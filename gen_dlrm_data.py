"""Generate tasks from DLRM synthetic datasets
"""

import argparse
import os
import json

import numpy as np
import torch


def gen_table_configs(
    lS_rows,
    lS_pooling_factors,
    lS_bin_counts,
    args
):
    T = len(lS_rows) # number of tables
    lS_dims = np.random.choice(args.dim_range, T)

    table_configs = {}
    table_configs["tables"] = []
    for i in range(T):
        table_config = {}
        table_config["index"] = i
        table_config["row"] = int(lS_rows[i])
        table_config["dim"] = int(lS_dims[i])
        table_config["pooling_factor"] = int(lS_pooling_factors[i])
        for j, bin_count in enumerate(lS_bin_counts[i]):
            table_config["bin_"+str(j)] = lS_bin_counts[i][j]

        table_configs["tables"].append(table_config)
    return table_configs

def process_data(data_path):
    indices, offsets, lengths = torch.load(data_path)
    num_tables, batch_size = lengths.shape

    indices = indices.cuda()
    offsets = offsets.cuda()

    lS_pooling_factors = list(map(int, lengths.float().mean(dim=1).tolist()))

    # Split the tables
    lS_offsets, lS_indices, lS_rows, lS_bin_counts = [], [], [], []
    for t in range(num_tables):
        start = t * batch_size
        end = (t + 1) * batch_size + 1
        table_offsets = offsets[start:end]
        table_indices = indices[table_offsets[0]:table_offsets[-1]]
        table_offsets = table_offsets - offsets[start]

        row = table_indices.max().int().item() + 1 if len(table_indices) > 0 else 1
        row = max(100, row)


        _, indices_counts = torch.unique(table_indices, return_counts=True)
        unique_counts, counts_counts = torch.unique(indices_counts, return_counts=True)
        total_counts = counts_counts.sum().item()

        if total_counts == 0:
            bin_counts = [0.0 for _ in range(17)]
        else:
            start, end = 0, 1
            bin_counts = []
            for i in range(16):
                bin_counts.append(counts_counts[(unique_counts > start) & (unique_counts <= end)].sum().item())
                start = end
                end *= 2
            bin_counts.append(counts_counts[unique_counts > start].sum().item())
            bin_counts = [x/total_counts for x in bin_counts]


        lS_offsets.append(table_offsets.cpu())
        lS_indices.append(table_indices.cpu())
        lS_rows.append(row)
        lS_bin_counts.append(bin_counts)


    return lS_offsets, lS_indices, lS_rows, lS_pooling_factors, lS_bin_counts


def main():
    parser = argparse.ArgumentParser("Generate synthetic data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data', type=str, default="../dlrm_datasets/embedding_bag/fbgemm_t856_bs65536.pt")
    parser.add_argument('--out-dir', type=str, default="data/dlrm_datasets")
    parser.add_argument('--dim-range', type=str, default="16,32") # Randomly select one of them
    parser.add_argument('--T-range', type=str, default="80,81")
    parser.add_argument('--num-train', type=int, default=1)
    parser.add_argument('--num-test', type=int, default=1)
    parser.add_argument('--test-as-train', action="store_true", help="Using train as test table set.")
    parser.add_argument('--transfer', action="store_true", help="Transfer setting. The first half for training, the second half for testing")
    parser.add_argument('--transfer-ratio', type=float, default=1.0)

    args = parser.parse_args()
    np.random.seed(args.seed)
    args.dim_range = list(map(int, args.dim_range.split(",")))
    args.T_range = list(map(int, args.T_range.split(",")))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print("Procssing DLRM data...")
    lS_offsets, lS_indices, lS_rows, lS_pooling_factors, lS_bin_counts = process_data(args.data)
    data = {
        "lS_offsets": lS_offsets,
        "lS_indices": lS_indices,
    }
    torch.save(data, os.path.join(args.out_dir, "data.pt"))

    print("Generating table configs...")
    table_configs = gen_table_configs(
        lS_rows,
        lS_pooling_factors,
        lS_bin_counts,
        args
    )
    with open(os.path.join(args.out_dir, "table_configs.json"), "w") as f:
        json.dump(table_configs, f)

    print("Generating training and testing indices...")
    T = len(lS_offsets)
    with open(os.path.join(args.out_dir, "train.txt"), "w") as f:
        for _ in range(args.num_train):
            task_T = np.random.randint(args.T_range[0], args.T_range[1])
            if args.transfer:
                indices = np.random.choice([i for i in range(T//2)], size=task_T, replace=False)
            else:
                indices = np.random.choice([i for i in range(T)], size=task_T, replace=False)
            f.write(",".join(list(map(str, indices))) + "\n")
    with open(os.path.join(args.out_dir, "test.txt"), "w") as f:
        if args.test_as_train:
            assert args.num_train == 1
            assert args.num_test == 1
            f.write(",".join(list(map(str, indices))) + "\n")
        else: 
            for _ in range(args.num_test):
                task_T = np.random.randint(args.T_range[0], args.T_range[1])
                if args.transfer:
                    indices = np.random.choice([i for i in range(int(T//2 * (1.0-args.transfer_ratio)))] + [i for i in range(T//2, T//2 + int((T-T//2) * args.transfer_ratio))], size=task_T, replace=False)
                else:
                    indices = np.random.choice([i for i in range(T)], size=task_T, replace=False)
                f.write(",".join(list(map(str, indices))) + "\n")


if __name__ == '__main__':
    main()
