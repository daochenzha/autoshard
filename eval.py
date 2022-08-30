import os
import argparse
import json
import torch
import numpy as np
import copy
import time

from autoshard.bench import Env, table_size
from autoshard import sharders
from autoshard.utils import allocation2plan, select_data

def main():
    parser = argparse.ArgumentParser("Benchmark sharding")
    parser.add_argument('--data-dir', type=str, default="data/dlrm_datasets")
    parser.add_argument('--checkpointpath', type=str, default="logs/autoshard/model.tar")
    parser.add_argument('--alg', type=str, default="random", choices=["random", "dim_greedy", "lookup_greedy", "size_greedy", "autoshard"])
    parser.add_argument('--ndevices', type=int, default=8)
    parser.add_argument('--max-memory', type=int, default=7, help="Max memory for each shard in GB")
    parser.add_argument('--gpu-devices', type=str, default="0,1,2,3,4,5,6,7")

    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices

    # We don't want too many threads for stable benchmarks
    torch.set_num_threads(1)

    test_indices = []
    with open(os.path.join(flags.data_dir,"test.txt"), "r") as f:
        for line in f.readlines():
            table_indices = list(map(int, line.strip().split(",")))
            test_indices.append(table_indices)

    table_config_path = os.path.join(flags.data_dir, "table_configs.json")
    data_path = os.path.join(flags.data_dir, "data.pt")
    with open(table_config_path) as f:
        table_configs = json.load(f)["tables"]
    data = torch.load(data_path)

    for task_id, table_indices in enumerate(test_indices):
        selected_table_configs, selected_data = select_data(copy.deepcopy(table_configs), data, table_indices)
        
        env = Env(
            table_configs=selected_table_configs,
            data=selected_data,
            ndevices=flags.ndevices,
            max_memory=flags.max_memory,
        )
        env.checkpointpath = flags.checkpointpath

        start = time.time()
        allocation = sharders.shard(
            env=env,
            alg=flags.alg,
        )
        end = time.time()
        if flags.alg == "autoshard":
            allocation, sharding_time = allocation
        else:
            sharding_time = end - start
        print("Task", task_id)
        print("Allocation of", flags.alg+":", allocation)

        # Check size
        sizes = [table_size(config["row"], config["dim"], fp16=True) for config in env.table_configs]
        plan = allocation2plan(allocation, env.ndevices)
        size_sums = [sum([sizes[i] for i in shard]) for shard in plan]
        print("Sizes:", size_sums)
        max_size_sum = max(size_sums)
        if max_size_sum > flags.max_memory:
            print("Out of memory")
            return

        
        # Evaluate sharding plan
        latencies = env.step(plan)

        print("----- Results -----")
        print("Latencies:", latencies)
        print("Max:", np.max(latencies))
        print("Min:", np.min(latencies))
        print("Mean:", np.mean(latencies))
        print("Min/Max:", np.min(latencies)/np.max(latencies))
        print("Sharding time:", sharding_time)


if __name__ == '__main__':
    main()

