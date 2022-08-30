import argparse

from autoshard.training import train

def main():
    parser = argparse.ArgumentParser("AutoShard")
    parser.add_argument('--data-dir', type=str, default="data/dlrm_datasets")
    parser.add_argument('--ndevices', type=int, default=8)
    parser.add_argument('--gpu-devices', type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument('--max-memory', type=int, default=7, help="Max memory for each shard in GB")
    parser.add_argument("--xpid", default="autoshard",
                    help="Experiment id (default: autoshard).")

    # Training settings.
    parser.add_argument("--disable-cost-model", action="store_true",
                        help="Disable perf model as features.")
    parser.add_argument("--disable-dims", action="store_true")
    parser.add_argument("--disable-rows", action="store_true")
    parser.add_argument("--disable-pooling-factors", action="store_true")
    parser.add_argument("--disable-sizes", action="store_true")
    parser.add_argument("--disable-bins", action="store_true")
    parser.add_argument("--checkpoint-history", action="store_true")
    parser.add_argument("--checkpoint-history-every", default=2400, type=int)
    parser.add_argument("--disable-checkpoint", action="store_true",
                        help="Disable saving checkpoint.")
    parser.add_argument("--savedir", default="logs/",
                        help="Root dir where experiment data will be saved.")
    parser.add_argument("--total-steps", default=100000, type=int, metavar="T",
                        help="Total environment steps to train for.")
    parser.add_argument("--batch-size", default=8, type=int, metavar="B",
                        help="Learner batch size.")
    parser.add_argument("--unroll-length", default=100, type=int, metavar="T",
                        help="The unroll length (time dimension).")
    parser.add_argument("--num-learner-threads", "--num-threads", default=1, type=int,
                        metavar="N", help="Number learner threads.")
    parser.add_argument("--num-cost-updates", default=20, type=int, help="Number of cost updates.")
    parser.add_argument("--num-min-buf-size", default=0, type=int, help="Number of min buffer size to start training.")

    # Loss settings.
    parser.add_argument("--entropy-cost", default=0.001,
                        type=float, help="Entropy cost/multiplier.")
    parser.add_argument("--baseline-cost", default=0.5,
                        type=float, help="Baseline cost/multiplier.")
    parser.add_argument("--discounting", default=1.0,
                        type=float, help="Discounting factor.")
    parser.add_argument("--reward-clipping", default="none",
                        choices=["abs_one", "none"],
                        help="Reward clipping.")

    # Optimizer settings.
    parser.add_argument("--learning-rate", default=0.001,
                        type=float, metavar="LR", help="Learning rate.")
    parser.add_argument("--grad-norm-clipping", default=40.0, type=float,
                        help="Global gradient norm clip.")
    flags = parser.parse_args()
    flags.num_actors = len(flags.gpu_devices.split(","))

    train(flags)

if __name__ == '__main__':
    main()

