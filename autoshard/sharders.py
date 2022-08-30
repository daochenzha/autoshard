from dataclasses import dataclass
import numpy as np
import time

from autoshard.utils import allocation2plan, plan2allocation, table_size

_sharders = {}

@dataclass
class TableInfo:
    index: int
    cost: float
    size: float

    def __lt__(self, o: "TableInfo"):
        return (self.cost, self.size, self.index) < (o.cost, o.size, o.index)

def register_sharder(sharder_name):
    def decorate(func):
        _sharders[sharder_name] = func
        return func
    return decorate

# get device indices for tables
# e.g 8 tables, No. [1,3,5,6] on device 0, No. [2,4,7,8] on device 1, then
# return [0, 1, 0, 1, 0, 0, 1, 1]
def shard(env, alg="random"):
    if alg not in _sharders:
        import sys
        sys.exit("ERROR: sharder not found")
    return _sharders[alg](env)

@register_sharder("dim_greedy")
def dim_greedy_shard(env, return_plan=False):
    # Get the embedding dims
    idx_weight_pairs = []
    for i, config in enumerate(env.table_configs):

        index = config["index"]
        dim = config["dim"]
        row = config["row"]
        size = table_size(row, dim, fp16=True)
        idx_weight_pairs.append(TableInfo(index=index, cost=dim, size=size))

    # Greedy algorithm
    num_bins = env.ndevices
    sorted_idx_weight_pairs = sorted(idx_weight_pairs)
    partitions = [[] for p in range(num_bins)]
    partition_sums = [0.0] * num_bins
    partition_size_sums = [0.0] * num_bins

    mem_cap = [env.max_memory] * num_bins

    while sorted_idx_weight_pairs:
        table_info = sorted_idx_weight_pairs.pop()
        min_sum = np.inf
        min_size_taken = np.inf
        min_r = -1
        for r in range(num_bins):
            if partition_size_sums[r] + table_info.size <= mem_cap[r]:
                if partition_sums[r] < min_sum or (
                    partition_sums[r] == min_sum
                    and partition_size_sums[r] < min_size_taken
                ):
                    min_sum = partition_sums[r]
                    min_r = r
                    min_size_taken = partition_size_sums[r]

        partitions[min_r].append(table_info)
        partition_sums[min_r] += table_info.cost
        partition_size_sums[min_r] += table_info.size

    partitions = [[table_info.index for table_info in partition] for partition in partitions]

    return partitions if return_plan else plan2allocation(partitions)

@register_sharder("size_greedy")
def size_greedy_shard(env, return_plan=False):
    # Get the embedding dims
    idx_weight_pairs = []
    for i, config in enumerate(env.table_configs):

        index = config["index"]
        dim = config["dim"]
        row = config["row"]
        size = table_size(row, dim, fp16=True)
        idx_weight_pairs.append(TableInfo(index=index, cost=size, size=size))

    # Greedy algorithm
    num_bins = env.ndevices
    sorted_idx_weight_pairs = sorted(idx_weight_pairs)
    partitions = [[] for p in range(num_bins)]
    partition_sums = [0.0] * num_bins
    partition_size_sums = [0.0] * num_bins

    mem_cap = [env.max_memory] * num_bins

    while sorted_idx_weight_pairs:
        table_info = sorted_idx_weight_pairs.pop()
        min_sum = np.inf
        min_size_taken = np.inf
        min_r = -1
        for r in range(num_bins):
            if partition_size_sums[r] + table_info.size <= mem_cap[r]:
                if partition_sums[r] < min_sum or (
                    partition_sums[r] == min_sum
                    and partition_size_sums[r] < min_size_taken
                ):
                    min_sum = partition_sums[r]
                    min_r = r
                    min_size_taken = partition_size_sums[r]

        partitions[min_r].append(table_info)
        partition_sums[min_r] += table_info.cost
        partition_size_sums[min_r] += table_info.size

    partitions = [[table_info.index for table_info in partition] for partition in partitions]

    return partitions if return_plan else plan2allocation(partitions)

@register_sharder("lookup_greedy")
def lookup_greedy_shard(env, return_plan=False):
    # Get the embedding dims
    idx_weight_pairs = []
    for i, config in enumerate(env.table_configs):

        index = config["index"]
        dim = config["dim"]
        row = config["row"]
        pooling_factor = config["pooling_factor"]
        size = table_size(row, dim, fp16=True)
        idx_weight_pairs.append(TableInfo(index=index, cost=dim*pooling_factor, size=size))

    # Greedy algorithm
    num_bins = env.ndevices
    sorted_idx_weight_pairs = sorted(idx_weight_pairs)
    partitions = [[] for p in range(num_bins)]
    partition_sums = [0.0] * num_bins
    partition_size_sums = [0.0] * num_bins

    mem_cap = [env.max_memory] * num_bins

    while sorted_idx_weight_pairs:
        table_info = sorted_idx_weight_pairs.pop()
        min_sum = np.inf
        min_size_taken = np.inf
        min_r = -1
        for r in range(num_bins):
            if partition_size_sums[r] + table_info.size <= mem_cap[r]:
                if partition_sums[r] < min_sum or (
                    partition_sums[r] == min_sum
                    and partition_size_sums[r] < min_size_taken
                ):
                    min_sum = partition_sums[r]
                    min_r = r
                    min_size_taken = partition_size_sums[r]

        partitions[min_r].append(table_info)
        partition_sums[min_r] += table_info.cost
        partition_size_sums[min_r] += table_info.size

    partitions = [[table_info.index for table_info in partition] for partition in partitions]

    return partitions if return_plan else plan2allocation(partitions)

@register_sharder("random")
def random_shard(env):
    table_device_indices = []
    for _ in range(env.num_tables):
        table_device_indices.append(np.random.randint(env.ndevices))
    return table_device_indices

@register_sharder("autoshard")
def rl_shard(env):
    from autoshard.bench import GymWrapper
    from autoshard.training import Net
    from autoshard.utils import Environment
    import torch
    checkpointpath = env.checkpointpath
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model = Net(GymWrapper([env]).observation_space)
    model.load_state_dict(checkpoint["model_state_dict"])

    gym_env = GymWrapper([env], model, infer_only=True, feature_means=checkpoint["feature_means"], feature_stds=checkpoint["feature_stds"])
    env = Environment(gym_env)
    model.eval()
    
    env_output = env.initial()
    agent_state = model.initial_state(batch_size=1)
    returns = []

    start = time.time()
    while True:
        agent_output, agent_state = model(env_output, agent_state)
        env_output = env.step(agent_output["action"])
        if env_output["done"].item():
            plan = env_output["plan"]
            break
    end = time.time()

    return plan2allocation(plan), end-start
