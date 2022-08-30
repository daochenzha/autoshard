import os
import time
import json
import numpy as np
from typing import Dict, Any, Callable
from collections import OrderedDict
import gc

import torch
from fbgemm_gpu import split_table_batched_embeddings_ops
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType

import gym
from gym import spaces

from autoshard.utils import table_size


# Timer in seconds
class Timer:
    def __init__(self, device: str):
        self.device: str = device
        self.start_time: float = 0
        self.end_time: float = 0
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if self.device == "cpu":
            self.start_time = time.perf_counter()
        else:
            torch.cuda.synchronize()
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
            self.start_time = 0
        return self

    def __exit__(self, type, value, traceback):
        if self.device == "cpu":
            self.end_time = time.perf_counter()
        else:
            self.end_event.record()
            torch.cuda.synchronize()
            self.end_time = self.start_event.elapsed_time(self.end_event) * 1.0e-3

    # returns time in seconds
    def elapsed_time(self):
        return self.end_time - self.start_time

def benchmark_op(op: Callable, args: Any, kwargs: Any, grads_tensor: Any, device: str, num_iter: int):
    time_records = []
    for _ in range(num_iter):
        # flush cache
        if device.startswith("cuda"):
            _ = torch.rand(6 * 1024 * 1024 // 4).float() * 2  # V100 6MB L2 cache
            #_ = torch.rand(11 * 512 * 1024 // 4).float() * 2  # 2080 5.5MB L2 cache
            torch.cuda.empty_cache()

        with Timer(device) as timer:
            op(*args, **kwargs).backward(grads_tensor)
        time_records.append(timer.elapsed_time() * 1000)
    return time_records

def warmup(
    op: Callable,
    args: Any,
    kwargs: Any,
    grads_tensor: Any,
    device: str,
    num_iter: int,
):
    # warm up
    time_records = benchmark_op(op, args, kwargs, grads_tensor, device, num_iter)
    return time_records

def measure_latency(
    op: Callable,
    args: Any,
    kwargs: Any,
    grads_tensor: Any,
    device: str,
    num_iter: int,
):
    torch.cuda.nvtx.range_push("op_bench")
    time_records = benchmark_op(op, args, kwargs, grads_tensor, device, num_iter)

    return time_records

class Env:
    def __init__(
        self,
        table_configs,
        data,
        ndevices,
        max_memory,
        warmup_iter=5,
        num_iter=10,
        device="cuda:0",
        verbose=False,
    ):

        self.table_configs = table_configs
        self.num_tables = len(self.table_configs)

        # Load indices and offsets
        self.offsets = data["lS_offsets"]
        self.indices = data["lS_indices"]

        self.ndevices = ndevices
        self.max_memory = max_memory
        self.warmup_iter = warmup_iter
        self.num_iter = num_iter
        self.device = device
        self.verbose = verbose

    def step(self, plan):

        latenceis = []
        for shard in plan:
            latency = self.single_step(shard)
            latenceis.append(latency)

        return latenceis

    def single_step(self, table_indices):
        if len(table_indices) == 0:
            return 0

        gc.collect()
        torch.cuda.empty_cache()

        # Build the op
        shard_table_configs = [self.table_configs[i] for i in table_indices]

        op = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    table_config["row"],
                    table_config["dim"],
                    split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                    split_table_batched_embeddings_ops.ComputeDevice.CUDA,
                )
                for table_config in shard_table_configs
            ],
            optimizer=OptimType.EXACT_SGD,
            cache_algorithm=split_table_batched_embeddings_ops.CacheAlgorithm.LFU,
            cache_reserved_memory=8.0,
            eps=0.01,
            device=self.device,
            weights_precision=SparseType.FP16,
        )

        # Get data
        shard_offsets = [self.offsets[i] for i in table_indices]
        shard_indices = [self.indices[i] for i in table_indices]
        dims = [config["dim"] for config in shard_table_configs]
        args, kwargs, grads_tensor = get_data(shard_offsets, shard_indices, dims, self.device)

        # Warmup
        warmup_time_records = warmup(
            op,
            args,
            kwargs,
            grads_tensor,
            self.device,
            num_iter=self.warmup_iter
        )
        if self.verbose:
            print("Warmup:", warmup_time_records, table_indices)

        # Benchmark
        time_records = measure_latency(
            op,
            args,
            kwargs,
            grads_tensor,
            self.device,
            num_iter=self.num_iter
        )

        if self.verbose:
            print("Benchmark:", time_records, table_indices)

        # Remove the largest two and smallest two values
        time_records.sort()
        return np.mean(time_records[2:-2])

class GymWrapper(gym.Env):
    def __init__(self, envs, model=None, infer_only=False, feature_means=None, feature_stds=None):
        self.infer_only = infer_only
        self._envs = envs
        self.num_envs = len(envs)
        self.num_tables = [_env.num_tables for _env in self._envs]
        self.ndevices = envs[0].ndevices
        self.max_memory = envs[0].max_memory
        self.model = model
        self.cost_data = {
            "offset": [],
            "tables": [],
            "cost": [],
        }

        # Table features
        self.table_features = []
        for _env in self._envs:
            _table_features = {
                "dims": [config["dim"] for config in _env.table_configs],
                "rows": [config["row"] for config in _env.table_configs],
                "pooling_factors": [config["pooling_factor"] for config in _env.table_configs],
                "sizes": [table_size(config["row"], config["dim"], fp16=True) for config in _env.table_configs],
            }
            for i in range(17):
                _table_features["bin_"+str(i)] = [config["bin_"+str(i)] for config in _env.table_configs]
            self.table_features.append(OrderedDict(_table_features))

        print("Total table sizes:", [sum(self.table_features[i]["sizes"]) for i in range(self.num_envs)])

        # Save means, stds
        if feature_means is not None:
            self.feature_means, self.feature_stds = feature_means, feature_stds
        else:
            self.feature_means, self.feature_stds = {}, {}

        for key in self.table_features[0]:
            if key != "sizes" and "bin" not in key:
                if key in self.feature_means:
                    mean = self.feature_means[key]
                else:
                    mean = np.mean([x for table_features in self.table_features for x in table_features[key]])
                    self.feature_means[key] = mean
                if key in self.feature_stds:
                    std = self.feature_stds[key]
                else:
                    std = np.std([x for table_features in self.table_features for x in table_features[key]])
                    self.feature_stds[key] = std
                for i in range(self.num_envs):
                    # print("mean, std", mean, std)
                    # print("key", key, self.table_features[i][key])
                    if std == 0:
                        self.table_features[i][key] = [(float(x) - mean) for x in self.table_features[i][key]] # Normalize the features by substracting mean and dividing by std
                    else:
                        self.table_features[i][key] = [(float(x) - mean) / std for x in self.table_features[i][key]] # Normalize the features by substracting mean and dividing by std

        self.used_table_features = [
            "dims",
            "rows",
            "pooling_factors",
            "sizes",
            "bin_0",
            "bin_1",
            "bin_2",
            "bin_3",
            "bin_4",
            "bin_5",
            "bin_6",
            "bin_7",
            "bin_8",
            "bin_9",
            "bin_10",
            "bin_11",
            "bin_12",
            "bin_13",
            "bin_14",
            "bin_15",
            "bin_16",
        ]
        self.num_table_features = len(self.used_table_features)

        # State and action spaces for gym
        self.action_space = spaces.Discrete(self.ndevices)
        self.observation_space = gym.spaces.Dict({
            "state_table": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_table_features,)),
            "state_other": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            "actions_table": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.ndevices,self.num_table_features)),
            "actions_other": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.ndevices,2)),
        })

        self.table_indices = [[config["index"] for config in _env.table_configs] for _env in self._envs]
        self.plan = [[] for _ in range(self.ndevices)]
        self.env_id = np.random.randint(self.num_envs)

    def reset(self):
        real_plan = [[self.table_indices[self.env_id][i] for i in shard] for shard in self.plan]
        self.env_id = np.random.randint(self.num_envs)
        self.plan = [[] for _ in range(self.ndevices)]
        self.index = 0

        self._sort()

        return self._get_obs(), real_plan

    def step(self, action):
        self.plan[action].append(self.index)
        self.index += 1
        obs = self._get_obs()
        if len(self.cost_data["offset"]) > 0:
            self.cost_data["offset"] = np.concatenate(self.cost_data["offset"])
            self.cost_data["cost"] = np.array(self.cost_data["cost"], dtype=np.float32)
            info = {"cost_data": self.cost_data}
            self.cost_data = {key: [] for key in self.cost_data}
        else:
            info = {"cost_data": None}
        if self.index < self.num_tables[self.env_id]:
            return obs, 0, False, info
        else:
            if self.infer_only:
                reward = -1
            else:
                reward = self._get_reward()
            return obs, reward, True, info


    def _get_cost(self, indices_batch, predictor=False): # Reordered index here
        # Identify the non-empty list
        non_empty = [j for j in range(len(indices_batch)) if len(indices_batch[j]) > 0]
        if len(non_empty) > 0:
            inputs = {
                "offset": np.array([len(indices) for j, indices in enumerate(indices_batch) if j in non_empty], dtype=np.int64),
                "tables": [np.array([[self.table_features[self.env_id][key][i] for key in self.used_table_features] for i in indices], dtype=np.float32) for j, indices in enumerate(indices_batch) if j in non_empty]
            }
            if not predictor:
                costs = [self._envs[self.env_id].single_step([self.table_indices[self.env_id][i] for i in indices]) for j, indices in enumerate(indices_batch) if j in non_empty]
                self.cost_data["offset"].append(inputs["offset"])
                self.cost_data["tables"].extend(inputs["tables"])
                self.cost_data["cost"].extend(costs)
            else:
                with torch.no_grad():
                    costs = self.model.cost_forward({
                        "offset": torch.from_numpy(inputs["offset"]),
                        "tables": [torch.from_numpy(tables) for tables in inputs["tables"]],
                    }).tolist()

        # Recover the costs with 0
        actual_costs = [0 for _ in range(len(indices_batch))]
        for i, j in enumerate(non_empty):
            actual_costs[j] = costs[i]

        return actual_costs

    def _sort(self):
        single_table_costs = self._get_cost([[i] for i in range(self.num_tables[self.env_id])], predictor=True)
        # Sort them based on single table costs
        (
            self.table_indices[self.env_id],
            single_table_costs,
            self.table_features[self.env_id]["dims"],
            self.table_features[self.env_id]["rows"],
            self.table_features[self.env_id]["pooling_factors"],
            self.table_features[self.env_id]["sizes"],
            self.table_features[self.env_id]["bin_0"],
            self.table_features[self.env_id]["bin_1"],
            self.table_features[self.env_id]["bin_2"],
            self.table_features[self.env_id]["bin_3"],
            self.table_features[self.env_id]["bin_4"],
            self.table_features[self.env_id]["bin_5"],
            self.table_features[self.env_id]["bin_6"],
            self.table_features[self.env_id]["bin_7"],
            self.table_features[self.env_id]["bin_8"],
            self.table_features[self.env_id]["bin_9"],
            self.table_features[self.env_id]["bin_10"],
            self.table_features[self.env_id]["bin_11"],
            self.table_features[self.env_id]["bin_12"],
            self.table_features[self.env_id]["bin_13"],
            self.table_features[self.env_id]["bin_14"],
            self.table_features[self.env_id]["bin_15"],
            self.table_features[self.env_id]["bin_16"],
        ) = tuple(map(list, zip(*sorted(
            list(zip(
                self.table_indices[self.env_id],
                single_table_costs,
                self.table_features[self.env_id]["dims"],
                self.table_features[self.env_id]["rows"],
                self.table_features[self.env_id]["pooling_factors"],
                self.table_features[self.env_id]["sizes"],
                self.table_features[self.env_id]["bin_0"],
                self.table_features[self.env_id]["bin_1"],
                self.table_features[self.env_id]["bin_2"],
                self.table_features[self.env_id]["bin_3"],
                self.table_features[self.env_id]["bin_4"],
                self.table_features[self.env_id]["bin_5"],
                self.table_features[self.env_id]["bin_6"],
                self.table_features[self.env_id]["bin_7"],
                self.table_features[self.env_id]["bin_8"],
                self.table_features[self.env_id]["bin_9"],
                self.table_features[self.env_id]["bin_10"],
                self.table_features[self.env_id]["bin_11"],
                self.table_features[self.env_id]["bin_12"],
                self.table_features[self.env_id]["bin_13"],
                self.table_features[self.env_id]["bin_14"],
                self.table_features[self.env_id]["bin_15"],
                self.table_features[self.env_id]["bin_16"],
            )),
            key=lambda x: x[1],
            reverse=True
        ))))

    def _get_obs(self):
        if self.index >= self.num_tables[self.env_id]:
            return None
        else:
            return {
                "state_table": self._get_state_table(),
                "state_other": self._get_state_other(),
                "actions_table": self._get_actions_table(),
                "actions_other": self._get_actions_other(),
            }

    def _get_state_table(self):
        return [self.table_features[self.env_id][key][self.index] for key in self.used_table_features]

    def _get_state_other(self):
        return [
            float(self.index) / self.num_tables[self.env_id], # The proportion of assigned tables
        ]

    def _get_actions_table(self):
        # Table features
        shard_table_features = [[[self.table_features[self.env_id][key][i] for key in self.used_table_features] for i in shard] for shard in self.plan]
        # Add dummy features to avoid being empty
        for i in range(self.ndevices):
            shard_table_features[i].append([0 for _ in range(self.num_table_features)])
        return shard_table_features

    def _get_actions_other(self):
        return [[
            self._get_cost([shard], predictor=True)[0],
            sum([self.table_features[self.env_id]["sizes"][i] for i in shard]),
        ] for shard in self.plan]

    def _get_reward(self):
        # Check size
        size_sums = [sum([self.table_features[self.env_id]["sizes"][i] for i in shard]) for shard in self.plan]
        max_size_sum = max(size_sums)
        if max_size_sum > self.max_memory:
            return (self.max_memory - max_size_sum) / self.max_memory

        latencies = self._get_cost(self.plan, predictor=False)
        reward = min(latencies) / max(latencies)
        #print("Latencies:", latencies, reward)

        return reward

def get_data(offsets, indices, dims, device):
    args_indices = torch.cat([x.view(-1) for x in indices], dim=0).int()
    E_offsets = [0] + np.cumsum([x.view(-1).shape[0] for x in indices]).tolist()
    args_offsets = torch.cat([torch.tensor([0])] + [x[1:] + y for x, y in zip(offsets, E_offsets[:-1])], dim=0).int()

    batch_size = offsets[0].shape[0] - 1

    grads_tensor = (
        torch.randn(batch_size, sum(dims))
    )

    return (
        [
            args_indices.to(device),
            args_offsets.to(device),
        ],
        {},
        grads_tensor.to(device),

    )
