import copy
import csv
import datetime
import json
import logging
import os
import time
import collections
from typing import Dict

import numpy as np

import torch
import torch.nn.functional as F


def table_size(hash_size, dim, fp16=False):
    gb_size = hash_size * dim / (1024 * 1024 * 1024)
    if fp16:
        gb_size = 2 * gb_size
    else:
        gb_size = 4 * gb_size
    return gb_size

def _format_frame(frame):
    frame["state_table"] = torch.tensor(frame["state_table"]).float()
    frame["state_other"] = torch.tensor(frame["state_other"]).float()
    frame["offset"] = torch.tensor([len(shard) for shard in frame["actions_table"]])
    frame["actions_table"] = [[torch.cat([torch.tensor(shard) for shard in frame["actions_table"]], dim=0).float()]]
    frame["actions_other"] = torch.tensor(frame["actions_other"]).float()

    frame["state_table"] = frame["state_table"].view((1,1) + frame["state_table"].shape)
    frame["state_other"] = frame["state_other"].view((1,1) + frame["state_other"].shape)
    frame["offset"] = frame["offset"].view((1,1) + frame["offset"].shape)
    frame["actions_other"] = frame["actions_other"].view((1,1) + frame["actions_other"].shape)
    return frame


class Environment:
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None
        self.ndevices = gym_env.ndevices
        self.num_tables = gym_env.num_tables

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        initial_frame  = _format_frame(self.gym_env.reset()[0])
        state_table, state_other, offset, actions_table, actions_other = initial_frame["state_table"], initial_frame["state_other"], initial_frame["offset"], initial_frame["actions_table"], initial_frame["actions_other"]
        return dict(
            state_table=state_table,
            state_other=state_other,
            offset=offset,
            actions_table=actions_table,
            actions_other=actions_other,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
            cost_data=None,
        )

    def step(self, action):
        plan = None
        frame, reward, done, info = self.gym_env.step(action.item())
        reb_data = None
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            frame, plan = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        frame = _format_frame(frame)
        state_table, state_other, offset, actions_table, actions_other = frame["state_table"], frame["state_other"], frame["offset"], frame["actions_table"], frame["actions_other"]
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        return dict(
            state_table=state_table,
            state_other=state_other,
            offset=offset,
            actions_table=actions_table,
            actions_other=actions_other,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
            cost_data=info["cost_data"],
            plan = plan,
        )

    def close(self):
        self.gym_env.close()



def gather_metadata() -> Dict:
    date_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # Gathering git metadata.
    try:
        import git

        try:
            repo = git.Repo(search_parent_directories=True)
            git_sha = repo.commit().hexsha
            git_data = dict(
                commit=git_sha,
                branch=None if repo.head.is_detached else repo.active_branch.name,
                is_dirty=repo.is_dirty(),
                path=repo.git_dir,
            )
        except git.InvalidGitRepositoryError:
            git_data = None
    except ImportError:
        git_data = None
    # Gathering slurm metadata.
    if "SLURM_JOB_ID" in os.environ:
        slurm_env_keys = [k for k in os.environ if k.startswith("SLURM")]
        slurm_data = {}
        for k in slurm_env_keys:
            d_key = k.replace("SLURM_", "").replace("SLURMD_", "").lower()
            slurm_data[d_key] = os.environ[k]
    else:
        slurm_data = None
    return dict(
        date_start=date_start,
        date_end=None,
        successful=False,
        git=git_data,
        slurm=slurm_data,
        env=os.environ.copy(),
    )


class FileWriter:
    def __init__(
        self,
        xpid: str = None,
        xp_args: dict = None,
        rootdir: str = "~/logs",
        symlink_to_latest: bool = True,
    ):
        if not xpid:
            # Make unique id.
            xpid = "{proc}_{unixtime}".format(
                proc=os.getpid(), unixtime=int(time.time())
            )
        self.xpid = xpid
        self._tick = 0

        # Metadata gathering.
        if xp_args is None:
            xp_args = {}
        self.metadata = gather_metadata()
        # We need to copy the args, otherwise when we close the file writer
        # (and rewrite the args) we might have non-serializable objects (or
        # other unwanted side-effects).
        self.metadata["args"] = copy.deepcopy(xp_args)
        self.metadata["xpid"] = self.xpid

        formatter = logging.Formatter("%(message)s")
        self._logger = logging.getLogger("logs/out")

        # To stdout handler.
        shandle = logging.StreamHandler()
        shandle.setFormatter(formatter)
        self._logger.addHandler(shandle)
        self._logger.setLevel(logging.INFO)

        rootdir = os.path.expandvars(os.path.expanduser(rootdir))
        # To file handler.
        self.basepath = os.path.join(rootdir, self.xpid)
        if not os.path.exists(self.basepath):
            self._logger.info("Creating log directory: %s", self.basepath)
            os.makedirs(self.basepath, exist_ok=True)
        else:
            self._logger.info("Found log directory: %s", self.basepath)

        if symlink_to_latest:
            # Add 'latest' as symlink unless it exists and is no symlink.
            symlink = os.path.join(rootdir, "latest")
            try:
                if os.path.islink(symlink):
                    os.remove(symlink)
                if not os.path.exists(symlink):
                    os.symlink(self.basepath, symlink)
                    self._logger.info("Symlinked log directory: %s", symlink)
            except OSError:
                # os.remove() or os.symlink() raced. Don't do anything.
                pass

        self.paths = dict(
            msg="{base}/out.log".format(base=self.basepath),
            logs="{base}/logs.csv".format(base=self.basepath),
            fields="{base}/fields.csv".format(base=self.basepath),
            meta="{base}/meta.json".format(base=self.basepath),
        )

        self._logger.info("Saving arguments to %s", self.paths["meta"])
        if os.path.exists(self.paths["meta"]):
            self._logger.warning(
                "Path to meta file already exists. " "Not overriding meta."
            )
        else:
            self._save_metadata()

        self._logger.info("Saving messages to %s", self.paths["msg"])
        if os.path.exists(self.paths["msg"]):
            self._logger.warning(
                "Path to message file already exists. " "New data will be appended."
            )

        fhandle = logging.FileHandler(self.paths["msg"])
        fhandle.setFormatter(formatter)
        self._logger.addHandler(fhandle)

        self._logger.info("Saving logs data to %s", self.paths["logs"])
        self._logger.info("Saving logs' fields to %s", self.paths["fields"])
        self.fieldnames = ["_tick", "_time"]
        if os.path.exists(self.paths["logs"]):
            self._logger.warning(
                "Path to log file already exists. " "New data will be appended."
            )
            # Override default fieldnames.
            with open(self.paths["fields"], "r") as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                if len(lines) > 0:
                    self.fieldnames = lines[-1]
            # Override default tick: use the last tick from the logs file plus 1.
            with open(self.paths["logs"], "r") as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                # Need at least two lines in order to read the last tick:
                # the first is the csv header and the second is the first line
                # of data.
                if len(lines) > 1:
                    self._tick = int(lines[-1][0]) + 1

        self._fieldfile = open(self.paths["fields"], "a")
        self._fieldwriter = csv.writer(self._fieldfile)
        self._logfile = open(self.paths["logs"], "a")
        self._logwriter = csv.DictWriter(self._logfile, fieldnames=self.fieldnames)

    def log(self, to_log: Dict, tick: int = None, verbose: bool = False) -> None:
        if tick is not None:
            raise NotImplementedError
        else:
            to_log["_tick"] = self._tick
            self._tick += 1
        to_log["_time"] = time.time()

        old_len = len(self.fieldnames)
        for k in to_log:
            if k not in self.fieldnames:
                self.fieldnames.append(k)
        if old_len != len(self.fieldnames):
            self._fieldwriter.writerow(self.fieldnames)
            self._logger.info("Updated log fields: %s", self.fieldnames)

        if to_log["_tick"] == 0:
            self._logfile.write("# %s\n" % ",".join(self.fieldnames))

        if verbose:
            self._logger.info(
                "LOG | %s",
                ", ".join(["{}: {}".format(k, to_log[k]) for k in sorted(to_log)]),
            )

        self._logwriter.writerow(to_log)
        self._logfile.flush()

    def close(self, successful: bool = True) -> None:
        self.metadata["date_end"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )
        self.metadata["successful"] = successful
        self._save_metadata()

        for f in [self._logfile, self._fieldfile]:
            f.close()

    def _save_metadata(self) -> None:
        with open(self.paths["meta"], "w") as jsonfile:
            json.dump(self.metadata, jsonfile, indent=4, sort_keys=True)

VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")


def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(torch.flatten(policy_logits, 0, -2), dim=-1),
        torch.flatten(actions),
        reduction="none",
    ).view_as(actions)


def from_logits(
    behavior_policy_logits,
    target_policy_logits,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace for softmax policies."""

    target_action_log_probs = action_log_probs(target_policy_logits, actions)
    behavior_action_log_probs = action_log_probs(behavior_policy_logits, actions)
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


@torch.no_grad()
def from_importance_weights(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace from log importance weights."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = torch.clamp(rhos, max=1.0)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
        vs_t_plus_1 = torch.cat(
            [vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0
        )
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)

def plan2allocation(plan):
    num_tables = sum([len(shard) for shard in plan])
    table_device_indices = [-1] * num_tables
    for bin_id, partition in enumerate(plan):
        for index in partition:
            table_device_indices[index] = bin_id
    return table_device_indices

def allocation2plan(allocation, ndevices):
    plan = [[] for _ in range(ndevices)]
    for i, d in enumerate(allocation):
        plan[d].append(i)
    return plan

class RingBuffer(object):
    def __init__(self, maxlen, batch_size):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.batch_size = batch_size
        self.offset_buf = np.empty(shape=(maxlen,), dtype=np.int64)
        self.cost_buf = np.empty(shape=(maxlen,), dtype=np.float32)
        self.tables_buf = [None] * maxlen

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return {
            "offset": self.offset_buf[idx],
            "cost": self.cost_buf[idx],
            "tables": self.tables_buf[idx],
        }

    def __len__(self):
        return self.length

    def get_batch(self):
        if self.length <= 0:
            return None
        idxs = np.random.randint(self.length, size=self.batch_size)
        return {
            "offset": self.offset_buf[idxs],
            "cost": self.cost_buf[idxs],
            "tables": [self.tables_buf[i] for i in idxs],
        }

    def append(self, cost_data):
        end = self.start + len(cost_data["offset"])
        remain = 0
        if end > self.maxlen:
            remain = end - self.maxlen
            end = self.maxlen
        self.offset_buf[self.start:end] = cost_data["offset"][:end-self.start]
        self.cost_buf[self.start:end] = cost_data["cost"][:end-self.start]
        self.tables_buf[self.start:end] = cost_data["tables"][:end-self.start]
        if remain > 0:
            self.offset_buf[:remain] = cost_data["offset"][end-self.start:]
            self.cost_buf[:remain] = cost_data["cost"][end-self.start:]
            self.tables_buf[:remain] = cost_data["tables"][end-self.start:]

        self.start = (self.start + len(cost_data["offset"])) % self.maxlen
        self.length = min(self.length + len(cost_data["offset"]), self.maxlen)

def select_data(table_configs, data, table_indices):
    selected_table_configs = []
    for i in table_indices:
        table_config = table_configs[i]
        table_config["index"] = len(selected_table_configs)
        selected_table_configs.append(table_config)

    selected_data = {key: [None] for key in data}
    selected_data["lS_offsets"] = [data["lS_offsets"][i] for i in table_indices]
    selected_data["lS_indices"] = [data["lS_indices"][i] for i in table_indices]

    return selected_table_configs, selected_data
