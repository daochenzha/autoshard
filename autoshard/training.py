import os
import traceback
import threading
import logging
import typing
import time
import timeit
import pprint
import numpy as np
import json
import copy

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F
torch.multiprocessing.set_sharing_strategy('file_system')

from autoshard.bench import Env, GymWrapper
from autoshard.utils import Environment, FileWriter, from_logits, RingBuffer, select_data

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=20,
)

def train(flags):

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    plogger = FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    T = flags.unroll_length
    B = flags.batch_size

    flags.train_indices = []
    with open(os.path.join(flags.data_dir,"train.txt"), "r") as f:
        for line in f.readlines():
            table_indices = list(map(int, line.strip().split(",")))
            flags.train_indices.append(table_indices)

    table_config_path = os.path.join(flags.data_dir, "table_configs.json")
    data_path = os.path.join(flags.data_dir, "data.pt")
    with open(table_config_path) as f:
        flags.table_configs = json.load(f)["tables"]
    flags.data = torch.load(data_path)


    envs = []
    for table_indices in flags.train_indices:
        selected_table_configs, selected_data = select_data(copy.deepcopy(flags.table_configs), flags.data, table_indices)
        env = Env(
            table_configs=selected_table_configs,
            data=selected_data,
            ndevices=flags.ndevices,
            max_memory=flags.max_memory,
        )
        envs.append(env)
    env = GymWrapper(envs)
    feature_means = env.feature_means
    feature_stds = env.feature_stds
    
    model = Net(env.observation_space)
    model.share_memory()

    replay_buffer = RingBuffer(5000, 512)

    actor_processes = []
    ctx = mp.get_context("fork")
    data_queue = ctx.Queue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                data_queue,
                model,
            ),
        )
        actor.start()
        actor_processes.append(actor)
    learner_model = Net(env.observation_space)

    optimizer = torch.optim.Adam(
        learner_model.parameters(),
        lr=flags.learning_rate,
    )
    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        while step < flags.total_steps:
            batch, agent_state = get_batch(
                flags,
                data_queue,
            )
            stats = learn(
                flags, model, learner_model, batch, agent_state, optimizer, replay_buffer, scheduler
            )
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                step += T * B

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "feature_means": feature_means,
                "feature_stds": feature_stds,
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        last_step = -flags.checkpoint_history_every
        while step < flags.total_steps:
            if flags.checkpoint_history:
                if step - last_step >= flags.checkpoint_history_every:
                    history_checkpointpath = os.path.expandvars(
                        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model_"+str(step)+".tar"))
                    )
                    logging.info("Saving checkpoint to %s", history_checkpointpath)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "feature_means": feature_means,
                            "feature_stds": feature_stds,
                        },
                        history_checkpointpath,
                    )
                    last_step = step

                    
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.3f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        for thread in threads:
            thread.join()
        checkpoint()
        plogger.close()
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for actor in actor_processes:
            actor.join(timeout=1)
            actor.terminate()

    checkpoint()
    plogger.close()

def get_batch(
    flags,
    data_queue: mp.Queue,
    lock=threading.Lock(),
):
    with lock:
        buffers = [data_queue.get() for _ in range(flags.batch_size)]

    batch = {
        key: torch.stack([buf[key] for buf in buffers], dim=1) for key in buffers[0] if key != "actions_table" and key != "initial_agent_state" and key != "cost_data"
    }

    # Action tables
    batch["actions_table"] = [[buf["actions_table"][i] for buf in buffers] for i in range(len(buffers[0]["actions_table"]))]

    # Cost data
    batch["cost_data"] = {key: [] for key in buffers[0]["cost_data"]}
    for buf in buffers:
        for key in batch["cost_data"]:
            if key == "tables":
                batch["cost_data"][key].extend(buf["cost_data"][key])
            else:
                batch["cost_data"][key].append(buf["cost_data"][key])
    for cost_data_key in ["offset", "cost"]:
        batch["cost_data"][cost_data_key] = np.concatenate(batch["cost_data"][cost_data_key])

    # Initial agent state
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[buf["initial_agent_state"] for buf in buffers])
    )
    batch = {k: t for k, t in batch.items()}
    initial_agent_state = tuple(
        t for t in initial_agent_state
    )
    return batch, initial_agent_state

def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    replay_buffer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        cost_data = batch["cost_data"] # First fetch cost data to avoid being flushed
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items() if key != "cost_data"}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting

        vtrace_returns = from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        episode_returns = batch["episode_return"][batch["done"]]

        total_loss = pg_loss + baseline_loss + entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        if len(cost_data["offset"]) > 0:
            replay_buffer.append(cost_data)

        sampled_cost_loss = -1.0
        if not flags.disable_cost_model and len(replay_buffer) > flags.num_min_buf_size:
            for _ in range(flags.num_cost_updates):
                sampled_cost_data = replay_buffer.get_batch()
                if sampled_cost_data is not None:
                    sampled_pred_cost = model.cost_forward({
                        "offset": torch.from_numpy(sampled_cost_data["offset"]),
                        "tables": [torch.from_numpy(tables) for tables in sampled_cost_data["tables"]],
                    })
                    sampled_cost_loss = compute_cost_loss(sampled_pred_cost, torch.from_numpy(sampled_cost_data["cost"]))
                    optimizer.zero_grad()
                    sampled_cost_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
                    optimizer.step()
                    sampled_cost_loss = sampled_cost_loss.item()


        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "sampled_cost_loss": sampled_cost_loss,
            "buffer_size": len(replay_buffer),
        }

        actor_model.load_state_dict(model.state_dict())
        return stats

def compute_cost_loss(pred_cost, cost):
    return torch.mean((pred_cost - cost) ** 2)

def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)

def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)

def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())

def act(
    flags,
    actor_index: int,
    data_queue: mp.Queue,
    model: torch.nn.Module,
):
    try:
        logging.info("Actor %i started.", actor_index)
        os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices.split(",")[actor_index]

        # We don't want too many threads for stable benchmarks
        torch.set_num_threads(1)

        envs = []
        for table_indices in flags.train_indices:
            selected_table_configs, selected_data = select_data(copy.deepcopy(flags.table_configs), flags.data, table_indices)
            env = Env(
                table_configs=selected_table_configs,
                data=selected_data,
                ndevices=flags.ndevices,
                max_memory=flags.max_memory,
            )
            envs.append(env)
        gym_env = GymWrapper(envs)

        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        torch.manual_seed(seed)
        np.random.seed(seed)
        gym_env.seed(seed)
        env = Environment(gym_env)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        with torch.no_grad():
            agent_output, unused_state = model(env_output, agent_state)

        while True:
            buf = create_buffer(flags, gym_env.observation_space)
            # Write old rollout end.
            for key in env_output:
                if key == "plan":
                    continue
                if key == "actions_table":
                    buf[key].append(env_output[key][0][0])
                elif key == "cost_data":
                    if env_output[key] is not None:
                        for cost_data_key in buf[key]:
                            if cost_data_key == "tables":
                                buf[key][cost_data_key].extend(env_output[key][cost_data_key])
                            else:
                                buf[key][cost_data_key].append(env_output[key][cost_data_key])
                else:
                    buf[key][0, ...] = env_output[key]
            for key in agent_output:
                buf[key][0, ...] = agent_output[key]
            buf["initial_agent_state"] = agent_state

            # Do new rollout.
            for t in range(flags.unroll_length):

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                env_output = env.step(agent_output["action"])

                for key in env_output:
                    if key == "plan":
                        continue
                    if key == "actions_table":
                        buf[key].append(env_output[key][0][0])
                    elif key == "cost_data":
                        if env_output[key] is not None:
                            for cost_data_key in buf[key]:
                                if cost_data_key == "tables":
                                    buf[key][cost_data_key].extend(env_output[key][cost_data_key])
                                else:
                                    buf[key][cost_data_key].append(env_output[key][cost_data_key])
                    else:
                        buf[key][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buf[key][t + 1, ...] = agent_output[key]

            for cost_data_key in ["offset", "cost"]:
                if len(buf["cost_data"][cost_data_key]) > 0:
                    buf["cost_data"][cost_data_key] = np.concatenate(buf["cost_data"][cost_data_key])
                else:
                    if cost_data_key == "offset":
                        buf["cost_data"][cost_data_key] = np.array([], dtype=np.int64)
                    else:
                        buf["cost_data"][cost_data_key] = np.array([], dtype=np.float32)
            while data_queue.qsize() > 2 * flags.batch_size:
                time.sleep(0.05)
            data_queue.put(buf)

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def create_buffer(flags, observation_space):
    T = flags.unroll_length
    specs = dict(
        state_table=dict(size=(T + 1, *observation_space["state_table"].shape), dtype=torch.float32),
        state_other=dict(size=(T + 1, *observation_space["state_other"].shape), dtype=torch.float32),
        actions_other=dict(size=(T + 1, *observation_space["actions_other"].shape), dtype=torch.float32), # Hardcode
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        offset=dict(size=(T + 1, flags.ndevices), dtype=torch.int64),
        policy_logits=dict(size=(T + 1, flags.ndevices), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buf = {key: [] for key in specs}
    for key in buf:
        buf[key] = torch.empty(**specs[key])
    buf["actions_table"] = []
    buf["cost_data"] = {
        "offset": [],
        "tables": [],
        "cost": [],
    }
    buf["initial_agent_state"] = None
    return buf

class Net(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        self.num_actions, self.table_dim = observation_space["actions_table"].shape
        self.state_other_dim = observation_space["state_other"].shape[0]
        self.actions_other_dim = observation_space["actions_other"].shape[1]

        # Table feature extraction
        self.t_fc_0 = nn.Linear(self.table_dim, 128)
        self.t_fc_1 = nn.Linear(128, 32)

        # Other state feature extraction
        self.s_fc_0 = nn.Linear(self.state_other_dim, 32)

        # Other actions feature extraction
        self.a_fc_0 = nn.Linear(self.actions_other_dim, 32)

        # LSTM
        core_output_size = self.t_fc_1.out_features + self.s_fc_0.out_features
        self.core = nn.LSTM(core_output_size, core_output_size, 2)

        # Final MLPs of policy head
        self.p_fc_0 = nn.Linear(192, 128)
        self.p_fc_1 = nn.Linear(128, 128)
        self.p_fc_2 = nn.Linear(128, 64)
        self.p_fc_3 = nn.Linear(64, 1)

        # Final MLPs of baseline head
        self.b_fc_0 = nn.Linear(self.t_fc_1.out_features + self.s_fc_0.out_features, 64)
        self.b_fc_1 = nn.Linear(64, 1)

        # Final MLPs of cost estimation head
        self.c_fc_0 = nn.Linear(self.t_fc_1.out_features, 64)
        self.c_fc_1 = nn.Linear(64, 1)

    def initial_state(self, batch_size):
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        state_table = inputs["state_table"] # T * B * table_dim
        state_other = inputs["state_other"] # T * B * state_other_dim
        actions_table = inputs["actions_table"] # list(T) * list(B) * num_actions * num_tables * table_dim
        actions_other = inputs["actions_other"] # T, B, num_actions, actions_other_dim
        offset = inputs["offset"] # T * B * num_actions
        T, B, _ = state_table.shape

        # State table representation
        state_table = torch.flatten(state_table, 0, 1)  # Merge time and batch.
        state_table = F.relu(self.t_fc_0(state_table))
        state_table = F.relu(self.t_fc_1(state_table))

        # State other representation
        state_other = state_other.flatten(0,1)
        state_other = F.relu(self.s_fc_0(state_other))

        # Merge to have the state
        state = torch.cat((state_table, state_other), dim=-1)

        # LSTM encoding historical state info
        core_input = state.view(T,B,-1)
        core_output_list = []
        notdone = (~inputs["done"]).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, hidden_size)
            # states:
            nd = nd.view(1, -1, 1)
            core_state = tuple(nd * s for s in core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        state = torch.flatten(torch.cat(core_output_list), 0, 1)

        # Value head
        baseline = F.relu(self.b_fc_0(state))
        baseline = self.b_fc_1(baseline)


 
        # Flatten list and offsets, extract table features for actions
        actions_table = [a for b in actions_table for a in b]
        actions_table = torch.cat(actions_table, dim=0)
        actions_table = F.relu(self.t_fc_0(actions_table))
        actions_table = F.relu(self.t_fc_1(actions_table))

        # Sum table features for each shard
        offset = offset.flatten(0,2)
        ind = torch.repeat_interleave(torch.arange(len(offset)), offset)
        tmp = torch.zeros((offset.shape[0], actions_table.shape[1]))
        tmp.index_add_(0, ind, actions_table)
        actions_table = tmp

        # Extract other action features
        actions_other = actions_other.flatten(0,2)
        actions_other = F.relu(self.a_fc_0(actions_other))

        # Policy head
        state = state.repeat(1,self.num_actions).view(-1,state.shape[1])
        actions = torch.cat((actions_table, actions_other), -1)
        interaction = torch.mul(state, actions)
        policy_logits = torch.cat((state, actions, interaction), -1)
        policy_logits = F.relu(self.p_fc_0(policy_logits))
        policy_logits = F.relu(self.p_fc_1(policy_logits))
        policy_logits = F.relu(self.p_fc_2(policy_logits))
        policy_logits = self.p_fc_3(policy_logits).view(-1,self.num_actions)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )

    def cost_forward(self, inputs):
        tables = inputs["tables"] # list(B) * num_tables * table_dim
        offset = inputs["offset"] # B
        B = offset.shape[0]

        tables = torch.cat(tables, dim=0)
        tables = F.relu(self.t_fc_0(tables))
        tables = F.relu(self.t_fc_1(tables))
        
        ind = torch.repeat_interleave(torch.arange(len(offset)), offset)
        tmp = torch.zeros((offset.shape[0], tables.shape[1]))
        tmp.index_add_(0, ind, tables)
        tables = tmp

        cost = F.relu(self.c_fc_0(tables))
        cost = self.c_fc_1(cost)

        cost = cost.view(B)

        return cost
