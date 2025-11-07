from __future__ import annotations

import math
import queue
import random
import time
from dataclasses import dataclass
from typing import Any, Dict

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from multiprocessing import shared_memory

from FastBuffer import FastBuffer

# ---------------------
# Hyperparameters
# ---------------------
NUM_STEPS = 100000
BUFFER_LEN = 10000
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-4
MODEL_SYNC_EVERY = 5
MEMORY_SYNC_EVERY = 1024
# Size of batches staged in shared memory from collector to learner.
STAGING_BATCH = MEMORY_SYNC_EVERY
DELAY_ENV = True
if DELAY_ENV:
    NUM_STEPS = 5000


# ---------------------
# Model definition
# ---------------------
class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# ---------------------
# Utilities
# ---------------------
@dataclass
class Transition:
    obs: Any
    obs_: Any
    action: int
    reward: float
    terminated: int  # 0/1


def get_action(
    model: nn.Module, obs: torch.Tensor, epsilon: float, max_action: int
) -> int:
    if random.random() < epsilon:
        return random.randint(0, max_action - 1)
    with torch.no_grad():
        return model(obs).argmax(-1).item()


def update_model(
    model: nn.Module, batch: Dict[str, torch.Tensor], optimizer: optim.Optimizer
):
    old_q = (
        model(batch["obs"])
        .gather(-1, batch["discrete_actions"].unsqueeze(-1))
        .squeeze(-1)
    )
    with torch.no_grad():
        new_q: torch.Tensor = torch.max(model(batch["obs_"]), dim=-1).values
        target = batch["rewards"] + GAMMA * (1 - batch["terminated"]) * new_q
    loss = ((old_q - target) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ---------------------
# Shared-memory staging (double-buffered)
# ---------------------


@dataclass
class ShmSpec:
    name: str
    shape: tuple
    dtype: str  # numpy dtype string like 'float32' / 'int64'


@dataclass
class StagingSlotSpec:
    obs: ShmSpec
    obs_: ShmSpec
    actions: ShmSpec
    rewards: ShmSpec
    terminated: ShmSpec


def create_shm_array(ctx, shape, dtype):
    import numpy as np

    nbytes = int(np.empty(shape, dtype=dtype).nbytes)
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    return shm, ShmSpec(name=shm.name, shape=tuple(shape), dtype=np.dtype(dtype).name)


def shm_to_ndarray(spec: ShmSpec):
    import numpy as np

    shm = shared_memory.SharedMemory(name=spec.name)
    arr = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype), buffer=shm.buf)
    return shm, arr


# ---------------------
# Collector process
# ---------------------


def collector_process(
    env_name: str,
    slot_specs: dict,
    slot_flags,
    slot_sizes,
    weight_queue,
    stop_event,
    steps_limit: int,
):
    """Runs the environment and publishes micro-batches into shared-memory slots.

    The learner drains these slots into its own FastBuffer (with pinned CPU tensors) and
    performs H2D copies on its own streams, preserving the original FastBuffer semantics.
    """
    import numpy as np
    from gymnasium import spaces

    env = gym.make(env_name)
    obs0, _ = env.reset()
    n_observations = len(obs0)
    assert isinstance(
        env.action_space, spaces.Discrete
    ), "This runner expects a discrete action space"
    n_actions = int(env.action_space.n)

    # Double-buffered CPU models for action selection
    cpu_models = [DQN(n_observations, n_actions), DQN(n_observations, n_actions)]
    active_model = 0

    # Attach to shared-memory slot arrays
    slot0 = slot_specs["slot0"]
    slot1 = slot_specs["slot1"]
    s0 = {}
    s1 = {}
    s0["obs_shm"], s0["obs"] = shm_to_ndarray(slot0["obs"])  # type: ignore[index]
    s0["obs__shm"], s0["obs_"] = shm_to_ndarray(slot0["obs_"])  # type: ignore[index]
    s0["act_shm"], s0["actions"] = shm_to_ndarray(slot0["actions"])  # type: ignore[index]
    s0["rew_shm"], s0["rewards"] = shm_to_ndarray(slot0["rewards"])  # type: ignore[index]
    s0["term_shm"], s0["terminated"] = shm_to_ndarray(slot0["terminated"])  # type: ignore[index]

    s1["obs_shm"], s1["obs"] = shm_to_ndarray(slot1["obs"])  # type: ignore[index]
    s1["obs__shm"], s1["obs_"] = shm_to_ndarray(slot1["obs_"])  # type: ignore[index]
    s1["act_shm"], s1["actions"] = shm_to_ndarray(slot1["actions"])  # type: ignore[index]
    s1["rew_shm"], s1["rewards"] = shm_to_ndarray(slot1["rewards"])  # type: ignore[index]
    s1["term_shm"], s1["terminated"] = shm_to_ndarray(slot1["terminated"])  # type: ignore[index]

    # Local accumulation buffers
    acc = {
        "obs": np.zeros((STAGING_BATCH, n_observations), dtype=np.float32),
        "obs_": np.zeros((STAGING_BATCH, n_observations), dtype=np.float32),
        "actions": np.zeros((STAGING_BATCH,), dtype=np.int64),
        "rewards": np.zeros((STAGING_BATCH,), dtype=np.float32),
        "terminated": np.zeros((STAGING_BATCH,), dtype=np.int64),
    }
    acc_i = 0

    overall_steps = 0
    episodes = 0
    last_print = 0
    smooth_reward = 0.0
    step_start_time = time.time()
    obs = torch.tensor(obs0, dtype=torch.float32)
    slot_toggle = 0

    def try_publish(batch_size: int):
        nonlocal slot_toggle
        if batch_size <= 0:
            return
        # Choose slot
        s = s0 if slot_toggle == 0 else s1
        flag = slot_flags[slot_toggle]
        size_val = slot_sizes[slot_toggle]
        if flag.value == 1:
            # Slot full; drop this micro-batch (no blocking). Toggle to try the other next time.
            slot_toggle = 1 - slot_toggle
            return
        # Copy into slot arrays
        s["obs"][:batch_size] = acc["obs"][:batch_size]
        s["obs_"][:batch_size] = acc["obs_"][:batch_size]
        s["actions"][:batch_size] = acc["actions"][:batch_size]
        s["rewards"][:batch_size] = acc["rewards"][:batch_size]
        s["terminated"][:batch_size] = acc["terminated"][:batch_size]
        size_val.value = int(batch_size)
        flag.value = 1  # mark ready
        slot_toggle = 1 - slot_toggle

    while overall_steps < steps_limit and not stop_event.is_set():
        # Apply any pending weight updates
        try:
            while True:
                state_dict = weight_queue.get_nowait()
                if state_dict is None:
                    break
                inactive = 1 - active_model
                cpu_models[inactive].load_state_dict(state_dict)
                active_model = inactive
        except queue.Empty:
            pass

        # New episode
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        done = False
        ep_reward = 0.0

        while not done and overall_steps < steps_limit and not stop_event.is_set():
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(
                -1.0 * overall_steps / EPS_DECAY
            )
            model = cpu_models[active_model]
            action = get_action(model, obs.unsqueeze(0), epsilon, n_actions)
            next_obs, reward, terminated, truncated, _info = env.step(action)

            if DELAY_ENV:
                for q in range(100000):
                    _ = q / 7

            # Accumulate locally
            acc["obs"][acc_i] = obs.numpy()
            acc["obs_"][acc_i] = torch.as_tensor(next_obs, dtype=torch.float32).numpy()
            acc["actions"][acc_i] = int(action)
            acc["rewards"][acc_i] = float(reward)
            acc["terminated"][acc_i] = int(bool(terminated))
            acc_i += 1

            # Publish when full
            if acc_i >= STAGING_BATCH:
                try_publish(acc_i)
                acc_i = 0

            obs = torch.tensor(next_obs, dtype=torch.float32)
            ep_reward += float(reward)
            done = bool(terminated) or bool(truncated)
            overall_steps += 1

            if overall_steps > last_print:
                smooth_reward = (
                    0.95 * smooth_reward + 0.05 * ep_reward
                    if episodes > 0
                    else ep_reward
                )
                rate = overall_steps / max(1e-6, (time.time() - step_start_time))
                print(
                    f"Collector | Step {overall_steps} at {rate:.1f} s/s | EpR: {ep_reward:.2f} | Smooth: {smooth_reward:.2f}"
                )
                last_print = overall_steps + 1000

        if done:
            episodes += 1

    # Flush any partial batch
    try_publish(acc_i)
    env.close()


# ---------------------
# Learner process
# ---------------------


def learner_process(
    env_name: str, slot_specs: dict, slot_flags, slot_sizes, weight_queue, stop_event
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Infer obs/action shapes from a short-lived env
    env = gym.make(env_name)
    obs0, _ = env.reset()
    n_observations = len(obs0)
    from gymnasium import spaces

    assert isinstance(
        env.action_space, spaces.Discrete
    ), "This runner expects a discrete action space"
    n_actions = env.action_space.n
    env.close()

    # GPU model + optimizer
    gpu_model = DQN(n_observations, int(n_actions)).to(device)
    optimizer = optim.Adam(gpu_model.parameters(), lr=LR)

    # Replay buffer owned by learner (with GPU double buffer)
    buffer = FastBuffer(
        buffer_len=BUFFER_LEN,
        n_agents=1,
        gpu_double_buffer=True,
        action_mask=False,
        discrete_cardonalities=None,
        registered_vars={
            "rewards": {"dim": None, "dtype": torch.float32, "per_agent": False},
            "obs": {"dim": n_observations, "dtype": torch.float32, "per_agent": False},
            "obs_": {"dim": n_observations, "dtype": torch.float32, "per_agent": False},
            "discrete_actions": {"dim": None, "dtype": torch.int64, "per_agent": False},
            "terminated": {"dim": None, "dtype": torch.int64, "per_agent": False},
        },
    )

    # Attach to shared-memory slots
    s0_spec = slot_specs["slot0"]
    s1_spec = slot_specs["slot1"]
    s0 = {}
    s1 = {}
    s0["obs_shm"], s0["obs"] = shm_to_ndarray(s0_spec["obs"])  # type: ignore[index]
    s0["obs__shm"], s0["obs_"] = shm_to_ndarray(s0_spec["obs_"])  # type: ignore[index]
    s0["act_shm"], s0["actions"] = shm_to_ndarray(s0_spec["actions"])  # type: ignore[index]
    s0["rew_shm"], s0["rewards"] = shm_to_ndarray(s0_spec["rewards"])  # type: ignore[index]
    s0["term_shm"], s0["terminated"] = shm_to_ndarray(s0_spec["terminated"])  # type: ignore[index]

    s1["obs_shm"], s1["obs"] = shm_to_ndarray(s1_spec["obs"])  # type: ignore[index]
    s1["obs__shm"], s1["obs_"] = shm_to_ndarray(s1_spec["obs_"])  # type: ignore[index]
    s1["act_shm"], s1["actions"] = shm_to_ndarray(s1_spec["actions"])  # type: ignore[index]
    s1["rew_shm"], s1["rewards"] = shm_to_ndarray(s1_spec["rewards"])  # type: ignore[index]
    s1["term_shm"], s1["terminated"] = shm_to_ndarray(s1_spec["terminated"])  # type: ignore[index]

    updates = 0
    learn_start = time.time()
    last_sync = 0
    t_sample = 0.0
    t_compute = 0.0

    def drain_slot(s, flag, size_val):
        nonlocal buffer
        if flag.value != 1:
            return 0
        n = int(size_val.value)
        if n <= 0:
            flag.value = 0
            return 0
        # Vectorized batch copy into FastBuffer (CPU side)
        batch_dict = {
            "obs": torch.as_tensor(s["obs"][:n], dtype=torch.float32),
            "obs_": torch.as_tensor(s["obs_"][:n], dtype=torch.float32),
            "discrete_actions": torch.as_tensor(s["actions"][:n], dtype=torch.int64),
            "rewards": torch.as_tensor(s["rewards"][:n], dtype=torch.float32),
            "terminated": torch.as_tensor(s["terminated"][:n], dtype=torch.int64),
        }
        buffer.save_batch(batch_dict, n)
        flag.value = 0
        # Trigger H2D after each full micro-batch to maintain cadence
        if buffer.steps_recorded % MEMORY_SYNC_EVERY == 0:
            buffer.update_gpu()
        return n

    # Main learner loop: train continuously
    while not stop_event.is_set():
        # Drain any ready slot data without blocking
        drained = 0
        drained += drain_slot(s0, slot_flags[0], slot_sizes[0])
        drained += drain_slot(s1, slot_flags[1], slot_sizes[1])

        # Train if enough GPU-side samples are available
        if buffer.gpu_steps_recorded >= BATCH_SIZE * 2:
            if getattr(buffer, "copy_event", None) is not None:
                torch.cuda.current_stream().wait_event(buffer.copy_event)  # type: ignore[arg-type]
            t0 = time.time()
            batch, _ = buffer.sample_transitions(BATCH_SIZE)
            t1 = time.time()
            update_model(gpu_model, batch, optimizer)
            t2 = time.time()
            updates += 1
            t_sample += t1 - t0
            t_compute += t2 - t1

            if updates - last_sync >= MODEL_SYNC_EVERY:
                state_cpu = {
                    k: v.detach().cpu() for k, v in gpu_model.state_dict().items()
                }
                try:
                    while True:
                        try:
                            weight_queue.put(state_cpu, timeout=0.01)
                            break
                        except queue.Full:
                            try:
                                _ = weight_queue.get_nowait()
                            except queue.Empty:
                                break
                except Exception:
                    pass
                last_sync = updates

            if updates % 1024 == 0:
                rate = updates / max(1e-6, (time.time() - learn_start))
                print(
                    f"Learner | updates: {updates} at {rate:.1f} u/s | sample {t_sample:.3f}s | compute {t_compute:.3f}s"
                )

        else:
            # Brief sleep to yield if nothing to train yet
            time.sleep(0.001)


# ---------------------
# Main (Windows-safe)
# ---------------------


def main():
    import numpy as np
    import multiprocessing as mp

    env_name = "CartPole-v1"

    ctx = mp.get_context("spawn")  # Windows safe

    # Build short-lived env to infer shapes
    env = gym.make(env_name)
    obs0, _ = env.reset()
    n_observations = len(obs0)
    env.close()

    # Create two shared-memory slots
    slot_specs: dict = {"slot0": {}, "slot1": {}}
    # slot arrays shapes
    obs_shape = (STAGING_BATCH, n_observations)
    vec_shape = (STAGING_BATCH,)

    # Slot 0
    shm0_obs, spec0_obs = create_shm_array(ctx, obs_shape, np.float32)
    shm0_obs_, spec0_obs_ = create_shm_array(ctx, obs_shape, np.float32)
    shm0_act, spec0_act = create_shm_array(ctx, vec_shape, np.int64)
    shm0_rew, spec0_rew = create_shm_array(ctx, vec_shape, np.float32)
    shm0_term, spec0_term = create_shm_array(ctx, vec_shape, np.int64)
    slot_specs["slot0"] = {
        "obs": spec0_obs,
        "obs_": spec0_obs_,
        "actions": spec0_act,
        "rewards": spec0_rew,
        "terminated": spec0_term,
    }

    # Slot 1
    shm1_obs, spec1_obs = create_shm_array(ctx, obs_shape, np.float32)
    shm1_obs_, spec1_obs_ = create_shm_array(ctx, obs_shape, np.float32)
    shm1_act, spec1_act = create_shm_array(ctx, vec_shape, np.int64)
    shm1_rew, spec1_rew = create_shm_array(ctx, vec_shape, np.float32)
    shm1_term, spec1_term = create_shm_array(ctx, vec_shape, np.int64)
    slot_specs["slot1"] = {
        "obs": spec1_obs,
        "obs_": spec1_obs_,
        "actions": spec1_act,
        "rewards": spec1_rew,
        "terminated": spec1_term,
    }

    # Slot flags and sizes (synchronized Values)
    slot_flags = [ctx.Value("i", 0), ctx.Value("i", 0)]  # 0=empty, 1=ready
    slot_sizes = [ctx.Value("i", 0), ctx.Value("i", 0)]  # actual batch sizes

    # Tiny queue for weights
    weight_queue = ctx.Queue(maxsize=2)
    stop_event = ctx.Event()

    collector = ctx.Process(
        target=collector_process,
        args=(
            env_name,
            slot_specs,
            slot_flags,
            slot_sizes,
            weight_queue,
            stop_event,
            NUM_STEPS,
        ),
        name="collector",
    )
    learner = ctx.Process(
        target=learner_process,
        args=(env_name, slot_specs, slot_flags, slot_sizes, weight_queue, stop_event),
        name="learner",
    )

    start_time = time.time()
    collector.start()
    learner.start()

    collector.join()
    stop_event.set()
    learner.join()

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds (MP, shared-mem).")


if __name__ == "__main__":
    main()
