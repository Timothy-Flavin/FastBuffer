# Threaded RL runner with double models and per-thread FastBuffer
from threading import Thread, Lock, Event, Condition
import gymnasium as gym
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from FastBuffer import FastBuffer
import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ModelManager:
    """Manages two CPU actor models and a single GPU training model.

    Pipeline:
    1) Actors collect with cpu_models[active_idx]; gpu_model is a copy at swap time.
    2) Trainer updates gpu_model from buffers.
    3) Occasionally copy gpu_model -> cpu_models[inactive_idx] and then flip active_idx.
    """

    def __init__(self, cpu_models: list[nn.Module], gpu_model: nn.Module):
        assert len(cpu_models) == 2, "Expect exactly two CPU models"
        self.cpu_models = cpu_models
        self.gpu_model = gpu_model
        self._active_idx = 0
        self._usage = [0, 0]
        self._lock = Lock()
        self._cond = Condition(self._lock)

    def get_active_idx(self) -> int:
        with self._lock:
            return self._active_idx

    def get_inactive_idx(self) -> int:
        with self._lock:
            return 1 - self._active_idx

    def acquire_active_cpu(self) -> tuple[nn.Module, int]:
        """Increment usage for active model and return it with its index.
        Call release(idx) when done.
        """
        with self._lock:
            idx = self._active_idx
            self._usage[idx] += 1
            return self.cpu_models[idx], idx

    def release(self, idx: int):
        with self._lock:
            self._usage[idx] -= 1
            if self._usage[idx] == 0:
                self._cond.notify_all()

    def copy_active_cpu_to_gpu(self):
        """Copy active CPU model weights into the single GPU model."""
        active_sd = self.cpu_models[self._active_idx].state_dict()
        self.gpu_model.load_state_dict(
            {
                k: v.to(next(self.gpu_model.parameters()).device)
                for k, v in active_sd.items()
            }
        )

    def swap_from_gpu(self):
        """Copy GPU -> inactive CPU, then make it active.

        Ensures old active has zero readers before flipping.
        After swap, actors use the CPU model that was just synced from GPU.
        """
        with self._lock:
            old_active = self._active_idx
            new_active = 1 - old_active
            # Wait until old active CPU model has no active users
            while self._usage[old_active] > 0:
                self._cond.wait(timeout=0.01)

            # Copy GPU -> inactive CPU
            gpu_sd = self.gpu_model.state_dict()
            self.cpu_models[new_active].load_state_dict(
                {k: v.cpu() for k, v in gpu_sd.items()}
            )

            # Flip active
            self._active_idx = new_active

    def get_training_gpu(self) -> nn.Module:
        return self.gpu_model


def select_action(state, manager: ModelManager, device, n_actions):
    with torch.no_grad():
        # Inference on CPU for better throughput with many small batches
        state_t = torch.as_tensor(state, device="cpu").unsqueeze(0).float()
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * steps_done / EPS_DECAY
        )
        steps_done += 1
        if sample > eps_threshold:
            model, idx = manager.acquire_active_cpu()
            try:
                return model(state_t).argmax(dim=1).item()
            finally:
                manager.release(idx)
        else:
            return random.randrange(n_actions)


def optimize_model(model, buffer: FastBuffer, optimizer, device):
    global BATCH_SIZE, GAMMA

    if buffer.gpu_steps_recorded < BATCH_SIZE:
        return
    transitions, idx = buffer.sample_transitions(BATCH_SIZE)

    obs = transitions["obs"].to(device)
    obs_ = transitions["obs_"].to(device)
    acts = transitions["discrete_actions"].to(device)
    rews = transitions["global_rewards"].to(device)
    term = transitions.get("terminated")
    if term is None:
        term = torch.zeros((BATCH_SIZE,), dtype=torch.bool, device=device)
    else:
        term = term.to(device).bool()

    # Q(s,a)
    q_values = model(obs)
    q = q_values.gather(1, acts.long().unsqueeze(-1))[:, 0]

    with torch.no_grad():
        q_next = model(obs_).max(1).values
        target = GAMMA * q_next * (~term) + rews

    criterion = nn.MSELoss()
    loss = criterion(q, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def update_worker(
    manager: ModelManager,
    buffers,
    buffer_locks,
    optimizer,
    device,
    stop_event: Event,
    switch_every: int = 5,
):
    n_step = 0
    # Ratio gating to prevent training from outpacing data collection
    global env_steps_counter, update_steps_counter
    while not stop_event.is_set():
        # throttle updates based on collected env steps

        model = manager.get_training_gpu()
        # Train across buffers
        for i, buffer in enumerate(buffers):
            if stop_event.is_set():
                break
            if buffer.gpu_steps_recorded >= BATCH_SIZE * 4:
                with buffer_locks[i]:
                    optimize_model(model, buffer, optimizer, device)

        n_step += 1
        # print(f"Update step {n_step} done")
        # Periodic swap
        if n_step % switch_every == 0:
            # After some training, deploy GPU -> inactive CPU and flip
            manager.swap_from_gpu()


def rollout_worker(
    env,
    manager: ModelManager,
    buffer: FastBuffer,
    buffer_lock: Lock,
    num_steps: int,
    device,
    n_actions: int,
    thread_id: int,
    ep_lengths: list,
    stop_event: Event,
):
    global env_steps_counter
    i_episode = 0
    step = 0
    while step < num_steps and not stop_event.is_set():
        state, info = env.reset()
        done = False
        t = 0
        while not done and not stop_event.is_set():
            t += 1
            action = select_action(state, manager, device, n_actions)
            state_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.save_transition(
                data={
                    "global_rewards": float(reward),
                    "obs": np.asarray(state, dtype=np.float32),
                    "obs_": np.asarray(state_, dtype=np.float32),
                    "discrete_actions": int(action),
                    "terminated": int(terminated),
                },
            )

            state = state_

            # Periodically push to GPU
            if buffer.steps_recorded % 512 == 0:
                with buffer_lock:
                    buffer.update_gpu()

            if done:
                # print(f"[Thread {thread_id}] ep_len: {t} ep: {i_episode} total_step: {step}")
                ep_lengths.append(t)
                i_episode += 1
            step += 1

    # Final GPU sync for any remaining data
    with buffer_lock:
        buffer.update_gpu()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_STEPS = 200000
    BATCH_SIZE = 512
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 50000
    LR = 3e-4
    # shared counters for gating
    env_steps_counter = 0
    update_steps_counter = 0
    counters_lock = Lock()

    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n  # type: ignore
    state, info = env.reset()
    n_observations = len(state)

    # Two CPU actor models and one GPU training model
    cpu_model0 = DQN(n_observations, n_actions).to("cpu").float()
    cpu_model1 = DQN(n_observations, n_actions).to("cpu").float()
    gpu_model = DQN(n_observations, n_actions).to(device).float()
    # Initialize the inactive cpu model with the same weights
    cpu_model1.load_state_dict(cpu_model0.state_dict())
    manager = ModelManager([cpu_model0, cpu_model1], gpu_model)
    # Copy active CPU -> GPU at start (pipeline step 1 -> 2)
    manager.copy_active_cpu_to_gpu()
    optimizer = optim.AdamW(gpu_model.parameters(), lr=LR, amsgrad=True)
    steps_done = 0

    # Memory buffers: one per CPU worker
    total_memory = 20000
    num_cpu_workers = 8
    per_buffer = total_memory // num_cpu_workers
    buffers = []
    buffer_locks: list[Lock] = []
    for _ in range(num_cpu_workers):
        buffers.append(
            FastBuffer(
                buffer_len=per_buffer,
                n_agents=1,
                discrete_cardonalities=[n_actions],
                registered_vars={
                    "global_rewards": {
                        "dim": None,
                        "dtype": torch.float32,
                        "per_agent": False,
                    },
                    "obs": {
                        "dim": n_observations,
                        "dtype": torch.float32,
                        "per_agent": False,
                    },
                    "obs_": {
                        "dim": n_observations,
                        "dtype": torch.float32,
                        "per_agent": False,
                    },
                    "discrete_actions": {
                        "dim": None,
                        "dtype": torch.int64,
                        "per_agent": False,
                    },
                    "terminated": {
                        "dim": None,
                        "dtype": torch.int64,
                        "per_agent": False,
                    },
                },
                gpu=(device == "cuda"),
            )
        )
        buffer_locks.append(Lock())

    # Thread management
    stop_event = Event()
    start = time.time()

    # Start update worker
    updater = Thread(
        target=update_worker,
        args=(manager, buffers, buffer_locks, optimizer, device, stop_event, 200),
        daemon=True,
    )
    updater.start()

    # Start rollout workers and collect per-thread episode lengths
    per_thread_ep_lengths: list[list[int]] = [[] for _ in range(num_cpu_workers)]
    workers = []
    for i in range(num_cpu_workers):
        w = Thread(
            target=rollout_worker,
            args=(
                gym.make("CartPole-v1"),
                manager,
                buffers[i],
                buffer_locks[i],
                NUM_STEPS // num_cpu_workers,
                device,
                n_actions,
                i,
                per_thread_ep_lengths[i],
                stop_event,
            ),
            daemon=True,
        )
        w.start()
        workers.append(w)

    for w in workers:
        w.join()
    stop_event.set()
    updater.join(timeout=1.0)
    end = time.time()
    print(f"Finished in {end - start:.2f}s on device {device}")

    # Combine episode lengths from all threads for later plotting
    all_ep_lengths = [length for lst in per_thread_ep_lengths for length in lst]
    for lst in per_thread_ep_lengths:
        plt.plot(lst)
    plt.title("Episode Lengths over Time")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.show()
    print(f"Collected {len(all_ep_lengths)} episodes across {num_cpu_workers} threads")
    for i, lst in enumerate(per_thread_ep_lengths):
        tail = lst[-5:] if len(lst) >= 5 else lst
        print(f"Thread {i}: {len(lst)} episodes, last up to 5: {tail}")
