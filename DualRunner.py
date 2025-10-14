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

NUM_STEPS = 50000
BUFFER_LEN = 10000
BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-4
MODEL_SYNC_EVERY = 10
MEMORY_SYNC_EVERY = 1024


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
    def __init__(self, cpu_models, gpu_model, optimizer, n_buffers=1):
        self.active_model = 0
        self.cpu_models = cpu_models
        self.gpu_model = gpu_model
        self.gpu_buffer_locks = []
        for _ in range(n_buffers):
            self.gpu_buffer_locks.append(Lock())
        self.n_buffers = n_buffers
        self.optimizer = optimizer
        self.collectors_done = [False for _ in range(n_buffers)]

    def get_active_model(self):
        return self.active_model


def get_action(model, obs, epsilon, max_action):
    if random.random() < epsilon:
        return random.randint(0, max_action - 1)
    else:
        with torch.no_grad():
            return model(obs).argmax(-1).cpu().item()


def update_model(model: nn.Module, batch: dict, optimizer: optim.Optimizer):
    global GAMMA, MODEL_SYNC_EVERY, MEMORY_SYNC_EVERY
    old_q = (
        model(batch["obs"])
        .gather(-1, batch["discrete_actions"].unsqueeze(-1))
        .squeeze(-1)
    )
    # print(old_q.shape, batch["rewards"].shape, batch["terminated"].shape)
    with torch.no_grad():
        new_q: torch.Tensor = torch.max(model(batch["obs_"]), dim=-1).values
        # print(new_q.shape)
        target = batch["rewards"] + GAMMA * (1 - batch["terminated"]) * new_q
        # print(target.shape)
    loss = ((old_q - target) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def update_thread(model_manager: ModelManager, memory_buffers: list[FastBuffer]):
    global BATCH_SIZE
    step = 1
    gpu_net = model_manager.gpu_model
    while not all(model_manager.collectors_done):
        step_inc = 0
        for i, m in enumerate(memory_buffers):
            if m.gpu_steps_recorded < BATCH_SIZE * 2:
                time.sleep(0.01)
                continue
            step_inc = 1
            try:
                model_manager.gpu_buffer_locks[i].acquire()
                batch, idx = m.sample_transitions(BATCH_SIZE)
                update_model(gpu_net, batch, model_manager.optimizer)
            finally:
                model_manager.gpu_buffer_locks[i].release()
        step += step_inc
        if step % MODEL_SYNC_EVERY == 0:
            net = model_manager.cpu_models[1 - model_manager.active_model]
            net.load_state_dict(gpu_net.state_dict())
            model_manager.active_model = 1 - model_manager.active_model
            print(f"Model updated to {model_manager.active_model}")


def runner_thread(
    thread_id: int,
    model_manager: ModelManager,
    memory_buffer: FastBuffer,
    env_name: str,
    max_steps: int,
):
    global EPS_START, EPS_END, EPS_DECAY, MEMORY_SYNC_EVERY
    env = gym.make(env_name)
    n_actions = env.action_space.n
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    episode_rewards = []
    overall_steps = 0
    episodes = 0
    ep_rewards = []
    smooth_reward = 0.0
    while overall_steps < max_steps:
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        total_reward = 0.0
        done = False
        while not done:
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(
                -1.0 * (overall_steps) / EPS_DECAY
            )
            model_idx = model_manager.get_active_model()
            model = model_manager.cpu_models[model_idx]
            action = get_action(model, obs.unsqueeze(0), epsilon, n_actions)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            memory_buffer.save_transition(
                data={
                    "obs": obs,
                    "obs_": next_obs,
                    "discrete_actions": action,
                    "rewards": reward,
                    "terminated": terminated,
                }
            )
            obs = next_obs
            total_reward += float(reward)
            done = terminated or truncated
            overall_steps += 1

            if overall_steps % MEMORY_SYNC_EVERY == 0:
                try:
                    model_manager.gpu_buffer_locks[thread_id].acquire()
                    memory_buffer.update_gpu()
                finally:
                    model_manager.gpu_buffer_locks[thread_id].release()

        episode_rewards.append(total_reward)

        if done:
            smooth_reward = (
                0.95 * smooth_reward + 0.05 * total_reward
                if episodes > 0
                else total_reward
            )
            print(
                f"Thread {thread_id}, Overall Step {overall_steps}, Reward: {total_reward}, Smooth Reward: {smooth_reward}"
            )
            episodes += 1
            ep_rewards.append(total_reward)

    env.close()
    print(f"Thread {thread_id} finished.")
    model_manager.collectors_done[thread_id] = True
    return episode_rewards


def main():
    global NUM_STEPS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = "CartPole-v1"
    n_threads = 2
    env = gym.make(env_name)
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)
    env.close()

    cpu_models = [DQN(n_observations, n_actions), DQN(n_observations, n_actions)]
    for model in cpu_models:
        model.to("cpu")
    gpu_model = DQN(n_observations, n_actions).to(device)
    gpu_model.load_state_dict(cpu_models[0].state_dict())
    optimizer = optim.Adam(gpu_model.parameters(), lr=LR)

    model_manager = ModelManager(cpu_models, gpu_model, optimizer, n_buffers=n_threads)
    memory_buffers = []
    for _ in range(n_threads):
        memory_buffers.append(
            FastBuffer(
                buffer_len=BUFFER_LEN,
                n_agents=1,
                gpu=True,
                action_mask=False,
                discrete_cardonalities=None,
                registered_vars={
                    "rewards": {
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
            )
        )

    start_time = time.time()

    threads = []
    for i in range(n_threads):
        t = Thread(
            target=runner_thread,
            args=(
                i,
                model_manager,
                memory_buffers[i],
                env_name,
                NUM_STEPS // n_threads,
            ),
        )
        t.start()
        threads.append(t)

    update_t = Thread(target=update_thread, args=(model_manager, memory_buffers))
    update_t.start()

    for t in threads:
        t.join()
    update_t.join()
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    print("Training complete.")


if __name__ == "__main__":
    main()
