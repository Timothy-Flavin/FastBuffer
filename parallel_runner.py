# Example single agent using flexibuff
# with torch dqn example modified from
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
from threading import Thread
import gymnasium as gym
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
from FastBuffer import FastBuffer


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


def select_action(state, model):
    with torch.no_grad():
        state = torch.from_numpy(state)[None, :].to(device)
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * steps_done / EPS_DECAY
        )
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return model(state).max(1).indices.view(1, 1).cpu().item()
        else:
            return (
                torch.tensor(
                    [[env.action_space.sample()]], device=device, dtype=torch.long
                )
                .cpu()
                .item()
            )


def optimize_model(model, buffer, optimizer):
    global BATCH_SIZE
    global GAMMA

    if buffer.steps_recorded < BATCH_SIZE:
        return
    transitions, idx = buffer.sample_transitions(BATCH_SIZE)
    term = transitions["terminated"]
    # print(transitions)
    if term is None:
        term = torch.zeros((BATCH_SIZE,), dtype=torch.bool, device=device)
    # print(transitions)
    # [0] because we are doing this for just the first agent because there is only one agent
    Q = model(transitions["obs"]).gather(
        1, transitions["discrete_actions"].unsqueeze(-1)
    )[:, 0]

    # print(
    #    f"Gamma: {GAMMA} * {policy_net(transitions.obs_[0]).max(1).values} * { (1 - transitions.terminated)} + {transitions.global_rewards}"
    # )
    with torch.no_grad():  # no need to track gradient for next Q value
        Q_NEXT = (
            GAMMA  # obs [0] because we are only using 1 agent again
            * model(transitions["obs_"]).max(1).values
            * (1 - term)  # Terminated and global rewards do
            + transitions["global_rewards"]  # not need to be [0] because they
        )  # are for all agents
    # Compute MSE loss
    criterion = nn.MSELoss()
    loss = criterion(Q, Q_NEXT)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def update_worker(active_net: int, buffer_locks, models, buffers, optimizer):
    n_step = 0
    training_net = 1 - active_net
    while True:
        for i, buffer in enumerate(buffers):
            if buffer.steps_recorded > 512:
                # Perform one step of the optimization (on the policy network)
                buffer_locks[i].acquire()
                optimize_model(models[training_net], buffer, optimizer)
                buffer_locks[i].release()
            n_step += 1
        if n_step % 10 == 0:
            active_net = 1 - active_net
            training_net = 1 - active_net
            # TODO: wait until all actions on the active net are done then copy the weights
            models[training_net].load_state_dict(models[active_net].state_dict())


def rollout_worker(
    env, models, buffers, buffer_locks, num_steps, thread_id=0, active_net: int = 0
):
    ts = []
    es = []
    i_episode = 0
    step = 0
    while step < num_steps:
        # Initialize the environment and get its state
        es.append(0)
        done = False
        state, info = env.reset()
        t = 0
        while not done:
            t += 1
            # TODO: Mark that this network is in action so the update thread waits to copy weights
            action = select_action(state, models[active_net])
            state_, reward, terminated, truncated, _ = env.step(action)
            es[-1] += reward
            done = terminated or truncated

            # Store the transition in memory which accepts np arrays
            buffers[thread_id].save_transition(
                data={
                    "global_rewards": reward,
                    "obs": state,
                    "obs_": state_,
                    "discrete_actions": [action],
                    "terminated": terminated,
                },
            )

            # Move to the next state
            state = state_
            if buffers[thread_id].steps_recorded % 512 == 0:
                # acquire the gpu lock and update gpu tensors
                buffer_locks[thread_id].acquire()
                buffers[thread_id].update_gpu()
                buffer_locks[thread_id].release()

            if done:
                print(f"len: {t} ep: {i_episode} total_step: {step}")
                ts.append(t)
                i_episode += 1
            step += 1
    return ts, es


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_STEPS = 50000
    BATCH_SIZE = 256
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    LR = 1e-4

    env = gym.make("CartPole-v1")
    # Get number of actions from gym action space
    n_actions = env.action_space.n  # type: ignore
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    policy_net1 = DQN(n_observations, n_actions).to(device).float()
    policy_net2 = DQN(n_observations, n_actions).to(device).float()
    policy_net2.load_state_dict(policy_net1.state_dict())
    optimizer1 = optim.AdamW(policy_net1.parameters(), lr=LR, amsgrad=True)
    optimizer2 = optim.AdamW(policy_net2.parameters(), lr=LR, amsgrad=True)

    active_net = policy_net1
    active_opt = optimizer1
    steps_done = 0

    # Set up the memory buffer for use with one agent,
    # global reward and one discrete action output
    total_memory = 10000
    num_cpus_buffers = 1
    memories = []
    for _ in range(num_cpus_buffers):
        memories.append(
            FastBuffer(
                buffer_len=total_memory // num_cpus_buffers,
                n_agents=1,
                discrete_cardonalities=[2],
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
                gpu=device == "cuda",
            )
        )

    start = time.time()
    Thread(
        target=update_worker, args=(env, policy_net2, memories, optimizer2, NUM_STEPS)
    ).start()
    for c in range(num_cpus_buffers):
        Thread(
            target=rollout_worker,
            args=(env, policy_net1, memories[c], NUM_STEPS),
        ).start()
    end = time.time()

    # print(f"Time elapsed: {end-start}s on device: {device}")
    # print("Complete")
    # plt.plot(ep_rewards)
    # plt.title(f"rewards over {end-start}s")
    # plt.show()
