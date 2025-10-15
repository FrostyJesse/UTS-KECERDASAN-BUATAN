%%writefile train_dqn.py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from rocket_env import SimpleRocketEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.model(x)

def select_action(state, policy_net, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state)
        return q_values.argmax().item()

def compute_loss(batch, policy_net, target_net, gamma):
    states = torch.FloatTensor(np.vstack(batch.state)).to(device)
    actions = torch.LongTensor(batch.action).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(np.vstack(batch.next_state)).to(device)
    dones = torch.FloatTensor(batch.done).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
        q_targets = rewards + (1 - dones) * gamma * next_q_values
    return nn.MSELoss()(q_values, q_targets)

def train_dqn():
    env = SimpleRocketEnv(render_mode=None)
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    policy_net = DQN(state_dim, n_actions).to(device)
    target_net = DQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    buffer = ReplayBuffer(100000)
    gamma = 0.99
    eps_start, eps_end, eps_decay = 1.0, 0.05, 200000
    batch_size = 64
    total_steps = 300000
    update_every, target_update = 4, 1000

    rewards_log = []
    state, _ = env.reset()
    episode_reward = 0

    for step in range(total_steps):
        eps = eps_end + (eps_start - eps_end) * np.exp(-1. * step / eps_decay)
        action = select_action(state, policy_net, eps, n_actions)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if len(buffer) > batch_size and step % update_every == 0:
            batch = buffer.sample(batch_size)
            loss = compute_loss(batch, policy_net, target_net, gamma)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

        if step % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            rewards_log.append(episode_reward)
            episode_reward = 0
            state, _ = env.reset()

        if (step + 1) % 5000 == 0:
            avg = np.mean(rewards_log[-20:]) if len(rewards_log) else 0
            print(f"Step {step+1}, Epsilon {eps:.3f}, Avg Reward {avg:.2f}")

    torch.save(policy_net.state_dict(), "dqn_rocket_model.pth")
    env.close()
    plt.figure(figsize=(8,4))
    plt.plot(rewards_log, color='black')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve - DQN Rocket")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("learning_curve_bw.png", dpi=200)
    plt.close()
    print("Training selesai, model disimpan sebagai dqn_rocket_model.pth")

if __name__ == "__main__":
    train_dqn()
