"""
15x15 Grid Maze RL: SAC-Discrete vs PPO Comparison
- Environment: 15x15 maze with random obstacles
- Agent: point robot, 4 discrete actions (up/down/left/right)
- State: 8-dim obstacle detection + 8-dim goal direction = 16 dim
- Training: 500K steps each
- Output: training curves, comparison plots, importance heatmap
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# Environment
# ============================================================

class MazeEnv:
    DIRECTIONS_8 = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]  # N,NE,E,SE,S,SW,W,NW
    ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right

    def __init__(self, size=15, obstacle_ratio=0.2, max_steps=200, seed=None):
        self.size = size
        self.obstacle_ratio = obstacle_ratio
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0
        self._generate_maze()

    def _generate_maze(self):
        while True:
            self.grid = np.zeros((self.size, self.size), dtype=np.int32)
            n_obstacles = int(self.size * self.size * self.obstacle_ratio)
            positions = [(r, c) for r in range(self.size) for c in range(self.size)]
            positions.remove((0, 0))
            positions.remove((self.size-1, self.size-1))
            chosen = self.rng.choice(len(positions), size=min(n_obstacles, len(positions)), replace=False)
            for idx in chosen:
                r, c = positions[idx]
                self.grid[r, c] = 1
            if self._bfs_connected((0, 0), (self.size-1, self.size-1)):
                break

    def _bfs_connected(self, start, end):
        visited = set()
        queue = deque([start])
        visited.add(start)
        while queue:
            r, c = queue.popleft()
            if (r, c) == end:
                return True
            for dr, dc in self.ACTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in visited and self.grid[nr, nc] == 0:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        state = np.zeros(16, dtype=np.float32)
        r, c = self.agent_pos
        for i, (dr, dc) in enumerate(self.DIRECTIONS_8):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size or self.grid[nr, nc] == 1:
                state[i] = 1.0
        gr, gc = self.goal_pos
        dr, dc = gr - r, gc - c
        if dr == 0 and dc == 0:
            state[8:16] = 0.0
        else:
            angle = np.arctan2(dc, -dr)
            if angle < 0:
                angle += 2 * np.pi
            sector = int(angle / (2 * np.pi / 8)) % 8
            state[8 + sector] = 1.0
        return state

    def step(self, action):
        self.steps += 1
        dr, dc = self.ACTIONS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        old_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])

        if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size or self.grid[nr, nc] == 1:
            reward = -0.2
            done = False
        else:
            self.agent_pos = [nr, nc]
            if self.agent_pos == self.goal_pos:
                reward = 10.0
                done = True
            else:
                new_dist = abs(nr - self.goal_pos[0]) + abs(nc - self.goal_pos[1])
                reward = 0.1 if new_dist < old_dist else -0.05
                done = False

        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done, {}


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ============================================================
# SAC-Discrete
# ============================================================

class SACActorDiscrete(nn.Module):
    def __init__(self, state_dim=16, action_dim=4, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = F.softmax(logits, dim=-1)
        return probs


class SACCriticDiscrete(nn.Module):
    def __init__(self, state_dim=16, action_dim=4, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SACDiscrete:
    def __init__(self, state_dim=16, action_dim=4, lr=3e-4, gamma=0.99, tau=0.005, alpha_lr=3e-4):
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.actor = SACActorDiscrete(state_dim, action_dim).to(device)
        self.critic1 = SACCriticDiscrete(state_dim, action_dim).to(device)
        self.critic2 = SACCriticDiscrete(state_dim, action_dim).to(device)
        self.critic1_target = SACCriticDiscrete(state_dim, action_dim).to(device)
        self.critic2_target = SACCriticDiscrete(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr)

        self.target_entropy = -np.log(1.0 / action_dim) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = self.actor(state_t)
        if evaluate:
            return probs.argmax(dim=-1).item()
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)

        with torch.no_grad():
            next_probs = self.actor(next_states_t)
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q1 = self.critic1_target(next_states_t)
            next_q2 = self.critic2_target(next_states_t)
            next_q = torch.min(next_q1, next_q2)
            next_v = (next_probs * (next_q - self.alpha.detach() * next_log_probs)).sum(dim=-1, keepdim=True)
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_v

        q1 = self.critic1(states_t).gather(1, actions_t.unsqueeze(1))
        q2 = self.critic2(states_t).gather(1, actions_t.unsqueeze(1))
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        probs = self.actor(states_t)
        log_probs = torch.log(probs + 1e-8)
        q1_val = self.critic1(states_t)
        q2_val = self.critic2(states_t)
        q_val = torch.min(q1_val, q2_val)
        actor_loss = (probs * (self.alpha.detach() * log_probs - q_val)).sum(dim=-1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        entropy = -(probs * log_probs).sum(dim=-1).mean()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach())
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# ============================================================
# PPO
# ============================================================

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim=16, action_dim=4, hidden=128):
        super().__init__()
        self.shared1 = nn.Linear(state_dim, hidden)
        self.shared2 = nn.Linear(hidden, hidden)
        self.actor_head = nn.Linear(hidden, action_dim)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.shared1(x))
        x = F.relu(self.shared2(x))
        logits = self.actor_head(x)
        value = self.critic_head(x)
        return logits, value

    def get_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = self.forward(state_t)
        probs = F.softmax(logits, dim=-1)
        if evaluate:
            return probs.argmax(dim=-1).item(), None, None
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()


class PPO:
    def __init__(self, state_dim=16, action_dim=4, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_ratio=0.2, epochs=10, batch_size=64):
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size

        self.ac = PPOActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            if dones[t]:
                gae = delta
            else:
                gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, states, actions, log_probs_old, returns, advantages):
        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        log_probs_old_t = torch.FloatTensor(log_probs_old).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        dataset_size = len(states)
        for _ in range(self.epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                logits, values = self.ac(states_t[idx])
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs_new = dist.log_prob(actions_t[idx])
                entropy = dist.entropy().mean()

                ratio = (log_probs_new - log_probs_old_t[idx]).exp()
                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_t[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values.squeeze(), returns_t[idx])
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()


# ============================================================
# Training Functions
# ============================================================

def train_sac(env, total_steps=500000, warmup=10000, batch_size=256, update_every=1):
    agent = SACDiscrete()
    buffer = ReplayBuffer()
    episode_rewards = []
    ep_reward = 0
    state = env.reset()
    step = 0

    while step < total_steps:
        if step < warmup:
            action = random.randint(0, 3)
        else:
            action = agent.select_action(state)

        next_state, reward, done, _ = env.step(action)
        buffer.push(state, action, reward, next_state, float(done))
        ep_reward += reward
        state = next_state
        step += 1

        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0
            state = env.reset()

        if step >= warmup and len(buffer) >= batch_size and step % update_every == 0:
            batch = buffer.sample(batch_size)
            agent.update(batch)

        if step % 50000 == 0:
            print(f"  SAC step {step}/{total_steps}, episodes: {len(episode_rewards)}, "
                  f"avg reward (last 100): {np.mean(episode_rewards[-100:]) if episode_rewards else 0:.2f}")

    return agent, episode_rewards


def train_ppo(env, total_steps=500000, rollout_len=2048):
    agent = PPO()
    episode_rewards = []
    ep_reward = 0
    state = env.reset()
    step = 0

    while step < total_steps:
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        for _ in range(rollout_len):
            action, log_prob, value = agent.ac.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(float(done))
            values.append(value)

            ep_reward += reward
            state = next_state
            step += 1

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0
                state = env.reset()

            if step >= total_steps:
                break

        with torch.no_grad():
            _, next_value = agent.ac(torch.FloatTensor(state).unsqueeze(0).to(device))
            next_value = next_value.item()

        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)
        agent.update(states, actions, log_probs, returns, advantages)

        if step % 50000 < rollout_len:
            print(f"  PPO step {step}/{total_steps}, episodes: {len(episode_rewards)}, "
                  f"avg reward (last 100): {np.mean(episode_rewards[-100:]) if episode_rewards else 0:.2f}")

    return agent, episode_rewards


# ============================================================
# Evaluation
# ============================================================

def evaluate_agent(env, select_action_fn, n_episodes=100):
    successes = 0
    total_steps = []
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            action = select_action_fn(state)
            state, reward, done, _ = env.step(action)
            steps += 1
        if env.agent_pos == env.goal_pos:
            successes += 1
        total_steps.append(steps)
    return successes / n_episodes, np.mean(total_steps)


# ============================================================
# Visualization
# ============================================================

def compute_input_importance(actor, env, n_episodes=50):
    """Gradient-based saliency for input dimension importance."""
    importances = np.zeros(16)
    count = 0

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            state_t.requires_grad_(True)
            actor.zero_grad()
            probs = actor(state_t)
            action = probs.argmax(dim=-1)
            selected_prob = probs[0, action]
            if state_t.grad is not None:
                state_t.grad.zero_()
            selected_prob.backward()
            importances += np.abs(state_t.grad.cpu().numpy()[0])
            count += 1
            with torch.no_grad():
                action_np = action.item()
            state, _, done, _ = env.step(action_np)

    importances /= count
    return importances


def plot_maze(env, path=None, filename="maze_visualization.png"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    maze = env.grid.copy()

    cmap = LinearSegmentedColormap.from_list("maze", ["white", "black"])
    ax.imshow(maze, cmap=cmap, origin="upper")

    ax.plot(env.goal_pos[1], env.goal_pos[0], "r*", markersize=20, label="Goal")
    ax.plot(0, 0, "go", markersize=15, label="Start")

    if path:
        path_arr = np.array(path)
        ax.plot(path_arr[:, 1], path_arr[:, 0], "b-", linewidth=2, alpha=0.7, label="Path")

    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=12)
    ax.set_title("15x15 Maze Environment", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_training_curves(sac_rewards, ppo_rewards, filename="training_curves.png"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode="valid")

    sac_smooth = smooth(sac_rewards)
    ppo_smooth = smooth(ppo_rewards)

    ax.plot(sac_smooth, label="SAC-Discrete", color="blue", alpha=0.8)
    ax.plot(ppo_smooth, label="PPO", color="red", alpha=0.8)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Training Curves: SAC vs PPO", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_comparison(sac_results, ppo_results, filename="comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["SAC-Discrete", "PPO"]
    success_rates = [sac_results[0] * 100, ppo_results[0] * 100]
    avg_steps = [sac_results[1], ppo_results[1]]
    colors = ["#2196F3", "#F44336"]

    axes[0].bar(labels, success_rates, color=colors, width=0.5, edgecolor="black")
    axes[0].set_ylabel("Success Rate (%)", fontsize=12)
    axes[0].set_title("Success Rate Comparison", fontsize=14)
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")

    axes[1].bar(labels, avg_steps, color=colors, width=0.5, edgecolor="black")
    axes[1].set_ylabel("Average Steps", fontsize=12)
    axes[1].set_title("Average Steps Comparison", fontsize=14)
    for i, v in enumerate(avg_steps):
        axes[1].text(i, v + 1, f"{v:.1f}", ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_importance_heatmap(importances, filename="importance_heatmap.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dim_labels = ["N_obs", "NE_obs", "E_obs", "SE_obs", "S_obs", "SW_obs", "W_obs", "NW_obs",
                  "Goal_N", "Goal_NE", "Goal_E", "Goal_SE", "Goal_S", "Goal_SW", "Goal_W", "Goal_NW"]

    # Bar chart
    colors = ["#FF6B6B"] * 8 + ["#4ECDC4"] * 8
    axes[0].barh(range(16), importances, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_yticks(range(16))
    axes[0].set_yticklabels(dim_labels, fontsize=10)
    axes[0].set_xlabel("Importance (Mean |Gradient|)", fontsize=11)
    axes[0].set_title("Input Dimension Importance\n(Policy Network Saliency)", fontsize=13)
    legend_elements = [mpatches.Patch(facecolor="#FF6B6B", label="Obstacle Detection"),
                       mpatches.Patch(facecolor="#4ECDC4", label="Goal Direction")]
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=10)

    # Heatmap
    imp_2d = importances.reshape(2, 8)
    im = axes[1].imshow(imp_2d, cmap="YlOrRd", aspect="auto")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Obstacle\nDetection", "Goal\nDirection"], fontsize=11)
    axes[1].set_xticks(range(8))
    axes[1].set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=10)
    axes[1].set_title("Importance Heatmap", fontsize=13)
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    for i in range(2):
        for j in range(8):
            axes[1].text(j, i, f"{imp_2d[i,j]:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if imp_2d[i,j] > imp_2d.max()*0.6 else "black")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def get_agent_path(env, select_action_fn):
    """Record the path taken by the agent for visualization."""
    state = env.reset()
    path = [list(env.agent_pos)]
    done = False
    while not done and len(path) < 300:
        action = select_action_fn(state)
        state, _, done, _ = env.step(action)
        path.append(list(env.agent_pos))
    return path


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  15x15 Maze RL: SAC-Discrete vs PPO")
    print("=" * 60)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sac_env = MazeEnv(size=15, obstacle_ratio=0.2, seed=seed)
    ppo_env = MazeEnv(size=15, obstacle_ratio=0.2, seed=seed + 1)
    print(f"\nSAC Maze: {sac_env.size}x{sac_env.size}, obstacles: {sac_env.grid.sum()}/{sac_env.size**2}")
    print(f"PPO Maze: {ppo_env.size}x{ppo_env.size}, obstacles: {ppo_env.grid.sum()}/{ppo_env.size**2}")

    # Train SAC
    print("\n[1/4] Training SAC-Discrete (500K steps)...")
    sac_agent, sac_rewards = train_sac(sac_env, total_steps=500000)

    # Train PPO
    print("\n[2/4] Training PPO (500K steps)...")
    ppo_agent, ppo_rewards = train_ppo(ppo_env, total_steps=500000)

    # Evaluate
    print("\n[3/4] Evaluating agents (100 episodes each)...")
    sac_select = lambda s: sac_agent.select_action(s, evaluate=True)
    ppo_select = lambda s: ppo_agent.ac.get_action(s, evaluate=True)[0]

    sac_results = evaluate_agent(sac_env, sac_select)
    ppo_results = evaluate_agent(ppo_env, ppo_select)

    print(f"\n{'='*50}")
    print(f"  Results Summary")
    print(f"{'='*50}")
    print(f"  {'Metric':<20} {'SAC-Discrete':<15} {'PPO':<15}")
    print(f"  {'-'*50}")
    print(f"  {'Success Rate':<20} {sac_results[0]*100:.1f}%{'':<10} {ppo_results[0]*100:.1f}%")
    print(f"  {'Avg Steps':<20} {sac_results[1]:.1f}{'':<12} {ppo_results[1]:.1f}")
    print(f"{'='*50}")

    # Visualizations
    print("\n[4/4] Generating visualizations...")
    path = get_agent_path(sac_env, sac_select)
    plot_maze(sac_env, path=path, filename="maze_visualization_sac.png")
    path = get_agent_path(ppo_env, ppo_select)
    plot_maze(ppo_env, path=path, filename="maze_visualization_ppo.png")
    plot_training_curves(sac_rewards, ppo_rewards)
    plot_comparison(sac_results, ppo_results)

    print("  Computing input importance (gradient saliency)...")
    importances = compute_input_importance(sac_agent.actor, sac_env)
    plot_importance_heatmap(importances)

    print("\nDone! All plots saved.")


if __name__ == "__main__":
    main()
