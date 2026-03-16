import os
import random
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

import torch
import torch.nn as nn
import torch.optim as optim

import robust_gymnasium as gym
from robust_gymnasium.configs.robust_setting import get_config

# Hyperparameters 
ENV_NAME        = "LunarLander-v3"
SEED            = 42
TOTAL_EPISODES  = 1000
MAX_STEPS       = 1000
GAMMA           = 0.99
LR              = 5e-4
BATCH_SIZE      = 64
BUFFER_SIZE     = 100_000
TAU             = 1e-3          # soft-update rate for target network
EPS_START       = 1.0
EPS_END         = 0.01
EPS_DECAY       = 0.995        # multiplicative decay per episode
HIDDEN_DIM      = 128
UPDATE_EVERY    = 4             # learn every N steps
SOLVE_SCORE     = 200.0 # task considered solved
SAVE_DIR        = "results/train_dqn"


# Q-Network 
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Replay Buffer 
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# DQN Agent 
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, device: torch.device):
        self.action_dim = action_dim
        self.device = device

        # Online and target networks
        self.qnet = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.qnet.state_dict())

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPS_START
        self.step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.qnet(state_t)
        return int(q_values.argmax(dim=1).item())

    def step(self, state, action, reward, next_state, done):
        """Store transition and learn if enough samples are collected."""
        self.buffer.push(state, action, reward, next_state, done)
        self.step_count += 1
        if self.step_count % UPDATE_EVERY == 0 and len(self.buffer) >= BATCH_SIZE:
            self._learn()

    def _learn(self):
        """Sample a batch and perform one gradient step"""
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        states_t      = torch.tensor(states, device=self.device)
        actions_t     = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t     = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t       = torch.tensor(dones, device=self.device).unsqueeze(1)

        # Current Q values for chosen actions
        q_values = self.qnet(states_t).gather(1, actions_t)

        # Target Q values (no grad)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target = rewards_t + GAMMA * max_next_q * (1.0 - dones_t)

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._soft_update()

    def _soft_update(self):
        for tp, op in zip(self.target_net.parameters(), self.qnet.parameters()):
            tp.data.copy_(TAU * op.data + (1.0 - TAU) * tp.data)

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)


#Visualization 
def plot_results(scores: list, avg_scores: list, epsilons: list, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    episodes = range(1, len(scores) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Score plot
    ax1.plot(episodes, scores, alpha=0.3, color="steelblue", label="Episode reward")
    ax1.plot(episodes, avg_scores, color="darkorange", linewidth=2, label="Avg reward (100 ep)")
    ax1.axhline(y=SOLVE_SCORE, color="green", linestyle="--", label=f"Solved ({SOLVE_SCORE})")
    ax1.set_ylabel("Reward")
    ax1.set_title("DQN on LunarLander-v3 (Discrete)")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Epsilon plot
    ax2.plot(episodes, epsilons, color="crimson")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon")
    ax2.set_title("Epsilon Decay")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"[INFO] Training curves saved to {save_dir}/training_curves.png")


def record_animation(agent, args, save_dir: str, num_episodes: int = 3):
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    best_frames, best_reward = [], -float("inf")

    for ep in range(num_episodes):
        state, _ = env.reset(seed=SEED + 10000 + ep)
        frames, total_reward = [], 0.0

        for _ in range(MAX_STEPS):
            frames.append(env.render())
            state_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                action = int(agent.qnet(state_t).argmax(dim=1).item())

            robust_input = {"action": action, "robust_type": "action", "robust_config": args}
            state, reward, terminated, truncated, _ = env.step(robust_input)
            total_reward += reward
            if terminated or truncated:
                frames.append(env.render())
                break

        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames
        print(f"[ANIM] Episode {ep+1}/{num_episodes} | Reward: {total_reward:.1f}")

    env.close()

    # Save best episode as GIF
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    img = ax.imshow(best_frames[0])
    ax.set_title(f"DQN LunarLander-v3 | Reward: {best_reward:.1f}", fontsize=12)

    def update(i):
        img.set_data(best_frames[i])
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=len(best_frames), interval=30, blit=True)
    gif_path = os.path.join(save_dir, "lunarlander_dqn.gif")
    ani.save(gif_path, writer="pillow", fps=30)
    plt.close(fig)
    print(f"[INFO] Animation saved to {gif_path}")


#  Main Training Loop 
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Robust-Gymnasium config
    args = get_config().parse_args([])
    args.noise_factor = "none"          # disable all perturbations
    args.noise_sigma  = 0.0

    # Seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make(ENV_NAME)
    state_dim  = env.observation_space.shape[0]   # 8
    action_dim = env.action_space.n               # 4

    agent = DQNAgent(state_dim, action_dim, device)

    scores, avg_scores, epsilons = [], [], []
    recent_scores = deque(maxlen=100)
    solved = False

    for ep in range(1, TOTAL_EPISODES + 1):
        state, _ = env.reset(seed=SEED + ep)
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            action = agent.select_action(state)

            robust_input = {
                "action": action,
                "robust_type": "action",
                "robust_config": args,
            }
            next_state, reward, terminated, truncated, _ = env.step(robust_input)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        scores.append(total_reward)
        recent_scores.append(total_reward)
        avg = np.mean(recent_scores)
        avg_scores.append(avg)
        epsilons.append(agent.epsilon)

        if ep % 20 == 0:
            print(f"Episode {ep:4d} | Reward: {total_reward:7.1f} | Avg(100): {avg:7.1f} | Eps: {agent.epsilon:.3f}")

        if avg >= SOLVE_SCORE and not solved:
            solved = True
            print(f"\n*** Solved at episode {ep} with avg reward {avg:.1f} ***\n")
            torch.save(agent.qnet.state_dict(), os.path.join(SAVE_DIR, "dqn_solved.pth"))

    # Save final model
    torch.save(agent.qnet.state_dict(), os.path.join(SAVE_DIR, "dqn_final.pth"))
    env.close()

    plot_results(scores, avg_scores, epsilons, SAVE_DIR)
    record_animation(agent, args, SAVE_DIR)
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
