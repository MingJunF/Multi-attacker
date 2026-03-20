import os
import random
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import robust_gymnasium as gym
from robust_gymnasium.configs.robust_setting import get_config

ENV_NAME       = "LunarLander-v3"
SEED           = 42
TOTAL_EPISODES = 1000
MAX_STEPS      = 1000
GAMMA          = 0.99
LR             = 5e-4
BATCH_SIZE     = 64
BUFFER_SIZE    = 100_000
TAU            = 1e-3
EPS_START      = 1.0
EPS_END        = 0.01
EPS_DECAY      = 0.995
HIDDEN_DIM     = 256
UPDATE_EVERY   = 4
SOLVE_SCORE    = 200.0
SAVE_DIR       = "results/train_dueling_double_dqn"

PER_ALPHA  = 0.6
BETA_START = 0.4
BETA_END   = 1.0

class DuelingQNetwork(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        value = self.value_stream(feat)            # (batch, 1)
        advantage = self.advantage_stream(feat)    # (batch, n_actions)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class PrioritizedReplayBuffer:

    def __init__(self, capacity: int, alpha: float = PER_ALPHA):
        self.capacity  = capacity
        self.alpha     = alpha
        self.buffer: list = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos  = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = float(self.priorities[: self.size].max()) if self.size > 0 else 1.0
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        prios = self.priorities[: self.size] ** self.alpha
        probs = prios / prios.sum()
        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors: np.ndarray, epsilon: float = 1e-6):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(abs(err)) + epsilon

    def __len__(self):
        return self.size

class DuelingDDQNAgent:

    def __init__(self, state_dim: int, action_dim: int, device: torch.device):
        self.action_dim = action_dim
        self.device     = device

        self.qnet       = DuelingQNetwork(state_dim, action_dim).to(device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.qnet.state_dict())

        self.optimizer  = optim.Adam(self.qnet.parameters(), lr=LR)
        self.buffer     = PrioritizedReplayBuffer(BUFFER_SIZE)
        self.epsilon    = EPS_START
        self.step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.qnet(s).argmax(dim=1).item())

    def step(self, state, action, reward, next_state, done, beta: float = 0.4):
        self.buffer.push(state, action, reward, next_state, done)
        self.step_count += 1
        if self.step_count % UPDATE_EVERY == 0 and len(self.buffer) >= BATCH_SIZE:
            self._learn(beta)

    def _learn(self, beta: float):
        states, actions, rewards, next_states, dones, indices, weights = \
            self.buffer.sample(BATCH_SIZE, beta)

        s  = torch.tensor(states,      dtype=torch.float32, device=self.device)
        a  = torch.tensor(actions,     dtype=torch.int64,   device=self.device).unsqueeze(1)
        r  = torch.tensor(rewards,     dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        d  = torch.tensor(dones,       dtype=torch.float32, device=self.device).unsqueeze(1)
        w  = torch.tensor(weights,     dtype=torch.float32, device=self.device).unsqueeze(1)

        q_pred = self.qnet(s).gather(1, a)

        with torch.no_grad():
            best_next_a = self.qnet(ns).argmax(dim=1, keepdim=True)
            q_next      = self.target_net(ns).gather(1, best_next_a)
            q_target    = r + GAMMA * q_next * (1.0 - d)

        td_errors = (q_pred - q_target).detach().cpu().numpy().flatten()

        element_loss = nn.functional.smooth_l1_loss(q_pred, q_target, reduction="none")
        loss = (w * element_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_errors)
        self._soft_update()

    def _soft_update(self):
        for tp, op in zip(self.target_net.parameters(), self.qnet.parameters()):
            tp.data.copy_(TAU * op.data + (1.0 - TAU) * tp.data)

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)


def plot_results(scores: list, avg_scores: list, epsilons: list, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    episodes = range(1, len(scores) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(episodes, scores,     alpha=0.3, color="steelblue",  label="Episode reward")
    ax1.plot(episodes, avg_scores, linewidth=2, color="darkorange", label="Avg reward (100 ep)")
    ax1.axhline(y=SOLVE_SCORE, color="green", linestyle="--", label=f"Solved ({SOLVE_SCORE})")
    ax1.set_ylabel("Reward")
    ax1.set_title("Dueling Double DQN + PER on LunarLander-v3 (Discrete)")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, epsilons, color="crimson")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon")
    ax2.set_title("Epsilon Decay")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves_dueling_ddqn.png"), dpi=150)
    plt.close(fig)
    print(f"[INFO] Training curves saved to {save_dir}/training_curves_dueling_ddqn.png")


def record_animation(agent: DuelingDDQNAgent, args, save_dir: str, num_episodes: int = 3):
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    best_frames, best_reward = [], -float("inf")

    for ep in range(num_episodes):
        state, _ = env.reset(seed=SEED + 10000 + ep)
        frames, total_reward = [], 0.0
        for _ in range(MAX_STEPS):
            frames.append(env.render())
            s = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                action = int(agent.qnet(s).argmax(dim=1).item())
            robust_input = {"action": action, "robust_type": "action", "robust_config": args}
            state, reward, terminated, truncated, _ = env.step(robust_input)
            total_reward += float(reward)
            if terminated or truncated:
                frames.append(env.render())
                break
        if total_reward > best_reward:
            best_reward, best_frames = total_reward, frames
        print(f"[ANIM] Episode {ep + 1}/{num_episodes} | Reward: {total_reward:.1f}")

    env.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    img = ax.imshow(best_frames[0])
    ax.set_title(f"Dueling DDQN+PER | Reward: {best_reward:.1f}", fontsize=12)

    def update(i):
        img.set_data(best_frames[i])
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=len(best_frames), interval=30, blit=True)
    gif_path = os.path.join(save_dir, "lunarlander_dueling_ddqn.gif")
    ani.save(gif_path, writer="pillow", fps=30)
    plt.close(fig)
    print(f"[INFO] Animation saved to {gif_path}")

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Dueling Double DQN + PER | hidden={HIDDEN_DIM} | "
          f"PER alpha={PER_ALPHA} | beta {BETA_START}→{BETA_END}")

    args              = get_config().parse_args([])
    args.noise_factor = "none"
    args.noise_sigma  = 0.0

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env        = gym.make(ENV_NAME)
    state_dim  = env.observation_space.shape[0]   # 8
    action_dim = env.action_space.n               # 4
    agent      = DuelingDDQNAgent(state_dim, action_dim, device)

    scores, avg_scores, epsilons = [], [], []
    recent_scores = deque(maxlen=100)
    solved = False

    for ep in range(1, TOTAL_EPISODES + 1):
        beta = min(BETA_END, BETA_START + (BETA_END - BETA_START) * ep / TOTAL_EPISODES)

        state, _ = env.reset(seed=SEED + ep)
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            robust_input = {
                "action":        action,
                "robust_type":   "action",
                "robust_config": args,
            }
            next_state, reward, terminated, truncated, _ = env.step(robust_input)
            done = terminated or truncated

            agent.step(state, action, float(reward), next_state, done, beta)
            state        = next_state
            total_reward += float(reward)
            if done:
                break

        agent.decay_epsilon()
        scores.append(total_reward)
        recent_scores.append(total_reward)
        avg_reward = float(np.mean(recent_scores))
        avg_scores.append(avg_reward)
        epsilons.append(agent.epsilon)

        if ep % 20 == 0:
            print(f"Episode {ep:4d} | Reward: {total_reward:7.1f} | "
                  f"Avg(100): {avg_reward:7.1f} | Eps: {agent.epsilon:.3f} | Beta: {beta:.3f}")

        if avg_reward >= SOLVE_SCORE and not solved:
            solved = True
            print(f"\n*** Solved at episode {ep} with avg reward {avg_reward:.1f} ***\n")
            torch.save(agent.qnet.state_dict(),
                       os.path.join(SAVE_DIR, "dueling_ddqn_solved.pth"))

    torch.save(agent.qnet.state_dict(), os.path.join(SAVE_DIR, "dueling_ddqn_final.pth"))
    env.close()

    plot_results(scores, avg_scores, epsilons, SAVE_DIR)
    record_animation(agent, args, SAVE_DIR)
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
