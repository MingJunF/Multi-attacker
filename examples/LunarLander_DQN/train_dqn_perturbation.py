
import os
import random
import json
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

#  Hyperparameters 
ENV_NAME        = "LunarLander-v3"
SEED            = 42
TOTAL_EPISODES  = 800
MAX_STEPS       = 1000
GAMMA           = 0.99
LR              = 5e-4
BATCH_SIZE      = 64
BUFFER_SIZE     = 100_000
TAU             = 1e-3
EPS_START       = 1.0
EPS_END         = 0.01
EPS_DECAY       = 0.995
HIDDEN_DIM      = 128
UPDATE_EVERY    = 4
SOLVE_SCORE     = 200.0
SAVE_DIR        = "results/train_dqn_perturbation"

# Perturbation Configs 
PERTURBATION_CONFIGS = [
    ("No Perturbation",         "none",   "gauss", 0.0),
    ("State Gauss σ=0.01",      "state",  "gauss", 0.01),
    ("State Gauss σ=0.05",      "state",  "gauss", 0.05),
    ("State Gauss σ=0.1",       "state",  "gauss", 0.1),
    ("Reward Gauss σ=1.0",      "reward", "gauss", 1.0),
    ("Reward Gauss σ=5.0",      "reward", "gauss", 5.0),
    ("Reward Gauss σ=10.0",     "reward", "gauss", 10.0),
]


#  Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


#  Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


#  DQN Agent 
class DQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.action_dim = action_dim
        self.device = device
        self.qnet = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPS_START
        self.step_count = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.qnet(s).argmax(1).item())

    def step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.step_count += 1
        if self.step_count % UPDATE_EVERY == 0 and len(self.buffer) >= BATCH_SIZE:
            self._learn()

    def _learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        s  = torch.tensor(states, device=self.device)
        a  = torch.tensor(actions, device=self.device).unsqueeze(1)
        r  = torch.tensor(rewards, device=self.device).unsqueeze(1)
        ns = torch.tensor(next_states, device=self.device)
        d  = torch.tensor(dones, device=self.device).unsqueeze(1)

        q = self.qnet(s).gather(1, a)
        with torch.no_grad():
            target = r + GAMMA * self.target_net(ns).max(1, keepdim=True)[0] * (1.0 - d)

        loss = nn.functional.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 1.0)
        self.optimizer.step()

        for tp, op in zip(self.target_net.parameters(), self.qnet.parameters()):
            tp.data.copy_(TAU * op.data + (1.0 - TAU) * tp.data)

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)


#  Single Experiment Run 
def train_single(label, noise_factor, noise_type, noise_sigma, device):
    """Train DQN under one perturbation setting. Returns scores and avg_scores."""
    print(f"\n{'='*60}")
    print(f"  Experiment: {label}")
    print(f"  noise_factor={noise_factor}, noise_type={noise_type}, sigma={noise_sigma}")
    print(f"{'='*60}")

    args = get_config().parse_args([])
    args.noise_factor = noise_factor
    args.noise_type   = noise_type
    args.noise_sigma  = noise_sigma
    args.noise_mu     = 0.0

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make(ENV_NAME)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, device)

    scores, avg_scores = [], []
    recent = deque(maxlen=100)

    for ep in range(1, TOTAL_EPISODES + 1):
        state, _ = env.reset(seed=SEED + ep)
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            robust_input = {"action": action, "robust_type": "action", "robust_config": args}
            next_state, reward, terminated, truncated, _ = env.step(robust_input)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        scores.append(total_reward)
        recent.append(total_reward)
        avg = np.mean(recent)
        avg_scores.append(avg)

        if ep % 100 == 0:
            print(f"  [{label}] Ep {ep:4d} | Reward: {total_reward:7.1f} | Avg(100): {avg:7.1f}")

    env.close()

    # Record best animation episode
    frames = record_best_episode(agent, args, device)

    return {
        "label": label,
        "scores": scores,
        "avg_scores": avg_scores,
        "final_avg": float(np.mean(scores[-100:])),
        "final_std": float(np.std(scores[-100:])),
        "frames": frames,
    }


#  Record Best Episode
def record_best_episode(agent, args, device, num_eval=3):
    """Evaluate agent for a few episodes and return frames of the best one."""
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    best_frames, best_r = [], -float("inf")

    for i in range(num_eval):
        state, _ = env.reset(seed=SEED + 20000 + i)
        frames, total_r = [], 0.0
        for _ in range(MAX_STEPS):
            frames.append(env.render())
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(agent.qnet(s).argmax(1).item())
            robust_input = {"action": action, "robust_type": "action", "robust_config": args}
            state, reward, terminated, truncated, _ = env.step(robust_input)
            total_r += reward
            if terminated or truncated:
                frames.append(env.render())
                break
        if total_r > best_r:
            best_r, best_frames = total_r, frames

    env.close()
    return best_frames


#  Visualization
def plot_comparison(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Training curves comparison 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for r in results:
        if "State" in r["label"] or "No Perturbation" in r["label"]:
            ax.plot(r["avg_scores"], label=r["label"], linewidth=1.5)
    ax.axhline(y=SOLVE_SCORE, color="gray", linestyle="--", alpha=0.5, label="Solved (200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward (100 ep)")
    ax.set_title("State Perturbation Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Reward perturbation group
    ax = axes[1]
    for r in results:
        if "Reward" in r["label"] or "No Perturbation" in r["label"]:
            ax.plot(r["avg_scores"], label=r["label"], linewidth=1.5)
    ax.axhline(y=SOLVE_SCORE, color="gray", linestyle="--", alpha=0.5, label="Solved (200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward (100 ep)")
    ax.set_title("Reward Perturbation Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "perturbation_training_curves.png"), dpi=150)
    plt.close(fig)

    #  Final performance bar chart 
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [r["label"] for r in results]
    means  = [r["final_avg"] for r in results]
    stds   = [r["final_std"] for r in results]
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    bars = ax.bar(range(len(results)), means, yerr=stds, color=colors,
                  edgecolor="black", linewidth=0.5, capsize=4)
    ax.axhline(y=SOLVE_SCORE, color="red", linestyle="--", alpha=0.7, label="Solved (200)")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Avg Reward (last 100 ep)")
    ax.set_title("DQN Performance Under Different Perturbations")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{m:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "perturbation_bar_chart.png"), dpi=150)
    plt.close(fig)

    #All training curves on one plot 
    fig, ax = plt.subplots(figsize=(12, 6))
    for r in results:
        ax.plot(r["avg_scores"], label=r["label"], linewidth=1.5)
    ax.axhline(y=SOLVE_SCORE, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward (100 ep)")
    ax.set_title("DQN Robustness: All Perturbation Settings")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "perturbation_all_curves.png"), dpi=150)
    plt.close(fig)

    print(f"[INFO] Comparison plots saved to {save_dir}/")


def save_animations(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for r in results:
        frames = r["frames"]
        if not frames:
            continue
        # Sanitize filename
        fname = r["label"].replace(" ", "_").replace("σ=", "s").replace(".", "p") + ".gif"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        img = ax.imshow(frames[0])
        ax.set_title(f"{r['label']} | Avg: {r['final_avg']:.1f}", fontsize=10)

        def update(i, img=img, frames=frames):
            img.set_data(frames[i])
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=30, blit=True)
        ani.save(os.path.join(save_dir, fname), writer="pillow", fps=30)
        plt.close(fig)
        print(f"[INFO] Animation saved: {fname}")


def save_summary(results, save_dir):
    """Save a summary table to text and JSON."""
    os.makedirs(save_dir, exist_ok=True)

    # Text summary
    lines = ["=" * 70, "  Perturbation Experiment Summary", "=" * 70, ""]
    lines.append(f"{'Setting':<30} {'Avg Reward':>12} {'Std':>8} {'Δ vs Baseline':>14}")
    lines.append("-" * 70)
    baseline = results[0]["final_avg"]
    for r in results:
        delta = r["final_avg"] - baseline
        sign = "+" if delta >= 0 else ""
        lines.append(f"{r['label']:<30} {r['final_avg']:>12.1f} {r['final_std']:>8.1f} {sign}{delta:>13.1f}")
    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    with open(os.path.join(save_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    # JSON summary 
    json_data = [{k: v for k, v in r.items() if k != "frames"} for r in results]
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(json_data, f, indent=2)


# Main
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Running {len(PERTURBATION_CONFIGS)} experiments, {TOTAL_EPISODES} episodes each\n")

    results = []
    for label, nf, nt, ns in PERTURBATION_CONFIGS:
        result = train_single(label, nf, nt, ns, device)
        results.append(result)

    plot_comparison(results, SAVE_DIR)
    save_animations(results, SAVE_DIR)
    save_summary(results, SAVE_DIR)
    print("\n[INFO] All experiments complete.")


if __name__ == "__main__":
    main()
