import json
import os
import random
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import robust_gymnasium as gym
from robust_gymnasium.configs.robust_setting import get_config


ENV_NAME       = "LunarLander-v3"
SEED           = 42
TOTAL_EPISODES = 600
MAX_STEPS      = 1000
GAMMA          = 0.99
LR             = 5e-4
BATCH_SIZE     = 64
BUFFER_SIZE    = 100_000
TAU            = 1e-3
EPS_START      = 1.0
EPS_END        = 0.01
EPS_DECAY      = 0.995
UPDATE_EVERY   = 4
SOLVE_SCORE    = 200.0
SAVE_DIR       = "results/compare_algorithms_perturbation_3DQN"

HIDDEN_STD     = 128
HIDDEN_DUEL    = 256

PER_ALPHA      = 0.6
BETA_START     = 0.4
BETA_END       = 1.0
ALGO_NAMES = ["DQN", "DoubleDQN", "DuelingDDQN"]

PERTURBATION_CONFIGS = [
    ("No Perturbation",    "none",       "gauss",    0.0),
    ("State σ=0.05",       "state",      "gauss",    0.05),
    ("State σ=0.10",       "state",      "gauss",    0.10),
    ("Reward σ=5.0",       "reward",     "gauss",    5.0),
    ("Reward σ=10.0",      "reward",     "gauss",    10.0),
]


# Networks 

class MLP(nn.Module):
    """Standard two-hidden-layer Q-network used by DQN and Double DQN."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_STD):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingMLP(nn.Module):
    """Dueling Q-network: shared features → separate V(s) and A(s,a) heads."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DUEL):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        v = self.value(feat)
        a = self.advantage(feat)
        return v + a - a.mean(dim=1, keepdim=True)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = PER_ALPHA):
        self.capacity  = capacity
        self.alpha     = alpha
        self.buffer: list = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos  = 0
        self.size = 0

    def push(self, s, a, r, ns, d):
        max_prio = float(self.priorities[: self.size].max()) if self.size > 0 else 1.0
        if self.size < self.capacity:
            self.buffer.append((s, a, r, ns, d))
        else:
            self.buffer[self.pos] = (s, a, r, ns, d)
        self.priorities[self.pos] = max_prio
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        prios = self.priorities[: self.size] ** self.alpha
        probs = prios / prios.sum()
        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        batch = [self.buffer[i] for i in indices]
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32), indices, weights.astype(np.float32))

    def update_priorities(self, indices, td_errors, eps: float = 1e-6):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(abs(err)) + eps

    def __len__(self):
        return self.size


#Agents

class DQNAgent:
    """Vanilla DQN — uniform replay, standard Bellman target."""

    def __init__(self, state_dim, action_dim, device):
        self.action_dim  = action_dim
        self.device      = device
        self.qnet        = MLP(state_dim, action_dim).to(device)
        self.target_net  = MLP(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.qnet.state_dict())
        self.optimizer   = optim.Adam(self.qnet.parameters(), lr=LR)
        self.buffer      = ReplayBuffer(BUFFER_SIZE)
        self.epsilon     = EPS_START
        self.step_count  = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.qnet(s).argmax(1).item())

    def step(self, s, a, r, ns, d, beta=0.4):
        self.buffer.push(s, a, r, ns, d)
        self.step_count += 1
        if self.step_count % UPDATE_EVERY == 0 and len(self.buffer) >= BATCH_SIZE:
            self._learn()

    def _learn(self):
        s, a, r, ns, d = self.buffer.sample(BATCH_SIZE)
        s  = torch.tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1)
        r  = torch.tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d  = torch.tensor(d,  dtype=torch.float32, device=self.device).unsqueeze(1)
        q  = self.qnet(s).gather(1, a)
        with torch.no_grad():
            target = r + GAMMA * self.target_net(ns).max(1, keepdim=True)[0] * (1 - d)
        loss = nn.functional.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 1.0)
        self.optimizer.step()
        for tp, op in zip(self.target_net.parameters(), self.qnet.parameters()):
            tp.data.copy_(TAU * op.data + (1 - TAU) * tp.data)

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)


class DoubleDQNAgent(DQNAgent):
    """Double DQN — same as DQN but online net selects next action, target net evaluates."""

    def _learn(self):
        s, a, r, ns, d = self.buffer.sample(BATCH_SIZE)
        s  = torch.tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1)
        r  = torch.tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d  = torch.tensor(d,  dtype=torch.float32, device=self.device).unsqueeze(1)
        q  = self.qnet(s).gather(1, a)
        with torch.no_grad():
            best_a  = self.qnet(ns).argmax(1, keepdim=True)
            q_next  = self.target_net(ns).gather(1, best_a)
            target  = r + GAMMA * q_next * (1 - d)
        loss = nn.functional.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 1.0)
        self.optimizer.step()
        for tp, op in zip(self.target_net.parameters(), self.qnet.parameters()):
            tp.data.copy_(TAU * op.data + (1 - TAU) * tp.data)


class DuelingDDQNAgent:
    """Dueling Double DQN + PER — Dueling network, PER buffer, Double DQN target, Huber loss."""

    def __init__(self, state_dim, action_dim, device):
        self.action_dim  = action_dim
        self.device      = device
        self.qnet        = DuelingMLP(state_dim, action_dim).to(device)
        self.target_net  = DuelingMLP(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.qnet.state_dict())
        self.optimizer   = optim.Adam(self.qnet.parameters(), lr=LR)
        self.buffer      = PrioritizedReplayBuffer(BUFFER_SIZE)
        self.epsilon     = EPS_START
        self.step_count  = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.qnet(s).argmax(1).item())

    def step(self, s, a, r, ns, d, beta=0.4):
        self.buffer.push(s, a, r, ns, d)
        self.step_count += 1
        if self.step_count % UPDATE_EVERY == 0 and len(self.buffer) >= BATCH_SIZE:
            self._learn(beta)

    def _learn(self, beta):
        s, a, r, ns, d, indices, w = self.buffer.sample(BATCH_SIZE, beta)
        s  = torch.tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1)
        r  = torch.tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d  = torch.tensor(d,  dtype=torch.float32, device=self.device).unsqueeze(1)
        w  = torch.tensor(w,  dtype=torch.float32, device=self.device).unsqueeze(1)
        q_pred = self.qnet(s).gather(1, a)
        with torch.no_grad():
            best_a   = self.qnet(ns).argmax(1, keepdim=True)
            q_next   = self.target_net(ns).gather(1, best_a)
            q_target = r + GAMMA * q_next * (1 - d)
        td_err = (q_pred - q_target).detach().cpu().numpy().flatten()
        loss   = (w * nn.functional.smooth_l1_loss(q_pred, q_target, reduction="none")).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.optimizer.step()
        self.buffer.update_priorities(indices, td_err)
        for tp, op in zip(self.target_net.parameters(), self.qnet.parameters()):
            tp.data.copy_(TAU * op.data + (1 - TAU) * tp.data)

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

def make_agent(algo: str, state_dim: int, action_dim: int, device: torch.device):
    if algo == "DQN":
        return DQNAgent(state_dim, action_dim, device)
    if algo == "DoubleDQN":
        return DoubleDQNAgent(state_dim, action_dim, device)
    if algo == "DuelingDDQN":
        return DuelingDDQNAgent(state_dim, action_dim, device)
    raise ValueError(f"Unknown algorithm: {algo}")

def train_single(algo: str, pert_label: str, noise_factor: str,
                 noise_type: str, noise_sigma: float, device) -> dict:
    print(f"\n{'─' * 60}")
    print(f"  Algo: {algo:15s} | Pert: {pert_label}")
    print(f"{'─' * 60}")

    args              = get_config().parse_args([])
    args.noise_factor = noise_factor
    args.noise_type   = noise_type
    args.noise_sigma  = noise_sigma
    args.noise_mu     = 0.0

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env        = gym.make(ENV_NAME)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent      = make_agent(algo, state_dim, action_dim, device)

    scores, avg_scores = [], []
    recent = deque(maxlen=100)

    for ep in range(1, TOTAL_EPISODES + 1):
        beta  = min(BETA_END, BETA_START + (BETA_END - BETA_START) * ep / TOTAL_EPISODES)
        state, _ = env.reset(seed=SEED + ep)
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            robust_input = {
                "action": action, "robust_type": "action", "robust_config": args
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
        recent.append(total_reward)
        avg = float(np.mean(recent))
        avg_scores.append(avg)

        if ep % 100 == 0:
            print(f"  [{algo}|{pert_label}] ep {ep:4d} | "
                  f"Reward: {total_reward:7.1f} | Avg(100): {avg:7.1f}")

    env.close()
    final_avg = float(np.mean(scores[-100:]))
    final_std = float(np.std(scores[-100:]))
    print(f"  → Final avg (last 100): {final_avg:.1f} ± {final_std:.1f}")

    return {
        "algo":        algo,
        "pert_label":  pert_label,
        "noise_factor": noise_factor,
        "noise_sigma":  noise_sigma,
        "scores":      scores,
        "avg_scores":  avg_scores,
        "final_avg":   final_avg,
        "final_std":   final_std,
    }


COLORS = {
    "DQN":         "#4c8bca",
    "DoubleDQN":   "#f48024",
    "DuelingDDQN": "#27ae60",
}
LINESTYLES = {
    "DQN":         "-",
    "DoubleDQN":   "--",
    "DuelingDDQN": "-.",
}


def _plot_group(ax, results, algos, pert_label, title):
    for algo in algos:
        row = next((r for r in results if r["algo"] == algo and r["pert_label"] == pert_label), None)
        if row is None:
            continue
        ax.plot(row["avg_scores"],
                label=f"{algo} ({row['final_avg']:.0f})",
                color=COLORS[algo], linestyle=LINESTYLES[algo], linewidth=1.8)
    ax.axhline(y=SOLVE_SCORE, color="gray", linestyle=":", alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward (100 ep)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_all(results: list, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    algos     = ALGO_NAMES
    pert_list = [p[0] for p in PERTURBATION_CONFIGS]

    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_group(ax, results, algos, "No Perturbation", "Algorithm Comparison — No Perturbation")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "comparison_baseline.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, lbl in zip(axes, ["State σ=0.05", "State σ=0.10"]):
        _plot_group(ax, results, algos, lbl, f"State Noise — {lbl}")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "comparison_state_noise.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, lbl in zip(axes, ["Reward σ=5.0", "Reward σ=10.0"]):
        _plot_group(ax, results, algos, lbl, f"Reward Noise — {lbl}")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "comparison_reward_noise.png"), dpi=150)
    plt.close(fig)

    n_pert  = len(pert_list)
    n_algos = len(algos)
    x       = np.arange(n_pert)
    width   = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, algo in enumerate(algos):
        vals = []
        errs = []
        for pl in pert_list:
            row = next((r for r in results if r["algo"] == algo and r["pert_label"] == pl), None)
            vals.append(row["final_avg"] if row else 0)
            errs.append(row["final_std"] if row else 0)
        ax.bar(x + i * width, vals, width, yerr=errs, capsize=3,
               label=algo, color=COLORS[algo], alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.axhline(y=SOLVE_SCORE, color="red", linestyle="--", alpha=0.6, label=f"Solved ({SOLVE_SCORE})")
    ax.set_xticks(x + width)
    ax.set_xticklabels(pert_list, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Avg Reward (last 100 ep)")
    ax.set_title("Algorithm Robustness under Perturbations — LunarLander-v3")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "robustness_bar.png"), dpi=150)
    plt.close(fig)

    baselines = {
        algo: next((r["final_avg"] for r in results
                    if r["algo"] == algo and r["pert_label"] == "No Perturbation"), 1.0)
        for algo in algos
    }
    pert_noise_labels = [p[0] for p in PERTURBATION_CONFIGS[1:]]
    matrix = np.zeros((len(algos), len(pert_noise_labels)))
    for i, algo in enumerate(algos):
        baseline = baselines[algo] if baselines[algo] != 0 else 1.0
        for j, pl in enumerate(pert_noise_labels):
            row = next((r for r in results if r["algo"] == algo and r["pert_label"] == pl), None)
            if row and baseline > 0:
                matrix[i, j] = 100.0 * (row["final_avg"] - baseline) / abs(baseline)

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-60, vmax=10, aspect="auto")
    ax.set_xticks(range(len(pert_noise_labels)))
    ax.set_xticklabels(pert_noise_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos)
    ax.set_title("Performance Change vs Each Algorithm's Own Baseline (%)")
    plt.colorbar(im, ax=ax, label="% change")
    for i in range(len(algos)):
        for j in range(len(pert_noise_labels)):
            ax.text(j, i, f"{matrix[i, j]:+.1f}%",
                    ha="center", va="center", fontsize=8,
                    color="black" if -30 < matrix[i, j] < 5 else "white")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "degradation_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"[INFO] All plots saved to {save_dir}/")


def save_summary(results: list, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    header = f"{'Algorithm':<16} {'Perturbation':<22} {'Avg Reward':>12} {'Std':>8}"
    sep    = "─" * len(header)
    lines  = [sep, header, sep]
    for r in results:
        lines.append(f"{r['algo']:<16} {r['pert_label']:<22} "
                     f"{r['final_avg']:>12.1f} {r['final_std']:>8.1f}")
    lines.append(sep)
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(save_dir, "summary.txt"), "w") as f:
        f.write(text)
    json_data = [{k: v for k, v in r.items() if k not in ("scores", "avg_scores")}
                 for r in results]
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(json_data, f, indent=2)


# Main
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_runs = len(ALGO_NAMES) * len(PERTURBATION_CONFIGS)
    print(f"[INFO] Device: {device} | Total runs: {total_runs} "
          f"({TOTAL_EPISODES} ep each)")

    results = []
    run_idx = 0
    for algo in ALGO_NAMES:
        for pert_label, nf, nt, ns in PERTURBATION_CONFIGS:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}]", end="")
            result = train_single(algo, pert_label, nf, nt, ns, device)
            results.append(result)

    plot_all(results, SAVE_DIR)
    save_summary(results, SAVE_DIR)
    print("\n[INFO] All experiments complete.")


if __name__ == "__main__":
    main()
