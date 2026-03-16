from __future__ import annotations

import argparse
import csv
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import robust_gymnasium as gym
from robust_gymnasium.configs.robust_setting import get_config


@dataclass
class ExpConfig:
    env_name: str = "LunarLander-v3"
    seed: int = 42
    total_episodes: int = 600
    max_steps: int = 1000
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 64
    buffer_size: int = 100_000
    tau: float = 1e-3
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.995
    hidden_dim: int = 128
    update_every: int = 4
    eval_interval: int = 50
    eval_episodes: int = 5
    eval_seed_offset: int = 100000
    return_window: int = 100


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int):
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

    def __len__(self) -> int:
        return len(self.buffer)


class DQNFamilyAgent:

    def __init__(self, state_dim: int, action_dim: int, device: torch.device, cfg: ExpConfig, use_double: bool):
        self.action_dim = action_dim
        self.device = device
        self.cfg = cfg
        self.use_double = use_double

        self.qnet = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(device)
        self.target_net.load_state_dict(self.qnet.state_dict())

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)

        self.epsilon = cfg.eps_start
        self.step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.qnet(state_t)
        return int(q_values.argmax(dim=1).item())

    def step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.step_count += 1
        if self.step_count % self.cfg.update_every == 0 and len(self.buffer) >= self.cfg.batch_size:
            self._learn()

    def _learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.cfg.batch_size)

        s = torch.tensor(states, dtype=torch.float32, device=self.device)
        a = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_pred = self.qnet(s).gather(1, a)

        with torch.no_grad():
            if self.use_double:
                next_actions = self.qnet(ns).argmax(dim=1, keepdim=True)
                next_q = self.target_net(ns).gather(1, next_actions)
            else:
                next_q = self.target_net(ns).max(dim=1, keepdim=True)[0]
            target = r + self.cfg.gamma * next_q * (1.0 - d)

        loss = nn.functional.mse_loss(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._soft_update()

    def _soft_update(self):
        for tp, op in zip(self.target_net.parameters(), self.qnet.parameters()):
            tp.data.copy_(self.cfg.tau * op.data + (1.0 - self.cfg.tau) * tp.data)

    def decay_epsilon(self):
        self.epsilon = max(self.cfg.eps_end, self.epsilon * self.cfg.eps_decay)


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_robust_config():
    robust_cfg = get_config().parse_args([])
    robust_cfg.noise_factor = "none"
    robust_cfg.noise_sigma = 0.0
    return robust_cfg


def _discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    out = [0.0] * len(rewards)
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        g = float(rewards[i]) + gamma * g
        out[i] = g
    return out


def evaluate_overestimation_on_policy(
    agent: DQNFamilyAgent,
    cfg: ExpConfig,
    robust_cfg,
    eval_seed_base: int,
) -> Tuple[float, float, float, float]:

    env = gym.make(cfg.env_name)

    q_preds_all: List[float] = []
    g_true_all: List[float] = []
    episodic_returns: List[float] = []

    for ep in range(cfg.eval_episodes):
        state, _ = env.reset(seed=eval_seed_base + ep)
        rewards: List[float] = []
        q_preds_episode: List[float] = []

        for _ in range(cfg.max_steps):
            s = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.qnet(s)
            action = int(q_values.argmax(dim=1).item())
            q_preds_episode.append(float(q_values.max(dim=1)[0].item()))

            robust_input = {"action": action, "robust_type": "action", "robust_config": robust_cfg}
            next_state, reward, terminated, truncated, _ = env.step(robust_input)
            rewards.append(float(reward))
            state = next_state

            if terminated or truncated:
                break

        g_true_episode = _discounted_returns(rewards, cfg.gamma)
        q_preds_all.extend(q_preds_episode[: len(g_true_episode)])
        g_true_all.extend(g_true_episode)
        episodic_returns.append(float(np.sum(rewards)))

    env.close()

    q_mean = float(np.mean(q_preds_all)) if q_preds_all else 0.0
    g_mean = float(np.mean(g_true_all)) if g_true_all else 0.0
    gap_mean = q_mean - g_mean
    epi_return_mean = float(np.mean(episodic_returns)) if episodic_returns else 0.0
    return q_mean, g_mean, gap_mean, epi_return_mean


def run_training(
    algo_name: str,
    use_double: bool,
    cfg: ExpConfig,
    device: torch.device,
) -> List[Dict[str, float]]:
    set_global_seeds(cfg.seed)
    robust_cfg = build_robust_config()

    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNFamilyAgent(state_dim, action_dim, device, cfg, use_double=use_double)
    recent_returns = deque(maxlen=cfg.return_window)

    metrics: List[Dict[str, float]] = []

    for episode in range(1, cfg.total_episodes + 1):
        state, _ = env.reset(seed=cfg.seed + episode)
        episode_return = 0.0

        for _ in range(cfg.max_steps):
            action = agent.select_action(state)
            robust_input = {"action": action, "robust_type": "action", "robust_config": robust_cfg}
            next_state, reward, terminated, truncated, _ = env.step(robust_input)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_return += float(reward)

            if done:
                break

        recent_returns.append(episode_return)
        agent.decay_epsilon()

        if episode % cfg.eval_interval == 0:
            q_pred, true_discounted_return, gap, eval_return = evaluate_overestimation_on_policy(
                agent=agent,
                cfg=cfg,
                robust_cfg=robust_cfg,
                eval_seed_base=cfg.seed + cfg.eval_seed_offset + episode,
            )
            train_return = float(np.mean(recent_returns))
            metrics.append(
                {
                    "algorithm": algo_name,
                    "episode": float(episode),
                    "predicted_q": q_pred,
                    "true_discounted_return": true_discounted_return,
                    "overestimation_gap": gap,
                    "train_true_return": train_return,
                    "eval_true_return": eval_return,
                }
            )
            print(
                f"[{algo_name}] Ep {episode:4d} | Q_predict={q_pred:8.3f} | "
                f"G_true={true_discounted_return:8.3f} | Gap={gap:8.3f} | "
                f"EvalReturn={eval_return:8.3f} | Eps={agent.epsilon:.3f}"
            )

    env.close()
    return metrics


def save_metrics_csv(metrics: List[Dict[str, float]], out_path: str):
    fieldnames = [
        "algorithm",
        "episode",
        "predicted_q",
        "true_discounted_return",
        "overestimation_gap",
        "train_true_return",
        "eval_true_return",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def plot_aligned_values(dqn_metrics: List[Dict[str, float]], ddqn_metrics: List[Dict[str, float]], out_path: str):
    ep_dqn = [int(m["episode"]) for m in dqn_metrics]
    ep_ddqn = [int(m["episode"]) for m in ddqn_metrics]

    q_dqn = [m["predicted_q"] for m in dqn_metrics]
    q_ddqn = [m["predicted_q"] for m in ddqn_metrics]

    g_dqn = [m["true_discounted_return"] for m in dqn_metrics]
    g_ddqn = [m["true_discounted_return"] for m in ddqn_metrics]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(ep_dqn, q_dqn, color="tab:red", linewidth=2.0, label="DQN Predicted Q")
    ax.plot(ep_ddqn, q_ddqn, color="tab:orange", linewidth=2.0, label="Double DQN Predicted Q")
    ax.plot(ep_dqn, g_dqn, color="tab:blue", linestyle="--", linewidth=2.0, label="DQN True Discounted Return")
    ax.plot(ep_ddqn, g_ddqn, color="tab:green", linestyle="--", linewidth=2.0, label="Double DQN True Discounted Return")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Value (Same Scale)")
    ax.set_title("Experiment 1A: Q Prediction vs True Discounted Return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_true_return_curve(dqn_metrics: List[Dict[str, float]], ddqn_metrics: List[Dict[str, float]], out_path: str):
    ep_dqn = [int(m["episode"]) for m in dqn_metrics]
    ep_ddqn = [int(m["episode"]) for m in ddqn_metrics]
    r_dqn = [m["eval_true_return"] for m in dqn_metrics]
    r_ddqn = [m["eval_true_return"] for m in ddqn_metrics]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ep_dqn, r_dqn, color="tab:blue", linewidth=2.0, label="DQN Eval Episodic Return")
    ax.plot(ep_ddqn, r_ddqn, color="tab:green", linewidth=2.0, label="Double DQN Eval Episodic Return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Experiment 1A: Evaluation Episodic Return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_summary(dqn_metrics: List[Dict[str, float]], ddqn_metrics: List[Dict[str, float]], out_path: str):
    dqn_gap = np.array([m["overestimation_gap"] for m in dqn_metrics], dtype=np.float32)
    ddqn_gap = np.array([m["overestimation_gap"] for m in ddqn_metrics], dtype=np.float32)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Experiment 1A Summary\n")
        f.write("=====================\n\n")
        f.write("Metric: overestimation_gap = Q_predict - TrueDiscountedReturn\n")
        f.write("Evaluation is on-policy: states visited by current greedy policy.\n\n")
        f.write(f"DQN average gap: {dqn_gap.mean():.4f}\n")
        f.write(f"Double DQN average gap: {ddqn_gap.mean():.4f}\n")
        f.write(f"DQN max gap: {dqn_gap.max():.4f}\n")
        f.write(f"Double DQN max gap: {ddqn_gap.max():.4f}\n")
        f.write(f"Gap reduction (avg): {(dqn_gap.mean() - ddqn_gap.mean()):.4f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1A: DQN vs Double DQN overestimation bias")
    parser.add_argument("--episodes", type=int, default=1000, help="Total training episodes per algorithm")
    parser.add_argument("--eval-interval", type=int, default=50, help="Evaluate every N episodes")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Greedy evaluation episodes per checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output folder under this script directory")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = ExpConfig(
        seed=args.seed,
        total_episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
    )

    set_global_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"[INFO] On-policy evaluation enabled | eval_interval={cfg.eval_interval} | "
        f"eval_episodes={cfg.eval_episodes}"
    )

    dqn_metrics = run_training(
        algo_name="DQN",
        use_double=False,
        cfg=cfg,
        device=device,
    )
    ddqn_metrics = run_training(
        algo_name="DoubleDQN",
        use_double=True,
        cfg=cfg,
        device=device,
    )

    all_metrics = dqn_metrics + ddqn_metrics
    save_metrics_csv(all_metrics, os.path.join(out_dir, "metrics.csv"))
    plot_aligned_values(
        dqn_metrics,
        ddqn_metrics,
        os.path.join(out_dir, "overestimation_aligned.png"),
    )
    plot_true_return_curve(
        dqn_metrics,
        ddqn_metrics,
        os.path.join(out_dir, "eval_return_curve.png"),
    )
    save_summary(dqn_metrics, ddqn_metrics, os.path.join(out_dir, "summary.txt"))

    print(f"[DONE] Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
