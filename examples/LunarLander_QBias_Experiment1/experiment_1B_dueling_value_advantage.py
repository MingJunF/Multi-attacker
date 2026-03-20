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
    update_every: int = 4
    return_window: int = 100
    hidden_ddqn: int = 128
    hidden_dueling: int = 256


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


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int):
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

    def forward_with_streams(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.feature(x)
        value = self.value_stream(feat)
        advantage_raw = self.advantage_stream(feat)
        advantage = advantage_raw - advantage_raw.mean(dim=1, keepdim=True)
        q_values = value + advantage
        return q_values, value, advantage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_values, _, _ = self.forward_with_streams(x)
        return q_values


class DoubleDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, device: torch.device, cfg: ExpConfig):
        self.action_dim = action_dim
        self.device = device
        self.cfg = cfg

        self.qnet = QNetwork(state_dim, action_dim, cfg.hidden_ddqn).to(device)
        self.target_net = QNetwork(state_dim, action_dim, cfg.hidden_ddqn).to(device)
        self.target_net.load_state_dict(self.qnet.state_dict())

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.epsilon = cfg.eps_start
        self.step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.qnet(s)
        return int(q.argmax(dim=1).item())

    def select_action_greedy(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.qnet(s)
        q_np = q.squeeze(0).detach().cpu().numpy()
        return int(np.argmax(q_np)), q_np

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
            next_actions = self.qnet(ns).argmax(dim=1, keepdim=True)
            next_q = self.target_net(ns).gather(1, next_actions)
            q_target = r + self.cfg.gamma * next_q * (1.0 - d)

        loss = nn.functional.mse_loss(q_pred, q_target)
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


class DuelingDoubleDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, device: torch.device, cfg: ExpConfig):
        self.action_dim = action_dim
        self.device = device
        self.cfg = cfg

        self.qnet = DuelingQNetwork(state_dim, action_dim, cfg.hidden_dueling).to(device)
        self.target_net = DuelingQNetwork(state_dim, action_dim, cfg.hidden_dueling).to(device)
        self.target_net.load_state_dict(self.qnet.state_dict())

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.epsilon = cfg.eps_start
        self.step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.qnet(s)
        return int(q.argmax(dim=1).item())

    def select_action_greedy_with_streams(self, state: np.ndarray) -> Tuple[int, np.ndarray, float, np.ndarray]:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q, v, a = self.qnet.forward_with_streams(s)

        q_np = q.squeeze(0).detach().cpu().numpy()
        v_np = float(v.squeeze(0).item())
        a_np = a.squeeze(0).detach().cpu().numpy()
        return int(np.argmax(q_np)), q_np, v_np, a_np

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
            next_actions = self.qnet(ns).argmax(dim=1, keepdim=True)
            next_q = self.target_net(ns).gather(1, next_actions)
            q_target = r + self.cfg.gamma * next_q * (1.0 - d)

        loss = nn.functional.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=5.0)
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


def train_agent(agent, cfg: ExpConfig, robust_cfg, env_seed_offset: int, tag: str) -> List[float]:
    env = gym.make(cfg.env_name)
    returns: List[float] = []
    recent = deque(maxlen=cfg.return_window)

    for ep in range(1, cfg.total_episodes + 1):
        state, _ = env.reset(seed=cfg.seed + env_seed_offset + ep)
        total_reward = 0.0

        for _ in range(cfg.max_steps):
            action = agent.select_action(state)
            robust_input = {"action": action, "robust_type": "action", "robust_config": robust_cfg}
            next_state, reward, terminated, truncated, _ = env.step(robust_input)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += float(reward)

            if done:
                break

        agent.decay_epsilon()
        returns.append(total_reward)
        recent.append(total_reward)

        if ep % 20 == 0:
            print(
                f"[{tag}] Episode {ep:4d} | Reward {total_reward:8.3f} | "
                f"Avg100 {np.mean(recent):8.3f} | Eps {agent.epsilon:.3f}"
            )

    env.close()
    return returns


def evaluate_double_ddqn_episode(agent: DoubleDQNAgent, cfg: ExpConfig, robust_cfg, eval_seed: int) -> Tuple[List[Dict[str, float]], float]:
    env = gym.make(cfg.env_name)
    state, _ = env.reset(seed=eval_seed)

    trace: List[Dict[str, float]] = []
    total_reward = 0.0

    for t in range(cfg.max_steps):
        action, q_values = agent.select_action_greedy(state)
        robust_input = {"action": action, "robust_type": "action", "robust_config": robust_cfg}
        next_state, reward, terminated, truncated, _ = env.step(robust_input)

        trace.append(
            {
                "t": float(t),
                "action": float(action),
                "x": float(state[0]),
                "y": float(state[1]),
                "vx": float(state[2]),
                "vy": float(state[3]),
                "angle": float(state[4]),
                "omega": float(state[5]),
                "left_leg": float(state[6]),
                "right_leg": float(state[7]),
                "q_max": float(np.max(q_values)),
            }
        )

        state = next_state
        total_reward += float(reward)
        if terminated or truncated:
            break

    env.close()
    return trace, total_reward


def evaluate_dueling_episode(agent: DuelingDoubleDQNAgent, cfg: ExpConfig, robust_cfg, eval_seed: int) -> Tuple[List[Dict[str, float]], float]:
    env = gym.make(cfg.env_name)
    state, _ = env.reset(seed=eval_seed)

    trace: List[Dict[str, float]] = []
    total_reward = 0.0

    for t in range(cfg.max_steps):
        action, q_values, value, advantage = agent.select_action_greedy_with_streams(state)
        robust_input = {"action": action, "robust_type": "action", "robust_config": robust_cfg}
        next_state, reward, terminated, truncated, _ = env.step(robust_input)

        trace.append(
            {
                "t": float(t),
                "action": float(action),
                "x": float(state[0]),
                "y": float(state[1]),
                "vx": float(state[2]),
                "vy": float(state[3]),
                "angle": float(state[4]),
                "omega": float(state[5]),
                "left_leg": float(state[6]),
                "right_leg": float(state[7]),
                "v_s": float(value),
                "a0": float(advantage[0]),
                "a1": float(advantage[1]),
                "a2": float(advantage[2]),
                "a3": float(advantage[3]),
                "a_max": float(np.max(advantage)),
                "a_min": float(np.min(advantage)),
                "q_max": float(np.max(q_values)),
            }
        )

        state = next_state
        total_reward += float(reward)
        if terminated or truncated:
            break

    env.close()
    return trace, total_reward


def _longest_true_segment(mask: np.ndarray) -> Tuple[int, int]:
    best_start, best_end = 0, 0
    start = None

    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            if i - start > best_end - best_start:
                best_start, best_end = start, i
            start = None

    if start is not None and len(mask) - start > best_end - best_start:
        best_start, best_end = start, len(mask)

    return best_start, best_end


def select_high_altitude_window(trace: List[Dict[str, float]], min_len: int = 15) -> Tuple[int, int, str]:
    y = np.array([row["y"] for row in trace], dtype=np.float32)
    vy = np.array([row["vy"] for row in trace], dtype=np.float32)
    left_leg = np.array([row["left_leg"] for row in trace], dtype=np.float32)
    right_leg = np.array([row["right_leg"] for row in trace], dtype=np.float32)

    y_thr = float(np.quantile(y, 0.7))
    primary = (y >= y_thr) & (vy < -0.02) & (left_leg < 0.5) & (right_leg < 0.5)

    s, e = _longest_true_segment(primary)
    if e - s >= min_len:
        return s, e, "primary: high altitude + descending + no leg contact"

    fallback = (y >= float(np.quantile(y, 0.6))) & (left_leg < 0.5) & (right_leg < 0.5)
    s, e = _longest_true_segment(fallback)
    if e - s >= min_len:
        return s, e, "fallback: top 40% altitude + no leg contact"

    final_len = min(max(min_len, 10), len(trace))
    return 0, final_len, "fallback: first window"


def save_csv(rows: List[Dict[str, float]], out_path: str):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_value_advantage_window(dueling_window: List[Dict[str, float]], out_path: str):
    t = [int(r["t"]) for r in dueling_window]
    v_s = [r["v_s"] for r in dueling_window]
    a_max = [r["a_max"] for r in dueling_window]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, v_s, label="V(s)", color="tab:blue", linewidth=2.2)
    ax.plot(t, a_max, label="max A(s,a)", color="tab:orange", linewidth=2.2)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title("Experiment 1B: High-Altitude Phase in Dueling Double DQN")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_qmax_comparison(
    ddqn_window: List[Dict[str, float]],
    dueling_window: List[Dict[str, float]],
    out_path: str,
):
    t1 = [int(r["t"]) for r in ddqn_window]
    t2 = [int(r["t"]) for r in dueling_window]

    q1 = [r["q_max"] for r in ddqn_window]
    q2 = [r["q_max"] for r in dueling_window]
    v2 = [r["v_s"] for r in dueling_window]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t1, q1, label="Double DQN max Q", color="tab:red", linewidth=2.0)
    ax.plot(t2, q2, label="Dueling DDQN max Q", color="tab:green", linewidth=2.0)
    ax.plot(t2, v2, label="Dueling V(s)", color="tab:blue", linestyle="--", linewidth=2.0)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title("High-Altitude Window: Qmax Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_summary(
    ddqn_return: float,
    dueling_return: float,
    dueling_window: List[Dict[str, float]],
    window_reason: str,
    out_path: str,
):
    a_all = np.array([[r["a0"], r["a1"], r["a2"], r["a3"]] for r in dueling_window], dtype=np.float32)
    v_s = np.array([r["v_s"] for r in dueling_window], dtype=np.float32)
    q_max = np.array([r["q_max"] for r in dueling_window], dtype=np.float32)

    mean_abs_adv = float(np.mean(np.abs(a_all))) if len(a_all) > 0 else 0.0
    mean_adv_span = float(np.mean(np.max(a_all, axis=1) - np.min(a_all, axis=1))) if len(a_all) > 0 else 0.0
    mean_abs_v = float(np.mean(np.abs(v_s))) if len(v_s) > 0 else 0.0
    dominance_ratio = float(mean_abs_v / (mean_abs_adv + 1e-8))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Experiment 1B Summary\n")
        f.write("=====================\n\n")
        f.write(f"Double DQN eval return: {ddqn_return:.4f}\n")
        f.write(f"Dueling DDQN eval return: {dueling_return:.4f}\n\n")
        f.write(f"Selected window reason: {window_reason}\n")
        f.write(f"Window length: {len(dueling_window)} steps\n\n")
        f.write("Dueling decomposition stats (selected high-altitude window)\n")
        f.write(f"mean(|V(s)|): {mean_abs_v:.6f}\n")
        f.write(f"mean(|A(s,a)|): {mean_abs_adv:.6f}\n")
        f.write(f"mean(max(A)-min(A)): {mean_adv_span:.6f}\n")
        f.write(f"|V| / |A| ratio: {dominance_ratio:.6f}\n")
        if len(q_max) > 0:
            f.write(f"mean(max Q): {float(np.mean(q_max)):.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1B: Dueling value-advantage visualization")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes per model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="outputs_1b", help="Output folder under script dir")
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=20260313,
        help="Evaluation episode seed",
    )
    parser.add_argument(
        "--load-double-path",
        type=str,
        default="",
        help="Optional pre-trained Double DQN weights path (.pth)",
    )
    parser.add_argument(
        "--load-dueling-path",
        type=str,
        default="",
        help="Optional pre-trained Dueling Double DQN weights path (.pth)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ExpConfig(seed=args.seed, total_episodes=args.episodes)
    set_global_seeds(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robust_cfg = build_robust_config()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    tmp_env = gym.make(cfg.env_name)
    state_dim = tmp_env.observation_space.shape[0]
    action_dim = tmp_env.action_space.n
    tmp_env.close()

    double_agent = DoubleDQNAgent(state_dim, action_dim, device, cfg)
    dueling_agent = DuelingDoubleDQNAgent(state_dim, action_dim, device, cfg)

    if args.load_double_path and os.path.isfile(args.load_double_path):
        double_agent.qnet.load_state_dict(torch.load(args.load_double_path, map_location=device))
        double_agent.target_net.load_state_dict(double_agent.qnet.state_dict())
        print(f"[INFO] Loaded Double DQN weights from {args.load_double_path}")
    else:
        print("[INFO] Training Double DQN...")
        train_agent(double_agent, cfg, robust_cfg, env_seed_offset=0, tag="DoubleDQN")
        torch.save(double_agent.qnet.state_dict(), os.path.join(out_dir, "double_ddqn_1b.pth"))

    if args.load_dueling_path and os.path.isfile(args.load_dueling_path):
        dueling_agent.qnet.load_state_dict(torch.load(args.load_dueling_path, map_location=device))
        dueling_agent.target_net.load_state_dict(dueling_agent.qnet.state_dict())
        print(f"[INFO] Loaded Dueling DDQN weights from {args.load_dueling_path}")
    else:
        print("[INFO] Training Dueling Double DQN...")
        train_agent(dueling_agent, cfg, robust_cfg, env_seed_offset=100000, tag="DuelingDDQN")
        torch.save(dueling_agent.qnet.state_dict(), os.path.join(out_dir, "dueling_ddqn_1b.pth"))

    print("[INFO] Running evaluation episodes...")
    ddqn_trace, ddqn_return = evaluate_double_ddqn_episode(double_agent, cfg, robust_cfg, args.eval_seed)
    dueling_trace, dueling_return = evaluate_dueling_episode(dueling_agent, cfg, robust_cfg, args.eval_seed)

    save_csv(ddqn_trace, os.path.join(out_dir, "eval_trace_double_ddqn.csv"))
    save_csv(dueling_trace, os.path.join(out_dir, "eval_trace_dueling_ddqn.csv"))

    s, e, reason = select_high_altitude_window(dueling_trace)
    ddqn_window = ddqn_trace[s:e]
    dueling_window = dueling_trace[s:e]

    save_csv(ddqn_window, os.path.join(out_dir, "window_trace_double_ddqn.csv"))
    save_csv(dueling_window, os.path.join(out_dir, "window_trace_dueling_ddqn.csv"))

    plot_value_advantage_window(
        dueling_window,
        os.path.join(out_dir, "dueling_value_vs_max_advantage.png"),
    )
    plot_qmax_comparison(
        ddqn_window,
        dueling_window,
        os.path.join(out_dir, "ddqn_vs_dueling_qmax_window.png"),
    )

    save_summary(
        ddqn_return,
        dueling_return,
        dueling_window,
        reason,
        os.path.join(out_dir, "summary_1b.txt"),
    )

    print(f"[DONE] 1B outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
