"""
Delayed Reward Structure Analysis for REINFORCE+Normalization on LunarLander-v3.

Trains a pure policy-gradient agent (no critic) under five reward delivery modes:
  baseline  – original LunarLander reward (dense potential-based shaping)
  dense     – original + extra per-step shaping (centering, orientation, landing)
  medium_10 – accumulated reward delivered every 10 steps
  medium_20 – accumulated reward delivered every 20 steps
  sparse    – accumulated reward delivered only at episode end

After training each configuration, evaluates the agent and collects step-level
metrics (return-to-go G_t, gradient signal magnitude |G̃_t|, delivered vs
original rewards).  A robustness sweep over varying delay lengths is also
performed.

All results are saved to results/delayed_reward_reinforce/experiment_results.pkl
for post-hoc visualization by visualize_delayed_reward.py.
"""

import os
import random
import pickle
import time
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import robust_gymnasium as gym
from robust_gymnasium.configs.robust_setting import get_config

# ─── Hyperparameters (consistent with train_REINFORCE_normalization.py) ──────
ENV_NAME       = "LunarLander-v3"
SEED           = 42
TOTAL_EPISODES = 1000
MAX_STEPS      = 1000
GAMMA          = 0.99
LR             = 1e-3
HIDDEN_DIM     = 128
EVAL_EPISODES  = 20
SAVE_DIR       = "results/delayed_reward_reinforce"
ROBUSTNESS_EPS = 500

REWARD_CONFIGS = {
    "baseline":  {"mode": "baseline", "delay": 1,  "label": "Baseline (Default)"},
    "dense":     {"mode": "dense",    "delay": 1,  "label": "Dense (Extra Shaping)"},
    "medium_10": {"mode": "medium",   "delay": 10, "label": "Medium Delay (K=10)"},
    "medium_20": {"mode": "medium",   "delay": 20, "label": "Medium Delay (K=20)"},
    "sparse":    {"mode": "sparse",   "delay": 1,  "label": "Sparse (Terminal Only)"},
}


# ─── Delayed-Reward Wrapper ──────────────────────────────────────────────────
class DelayedRewardWrapper:
    """Intercepts environment rewards and controls *when* they are delivered."""

    def __init__(self, env, mode="baseline", delay_steps=1):
        self.env = env
        self.mode = mode
        self.delay_steps = max(1, delay_steps)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._buffer = 0.0
        self._steps = 0

    def reset(self, **kwargs):
        self._buffer = 0.0
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info["original_reward"] = reward

        if self.mode == "baseline":
            delivered = reward
        elif self.mode == "dense":
            x, angle = next_state[0], next_state[4]
            leg1, leg2 = next_state[6], next_state[7]
            bonus = 0.2 * (leg1 + leg2) - 0.3 * abs(x) - 0.2 * abs(angle)
            delivered = reward + bonus
        elif self.mode == "medium":
            self._buffer += reward
            self._steps += 1
            if self._steps % self.delay_steps == 0 or done:
                delivered = self._buffer
                self._buffer = 0.0
            else:
                delivered = 0.0
        elif self.mode == "sparse":
            self._buffer += reward
            if done:
                delivered = self._buffer
                self._buffer = 0.0
            else:
                delivered = 0.0
        else:
            delivered = reward

        return next_state, delivered, terminated, truncated, info

    def close(self):
        self.env.close()


# ─── Policy Network ──────────────────────────────────────────────────────────
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ─── Discounted Returns ──────────────────────────────────────────────────────
def compute_returns(rewards, gamma=GAMMA):
    """Compute discounted return-to-go G_t for each timestep."""
    G = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        G.append(running)
    G.reverse()
    return G


# ─── REINFORCE Agent ─────────────────────────────────────────────────────────
class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.action_dim = action_dim
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

    def select_action(self, state):
        s = torch.as_tensor(state, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        logits = self.policy(s)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).squeeze(0)

    def greedy_action(self, state):
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            return int(self.policy(s).argmax(-1).item())

    def update(self, log_probs, rewards):
        """REINFORCE with return normalization (per-episode update)."""
        returns = compute_returns(rewards, GAMMA)
        G = torch.tensor(returns, dtype=torch.float32, device=self.device)
        G = (G - G.mean()) / (G.std() + 1e-8)

        lps = torch.stack(log_probs)
        loss = -(lps * G).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ─── Training Loop ───────────────────────────────────────────────────────────
def train(wrapper, agent, args, n_episodes):
    """Train REINFORCE agent; returns list of per-episode *original* rewards."""
    rewards_history = []
    recent = deque(maxlen=100)

    for ep in range(1, n_episodes + 1):
        state, _ = wrapper.reset(seed=SEED + ep)
        ep_original = 0.0
        ep_log_probs = []
        ep_delivered = []

        for _ in range(MAX_STEPS):
            action, log_prob = agent.select_action(state)
            robust_input = {"action": action, "robust_type": "action",
                            "robust_config": args}
            ns, delivered, term, trunc, info = wrapper.step(robust_input)
            done = term or trunc

            ep_log_probs.append(log_prob)
            ep_delivered.append(delivered)
            ep_original += info.get("original_reward", delivered)
            state = ns
            if done:
                break

        agent.update(ep_log_probs, ep_delivered)
        rewards_history.append(ep_original)
        recent.append(ep_original)

        if ep % 50 == 0:
            print(f"  Ep {ep:4d} | R={ep_original:7.1f} "
                  f"| Avg100={np.mean(recent):7.1f}")

    return rewards_history


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(wrapper, agent, args, n_episodes):
    """Greedy evaluation collecting step-level return-to-go and gradient signal."""
    episodes = []
    for ep in range(n_episodes):
        state, _ = wrapper.reset(seed=SEED + 80000 + ep)
        data = {"rewards": [], "delivered_rewards": [],
                "q_values": [], "td_errors": []}
        step_delivered = []

        for _ in range(MAX_STEPS):
            action = agent.greedy_action(state)
            robust_input = {"action": action, "robust_type": "action",
                            "robust_config": args}
            ns, delivered, term, trunc, info = wrapper.step(robust_input)
            done = term or trunc

            data["rewards"].append(info.get("original_reward", delivered))
            data["delivered_rewards"].append(delivered)
            step_delivered.append(delivered)
            state = ns
            if done:
                break

        # Post-episode: compute return-to-go G_t and normalized |G̃_t|
        # G_t is the REINFORCE analog of V(s) / Q(s,a)
        returns = compute_returns(step_delivered, GAMMA)
        G = np.array(returns, dtype=np.float64)
        G_norm = (G - G.mean()) / (G.std() + 1e-8)

        data["q_values"] = G.tolist()
        data["td_errors"] = np.abs(G_norm).tolist()
        episodes.append(data)

    return episodes


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}\n")

    args = get_config().parse_args([])
    args.noise_factor = "none"
    args.noise_sigma  = 0.0

    results = {}
    t0 = time.time()

    # ── Phase 1: train & evaluate each reward configuration ──────────────────
    for name, cfg in REWARD_CONFIGS.items():
        print(f"{'='*60}")
        print(f"[{name.upper()}] mode={cfg['mode']}, delay={cfg['delay']}")
        print(f"{'='*60}")

        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

        env = DelayedRewardWrapper(
            gym.make(ENV_NAME), mode=cfg["mode"], delay_steps=cfg["delay"])
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = REINFORCEAgent(state_dim, action_dim, device)

        train_rewards = train(env, agent, args, TOTAL_EPISODES)
        env.close()

        eval_env = DelayedRewardWrapper(
            gym.make(ENV_NAME), mode=cfg["mode"], delay_steps=cfg["delay"])
        eval_data = evaluate(eval_env, agent, args, EVAL_EPISODES)
        eval_env.close()

        results[name] = {"train_rewards": train_rewards,
                         "eval_data": eval_data, "config": cfg}
        torch.save(agent.policy.state_dict(),
                   os.path.join(SAVE_DIR, f"reinforce_{name}.pth"))

    # ── Phase 2: robustness sweep over delay values ──────────────────────────
    print(f"\n{'='*60}")
    print("ROBUSTNESS SWEEP – varying reward delay")
    print(f"{'='*60}")

    delay_values = [1, 5, 10, 15, 20, 30, 50, 100]
    rob = {"delays": [], "mean_rewards": [], "std_rewards": []}

    for d in delay_values:
        print(f"\n  delay = {d}")
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        env = DelayedRewardWrapper(gym.make(ENV_NAME), mode="medium",
                                   delay_steps=d)
        agent = REINFORCEAgent(env.observation_space.shape[0],
                               env.action_space.n, device)
        rews = train(env, agent, args, ROBUSTNESS_EPS)
        env.close()
        tail = rews[-50:]
        rob["delays"].append(d)
        rob["mean_rewards"].append(float(np.mean(tail)))
        rob["std_rewards"].append(float(np.std(tail)))

    # Sparse extreme
    print(f"\n  delay = sparse (terminal)")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    env = DelayedRewardWrapper(gym.make(ENV_NAME), mode="sparse")
    agent = REINFORCEAgent(env.observation_space.shape[0],
                           env.action_space.n, device)
    rews = train(env, agent, args, ROBUSTNESS_EPS)
    env.close()
    rob["delays"].append(1000)
    rob["mean_rewards"].append(float(np.mean(rews[-50:])))
    rob["std_rewards"].append(float(np.std(rews[-50:])))

    results["robustness"] = rob

    # ── Save ─────────────────────────────────────────────────────────────────
    pkl_path = os.path.join(SAVE_DIR, "experiment_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    elapsed = time.time() - t0
    print(f"\n[INFO] Total time: {elapsed / 60:.1f} min")
    print(f"[INFO] Results → {pkl_path}")
    print("[INFO] Next: python visualize_delayed_reward.py")


if __name__ == "__main__":
    main()
