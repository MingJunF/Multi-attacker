"""
Delayed Reward Structure Analysis for PPO on LunarLander-v3 (Discrete).

Trains PPO under five reward delivery modes:
  baseline  – original LunarLander reward (dense potential-based shaping)
  dense     – original + extra per-step shaping (centering, orientation, landing)
  medium_10 – accumulated reward delivered every 10 steps
  medium_20 – accumulated reward delivered every 20 steps
  sparse    – accumulated reward delivered only at episode end

After training each configuration, evaluates the agent and collects step-level
metrics (V(s), TD-errors, delivered vs original rewards).  A robustness sweep
over varying delay lengths is also performed.

All results are saved to results/delayed_reward_ppo/experiment_results.pkl
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
import torch.nn.functional as F
from torch.distributions import Categorical

import robust_gymnasium as gym
from robust_gymnasium.configs.robust_setting import get_config

# ─── Hyperparameters (consistent with train_PPO.py) ─────────────────────────
ENV_NAME       = "LunarLander-v3"
SEED           = 42
TOTAL_EPISODES = 500
MAX_STEPS      = 1000
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
LR             = 3e-4
CLIP_RANGE     = 0.2
VALUE_COEF     = 0.5
ENTROPY_COEF   = 0.01
BATCH_SIZE     = 64
N_EPOCHS       = 10
N_STEPS        = 2048
HIDDEN_DIM     = 64
MAX_GRAD_NORM  = 0.5
EVAL_EPISODES  = 20
SAVE_DIR       = "results/delayed_reward_ppo"
ROBUSTNESS_EPS = 300

REWARD_CONFIGS = {
    "baseline":  {"mode": "baseline", "delay": 1,  "label": "Baseline (Default)"},
    "dense":     {"mode": "dense",    "delay": 1,  "label": "Dense (Extra Shaping)"},
    "medium_10": {"mode": "medium",   "delay": 10, "label": "Medium Delay (K=10)"},
    "medium_20": {"mode": "medium",   "delay": 20, "label": "Medium Delay (K=20)"},
    "sparse":    {"mode": "sparse",   "delay": 1,  "label": "Sparse (Terminal Only)"},
}


# ─── Delayed-Reward Wrapper (identical to DQN version) ──────────────────────
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


# ─── Actor Network ───────────────────────────────────────────────────────────
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)

    def get_action(self, state, device):
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        probs = self.forward(s)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()


# ─── Critic Network ─────────────────────────────────────────────────────────
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─── Rollout Buffer ──────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def push(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get(self):
        return (np.array(self.states, dtype=np.float32),
                np.array(self.actions, dtype=np.int64),
                np.array(self.rewards, dtype=np.float32),
                np.array(self.dones, dtype=np.float32),
                np.array(self.log_probs, dtype=np.float32),
                np.array(self.values, dtype=np.float32))

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []

    def __len__(self):
        return len(self.states)


# ─── PPO Agent ───────────────────────────────────────────────────────────────
class PPOAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.action_dim = action_dim
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR)
        self.buffer = RolloutBuffer()

    def select_action(self, state):
        action, log_prob = self.actor.get_action(state, self.device)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            value = self.critic(s).item()
        return action, log_prob, value

    def get_value(self, state):
        """Return V(s) scalar for evaluation logging."""
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            return self.critic(s).item()

    def compute_gae(self, rewards, dones, values, next_value):
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + GAMMA * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        return advantages, advantages + values

    def update(self, next_state):
        states, actions, rewards, dones, old_lps, values = self.buffer.get()
        with torch.no_grad():
            s = torch.as_tensor(next_state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            next_val = self.critic(s).item()

        advantages, returns = self.compute_gae(rewards, dones, values, next_val)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        s_t = torch.as_tensor(states, device=self.device)
        a_t = torch.as_tensor(actions, device=self.device)
        old_lp_t = torch.as_tensor(old_lps, device=self.device)
        adv_t = torch.as_tensor(advantages, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)

        for _ in range(N_EPOCHS):
            idx = np.arange(len(states))
            np.random.shuffle(idx)
            for start in range(0, len(states), BATCH_SIZE):
                b = idx[start:start + BATCH_SIZE]
                probs = self.actor(s_t[b])
                dist = Categorical(probs)
                new_lp = dist.log_prob(a_t[b])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp_t[b])
                s1 = ratio * adv_t[b]
                s2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * adv_t[b]
                pi_loss = -torch.min(s1, s2).mean()

                v_pred = self.critic(s_t[b])
                v_loss = F.mse_loss(v_pred, ret_t[b])

                self.actor_opt.zero_grad()
                pi_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
                self.critic_opt.step()

        self.buffer.clear()


# ─── Training Loop ───────────────────────────────────────────────────────────
def train(wrapper, agent, args, n_episodes):
    """Train PPO agent; returns list of per-episode *original* rewards."""
    rewards_history = []
    recent = deque(maxlen=100)

    for ep in range(1, n_episodes + 1):
        state, _ = wrapper.reset(seed=SEED + ep)
        ep_original = 0.0

        for _ in range(MAX_STEPS):
            action, log_prob, value = agent.select_action(state)
            robust_input = {"action": action, "robust_type": "action",
                            "robust_config": args}
            ns, delivered, term, trunc, info = wrapper.step(robust_input)
            done = term or trunc

            agent.buffer.push(state, action, delivered, done, log_prob, value)
            ep_original += info.get("original_reward", delivered)
            state = ns

            # PPO update when buffer is full
            if len(agent.buffer) >= N_STEPS:
                agent.update(state)

            if done:
                break

        rewards_history.append(ep_original)
        recent.append(ep_original)

        if ep % 50 == 0:
            print(f"  Ep {ep:4d} | R={ep_original:7.1f} "
                  f"| Avg100={np.mean(recent):7.1f}")

    return rewards_history


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(wrapper, agent, args, n_episodes):
    """Greedy evaluation collecting step-level V(s) and TD-error."""
    episodes = []
    for ep in range(n_episodes):
        state, _ = wrapper.reset(seed=SEED + 80000 + ep)
        data = {"rewards": [], "delivered_rewards": [],
                "q_values": [], "td_errors": []}

        for _ in range(MAX_STEPS):
            vs = agent.get_value(state)
            # Greedy action from policy
            s_t = torch.as_tensor(state, dtype=torch.float32,
                                  device=agent.device).unsqueeze(0)
            with torch.no_grad():
                action = int(agent.actor(s_t).argmax(-1).item())

            robust_input = {"action": action, "robust_type": "action",
                            "robust_config": args}
            ns, delivered, term, trunc, info = wrapper.step(robust_input)
            done = term or trunc

            vs_next = agent.get_value(ns)
            # TD-error: |r + γ·V(s') − V(s)|
            td_target = delivered + GAMMA * vs_next * (1.0 - float(done))
            td_err = abs(td_target - vs)

            data["rewards"].append(info.get("original_reward", delivered))
            data["delivered_rewards"].append(delivered)
            data["q_values"].append(float(vs))  # V(s) stored in q_values key
            data["td_errors"].append(float(td_err))

            state = ns
            if done:
                break

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
        agent = PPOAgent(state_dim, action_dim, device)

        train_rewards = train(env, agent, args, TOTAL_EPISODES)
        env.close()

        eval_env = DelayedRewardWrapper(
            gym.make(ENV_NAME), mode=cfg["mode"], delay_steps=cfg["delay"])
        eval_data = evaluate(eval_env, agent, args, EVAL_EPISODES)
        eval_env.close()

        results[name] = {"train_rewards": train_rewards,
                         "eval_data": eval_data, "config": cfg}
        torch.save({"actor": agent.actor.state_dict(),
                     "critic": agent.critic.state_dict()},
                   os.path.join(SAVE_DIR, f"ppo_{name}.pth"))

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
        agent = PPOAgent(env.observation_space.shape[0],
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
    agent = PPOAgent(env.observation_space.shape[0],
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
