
import os
import random
import pickle
import time
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import robust_gymnasium as gym
from robust_gymnasium.configs.robust_setting import get_config

# Hyperparameters
ENV_NAME       = "LunarLander-v3"
SEED           = 42
TOTAL_EPISODES = 500
MAX_STEPS      = 1000
GAMMA          = 0.99
LR             = 5e-4
BATCH_SIZE     = 64
BUFFER_SIZE    = 100_000
TAU            = 1e-3
EPS_START      = 1.0
EPS_END        = 0.01
EPS_DECAY      = 0.995
HIDDEN_DIM     = 128
UPDATE_EVERY   = 4
EVAL_EPISODES  = 20
SAVE_DIR       = "results/delayed_reward_dqn"
ROBUSTNESS_EPS = 300

REWARD_CONFIGS = {
    "baseline":  {"mode": "baseline", "delay": 1,  "label": "Baseline (Default)"},
    "dense":     {"mode": "dense",    "delay": 1,  "label": "Dense (Extra Shaping)"},
    "medium_10": {"mode": "medium",   "delay": 10, "label": "Medium Delay (K=10)"},
    "medium_20": {"mode": "medium",   "delay": 20, "label": "Medium Delay (K=20)"},
    "sparse":    {"mode": "sparse",   "delay": 1,  "label": "Sparse (Terminal Only)"},
}


# Delayed-Reward Wrapper
class DelayedRewardWrapper:

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
            # Extra shaping: centering, uprightness, and landing contact
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


#Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# DQN Agent
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
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            return int(self.qnet(s).argmax(1).item())

    def q_values(self, state):
        """Return Q(s, ·) array for logging."""
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            return self.qnet(s).cpu().numpy().flatten()

    def step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.step_count += 1
        if self.step_count % UPDATE_EVERY == 0 and len(self.buffer) >= BATCH_SIZE:
            self._learn()

    def _learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        s  = torch.as_tensor(states,  device=self.device)
        a  = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        r  = torch.as_tensor(rewards, device=self.device).unsqueeze(1)
        ns = torch.as_tensor(next_states, device=self.device)
        d  = torch.as_tensor(dones,   device=self.device).unsqueeze(1)

        q = self.qnet(s).gather(1, a)
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


# Training Loop
def train(wrapper, agent, args, n_episodes):
    rewards_history = []
    recent = deque(maxlen=100)

    for ep in range(1, n_episodes + 1):
        state, _ = wrapper.reset(seed=SEED + ep)
        ep_original_reward = 0.0

        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            robust_input = {"action": action, "robust_type": "action",
                            "robust_config": args}
            ns, delivered, term, trunc, info = wrapper.step(robust_input)
            done = term or trunc

            agent.step(state, action, delivered, ns, done)
            ep_original_reward += info.get("original_reward", delivered)
            state = ns
            if done:
                break

        agent.decay_epsilon()
        rewards_history.append(ep_original_reward)
        recent.append(ep_original_reward)

        if ep % 50 == 0:
            print(f"  Ep {ep:4d} | R={ep_original_reward:7.1f} "
                  f"| Avg100={np.mean(recent):7.1f} | ε={agent.epsilon:.3f}")

    return rewards_history


#Evaluation
def evaluate(wrapper, agent, args, n_episodes):
    episodes = []
    for ep in range(n_episodes):
        state, _ = wrapper.reset(seed=SEED + 80000 + ep)
        data = {"rewards": [], "delivered_rewards": [],
                "q_values": [], "td_errors": []}

        for _ in range(MAX_STEPS):
            qv = agent.q_values(state)
            action = int(np.argmax(qv))
            robust_input = {"action": action, "robust_type": "action",
                            "robust_config": args}
            ns, delivered, term, trunc, info = wrapper.step(robust_input)
            done = term or trunc

            nq = agent.q_values(ns)
            td_target = delivered + GAMMA * np.max(nq) * (1.0 - float(done))
            td_err = abs(td_target - qv[action])

            data["rewards"].append(info.get("original_reward", delivered))
            data["delivered_rewards"].append(delivered)
            data["q_values"].append(float(qv[action]))
            data["td_errors"].append(float(td_err))

            state = ns
            if done:
                break

        episodes.append(data)
    return episodes


#Main 
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}\n")

    args = get_config().parse_args([])
    args.noise_factor = "none"
    args.noise_sigma  = 0.0

    results = {}
    t0 = time.time()

    # train and evaluate each reward configuration
    for name, cfg in REWARD_CONFIGS.items():
        print(f"{'='*60}")
        print(f"[{name.upper()}] mode={cfg['mode']}, delay={cfg['delay']}")
        print(f"{'='*60}")

        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

        env = DelayedRewardWrapper(
            gym.make(ENV_NAME), mode=cfg["mode"], delay_steps=cfg["delay"])
        agent = DQNAgent(env.observation_space.shape[0],
                         env.action_space.n, device)

        train_rewards = train(env, agent, args, TOTAL_EPISODES)
        env.close()

        eval_env = DelayedRewardWrapper(
            gym.make(ENV_NAME), mode=cfg["mode"], delay_steps=cfg["delay"])
        eval_data = evaluate(eval_env, agent, args, EVAL_EPISODES)
        eval_env.close()

        results[name] = {"train_rewards": train_rewards,
                         "eval_data": eval_data, "config": cfg}
        torch.save(agent.qnet.state_dict(),
                   os.path.join(SAVE_DIR, f"dqn_{name}.pth"))

    #  robustness sweep over delay values
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
        agent = DQNAgent(env.observation_space.shape[0],
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
    agent = DQNAgent(env.observation_space.shape[0],
                     env.action_space.n, device)
    rews = train(env, agent, args, ROBUSTNESS_EPS)
    env.close()
    rob["delays"].append(1000)
    rob["mean_rewards"].append(float(np.mean(rews[-50:])))
    rob["std_rewards"].append(float(np.std(rews[-50:])))

    results["robustness"] = rob

    #  Save 
    pkl_path = os.path.join(SAVE_DIR, "experiment_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    elapsed = time.time() - t0
    print(f"\n[INFO] Total time: {elapsed / 60:.1f} min")
    print(f"[INFO] Results → {pkl_path}")
    print("[INFO] Next: python visualize_delayed_reward.py")


if __name__ == "__main__":
    main()
