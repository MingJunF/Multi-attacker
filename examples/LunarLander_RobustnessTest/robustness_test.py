import sys
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "LunarLander_A2C"))

import robust_gymnasium as gym
from robust_gymnasium.configs.robust_setting import get_config

# Configuration

NUM_EPISODES = 50
MAX_STEPS = 1000
SEED = 2024
RESULTS_DIR = PROJECT_ROOT / "results" / "RobustnessTest"

MODEL_PATHS = {
    "DQN":          PROJECT_ROOT / "results" / "train_dqn" / "dqn_final.pth",
    "Double DQN":   PROJECT_ROOT / "results" / "train_double_dqn" / "double_dqn_final.pth",
    "Dueling DDQN": PROJECT_ROOT / "results" / "train_dueling_double_dqn" / "dueling_ddqn_final.pth",
    "PPO":          PROJECT_ROOT / "results" / "train_ppo" / "ppo_final.pth",
    "REINFORCE":    PROJECT_ROOT / "results" / "train_reinforce_normalization" / "reinforce_final.pth",
    "A2C-V":        PROJECT_ROOT / "results" / "train_A2C" / "v" / "seed_1" / "models" / "best_model.pt",
}

ENV_CONFIGS = {
    "Standard":    dict(gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5),
    "High-Wind":   dict(gravity=-10.0, enable_wind=True,  wind_power=20.0, turbulence_power=2.0),
    "Low-Gravity": dict(gravity=-5.0,  enable_wind=False, wind_power=15.0, turbulence_power=1.5),
}

AGENTS_ORDER = ["DQN", "Double DQN", "Dueling DDQN", "PPO", "REINFORCE", "A2C-V"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden=256):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, x):
        feat = self.feature(x)
        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# Model Loading 
def _load_state_dict(cls, key, **kwargs):
    model = cls(**kwargs).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATHS[key], map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def load_all_models():
    models = {}
    models["DQN"]          = _load_state_dict(QNetwork, "DQN")
    models["Double DQN"]   = _load_state_dict(QNetwork, "Double DQN")
    models["Dueling DDQN"] = _load_state_dict(DuelingQNetwork, "Dueling DDQN")

    ppo_actor = ActorNetwork().to(DEVICE)
    ckpt = torch.load(MODEL_PATHS["PPO"], map_location=DEVICE, weights_only=True)
    ppo_actor.load_state_dict(ckpt["actor"])
    ppo_actor.eval()
    models["PPO"] = ppo_actor

    models["REINFORCE"] = _load_state_dict(PolicyNetwork, "REINFORCE")

    a2c_ckpt = torch.load(MODEL_PATHS["A2C-V"], map_location=DEVICE, weights_only=False)
    actor_critic = a2c_ckpt["actor_critic"]
    actor_critic.to(DEVICE)
    actor_critic.eval()
    models["A2C-V"] = (actor_critic, a2c_ckpt.get("obs_rms"))

    return models


# Greedy Action Selection
def greedy_action(model, state):
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return int(model(state_t).argmax(dim=-1).item())


# Evaluation
def build_robust_args():
    args = get_config().parse_args([])
    args.noise_factor = "none"
    args.noise_type = "none"
    args.noise_sigma = 0.0
    args.noise_mu = 0.0
    args.noise_shift = 0.0
    return args


def evaluate_simple_agent(model, env_config, num_episodes=NUM_EPISODES):
    robust_args = build_robust_args()
    env = gym.make("LunarLander-v3", **env_config)
    rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset(seed=SEED + ep)
        total_reward = 0.0
        for _ in range(MAX_STEPS):
            action = greedy_action(model, state)
            robust_input = {"action": action, "robust_type": "action", "robust_config": robust_args}
            state, reward, terminated, truncated, _ = env.step(robust_input)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)

    env.close()
    return rewards


def evaluate_a2c_agent(actor_critic, obs_rms, env_config, num_episodes=NUM_EPISODES):
    """Evaluate A2C-V using VecEnv pipeline with observation normalization."""
    from a2c_v.envs import make_vec_envs
    from a2c_v import utils

    env_kwargs = {**env_config, "noise_factor": "none", "noise_type": "none",
                  "noise_mu": 0.0, "noise_sigma": 0.0, "noise_shift": 0.0}

    eval_envs = make_vec_envs(
        "RobustLunarLander-v3", SEED, 1, None,
        "/tmp/robustness_test_a2c", DEVICE, True, env_kwargs=env_kwargs,
    )

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    rewards = []
    obs = eval_envs.reset()
    rnn_hxs = torch.zeros(1, actor_critic.recurrent_hidden_state_size, device=DEVICE)
    masks = torch.zeros(1, 1, device=DEVICE)

    while len(rewards) < num_episodes:
        with torch.no_grad():
            _, action, _, rnn_hxs = actor_critic.act(obs, rnn_hxs, masks, deterministic=True)
        action = action.squeeze(1)
        obs, _, done, infos = eval_envs.step(action)
        masks = torch.tensor([[0.0] if done[0] else [1.0]], dtype=torch.float32, device=DEVICE)
        for info in infos:
            if "episode" in info:
                rewards.append(info["episode"]["r"])

    eval_envs.close()
    return rewards[:num_episodes]


# Main

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Evaluating {NUM_EPISODES} episodes × {len(AGENTS_ORDER)} agents × "
          f"{len(ENV_CONFIGS)} environments = {NUM_EPISODES * len(AGENTS_ORDER) * len(ENV_CONFIGS)} total episodes.\n")

    for name, path in MODEL_PATHS.items():
        if not path.exists():
            print(f"[ERROR] Model not found: {name} -> {path}")
            sys.exit(1)

    print("Loading models...")
    models = load_all_models()
    print(f"All {len(models)} models loaded.\n")

    all_results = {}

    for env_name, env_config in ENV_CONFIGS.items():
        print(f"{'='*70}")
        print(f"Environment: {env_name}  |  {env_config}")
        print(f"{'='*70}")

        for agent_name in AGENTS_ORDER:
            if agent_name == "A2C-V":
                actor_critic, obs_rms = models["A2C-V"]
                rewards = evaluate_a2c_agent(actor_critic, obs_rms, env_config)
            else:
                rewards = evaluate_simple_agent(models[agent_name], env_config)

            key = f"{agent_name}_{env_name}"
            all_results[key] = rewards
            mean_r, std_r = np.mean(rewards), np.std(rewards)
            print(f"  {agent_name:15s} | Mean: {mean_r:8.2f} | Std: {std_r:7.2f} | "
                  f"Min: {min(rewards):8.2f} | Max: {max(rewards):8.2f}")
        print()

    results_file = RESULTS_DIR / "robustness_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[INFO] Results saved to {results_file}")

    print_summary_table(all_results)


def print_summary_table(results):
    print(f"\n{'='*96}")
    print(f"{'Zero-Shot Robustness Test Summary':^96}")
    print(f"{'='*96}")
    header = f"{'Algorithm':<16}" + "".join(f"{'|':>2} {e:>22}" for e in ENV_CONFIGS)
    print(header)
    print("-" * 96)

    for agent in AGENTS_ORDER:
        row = f"{agent:<16}"
        for env in ENV_CONFIGS:
            key = f"{agent}_{env}"
            r = results[key]
            row += f"{'|':>2} {np.mean(r):8.1f} ± {np.std(r):6.1f}      "
        print(row)

    print(f"{'='*96}\n")


if __name__ == "__main__":
    main()
