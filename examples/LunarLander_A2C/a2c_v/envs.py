from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

# Some stable-baselines3 utilities still import `gym`, so alias it before those imports run.
sys.modules.setdefault("gym", gym)

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VecNormalize_


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import robust_gymnasium
from robust_gymnasium.configs.robust_setting import get_config


warnings.filterwarnings(
    "ignore",
    message=r".*Overriding environment .* already in registry.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"robust_gymnasium\.envs\.registration",
)


def build_robust_config() -> argparse.Namespace:
    robust_args = get_config().parse_args([])
    robust_args.noise_factor = "none"
    robust_args.noise_type = "none"
    robust_args.noise_mu = 0.0
    robust_args.noise_sigma = 0.0
    robust_args.noise_shift = 0.0
    return robust_args


class RobustLunarLanderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        noise_factor: str = "none",
        noise_type: str = "none",
        noise_mu: float = 0.0,
        noise_sigma: float = 0.0,
        noise_shift: float = 0.0,
    ):
        super().__init__()
        self._seed = None
        self.robust_config = build_robust_config()
        self.robust_config.noise_factor = noise_factor
        self.robust_config.noise_type = noise_type
        self.robust_config.noise_mu = noise_mu
        self.robust_config.noise_sigma = noise_sigma
        self.robust_config.noise_shift = noise_shift
        self.env = robust_gymnasium.make(
            "LunarLander-v3",
            gravity=gravity,
            enable_wind=enable_wind,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
        )

        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=np.asarray(obs_space.low, dtype=np.float32),
            high=np.asarray(obs_space.high, dtype=np.float32),
            shape=obs_space.shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(int(act_space.n))

    def seed(self, seed=None):
        self._seed = seed
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = self._seed
        obs, info = self.env.reset(seed=seed, options=options)
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        robust_input = {
            "action": int(action),
            "robust_type": "action",
            "robust_config": self.robust_config,
        }
        obs, reward, terminated, truncated, info = self.env.step(robust_input)
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def make_env(env_id, seed, rank, log_dir, allow_early_resets, env_kwargs=None):
    def _thunk():
        if env_id != "RobustLunarLander-v3":
            raise ValueError("Only RobustLunarLander-v3 is supported in this extracted package.")

        env = RobustLunarLanderEnv(**(env_kwargs or {}))
        env.reset(seed=seed + rank)

        if log_dir is not None:
            env = Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets,
            )

        return env

    return _thunk


def make_vec_envs(
    env_name,
    seed,
    num_processes,
    gamma,
    log_dir,
    device,
    allow_early_resets,
    env_kwargs=None,
):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, env_kwargs=env_kwargs)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, norm_reward=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)
    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super().__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)
            obs = np.clip(
                (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            )
            return obs
        return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
