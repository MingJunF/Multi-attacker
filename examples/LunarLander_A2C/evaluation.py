from __future__ import annotations

import numpy as np
import torch

from a2c_v import utils
from a2c_v.envs import make_vec_envs


def evaluate(
    actor_critic,
    obs_rms,
    env_name,
    seed,
    num_processes,
    eval_log_dir,
    device,
    num_episodes=10,
    env_kwargs=None,
):
    eval_envs = make_vec_envs(
        env_name,
        seed + num_processes,
        num_processes,
        None,
        eval_log_dir,
        device,
        True,
        env_kwargs=env_kwargs,
    )

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes,
        actor_critic.recurrent_hidden_state_size,
        device=device,
    )
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True,
            )

        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )

        for info in infos:
            if "episode" in info:
                eval_episode_rewards.append(info["episode"]["r"])

    eval_envs.close()
    eval_episode_rewards = eval_episode_rewards[:num_episodes]
    mean_reward = float(np.mean(eval_episode_rewards))
    std_reward = float(np.std(eval_episode_rewards))

    print(
        " Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards),
            mean_reward,
        )
    )
    return {
        "num_episodes": len(eval_episode_rewards),
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "rewards": [float(reward) for reward in eval_episode_rewards],
    }
