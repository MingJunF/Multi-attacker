#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from a2c_v.algo import A2C
from a2c_v.arguments import get_args
from a2c_v.envs import make_vec_envs
from a2c_v.experiment_logger import ExperimentLogger
from a2c_v.model import Policy
from a2c_v.storage import RolloutStorage
from a2c_v import utils
from evaluation import evaluate


def save_checkpoint(path: Path, actor_critic, obs_rms, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_critic": actor_critic,
            "obs_rms": obs_rms,
            "metadata": metadata,
        },
        path,
    )


def main() -> None:
    args = get_args()
    run_name = f"{args.critic_type}_seed_{args.seed}"
    results_dir = Path(os.path.expanduser(args.results_dir)) / args.critic_type / f"seed_{args.seed}"
    model_dir = results_dir / "models"
    best_model_path = model_dir / "best_model.pt"
    last_model_path = model_dir / "last_model.pt"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_root = os.path.expanduser(args.log_dir)
    log_dir = os.path.join(log_root, run_name)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    logger = ExperimentLogger(
        results_dir,
        {
            **vars(args),
            "device": str(device),
            "run_name": run_name,
        },
    )

    envs = make_vec_envs(
        args.env_name,
        args.seed,
        args.num_processes,
        args.gamma,
        log_dir,
        device,
        False,
    )

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={"recurrent": args.recurrent_policy},
        critic_type=args.critic_type,
    )
    actor_critic.to(device)

    agent = A2C(
        actor_critic,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        alpha=args.alpha,
        max_grad_norm=args.max_grad_norm,
    )

    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    best_eval_mean = float("-inf")
    best_eval_metadata = None

    for j in range(num_updates):
        if args.use_linear_lr_decay:
            utils.update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            with torch.no_grad():
                policy_value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )

            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if "episode" in info:
                    episode_rewards.append(info["episode"]["r"])

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info else [1.0] for info in infos]
            )

            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                policy_value,
                reward,
                masks,
                bad_masks,
            )

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value,
            args.use_gae,
            args.gamma,
            args.gae_lambda,
            args.use_proper_time_limits,
        )

        update_metrics = agent.update(rollouts)
        rollouts.after_update()
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        logger.append_training_metric(
            {
                "critic_type": args.critic_type,
                "seed": args.seed,
                "update": j,
                "timesteps": total_num_steps,
                "learning_rate": agent.optimizer.param_groups[0]["lr"],
                **update_metrics,
            }
        )

        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, f"a2c_{args.critic_type}")
            os.makedirs(save_path, exist_ok=True)
            torch.save(
                [
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), "obs_rms", None),
                ],
                os.path.join(save_path, f"{args.env_name}_seed{args.seed}.pt"),
            )

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n"
                " Last {} training episodes: mean/median reward {:.1f}/{:.1f},"
                " min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                )
            )
            print(
                " critic_type {}, critic_loss {:.5f}, action_loss {:.5f}, dist_entropy {:.5f}\n".format(
                    args.critic_type,
                    update_metrics["critic_loss"],
                    update_metrics["action_loss"],
                    update_metrics["dist_entropy"],
                )
            )

        if args.eval_interval is not None and j % args.eval_interval == 0:
            vec_norm = utils.get_vec_normalize(envs)
            obs_rms = getattr(vec_norm, "obs_rms", None)
            eval_result = evaluate(
                actor_critic,
                obs_rms,
                args.env_name,
                args.seed,
                args.num_processes,
                eval_log_dir,
                device,
                num_episodes=args.eval_episodes,
            )
            logger.append_evaluation(
                {
                    "critic_type": args.critic_type,
                    "seed": args.seed,
                    "update": j,
                    "timesteps": total_num_steps,
                    "stage": "checkpoint",
                    "num_episodes": eval_result["num_episodes"],
                    "mean_reward": eval_result["mean_reward"],
                    "std_reward": eval_result["std_reward"],
                }
            )
            if eval_result["mean_reward"] > best_eval_mean:
                best_eval_mean = eval_result["mean_reward"]
                best_eval_metadata = {
                    "critic_type": args.critic_type,
                    "seed": args.seed,
                    "stage": "checkpoint",
                    "update": j,
                    "timesteps": total_num_steps,
                    "mean_reward": eval_result["mean_reward"],
                    "std_reward": eval_result["std_reward"],
                    "num_episodes": eval_result["num_episodes"],
                }
                save_checkpoint(
                    best_model_path,
                    actor_critic,
                    obs_rms,
                    best_eval_metadata,
                )

    vec_norm = utils.get_vec_normalize(envs)
    obs_rms = getattr(vec_norm, "obs_rms", None)
    final_eval_result = evaluate(
        actor_critic,
        obs_rms,
        args.env_name,
        args.seed,
        args.num_processes,
        eval_log_dir,
        device,
        num_episodes=args.final_eval_episodes,
    )
    final_stage = f"final_{args.final_eval_episodes}"
    logger.append_evaluation(
        {
            "critic_type": args.critic_type,
            "seed": args.seed,
            "update": num_updates - 1,
            "timesteps": num_updates * args.num_processes * args.num_steps,
            "stage": final_stage,
            "num_episodes": final_eval_result["num_episodes"],
            "mean_reward": final_eval_result["mean_reward"],
            "std_reward": final_eval_result["std_reward"],
        }
    )
    logger.save_final_eval_rewards(
        {
            "critic_type": args.critic_type,
            "seed": args.seed,
            "timesteps": num_updates * args.num_processes * args.num_steps,
            "stage": final_stage,
            **final_eval_result,
        }
    )
    final_checkpoint_metadata = {
        "critic_type": args.critic_type,
        "seed": args.seed,
        "stage": final_stage,
        "update": num_updates - 1,
        "timesteps": num_updates * args.num_processes * args.num_steps,
        "mean_reward": final_eval_result["mean_reward"],
        "std_reward": final_eval_result["std_reward"],
        "num_episodes": final_eval_result["num_episodes"],
    }
    save_checkpoint(
        last_model_path,
        actor_critic,
        obs_rms,
        final_checkpoint_metadata,
    )
    if final_eval_result["mean_reward"] > best_eval_mean:
        best_eval_mean = final_eval_result["mean_reward"]
        best_eval_metadata = final_checkpoint_metadata
        save_checkpoint(
            best_model_path,
            actor_critic,
            obs_rms,
            best_eval_metadata,
        )
    logger.save_summary(
        {
            "critic_type": args.critic_type,
            "seed": args.seed,
            "total_timesteps": num_updates * args.num_processes * args.num_steps,
            "final_mean_reward": final_eval_result["mean_reward"],
            "final_std_reward": final_eval_result["std_reward"],
            "best_mean_reward": best_eval_mean,
            "best_model_path": str(best_model_path),
            "last_model_path": str(last_model_path),
            "best_model_stage": None if best_eval_metadata is None else best_eval_metadata["stage"],
        }
    )
    envs.close()


if __name__ == "__main__":
    main()
