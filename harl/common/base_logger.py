"""Base logger."""

import time
import os
import numpy as np


class BaseLogger:
    """Base logger class.
    Used for logging information in the on-policy training pipeline.
    """

    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        """Initialize the logger."""
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.task_name = self.get_task_name()
        self.num_agents = num_agents
        self.writter = writter
        self.run_dir = run_dir
        self.log_file = open(
            os.path.join(run_dir, "progress.txt"), "w", encoding="utf-8"
        )
        self._init_wandb()

    def _init_wandb(self):
        """Initialise Weights & Biases logging from the ``logger`` config block.

        This is the single standard switch for both victim and attacker runs.
        Set ``logger.use_wandb: True`` (or pass ``--use_wandb True`` on the CLI)
        to enable it; leave it ``False`` to fall back to TensorBoard only.
        Recognised ``logger`` keys (all optional):
            use_wandb, wandb_project, wandb_entity, wandb_group,
            wandb_name, wandb_tags, wandb_mode (online/offline/disabled).
        """
        logger_cfg = self.algo_args.get("logger", {})
        self.use_wandb = bool(logger_cfg.get("use_wandb", False))
        self.wandb_run = None
        if not self.use_wandb:
            return
        try:
            import wandb
        except ImportError:
            print("[logger] use_wandb=True but the wandb package is not "
                  "installed; run `pip install wandb`. Disabling wandb.")
            self.use_wandb = False
            return
        self._wandb = wandb
        default_name = "{}-{}-{}".format(
            self.args["env"], self.args["algo"], self.args["exp_name"]
        )
        self.wandb_run = wandb.init(
            project=logger_cfg.get("wandb_project", "robust-gymnasium"),
            entity=logger_cfg.get("wandb_entity", None),
            group=logger_cfg.get("wandb_group", None),
            name=logger_cfg.get("wandb_name", None) or default_name,
            tags=logger_cfg.get("wandb_tags", None),
            mode=logger_cfg.get("wandb_mode", "online"),
            dir=self.run_dir,
            config={
                "args": self.args,
                "algo_args": self.algo_args,
                "env_args": self.env_args,
            },
        )

    def wandb_log(self, data, step):
        """Log a flat dict of scalars to wandb (no-op when wandb is disabled)."""
        if self.use_wandb and self.wandb_run is not None:
            self._wandb.log(data, step=int(step))

    def get_task_name(self):
        """Get the task name."""
        raise NotImplementedError

    def init(self, episodes):
        """Initialize the logger."""
        self.start = time.time()
        self.episodes = episodes
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []

    def episode_init(self, episode):
        """Initialize the logger for each episode."""
        self.episode = episode

    def per_step(self, data):
        """Process data per step."""
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        dones_env = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env
        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[t]:
                self.done_episodes_rewards.append(self.train_episode_rewards[t])
                self.train_episode_rewards[t] = 0

    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        """Log information for each episode."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()
        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)

        print(
            "Average step reward is {}.".format(
                critic_train_info["average_step_rewards"]
            )
        )

        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(self.done_episodes_rewards)
            print(
                "Some episodes done, average episode reward is {}.\n".format(
                    aver_episode_rewards
                )
            )
            self.wandb_log(
                {"train/average_episode_rewards": aver_episode_rewards},
                self.total_num_steps,
            )

            self.writter.add_scalars(
                "train_episode_rewards",
                {"aver_rewards": aver_episode_rewards},
                self.total_num_steps,
            )
            self.done_episodes_rewards = []

    def eval_init(self):
        """Initialize the logger for evaluation."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.eval_episode_rewards = []
        self.one_episode_rewards = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards.append([])
            self.eval_episode_rewards.append([])

    def eval_per_step(self, eval_data):
        """Log evaluation information per step."""
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = eval_data
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
        self.eval_infos = eval_infos

    def eval_thread_done(self, tid):
        """Log evaluation information."""
        self.eval_episode_rewards[tid].append(
            np.sum(self.one_episode_rewards[tid], axis=0)
        )
        self.one_episode_rewards[tid] = []

    def eval_log(self, eval_episode):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )

        # 取每个 episode 的第一个智能体的奖励值，并转换为整数
        if len(self.eval_episode_rewards) > 0:
            global_rewards = [int(ep_rewards[0]) for ep_rewards in self.eval_episode_rewards]  # 取第一个智能体的值并转换为整数
        else:
            global_rewards = []

        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("Esisodes reward",self.eval_episode_rewards)
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))

        # 将当前总步数 + 全局奖励写入日志
        rewards_str = ",".join(map(str, global_rewards))  # 转换成字符串
        self.log_file.write(f"{self.total_num_steps},{rewards_str}\n")  # 记录到日志
        self.log_file.flush()  # 立即写入文件

    def log_train(self, actor_train_infos, critic_train_info):
        """Log training information."""
        # log actor
        for agent_id in range(self.num_agents):
            for k, v in actor_train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, self.total_num_steps)
                self.wandb_log({agent_k: v}, self.total_num_steps)
        # log critic
        for k, v in critic_train_info.items():
            critic_k = "critic/" + k
            self.writter.add_scalars(critic_k, {critic_k: v}, self.total_num_steps)
            self.wandb_log({critic_k: v}, self.total_num_steps)

    def log_env(self, env_infos):
        """Log environment information."""
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, self.total_num_steps)
                self.wandb_log({k: np.mean(v)}, self.total_num_steps)

    def close(self):
        """Close the logger."""
        self.log_file.close()
        if self.use_wandb and self.wandb_run is not None:
            self._wandb.finish()