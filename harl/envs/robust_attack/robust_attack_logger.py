"""Logger for the multi-attacker (robust_attack) environment."""

import numpy as np

from harl.common.base_logger import BaseLogger


class RobustAttackLogger(BaseLogger):
    """Logger that additionally tracks the victim's (undiscounted) return.

    The attacker reward is the negative victim reward, so monitoring the victim
    return directly shows how much the multi-attacker degrades the victim.
    """

    def get_task_name(self):
        return self.env_args["scenario"]

    def init(self, episodes):
        super().init(episodes)
        self.train_episode_victim_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_victim_rewards = []

    def per_step(self, data):
        super().per_step(data)
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
        n_threads = self.algo_args["train"]["n_rollout_threads"]
        for t in range(n_threads):
            victim_reward = infos[t][0].get("victim_reward", 0.0)
            self.train_episode_victim_rewards[t] += victim_reward
            if dones_env[t]:
                self.done_episodes_victim_rewards.append(
                    self.train_episode_victim_rewards[t]
                )
                self.train_episode_victim_rewards[t] = 0

    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        super().episode_log(
            actor_train_infos, critic_train_info, actor_buffer, critic_buffer
        )
        if len(self.done_episodes_victim_rewards) > 0:
            aver_victim = np.mean(self.done_episodes_victim_rewards)
            print(
                "Victim average episode return under attack is {}.\n".format(
                    aver_victim
                )
            )
            self.writter.add_scalars(
                "victim_episode_rewards",
                {"aver_rewards": aver_victim},
                self.total_num_steps,
            )
            self.wandb_log(
                {"attack/victim_episode_rewards": aver_victim},
                self.total_num_steps,
            )
            self.done_episodes_victim_rewards = []


class RobustVictimLogger(BaseLogger):
    """Logger for single-agent victim PPO training on robust_gymnasium."""

    def get_task_name(self):
        return self.env_args["scenario"]

