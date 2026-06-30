"""Runner for off-policy Stage-Aware MADDPG (Stage-MADDPG).

Off-policy analogue of ``OnPolicyStageRunner``. Two ordered attackers share a
single centralized continuous Q critic with a coupled two-stage Bellman target
(see ``StageQCritic``). The rollout is sequential, identical to Stage-MAPPO:
the leader (obs attacker) acts; ``begin_step`` commits its perturbation and
recomputes the victim action; the follower (act attacker) then acts on the
CURRENT-step victim action. The env's causal FP centralized state encodes the
stage (x^o = [s, 0] for the leader, x^a = [s, victim.act] for the follower).

Launch with the robust_attack env set to ``--state_type FP
--causal_critic_state True`` (so one critic produces a per-agent stage state and
the leader's victim-action slot is masked).
"""
import copy

import numpy as np
import torch

from harl.runners.off_policy_ma_runner import OffPolicyMARunner
from harl.utils.trans_tools import _t2n


class OffPolicyStageRunner(OffPolicyMARunner):
    """Stage-Aware MADDPG runner: sequential rollout + coupled stage Q critic."""

    LEADER_ID = 0
    FOLLOWER_ID = 1

    def __init__(self, args, algo_args, env_args):
        super().__init__(args, algo_args, env_args)
        assert self.state_type == "FP", (
            "Stage-MADDPG requires state_type: FP (one shared critic producing a "
            "per-agent stage value). Pass --state_type FP."
        )
        assert self.num_agents == 2, "Stage-MADDPG assumes two ordered agents."
        assert env_args.get("causal_critic_state", False), (
            "Stage-MADDPG needs the stage one-hot / masked leader state in the "
            "critic input; pass --causal_critic_state True."
        )
        # victim-return tracking (the attacker reward is its negative)
        self.train_episode_victim_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_victim_rewards = []
        self._init_wandb()

    # ------------------------------------------------------------------ wandb
    def _init_wandb(self):
        """Initialise Weights & Biases logging from the ``logger`` config block.

        Mirrors ``BaseLogger._init_wandb`` (the off-policy base runner has no
        logger object). Logs the victim's average episode return under attack as
        ``attack/victim_episode_rewards`` so Stage-MADDPG curves are directly
        comparable with the on-policy attackers.
        """
        logger_cfg = self.algo_args.get("logger", {})
        self.use_wandb = bool(logger_cfg.get("use_wandb", False))
        self.wandb_run = None
        if not self.use_wandb:
            return
        try:
            import wandb
        except ImportError:
            print(
                "[logger] use_wandb=True but the wandb package is not installed; "
                "run `pip install wandb`. Disabling wandb."
            )
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
        """Log a flat dict of scalars to wandb (no-op when disabled)."""
        if self.use_wandb and self.wandb_run is not None:
            self._wandb.log(data, step=int(step))

    # ---------------------------------------------------------------- rollout
    @torch.no_grad()
    def get_actions(self, obs, available_actions=None, add_random=True):
        """Two-phase sequential action selection on the training envs.

        Phase 1: the leader acts on its observation. ``begin_step`` commits the
        obs attack and returns per-agent observations whose follower view holds
        the CURRENT-step victim action. Phase 2: the follower acts on that
        updated observation. ``self.envs.step`` then reuses the committed obs
        attack (it does not re-apply it).
        """
        leader = self.LEADER_ID
        follower = self.FOLLOWER_ID
        leader_act = _t2n(
            self.actor[leader].get_actions(obs[:, leader], add_random)
        )  # (n_threads, dim)
        new_obs, _ = self.envs.begin_step(leader_act)
        follower_act = _t2n(
            self.actor[follower].get_actions(new_obs[:, follower], add_random)
        )  # (n_threads, dim)
        # (n_agents, n_threads, dim) -> (n_threads, n_agents, dim)
        actions = np.stack([leader_act, follower_act], axis=0).transpose(1, 0, 2)
        return actions

    def _track_victim(self, infos, dones):
        """Accumulate per-thread victim episode returns from step infos."""
        n_threads = self.algo_args["train"]["n_rollout_threads"]
        dones_env = np.all(dones, axis=1)
        for t in range(n_threads):
            self.train_episode_victim_rewards[t] += infos[t][0].get(
                "victim_reward", 0.0
            )
            if dones_env[t]:
                self.done_episodes_victim_rewards.append(
                    self.train_episode_victim_rewards[t]
                )
                self.train_episode_victim_rewards[t] = 0.0

    def run(self):
        """Training pipeline with the two-phase stage rollout + victim logging."""
        if self.algo_args["render"]["use_render"]:
            self.render()
            return
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []
        print("start warmup")
        obs, share_obs, available_actions = self.warmup()
        print("finish warmup, start training")
        steps = (
            self.algo_args["train"]["num_env_steps"]
            // self.algo_args["train"]["n_rollout_threads"]
        )
        update_num = int(
            self.algo_args["train"]["update_per_train"]
            * self.algo_args["train"]["train_interval"]
        )
        for step in range(1, steps + 1):
            actions = self.get_actions(
                obs, available_actions=available_actions, add_random=True
            )
            (
                new_obs,
                new_share_obs,
                rewards,
                dones,
                infos,
                new_available_actions,
            ) = self.envs.step(actions)
            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            next_available_actions = new_available_actions.copy()
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                next_available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
            )
            self._track_victim(infos, dones)
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions
            if step % self.algo_args["train"]["train_interval"] == 0:
                if self.algo_args["train"]["use_linear_lr_decay"]:
                    if self.share_param:
                        self.actor[0].lr_decay(step, steps)
                    else:
                        for agent_id in range(self.num_agents):
                            self.actor[agent_id].lr_decay(step, steps)
                    self.critic.lr_decay(step, steps)
                for _ in range(update_num):
                    self.train()
            if step % self.algo_args["train"]["eval_interval"] == 0:
                cur_step = (
                    self.algo_args["train"]["warmup_steps"]
                    + step * self.algo_args["train"]["n_rollout_threads"]
                )
                print(
                    f"Env {self.args['env']} Task {self.task_name} Algo {self.args['algo']} "
                    f"Exp {self.args['exp_name']} Step {cur_step} / "
                    f"{self.algo_args['train']['num_env_steps']}, average step reward "
                    f"in buffer: {self.buffer.get_mean_rewards()}.\n"
                )
                log_data = {}
                if len(self.done_episodes_rewards) > 0:
                    aver_attacker = np.mean(self.done_episodes_rewards)
                    print(
                        "Some episodes done, average attacker episode reward is "
                        "{}.\n".format(aver_attacker)
                    )
                    self.log_file.write(
                        ",".join(map(str, [cur_step, aver_attacker])) + "\n"
                    )
                    self.log_file.flush()
                    log_data["attack/attacker_episode_rewards"] = aver_attacker
                    self.done_episodes_rewards = []
                if len(self.done_episodes_victim_rewards) > 0:
                    aver_victim = np.mean(self.done_episodes_victim_rewards)
                    print(
                        "Victim average episode return under attack is {}.\n".format(
                            aver_victim
                        )
                    )
                    log_data["attack/victim_episode_rewards"] = aver_victim
                    self.done_episodes_victim_rewards = []
                if log_data:
                    self.wandb_log(log_data, cur_step)
                self.save()

    # ------------------------------------------------------------------ train
    def train(self):
        """Stage-aware off-policy update: coupled stage critic + per-stage PG."""
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,  # (n_agents * batch, dim)
            sp_obs,  # (n_agents, batch, dim)
            sp_actions,  # (n_agents, batch, dim)
            sp_available_actions,
            sp_reward,  # (n_agents * batch, 1)
            sp_done,  # (n_agents * batch, 1)
            sp_valid_transition,
            sp_term,  # (n_agents * batch, 1)
            sp_next_share_obs,  # (n_agents * batch, dim)
            sp_next_obs,  # (n_agents, batch, dim)
            sp_next_available_actions,
            sp_gamma,  # (n_agents * batch, 1)
        ) = data

        leader = self.LEADER_ID
        follower = self.FOLLOWER_ID
        batch = sp_actions.shape[1]

        # --- critic update (coupled two-stage target) -----------------------
        self.critic.turn_on_grad()
        next_actions = []
        for agent_id in range(self.num_agents):
            next_actions.append(
                self.actor[agent_id].get_target_actions(sp_next_obs[agent_id])
            )
        self.critic.train(
            sp_share_obs,
            sp_actions,
            sp_reward,
            sp_done,
            sp_term,
            sp_next_share_obs,
            next_actions,
            sp_gamma,
        )
        self.critic.turn_off_grad()

        if self.total_it % self.policy_freq == 0:
            # FP rows are agent-major: leader rows (x^o) first, follower (x^a).
            x_o = sp_share_obs[:batch]
            x_a = sp_share_obs[batch : 2 * batch]
            # behaviour obs attack, held fixed for the follower's gradient
            delta_o_buf = torch.tensor(sp_actions[leader]).to(self.device)

            # --- follower (action attacker): grad through Q^a wrt delta_a ----
            self.actor[follower].turn_on_grad()
            delta_a_pol = self.actor[follower].get_actions(sp_obs[follower], False)
            joint_a = torch.cat([delta_o_buf, delta_a_pol], dim=-1)
            q_a = self.critic.get_values(x_a, joint_a)
            follower_loss = -torch.mean(q_a)
            self.actor[follower].actor_optimizer.zero_grad()
            follower_loss.backward()
            self.actor[follower].actor_optimizer.step()
            self.actor[follower].turn_off_grad()

            # --- leader (obs attacker): grad through Q^o wrt delta_o ---------
            # obs stage masks delta_a -> 0 (leakage-free), so the gradient flows
            # only through delta_o.
            self.actor[leader].turn_on_grad()
            delta_o_pol = self.actor[leader].get_actions(sp_obs[leader], False)
            joint_o = torch.cat([delta_o_pol, torch.zeros_like(delta_o_pol)], dim=-1)
            q_o = self.critic.get_values(x_o, joint_o)
            leader_loss = -torch.mean(q_o)
            self.actor[leader].actor_optimizer.zero_grad()
            leader_loss.backward()
            self.actor[leader].actor_optimizer.step()
            self.actor[leader].turn_off_grad()

            # --- soft updates -----------------------------------------------
            for agent_id in range(self.num_agents):
                self.actor[agent_id].soft_update()
            self.critic.soft_update()
