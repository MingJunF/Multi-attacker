"""Runner for Independent PPO (IPPO).

Unlike MAPPO (single centralized critic on the shared/global state) and HAPPO
(per-agent actors but still a single shared critic), IPPO gives **every agent its
own independent actor and its own independent critic**. Each critic is trained
purely on that agent's local observation, so the agents learn fully decentralized
value functions.

This runner reuses all of the base on-policy machinery (env setup, actor buffers,
rollout/eval loops). It only replaces the single ``self.critic`` /
``self.critic_buffer`` with per-agent lists and overrides the few methods that
touch the critic. The existing MAPPO / HAPPO runners are left untouched.
"""
import numpy as np
import torch

from harl.common.valuenorm import ValueNorm
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.algorithms.critics.v_critic import VCritic
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner
from harl.utils.trans_tools import _t2n


class OnPolicyIPPORunner(OnPolicyBaseRunner):
    """Runner for Independent PPO (per-agent actor + per-agent critic)."""

    def __init__(self, args, algo_args, env_args):
        # Build everything via the base runner first (env, actors, a single
        # placeholder critic, logger, ...). We then discard the shared critic and
        # replace it with one independent critic per agent.
        super().__init__(args, algo_args, env_args)

        # Sequential (AR) rollout: when running the robust_attack env with the
        # two ordered attackers, query the leader (observation attacker) first,
        # commit its perturbation, then query the follower (action attacker) on
        # the CURRENT-step victim action -- exactly the env's intended ordering.
        # Guarded so IPPO stays the standard simultaneous algorithm everywhere
        # else (the env exposes ``begin_step`` only for robust_attack).
        self._ar_rollout = (
            args["env"] == "robust_attack" and self.num_agents == 2
        )

        if self.algo_args["render"]["use_render"]:
            return

        # Each agent's critic consumes ONLY its own local observation (truly
        # decentralized IPPO). The env exposes a global 80-d ``share_obs`` for
        # MAPPO's centralized critic, but IPPO deliberately ignores it and feeds
        # each critic the agent's own local ``observation_space`` / ``obs`` so the
        # value functions stay fully decentralized.
        self.critic = []
        self.critic_buffer = []
        self.value_normalizer = []
        for agent_id in range(self.num_agents):
            critic = VCritic(
                {**self.algo_args["model"], **self.algo_args["algo"]},
                self.envs.observation_space[agent_id],
                device=self.device,
            )
            self.critic.append(critic)
            self.critic_buffer.append(
                OnPolicyCriticBufferEP(
                    {
                        **self.algo_args["train"],
                        **self.algo_args["model"],
                        **self.algo_args["algo"],
                    },
                    self.envs.observation_space[agent_id],
                )
            )
            if self.algo_args["train"]["use_valuenorm"] is True:
                self.value_normalizer.append(ValueNorm(1, device=self.device))
            else:
                self.value_normalizer.append(None)

        # Re-restore now that the per-agent critics exist (base __init__ already
        # ran restore() against the placeholder critic, which our override turned
        # into a no-op because the per-agent list did not exist yet).
        if self.algo_args["train"]["model_dir"] is not None:
            self.restore()

    def run(self):
        """Run the IPPO training pipeline (mirrors the base loop, but decays and
        logs every per-agent critic)."""
        if self.algo_args["render"]["use_render"] is True:
            self.render()
            return
        print("start running")
        self.warmup()

        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        self.logger.init(episodes)

        for episode in range(1, episodes + 1):
            if self.algo_args["train"]["use_linear_lr_decay"]:
                if self.share_param:
                    self.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].lr_decay(episode, episodes)
                for agent_id in range(self.num_agents):
                    self.critic[agent_id].lr_decay(episode, episodes)

            self.logger.episode_init(episode)

            self.prep_rollout()
            for step in range(self.algo_args["train"]["episode_length"]):
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                data = (
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
                )

                self.logger.per_step(data)
                self.insert(data)

            self.compute()
            self.prep_training()

            actor_train_infos, critic_train_info = self.train()

            if episode % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer[0],
                )

            if episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()
                self.save()

            self.after_update()

    def warmup(self):
        """Warm up actor buffers and every per-agent critic buffer."""
        obs, share_obs, available_actions = self.envs.reset()
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[
                    :, agent_id
                ].copy()
            # IPPO critic input = agent's own LOCAL observation (not the global
            # share_obs), keeping the value function decentralized.
            self.critic_buffer[agent_id].share_obs[0] = obs[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        """Collect actions from actors and values from per-agent critics."""
        if getattr(self, "_ar_rollout", False):
            return self._collect_sequential(step)
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        value_collector = []
        rnn_state_critic_collector = []
        for agent_id in range(self.num_agents):
            action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(
                self.actor_buffer[agent_id].obs[step],
                self.actor_buffer[agent_id].rnn_states[step],
                self.actor_buffer[agent_id].masks[step],
                self.actor_buffer[agent_id].available_actions[step]
                if self.actor_buffer[agent_id].available_actions is not None
                else None,
            )
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))

            value, rnn_state_critic = self.critic[agent_id].get_values(
                self.critic_buffer[agent_id].share_obs[step],
                self.critic_buffer[agent_id].rnn_states_critic[step],
                self.critic_buffer[agent_id].masks[step],
            )
            value_collector.append(_t2n(value))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))

        # (n_agents, n_threads, dim) -> (n_threads, n_agents, dim)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        values = np.array(value_collector).transpose(1, 0, 2)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    @torch.no_grad()
    def _collect_sequential(self, step):
        """Two-phase (leader -> victim -> follower) collection for robust_attack.

        The observation attacker (agent 0, leader) acts first; the env commits
        its perturbation and recomputes the victim action; only then does the
        action attacker (agent 1, follower) act -- so the follower observes the
        CURRENT-step victim action it is about to perturb (not the one-step-stale
        value seen by the plain simultaneous rollout).

        Unlike MAPPO's single shared critic, IPPO gives every agent its OWN
        critic, so we also feed the follower's independent critic the current
        victim action: a per-agent follower baseline that depends on delta_o is
        still unbiased because delta_o is the LEADER's action, not the
        follower's own. Training, buffers and everything else are unchanged.
        """
        leader, follower = 0, 1
        actions_l = [None, None]
        alps_l = [None, None]
        rnns_l = [None, None]
        values_l = [None, None]
        rnns_c_l = [None, None]

        def query(agent_id):
            action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(
                self.actor_buffer[agent_id].obs[step],
                self.actor_buffer[agent_id].rnn_states[step],
                self.actor_buffer[agent_id].masks[step],
                self.actor_buffer[agent_id].available_actions[step]
                if self.actor_buffer[agent_id].available_actions is not None
                else None,
            )
            value, rnn_state_critic = self.critic[agent_id].get_values(
                self.critic_buffer[agent_id].share_obs[step],
                self.critic_buffer[agent_id].rnn_states_critic[step],
                self.critic_buffer[agent_id].masks[step],
            )
            actions_l[agent_id] = _t2n(action)
            alps_l[agent_id] = _t2n(action_log_prob)
            rnns_l[agent_id] = _t2n(rnn_state)
            values_l[agent_id] = _t2n(value)
            rnns_c_l[agent_id] = _t2n(rnn_state_critic)

        # --- phase 1: leader (observation attacker) acts --------------------
        query(leader)

        # --- commit leader attack; fetch follower obs w/ current victim act --
        obs, _ = self.envs.begin_step(actions_l[leader])
        # Follower observes -- and its independent critic is evaluated on -- the
        # current victim action. Keep actor/critic information sets identical.
        self.actor_buffer[follower].obs[step] = obs[:, follower].copy()
        self.critic_buffer[follower].share_obs[step] = obs[:, follower].copy()

        # --- phase 2: follower (action attacker) acts -----------------------
        query(follower)

        actions = np.array(actions_l).transpose(1, 0, 2)
        action_log_probs = np.array(alps_l).transpose(1, 0, 2)
        rnn_states = np.array(rnns_l).transpose(1, 0, 2, 3)
        values = np.array(values_l).transpose(1, 0, 2)
        rnn_states_critic = np.array(rnns_c_l).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        """Insert collected data into actor and per-agent critic buffers."""
        (
            obs,  # (n_threads, n_agents, obs_dim)
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # list, len n_threads
            available_actions,  # (n_threads, ) of None or (n_threads, n_agents, action_number)
            values,  # (n_threads, n_agents, 1)
            actions,  # (n_threads, n_agents, action_dim)
            action_log_probs,  # (n_threads, n_agents, action_dim)
            rnn_states,  # (n_threads, n_agents, recurrent_n, hidden)
            rnn_states_critic,  # (n_threads, n_agents, recurrent_n, hidden)
        ) = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        rnn_states_critic[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        active_masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32
        )
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # Per-agent bad_masks (truncation vs termination), read from each agent's info.
        bad_masks = np.array(
            [
                [
                    [0.0]
                    if "bad_transition" in info[agent_id].keys()
                    and info[agent_id]["bad_transition"] == True
                    else [1.0]
                    for agent_id in range(self.num_agents)
                ]
                for info in infos
            ]
        )

        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )
            self.critic_buffer[agent_id].insert(
                obs[:, agent_id],
                rnn_states_critic[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
                bad_masks[:, agent_id],
            )

    @torch.no_grad()
    def compute(self):
        """Compute returns/advantages independently for every agent's critic."""
        for agent_id in range(self.num_agents):
            next_value, _ = self.critic[agent_id].get_values(
                self.critic_buffer[agent_id].share_obs[-1],
                self.critic_buffer[agent_id].rnn_states_critic[-1],
                self.critic_buffer[agent_id].masks[-1],
            )
            next_value = _t2n(next_value)
            self.critic_buffer[agent_id].compute_returns(
                next_value, self.value_normalizer[agent_id]
            )

    def train(self):
        """Independent PPO update: each agent trains its own actor and critic."""
        actor_train_infos = []
        critic_train_infos = []

        for agent_id in range(self.num_agents):
            value_normalizer = self.value_normalizer[agent_id]
            critic_buffer = self.critic_buffer[agent_id]
            if value_normalizer is not None:
                advantages = critic_buffer.returns[
                    :-1
                ] - value_normalizer.denormalize(critic_buffer.value_preds[:-1])
            else:
                advantages = (
                    critic_buffer.returns[:-1] - critic_buffer.value_preds[:-1]
                )

            actor_train_info = self.actor[agent_id].train(
                self.actor_buffer[agent_id], advantages.copy(), "EP"
            )
            critic_train_info = self.critic[agent_id].train(
                critic_buffer, value_normalizer
            )

            actor_train_infos.append(actor_train_info)
            critic_train_infos.append(critic_train_info)

        # Aggregate per-agent critic infos into one dict for logging. Some values
        # (e.g. grad norms) may be torch tensors on GPU, so move to CPU floats.
        def _to_float(v):
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().item()
            return float(v)

        critic_train_info = {
            key: float(np.mean([_to_float(info[key]) for info in critic_train_infos]))
            for key in critic_train_infos[0]
        }

        return actor_train_infos, critic_train_info

    def after_update(self):
        """Copy the last-step data to the first position of every buffer."""
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
            self.critic_buffer[agent_id].after_update()

    def prep_rollout(self):
        """Set actors and per-agent critics to eval mode."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_rollout()
            self.critic[agent_id].prep_rollout()

    def prep_training(self):
        """Set actors and per-agent critics to train mode."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_training()
            self.critic[agent_id].prep_training()

    def save(self):
        """Save each agent's actor, critic, and value normalizer."""
        for agent_id in range(self.num_agents):
            policy_actor = self.actor[agent_id].actor
            torch.save(
                policy_actor.state_dict(),
                str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt",
            )
            policy_critic = self.critic[agent_id].critic
            torch.save(
                policy_critic.state_dict(),
                str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt",
            )
            if self.value_normalizer[agent_id] is not None:
                torch.save(
                    self.value_normalizer[agent_id].state_dict(),
                    str(self.save_dir) + "/value_normalizer_agent" + str(agent_id) + ".pt",
                )

    def restore(self):
        """Restore each agent's actor, critic, and value normalizer."""
        # During base __init__ the per-agent critics do not yet exist; fall back
        # to the base (single-critic) behaviour so that early call is harmless.
        if not isinstance(self.critic, list):
            super().restore()
            return

        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(
                str(self.algo_args["train"]["model_dir"])
                + "/actor_agent"
                + str(agent_id)
                + ".pt"
            )
            self.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
            if not self.algo_args["render"]["use_render"]:
                policy_critic_state_dict = torch.load(
                    str(self.algo_args["train"]["model_dir"])
                    + "/critic_agent"
                    + str(agent_id)
                    + ".pt"
                )
                self.critic[agent_id].critic.load_state_dict(policy_critic_state_dict)
                if self.value_normalizer[agent_id] is not None:
                    value_normalizer_state_dict = torch.load(
                        str(self.algo_args["train"]["model_dir"])
                        + "/value_normalizer_agent"
                        + str(agent_id)
                        + ".pt"
                    )
                    self.value_normalizer[agent_id].load_state_dict(
                        value_normalizer_state_dict
                    )
