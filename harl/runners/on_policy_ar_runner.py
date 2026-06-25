"""Runner for AR-MAPPO (autoregressive / ordered-intervention MAPPO).

AR-MAPPO turns the standard *simultaneous* joint policy of MAPPO into an
*ordered* one that matches the sequential attack chain

    u_o (obs attack) -> victim acts on corrupted obs -> u_a (action attack).

Within each environment step the leader (observation attacker, agent 0) acts
first; the environment commits its perturbation and recomputes the victim
action; only then does the follower (action attacker, agent 1) act, conditioned
on the CURRENT-step victim action (not the one-step-stale value used by the
plain simultaneous rollout). Buffers, critic and training are identical to
MAPPO -- only the rollout ordering changes.
"""
import numpy as np
import torch

from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.utils.trans_tools import _t2n


class OnPolicyARRunner(OnPolicyMARunner):
    """AR-MAPPO runner: within-step sequential (leader -> victim -> follower)."""

    LEADER_ID = 0
    FOLLOWER_ID = 1

    def _get_actions_for(self, agent_id, step):
        """Query one actor from its buffered observation at ``step``."""
        action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(
            self.actor_buffer[agent_id].obs[step],
            self.actor_buffer[agent_id].rnn_states[step],
            self.actor_buffer[agent_id].masks[step],
            self.actor_buffer[agent_id].available_actions[step]
            if self.actor_buffer[agent_id].available_actions is not None
            else None,
        )
        return _t2n(action), _t2n(action_log_prob), _t2n(rnn_state)

    @torch.no_grad()
    def collect(self, step):
        """Two-phase sequential collection (leader, then follower)."""
        assert self.num_agents == 2, "AR-MAPPO assumes exactly two ordered agents."
        leader = self.LEADER_ID
        follower = self.FOLLOWER_ID

        action_collector = [None, None]
        action_log_prob_collector = [None, None]
        rnn_state_collector = [None, None]

        # --- phase 1: leader (observation attacker) acts --------------------
        (
            action_collector[leader],
            action_log_prob_collector[leader],
            rnn_state_collector[leader],
        ) = self._get_actions_for(leader, step)

        # --- commit leader attack; fetch follower obs w/ current victim act --
        # begin_step applies the leader's perturbation, queries the victim, and
        # returns per-agent observations whose follower view is the CURRENT-step
        # victim action. We overwrite ONLY the follower's actor observation so it
        # acts on -- and is trained on -- exactly this observation.
        #
        # IMPORTANT: we deliberately do NOT touch the centralized critic's
        # share_obs. Injecting the current-step victim action (a function of the
        # leader's own delta_o) into the critic would turn the shared EP baseline
        # into V(s, a^v(delta_o)) ~= Q_o(s, delta_o), which cancels the leader's
        # own advantage (A_leader = R - Q_o ~= 0) and starves the obs-attacker of
        # gradient. The baseline must stay a clean state-value V(s) independent of
        # the current delta_o, so we keep the standard buffered share_obs[step].
        obs, _ = self.envs.begin_step(action_collector[leader])
        self.actor_buffer[follower].obs[step] = obs[:, follower].copy()

        # --- phase 2: follower (action attacker) acts -----------------------
        (
            action_collector[follower],
            action_log_prob_collector[follower],
            rnn_state_collector[follower],
        ) = self._get_actions_for(follower, step)

        # (n_agents, n_threads, dim) -> (n_threads, n_agents, dim)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)

        # --- critic values on the (updated) share_obs -----------------------
        if self.state_type == "EP":
            value, rnn_state_critic = self.critic.get_values(
                self.critic_buffer.share_obs[step],
                self.critic_buffer.rnn_states_critic[step],
                self.critic_buffer.masks[step],
            )
            values = _t2n(value)
            rnn_states_critic = _t2n(rnn_state_critic)
        elif self.state_type == "FP":
            value, rnn_state_critic = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[step]),
                np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                np.concatenate(self.critic_buffer.masks[step]),
            )
            values = np.array(
                np.split(_t2n(value), self.algo_args["train"]["n_rollout_threads"])
            )
            rnn_states_critic = np.array(
                np.split(
                    _t2n(rnn_state_critic),
                    self.algo_args["train"]["n_rollout_threads"],
                )
            )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic
