"""Runner for Independent DDPG (IDDPG).

The off-policy analogue of IPPO. Unlike MADDPG (a single centralized critic
Q(share_obs, joint_action) shared across agents), IDDPG gives **every agent its
own independent critic** Q_i(o_i, a_i) trained purely on that agent's LOCAL
observation and its OWN action. Each actor is optimized to maximize only its own
Q_i, so the agents learn fully decentralized, non-cooperative value functions --
no other agent's action ever enters an agent's critic.

This runner reuses all of the base off-policy machinery (env setup, replay
buffer, warmup/rollout/eval loops). It only replaces the single centralized
``self.critic`` with a list of per-agent decentralized critics and overrides
``train`` accordingly. The existing MADDPG / stage_maddpg runners are untouched.
"""
import copy

import numpy as np
import torch

from harl.algorithms.critics.continuous_q_critic import ContinuousQCritic
from harl.runners.off_policy_ma_runner import OffPolicyMARunner


class _IndependentCritics:
    """A thin container over per-agent decentralized critics that exposes the
    same ``lr_decay`` / ``save`` / ``restore`` interface the base runner expects
    from a single centralized critic, so the base off-policy loop stays
    unchanged. Individual critics are reached via indexing (``self.critic[i]``).
    """

    def __init__(self, critics):
        self.critics = critics

    def __getitem__(self, agent_id):
        return self.critics[agent_id]

    def __len__(self):
        return len(self.critics)

    def lr_decay(self, step, steps):
        for critic in self.critics:
            critic.lr_decay(step, steps)

    def soft_update(self):
        for critic in self.critics:
            critic.soft_update()

    def save(self, save_dir):
        import os

        for agent_id, critic in enumerate(self.critics):
            agent_dir = os.path.join(str(save_dir), f"critic_agent{agent_id}")
            os.makedirs(agent_dir, exist_ok=True)
            critic.save(agent_dir)

    def restore(self, model_dir):
        import os

        for agent_id, critic in enumerate(self.critics):
            critic.restore(os.path.join(str(model_dir), f"critic_agent{agent_id}"))


class OffPolicyIDDPGRunner(OffPolicyMARunner):
    """Runner for Independent DDPG (per-agent actor + per-agent local critic)."""

    def __init__(self, args, algo_args, env_args):
        # Build everything via the base runner first (env, actors, a single
        # placeholder centralized critic, buffer, ...). We then discard the
        # centralized critic and replace it with one independent critic per agent.
        super().__init__(args, algo_args, env_args)

        if self.algo_args["render"]["use_render"]:
            return

        # Each agent's critic consumes ONLY its own local observation and its own
        # action (truly decentralized IDDPG). The env exposes a global share_obs
        # for MADDPG's centralized critic, but IDDPG deliberately ignores it and
        # feeds each critic the agent's own ``observation_space`` / action so the
        # value functions stay fully decentralized.
        merged_args = {
            **self.algo_args["train"],
            **self.algo_args["model"],
            **self.algo_args["algo"],
        }
        critics = []
        for agent_id in range(self.num_agents):
            critics.append(
                ContinuousQCritic(
                    merged_args,
                    self.envs.observation_space[agent_id],
                    [self.envs.action_space[agent_id]],
                    1,  # single-agent critic
                    "EP",
                    device=self.device,
                )
            )
        self.critic = _IndependentCritics(critics)

        # Re-restore now that the per-agent critics exist (base __init__ already
        # ran restore() against the placeholder centralized critic).
        if self.algo_args["train"]["model_dir"] is not None:
            self.restore()

    def train(self):
        """Independent DDPG update: each agent trains its own local critic and
        actor, using only its own observation and action."""
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,  # (batch_size, dim)  -- unused by IDDPG
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # (batch_size, 1)
            sp_done,  # (batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # (batch_size, 1)
            sp_next_share_obs,  # (batch_size, dim)  -- unused by IDDPG
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # (batch_size, 1)
        ) = data

        # --- train each agent's independent critic on its LOCAL obs/action ----
        for agent_id in range(self.num_agents):
            self.critic[agent_id].turn_on_grad()
            next_action = self.actor[agent_id].get_target_actions(
                sp_next_obs[agent_id]
            )
            self.critic[agent_id].train(
                sp_obs[agent_id],
                np.array([sp_actions[agent_id]]),  # (1, batch, dim)
                sp_reward,
                sp_done,
                sp_term,
                sp_next_obs[agent_id],
                [next_action],
                sp_gamma,
            )
            self.critic[agent_id].turn_off_grad()

        if self.total_it % self.policy_freq == 0:
            # --- train each agent's actor against its OWN critic only ---------
            for agent_id in range(self.num_agents):
                self.actor[agent_id].turn_on_grad()
                action = self.actor[agent_id].get_actions(sp_obs[agent_id], False)
                value_pred = self.critic[agent_id].get_values(
                    sp_obs[agent_id], action
                )
                actor_loss = -torch.mean(value_pred)
                self.actor[agent_id].actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor[agent_id].actor_optimizer.step()
                self.actor[agent_id].turn_off_grad()
            # --- soft update every agent's actor and critic -------------------
            for agent_id in range(self.num_agents):
                self.actor[agent_id].soft_update()
                self.critic[agent_id].soft_update()
