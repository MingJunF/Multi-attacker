"""Stage-aware FP critic buffer for ordered (leader -> follower) attacks.

This buffer implements the *Stage-Aware MAPPO* value structure. Each
environment step is treated as TWO sequential decision stages driven by a
single shared critic ``V``:

    o-stage (obs attacker, agent 0): state x^o_t = [s_t, 0]
    a-stage (act attacker, agent 1): state x^a_t = [s_t, victim.act(s_t + delta_o)]

The two stage values are coupled by an intra-step Bellman relation:

    V^o(x^o_t) = E_{u^o}[ V^a(x^a_t) ]                         (o -> a: no reward, no discount)
    V^a(x^a_t) = E_{u^a}[ r_t + gamma * V^o(x^o_{t+1}) ]        (a -> o(t+1): reward r, discount gamma)

Running GAE over the doubled stage sequence
    ..., x^o_t, x^a_t, x^o_{t+1}, x^a_{t+1}, ...
yields the exact stage advantage decomposition

    A^o_t (obs attacker) = GAE at the o-stage   ~  V^a(x^a_t) - V^o(x^o_t) + ...
    A^a_t (act attacker) = GAE at the a-stage   ~  r_t + gamma V^o(x^o_{t+1}) - V^a(x^a_t) + ...

so that obs-attack credit and act-attack credit are separated by construction.
Only ``compute_returns`` differs from the standard FP buffer; everything else
(storage, generators) is inherited. The per-agent ``returns`` written here feed
both the actor advantages (returns - value_preds) and the shared critic target.
"""
import numpy as np

from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP


class OnPolicyCriticBufferStage(OnPolicyCriticBufferFP):
    """FP critic buffer with stage-aware (intra-step bootstrapping) GAE."""

    # Agent ordering must match the env's agent_roles: 0 = obs attacker
    # (leader), 1 = act attacker (follower).
    LEADER = 0
    FOLLOWER = 1

    def compute_returns(self, next_value, value_normalizer=None):
        """Compute stage-aware returns/advantages via GAE over the doubled
        (o-stage, a-stage) sequence.

        Args:
            next_value: (np.ndarray) per-agent value predictions for the step
                after the last episode step, shape (n_threads, num_agents, 1).
                Only the leader column (V^o_T) is used as the terminal bootstrap.
            value_normalizer: (ValueNorm) optional value normalizer.
        """
        assert self.use_gae, "Stage-aware critic buffer requires use_gae=True."

        self.value_preds[-1] = next_value
        gamma = self.gamma
        # Two decoupled GAE lambdas:
        #   lam_time  -- temporal credit across env steps (a_t -> o_{t+1}).
        #   lam_stage -- intra-step leader<-follower coupling (o_t -> a_t), i.e.
        #                how much of the act (follower) advantage flows into the
        #                obs (leader) advantage. Defaults to lam_time for exact
        #                backward compatibility; set algo.stage_lambda to
        #                decouple (lam_stage=0 -> the obs actor trains on the
        #                pure stage gap delta_o = V^a - V^o, dropping the
        #                follower-coupling term entirely).
        lam_time = self.gae_lambda
        lam_stage = getattr(self, "stage_lambda", self.gae_lambda)

        o = self.LEADER
        a = self.FOLLOWER

        # Per-stage denormalization: V^o and V^a may maintain independent
        # running statistics (StageValueNorm). Fall back to a single shared
        # normalizer (or identity) otherwise.
        from harl.common.stage_value_norm import StageValueNorm

        def denorm(x, stage):
            if value_normalizer is None:
                return x
            if isinstance(value_normalizer, StageValueNorm):
                return value_normalizer[stage].denormalize(x)
            return value_normalizer.denormalize(x)

        num_steps = self.rewards.shape[0]

        # GAE advantage of the o-stage exactly one ENV step ahead (forward
        # neighbour of the current a-stage). Starts at 0 at the trajectory tail.
        adv_o_next = 0.0
        for step in reversed(range(num_steps)):
            v_o_t = denorm(self.value_preds[step, :, o], o)        # V^o(x^o_t)
            v_a_t = denorm(self.value_preds[step, :, a], a)        # V^a(x^a_t)
            v_o_tp1 = denorm(self.value_preds[step + 1, :, o], o)  # V^o(x^o_{t+1})

            r_t = self.rewards[step, :, o]            # shared reward (same for both)
            cont = self.masks[step + 1, :, o]         # 0.0 if the episode ended at t
            bad = self.bad_masks[step + 1, :, o]      # 0.0 if the step was truncated

            # --- a-stage (act attacker): a_t -> o_{t+1}, reward r_t, discount gamma
            delta_a = r_t + gamma * v_o_tp1 * cont - v_a_t
            adv_a = delta_a + gamma * lam_time * cont * adv_o_next
            # On truncation, do not propagate the (bootstrapped) future credit.
            adv_a = adv_a * bad

            # --- o-stage (obs attacker): o_t -> a_t, no reward, no discount
            delta_o = v_a_t - v_o_t
            adv_o = delta_o + lam_stage * adv_a

            self.returns[step, :, o] = adv_o + v_o_t
            self.returns[step, :, a] = adv_a + v_a_t

            adv_o_next = adv_o
