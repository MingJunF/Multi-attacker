"""Runner for Stage-Aware MAPPO (ordered obs->act attack with stage values).

Stage-Aware MAPPO keeps a SINGLE shared centralized critic but, unlike vanilla
(EP) MAPPO or the input-masked causal MA-MAPPO, it gives that critic a genuine
two-stage value structure with intra-step Bellman bootstrapping:

    o-stage (obs attacker, agent 0): x^o_t = [s_t, 0]
    a-stage (act attacker, agent 1): x^a_t = [s_t, victim.act(s_t + delta_o)]

    V^o(x^o_t) = E_{u^o}[ V^a(x^a_t) ]                  (within-step, no reward, no discount)
    V^a(x^a_t) = E_{u^a}[ r_t + gamma V^o(x^o_{t+1}) ]  (across-step, reward r, discount gamma)

The rollout is sequential (leader acts -> env commits obs attack and recomputes
the victim action -> follower acts on the CURRENT victim action), identical to
AR-MAPPO. The difference from AR-MAPPO lives entirely in the value learning:
the critic is evaluated on the CURRENT-step per-agent states x^o_t, x^a_t and
the returns/advantages are computed by a stage-aware GAE
(``OnPolicyCriticBufferStage``), yielding the exact decomposition A = A_obs + A_act.

Requirements (set on the command line / env config):
    state_type: FP            (single critic produces a value per agent)
    causal_critic_state: True (the env masks the leader's victim-action slot so
                               x^o = [s, 0]; the follower keeps x^a = [s, a^v])
"""
import numpy as np
import torch

from harl.algorithms.critics.stage_v_critic import StageVCritic
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.common.buffers.on_policy_critic_buffer_stage import (
    OnPolicyCriticBufferStage,
)
from harl.common.stage_value_norm import StageValueNorm
from harl.models.policy_models.masked_stochastic_policy import MaskedStochasticPolicy
from harl.models.value_function_models.stage_v_net import StageVNet
from harl.runners.on_policy_ar_runner import OnPolicyARRunner
from harl.utils.envs_tools import check
from harl.utils.models_tools import huber_loss, mse_loss
from harl.utils.trans_tools import _t2n


class OnPolicyStageRunner(OnPolicyARRunner):
    """Stage-Aware MAPPO runner: sequential rollout + stage-aware critic."""

    def __init__(self, args, algo_args, env_args):
        super().__init__(args, algo_args, env_args)
        assert self.state_type == "FP", (
            "Stage-Aware MAPPO requires state_type: FP (one shared critic "
            "producing a per-agent value). Pass --state_type FP."
        )
        assert self.num_agents == 2, "Stage-Aware MAPPO assumes two ordered agents."
        # Swap in the stage-aware GAE without rebuilding the buffer: the stage
        # buffer is a pure subclass of the FP buffer that only overrides
        # compute_returns, so re-tagging the class is sufficient and safe.
        assert isinstance(self.critic_buffer, OnPolicyCriticBufferFP), (
            "Stage-Aware MAPPO expects the FP critic buffer (state_type: FP)."
        )
        self.critic_buffer.__class__ = OnPolicyCriticBufferStage

        # --- per-stage value normalization (#2) -----------------------------
        # Replace the single critic with a stage-aware one that normalizes the
        # o-stage value V^o and the a-stage value V^a with independent running
        # statistics, routed by the stage one-hot in the centralized state.
        if self.value_normalizer is not None:
            assert self.env_args.get("causal_critic_state", False), (
                "Stage-Aware MAPPO's per-stage value normalization needs the "
                "stage one-hot in the critic state; pass --causal_critic_state "
                "True."
            )
            old_critic = self.critic
            self.critic = StageVCritic(
                old_critic.args,
                old_critic.share_obs_space,
                num_stages=self.num_agents,
                device=old_critic.device,
            )
            self.value_normalizer = StageValueNorm(
                num_stages=self.num_agents, device=self.device
            )

        # Optional two-head critic: replace the single shared value head with
        # two stage-specific heads V^o / V^a (shared backbone), routed by the
        # stage one-hot in the critic state. Requires the env to append the
        # stage id (causal_critic_state: True). Rebuild the optimizer so the new
        # head parameters are tracked.
        if self.env_args.get("two_head_critic", False):
            assert self.env_args.get("causal_critic_state", False), (
                "two_head_critic needs the stage one-hot in the critic state; "
                "pass --causal_critic_state True."
            )
            self.critic.critic = StageVNet(
                self.critic.args, self.critic.share_obs_space, self.critic.device
            )
            self.critic.critic_optimizer = torch.optim.Adam(
                self.critic.critic.parameters(),
                lr=self.critic.critic_lr,
                eps=self.critic.opti_eps,
                weight_decay=self.critic.weight_decay,
            )

        # Optional gradient-interference probe (logging only, default off): once
        # per update, measure cos(grad L^o, grad L^a) on the shared backbone and
        # on the value head(s) separately to tell apart backbone representation
        # conflict from head-level conflict. See ``_grad_cosine_diagnostics``.
        self.grad_cos_diag = bool(self.env_args.get("grad_cos_diag", False))

        # Optional obs-action value-spread probe (logging only, default off):
        # once per update, sample K obs-attacks u^o ~ pi^o at the current state,
        # query the victim for each, and measure the critic's perceived value
        # spread across obs-actions (pred_action_var/action_range) and the
        # intra-stage Bellman inconsistency E[V^o] vs mean_u V^a (bellman_gap).
        # See ``_obs_action_spread_diagnostics``.
        self.obs_spread_diag = bool(self.env_args.get("obs_spread_diag", False))
        self.obs_spread_k = int(self.env_args.get("obs_spread_k", 16))

        # --- follower masked log-prob (#1) ----------------------------------
        # The follower (action attacker) reports a padded action space but only
        # its first ``valid_action_dim`` dims drive the env. Restrict its PPO
        # log-prob/entropy to those dims so the padding dims never pollute the
        # importance ratio or entropy bonus. The leader (obs attacker) has no
        # padding (valid dim == pad dim), so it is left untouched.
        follower_space = self.envs.action_space[self.FOLLOWER_ID]
        valid_dim = int(
            getattr(follower_space, "valid_action_dim", follower_space.shape[0])
        )
        fa = self.actor[self.FOLLOWER_ID]
        if valid_dim < fa.act_space.shape[0]:
            fa.actor = MaskedStochasticPolicy(
                fa.args, fa.obs_space, fa.act_space, valid_dim, fa.device
            )
            fa.actor_optimizer = torch.optim.Adam(
                fa.actor.parameters(),
                lr=fa.lr,
                eps=fa.opti_eps,
                weight_decay=fa.weight_decay,
            )

    @torch.no_grad()
    def collect(self, step):
        """Two-phase sequential collection with CURRENT-step stage critic states.

        Phase 1 queries the leader; ``begin_step`` commits its obs attack and
        returns per-agent observations/share_obs whose follower view holds the
        CURRENT-step victim action. We overwrite the follower's actor obs AND
        BOTH agents' centralized-critic states with these current-step values
        (leader -> x^o = [s, 0]; follower -> x^a = [s, victim.act(s + delta_o)]),
        then query the follower and evaluate the shared critic on x^o, x^a.
        """
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

        # --- commit leader attack; fetch CURRENT-step per-agent obs/share_obs
        obs, share_obs = self.envs.begin_step(action_collector[leader])
        # Follower acts on the current-step victim action.
        self.actor_buffer[follower].obs[step] = obs[:, follower].copy()
        # Stage-aware critic: evaluate and TRAIN both stage values on the
        # current-step states. Overwriting share_obs[step] (not just reading it)
        # means the critic target stored for this step is x^o_t / x^a_t.
        self.critic_buffer.share_obs[step] = share_obs.copy()

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

        # --- shared critic values on the current-step per-agent states (FP) --
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
                _t2n(rnn_state_critic), self.algo_args["train"]["n_rollout_threads"]
            )
        )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def train(self):
        """Stage-aware MAPPO training update.

        Differs from vanilla FP MAPPO in two places:
          * advantages are de-normalized per stage (StageValueNorm), and
          * advantages are normalized per stage (the within-step o-stage and
            the reward-scale a-stage have very different magnitudes, so a single
            global normalization would let the larger stage dominate).
        """
        actor_train_infos = []

        # --- advantages with per-stage value de-normalization ---------------
        returns = self.critic_buffer.returns[:-1]
        value_preds = self.critic_buffer.value_preds[:-1]
        if self.value_normalizer is not None:
            if isinstance(self.value_normalizer, StageValueNorm):
                denorm = np.zeros_like(returns)
                for s in range(self.num_agents):
                    denorm[:, :, s] = self.value_normalizer[s].denormalize(
                        value_preds[:, :, s]
                    )
            else:
                denorm = self.value_normalizer.denormalize(value_preds)
            advantages = returns - denorm
        else:
            denorm = value_preds
            advantages = returns - value_preds

        # --- per-stage advantage normalization (FP) -------------------------
        active_masks_collector = [
            self.actor_buffer[i].active_masks for i in range(self.num_agents)
        ]
        active_masks_array = np.stack(active_masks_collector, axis=2)

        # --- stage diagnostics (logging only; RAW pre-norm advantages) ------
        stage_diag = self._stage_diagnostics(advantages, denorm, active_masks_array)

        for s in range(self.num_agents):
            adv_s = advantages[:, :, s].copy()
            adv_s[active_masks_array[:-1, :, s] == 0.0] = np.nan
            mean_s = np.nanmean(adv_s)
            std_s = np.nanstd(adv_s)
            advantages[:, :, s] = (advantages[:, :, s] - mean_s) / (std_s + 1e-5)

        # --- update actors --------------------------------------------------
        if self.share_param:
            actor_train_info = self.actor[0].share_param_train(
                self.actor_buffer, advantages.copy(), self.num_agents, self.state_type
            )
            for _ in torch.randperm(self.num_agents):
                actor_train_infos.append(actor_train_info)
        else:
            for agent_id in range(self.num_agents):
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id],
                    advantages[:, :, agent_id].copy(),
                    "FP",
                )
                actor_train_infos.append(actor_train_info)

        # --- update critic --------------------------------------------------
        critic_train_info = self.critic.train(
            self.critic_buffer, self.value_normalizer
        )
        critic_train_info.update(stage_diag)
        if self.grad_cos_diag:
            critic_train_info.update(self._grad_cosine_diagnostics())
        if self.obs_spread_diag:
            critic_train_info.update(self._obs_action_spread_diagnostics())

        return actor_train_infos, critic_train_info

    def _grad_cosine_diagnostics(self):
        """Gradient-interference probe between the o-stage and a-stage critic
        losses, measured separately on the shared backbone and on the value
        head(s).

        Diagnostic only: uses ``torch.autograd.grad`` so it never touches the
        optimizer state or the parameter ``.grad`` buffers, performs no update,
        and (via ``normalize`` without ``update``) leaves the value-norm running
        statistics untouched. One full-batch forward + two backward passes.

        Interpretation (o = obs stage, a = act stage):
          * ``diag_gradcos_backbone`` < 0  -> the two stage targets pull the
            SHARED representation in opposing directions. A single-head critic
            cannot fit both, which collapses the residual delta_o (explains
            ``diag_corr_resid_return`` ~ 0). A two-head critic only decouples
            the head, so a negative BACKBONE cosine means two heads cannot help
            -- the fix is a separate encoder per stage.
          * ``diag_gradcos_head``: for a single shared head this measures
            head-level conflict; for ``two_head_critic`` the heads are disjoint
            (gated), so this is ~0 by construction and serves as a sanity check.
        """
        net = self.critic.critic
        if not hasattr(net, "base"):
            return {}

        o = OnPolicyCriticBufferStage.LEADER    # obs attacker = 0
        a = OnPolicyCriticBufferStage.FOLLOWER  # act attacker = 1

        # one full-batch sample (critic_num_mini_batch -> 1). The generator
        # draws ``torch.randperm`` to shuffle the minibatch, which would consume
        # the global RNG and desynchronize the rest of training relative to a
        # run with the probe disabled. Save/restore the RNG state so enabling
        # this diagnostic leaves the training trajectory bit-identical.
        rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state_all()
        sample = next(self.critic_buffer.feed_forward_generator_critic(1))
        torch.set_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)
        (
            share_obs_batch,
            rnn_states_critic_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
        ) = sample
        return_batch = check(return_batch).to(**self.critic.tpdv)
        stage_ids = check(share_obs_batch[..., -self.num_agents :]).to(
            **self.critic.tpdv
        )
        stage_ids = stage_ids.argmax(dim=-1, keepdim=True)

        values, _ = self.critic.get_values(
            share_obs_batch, rnn_states_critic_batch, masks_batch
        )

        def stage_loss(s):
            row = (stage_ids == s).squeeze(-1)
            if not bool(row.any()):
                return None
            v = values[row]
            tgt = return_batch[row]
            if isinstance(self.value_normalizer, StageValueNorm):
                tgt = self.value_normalizer[s].normalize(tgt)
            elif self.value_normalizer is not None:
                tgt = self.value_normalizer.normalize(tgt)
            err = tgt - v
            if self.critic.use_huber_loss:
                return huber_loss(err, self.critic.huber_delta).mean()
            return mse_loss(err).mean()

        loss_o = stage_loss(o)
        loss_a = stage_loss(a)
        if loss_o is None or loss_a is None:
            return {}

        base_params = list(net.base.parameters())
        head_params = [
            p for n, p in net.named_parameters() if not n.startswith("base.")
        ]
        all_params = base_params + head_params
        n_base = sum(p.numel() for p in base_params)

        def flat_grad(loss, retain):
            grads = torch.autograd.grad(
                loss, all_params, retain_graph=retain, allow_unused=True
            )
            parts = [
                (g if g is not None else torch.zeros_like(p)).reshape(-1)
                for g, p in zip(grads, all_params)
            ]
            return torch.cat(parts)

        g_o = flat_grad(loss_o, retain=True)
        g_a = flat_grad(loss_a, retain=False)

        def cos(x, y):
            denom = x.norm() * y.norm()
            if float(denom) < 1e-12:
                return 0.0
            return float((x @ y) / denom)

        return {
            "diag_gradcos_backbone": cos(g_o[:n_base], g_a[:n_base]),
            "diag_gradcos_head": cos(g_o[n_base:], g_a[n_base:]),
            "diag_gradcos_full": cos(g_o, g_a),
            "diag_gradnorm_o": float(g_o.norm()),
            "diag_gradnorm_a": float(g_a.norm()),
        }

    def _obs_action_spread_diagnostics(self):
        """Logging-only probe of the critic's obs-action value spread and of the
        intra-stage Bellman consistency ``V^o(x^o) ?= E_{u^o ~ pi^o}[V^a(x^a)]``.

        For each parallel env's CURRENT state ``s`` (the bootstrap row, aligned
        with the leader's actor ``obs[-1]``), sample ``K`` obs-attacks
        ``u^o ~ pi^o``, query the victim for each (read-only) to build
        ``x^a_k = [s, victim.act(s + u^o), onehot_a]``, and evaluate the shared
        critic::

            q_k = V^a(x^a_k),   mu = mean_k q_k,   A_k = q_k - mu,   V^o = V^o(x^o)

        Metrics (uniform 1/K weights -- ``u^o`` is continuous, so we SAMPLE, not
        enumerate; hence the baseline is the simple mean, not a pi-weighted sum):

          * ``diag_spread_pred_action_var`` = E_i mean_k A_k^2 -- the critic's
            PERCEIVED spread of value across obs-actions. NOT the true marginal
            effect: each V^a still carries the critic's V^a error e_a, so this
            cannot by itself separate "true Delta small" from "e_a large".
          * ``diag_spread_bellman_gap`` = E_i (V^o_i - mu_i)^2 -- intra-stage
            Bellman inconsistency. Since V^o - mu = e_o - mean_u e_a (independent
            of the true gap), a large value flags that the V^o / V^a heads are
            mutually inconsistent -- exactly what a mean-zero dueling head would
            remove. This is the most trustworthy of the four (no true-Delta
            knowledge required).
          * ``diag_spread_mean_abs_adv`` = E_ik |A_k|.
          * ``diag_spread_action_range`` = E_i (max_k q - min_k q).

        Read-only: ``torch.no_grad``, eval critic, victim recurrent state is
        snapshotted/restored in the env, no optimizer/backward, no env-state
        mutation. The ``u^o`` sampling consumes the global torch RNG, so the RNG
        state is saved/restored to keep a run with the probe enabled bit-identical
        to one without it.
        """
        # Stage routing (one-hot) is required to tell V^o from V^a.
        if not self.env_args.get("causal_critic_state", False):
            return {}

        leader = self.LEADER_ID
        K = self.obs_spread_k
        lo = self.actor_buffer[leader]
        obs = lo.obs[-1]            # (n, aug_obs_dim) current leader observation
        rnn = lo.rnn_states[-1]     # (n, recurrent_n, hidden)
        masks = lo.masks[-1]        # (n, 1)
        n = obs.shape[0]

        obs_t = np.repeat(obs, K, axis=0)
        rnn_t = np.repeat(rnn, K, axis=0)
        masks_t = np.repeat(masks, K, axis=0)
        avail = (
            np.repeat(lo.available_actions[-1], K, axis=0)
            if lo.available_actions is not None
            else None
        )

        # Sample K obs-attacks per env from pi^o. Save/restore RNG so enabling
        # the probe leaves the training trajectory bit-identical.
        rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state_all()
        with torch.no_grad():
            actions, _, _ = self.actor[leader].get_actions(
                obs_t, rnn_t, masks_t, avail, deterministic=False
            )
        torch.set_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)
        deltas = _t2n(actions).reshape(n, K, -1)   # (n, K, pad_dim)

        clean_states, act_views = self.envs.probe_act_views(deltas)
        clean_states = np.asarray(clean_states, dtype=np.float32)  # (n, obs_dim)
        act_views = np.asarray(act_views, dtype=np.float32)        # (n, K, act_dim)
        act_dim = act_views.shape[-1]

        o = OnPolicyCriticBufferStage.LEADER
        a = OnPolicyCriticBufferStage.FOLLOWER
        onehot_o = np.zeros(self.num_agents, dtype=np.float32)
        onehot_o[o] = 1.0
        onehot_a = np.zeros(self.num_agents, dtype=np.float32)
        onehot_a[a] = 1.0

        # x^o = [s, 0, onehot_o]  (n, share_dim)
        xo = np.concatenate(
            [
                clean_states,
                np.zeros((n, act_dim), dtype=np.float32),
                np.tile(onehot_o, (n, 1)),
            ],
            axis=1,
        )
        # x^a = [s, victim.act(s + u^o), onehot_a]  (n*K, share_dim)
        s_tiled = np.repeat(clean_states, K, axis=0)
        av_flat = act_views.reshape(n * K, act_dim)
        xa = np.concatenate(
            [s_tiled, av_flat, np.tile(onehot_a, (n * K, 1))], axis=1
        )

        crit_rnn_shape = self.critic_buffer.rnn_states_critic.shape[-2:]

        def values_for(share_obs):
            m = share_obs.shape[0]
            rnn0 = np.zeros((m,) + crit_rnn_shape, dtype=np.float32)
            msk = np.ones((m, 1), dtype=np.float32)
            with torch.no_grad():
                v, _ = self.critic.get_values(share_obs, rnn0, msk)
            return _t2n(v).reshape(m)

        v_o = values_for(xo)        # (n,)
        v_a = values_for(xa)        # (n*K,)

        if isinstance(self.value_normalizer, StageValueNorm):
            v_o = self.value_normalizer[o].denormalize(v_o)
            v_a = self.value_normalizer[a].denormalize(v_a)
        elif self.value_normalizer is not None:
            v_o = self.value_normalizer.denormalize(v_o)
            v_a = self.value_normalizer.denormalize(v_a)

        v_o = np.asarray(v_o, dtype=np.float64).reshape(n)
        q = np.asarray(v_a, dtype=np.float64).reshape(n, K)
        mu = q.mean(axis=1)
        adv = q - mu[:, None]

        return {
            "diag_spread_pred_action_var": float(np.mean(adv ** 2)),
            "diag_spread_bellman_gap": float(np.mean((v_o - mu) ** 2)),
            "diag_spread_mean_abs_adv": float(np.mean(np.abs(adv))),
            "diag_spread_action_range": float(
                np.mean(q.max(axis=1) - q.min(axis=1))
            ),
        }

    def _stage_diagnostics(self, advantages, denorm, active_masks_array):
        """Logging-only diagnostics for the stage-aware obs advantage.

        Tests whether the obs-stage signal (a) predicts the true discounted
        return and (b) points the same way as a plain return-based (IPPO-like)
        reference advantage ``G_t - V^o(x^o_t)``. A low correlation / sign
        agreement means the stage residual is steering the obs actor against
        the return-based gradient, which would explain underperforming IPPO.

        All quantities use the RAW (pre-normalization) per-stage advantages and
        are masked to active obs-stage steps. Returns a dict of ``diag_*``
        scalars merged into ``critic_train_info`` for the logger.
        """
        o = OnPolicyCriticBufferStage.LEADER    # obs attacker = 0
        a = OnPolicyCriticBufferStage.FOLLOWER  # act attacker = 1
        cb = self.critic_buffer
        T = advantages.shape[0]

        # Bootstrap V^o at step T (denormalized per stage).
        if isinstance(self.value_normalizer, StageValueNorm):
            boot = self.value_normalizer[o].denormalize(cb.value_preds[T, :, o])
        elif self.value_normalizer is not None:
            boot = self.value_normalizer.denormalize(cb.value_preds[T, :, o])
        else:
            boot = cb.value_preds[T, :, o]

        # Monte-Carlo discounted return-to-go seen at each obs stage, built from
        # the shared env reward (the signal a return-based PG would regress on).
        gamma = cb.gamma
        G = np.zeros((T,) + boot.shape, dtype=np.float32)
        future = boot
        for t in reversed(range(T)):
            future = cb.rewards[t, :, o] + gamma * cb.masks[t + 1, :, o] * future
            G[t] = future

        v_o = denorm[:, :, o]         # V^o(x^o_t)
        v_a = denorm[:, :, a]         # V^a(x^a_t)
        resid = v_a - v_o             # delta_o (intra-step stage residual)
        adv_o = advantages[:, :, o]   # full stage GAE advantage of the obs actor
        ref_adv = G - v_o             # return-based (IPPO-like) advantage
        eta_a = G - v_a               # residual return below V^a (downstream/env/critic noise)

        mask = active_masks_array[:-1, :, o] != 0.0

        def sel(x):
            return x[mask].astype(np.float64).ravel()

        resid_f = sel(resid)
        advo_f = sel(adv_o)
        ref_f = sel(ref_adv)
        G_f = sel(G)
        vo_f = sel(v_o)
        eta_f = sel(eta_a)

        def corr(x, y):
            if x.size < 2 or x.std() < 1e-8 or y.std() < 1e-8:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1])

        def sign_agree(x, y):
            if x.size == 0:
                return 0.0
            return float(np.mean(np.sign(x) == np.sign(y)))

        eps = 1e-8

        # --- variance decomposition of G - V^o = delta_o + eta_a ------------
        # delta_o = V^a - V^o (obs intervention's conditional value gap),
        # eta_a   = G - V^a   (downstream action sampling + env + critic error).
        # If V^a fits the return well (high EV) AND Var(delta_o) << Var(eta_a),
        # then the obs intervention's marginal value is genuinely small relative
        # to downstream/env variance -- NOT merely an unfit critic. EV is the
        # standard explained variance 1 - Var(G - V)/Var(G).
        var_g = float(G_f.var()) if G_f.size else 0.0
        var_resid = float(resid_f.var()) if resid_f.size else 0.0
        var_eta = float(eta_f.var()) if eta_f.size else 0.0
        cov_resid_eta = (
            float(np.cov(resid_f, eta_f)[0, 1]) if resid_f.size > 1 else 0.0
        )
        ev_vo = 1.0 - float(ref_f.var()) / (var_g + eps) if G_f.size else 0.0
        ev_va = 1.0 - var_eta / (var_g + eps) if G_f.size else 0.0

        # --- covariance-consistency probes (delta_o vs downstream residual) ---
        # identity_error: ref_adv - delta_o - eta_a is an ALGEBRAIC tautology
        # given the current construction (all three use the same G, v_o, v_a at
        # the same index), so this is ~0 by machine precision. It only guards
        # against NaN / masking / dtype corruption -- it CANNOT detect a stage
        # time-index misalignment (that must be checked in the buffer).
        identity_error = (
            float(np.max(np.abs(ref_f - resid_f - eta_f))) if ref_f.size else 0.0
        )
        # orth_ratio = Cov(delta_o, eta_a) / Var(delta_o). For consistent nested
        # conditional expectations this should be ~0; a value near -1 means the
        # estimated delta_o is cancelled by the downstream residual (critic
        # smoothing the small stage increment / misalignment / non-nested info).
        orth_ratio = cov_resid_eta / (var_resid + eps)
        # signal_recovery = Cov(delta_o, G - V^o) / Var(delta_o). ~1 means the
        # estimated delta_o is fully recovered inside the residual return; ~0
        # means it does not show up in the realized return at all.
        cov_resid_ref = (
            float(np.cov(resid_f, ref_f)[0, 1]) if resid_f.size > 1 else 0.0
        )
        signal_recovery = cov_resid_ref / (var_resid + eps)

        return {
            "diag_resid_std": float(resid_f.std()) if resid_f.size else 0.0,
            "diag_value_o_std": float(vo_f.std()) if vo_f.size else 0.0,
            "diag_resid_to_value_ratio": (
                float(resid_f.std() / (vo_f.std() + eps)) if resid_f.size else 0.0
            ),
            "diag_corr_resid_return": corr(resid_f, G_f),
            "diag_corr_advo_refadv": corr(advo_f, ref_f),
            "diag_sign_agree_advo_refadv": sign_agree(advo_f, ref_f),
            "diag_sign_agree_resid_refadv": sign_agree(resid_f, ref_f),
            # variance decomposition (delta_o vs eta_a) + critic explained variance
            "diag_var_resid": var_resid,
            "diag_var_eta_a": var_eta,
            "diag_var_ratio_resid_eta": var_resid / (var_eta + eps),
            "diag_cov_resid_eta": cov_resid_eta,
            "diag_ev_vo_return": ev_vo,
            "diag_ev_va_return": ev_va,
            # covariance-consistency probes (delta_o vs downstream residual)
            "diag_corr_resid_refadv": corr(resid_f, ref_f),
            "diag_identity_error": identity_error,
            "diag_orth_ratio": orth_ratio,
            "diag_signal_recovery": signal_recovery,
        }
