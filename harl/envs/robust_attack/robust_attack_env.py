"""Multi-attacker environment built on top of robust_gymnasium.

This module formulates the adversarial attack on a *fixed* victim RL policy as a
cooperative multi-agent problem that can be solved with any HARL on-policy /
off-policy algorithm (independent PPO, MAPPO, HAPPO, ...).

Two attacker agents jointly degrade the victim's return:

    agent 0 -- ObservationAttacker
        Outputs an additive perturbation ``delta_o`` that is added to the
        observation *before* the victim sees it. Its magnitude is bounded by a
        fixed absolute L_inf cap ``epsilon_observation`` (default 0.5).

    agent 1 -- ActionAttacker
        Outputs an additive perturbation ``delta_a`` that tampers with the
        victim action *before* it is applied. Its magnitude is bounded by a
        fixed absolute L_inf cap ``epsilon_action`` (default 0.5). It can react
        to the victim because the victim's nominal action is part of the
        attacker observation.

The attacker capability is constrained by TWO independent limits: the
perturbation *magnitude* (``epsilon_*``, unrelated to the budget) and the
*number of attacks* (the per-attacker budget below).

Both attackers receive the same augmented observation:

    [ true victim state (obs_dim),
      victim nominal action (act_dim),
      remaining observation budget (normalised),
      remaining action budget      (normalised) ]

Per-step interaction (sequential flow):

    s_t                      (true state, stored from previous step)
    s~_t = s_t + delta_o     (observation attack, |delta_o| <= eps_obs, budget)
    a_t  = pi_victim(s~_t)   (fixed victim acts on the corrupted observation)
    a~_t = a_t + delta_a     (action attack, |delta_a| <= eps_act, budget)
    s_{t+1}, r_t = env(a~_t) (robust_gymnasium transition)
    r_attack = -r_t - penalty(t)   (cooperative reward + early-spend penalty)

The two attackers therefore *share* a common (cooperative) reward, which makes
the problem directly compatible with MAPPO / HAPPO. Using ``mappo`` with
``share_param: False`` recovers two independent PPO attackers.

Each attacker additionally owns an *independent*, sparse attack budget
(``budget_observation`` / ``budget_action``) that is consumed whenever it
perturbs the victim. Once a budget is spent that attacker can no longer attack
for the remainder of the episode. A well-trained victim survives ~300 steps in
this env, so a ``count`` budget of ~30 lets each attacker tamper on roughly
1/10 of the steps (they spend independently and may attack the same step). To
stop the attackers from blowing this sparse budget in the first random steps, an
annealed regularization penalty discourages early spending. Budgets are
replenished on ``reset()`` and only constrain the learned MARL attackers -- they
never affect robust_gymnasium's own (disabled) perturbation machinery.

HARL multi-agent interface (mirrors ``harl.envs.gym.gym_env.GYMEnv``):
    step(actions) -> (obs_n, share_obs_n, rewards_n, dones_n, infos_n, avail)
    reset()       -> (obs_n, share_obs_n, avail)

Heterogeneous action dimensions (``obs_dim`` vs ``act_dim``) are handled the
same way as HARL's mamujoco wrapper: every agent reports a zero-padded action
space of size ``max(obs_dim, act_dim)`` so the on-policy runner can stack the
actions, while the environment slices each agent's real action back out.
"""

import copy

import numpy as np
import robust_gymnasium as rgym
from robust_gymnasium.spaces.box import Box

from harl.envs.robust_attack.victim_policy import make_victim_policy


# Attacker roles. Order matters: it defines the agent indexing seen by HARL.
OBSERVATION_ATTACKER = "observation"
ACTION_ATTACKER = "action"


def _build_disabled_robust_config(background_noise=None):
    """Build a robust_gymnasium config namespace with internal noise disabled.

    The attacker agents generate the perturbations themselves, so the
    environment's *built-in* noise is switched off by default (``noise_factor``
    is set to a sentinel that matches none of the perturbation branches).

    Args:
        background_noise: (dict | None) optional overrides to re-enable the
            environment's intrinsic noise on top of the attacker perturbations.
    Returns:
        argparse.Namespace consumable as ``robust_input["robust_config"]``.
    """
    from robust_gymnasium.configs.robust_setting import get_config

    args = get_config().parse_args([])
    # Disable all built-in perturbations; the attackers are the only source.
    args.noise_factor = "disable"
    if background_noise:
        for key, value in background_noise.items():
            setattr(args, key, value)
    return args


class RobustAttackEnv:
    """Cooperative multi-attacker wrapper around a robust_gymnasium env."""

    def __init__(self, args):
        self.args = copy.deepcopy(args)

        scenario = self.args["scenario"]
        self.scenario = scenario

        # --- build the underlying victim environment -------------------------
        render_mode = self.args.get("render_mode", None)
        if render_mode is not None:
            self.env = rgym.make(scenario, render_mode=render_mode)
        else:
            self.env = rgym.make(scenario)

        # Maze tasks (PointMaze/AntMaze) use a plain-action ``step(action)`` API
        # (not the mujoco/fetch ``robust_input`` dict) and expose a Dict
        # observation {observation, achieved_goal, desired_goal}. Mirror
        # RobustVictimEnv so the attacker operates on the SAME flattened Box
        # state the victim policy was trained on.
        self._maze_api = scenario.startswith("PointMaze") or scenario.startswith(
            "AntMaze"
        )
        if self._maze_api:
            # Disable the maze's built-in global-args noise; the attackers are
            # the only perturbation source.
            try:
                import robust_gymnasium.envs.robust_maze.point_maze as _pm

                _pm.args.noise_factor = "disable"
            except Exception:
                pass

        raw_obs_space = self.env.observation_space
        victim_act_space = self.env.action_space
        if victim_act_space.__class__.__name__ != "Box":
            raise NotImplementedError(
                "RobustAttackEnv currently supports continuous (Box) victim "
                f"action spaces, got {type(victim_act_space)}."
            )

        # Goal-conditioned tasks (maze) expose a Dict observation; flatten it to
        # concat(observation, desired_goal) exactly like RobustVictimEnv so the
        # obs dims (and the loaded victim policy) match.
        self._goal_conditioned = raw_obs_space.__class__.__name__ == "Dict"
        if self._goal_conditioned:
            flat_dim = int(np.prod(raw_obs_space["observation"].shape)) + int(
                np.prod(raw_obs_space["desired_goal"].shape)
            )
            victim_obs_space = Box(
                low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
            )
        else:
            victim_obs_space = raw_obs_space

        self.obs_dim = int(np.prod(victim_obs_space.shape))
        self.act_dim = int(np.prod(victim_act_space.shape))
        self.victim_obs_space = victim_obs_space
        self.victim_act_low = np.asarray(victim_act_space.low, dtype=np.float32)
        self.victim_act_high = np.asarray(victim_act_space.high, dtype=np.float32)
        # Observation clipping bounds (only used when finite & enabled).
        self._obs_low = np.asarray(victim_obs_space.low, dtype=np.float32)
        self._obs_high = np.asarray(victim_obs_space.high, dtype=np.float32)

        # --- attacker configuration -----------------------------------------
        # The attacker capability is constrained by TWO independent limits:
        #   1. perturbation MAGNITUDE: a fixed absolute L_inf bound on every
        #      perturbation, epsilon_observation / epsilon_action (default 0.5).
        #      This is independent of the budget.
        #   2. number of ATTACKS: the per-attacker budget below (default 30).
        self.epsilon_observation = float(self.args.get("epsilon_observation", 0.5))
        self.epsilon_action = float(self.args.get("epsilon_action", 0.5))

        self.clip_observation = bool(self.args.get("clip_observation", False))
        self.reward_scale = float(self.args.get("reward_scale", 1.0))

        # --- per-attacker attack budgets ------------------------------------
        # Each attacker (observation / action) owns an *independent* budget that
        # is consumed every time it perturbs the victim. Once a budget is spent
        # the corresponding attacker can no longer attack (its perturbation is
        # forced to zero for the rest of the episode). A well-trained victim
        # survives ~300 steps in this env, so a ``count`` budget of ~30 lets an
        # attacker tamper on roughly 1/10 of the steps (the two attackers spend
        # independently and may attack the same step). These budgets only govern
        # the learned MARL attackers; they never touch robust_gymnasium's own
        # (disabled) perturbation machinery.
        #   budget_observation / budget_action: total budget per episode
        #       (use None / inf for the original unbounded behaviour).
        #   budget_cost: how each step's spend is measured, one of
        #       "count" (1 per attack step, default), "l1", "l2", or "linf".
        self.budget_observation = self._parse_budget(
            self.args.get("budget_observation", None)
        )
        self.budget_action = self._parse_budget(
            self.args.get("budget_action", None)
        )
        self.budget_cost = str(self.args.get("budget_cost", "count")).lower()
        if self.budget_cost not in ("l1", "l2", "linf", "count"):
            raise ValueError(
                "budget_cost must be one of 'l1', 'l2', 'linf', 'count'; "
                f"got {self.budget_cost!r}."
            )
        # --- attack deadzone (sparsity gate) --------------------------------
        # With a continuous policy the actor almost never outputs exactly zero,
        # so under a "count" budget *every* step would spend budget even for a
        # negligible perturbation. The deadzone gives the attacker a learnable
        # "do nothing" option: if the (clipped) perturbation's L_inf magnitude
        # is below ``attack_threshold`` it is treated as NO attack -- the
        # perturbation is zeroed and NO budget is charged. The attacker can thus
        # abstain for free by keeping its output small and only "commit" an
        # attack (cross the threshold) at critical moments.
        self.attack_threshold = float(self.args.get("attack_threshold", 0.0))
        if self.attack_threshold < 0.0:
            raise ValueError(
                f"attack_threshold must be non-negative, got {self.attack_threshold}."
            )
        # When True, the action attacker additionally observes the victim's
        # observation (the obs-corrupted state s~ the victim sees) instead of a
        # zero-masked state slot, so it sees BOTH the victim observation and the
        # victim action. Default False keeps the strict state/action separation.
        self.act_observe_state = bool(self.args.get("act_observe_state", False))
        # When True (MA-MAPPO), the centralized critic state is built PER AGENT
        # and conditioned on exactly the information realized BEFORE that agent
        # acts in the sequential attack order (obs-attack -> victim -> act-attack).
        # This yields the exact chain advantage decomposition A = A_obs + A_act:
        #   * obs-attacker (leader) baseline must NOT depend on its own delta_o,
        #     so the delta_o-dependent slot of share_obs is zero-masked -> V_o(s).
        #   * act-attacker (follower) baseline conditions on delta_o (via that
        #     slot) -> Q_o(s, delta_o).
        # Requires state_type "FP" so the single shared critic produces a
        # per-agent value. Default False keeps the standard EP-style global state
        # (identical across agents) used by vanilla MAPPO.
        self.causal_critic_state = bool(self.args.get("causal_critic_state", False))
        # Per-episode remaining budgets (initialised on reset()).
        self._remaining_budget_obs = self.budget_observation
        self._remaining_budget_act = self.budget_action

        # --- early-training budget regularization ---------------------------
        # Penalise budget spending, strongly at the start of training and then
        # annealed away, so the attackers do not blow their whole (sparse)
        # budget in the first few (random) steps and instead learn to time their
        # attacks. The penalty is subtracted from the attacker reward:
        #   penalty = reg_coef(t) * (spent_obs/budget_obs + spent_act/budget_act)
        # reg_coef anneals linearly from *_start to *_end over *_anneal_steps,
        # counted in per-environment steps (each parallel worker anneals on its
        # own step counter).
        self.budget_reg_coef_start = float(
            self.args.get("budget_reg_coef_start", 10.0)
        )
        self.budget_reg_coef_end = float(self.args.get("budget_reg_coef_end", 0.0))
        self.budget_reg_anneal_steps = int(
            self.args.get("budget_reg_anneal_steps", 200000)
        )
        self._global_step = 0

        attack_observation = bool(self.args.get("attack_observation", True))
        attack_action = bool(self.args.get("attack_action", True))
        self.agent_roles = []
        if attack_observation:
            self.agent_roles.append(OBSERVATION_ATTACKER)
        if attack_action:
            self.agent_roles.append(ACTION_ATTACKER)
        if not self.agent_roles:
            raise ValueError(
                "At least one of attack_observation / attack_action must be True."
            )
        self.n_agents = len(self.agent_roles)

        # --- per-agent true action spaces and padded reported spaces ---------
        self.true_action_space = []
        for role in self.agent_roles:
            if role == OBSERVATION_ATTACKER:
                self.true_action_space.append(
                    Box(
                        low=-self.epsilon_observation,
                        high=self.epsilon_observation,
                        shape=(self.obs_dim,),
                        dtype=np.float32,
                    )
                )
            else:  # ACTION_ATTACKER (absolute L_inf bound epsilon_action)
                self.true_action_space.append(
                    Box(
                        low=-self.epsilon_action,
                        high=self.epsilon_action,
                        shape=(self.act_dim,),
                        dtype=np.float32,
                    )
                )

        # Pad every reported action space to the same dimension so the HARL
        # on-policy runner can stack heterogeneous attacker actions. The number
        # of *valid* (non-padding) dimensions per agent is recorded on the Box
        # via ``valid_action_dim`` so a stage-aware runner can restrict the
        # follower's PPO objective to its true action dimensions.
        self.pad_dim = max(self.obs_dim, self.act_dim)
        budget_max = float(max(self.epsilon_observation, self.epsilon_action))
        self.action_space = []
        for true_space in self.true_action_space:
            padded = Box(
                low=-budget_max,
                high=budget_max,
                shape=(self.pad_dim,),
                dtype=np.float32,
            )
            padded.valid_action_dim = int(true_space.shape[0])
            self.action_space.append(padded)

        # --- augmented observation ------------------------------------------
        # Each attacker observes:
        #   [ true victim state (obs_dim),
        #     victim's nominal action (act_dim) -> lets the action attacker
        #         react to what the victim is about to do,
        #     its OWN remaining attack budget (normalised, 1 scalar) ].
        # An attacker only sees its own budget, never the other attacker's, so
        # the per-agent observation is built individually in reset()/step().
        self.aug_obs_dim = self.obs_dim + self.act_dim + 1
        aug_obs_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.aug_obs_dim,),
            dtype=np.float32,
        )
        self.observation_space = [aug_obs_space for _ in range(self.n_agents)]

        # --- centralized critic state (share_obs) ----------------------------
        # Each *actor* only ever sees its own restricted local observation
        # above. The MAPPO critic gets exactly two things:
        #   [ clean victim observation   (obs_dim) = s (no obs attack),
        #     victim.act(s + delta_o)     (act_dim) = action induced by the
        #                                             obs-corrupted observation ]
        # plus a 2-dim stage one-hot id [obs_stage, act_stage] so the critic can
        # cleanly tell the obs stage (x^o = [s, 0, 1, 0]) from the act stage
        # (x^a = [s, victim.act, 0, 1]) instead of inferring it from the masked
        # action slot (a zero victim action would otherwise be ambiguous).
        # Shared identically across agents only in the non-causal (EP) case.
        self.share_obs_dim = (
            self.obs_dim          # clean victim observation s
            + self.act_dim        # victim.act(s + delta_o)
            + 2                   # stage one-hot id [obs_stage, act_stage]
        )
        share_obs_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.share_obs_dim,),
            dtype=np.float32,
        )
        self.share_observation_space = [
            share_obs_space for _ in range(self.n_agents)
        ]
        self.discrete = False

        # --- victim policy (built on the TRUE victim spaces) -----------------
        self.victim = make_victim_policy(
            self.args.get("victim", {"type": "random"}),
            victim_obs_space,
            victim_act_space,
        )

        # Buffers populated on reset()/step() for the augmented observation.
        self._cur_state = None
        self._cur_victim_action = np.zeros(self.act_dim, dtype=np.float32)
        # AR rollout: set by begin_step() to hold the leader's committed
        # observation attack so the following step() reuses it instead of
        # re-applying the attack. None for standard (simultaneous) rollout.
        self._committed = None
        # Most recent observation perturbation actually applied by the
        # observation attacker. The action attacker only ever sees the victim
        # state *after* this perturbation (it never observes the clean state),
        # so it is stored here and reused to corrupt the next observation.
        self._last_delta_o = np.zeros(self.obs_dim, dtype=np.float32)
        # Most recent action perturbation actually applied by the action
        # attacker. Only exposed to the centralized critic (global state), never
        # to either actor's local observation.
        self._last_delta_a = np.zeros(self.act_dim, dtype=np.float32)

        # --- robust_gymnasium step config ------------------------------------
        self._robust_config = _build_disabled_robust_config(
            self.args.get("background_noise")
        )
        self._robust_type = self.args.get("robust_type", "disable")

        self._seed = self.args.get("seed", None)

    # ------------------------------------------------------------------ utils
    def _agent_index(self, role):
        return self.agent_roles.index(role) if role in self.agent_roles else None

    @staticmethod
    def _parse_budget(value):
        """Normalise a budget config value to a float (inf == unlimited)."""
        if value is None:
            return float("inf")
        value = float(value)
        if value < 0.0:
            raise ValueError(f"attack budget must be non-negative, got {value}.")
        return value

    def _apply_deadzone(self, delta):
        """Zero a perturbation that is below the attack threshold (no attack).

        A perturbation whose L_inf magnitude is below ``attack_threshold`` counts
        as "do nothing": it is not applied and -- via ``_consume_budget`` --
        costs no budget. This lets the attacker abstain for free and learn to
        attack only at key moments instead of dribbling budget away every step.
        """
        if self.attack_threshold <= 0.0:
            return delta
        if delta.size == 0 or float(np.max(np.abs(delta))) < self.attack_threshold:
            return np.zeros_like(delta)
        return delta

    def _perturbation_cost(self, delta):
        """Cost charged against the budget for a single perturbation."""
        if self.budget_cost == "count":
            return 1.0 if np.any(delta != 0.0) else 0.0
        if self.budget_cost == "l1":
            return float(np.sum(np.abs(delta)))
        if self.budget_cost == "linf":
            return float(np.max(np.abs(delta))) if delta.size else 0.0
        return float(np.linalg.norm(delta))  # l2 (default)

    def _consume_budget(self, delta, remaining):
        """Charge ``delta`` against ``remaining`` budget.

        Returns ``(effective_delta, new_remaining, spent)``. When the remaining
        budget cannot cover the requested perturbation it is either scaled down
        (norm-based costs) so it spends exactly what is left, or dropped to zero
        (count-based cost), after which the attacker can no longer attack.
        """
        if not np.isfinite(remaining):
            return delta, remaining, 0.0  # unlimited budget
        if remaining <= 0.0:
            return np.zeros_like(delta), 0.0, 0.0  # budget exhausted
        if self.budget_cost == "count":
            if np.any(delta != 0.0):
                if remaining >= 1.0:
                    return delta, remaining - 1.0, 1.0
                return np.zeros_like(delta), remaining, 0.0  # cannot afford
            return delta, remaining, 0.0
        cost = self._perturbation_cost(delta)
        if cost <= 0.0:
            return delta, remaining, 0.0
        if cost > remaining:
            factor = remaining / cost
            return delta * factor, 0.0, remaining  # partial final attack
        return delta, remaining - cost, cost

    def _normalized_remaining(self, remaining, total):
        """Remaining budget as a fraction in [0, 1] (1.0 when unlimited)."""
        if not np.isfinite(total) or total <= 0.0:
            return 1.0
        return float(np.clip(remaining / total, 0.0, 1.0))

    def _reg_coef(self):
        """Current (annealed) budget-spending regularization coefficient."""
        if self.budget_reg_anneal_steps <= 0:
            return self.budget_reg_coef_end
        frac = min(1.0, self._global_step / float(self.budget_reg_anneal_steps))
        return (
            self.budget_reg_coef_start
            + (self.budget_reg_coef_end - self.budget_reg_coef_start) * frac
        )

    def _nominal_victim_action(self, state):
        """Victim's clean action at ``state`` without mutating its RNN state."""
        rnn_backup = getattr(self.victim, "_rnn_states", None)
        action = np.asarray(self.victim.act(state), dtype=np.float32).reshape(-1)
        if rnn_backup is not None:
            # Restore so this "peek" does not advance a recurrent victim.
            self.victim._rnn_states = rnn_backup
        return action

    def _agent_remaining_frac(self, role):
        """Normalised remaining budget for a single attacker role."""
        if role == OBSERVATION_ATTACKER:
            return self._normalized_remaining(
                self._remaining_budget_obs, self.budget_observation
            )
        return self._normalized_remaining(
            self._remaining_budget_act, self.budget_action
        )

    def _build_obs(self, state, victim_action, remaining_frac):
        """Assemble one attacker's augmented observation vector.

        ``remaining_frac`` is *this* attacker's own normalised remaining budget;
        an attacker never observes the other attacker's budget.
        """
        return np.concatenate(
            [
                np.asarray(state, dtype=np.float32).reshape(-1),
                np.asarray(victim_action, dtype=np.float32).reshape(-1),
                np.array([remaining_frac], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _build_obs_all(self, state, victim_action, act_view_action_override=None):
        """Build the per-agent observation list.

        ``act_view_action_override``: when given (AR rollout), the action
        attacker's victim-action view is set to this CURRENT-step victim action
        instead of the one-step-stale recompute from ``_last_delta_o``.

        The two attackers see *different*, strictly non-overlapping views:

        * The observation attacker is **blind to the victim action**: it only
          sees the clean victim state and its own remaining budget. It corrupts
          the observation *before* the victim acts, so it conditions purely on
          the state -- the victim-action slot of its observation is zero-masked.
        * The action attacker is **blind to the state**: it only observes the
          victim's action (computed on the obs-corrupted observation
          ``s~ = s + delta_o``) and its own remaining budget. The state slot of
          its observation is zero-masked so it never sees the (clean or
          corrupted) victim state -- only the victim action it is about to
          perturb.

        So observation = state info (obs-attacker) XOR action info (act-attacker);
        each agent sees only its own dimension. ``state`` is the clean victim
        state and ``victim_action`` the victim's nominal action on it (used only
        for the centralized critic's global state, not for the obs-attacker).
        The observation dimension stays homogeneous across agents so the HARL
        runner can stack observations.

        Returns ``(obs_list, share_obs_list)`` where ``share_obs_list`` holds the
        per-agent centralized-critic state. By default every agent gets the same
        global vector (state_type "EP"); with ``causal_critic_state`` (MA-MAPPO)
        each agent's state is conditioned on the information realized before it
        acts in the sequential attack order (see ``share_obs_dim``).
        """
        clean_state = np.asarray(state, dtype=np.float32).reshape(-1)
        # Victim action on the obs-corrupted state (act-attacker's view). This
        # is also fed to the centralized critic so it sees the obs-attacker's
        # effect. Computed once and reused for both.
        corrupted = clean_state + self._last_delta_o
        if self.clip_observation:
            corrupted = np.clip(corrupted, self._obs_low, self._obs_high)
        if act_view_action_override is not None:
            # AR rollout: use the current-step victim action committed by
            # begin_step (avoids a redundant victim query and guarantees the
            # follower observes exactly the victim action that will execute).
            act_view_action = np.asarray(
                act_view_action_override, dtype=np.float32
            ).reshape(-1)
        else:
            act_view_action = self._nominal_victim_action(corrupted)

        obs_list = []
        for role in self.agent_roles:
            if role == ACTION_ATTACKER:
                # The action attacker sees the victim action. Its state slot is
                # zero-masked by default, unless ``act_observe_state`` is set, in
                # which case it also sees the victim's observation (s~).
                view_state = (
                    corrupted
                    if self.act_observe_state
                    else np.zeros(self.obs_dim, dtype=np.float32)
                )
                view_action = act_view_action
            else:
                # The observation attacker only sees the state, NOT the victim
                # action -> zero-mask the action slot.
                view_state = clean_state
                view_action = np.zeros(self.act_dim, dtype=np.float32)
            obs_list.append(
                self._build_obs(
                    view_state, view_action, self._agent_remaining_frac(role)
                )
            )

        # Global critic state: the clean victim observation (s, without the
        # obs-attacker's perturbation) and the victim action induced by the
        # obs-corrupted observation (victim.act(s + delta_o)). Just these two.
        full_share_obs = np.concatenate(
            [
                clean_state,      # victim observation without obs attack
                act_view_action,  # victim.act(s + delta_o)
            ]
        ).astype(np.float32)
        if self.causal_critic_state:
            # Per-agent, causally-ordered critic state (MA-MAPPO / Stage-Aware).
            # The leader (acts first) only conditions on the victim observation;
            # the follower additionally conditions on the executed victim action.
            # A trailing stage one-hot id makes the two stages unambiguous.
            share_obs_list = []
            for role in self.agent_roles:
                z = full_share_obs.copy()
                if role == OBSERVATION_ATTACKER:
                    z[self.obs_dim : self.obs_dim + self.act_dim] = 0.0
                    stage_id = np.array([1.0, 0.0], dtype=np.float32)  # obs stage
                else:
                    stage_id = np.array([0.0, 1.0], dtype=np.float32)  # act stage
                share_obs_list.append(
                    np.concatenate([z, stage_id]).astype(np.float32)
                )
        else:
            # Standard EP-style global state: identical across agents. No stage
            # distinction here, so the stage id is left all-zero.
            stage_id = np.zeros(2, dtype=np.float32)
            share_obs_list = [
                np.concatenate([full_share_obs, stage_id]).astype(np.float32)
                for _ in range(self.n_agents)
            ]
        return obs_list, share_obs_list

    def _victim_step(self, action):
        """Step the underlying robust_gymnasium env with a (possibly attacked) action."""
        if self._maze_api:
            # Maze tasks take a plain action and ignore the robust_input dict.
            return self.env.step(np.asarray(action, dtype=np.float32))
        robust_input = {
            "action": np.asarray(action, dtype=np.float32),
            "robust_type": self._robust_type,
            "robust_config": self._robust_config,
        }
        return self.env.step(robust_input)

    def _flatten_obs(self, obs):
        if self._goal_conditioned:
            return np.concatenate(
                [
                    np.asarray(obs["observation"], dtype=np.float32).reshape(-1),
                    np.asarray(obs["desired_goal"], dtype=np.float32).reshape(-1),
                ]
            )
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    # ----------------------------------------------------------------- API
    def _apply_obs_attack(self, leader_padded):
        """Apply the observation attack (leader) and query the victim.

        Returns ``(victim_obs, spent_obs, victim_action)`` and updates
        ``_last_delta_o`` and the observation attacker's budget. ``leader_padded``
        is the observation attacker's padded action vector, or None when there
        is no observation attacker.
        """
        victim_obs = self._cur_state.copy()
        spent_obs = 0.0
        if leader_padded is not None:
            delta_o = np.asarray(leader_padded, dtype=np.float32)[: self.obs_dim]
            # Magnitude limit: fixed absolute L_inf bound (independent of budget).
            delta_o = np.clip(
                delta_o, -self.epsilon_observation, self.epsilon_observation
            )
            # Deadzone gate: a sub-threshold perturbation counts as "no attack".
            delta_o = self._apply_deadzone(delta_o)
            # Charge the observation attacker's independent budget; once spent
            # the perturbation is forced to zero (no more attacks).
            delta_o, self._remaining_budget_obs, spent_obs = self._consume_budget(
                delta_o, self._remaining_budget_obs
            )
            victim_obs = victim_obs + delta_o
            if self.clip_observation:
                victim_obs = np.clip(victim_obs, self._obs_low, self._obs_high)
            # Remember the perturbation actually applied so the action attacker
            # only ever sees the obs-corrupted state (never the clean one).
            self._last_delta_o = np.asarray(delta_o, dtype=np.float32).reshape(-1)
        victim_action = np.asarray(
            self.victim.act(victim_obs), dtype=np.float32
        ).reshape(-1)
        return victim_obs, spent_obs, victim_action

    def probe_act_views(self, deltas):
        """Logging-only probe: victim act-views for K candidate obs-attacks.

        Given ``deltas`` of shape ``(K, pad_dim)`` (candidate leader / obs-attacker
        padded actions), return the victim actions
        ``victim.act(s + clip/deadzone(delta_o))`` at the CURRENT clean state
        ``s = self._cur_state``, for each candidate, WITHOUT mutating any env
        state: the per-episode budget is NOT charged (read-only), and
        ``_committed`` / ``_last_delta_o`` / ``_cur_victim_action`` are left
        untouched. The victim's recurrent state (if any) is snapshotted and
        restored so a recurrent victim is unaffected. Used by the stage-aware
        runner's obs-action value-spread diagnostic.

        Returns ``(clean_state (obs_dim,), act_views (K, act_dim))``.
        """
        deltas = np.asarray(deltas, dtype=np.float32)
        if deltas.ndim == 1:
            deltas = deltas[None, :]
        clean_state = self._cur_state.copy()
        victim_rnn = getattr(self.victim, "_rnn_states", None)
        if victim_rnn is not None:
            victim_rnn = np.array(victim_rnn, copy=True)
        act_views = np.zeros((deltas.shape[0], self.act_dim), dtype=np.float32)
        for k in range(deltas.shape[0]):
            delta_o = np.asarray(deltas[k], dtype=np.float32)[: self.obs_dim]
            delta_o = np.clip(
                delta_o, -self.epsilon_observation, self.epsilon_observation
            )
            delta_o = self._apply_deadzone(delta_o)
            victim_obs = clean_state + delta_o
            if self.clip_observation:
                victim_obs = np.clip(victim_obs, self._obs_low, self._obs_high)
            act_views[k] = np.asarray(
                self.victim.act(victim_obs), dtype=np.float32
            ).reshape(-1)
        if victim_rnn is not None:
            self.victim._rnn_states = victim_rnn
        return clean_state, act_views

    def begin_step(self, leader_action):
        """AR rollout phase 1: commit the leader (observation) attack.

        Applies the observation attacker's perturbation, queries the victim, and
        returns per-agent ``(obs_list, share_obs_list)`` whose action-attacker
        (follower) view holds the CURRENT-step victim action. The follower actor
        is queried on this observation; ``step`` is then called with both actions
        and reuses the committed result (it does not re-apply the obs attack).
        """
        leader_action = np.asarray(leader_action, dtype=np.float32).reshape(-1)
        victim_obs, spent_obs, victim_action = self._apply_obs_attack(leader_action)
        self._committed = {
            "victim_obs": victim_obs,
            "spent_obs": spent_obs,
            "victim_action": victim_action,
        }
        obs_list, share_obs_list = self._build_obs_all(
            self._cur_state,
            self._cur_victim_action,
            act_view_action_override=victim_action,
        )
        return obs_list, share_obs_list

    def step(self, actions):
        """Apply attacker perturbations, advance the victim, return MARL tuple.

        Args:
            actions: (np.ndarray) shape (n_agents, pad_dim); padded attacker
                perturbations produced by the HARL actors.
        Returns:
            obs_n, share_obs_n, rewards_n, dones_n, infos_n, available_actions
        """
        actions = np.asarray(actions, dtype=np.float32)
        state = self._cur_state  # true (un-attacked) victim state

        if self._committed is not None:
            # AR rollout: the observation attack was already applied by
            # begin_step (leader committed first); reuse its results so the
            # victim action the follower observed equals the one executed.
            victim_obs = self._committed["victim_obs"]
            spent_obs = self._committed["spent_obs"]
            victim_action = self._committed["victim_action"]
            self._committed = None
        else:
            # --- observation attack + victim acts (standard simultaneous) ----
            obs_idx = self._agent_index(OBSERVATION_ATTACKER)
            leader_padded = actions[obs_idx] if obs_idx is not None else None
            victim_obs, spent_obs, victim_action = self._apply_obs_attack(
                leader_padded
            )

        # --- action attack ---------------------------------------------------
        applied_action = victim_action.copy()
        act_idx = self._agent_index(ACTION_ATTACKER)
        spent_act = 0.0
        if act_idx is not None:
            # Magnitude limit: fixed absolute L_inf bound (independent of budget).
            delta_a = np.clip(
                actions[act_idx][: self.act_dim],
                -self.epsilon_action,
                self.epsilon_action,
            )
            # Deadzone gate: a sub-threshold perturbation counts as "no attack"
            # (zeroed, costs no budget) so the attacker can abstain for free.
            delta_a = self._apply_deadzone(delta_a)
            # Charge the action attacker's independent budget.
            delta_a, self._remaining_budget_act, spent_act = self._consume_budget(
                delta_a, self._remaining_budget_act
            )
            applied_action = applied_action + delta_a
            # Remember the perturbation actually applied (exposed to the
            # centralized critic via the global share_obs, never to the actors).
            self._last_delta_a = np.asarray(delta_a, dtype=np.float32).reshape(-1)
        applied_action = np.clip(
            applied_action, self.victim_act_low, self.victim_act_high
        )

        # --- environment transition -----------------------------------------
        next_obs, reward, terminated, truncated, info = self._victim_step(
            applied_action
        )
        next_state = self._flatten_obs(next_obs)
        self._cur_state = next_state
        # Victim's nominal action at the next (clean) state -> exposed to the
        # action attacker via the augmented observation.
        self._cur_victim_action = self._nominal_victim_action(next_state)

        done = bool(terminated or truncated)
        if done and truncated and not terminated:
            info = dict(info)
            info["bad_transition"] = True

        # Cooperative attacker reward = negative victim reward, minus an
        # annealed regularization penalty discouraging early budget dumping.
        victim_reward = float(reward)
        reg_coef = self._reg_coef()
        frac_obs = (
            spent_obs / self.budget_observation
            if np.isfinite(self.budget_observation) and self.budget_observation > 0
            else 0.0
        )
        frac_act = (
            spent_act / self.budget_action
            if np.isfinite(self.budget_action) and self.budget_action > 0
            else 0.0
        )
        penalty = reg_coef * (frac_obs + frac_act)
        attacker_reward = -victim_reward * self.reward_scale - penalty
        self._global_step += 1

        info = dict(info)
        info["victim_reward"] = victim_reward
        info["attacker_reward"] = attacker_reward
        info["spent_observation"] = float(spent_obs)
        info["spent_action"] = float(spent_act)
        info["budget_penalty"] = float(penalty)
        info["reg_coef"] = float(reg_coef)
        info["remaining_budget_observation"] = float(self._remaining_budget_obs)
        info["remaining_budget_action"] = float(self._remaining_budget_act)
        info["remaining_budget_observation_frac"] = self._normalized_remaining(
            self._remaining_budget_obs, self.budget_observation
        )
        info["remaining_budget_action_frac"] = self._normalized_remaining(
            self._remaining_budget_act, self.budget_action
        )

        obs, share_obs_list = self._build_obs_all(next_state, self._cur_victim_action)
        obs_n = obs
        share_obs_n = share_obs_list
        rewards_n = [[attacker_reward] for _ in range(self.n_agents)]
        dones_n = [done for _ in range(self.n_agents)]
        infos_n = [info for _ in range(self.n_agents)]
        return (
            obs_n,
            share_obs_n,
            rewards_n,
            dones_n,
            infos_n,
            self.get_avail_actions(),
        )

    def reset(self):
        """Reset victim env and victim policy, return initial MARL observations."""
        if self._seed is not None:
            obs, _ = self.env.reset(seed=self._seed)
            self._seed = None  # only seed the first reset explicitly
        else:
            obs, _ = self.env.reset()
        obs = self._flatten_obs(obs)
        self._cur_state = obs
        self.victim.reset()
        # Clear any pending AR commit from a previous episode.
        self._committed = None
        # Replenish each attacker's independent budget for the new episode.
        self._remaining_budget_obs = self.budget_observation
        self._remaining_budget_act = self.budget_action
        # No observation perturbation has been applied yet this episode.
        self._last_delta_o = np.zeros(self.obs_dim, dtype=np.float32)
        self._last_delta_a = np.zeros(self.act_dim, dtype=np.float32)
        # Victim's nominal action at the initial (clean) state.
        self._cur_victim_action = self._nominal_victim_action(obs)
        aug, share_obs_list = self._build_obs_all(obs, self._cur_victim_action)
        obs_n = aug
        share_obs_n = share_obs_list
        return obs_n, share_obs_n, self.get_avail_actions()

    def get_avail_actions(self):
        # Continuous attacker actions -> no action masking.
        return None

    def seed(self, seed):
        self._seed = seed

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
