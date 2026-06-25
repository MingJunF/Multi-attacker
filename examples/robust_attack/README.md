# Multi-Attacker (Observation + Action) on Robust-Gymnasium

A cooperative **multi-attacker** that degrades a *fixed* victim RL policy inside a
`robust_gymnasium` task. The attacker team is solved with the MARL algorithms in
`harl/` (independent PPO, MAPPO, or HAPPO).

## Threat model

The attacker is a **two-agent team** with a shared (cooperative) objective.
Both agents receive the **same augmented observation** `o_t = [s_t, a^nom_t,
b_obs, b_act]` (true state, victim's nominal action, and the two normalised
remaining budgets), so the action attacker can react to what the victim is about
to do:

| Agent | Role | Acts |
| ----- | ---- | ---- |
| 0 | **Observation attacker** | additive `δ_o`, capped *relatively* at `\|δ_o[i]\| ≤ tamper_ratio_obs·\|s_t[i]\|` (and abs cap `ε_obs`), added to the obs the victim sees |
| 1 | **Action attacker** | additive `δ_a`, capped at `\|δ_a[i]\| ≤ tamper_ratio_act·(action half-range)`, added to the victim action |

Per-step interaction (sequential flow):

```
o_t  = [s_t, a^nom_t, b_obs, b_act]   # augmented attacker observation
s̃_t = s_t + δ_o            # observation attack (tamper- & budget-clipped)
a_t  = π_victim(s̃_t)       # fixed victim acts on the corrupted observation
ã_t  = clip(a_t + δ_a)     # action attack (tamper- & budget-clipped)
s_{t+1}, r_t = env(ã_t)    # robust_gymnasium transition
r_attack = -r_t - penalty(t)   # cooperative reward + annealed early-spend penalty
```

Because both attackers share `r_attack = -r_victim`, the problem is a standard
cooperative MARL task — directly compatible with MAPPO / HAPPO. Using `mappo`
with `share_param: False` recovers **two independent PPO attackers**.

## Files

| File | Purpose |
| ---- | ------- |
| `harl/envs/robust_attack/robust_attack_env.py` | The cooperative multi-attacker environment |
| `harl/envs/robust_attack/victim_env.py` | Single-agent wrapper to train the victim PPO on the *same* robust_gymnasium task |
| `harl/envs/robust_attack/victim_policy.py` | Fixed victim policies (random / zero / HARL checkpoint / callable) |
| `harl/envs/robust_attack/robust_attack_logger.py` | Logs attacker return and the victim return under attack |
| `harl/configs/envs_cfgs/robust_attack.yaml` | Attacker environment configuration |
| `harl/configs/envs_cfgs/robust_victim.yaml` | Victim-training environment configuration |
| `examples/robust_attack/train_robust_attacker.py` | Generic training entry point (victim or attacker) |

## Observation & action spaces

For a victim task with observation dim `D_o` and action dim `D_a`
(e.g. Ant-v4: `D_o = 27`, `D_a = 8`):

Both attackers receive the augmented observation of dim `D_o + D_a + 2`
(e.g. Ant-v4: `27 + 8 + 2 = 37`):

| Component | Observation space | Action space |
| --------- | ----------------- | ------------ |
| **agent 0 — obs attacker** | `[s_t, a^nom_t, b_obs, b_act]`, `Box(D_o+D_a+2)` | perturbation `Box(-ε_obs, ε_obs, D_o)` |
| **agent 1 — act attacker** | `[s_t, a^nom_t, b_obs, b_act]`, `Box(D_o+D_a+2)` | per-dim perturbation `Box(D_a)`, bound `tamper_ratio_act·half-range` |
| **central critic** (MAPPO/HAPPO) | global state = `[s_t, a^nom_t, b_obs, b_act]`, `Box(D_o+D_a+2)` | none — it is a **V(s) value head**, outputs a scalar, no action |

Notes on the spaces:
- Both attackers **observe the same augmented vector** `[s_t, a^nom_t, b_obs,
  b_act]`: the true state, the victim's *nominal* action at the current state
  (so the action attacker can anticipate it), and the two normalised remaining
  budgets. `a^nom_t` is the victim action on the *clean* state — a one-step
  approximation, since the actually executed action is computed on the
  obs-attacked state.
- The two attackers have **heterogeneous true action dims** (`D_o` vs `D_a`).
  To let the on-policy runner batch them, every agent **reports** a zero-padded
  action space `Box(max(D_o, D_a))`; the environment slices each agent's real
  action (`true_action_space`) back out — `[(D_o,), (D_a,)]`.
- The central critic is **centralised but value-only**: it consumes the global
  state (here equal to `s_t`, dim `D_o`) and outputs `V(s)`. It never takes or
  produces actions. With `state_type: EP` it sees one shared global state.

## Full pipeline: train victim → freeze → train attacker

```bash
# ----------------------------------------------------------------------------
# Phase 1. Train a standard single-agent PPO victim on the MuJoCo task.
#   - env robust_victim wraps the SAME robust_gymnasium scenario, so the victim
#     observation/action spaces exactly match the attacker environment.
#   - one agent + centralised value critic == PPO.
#   - eval_interval 1 forces periodic checkpointing of actor_agent0.pt.
# ----------------------------------------------------------------------------
python examples/robust_attack/train_robust_attacker.py \
    --algo mappo --env robust_victim --exp_name victim \
    --scenario Ant-v4 --share_param True \
    --n_rollout_threads 20 --eval_interval 25

# The trained victim is saved to:
#   results/<logger_dir>/robust_victim/Ant-v4/mappo/victim/seed-XXXXX-<ts>/models/actor_agent0.pt

# ----------------------------------------------------------------------------
# Phase 2. Freeze the PPO policy as the victim.
#   Edit harl/configs/envs_cfgs/robust_attack.yaml -> set the victim block:
#
#     victim:
#       type: harl
#       model_path: /abs/path/to/.../models/actor_agent0.pt
#       device: cpu
#       deterministic: True
#       model_args:
#         hidden_sizes: [128, 128]   # must match the victim training config
#
#   The victim network is rebuilt from model_args and the weights are loaded;
#   its parameters are never updated during attacker training (frozen).
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Phase 3. Train the MAPPO-driven multi-attacker against the frozen victim.
# ----------------------------------------------------------------------------
python examples/robust_attack/train_robust_attacker.py \
    --algo mappo --env robust_attack --exp_name attacker \
    --scenario Ant-v4 --share_param False --n_rollout_threads 20
```

## Quick start

Run from the repository root:

```bash
# Two independent PPO attackers
python examples/robust_attack/train_robust_attacker.py \
    --algo mappo --env robust_attack --exp_name ind_ppo --share_param False

# Shared-parameter MAPPO attackers
python examples/robust_attack/train_robust_attacker.py \
    --algo mappo --env robust_attack --exp_name mappo --share_param True

# Heterogeneous-agent PPO (HAPPO) attackers
python examples/robust_attack/train_robust_attacker.py \
    --algo happo --env robust_attack --exp_name happo
```

## Configuration

Key options in `harl/configs/envs_cfgs/robust_attack.yaml` (override on the CLI):

| Key | Meaning |
| --- | ------- |
| `scenario` | robust_gymnasium victim task id (e.g. `Ant-v4`, `HalfCheetah-v4`) |
| `attack_observation` / `attack_action` | enable each attacker (both `True` ⇒ 2 agents) |
| `tamper_ratio_observation` | obs tamper limit, relative: `\|δ_o[i]\| ≤ ratio·\|s[i]\|` (default `0.5`) |
| `tamper_ratio_action` | act tamper limit: per-dim bound `= ratio·(action half-range)` (default `0.5` ⇒ ±0.5 on a [-1,1] action) |
| `epsilon_observation` | absolute safety cap + policy box for the obs perturbation (default `1.0`) |
| `epsilon_action` | optional extra absolute cap for the act perturbation (`~` = tamper ratio only) |
| `budget_observation` / `budget_action` | **independent** per-episode attack budget for each attacker (`~`/null = unlimited; default `30`) |
| `budget_cost` | how each step's spend is charged: `count` (default) / `l1` / `l2` / `linf` |
| `budget_reg_coef_start` / `budget_reg_coef_end` / `budget_reg_anneal_steps` | annealed early-spend penalty discouraging dumping the sparse budget too early |
| `clip_observation` | clip the corrupted obs back into the obs space |
| `reward_scale` | scale on the (negative victim) attacker reward |
| `victim.type` | `random` \| `zero` \| `harl` \| `callable` |
| `background_noise` | optional residual robust_gymnasium noise layered under the attack |

### Limiting attacker capability with budgets

By default each attacker can perturb the victim on **every** step (only bounded
per-step by `epsilon_*`). To make the attacker capability finite, give each
attacker its own budget that is spent whenever it attacks; once exhausted, that
attacker can no longer attack for the rest of the episode (its perturbation is
forced to zero). Budgets are replenished at every `reset()` and are completely
independent between the observation and action attackers. They only constrain
the learned MARL attackers and never touch robust_gymnasium's own perturbations.

```yaml
budget_observation: 30     # total per-episode budget for the obs attacker
budget_action: 30          # total per-episode budget for the act attacker
budget_cost: count         # count | l1 | l2 | linf
```

- `count` (default): each attack step costs `1` — a hard cap on the **number**
  of attacks per episode. A well-trained victim survives ~300 steps here, so
  `30` lets each attacker tamper on ~1/10 of the steps (the two budgets are
  independent and may both be spent on the same step).
- `l2` / `l1` / `linf`: each step spends the corresponding norm of the applied
  perturbation (an energy/magnitude budget). If the remaining budget cannot
  cover a requested perturbation it is scaled down to spend exactly what is
  left, then drops to zero.
- Because the budget is **sparse**, the env adds an **annealed early-spend
  penalty** so the initially-random attackers don't blow it all in the first
  few steps: `penalty = reg_coef(t)·(spent_obs/budget_obs + spent_act/budget_act)`,
  with `reg_coef(t)` annealing linearly from `budget_reg_coef_start` to
  `budget_reg_coef_end` over `budget_reg_anneal_steps` env steps. Set the start
  coef to `0` to disable.
- The remaining budgets (raw and normalised), per-step spend, and penalty are
  reported every step in `info["remaining_budget_observation"]`,
  `info["remaining_budget_action"]`, `info["spent_observation"]`,
  `info["spent_action"]`, and `info["budget_penalty"]`. The normalised remaining
  budgets are also part of the attacker observation.

### Attacking a trained victim

Set the victim to a trained HARL actor checkpoint:

```yaml
victim:
  type: harl
  model_path: /path/to/actor_agent0.pt
  device: cpu
  deterministic: True
  model_args:
    hidden_sizes: [128, 128]
```

## Logging (TensorBoard + optional Weights & Biases)

Both the victim and the attacker share **one standard logging switch** in the
algorithm config `logger` block (e.g. `harl/configs/algos_cfgs/mappo.yaml`,
`happo.yaml`). TensorBoard is always on; wandb is opt-in:

```yaml
logger:
  log_dir: "./results"
  use_wandb: False          # set True to enable wandb
  wandb_project: "robust-gymnasium"
  wandb_entity: ~           # your team/user (optional)
  wandb_group: ~            # group victim & attacker runs (optional)
  wandb_name: ~             # run name (defaults to env-algo-exp_name)
  wandb_mode: "online"      # online / offline / disabled
```

Enable it on the CLI for either run (victim or attacker):

```bash
# victim
python examples/robust_attack/train_robust_attacker.py \
    --algo mappo --env robust_victim --exp_name victim --scenario Ant-v4 \
    --use_wandb True --wandb_group ant_pipeline --wandb_name victim

# attacker
python examples/robust_attack/train_robust_attacker.py \
    --algo mappo --env robust_attack --exp_name attacker --scenario Ant-v4 \
    --use_wandb True --wandb_group ant_pipeline --wandb_name attacker
```

Logged metrics include `train/average_episode_rewards`, per-agent actor stats,
critic losses, and — for the attacker — `attack/victim_episode_rewards`
(the victim return under attack). Requires `pip install wandb`; if wandb is not
installed the switch silently falls back to TensorBoard only.

## Notes

- Attacker actions are **additive perturbations**, clipped to their L∞ budgets;
  the corrupted action is further clipped to the victim action bounds.
- Each attacker can be given an **independent depleting budget** (`budget_*`,
  `budget_cost`); when it runs out that attacker stops attacking for the episode.
- Heterogeneous attacker action dimensions (`obs_dim` vs `act_dim`) are zero-padded
  to `max(obs_dim, act_dim)` so the on-policy runner can batch them, then sliced
  back inside the environment (same trick as HARL's mamujoco wrapper).
- The built-in `robust_gymnasium` perturbations are **disabled by default** so the
  learned attackers are the only disturbance; re-enable them via `background_noise`.
