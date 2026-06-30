import wandb, numpy as np
api = wandb.Api(timeout=60)
ENT = "mingjun-fan-university-of-south-australia"
PROJ = "robust_attack_HalfCheetah-v4"
runs = list(api.runs(f"{ENT}/{PROJ}"))


def last20(r):
    h = r.history(keys=["attack/victim_episode_rewards"], samples=3000)
    v = h["attack/victim_episode_rewards"].dropna().values
    return (float(np.mean(v[-20:])), len(v)) if len(v) else (None, 0)


def digcfg(cfg, *keys):
    # config may be nested; search recursively for first matching key
    found = {}
    def rec(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k in keys and k not in found:
                    found[k] = v
                rec(v)
    rec(cfg)
    return found


targets = [
    "stage_mappo_nodelta_seed500",          # yesterday manual
    "stage_mappo_nodelta_eps010_seed500",   # campaign
    "stage_mappo_nodelta_eps010_seed1",
    "stage_mappo_nodelta_eps010_seed10",
    "stage_mappo_nodelta_eps010_seed1000",
    "stage_mappo_eps010_seed500",           # with-delta reference
]
for r in runs:
    if r.name in targets:
        val, n = last20(r)
        cfg = digcfg(r.config, "stage_lambda", "stage_drop_delta_o",
                     "epsilon_observation", "causal_critic_state", "state_type", "seed")
        print(f"{r.name:42} victimR={val:8.1f} n={n:4}  group={r.group}")
        print(f"    cfg: {cfg}")
