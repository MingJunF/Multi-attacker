"""Training-curve comparison (wandb-style) for HalfCheetah-v4 attackers.

For each eps it plots attack/victim_episode_rewards over training steps for the
three methods (IPPO, stage_mappo with delta^o, stage_mappo nodelta), averaged
across seeds with a +/- std band. One figure per eps.

    EPS    env var, default "005,010,015"
    SEEDS  env var, default "1,10,500,1000"
    Only stage runs with stage_lambda==0.95 are kept.
"""
import os
import re

import matplotlib
import numpy as np
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt

api = wandb.Api(timeout=60)
ENT = "mingjun-fan-university-of-south-australia"
PROJ = "robust_attack_HalfCheetah-v4"
METRIC = "attack/victim_episode_rewards"

PATS = [
    ("stage_nodelta", re.compile(r"^stage_mappo_nodelta_eps(\d{3})_seed(\d+)(_rerun)?$")),
    ("stage",         re.compile(r"^stage_mappo_eps(\d{3})_seed(\d+)(_rerun)?$")),
    ("ippo",          re.compile(r"^ippo_eps(\d{3})_seed(\d+)(_rerun)?$")),
]
EPS = os.environ.get("EPS", "005,010,015").split(",")
EPSV = {"005": 0.05, "010": 0.10, "015": 0.15, "020": 0.20}
SEEDS = os.environ.get("SEEDS", "1,10,500,1000").split(",")
GRID = int(os.environ.get("GRID", 300))

COL = {"ippo": "#1f77b4", "stage": "#d62728", "stage_nodelta": "#2ca02c"}
LBL = {"ippo": "IPPO", "stage": "stage_mappo (with $\\delta^o$)",
       "stage_nodelta": "stage_mappo nodelta (drop $\\delta^o$)"}


def stage_lambda_ok(r):
    try:
        lam = r.config["algo_args"]["algo"].get("stage_lambda", None)
    except (KeyError, TypeError):
        lam = None
    return lam is None or abs(float(lam) - 0.95) < 1e-6


# curves[(algo, eps)] = list of (steps, vals) per seed
curves = {}
for r in api.runs(f"{ENT}/{PROJ}"):
    name = r.name or ""
    for algo, rx in PATS:
        m = rx.match(name)
        if not m:
            continue
        eps, seed = m.group(1), m.group(2)
        if eps not in EPS or seed not in SEEDS:
            break
        if algo in ("stage", "stage_nodelta") and not stage_lambda_ok(r):
            break
        h = r.history(keys=[METRIC], samples=4000)
        if METRIC not in h:
            break
        sub = h[["_step", METRIC]].dropna()
        if len(sub) < 2:
            break
        steps = sub["_step"].values.astype(float)
        vals = sub[METRIC].values.astype(float)
        key = (algo, eps)
        # keep the longest run per seed if duplicates
        curves.setdefault(key, {})
        prev = curves[key].get(seed)
        if prev is None or steps[-1] > prev[0][-1]:
            curves[key][seed] = (steps, vals)
        break


def aligned_mean_std(seed_curves):
    """Interpolate each seed onto a common grid, return (xg, mean, std, n)."""
    series = list(seed_curves.values())
    if not series:
        return None
    xmax = min(s[0][-1] for s in series)  # common coverage (no extrapolation)
    xmin = max(s[0][0] for s in series)
    if xmax <= xmin:
        return None
    xg = np.linspace(xmin, xmax, GRID)
    stack = np.vstack([np.interp(xg, s[0], s[1]) for s in series])
    return xg, stack.mean(0), stack.std(0, ddof=1) if len(series) > 1 else stack.std(0), len(series)


for eps in EPS:
    plt.figure(figsize=(7.5, 5))
    for algo in ["ippo", "stage", "stage_nodelta"]:
        sc = curves.get((algo, eps))
        if not sc:
            continue
        res = aligned_mean_std(sc)
        if res is None:
            continue
        xg, mean, std, n = res
        plt.plot(xg, mean, color=COL[algo], lw=2, label=f"{LBL[algo]} (n={n})")
        plt.fill_between(xg, mean - std, mean + std, color=COL[algo], alpha=0.18)
    plt.axhline(5000, ls="--", color="gray", lw=1, alpha=0.7)
    plt.text(0.01, 5000, "no-attack baseline ~5000", color="gray",
             va="bottom", ha="left", fontsize=9, transform=plt.gca().get_yaxis_transform())
    plt.xlabel("training step")
    plt.ylabel("victim episode reward\nlower = stronger attack")
    plt.title(f"HalfCheetah-v4, $\\epsilon$={EPSV[eps]:.2f}: training curves "
              f"(mean $\\pm$ std over seeds {{{','.join(SEEDS)}}})")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = f"examples/robust_attack/hc_training_curve_eps{eps}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("saved:", out)
