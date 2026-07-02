#!/usr/bin/env python3
"""Training curves (wandb-style): victim_episode_rewards vs env steps.

One figure per eps. Each method (ippo / stage_mappo with delta^o / stage_mappo
nodelta) is averaged across seeds on a common ~50-bin step axis (0..NUM_STEPS),
drawn as a mean line with a +-std band. Only ~50 sampled points per run are
pulled from wandb (samples=NBINS), so this is cheap.

Run:
    EPS="005,010,015" SEEDS="1,10,500,1000" \
    /home/mingjun/miniconda3/envs/mujoko/bin/python \
        examples/robust_attack/plot_hc_curves.py
"""
import os
import re
import wandb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

api = wandb.Api(timeout=60)
ENT = "mingjun-fan-university-of-south-australia"
PROJ = os.environ.get("PROJ", "robust_attack_HalfCheetah-v4")
ENVNAME = os.environ.get("ENVNAME", PROJ.replace("robust_attack_", ""))
BASE = float(os.environ.get("BASE", "5000"))
PREFIX = os.environ.get("PREFIX", "hc")

EPS = os.environ.get("EPS", "005,010,015").split(",")
EPSV = {"005": 0.05, "010": 0.10, "015": 0.15, "020": 0.20}
SEEDS = os.environ.get("SEEDS", "1,10,500,1000").split(",")
NBINS = int(os.environ.get("NBINS", 50))
SAMPLES = int(os.environ.get("SAMPLES", 500))
NUM_STEPS = int(os.environ.get("NUM_STEPS", 6_000_000))
METRIC = "attack/victim_episode_rewards"
OUTDIR = os.environ.get("OUTDIR", "examples/robust_attack")
TAG = os.environ.get("TAG", "")

PATS = [
    ("stage_nodelta", re.compile(r"^stage_mappo_nodelta_eps(\d{3})_seed(\d+)(_rerun)?$")),
    ("stage_maddpg",  re.compile(r"^stage_maddpg_eps(\d{3})_seed(\d+)(_rerun)?$")),
    ("stage",         re.compile(r"^stage_mappo_eps(\d{3})_seed(\d+)(_rerun)?$")),
    ("mappo",         re.compile(r"^mappo_eps(\d{3})_seed(\d+)(_rerun)?$")),
    ("ippo",          re.compile(r"^ippo_eps(\d{3})_seed(\d+)(_rerun)?$")),
]
COL = {"ippo": "#1f77b4", "stage": "#d62728", "stage_nodelta": "#2ca02c",
       "mappo": "#ff7f0e", "stage_maddpg": "#9467bd"}
LBL = {"ippo": "IPPO", "stage": "stage_mappo (with $\\delta^o$)",
       "stage_nodelta": "stage_mappo nodelta (drop $\\delta^o$)",
       "mappo": "MAPPO", "stage_maddpg": "stage_maddpg"}
# Which algorithms to draw (comma-separated). Default: 4-way comparison.
PLOT_ALGOS = os.environ.get("ALGOS", "ippo,mappo,stage,stage_maddpg").split(",")

EDGES = np.linspace(0, NUM_STEPS, NBINS + 1)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])


def stage_lambda_ok(r):
    try:
        lam = r.config["algo_args"]["algo"].get("stage_lambda", None)
    except (KeyError, TypeError):
        lam = None
    return True if lam is None else abs(float(lam) - 0.95) < 1e-6


def binned_series(r):
    """Pull a densely-sampled history and interpolate it onto CENTERS.

    Each run is sampled with SAMPLES (>> NBINS) points, sorted by step, and
    linearly interpolated onto the common CENTERS grid. Points outside the
    run's actual [min_step, max_step] range stay NaN (no extrapolation), so the
    band is only drawn where every contributing seed truly has data. This kills
    the sparse single-seed bins that produced the zero-width-band artifact.
    """
    h = r.history(keys=[METRIC], samples=SAMPLES, x_axis="_step")
    h = h.dropna(subset=[METRIC])
    if len(h) == 0:
        return None
    order = np.argsort(h["_step"].values)
    steps = h["_step"].values[order]
    vals = h[METRIC].values[order]
    out = np.interp(CENTERS, steps, vals, left=np.nan, right=np.nan)
    return out


# collect: best[(eps,algo,seed)] = (binned array, n_valid) keeping the longest run
best = {}
runs = list(api.runs(f"{ENT}/{PROJ}"))
for r in runs:
    name = r.name or ""
    for algo, rx in PATS:
        m = rx.match(name)
        if not m:
            continue
        eps, seed = m.group(1), m.group(2)
        if eps not in EPS or seed not in SEEDS:
            break
        if algo in ("stage", "stage_nodelta", "stage_maddpg") and not stage_lambda_ok(r):
            break
        s = binned_series(r)
        if s is not None:
            nv = int(np.sum(~np.isnan(s)))
            key = (eps, algo, seed)
            if key not in best or nv > best[key][1]:
                best[key] = (s, nv)
        break

# data[eps][algo] = list of per-seed binned arrays
data = {e: {a: [] for a in COL} for e in EPS}
for (eps, algo, seed), (s, _nv) in best.items():
    data[eps][algo].append(s)

# one figure per eps
for eps in EPS:
    plt.figure(figsize=(8, 5))
    any_data = False
    for algo in PLOT_ALGOS:
        series = data[eps][algo]
        if not series:
            continue
        any_data = True
        arr = np.vstack(series)                       # (n_seeds, NBINS)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        valid = ~np.isnan(mean)
        plt.plot(CENTERS[valid] / 1e6, mean[valid], color=COL[algo], lw=2,
                 label=LBL[algo])
        plt.fill_between(CENTERS[valid] / 1e6, (mean - std)[valid], (mean + std)[valid],
                         color=COL[algo], alpha=0.18, lw=0)
    if not any_data:
        plt.close()
        print(f"eps{eps}: no data, skipped")
        continue
    plt.xlabel("environment steps (millions)")
    plt.ylabel("average episode reward under attack")
    plt.title(f"{ENVNAME}  $\\epsilon$={EPSV[eps]:.2f}")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = os.path.join(OUTDIR, f"{PREFIX}_curve_eps{eps}{TAG}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("saved:", out)
print("done")
