#!/usr/bin/env python3
"""Export wandb attack curves to local CSV so plotting no longer needs wandb.

For every matched run in each project it pulls the FULL-resolution history of
    METRIC = attack/victim_episode_rewards   vs   _step
(samples defaults to 4000 -> effectively the whole logged curve) and writes one
CSV per run:

    {OUTDIR}/{ENVNAME}/{algo}_eps{eps}_seed{seed}.csv     columns: step,value

plus a manifest:

    {OUTDIR}/{ENVNAME}/manifest.csv
        env,algo,eps,seed,stage_lambda,n_points,wandb_name,wandb_id,path

Matched algos: ippo, stage_mappo (with delta^o), stage_mappo_nodelta, stage_maddpg.
stage_* runs are only exported when stage_lambda == 0.95 (or absent -> 0.95),
matching the plotting filter.

Once exported, the plot scripts can be re-pointed at these CSVs and rebinned to
ANY number of points (50, 150, ...) offline.

Run (all three envs):
    /home/mingjun/miniconda3/envs/mujoko/bin/python \
        examples/robust_attack/export_wandb_data.py

    # or a subset / custom output dir
    PROJECTS=robust_attack_Hopper-v4 SEEDS=1,10,100,1000,10000 \
    OUTDIR=examples/robust_attack/plot_data \
        /home/mingjun/miniconda3/envs/mujoko/bin/python \
        examples/robust_attack/export_wandb_data.py
"""
import csv
import os
import re
import time

import wandb

ENT = "mingjun-fan-university-of-south-australia"
PROJECTS = os.environ.get(
    "PROJECTS",
    "robust_attack_HalfCheetah-v4,robust_attack_Ant-v4,robust_attack_Hopper-v4",
).split(",")
EPS = os.environ.get("EPS", "005,010,015,020").split(",")
SEEDS = os.environ.get("SEEDS", "1,10,100,1000,10000").split(",")
SAMPLES = int(os.environ.get("SAMPLES", 150))
METRIC = "attack/victim_episode_rewards"
OUTDIR = os.environ.get("OUTDIR", "examples/robust_attack/plot_data")

PATS = [
    ("stage_mappo_nodelta", re.compile(r"^stage_mappo_nodelta_eps(\d{3})_seed(\d+)(_rerun)?$")),
    ("stage_mappo",         re.compile(r"^stage_mappo_eps(\d{3})_seed(\d+)(_rerun)?$")),
    ("stage_maddpg",        re.compile(r"^stage_maddpg_eps(\d{3})_seed(\d+)(_rerun)?$")),
    ("ippo",                re.compile(r"^ippo_eps(\d{3})_seed(\d+)(_rerun)?$")),
]

api = wandb.Api(timeout=60)


def _retry(fn, tries=6, delay=4):
    """Call fn() with retries on transient wandb server errors (e.g. HTTP 500)."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # wandb HTTPError / CommError on flaky server
            last = e
            if i < tries - 1:
                time.sleep(delay * (i + 1))
    raise last


def stage_lambda_value(r):
    try:
        return r.config["algo_args"]["algo"].get("stage_lambda", None)
    except (KeyError, TypeError):
        return None


def stage_lambda_ok(lam):
    return True if lam is None else abs(float(lam) - 0.95) < 1e-6


def export_project(proj):
    env = proj.replace("robust_attack_", "")
    outdir = os.path.join(OUTDIR, env)
    os.makedirs(outdir, exist_ok=True)
    runs = _retry(lambda: list(api.runs(f"{ENT}/{proj}")))
    rows = []
    for r in runs:
        name = r.name or ""
        for algo, rx in PATS:
            m = rx.match(name)
            if not m:
                continue
            eps, seed = m.group(1), m.group(2)
            if eps not in EPS or seed not in SEEDS:
                break
            lam = None
            if algo.startswith("stage_"):
                lam = _retry(lambda: stage_lambda_value(r))
                if not stage_lambda_ok(lam):
                    print(f"  SKIP {name} (stage_lambda={lam})")
                    break
            h = _retry(lambda: r.history(keys=[METRIC], samples=SAMPLES, x_axis="_step"))
            h = h.dropna(subset=[METRIC])
            if len(h) == 0:
                print(f"  EMPTY {name}")
                break
            h = h.sort_values("_step")
            fname = f"{algo}_eps{eps}_seed{seed}.csv"
            path = os.path.join(outdir, fname)
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "value"])
                for step, val in zip(h["_step"].values, h[METRIC].values):
                    w.writerow([int(step), float(val)])
            rows.append({
                "env": env, "algo": algo, "eps": eps, "seed": seed,
                "stage_lambda": ("" if lam is None else lam),
                "n_points": len(h), "wandb_name": name, "wandb_id": r.id,
                "path": os.path.relpath(path, OUTDIR),
            })
            print(f"  saved {fname}  ({len(h)} pts)")
            break

    man = os.path.join(outdir, "manifest.csv")
    with open(man, "w", newline="") as f:
        cols = ["env", "algo", "eps", "seed", "stage_lambda",
                "n_points", "wandb_name", "wandb_id", "path"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in sorted(rows, key=lambda d: (d["algo"], d["eps"], d["seed"])):
            w.writerow(row)
    print(f"[{env}] {len(rows)} run(s) -> {outdir}  (manifest.csv written)")
    return len(rows)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"OUTDIR={OUTDIR}")
    print(f"projects={PROJECTS}")
    print(f"eps={EPS} seeds={SEEDS} samples={SAMPLES}")
    total = 0
    for proj in PROJECTS:
        proj = proj.strip()
        if not proj:
            continue
        print(f"=== {proj} ===")
        total += export_project(proj)
    print(f"DONE: exported {total} run(s) under {OUTDIR}")


if __name__ == "__main__":
    main()
