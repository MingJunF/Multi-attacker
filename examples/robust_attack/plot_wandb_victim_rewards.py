#!/usr/bin/env python3
"""Download `attack/victim_episode_rewards` from wandb and plot it.

Groups the robust_attack HalfCheetah runs by epsilon (one subplot per eps) and,
within each eps, draws one mean curve per algorithm (ippo vs stage_mappo)
aggregated over seeds with a shaded uncertainty band.

Run naming assumed (set by examples/robust_attack/run_server.py):
    wandb_name = "{algo}_eps{tag}_seed{seed}"   e.g. "stage_mappo_eps010_seed1"
    metric     = "attack/victim_episode_rewards" logged at step = env steps

Usage
-----
    wandb login                      # or export WANDB_API_KEY=...
    python examples/robust_attack/plot_wandb_victim_rewards.py
    # options:
    python examples/robust_attack/plot_wandb_victim_rewards.py \
        --entity mingjun-fan-university-of-south-australia \
        --project robust_attack_HalfCheetah-v4 \
        --band ci95 --grid 200 --out victim_rewards.png

Bands
-----
    --band ci95   mean +/- 1.96 * std / sqrt(n_seeds)   (95% CI of the mean, default)
    --band std    mean +/- 1.0  * std
    --band 95std  mean +/- 1.96 * std                   (~95% spread across seeds)
"""
import argparse
import re
from collections import defaultdict

import numpy as np


RUN_NAME_RE = re.compile(
    r"(?P<algo>ippo|stage_mappo)_eps(?P<eps>\d+)_seed(?P<seed>\d+)"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default="mingjun-fan-university-of-south-australia")
    p.add_argument("--project", default="robust_attack_HalfCheetah-v4")
    p.add_argument("--metric", default="attack/victim_episode_rewards")
    p.add_argument("--xkey", default="_step", help="x-axis key (env steps).")
    p.add_argument(
        "--band",
        default="ci95",
        choices=["ci95", "std", "95std"],
        help="Shaded band: 95%% CI of the mean / 1 std / 1.96 std.",
    )
    p.add_argument("--grid", type=int, default=200, help="Common-grid resolution.")
    p.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Points per run pulled via run.history (fast sampled download).",
    )
    p.add_argument(
        "--algos",
        nargs="+",
        default=["ippo", "stage_mappo"],
        help="Algorithms to plot (one colour each).",
    )
    p.add_argument(
        "--exclude-seeds",
        nargs="*",
        type=int,
        default=[],
        help="Seeds to drop from the aggregation (e.g. --exclude-seeds 100 10000).",
    )
    p.add_argument(
        "--include-seeds",
        nargs="*",
        type=int,
        default=None,
        help="If set, keep ONLY these seeds (overrides --exclude-seeds).",
    )
    p.add_argument("--out", default="victim_rewards.png")
    p.add_argument(
        "--csv", default=None, help="Optional path to dump the aggregated curves."
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Optional moving-average window (in grid points) for the mean.",
    )
    p.add_argument(
        "--per-seed",
        action="store_true",
        help="Plot one line PER SEED (no band) instead of the seed-aggregated mean.",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Print a table of each run's final-segment mean victim return.",
    )
    p.add_argument(
        "--tail-frac",
        type=float,
        default=0.1,
        help="Fraction of the END of each curve used for the summary mean.",
    )
    return p.parse_args()


def fetch_runs(entity, project, metric, xkey, samples):
    """Return {(eps, algo, seed): (steps, values)} keeping the newest duplicate."""
    import wandb

    api = wandb.Api(timeout=60)
    runs = api.runs(f"{entity}/{project}")
    print(f"[wandb] found {len(runs)} runs in {entity}/{project}")

    # newest run wins for a duplicated (eps, algo, seed)
    best = {}  # key -> (created_at, run)
    for run in runs:
        m = RUN_NAME_RE.search(run.name or "")
        if not m:
            # fall back to config if the name does not match
            cfg = run.config or {}
            algo = (cfg.get("args", {}) or {}).get("algo")
            seed = (cfg.get("args", {}) or {}).get("seed")
            eps = (cfg.get("env_args", {}) or {}).get("epsilon_observation")
            if algo is None or seed is None or eps is None:
                continue
            eps_tag = f"{int(round(float(eps) * 100)):03d}"
            key = (eps_tag, algo, int(seed))
        else:
            key = (m.group("eps"), m.group("algo"), int(m.group("seed")))
        created = getattr(run, "created_at", "") or ""
        if key not in best or created > best[key][0]:
            best[key] = (created, run)

    out = {}
    for key, (_, run) in sorted(best.items()):
        df = run.history(keys=[metric], x_axis=xkey, samples=samples, pandas=True)
        if df is None or df.empty or metric not in df or xkey not in df:
            print(f"[skip] {key} has no {metric}")
            continue
        df = df[[xkey, metric]].dropna()
        if len(df) < 2:
            print(f"[skip] {key} has <2 points")
            continue
        steps = df[xkey].to_numpy(dtype=float)
        vals = df[metric].to_numpy(dtype=float)
        order = np.argsort(steps)
        out[key] = (steps[order], vals[order])
        print(f"[ok]   eps{key[0]} {key[1]:<11} seed{key[2]:<6} {len(df)} pts")
    return out


def aggregate(curves, grid_n):
    """curves: list of (steps, vals). Returns (grid, mean, std, n)."""
    lo = max(s[0] for s, _ in curves)
    hi = min(s[-1] for s, _ in curves)
    if hi <= lo:  # no common overlap -> use the union range instead
        lo = min(s[0] for s, _ in curves)
        hi = max(s[-1] for s, _ in curves)
    grid = np.linspace(lo, hi, grid_n)
    stacked = np.vstack([np.interp(grid, s, v) for s, v in curves])
    return grid, stacked.mean(0), stacked.std(0), stacked.shape[0]


def band_halfwidth(std, n, mode):
    if mode == "std":
        return std
    if mode == "95std":
        return 1.96 * std
    return 1.96 * std / max(np.sqrt(n), 1.0)  # ci95


def moving_average(x, w):
    if w and w > 1:
        k = np.ones(w) / w
        return np.convolve(x, k, mode="same")
    return x


def main():
    args = parse_args()
    import matplotlib.pyplot as plt

    data = fetch_runs(args.entity, args.project, args.metric, args.xkey, args.samples)
    if not data:
        raise SystemExit("No matching runs / metric found.")

    # group by eps -> algo -> {seed: (steps, vals)}
    grouped = defaultdict(lambda: defaultdict(dict))
    for (eps, algo, seed), curve in data.items():
        if algo not in args.algos:
            continue
        if args.include_seeds is not None:
            if seed not in args.include_seeds:
                continue
        elif seed in args.exclude_seeds:
            continue
        grouped[eps][algo][seed] = curve

    eps_list = sorted(grouped.keys(), key=lambda t: int(t))

    # --- optional summary table: final-segment mean per (eps, algo, seed) -----
    if args.summary:
        def tail_mean(curve):
            v = curve[1]
            k = max(1, int(len(v) * args.tail_frac))
            return float(np.mean(v[-k:]))

        print(
            f"\n=== final {int(args.tail_frac*100)}% mean victim return "
            f"(lower = stronger attack) ==="
        )
        for eps in eps_list:
            print(f"\n  eps = 0.{eps}")
            seeds = sorted(
                {s for a in args.algos for s in grouped[eps].get(a, {})}
            )
            header = "    seed   " + "".join(f"{a:>16}" for a in args.algos)
            print(header)
            for s in seeds:
                cells = ""
                for a in args.algos:
                    cur = grouped[eps].get(a, {}).get(s)
                    cells += f"{tail_mean(cur):>16.1f}" if cur else f"{'--':>16}"
                print(f"    {s:<6}{cells}")
            # which seed drags stage_mappo (highest = weakest attack)
            if "stage_mappo" in grouped[eps]:
                worst = max(
                    grouped[eps]["stage_mappo"].items(),
                    key=lambda kv: tail_mean(kv[1]),
                )
                print(
                    f"    -> weakest stage_mappo seed: {worst[0]} "
                    f"({tail_mean(worst[1]):.1f})"
                )

    n = len(eps_list)
    ncol = min(n, 2)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(7 * ncol, 4.2 * nrow), squeeze=False, sharex=False
    )

    color = {"ippo": "tab:blue", "stage_mappo": "tab:red"}
    label = {"ippo": "IPPO (independent)", "stage_mappo": "Stage-Aware MAPPO"}
    seed_colors = plt.cm.tab10.colors
    algo_ls = {"ippo": "--", "stage_mappo": "-"}

    csv_rows = []
    for idx, eps in enumerate(eps_list):
        ax = axes[idx // ncol][idx % ncol]
        if args.per_seed:
            # one line per seed; colour = seed, linestyle = algo
            seeds = sorted({s for a in args.algos for s in grouped[eps].get(a, {})})
            scolor = {s: seed_colors[i % len(seed_colors)] for i, s in enumerate(seeds)}
            for algo in args.algos:
                for seed, curve in sorted(grouped[eps].get(algo, {}).items()):
                    steps, vals = curve
                    vals_s = moving_average(vals, args.smooth)
                    ax.plot(
                        steps, vals_s, color=scolor[seed], ls=algo_ls.get(algo, "-"),
                        lw=1.6, alpha=0.9,
                        label=f"{algo} seed{seed}",
                    )
        else:
            for algo in args.algos:
                curves = list(grouped[eps].get(algo, {}).values())
                if not curves:
                    continue
                grid, mean, std, m = aggregate(curves, args.grid)
                mean_s = moving_average(mean, args.smooth)
                hw = moving_average(band_halfwidth(std, m, args.band), args.smooth)
                c = color.get(algo, None)
                ax.plot(grid, mean_s, color=c, lw=2, label=f"{label.get(algo, algo)} (n={m})")
                ax.fill_between(grid, mean_s - hw, mean_s + hw, color=c, alpha=0.2, lw=0)
                if args.csv:
                    for g, mu, sd in zip(grid, mean, std):
                        csv_rows.append((eps, algo, g, mu, sd, m))
        ax.set_title(f"eps = 0.{eps}  (epsilon_obs = epsilon_act)")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Victim episode return under attack")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # hide any unused axes
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    band_txt = {"ci95": "95% CI of mean", "std": "+/-1 std", "95std": "+/-1.96 std"}
    fig.suptitle(
        f"{args.project}: victim return under attack "
        f"(shaded = {band_txt[args.band]} over seeds; lower = stronger attack)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=150)
    print(f"[saved] {args.out}")

    if args.csv and csv_rows:
        import csv as _csv

        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["eps", "algo", "step", "mean", "std", "n_seeds"])
            w.writerows(csv_rows)
        print(f"[saved] {args.csv}")


if __name__ == "__main__":
    main()
