#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CRITIC_TYPES = ("v", "q")
CRITIC_LABELS = {
    "v": "V-based A2C",
    "q": "Q-based A2C",
}
CRITIC_COLORS = {
    "v": "#1f77b4",
    "q": "#ff7f0e",
}


def parse_args():
    default_results_dir = Path(__file__).resolve().parent / "results" / "critic_compare"
    parser = argparse.ArgumentParser(description="Plot A2C-V/A2C-Q comparison figures from saved run metrics.")
    parser.add_argument(
        "--results-dir",
        default=str(default_results_dir),
        help="directory containing per-critic per-seed results",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory to save plots; defaults to <results-dir>/plots",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=200.0,
        help="reward threshold for the sample-efficiency bar chart (default: 200)",
    )
    parser.add_argument(
        "--late-eval-count",
        type=int,
        default=10,
        help="fallback number of late evaluations per seed for the boxplot when final eval data is missing (default: 10)",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in {"critic_type", "stage"}:
                    parsed[key] = value
                elif key in {"seed", "update", "timesteps", "num_episodes"}:
                    parsed[key] = int(float(value))
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
        return rows


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_runs(results_dir: Path) -> dict[str, list[dict]]:
    runs = {critic_type: [] for critic_type in CRITIC_TYPES}
    for critic_type in CRITIC_TYPES:
        critic_dir = results_dir / critic_type
        if not critic_dir.exists():
            continue
        for seed_dir in sorted(critic_dir.glob("seed_*")):
            training_path = seed_dir / "training_metrics.csv"
            evaluations_path = seed_dir / "evaluations.csv"
            if not training_path.exists() or not evaluations_path.exists():
                continue
            run = {
                "seed_dir": seed_dir,
                "seed": int(seed_dir.name.split("_")[-1]),
                "training_metrics": read_csv_rows(training_path),
                "evaluations": read_csv_rows(evaluations_path),
                "config": read_json(seed_dir / "config.json") if (seed_dir / "config.json").exists() else None,
                "final_eval": read_json(seed_dir / "final_eval_rewards.json")
                if (seed_dir / "final_eval_rewards.json").exists()
                else None,
            }
            runs[critic_type].append(run)
    return runs


def aggregate_series(runs: list[dict], source_key: str, value_key: str, stage: str | None = None):
    bucket = {}
    for run in runs:
        rows = run[source_key]
        for row in rows:
            if stage is not None and row.get("stage") != stage:
                continue
            timestep = row["timesteps"]
            bucket.setdefault(timestep, []).append(row[value_key])

    timesteps = sorted(bucket)
    means = np.asarray([np.mean(bucket[timestep]) for timestep in timesteps], dtype=np.float64)
    stds = np.asarray([np.std(bucket[timestep]) for timestep in timesteps], dtype=np.float64)
    return np.asarray(timesteps, dtype=np.int64), means, stds


def plot_mean_with_std(ax, x, mean, std, label, color):
    if len(x) == 0:
        return
    ax.plot(x, mean, label=label, color=color, linewidth=2)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


def get_checkpoint_rows(run: dict) -> list[dict]:
    return sorted(
        [row for row in run["evaluations"] if row["stage"] == "checkpoint"],
        key=lambda row: row["timesteps"],
    )


def get_boxplot_values(run: dict, late_eval_count: int) -> list[float]:
    if run["final_eval"] is not None and run["final_eval"].get("rewards"):
        rewards = run["final_eval"]["rewards"]
        return [float(np.mean(rewards))]

    late_rows = get_checkpoint_rows(run)[-late_eval_count:]
    if not late_rows:
        return []
    return [float(np.mean([row["mean_reward"] for row in late_rows]))]


def get_time_to_threshold(run: dict, threshold: float) -> tuple[float, bool]:
    checkpoint_rows = get_checkpoint_rows(run)
    if checkpoint_rows:
        budget = checkpoint_rows[-1]["timesteps"]
    elif run["training_metrics"]:
        budget = run["training_metrics"][-1]["timesteps"]
    else:
        budget = 0

    for row in checkpoint_rows:
        if row["mean_reward"] >= threshold:
            return float(row["timesteps"]), True
    return float(budget), False


def create_output_dirs(output_dir: Path) -> dict[str, Path]:
    subdirs = {
        "learning_curve": output_dir / "learning_curves",
        "final_performance": output_dir / "final_performance",
        "threshold": output_dir / "threshold",
        "critic_loss": output_dir / "critic_loss",
        "signal_variance": output_dir / "signal_variance",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def main():
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else results_dir / "plots"
    runs_by_type = load_runs(results_dir)

    if not any(runs_by_type.values()):
        raise FileNotFoundError(f"No run data found under {results_dir}")

    plt.style.use("default")
    subdirs = create_output_dirs(output_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    for critic_type in CRITIC_TYPES:
        timesteps, means, stds = aggregate_series(
            runs_by_type[critic_type],
            source_key="evaluations",
            value_key="mean_reward",
            stage="checkpoint",
        )
        plot_mean_with_std(
            ax,
            timesteps,
            means,
            stds,
            CRITIC_LABELS[critic_type],
            CRITIC_COLORS[critic_type],
        )
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Evaluation reward")
    ax.set_title("Learning Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(subdirs["learning_curve"] / "learning_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    boxplot_data = []
    positions = []
    present_critics = []
    for index, critic_type in enumerate(CRITIC_TYPES, start=1):
        values = []
        for run in runs_by_type[critic_type]:
            values.extend(get_boxplot_values(run, args.late_eval_count))
        if not values:
            continue
        boxplot_data.append(values)
        positions.append(index)
        present_critics.append(critic_type)
    if boxplot_data:
        box = ax.boxplot(boxplot_data, positions=positions, patch_artist=True, widths=0.5)
        for patch, critic_type in zip(box["boxes"], present_critics):
            patch.set_facecolor(CRITIC_COLORS[critic_type])
            patch.set_alpha(0.35)
        for position, values, critic_type in zip(positions, boxplot_data, present_critics):
            ax.scatter(
                np.full(len(values), position),
                values,
                color=CRITIC_COLORS[critic_type],
                alpha=0.8,
                s=25,
            )
    ax.set_xticks(
        positions if positions else [1, 2],
        [CRITIC_LABELS[critic_type] for critic_type in (present_critics or CRITIC_TYPES)],
    )
    ax.set_ylabel("Final evaluation reward")
    ax.set_title("Final Performance Boxplot")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(subdirs["final_performance"] / "final_performance_boxplot.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    bar_means = []
    bar_stds = []
    reach_counts = []
    for critic_type in CRITIC_TYPES:
        values = []
        reached = 0
        for run in runs_by_type[critic_type]:
            steps, did_reach = get_time_to_threshold(run, args.threshold)
            values.append(steps)
            if did_reach:
                reached += 1
        bar_means.append(float(np.mean(values)) if values else 0.0)
        bar_stds.append(float(np.std(values)) if values else 0.0)
        reach_counts.append((reached, len(values)))

    bars = ax.bar(
        [CRITIC_LABELS[critic_type] for critic_type in CRITIC_TYPES],
        bar_means,
        yerr=bar_stds,
        color=[CRITIC_COLORS[critic_type] for critic_type in CRITIC_TYPES],
        alpha=0.75,
        capsize=6,
    )
    for bar, (reached, total) in zip(bars, reach_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{reached}/{total} reached",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Timesteps to threshold")
    ax.set_title(f"Timesteps to Reach Reward > {args.threshold:g}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(subdirs["threshold"] / "time_to_threshold.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for critic_type in CRITIC_TYPES:
        timesteps, means, stds = aggregate_series(
            runs_by_type[critic_type],
            source_key="training_metrics",
            value_key="critic_loss",
        )
        plot_mean_with_std(
            ax,
            timesteps,
            means,
            stds,
            CRITIC_LABELS[critic_type],
            CRITIC_COLORS[critic_type],
        )
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Critic loss")
    ax.set_title("Critic Loss Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(subdirs["critic_loss"] / "critic_loss_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for critic_type in CRITIC_TYPES:
        timesteps, means, stds = aggregate_series(
            runs_by_type[critic_type],
            source_key="training_metrics",
            value_key="actor_signal_variance",
        )
        plot_mean_with_std(
            ax,
            timesteps,
            means,
            stds,
            CRITIC_LABELS[critic_type],
            CRITIC_COLORS[critic_type],
        )
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Actor update signal variance")
    ax.set_title("Advantage / Q-based Signal Variance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(subdirs["signal_variance"] / "signal_variance_curve.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
