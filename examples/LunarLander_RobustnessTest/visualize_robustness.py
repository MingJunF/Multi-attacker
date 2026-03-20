import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "RobustnessTest"
RESULTS_FILE = RESULTS_DIR / "robustness_results.json"

AGENTS = ["DQN", "Double DQN", "Dueling DDQN", "PPO", "REINFORCE", "A2C-V"]
ENVS = ["Standard", "High-Wind", "Low-Gravity"]
AGENT_COLORS = {
    "DQN": "#2196F3", "Double DQN": "#1565C0", "Dueling DDQN": "#0D47A1",
    "PPO": "#FF9800", "REINFORCE": "#4CAF50", "A2C-V": "#E91E63",
}


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def plot_boxplot(results):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for ax_idx, env in enumerate(ENVS):
        ax = axes[ax_idx]
        data = [results[f"{agent}_{env}"] for agent in AGENTS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker="o", markersize=4, alpha=0.5))
        for patch, agent in zip(bp["boxes"], AGENTS):
            patch.set_facecolor(AGENT_COLORS[agent])
            patch.set_alpha(0.75)

        ax.set_xticklabels(AGENTS, fontsize=8.5, rotation=25, ha="right")
        ax.set_title(env, fontsize=13, fontweight="bold")
        ax.axhline(y=200, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("Episode Reward", fontsize=11)

    fig.suptitle("Zero-Shot Robustness: Reward Distribution by Environment",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = RESULTS_DIR / "boxplot_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {path}")


def plot_grouped_bar(results):
    n_agents = len(AGENTS)
    x = np.arange(n_agents)
    bar_width = 0.24

    fig, ax = plt.subplots(figsize=(14, 6))
    env_colors = ["#5C6BC0", "#EF5350", "#66BB6A"]

    for i, env in enumerate(ENVS):
        means = [np.mean(results[f"{a}_{env}"]) for a in AGENTS]
        stds = [np.std(results[f"{a}_{env}"]) for a in AGENTS]
        offset = (i - 1) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds,
               label=env, color=env_colors[i], alpha=0.85,
               capsize=3, edgecolor="white", linewidth=0.5,
               error_kw=dict(lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(AGENTS, fontsize=10)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title("Zero-Shot Robustness: Mean Reward (± Std)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.axhline(y=200, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = RESULTS_DIR / "grouped_bar_chart.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {path}")


def plot_heatmap(results):
    matrix = np.zeros((len(AGENTS), len(ENVS)))
    for i, agent in enumerate(AGENTS):
        std_mean = np.mean(results[f"{agent}_Standard"])
        for j, env in enumerate(ENVS):
            env_mean = np.mean(results[f"{agent}_{env}"])
            if abs(std_mean) > 1e-6:
                matrix[i, j] = ((env_mean - std_mean) / abs(std_mean)) * 100
            else:
                matrix[i, j] = 0.0

    fig, ax = plt.subplots(figsize=(9, 6))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 1)
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(ENVS)))
    ax.set_xticklabels(ENVS, fontsize=11)
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels(AGENTS, fontsize=11)

    for i in range(len(AGENTS)):
        for j in range(len(ENVS)):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Reward Change (%)", fontsize=10)

    ax.set_title("Performance Change Relative to Standard (%)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = RESULTS_DIR / "performance_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {path}")


def plot_radar(results):
    metrics = []
    for env in ENVS:
        for stat_fn, label in [(np.mean, "Mean"), (lambda x: -np.std(x), "Consistency")]:
            metrics.append((env, label, stat_fn))

    labels = [f"{env}\n{label}" for env, label, _ in metrics]
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    all_values = []
    for agent in AGENTS:
        values = []
        for env, label, stat_fn in metrics:
            values.append(stat_fn(results[f"{agent}_{env}"]))
        all_values.append(values)

    flat = [v for vals in all_values for v in vals]
    vmin, vmax = min(flat), max(flat)
    rng = vmax - vmin if vmax != vmin else 1.0

    for agent, values in zip(AGENTS, all_values):
        normalized = [(v - vmin) / rng for v in values]
        normalized += normalized[:1]
        ax.plot(angles, normalized, "o-", linewidth=2, label=agent, color=AGENT_COLORS[agent])
        ax.fill(angles, normalized, alpha=0.08, color=AGENT_COLORS[agent])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Robustness Profile (normalized)", fontsize=13, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=9)
    plt.tight_layout()
    path = RESULTS_DIR / "radar_chart.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {path}")


def print_statistics(results):
    """Print detailed statistics table."""
    print(f"\n{'='*94}")
    print(f"{'Detailed Statistics':^94}")
    print(f"{'='*94}")
    print(f"{'Algorithm':<16} {'Environment':<14} {'Mean':>8} {'Std':>8} "
          f"{'Min':>8} {'Max':>8} {'Median':>8}")
    print("-" * 94)
    for agent in AGENTS:
        for env in ENVS:
            r = results[f"{agent}_{env}"]
            print(f"{agent:<16} {env:<14} {np.mean(r):8.1f} {np.std(r):8.1f} "
                  f"{np.min(r):8.1f} {np.max(r):8.1f} {np.median(r):8.1f}")
        print()
    print(f"{'='*94}\n")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_FILE.exists():
        print(f"[ERROR] Results file not found: {RESULTS_FILE}")
        print("       Run robustness_test.py first.")
        return

    results = load_results()
    print(f"[INFO] Loaded results from {RESULTS_FILE}")

    print_statistics(results)
    plot_boxplot(results)
    plot_grouped_bar(results)
    plot_heatmap(results)
    plot_radar(results)

    print(f"\n[INFO] All visualizations saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
