"""
Generates six publication-quality visualizations from the REINFORCE+Normalization
delayed reward experiment results saved by train_delayed_reward.py.

Plots
-----
1. Reward Timeline      – delivered vs original reward per timestep
2. Return Variance      – Var(return-to-go) across episodes per timestep
3. Return-to-Go Profile – mean G_t trajectory over an episode
4. Credit Assignment    – |G̃_t| (normalized return magnitude) heatmap
5. Robustness Curve     – final performance vs reward delay length
6. Learning Curve       – training episode reward for each reward structure

Note: REINFORCE is a pure policy-gradient method without a value network.
  Plot 3 uses the Monte Carlo return-to-go G_t (the analog of V(s)/Q(s,a)).
  Plot 4 uses |G̃_t| — the absolute normalized return — which directly
  represents the gradient signal magnitude at each timestep.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE_DIR = "results/delayed_reward_reinforce"
CONFIGS  = ["baseline", "dense", "medium_10", "medium_20", "sparse"]

COLORS = {
    "baseline":  "#1976D2",
    "dense":     "#388E3C",
    "medium_10": "#F57C00",
    "medium_20": "#D32F2F",
    "sparse":    "#7B1FA2",
}
LABELS = {
    "baseline":  "Baseline (Default)",
    "dense":     "Dense (Extra Shaping)",
    "medium_10": "Medium Delay (K=10)",
    "medium_20": "Medium Delay (K=20)",
    "sparse":    "Sparse (Terminal Only)",
}


def load_results():
    with open(os.path.join(SAVE_DIR, "experiment_results.pkl"), "rb") as f:
        return pickle.load(f)


def _smooth(arr, w=20):
    if len(arr) < w:
        return np.array(arr)
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def _save(fig, name):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {name}")


# ─── 1. Reward Timeline ──────────────────────────────────────────────────────
def plot_reward_timeline(R):
    Y_LIM = (-30, 30)
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    for i, name in enumerate(CONFIGS):
        ep = R[name]["eval_data"][0]
        raw_del = np.array(ep["delivered_rewards"], dtype=float)
        raw_ori = np.array(ep["rewards"], dtype=float)
        t = np.arange(len(raw_del))

        delivered = np.clip(raw_del, Y_LIM[0], Y_LIM[1])
        original  = np.clip(raw_ori, Y_LIM[0], Y_LIM[1])

        ax = axes[i]
        ax.bar(t, delivered, width=1.0,
               color=COLORS[name], alpha=0.7, label="Delivered reward")
        ax.plot(t, original, color="gray", alpha=0.55,
                linewidth=0.8, label="Original reward")

        for arr, va, y_anchor in [
            (raw_del, "bottom", Y_LIM[1] - 1),
            (raw_del, "top",    Y_LIM[0] + 1),
        ]:
            above = va == "bottom"
            mask = arr > Y_LIM[1] if above else arr < Y_LIM[0]
            for idx in np.where(mask)[0]:
                ax.annotate(
                    f"{arr[idx]:+.0f}", xy=(idx, y_anchor),
                    fontsize=6.5, fontweight="bold", color=COLORS[name],
                    ha="center", va=va,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec=COLORS[name], lw=0.6, alpha=0.85))

        ax.set_ylim(Y_LIM)
        ax.set_ylabel("Reward", fontsize=9)
        ax.set_title(LABELS[name], fontsize=10, fontweight="bold", loc="left")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Timestep")
    fig.suptitle("1 · Reward Timeline (REINFORCE+Norm): Delivered vs Original"
                 "  (y clipped ±30)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "1_reward_timeline.png")


# ─── 2. Return Variance ──────────────────────────────────────────────────────
def plot_return_variance(R):
    curves = {}
    for name in CONFIGS:
        evals = R[name]["eval_data"]
        max_len = max(len(e["rewards"]) for e in evals)
        var_t = []
        for t in range(max_len):
            rtg = [sum(e["rewards"][t:]) for e in evals if t < len(e["rewards"])]
            var_t.append(np.var(rtg) if len(rtg) >= 2 else 0.0)
        sm = _smooth(var_t, w=10)
        curves[name] = (np.arange(len(sm)) + 5, sm)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8),
                                          sharex=True, height_ratios=[1, 1])

    for name in CONFIGS:
        x, y = curves[name]
        ax_top.plot(x, np.maximum(y, 1), color=COLORS[name],
                    linewidth=1.8, label=LABELS[name])
    ax_top.set_yscale("log")
    ax_top.set_ylabel("Var(Return-to-Go)  [log]")
    ax_top.set_title("2 · Return Variance Over Episode Timesteps (REINFORCE+Norm)",
                     fontweight="bold")
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3, which="both")

    for name in CONFIGS:
        if name == "sparse":
            continue
        x, y = curves[name]
        ax_bot.plot(x, y, color=COLORS[name], linewidth=1.8,
                    label=LABELS[name])
    ax_bot.set_xlabel("Timestep")
    ax_bot.set_ylabel("Var(Return-to-Go)  [linear, excl. sparse]")
    ax_bot.legend(fontsize=8)
    ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "2_return_variance.png")


# ─── 3. Return-to-Go Profile (analog of Value Prediction) ────────────────────
def plot_value_prediction(R):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in CONFIGS:
        evals = R[name]["eval_data"]
        max_len = max(len(e["q_values"]) for e in evals)
        mean_g = []
        for t in range(max_len):
            vals = [e["q_values"][t] for e in evals if t < len(e["q_values"])]
            mean_g.append(np.mean(vals) if vals else 0.0)
        sm = _smooth(mean_g, w=5) if len(mean_g) > 5 else np.array(mean_g)
        ax.plot(np.arange(len(sm)) + 2, sm,
                color=COLORS[name], linewidth=1.8, label=LABELS[name])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("G_t  (Return-to-Go)")
    ax.set_title("3 · Return-to-Go Profile: Mean G_t Over Episode (REINFORCE+Norm)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "3_value_prediction.png")


# ─── 4. Credit Assignment Heatmap ────────────────────────────────────────────
def plot_credit_assignment(R):
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, name in enumerate(CONFIGS):
        evals = R[name]["eval_data"]
        max_len = max(len(e["td_errors"]) for e in evals)
        n_eps = min(len(evals), 15)
        mat = np.full((n_eps, max_len), np.nan)
        for j in range(n_eps):
            L = len(evals[j]["td_errors"])
            mat[j, :L] = evals[j]["td_errors"]

        valid = mat[np.isfinite(mat) & (mat > 0)]
        vmin = max(valid.min(), 1e-2) if len(valid) else 1e-2
        vmax = valid.max() if len(valid) else 1.0
        mat_log = np.where(np.isfinite(mat) & (mat > 0), mat, vmin)
        mat_log[~np.isfinite(mat)] = np.nan

        im = axes[i].imshow(mat_log, aspect="auto", cmap="YlOrRd",
                            interpolation="nearest",
                            norm=LogNorm(vmin=vmin, vmax=vmax))
        axes[i].set_title(LABELS[name], fontsize=8, fontweight="bold")
        axes[i].set_xlabel("Timestep", fontsize=8)
        if i == 0:
            axes[i].set_ylabel("Episode", fontsize=9)
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    fig.suptitle("4 · Credit Assignment: |G̃_t| Gradient Signal – log scale"
                 " (REINFORCE+Norm)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "4_credit_assignment.png")


# ─── 5. Delayed Reward Robustness ────────────────────────────────────────────
def plot_robustness(R):
    data = R["robustness"]
    delays = data["delays"]
    means  = np.array(data["mean_rewards"])
    stds   = np.array(data["std_rewards"])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(delays))
    ax.plot(x, means, "o-", color="#1976D2", linewidth=2.5,
            markersize=9, zorder=3)
    ax.fill_between(x, means - stds, means + stds,
                    alpha=0.18, color="#1976D2")
    ax.errorbar(x, means, yerr=stds, fmt="none",
                ecolor="#90CAF9", capsize=5, zorder=2)

    tick_labels = [str(d) if d < 1000 else "Sparse" for d in delays]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Reward Delay (steps)")
    ax.set_ylabel("Mean Episode Reward (last 50 training eps)")
    ax.set_title("5 · Delayed Reward Robustness: Performance vs Delay"
                 " (REINFORCE+Norm)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "5_robustness.png")


# ─── 6. Learning Curve ───────────────────────────────────────────────────────
def plot_learning_curve(R):
    fig, ax = plt.subplots(figsize=(12, 6))
    window = 30
    for name in CONFIGS:
        rews = np.array(R[name]["train_rewards"])
        eps  = np.arange(1, len(rews) + 1)
        ax.plot(eps, rews, alpha=0.12, color=COLORS[name])
        sm = _smooth(rews, w=window)
        ax.plot(np.arange(window // 2, window // 2 + len(sm)) + 1, sm,
                color=COLORS[name], linewidth=2.2, label=LABELS[name])

    ax.axhline(y=200, color="gray", linestyle="--", alpha=0.4,
               label="Solved (200)")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Episode Reward (Original)")
    ax.set_title("6 · Learning Curve: REINFORCE+Norm Under Different"
                 " Reward Structures", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "6_learning_curve.png")


# ─── Summary Table ───────────────────────────────────────────────────────────
def print_summary(R):
    print("\n" + "=" * 70)
    print("  Delayed Reward Experiment (REINFORCE+Norm) – Summary")
    print("=" * 70)
    header = f"{'Config':<28} {'Train Avg(last100)':>18} {'Eval Avg':>10}"
    print(header)
    print("-" * 70)
    for name in CONFIGS:
        train_avg = np.mean(R[name]["train_rewards"][-100:])
        eval_avg  = np.mean([sum(e["rewards"]) for e in R[name]["eval_data"]])
        print(f"  {LABELS[name]:<26} {train_avg:>16.1f} {eval_avg:>10.1f}")
    print("=" * 70)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("[INFO] Loading REINFORCE+Norm experiment results …")
    R = load_results()
    print("[INFO] Generating 6 visualizations …\n")

    plot_reward_timeline(R)
    plot_return_variance(R)
    plot_value_prediction(R)
    plot_credit_assignment(R)
    plot_robustness(R)
    plot_learning_curve(R)
    print_summary(R)

    print(f"\n[INFO] All plots saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
