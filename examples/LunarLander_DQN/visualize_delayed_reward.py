import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE_DIR = "results/delayed_reward"
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


#  Reward Timeline 
def plot_reward_timeline(R):
    Y_LIM = (-30, 30)
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    for i, name in enumerate(CONFIGS):
        ep = R[name]["eval_data"][0]
        raw_del = np.array(ep["delivered_rewards"], dtype=float)
        raw_ori = np.array(ep["rewards"], dtype=float)
        T = len(raw_del)
        t = np.arange(T)

        delivered = np.clip(raw_del, Y_LIM[0], Y_LIM[1])
        original  = np.clip(raw_ori, Y_LIM[0], Y_LIM[1])

        ax = axes[i]
        ax.bar(t, delivered, width=1.0,
               color=COLORS[name], alpha=0.7, label="Delivered reward")
        ax.plot(t, original, color="gray", alpha=0.55,
                linewidth=0.8, label="Original reward")

        for arr, va, y_anchor in [
            (raw_del, "bottom", Y_LIM[1] - 1),   # overflow above
            (raw_del, "top",    Y_LIM[0] + 1),    # overflow below
        ]:
            above = va == "bottom"
            mask = arr > Y_LIM[1] if above else arr < Y_LIM[0]
            idxs = np.where(mask)[0]
            for idx in idxs:
                ax.annotate(
                    f"{arr[idx]:+.0f}",
                    xy=(idx, y_anchor),
                    fontsize=6.5, fontweight="bold", color=COLORS[name],
                    ha="center", va=va,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec=COLORS[name], lw=0.6, alpha=0.85),
                )

        ax.set_ylim(Y_LIM)
        ax.set_ylabel("Reward", fontsize=9)
        ax.set_title(LABELS[name], fontsize=10, fontweight="bold", loc="left")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Timestep")
    fig.suptitle("1 · Reward Timeline: Delivered vs Original Reward per Step"
                 "  (y clipped to ±30)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "1_reward_timeline.png")


# Return Variance 
def plot_return_variance(R):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in CONFIGS:
        evals = R[name]["eval_data"]
        max_len = max(len(e["rewards"]) for e in evals)
        var_t = []
        for t in range(max_len):
            rtg = [sum(e["rewards"][t:]) for e in evals if t < len(e["rewards"])]
            var_t.append(np.var(rtg) if len(rtg) >= 2 else 0.0)
        sm = _smooth(var_t, w=10)
        ax.plot(np.arange(len(sm)) + 5, sm,
                color=COLORS[name], linewidth=1.8, label=LABELS[name])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Var(Return-to-Go)")
    ax.set_title("2 · Return Variance Over Episode Timesteps", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "2_return_variance.png")


# Value Prediction
def plot_value_prediction(R):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in CONFIGS:
        evals = R[name]["eval_data"]
        max_len = max(len(e["q_values"]) for e in evals)
        mean_q = []
        for t in range(max_len):
            vals = [e["q_values"][t] for e in evals if t < len(e["q_values"])]
            mean_q.append(np.mean(vals) if vals else 0.0)
        sm = _smooth(mean_q, w=5) if len(mean_q) > 5 else np.array(mean_q)
        ax.plot(np.arange(len(sm)) + 2, sm,
                color=COLORS[name], linewidth=1.8, label=LABELS[name])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Q(s, a)")
    ax.set_title("3 · Value Prediction: Mean Q(s, a) Over Episode", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "3_value_prediction.png")


#  Credit Assignment Heatmap 
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
    fig.suptitle("4 · Credit Assignment: TD-Error Magnitude (log scale)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "4_credit_assignment.png")


#  Delayed Reward Robustness
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
    ax.set_title("5 · Delayed Reward Robustness: Performance vs Delay",
                 fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "5_robustness.png")


#  Learning Curve 
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
    ax.set_title("6 · Learning Curve: DQN Under Different Reward Structures",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "6_learning_curve.png")


# Summary Table 
def print_summary(R):
    print("\n" + "=" * 70)
    print("  Delayed Reward Experiment – Summary")
    print("=" * 70)
    header = f"{'Config':<28} {'Train Avg(last100)':>18} {'Eval Avg':>10}"
    print(header)
    print("-" * 70)
    for name in CONFIGS:
        train_avg = np.mean(R[name]["train_rewards"][-100:])
        eval_avg  = np.mean([sum(e["rewards"]) for e in R[name]["eval_data"]])
        print(f"  {LABELS[name]:<26} {train_avg:>16.1f} {eval_avg:>10.1f}")
    print("=" * 70)


# Main
def main():
    print("[INFO] Loading experiment results …")
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
