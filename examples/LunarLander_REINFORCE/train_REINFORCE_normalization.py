"""
Strict REINFORCE for Robust Gymnasium LunarLander-v3 (Discrete).

This implementation follows the standard REINFORCE algorithm:
- Policy-only (no critic / no value network)
- Monte Carlo episode return
- Policy gradient update once per episode
- Visualization includes reward and moving average only
"""

import os
import random
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import robust_gymnasium as gym
from robust_gymnasium.configs.robust_setting import get_config


# Hyperparameters
ENV_NAME = "LunarLander-v3"
SEED = 42
TOTAL_EPISODES = 1000
MAX_STEPS = 1000
GAMMA = 0.99
LR = 1e-3
HIDDEN_DIM = 128
SOLVE_SCORE = 200.0
SAVE_DIR = "results/train_reinforce_normalization"


class PolicyNetwork(nn.Module):
    """MLP policy network that outputs action logits."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_discounted_returns(rewards: list, gamma: float = GAMMA) -> list:
    """Compute discounted returns G_t for a full episode trajectory."""
    returns = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    return returns


def record_best_episode_mp4(
    policy: "PolicyNetwork",
    args,
    device: torch.device,
    save_dir: str,
    num_eval_episodes: int = 5,
) -> None:
    """Run num_eval_episodes with greedy policy, save the best one as MP4."""
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    best_frames: list = []
    best_reward = -float("inf")

    policy.eval()
    with torch.no_grad():
        for ep in range(num_eval_episodes):
            state, _ = env.reset(seed=SEED + 20000 + ep)
            frames: list = []
            total_reward = 0.0

            for _ in range(MAX_STEPS):
                frames.append(env.render())
                state_t = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)
                logits = policy(state_t)
                action = int(logits.argmax(dim=-1).item())  # greedy

                robust_input = {
                    "action": action,
                    "robust_type": "action",
                    "robust_config": args,
                }
                state, reward, terminated, truncated, _ = env.step(robust_input)
                total_reward += float(reward)
                if terminated or truncated:
                    frames.append(env.render())
                    break

            print(f"[EVAL] Episode {ep + 1}/{num_eval_episodes} | Reward: {total_reward:.1f}")
            if total_reward > best_reward:
                best_reward = total_reward
                best_frames = frames

    env.close()
    policy.train()

    # Write MP4 using ffmpeg writer; fall back to pillow (gif) if unavailable
    os.makedirs(save_dir, exist_ok=True)
    mp4_path = os.path.join(save_dir, "best_episode.mp4")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    img_obj = ax.imshow(best_frames[0])
    ax.set_title(
        f"REINFORCE+Norm | Best Reward: {best_reward:.1f}", fontsize=11
    )

    def _update(i: int):
        img_obj.set_data(best_frames[i])
        return [img_obj]

    ani = mpl_animation.FuncAnimation(
        fig, _update, frames=len(best_frames), interval=33, blit=True
    )

    try:
        writer = mpl_animation.FFMpegWriter(fps=30, bitrate=1800)
        ani.save(mp4_path, writer=writer)
        print(f"[INFO] Best episode video saved to {mp4_path}")
    except Exception:
        # ffmpeg not installed — fall back to GIF
        gif_path = mp4_path.replace(".mp4", ".gif")
        ani.save(gif_path, writer="pillow", fps=30)
        print(f"[WARN] ffmpeg not found; saved as GIF instead: {gif_path}")

    plt.close(fig)


def plot_results(scores: list, avg_scores: list, save_dir: str) -> None:
    """Save training curves: episodic reward and moving-average reward."""
    os.makedirs(save_dir, exist_ok=True)
    episodes = range(1, len(scores) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, scores, alpha=0.3, color="steelblue", label="Episode reward")
    ax.plot(episodes, avg_scores, color="darkorange", linewidth=2, label="Moving Avg (100 ep)")
    ax.axhline(y=SOLVE_SCORE, color="green", linestyle="--", label=f"Solved ({SOLVE_SCORE})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("REINFORCE + Return Normalization on LunarLander-v3 (Discrete)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(save_dir, "training_curves_normalization.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Training curves saved to {output_path}")


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Robust-Gymnasium config: disable perturbations for standard training
    args = get_config().parse_args([])
    args.noise_factor = "none"
    args.noise_sigma = 0.0

    # Seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    env = gym.make(ENV_NAME)
    init_state, _ = env.reset(seed=SEED)
    state_dim = int(np.asarray(init_state, dtype=np.float32).shape[0])
    action_dim = int(getattr(env.action_space, "n"))

    policy = PolicyNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    scores = []
    avg_scores = []
    recent_scores = deque(maxlen=100)
    solved = False

    print(f"[INFO] Starting strict REINFORCE training for {TOTAL_EPISODES} episodes...")

    for episode in range(1, TOTAL_EPISODES + 1):
        state, _ = env.reset(seed=SEED + episode)
        episode_rewards = []
        episode_log_probs = []

        for _ in range(MAX_STEPS):
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits = policy(state_t)
            dist = Categorical(logits=logits)

            action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t)

            robust_input = {
                "action": int(action_t.item()),
                "robust_type": "action",
                "robust_config": args,
            }
            next_state, reward, terminated, truncated, _ = env.step(robust_input)

            episode_rewards.append(float(reward))
            episode_log_probs.append(log_prob_t.squeeze(0))

            state = next_state
            if terminated or truncated:
                break

        # Standard REINFORCE update uses full-episode Monte Carlo returns.
        returns = compute_discounted_returns(episode_rewards, gamma=GAMMA)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        log_probs_t = torch.stack(episode_log_probs)

        policy_loss = -(log_probs_t * returns_t).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        episode_reward = float(np.sum(episode_rewards))
        scores.append(episode_reward)
        recent_scores.append(episode_reward)
        avg_reward = float(np.mean(recent_scores))
        avg_scores.append(avg_reward)

        if episode % 20 == 0:
            print(
                f"Episode {episode:4d} | Reward: {episode_reward:7.1f} | "
                f"Avg(100): {avg_reward:7.1f} | PolicyLoss: {policy_loss.item():9.2f}"
            )

        if avg_reward >= SOLVE_SCORE and not solved:
            solved = True
            print(f"\n*** Solved at episode {episode} with avg reward {avg_reward:.1f} ***\n")
            torch.save(policy.state_dict(), os.path.join(SAVE_DIR, "reinforce_solved.pth"))

    torch.save(policy.state_dict(), os.path.join(SAVE_DIR, "reinforce_final.pth"))
    env.close()

    plot_results(scores, avg_scores, SAVE_DIR)
    record_best_episode_mp4(policy, args, device, SAVE_DIR)
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()

