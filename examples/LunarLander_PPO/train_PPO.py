"""
PPO Agent for Robust Gymnasium LunarLander-v3 (Discrete).
Actor-Critic with clipped surrogate objective and GAE.
No perturbation applied (noise_factor="none").
"""

import os
import random
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import robust_gymnasium as gym
from robust_gymnasium.configs.robust_setting import get_config

# ─── Hyperparameters ─────────────────────────────────────────────────────────
ENV_NAME        = "LunarLander-v3"
SEED            = 42
TOTAL_EPISODES  = 1000         
MAX_STEPS       = 1000
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
LR              = 3e-4
CLIP_RANGE      = 0.2
VALUE_COEF      = 0.5
ENTROPY_COEF    = 0.01
BATCH_SIZE      = 64
N_EPOCHS        = 10
N_STEPS         = 2048         # collect 2048 steps before update
HIDDEN_DIM      = 64
MAX_GRAD_NORM   = 0.5
SOLVE_SCORE     = 200.0
SAVE_DIR = "results/train_ppo"


# ─── Actor Network (Policy) ─────────────────────────────────────────────────
class ActorNetwork(nn.Module):
    """Policy network outputting action probabilities."""
    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
    
    def forward(self, x: torch.Tensor):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)
    
    def get_action(self, state: np.ndarray, device: torch.device):
        """Sample action from policy."""
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        probs = self.forward(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()


# ─── Critic Network (Value) ─────────────────────────────────────────────────
class CriticNetwork(nn.Module):
    """Value network estimating V(s)."""
    def __init__(self, state_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─── Rollout Buffer ─────────────────────────────────────────────────────────
class RolloutBuffer:
    """Store trajectories for on-policy learning."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def push(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get(self):
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
        )
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def __len__(self):
        return len(self.states)


# ─── PPO Agent ──────────────────────────────────────────────────────────────
class PPOAgent:
    """PPO agent with clipped objective and GAE."""
    def __init__(self, state_dim: int, action_dim: int, device: torch.device):
        self.device = device
        self.action_dim = action_dim
        
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        
        self.buffer = RolloutBuffer()
    
    def select_action(self, state: np.ndarray):
        """Select action and compute log_prob, value."""
        action, log_prob = self.actor.get_action(state, self.device)
        
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(state_t).item()
        
        return action, log_prob, value
    
    def compute_gae(self, rewards, dones, values, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + GAMMA * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, next_state):
        """Update policy and value networks using collected rollouts."""
        states, actions, rewards, dones, old_log_probs, values = self.buffer.get()
        
        # Compute next value for GAE
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            next_value = self.critic(next_state_t).item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, dones, values, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, device=self.device)
        advantages_t = torch.tensor(advantages, device=self.device)
        returns_t = torch.tensor(returns, device=self.device)
        
        # Multiple epochs of mini-batch updates
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        for _ in range(N_EPOCHS):
            # Generate random mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_idx = indices[start:end]
                
                # Get current policy distribution
                probs = self.actor(states_t[batch_idx])
                dist = Categorical(probs)
                
                new_log_probs = dist.log_prob(actions_t[batch_idx])
                entropy = dist.entropy().mean()
                
                # Compute ratio and clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs_t[batch_idx])
                surr1 = ratio * advantages_t[batch_idx]
                surr2 = torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * advantages_t[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values_pred = self.critic(states_t[batch_idx])
                value_loss = F.mse_loss(values_pred, returns_t[batch_idx])
                
                # Total loss
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
                
                # Update actor
                self.actor_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
                self.critic_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        
        self.buffer.clear()
        
        return (
            total_policy_loss / n_updates,
            total_value_loss / n_updates,
            total_entropy / n_updates
        )


# ─── Visualization ──────────────────────────────────────────────────────────
def plot_results(scores: list, avg_scores: list, value_losses: list, save_dir: str):
    """Generate and save training curves."""
    os.makedirs(save_dir, exist_ok=True)
    episodes = range(1, len(scores) + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # 共享x轴
    
    # Score plot
    ax1.plot(episodes, scores, alpha=0.3, color="steelblue", label="Episode reward")
    ax1.plot(episodes, avg_scores, color="darkorange", linewidth=2, label="Avg reward (100 ep)")
    ax1.axhline(y=SOLVE_SCORE, color="green", linestyle="--", label=f"Solved ({SOLVE_SCORE})")
    ax1.set_ylabel("Reward")
    ax1.set_title("PPO on LunarLander-v3 (Discrete)")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Value loss plot 
    ax2.plot(episodes, value_losses, color="crimson")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Value Loss")
    ax2.set_title("Critic Loss")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"[INFO] Training curves saved to {save_dir}/training_curves.png")


# ─── Animation ──────────────────────────────────────────────────────────────
def record_animation(agent, args, save_dir: str, num_episodes: int = 3):
    """Run trained agent and save a GIF animation of the best episode."""
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    best_frames, best_reward = [], -float("inf")
    
    for ep in range(num_episodes):
        state, _ = env.reset(seed=SEED + 10000 + ep)
        frames, total_reward = [], 0.0
        
        for _ in range(MAX_STEPS):
            frames.append(env.render())
            
            state_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                probs = agent.actor(state_t)
                action = probs.argmax(dim=-1).item()  # Greedy action
            
            robust_input = {"action": action, "robust_type": "action", "robust_config": args}
            state, reward, terminated, truncated, _ = env.step(robust_input)
            total_reward += reward
            if terminated or truncated:
                frames.append(env.render())
                break
        
        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames
        print(f"[ANIM] Episode {ep+1}/{num_episodes} | Reward: {total_reward:.1f}")
    
    env.close()
    
    # Save best episode as GIF
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    img = ax.imshow(best_frames[0])
    ax.set_title(f"PPO LunarLander-v3 | Reward: {best_reward:.1f}", fontsize=12)
    
    def update(i):
        img.set_data(best_frames[i])
        return [img]
    
    ani = animation.FuncAnimation(fig, update, frames=len(best_frames), interval=30, blit=True)
    gif_path = os.path.join(save_dir, "lunarlander_ppo.gif")
    ani.save(gif_path, writer="pillow", fps=30)
    plt.close(fig)
    print(f"[INFO] Animation saved to {gif_path}")


# ─── Main Training Loop ─────────────────────────────────────────────────────
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    
    # Robust-Gymnasium config (no perturbation)
    args = get_config().parse_args([])
    args.noise_factor = "none"
    args.noise_sigma = 0.0
    
    # Seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]  # 8
    action_dim = env.action_space.n  # 4
    
    agent = PPOAgent(state_dim, action_dim, device)
    
    scores, avg_scores, value_losses = [], [], []
    recent_scores = deque(maxlen=100)
    solved = False
    
    timesteps = 0
    last_value_loss = 0.0  
    
    print(f"[INFO] Starting PPO training for {TOTAL_EPISODES:,} episodes...")
    
    for episode in range(1, TOTAL_EPISODES + 1):
        episode_reward = 0.0
        state, _ = env.reset(seed=SEED + episode)
        
     
        for step in range(MAX_STEPS):
            action, log_prob, value = agent.select_action(state)
            
            robust_input = {
                "action": action,
                "robust_type": "action",
                "robust_config": args,
            }
            next_state, reward, terminated, truncated, _ = env.step(robust_input)
            done = terminated or truncated
            
            agent.buffer.push(state, action, reward, done, log_prob, value)
            
            state = next_state
            episode_reward += reward
            timesteps += 1
            
            if done:
                break
        
    
        scores.append(episode_reward)
        recent_scores.append(episode_reward)
        avg = np.mean(recent_scores) if recent_scores else episode_reward
        avg_scores.append(avg)
        
        # PPO update 
        if len(agent.buffer) >= N_STEPS:
            policy_loss, value_loss, entropy = agent.update(state)
            last_value_loss = value_loss
        
        
        value_losses.append(last_value_loss)
        
        
        if episode % 20 == 0:
            print(f"Episode {episode:4d} | Timestep {timesteps:7d} | Avg(100): {avg:7.1f}")
        
    
        if avg >= SOLVE_SCORE and not solved:
            solved = True
            print(f"\n*** Solved at episode {episode} with avg reward {avg:.1f} ***\n")
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
            }, os.path.join(SAVE_DIR, "ppo_solved.pth"))
    
    # Save final model
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
    }, os.path.join(SAVE_DIR, "ppo_final.pth"))
    env.close()
    
    # Plot training curves
    plot_results(scores, avg_scores, value_losses, SAVE_DIR)
    
    # Record animation of trained agent
    record_animation(agent, args, SAVE_DIR)
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()