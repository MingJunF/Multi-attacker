# PPO Solution for LunarLander-v3 (Discrete)

## 1. Problem Description

Environment: Robust Gymnasium `LunarLander-v3` (Discrete)

The agent controls a lunar lander and must land safely on the target pad.
An episode ends when the lander crashes, lands, or reaches the step limit.
A score >= 200 (averaged over 100 episodes) is considered solved.

| Property | Detail |
|---|---|
| Observation space | `Box(8,)`: position (x, y), velocity (vx, vy), angle, angular velocity, left/right leg contact |
| Action space | `Discrete(4)`: 0 = nop, 1 = left engine, 2 = main engine, 3 = right engine |
| Reward threshold | 200 |
| Max steps | 1000 |

Perturbation setting: none (`noise_factor="none"`, `noise_sigma=0.0`).
This is the baseline setup before adding robustness perturbations.

---

## 2. Algorithm: Proximal Policy Optimization (PPO)

### 2.1 Core Idea

PPO (Schulman et al., 2017) is an on-policy actor-critic algorithm.
It updates the policy with multiple epochs of mini-batch optimization while
controlling how far the new policy can move from the old one.

Compared with DQN, PPO directly optimizes a stochastic policy `pi(a|s)`
instead of only learning action values.

Key advantages over DQN:
1. On-policy updates with stable optimization behavior.
2. Direct policy optimization for stochastic action selection.
3. Clipped objective to avoid overly large policy updates.

### 2.2 PPO Objective

Main terms used in PPO:

```text
ratio_t = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
L_clip  = E[min(ratio_t * A_t, clip(ratio_t, 1 - eps, 1 + eps) * A_t)]
L_vf    = E[(V(s_t) - R_t)^2]
L_total = L_clip - c1 * L_vf + c2 * entropy
```

Where:
- `A_t` is the advantage estimate.
- `eps` is the clip range (0.2 in this implementation).
- `entropy` encourages exploration.

### 2.3 GAE (Generalized Advantage Estimation)

Advantages are computed with GAE:

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t     = sum_{l=0..inf} (gamma * lambda)^l * delta_{t+l}
```

GAE reduces gradient variance while preserving useful training signal.

---

## 3. Network Architecture

Actor network (policy):

```text
Input(8)
-> Linear(64)
-> Tanh
-> Linear(64)
-> Tanh
-> Linear(4)
-> Softmax(action probabilities)
```

Critic network (value):

```text
Input(8)
-> Linear(64)
-> Tanh
-> Linear(64)
-> Tanh
-> Linear(1)
-> V(s)
```

---

## 4. Hyperparameters

| Parameter | Value |
|---|---|
| Total episodes | 1000 |
| Max steps per episode | 1000 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Learning rate | 3e-4 |
| Clip range | 0.2 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0.01 |
| Batch size | 64 |
| PPO epochs per update | 10 |
| Rollout steps (`N_STEPS`) | 2048 |
| Hidden dim | 64 |
| Max grad norm | 0.5 |

---

## 5. Implementation Details

### 5.1 Main Files

```text
examples/LunarLander_PPO/
|- train_PPO.py
|- train_delayed_reward.py
|- visualize_delayed_reward.py
|- solution_PPO.md
|- solution_delayed_reward.md
```

### 5.2 Environment Step Interface

Robust Gymnasium uses dict-based input for `env.step(...)`:

```python
robust_input = {
	"action": action,
	"robust_type": "action",
	"robust_config": args,
}
next_state, reward, terminated, truncated, info = env.step(robust_input)
```

---

## 6. How to Run

```bash
conda activate robustgymnasium
cd Robust-Gymnasium
python examples/LunarLander_PPO/train_PPO.py
```

---