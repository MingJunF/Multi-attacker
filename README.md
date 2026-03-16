# LunarLander Experiments with Robust Gymnasium

This repository contains a focused set of reinforcement learning experiments on `LunarLander-v3` using the Robust Gymnasium interface.

It is designed as a practical benchmark to compare classic and modern RL methods under:
- standard training (no perturbation),
- perturbation-based robustness tests,
- delayed/sparse reward settings,
- value-estimation bias analysis.

## What is included

### Algorithms
- DQN
- Double DQN
- Dueling Double DQN (+ PER)
- PPO
- A2C (value critic and Q critic variants)
- REINFORCE

### Experiment themes
- **Baseline learning** on `LunarLander-v3`
- **Robustness to perturbations** (state noise / reward noise)
- **Delayed reward credit assignment**
- **Q-value overestimation bias** (DQN vs Double DQN)
- **Value-Advantage decomposition analysis** (Dueling architecture)

## Project structure (LunarLander v3-Discrete)

```text
examples/
  LunarLander_DQN/
    train_dqn.py
    train_dqn_perturbation.py
    train_delayed_reward.py
    visualize_delayed_reward.py

  LunarLander_double_DQN&DDQN/
    train_double_dqn.py
    train_dueling_double_dqn.py
    compare_algorithms_perturbation.py

  LunarLander_PPO/
    train_PPO.py
    train_delayed_reward.py

  LunarLander_A2C/
    main.py
    evaluation.py
    evaluate_saved_model.py
    plot_results.py

  LunarLander_REINFORCE/
    train_REINFORCE.py
    train_REINFORCE_normalization.py
    train_delayed_reward.py

  LunarLander_QBias_Experiment1/
    experiment_1A_overestimation_bias.py
    experiment_1B_dueling_value_advantage.py
```

## Quick start

```bash
conda create -n robustgymnasium python=3.11
conda activate robustgymnasium

git clone https://github.com/fangevo/Robust-Gymnasium.git
cd Robust-Gymnasium

pip install -r requirements.txt
pip install -e .
```

## Run experiments

From repository root:

```bash
# DQN baseline
python examples/LunarLander_DQN/train_dqn.py

# Double DQN
python "examples/LunarLander_double_DQN&DDQN/train_double_dqn.py"

# Dueling Double DQN (+ PER)
python "examples/LunarLander_double_DQN&DDQN/train_dueling_double_dqn.py"

# PPO baseline
python examples/LunarLander_PPO/train_PPO.py

# A2C (default args)
python examples/LunarLander_A2C/main.py

# Strict REINFORCE
python examples/LunarLander_REINFORCE/train_REINFORCE.py

# Perturbation comparison across DQN variants
python "examples/LunarLander_double_DQN&DDQN/compare_algorithms_perturbation.py"

# DQN delayed-reward experiments
python examples/LunarLander_DQN/train_delayed_reward.py
python examples/LunarLander_DQN/visualize_delayed_reward.py

# Q-bias experiments
python examples/LunarLander_QBias_Experiment1/experiment_1A_overestimation_bias.py
python examples/LunarLander_QBias_Experiment1/experiment_1B_dueling_value_advantage.py

# 6 Algorithm Robustness Tests (High Wind, Low Gravity)
python examples/LunarLander_RobustnessTest/robustness_test.py
python examples/LunarLander_RobustnessTest/visualize_robustness.py
```

## Notes on Robust Gymnasium interface

The scripts use Robust Gymnasium's dict-based step input:

```python
robust_input = {
    "action": action,
    "robust_type": "action",
    "robust_config": args,
}
next_state, reward, terminated, truncated, info = env.step(robust_input)
```

Set perturbations with fields in `robust_config` (for example `noise_factor`, `noise_type`, `noise_sigma`).

## Typical outputs

Depending on script, outputs include:
- training curves (`.png`),
- saved models (`.pth` / `.pt`),
- evaluation logs (`.csv` / `.json`),
- rollout visualizations (`.gif` / `.mp4`),
- experiment summaries (`.txt`).

## Citation

If this project is useful in your research, please cite:

```bibtex
@article{robustrl2024,
  title={Robust Gymnasium: A Unified Modular Benchmark for Robust Reinforcement Learning},
  author={Gu, Shangding and Shi, Laixi and Wen, Muning and Jin, Ming and Mazumdar, Eric and Chi, Yuejie and Wierman, Adam and Spanos, Costas},
  journal={ICLR},
  year={2025}
}
```
