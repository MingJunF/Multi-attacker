# A2C with V/Q Critic for RobustLunarLander-v3

This folder extracts the A2C training framework from
`pytorch-a2c-ppo-acktr-gail` into a standalone version that only targets
`RobustLunarLander-v3`.

The shared training loop is kept aligned with the original repository:

- MLP actor-critic policy
- RMSprop optimizer
- on-policy rollouts with `RolloutStorage`
- optional GAE
- vectorized environment collection
- observation normalization through `VecNormalize`

The package now supports two critic parameterizations behind one switch:

- original generic env interface -> fixed `RobustLunarLander-v3`
- `--critic-type v`: critic learns `V(s)` and actor uses `return - V(s)`
- `--critic-type q`: critic learns `Q(s,a)` and actor uses `Q(s,a) - E_pi[Q(s, .)]`

This keeps the environment, optimizer, rollout length, normalization, and
most of the surrounding code identical for later comparison.

## Layout

```text
main.py              # training entrypoint
evaluation.py        # evaluation helper
evaluate_saved_model.py
requirements.txt     # lightweight dependencies for this folder
a2c_v/
  arguments.py
  algo.py
  experiment_logger.py
  distributions.py
  envs.py
  model.py
  storage.py
  utils.py
plot_results.py      # aggregate metrics and draw the 5 comparison figures
```

## Install

Install `torch` separately according to your CUDA/CPU environment, then:

```bash
pip install -r A2C-V/requirements.txt
```

## Run

From the repository root:

```bash
for s in 0 1 2; do
  python A2C-V/main.py \
    --env-name RobustLunarLander-v3 \
    --critic-type v \
    --seed $s \
    --lr 7e-4 \
    --eps 1e-5 \
    --alpha 0.99 \
    --gamma 0.99 \
    --entropy-coef 0.01 \
    --value-loss-coef 0.5 \
    --max-grad-norm 0.5 \
    --num-processes 16 \
    --num-steps 5 \
    --num-env-steps 10000000 \
    --log-interval 10 \
    --save-interval 100 \
    --eval-interval 1000 \
    --eval-episodes 10 \
    --final-eval-episodes 100
done
```

To run the Q-based version with the same control variables, only change:

```bash
for s in 0 1 2; do
  python A2C-V/main.py \
    --env-name RobustLunarLander-v3 \
    --critic-type q \
    --seed $s \
    --lr 7e-4 \
    --eps 1e-5 \
    --alpha 0.99 \
    --gamma 0.99 \
    --entropy-coef 0.01 \
    --value-loss-coef 0.5 \
    --max-grad-norm 0.5 \
    --num-processes 16 \
    --num-steps 5 \
    --num-env-steps 10000000 \
    --log-interval 10 \
    --save-interval 100 \
    --eval-interval 1000 \
    --eval-episodes 10 \
    --final-eval-episodes 100
done
```

Each run writes structured data to:

```text
A2C-V/results/critic_compare/<critic_type>/seed_<seed>/
  config.json
  training_metrics.csv
  evaluations.csv
  final_eval_rewards.json
  summary.json
  models/
    best_model.pt
    last_model.pt
```

`training_metrics.csv` includes `critic_loss`, `action_loss`, `dist_entropy`,
`actor_signal_variance`, and `critic_target_variance`. `evaluations.csv`
stores checkpoint means/stds and the final evaluation summary. `best_model.pt`
tracks the checkpoint or final evaluation with the highest mean reward, while
`last_model.pt` stores the final trained policy.

## Evaluate A Saved Model

Evaluate the best saved value-based model in the standard environment:

```bash
for s in 0 1 2; do
  python A2C-V/evaluate_saved_model.py \
    --critic-type v \
    --seed $s \
    --model-kind best \
    --num-episodes 100 \
    --output-json A2C-V/results/critic_compare/v/seed_${s}/eval_standard.json
done
```

```bash
for s in 0 1 2; do
  python A2C-V/evaluate_saved_model.py \
    --critic-type q \
    --seed $s \
    --model-kind best \
    --num-episodes 100 \
    --output-json A2C-V/results/critic_compare/q/seed_${s}/eval_standard.json
done
```

## Plot

After running multiple seeds for both `--critic-type v` and `--critic-type q`,
generate the five requested figures with:

```bash
python A2C-V/plot_results.py \
  --results-dir A2C-V/results/critic_compare \
  --threshold 200
```

The plots are saved under:

```text
A2C-V/results/critic_compare/plots/
  learning_curves/learning_curve.png
  final_performance/final_performance_boxplot.png
  threshold/time_to_threshold.png
  critic_loss/critic_loss_curve.png
  signal_variance/signal_variance_curve.png
```