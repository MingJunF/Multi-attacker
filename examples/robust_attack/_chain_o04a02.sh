#!/usr/bin/env bash
# Wait for the current eps 0.2/0.2 runs (STAGE_MAPPO_twohead_o02a02 + IPPO_o02a02)
# to finish, then launch the eps obs0.4/act0.2 pair (same seed 10, t8).
set -u

REPO=/mnt/d/Github_Code/uncertainty/Robust-Gymnasium
PY=/home/mingjun/miniconda3/envs/mujoko/bin/python
LOGDIR=/tmp/ar_cmp
WAIT_TAGS="robust_attack-STAGE_MAPPO_twohead_o02a02|robust_attack-IPPO_o02a02"

mkdir -p "$LOGDIR"
echo "[chain] $(date) waiting for o02a02 runs to finish..."

# Poll /proc for the running exp-name tags (the train script rewrites argv to
# robust_attack-<expname>, so grep the cmdline tag, not the python path).
running() {
    for d in /proc/[0-9]*; do
        cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
        echo "$cmd" | grep -qE "$WAIT_TAGS" && return 0
    done
    return 1
}

while running; do
    sleep 120
done

echo "[chain] $(date) o02a02 finished. Launching o04a02 pair."
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"

COMMON="--env robust_attack --seed 10 --epsilon_observation 0.4 --epsilon_action 0.2 \
--num_env_steps 4050000 --n_rollout_threads 8 --share_param False \
--use_wandb True --wandb_entity mingjun-fan-university-of-south-australia \
--wandb_project robust-gymnasium --wandb_group ippo_ar_seed10_o04a02"

nohup $PY -u examples/robust_attack/train_robust_attacker.py \
    --algo stage_mappo --exp_name STAGE_MAPPO_twohead_o04a02 \
    --causal_critic_state True --state_type FP --two_head_critic True \
    $COMMON --wandb_name STAGE_MAPPO_twohead_o04a02 \
    > "$LOGDIR/twohead_o04a02.log" 2>&1 &
echo "[chain] launched STAGE_MAPPO_twohead_o04a02 PID $!"

nohup $PY -u examples/robust_attack/train_robust_attacker.py \
    --algo ippo --exp_name IPPO_o04a02 \
    $COMMON --wandb_name IPPO_o04a02 \
    > "$LOGDIR/ippo_o04a02.log" 2>&1 &
echo "[chain] launched IPPO_o04a02 PID $!"

echo "[chain] $(date) done."
