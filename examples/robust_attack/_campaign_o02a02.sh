#!/usr/bin/env bash
# Formal experiment campaign: IPPO vs Stage-Aware MAPPO (two-head) attackers.
# Setting: obs eps = act eps = 0.2 (default), 5M steps each, seeds 1/10/100/1000/10000.
# Per map (Ant-v4) wandb project; group = <algo>_<eps tag> so the 5 seeds of one
# algo+setting average together; run name carries algo+eps (NOT seed). Models are
# auto-saved every eval_interval to each run's results .../models/ dir.
#
# Concurrency: each wave runs ONE stage_mappo + ONE ippo (different argv tags ->
# distinguishable; 2 x 9 = 18 procs on 16 cores). Waves run sequentially.
set -u

REPO=/mnt/d/Github_Code/uncertainty/Robust-Gymnasium
PY=/home/mingjun/miniconda3/envs/mujoko/bin/python
LOGDIR=/tmp/ar_cmp
PROJECT=robust_attack_Ant-v4
ENTITY=mingjun-fan-university-of-south-australia
STEPS=5000000
EPS_O=0.2
EPS_A=0.2
TAG=o02a02
SEEDS=(1 10 100 1000 10000)

mkdir -p "$LOGDIR"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"

running_tag() {  # $1 = exp_name tag; 0 if a live proc carries robust_attack-<tag>
    for d in /proc/[0-9]*; do
        cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
        echo "$cmd" | grep -q "robust_attack-$1" && return 0
    done
    return 1
}

launch_stage() {  # $1 = seed
    nohup $PY -u examples/robust_attack/train_robust_attacker.py \
        --algo stage_mappo --env robust_attack --exp_name stage_mappo_${TAG} \
        --seed "$1" --epsilon_observation $EPS_O --epsilon_action $EPS_A \
        --num_env_steps $STEPS --n_rollout_threads 8 --share_param False \
        --causal_critic_state True --state_type FP --two_head_critic True \
        --use_wandb True --wandb_entity $ENTITY --wandb_project $PROJECT \
        --wandb_group stage_mappo_${TAG} --wandb_name stage_mappo_${TAG} \
        > "$LOGDIR/stage_mappo_${TAG}_s$1.log" 2>&1 &
    echo "[sched] launched stage_mappo_${TAG} seed=$1 PID $!"
}

launch_ippo() {  # $1 = seed
    nohup $PY -u examples/robust_attack/train_robust_attacker.py \
        --algo ippo --env robust_attack --exp_name ippo_${TAG} \
        --seed "$1" --epsilon_observation $EPS_O --epsilon_action $EPS_A \
        --num_env_steps $STEPS --n_rollout_threads 8 --share_param False \
        --use_wandb True --wandb_entity $ENTITY --wandb_project $PROJECT \
        --wandb_group ippo_${TAG} --wandb_name ippo_${TAG} \
        > "$LOGDIR/ippo_${TAG}_s$1.log" 2>&1 &
    echo "[sched] launched ippo_${TAG} seed=$1 PID $!"
}

echo "[sched] $(date) START campaign: 2 algos x ${#SEEDS[@]} seeds, ${STEPS} steps, eps ${EPS_O}/${EPS_A}"
for s in "${SEEDS[@]}"; do
    echo "[sched] $(date) ===== wave seed=$s ====="
    launch_stage "$s"
    launch_ippo "$s"
    sleep 30  # stagger startup / let argv tags appear
    while running_tag "stage_mappo_${TAG}" || running_tag "ippo_${TAG}"; do
        sleep 120
    done
    echo "[sched] $(date) wave seed=$s finished"
done
echo "[sched] $(date) ===== ALL WAVES DONE ====="
