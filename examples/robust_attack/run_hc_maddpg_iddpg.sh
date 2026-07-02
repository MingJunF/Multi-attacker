#!/usr/bin/env bash
# Launch BOTH the maddpg and iddpg attack campaigns for HalfCheetah-v4 only.
#
# HalfCheetah's seed set is {1, 500, 1000} (hard-coded in each campaign's
# ENV_SEEDS), so TARGETS=HalfCheetah-v4 runs exactly those three seeds x four
# eps {005,010,015,020} = 12 runs per algo (24 total). Both campaigns are
# idempotent: a run whose .done marker or final actor_agent0.pt already exists
# is skipped, and any (env, seed) whose victim checkpoint is missing is skipped
# with an error line (train the HC victims first).
#
# Usage (run from the repo root, in the mujoko env):
#     bash examples/robust_attack/run_hc_maddpg_iddpg.sh
#
# Env vars (optional):
#     CONC   concurrent runs PER campaign            (default 2)
#     STEPS  --num_env_steps per run                 (default 6000000)
#     PY     python interpreter                      (default: python)
set -u
cd "$(dirname "$0")/../.." || exit 1

PY=${PY:-python}
CONC=${CONC:-2}
STEPS=${STEPS:-6000000}

export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p logs/campaign

echo "launching HalfCheetah-v4 maddpg + iddpg (seeds {1,500,1000}, CONC=$CONC, STEPS=$STEPS)"

TARGETS=HalfCheetah-v4 CONC="$CONC" STEPS="$STEPS" nohup "$PY" -u \
    examples/robust_attack/run_maddpg_campaign.py \
    > logs/campaign/maddpg_hc.log 2>&1 &
echo "  maddpg campaign pid=$!  -> logs/campaign/maddpg_hc.log"

TARGETS=HalfCheetah-v4 CONC="$CONC" STEPS="$STEPS" nohup "$PY" -u \
    examples/robust_attack/run_iddpg_campaign.py \
    > logs/campaign/iddpg_hc.log 2>&1 &
echo "  iddpg  campaign pid=$!  -> logs/campaign/iddpg_hc.log"

echo "done. tail the logs or STATUS_{maddpg,iddpg}.txt to monitor."
