#!/usr/bin/env python3
"""LOCAL fill-in runner: HalfCheetah-v4 seed10, eps020, the two MISSING cells.

The three-algo comparison is missing two runs at eps=0.20 seed10:
    - ippo_eps020_seed10
    - stage_mappo_eps020_seed10   (WITH delta^o)
Both are produced here so the eps020 column reaches n=4. wandb names/groups
match the comparison plot's strict regex (no suffix => with-delta / ippo).

Run from the repo root with the mujoko env:
    PYTHONPATH=$PWD nohup /home/mingjun/miniconda3/envs/mujoko/bin/python -u \
        examples/robust_attack/run_eps020_seed10_fill.py \
        > logs/campaign/hc_eps020_seed10_fill.log 2>&1 &
    tail -f logs/campaign/STATUS_eps020_fill.txt
"""
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable
TRAIN = os.path.join("examples", "robust_attack", "train_robust_attacker.py")
LOGDIR = os.path.join(REPO, "logs", "campaign")
STATUS = os.path.join(LOGDIR, "STATUS_eps020_fill.txt")

ENV = "HalfCheetah-v4"
SEED = 10
EPS_TAG = "020"
EPS_VAL = 0.2

CONC = int(os.environ.get("CONC", 2))
THREADS = int(os.environ.get("THREADS", 8))
STEPS = int(os.environ.get("STEPS", 6_000_000))
USE_WANDB = os.environ.get("USE_WANDB", "True")
STAGE_LAMBDA = os.environ.get("STAGE_LAMBDA", "0.95")


def log_status(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def marker(name):
    return os.path.join(LOGDIR, name + ".done")


def victim_ckpt(seed):
    pats = glob.glob(
        f"{REPO}/results/robust_victim/{ENV}/mappo/*/seed-{seed:05d}-*/models/actor_agent0.pt"
    )
    pats = sorted(pats, key=os.path.getmtime)
    return pats[-1] if pats else None


def attack_cmd(algo, vpath):
    """algo in {ippo, stage_mappo}; stage_mappo here is WITH delta^o."""
    expname = f"attack_{algo}_hc_eps{EPS_TAG}_seed{SEED}"
    cmd = [
        PY, "-u", TRAIN,
        "--algo", algo, "--env", "robust_attack", "--exp_name", expname,
        "--scenario", ENV,
        "--epsilon_observation", str(EPS_VAL), "--epsilon_action", str(EPS_VAL),
        "--model_path", vpath,
        "--num_env_steps", str(STEPS), "--seed", str(SEED),
        "--n_rollout_threads", str(THREADS),
        "--share_param", "False",
        "--use_wandb", USE_WANDB,
        "--wandb_project", f"robust_attack_{ENV}",
        "--wandb_group", f"hc_eps{EPS_TAG}",
        "--wandb_name", f"{algo}_eps{EPS_TAG}_seed{SEED}",
    ]
    if algo == "stage_mappo":
        # WITH delta^o (no --stage_drop_delta_o). Keep stage_lambda high.
        cmd += ["--state_type", "FP", "--causal_critic_state", "True",
                "--stage_lambda", STAGE_LAMBDA]
    return expname, cmd


def run_pool(jobs, conc):
    queue = list(jobs)
    running = []
    while queue or running:
        while queue and len(running) < conc:
            name, cmd = queue.pop(0)
            logf = open(os.path.join(LOGDIR, name + ".log"), "w")
            env = dict(os.environ)
            env["PYTHONPATH"] = REPO
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            p = subprocess.Popen(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT, env=env)
            running.append((name, p, logf))
            log_status(f"START {name} (pid {p.pid})  [{len(queue)} queued, {len(running)} running]")
        time.sleep(15)
        still = []
        for name, p, logf in running:
            rc = p.poll()
            if rc is None:
                still.append((name, p, logf))
            else:
                logf.close()
                if rc == 0:
                    open(marker(name), "w").close()
                    log_status(f"DONE {name} rc=0")
                else:
                    log_status(f"FAIL {name} rc={rc}")
        running = still


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    log_status("==== HC eps020 seed10 FILL START ====")
    log_status(f"REPO={REPO}")
    log_status(f"PY={PY}")
    log_status(f"conc={CONC} threads={THREADS} steps={STEPS} stage_lambda={STAGE_LAMBDA} wandb={USE_WANDB}")

    vpath = victim_ckpt(SEED)
    if vpath is None:
        log_status(f"ERROR no HalfCheetah victim ckpt for seed{SEED}; aborting")
        return 1
    log_status(f"victim ckpt: {vpath}")

    jobs = []
    for algo in ["ippo", "stage_mappo"]:
        name, cmd = attack_cmd(algo, vpath)
        if os.path.exists(marker(name)):
            log_status(f"SKIP {name} (done marker exists)")
            continue
        jobs.append((name, cmd))

    if not jobs:
        log_status("nothing to do; all markers present")
        return 0

    run_pool(jobs, CONC)
    log_status("==== HC eps020 seed10 FILL COMPLETE ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
