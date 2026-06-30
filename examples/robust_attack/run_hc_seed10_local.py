#!/usr/bin/env python3
"""LOCAL (this machine) runner: HalfCheetah-v4 seed10 nodelta attacks.

Runs the nodelta stage_mappo attacker (--stage_drop_delta_o True, --stage_lambda 0.95)
for HalfCheetah-v4 seed 10 across eps {0.05, 0.10, 0.15, 0.20}, concurrency 2.
This complements the server campaign, which intentionally SKIPS HalfCheetah seed10.

wandb names/groups match the server campaign so the comparison plot picks them up:
    name  = stage_mappo_nodelta_eps{tag}_seed10
    group = hc_eps{tag}_nodelta
    proj  = robust_attack_HalfCheetah-v4

Run from the repo root with an env that has torch+mujoco (e.g. the `mujoko` env):
    PYTHONPATH=$PWD nohup /home/mingjun/miniconda3/envs/mujoko/bin/python -u \
        examples/robust_attack/run_hc_seed10_local.py \
        > logs/campaign/hc_seed10_local.log 2>&1 &
    tail -f logs/campaign/STATUS_hc_seed10.txt
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
STATUS = os.path.join(LOGDIR, "STATUS_hc_seed10.txt")

ENV = "HalfCheetah-v4"
SEED = 10
EPSES = [("005", 0.05), ("010", 0.1), ("015", 0.15), ("020", 0.2)]

CONC = int(os.environ.get("CONC", 2))           # 2 parallel runs on this machine
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


def attack_done(expname, seed):
    # complete only if it finished cleanly (marker); periodic ckpts don't count
    return os.path.exists(marker(expname))


def attack_cmd(eps_tag, eps_val, vpath):
    expname = f"attack_stage_mappo_nodelta_hc_eps{eps_tag}_seed{SEED}"
    cmd = [
        PY, "-u", TRAIN,
        "--algo", "stage_mappo", "--env", "robust_attack", "--exp_name", expname,
        "--scenario", ENV,
        "--epsilon_observation", str(eps_val), "--epsilon_action", str(eps_val),
        "--model_path", vpath,
        "--num_env_steps", str(STEPS), "--seed", str(SEED),
        "--n_rollout_threads", str(THREADS),
        "--share_param", "False",
        "--state_type", "FP", "--causal_critic_state", "True",
        "--stage_lambda", STAGE_LAMBDA, "--stage_drop_delta_o", "True",
        "--use_wandb", USE_WANDB,
        "--wandb_project", f"robust_attack_{ENV}",
        "--wandb_group", f"hc_eps{eps_tag}_nodelta",
        "--wandb_name", f"stage_mappo_nodelta_eps{eps_tag}_seed{SEED}",
    ]
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
    log_status("==== HC seed10 LOCAL CAMPAIGN START ====")
    log_status(f"REPO={REPO}")
    log_status(f"PY={PY}")
    log_status(f"conc={CONC} threads={THREADS} steps={STEPS} stage_lambda={STAGE_LAMBDA} wandb={USE_WANDB}")

    vpath = victim_ckpt(SEED)
    if vpath is None:
        log_status(f"ERROR no HalfCheetah victim ckpt for seed{SEED}; aborting")
        return 1
    log_status(f"victim ckpt: {vpath}")

    jobs = []
    for eps_tag, eps_val in EPSES:
        expname, cmd = attack_cmd(eps_tag, eps_val, vpath)
        if attack_done(expname, SEED):
            log_status(f"SKIP {expname} (marker exists)")
            continue
        jobs.append((expname, cmd))
    log_status(f"{len(jobs)} attack run(s) to train")
    run_pool(jobs, CONC)
    log_status("==== HC seed10 LOCAL CAMPAIGN COMPLETE ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
