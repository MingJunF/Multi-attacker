#!/usr/bin/env python3
"""Focused launcher: attack the STRONG HalfCheetah seed10 victim (~4518).

Runs only:  HC seed10, eps {0.1, 0.15}, algos {ippo, stage_mappo}  -> 4 runs.
Concurrency 2, n_rollout_threads 8, num_env_steps 6,000,000.
(HC seed10 eps0.2 already exists: attack_ippo_eps02 / attack_stagemappo_eps02.)
Idempotent: skips a run whose models/actor_agent0.pt already exists.
"""
import glob
import os
import subprocess
import sys
import time

REPO = "/mnt/d/Github_Code/uncertainty/Robust-Gymnasium"
PY = "/home/mingjun/miniconda3/envs/mujoko/bin/python"
TRAIN = "examples/robust_attack/train_robust_attacker.py"
LOGDIR = os.path.join(REPO, "logs", "campaign")
STATUS = os.path.join(LOGDIR, "STATUS_hc_seed10.txt")

ENV = "HalfCheetah-v4"
SEED = 10
THREADS = 8
CONC = 2
STEPS = 6_000_000
EPSES = [("010", 0.1), ("015", 0.15)]
ALGOS = ["ippo", "stage_mappo"]


def log_status(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def victim_ckpt():
    pats = glob.glob(
        f"{REPO}/results/robust_victim/{ENV}/mappo/*/seed-{SEED:05d}-*/models/actor_agent0.pt"
    )
    pats = sorted(pats, key=os.path.getmtime)
    return pats[-1] if pats else None


def attack_done(algo, expname):
    pats = glob.glob(
        f"{REPO}/results/robust_attack/{ENV}/{algo}/{expname}/seed-{SEED:05d}-*/models/actor_agent0.pt"
    )
    return bool(pats)


def attack_cmd(eps_tag, eps_val, algo, vpath):
    expname = f"attack_{algo}_hc_eps{eps_tag}_seed{SEED}"
    cmd = [
        PY, "-u", TRAIN,
        "--algo", algo, "--env", "robust_attack", "--exp_name", expname,
        "--scenario", ENV,
        "--epsilon_observation", str(eps_val), "--epsilon_action", str(eps_val),
        "--model_path", vpath,
        "--num_env_steps", str(STEPS), "--seed", str(SEED),
        "--n_rollout_threads", str(THREADS),
        "--share_param", "False",
        "--use_wandb", "True",
        "--wandb_project", f"robust_attack_{ENV}",
        "--wandb_group", f"hc_eps{eps_tag}",
        "--wandb_name", f"{algo}_eps{eps_tag}_seed{SEED}",
    ]
    if algo == "stage_mappo":
        cmd += ["--state_type", "FP", "--causal_critic_state", "True"]
    return expname, cmd


def run_pool(jobs):
    queue = list(jobs)
    running = []
    while queue or running:
        while queue and len(running) < CONC:
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
                log_status(f"{'DONE' if rc == 0 else 'FAIL'} {name} rc={rc}")
        running = still


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    log_status("==== HC seed10 focused run START ====")
    vpath = victim_ckpt()
    if vpath is None:
        log_status("ERROR: no HC seed10 victim checkpoint found")
        return 1
    log_status(f"victim: {vpath}")
    jobs = []
    # eps outer, algo inner -> pair {ippo, stage} of same eps run together (2 slots)
    for eps_tag, eps_val in EPSES:
        for algo in ALGOS:
            expname, cmd = attack_cmd(eps_tag, eps_val, algo, vpath)
            if attack_done(algo, expname):
                log_status(f"SKIP {expname} (already done)")
                continue
            jobs.append((expname, cmd))
    log_status(f"{len(jobs)} attack runs to train")
    run_pool(jobs)
    log_status("==== HC seed10 focused run COMPLETE ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
