#!/usr/bin/env python3
"""Campaign scheduler: train missing victims, then run the attack sweep.

Policy (per user request 2026-06-25):
  * concurrency = 2 runs at a time, each --n_rollout_threads 8 (16 cores total)
  * all NEW runs use --num_env_steps 6_000_000
  * Phase 1: finish the 4 missing victims FIRST (6 total, 2 already done):
        Ant-v4 seed 10, Ant-v4 seed 100, HalfCheetah-v4 seed 1, HalfCheetah-v4 seed 100
    (Ant-v4 seed 1 and HalfCheetah-v4 seed 10 already exist -> reused.)
  * Phase 2: attack sweep ordered by  seed(outer,1->10->100) -> eps -> env(map).
    i.e. for a fixed seed, finish ALL eps x map first, THEN move to the next seed
    (seed changes LAST). Each (seed, eps, env) launches the pair {ippo, stage_mappo}
    together (= the 2 parallel slots), so an ippo-vs-stage comparison finishes side by side.
    Skips:  Ant eps0.2 (already done, all seeds); HalfCheetah eps0.2 seed 10 (done).

Each attack resolves its victim checkpoint dynamically (attacker seed == victim seed).
Idempotent: a job whose models/actor_agent0.pt already exists is skipped, so the
scheduler can be re-run to resume.
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
STATUS = os.path.join(LOGDIR, "STATUS.txt")

THREADS = 8
CONC = 2
STEPS = 6_000_000

SEEDS = [1, 10, 100]
# (tag, value) ; ascending so fresh eps0.1 results come first
EPSES = [("010", 0.1), ("015", 0.15), ("020", 0.2)]
ENVS = ["Ant-v4", "HalfCheetah-v4"]  # "map" order
ALGOS = ["ippo", "stage_mappo"]

# Victims that still need training (the other 2 already exist and are reused).
VICTIM_JOBS = [("Ant-v4", 10), ("Ant-v4", 100), ("HalfCheetah-v4", 1), ("HalfCheetah-v4", 100)]

# Attack (env, eps_tag, seed) cells to SKIP because they are already done.
SKIP_ATTACK = set()
for s in SEEDS:
    SKIP_ATTACK.add(("Ant-v4", "020", s))      # Ant eps0.2 grandfathered as done
SKIP_ATTACK.add(("HalfCheetah-v4", "020", 10))  # HC eps0.2 seed10 already done


def short(env_full):
    return "ant" if env_full.startswith("Ant") else "hc"


def log_status(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def victim_ckpt(env_full, seed):
    """Latest victim actor checkpoint for (env, seed), or None."""
    pats = glob.glob(
        f"{REPO}/results/robust_victim/{env_full}/mappo/*/seed-{seed:05d}-*/models/actor_agent0.pt"
    )
    pats = sorted(pats, key=os.path.getmtime)
    return pats[-1] if pats else None


def victim6m_done(env_full, seed):
    pats = glob.glob(
        f"{REPO}/results/robust_victim/{env_full}/mappo/victim6m/seed-{seed:05d}-*/models/actor_agent0.pt"
    )
    return bool(pats)


def attack_done(env_full, algo, expname, seed):
    pats = glob.glob(
        f"{REPO}/results/robust_attack/{env_full}/{algo}/{expname}/seed-{seed:05d}-*/models/actor_agent0.pt"
    )
    return bool(pats)


def victim_cmd(env_full, seed):
    name = f"victim6m_{short(env_full)}_seed{seed}"
    cmd = [
        PY, "-u", TRAIN,
        "--algo", "mappo", "--env", "robust_victim", "--exp_name", "victim6m",
        "--scenario", env_full, "--seed", str(seed),
        "--share_param", "True",
        "--n_rollout_threads", str(THREADS),
        "--num_env_steps", str(STEPS),
        "--eval_interval", "25",
        "--use_wandb", "True",
        "--wandb_project", f"robust_victim_{env_full}",
        "--wandb_group", "victim6m",
        "--wandb_name", f"seed{seed}",
    ]
    return name, cmd


def attack_cmd(env_full, eps_tag, eps_val, seed, algo, vpath):
    expname = f"attack_{algo}_{short(env_full)}_eps{eps_tag}_seed{seed}"
    cmd = [
        PY, "-u", TRAIN,
        "--algo", algo, "--env", "robust_attack", "--exp_name", expname,
        "--scenario", env_full,
        "--epsilon_observation", str(eps_val), "--epsilon_action", str(eps_val),
        "--model_path", vpath,
        "--num_env_steps", str(STEPS), "--seed", str(seed),
        "--n_rollout_threads", str(THREADS),
        "--share_param", "False",
        "--use_wandb", "True",
        "--wandb_project", f"robust_attack_{env_full}",
        "--wandb_group", f"{short(env_full)}_eps{eps_tag}",
        "--wandb_name", f"{algo}_eps{eps_tag}_seed{seed}",
    ]
    if algo == "stage_mappo":
        cmd += ["--state_type", "FP", "--causal_critic_state", "True"]
    return expname, cmd


def run_pool(jobs):
    """Run (name, cmd) jobs with at most CONC concurrent subprocesses."""
    queue = list(jobs)
    running = []  # (name, proc, logf)
    while queue or running:
        while queue and len(running) < CONC:
            name, cmd = queue.pop(0)
            logf = open(os.path.join(LOGDIR, name + ".log"), "w")
            env = dict(os.environ)
            env["PYTHONPATH"] = REPO
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            p = subprocess.Popen(
                cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT, env=env
            )
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
    log_status("==== CAMPAIGN START ====")

    # ---- Phase 1: victims ----
    v_jobs = []
    for env_full, seed in VICTIM_JOBS:
        if victim6m_done(env_full, seed):
            log_status(f"SKIP victim {short(env_full)} seed{seed} (already trained)")
            continue
        v_jobs.append(victim_cmd(env_full, seed))
    log_status(f"Phase 1: {len(v_jobs)} victim runs to train")
    run_pool(v_jobs)
    log_status("==== Phase 1 (victims) COMPLETE ====")

    # ---- Phase 2: attacks, ordered seed(outer) -> eps -> env(map), pair {ippo, stage} together ----
    a_jobs = []
    for seed in SEEDS:
        for eps_tag, eps_val in EPSES:
            for env_full in ENVS:
                if (env_full, eps_tag, seed) in SKIP_ATTACK:
                    continue
                vpath = victim_ckpt(env_full, seed)
                if vpath is None:
                    log_status(f"ERROR no victim for {short(env_full)} seed{seed}; skipping eps{eps_tag}")
                    continue
                for algo in ALGOS:
                    expname, cmd = attack_cmd(env_full, eps_tag, eps_val, seed, algo, vpath)
                    if attack_done(env_full, algo, expname, seed):
                        log_status(f"SKIP {expname} (already done)")
                        continue
                    a_jobs.append((expname, cmd))
    log_status(f"Phase 2: {len(a_jobs)} attack runs to train")
    run_pool(a_jobs)
    log_status("==== CAMPAIGN COMPLETE ====")


if __name__ == "__main__":
    sys.exit(main())
