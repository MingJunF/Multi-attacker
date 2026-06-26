#!/usr/bin/env python3
"""Server campaign for HalfCheetah robustness attacks (portable + resumable).

Phase 1 -- train victims:   HalfCheetah-v4, seed {100000, 500}, --n_rollout_threads 16
Phase 2 -- train attackers:  for each victim seed, eps {0.05, 0.1, 0.15, 0.2}
           x algo {ippo, stage_mappo}, --n_rollout_threads 8
           (attacker seed == victim seed; each attacker loads its own victim ckpt)

Design notes
------------
* PORTABLE: REPO is inferred from this file's location; the Python interpreter is
  `sys.executable` (whatever runs this script). No machine-specific paths.
* RESUMABLE / IDEMPOTENT: a run is considered complete only when it exits rc==0,
  recorded by a marker file logs/campaign/<name>.done. Re-running the script skips
  completed runs and resumes the rest. (We do NOT rely on the periodic checkpoint
  files, which exist even for half-finished runs.)
* Phase 1 fully completes before Phase 2 starts (attackers need final victim ckpts).

Tunables (override via environment variables)
---------------------------------------------
  VICTIM_THREADS  (default 16)      ATTACK_THREADS (default 8)
  VICTIM_CONC     (default 2)       ATTACK_CONC    (default 2)
  VICTIM_STEPS    (default 6000000) ATTACK_STEPS   (default 6000000)
  USE_WANDB       (default True)    EVAL_INTERVAL  (default 25)

How to run on the rented server
-------------------------------
  cd <repo-root>
  conda activate <your-env>            # env with the project + mujoco installed
  # one of:  wandb login   |   export WANDB_API_KEY=...   |   USE_WANDB=False
  nohup python -u examples/robust_attack/run_server.py \
        > logs/campaign/server_campaign.log 2>&1 &
  # monitor:  tail -f logs/campaign/STATUS_server.txt
"""
import glob
import os
import subprocess
import sys
import time

# --- portable paths ---------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))            # .../examples/robust_attack
REPO = os.path.dirname(os.path.dirname(HERE))                # repo root
PY = sys.executable                                          # current interpreter
TRAIN = os.path.join("examples", "robust_attack", "train_robust_attacker.py")
LOGDIR = os.path.join(REPO, "logs", "campaign")
STATUS = os.path.join(LOGDIR, "STATUS_server.txt")

# --- tunables ---------------------------------------------------------------
def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default

ENV = "HalfCheetah-v4"
SEEDS = [100000, 500]
EPSES = [("005", 0.05), ("010", 0.1), ("015", 0.15), ("020", 0.2)]
ALGOS = ["ippo", "stage_mappo"]

VICTIM_THREADS = _int("VICTIM_THREADS", 16)
ATTACK_THREADS = _int("ATTACK_THREADS", 8)
VICTIM_CONC = _int("VICTIM_CONC", 2)
ATTACK_CONC = _int("ATTACK_CONC", 2)
VICTIM_STEPS = _int("VICTIM_STEPS", 6_000_000)
ATTACK_STEPS = _int("ATTACK_STEPS", 6_000_000)
EVAL_INTERVAL = _int("EVAL_INTERVAL", 25)
USE_WANDB = os.environ.get("USE_WANDB", "True")

VICTIM_EXP = "victim_hc16"   # exp_name; results/robust_victim/<ENV>/mappo/victim_hc16/seed-*/


# --- helpers ----------------------------------------------------------------
def log_status(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def marker(name):
    return os.path.join(LOGDIR, name + ".done")


def is_done(name):
    return os.path.exists(marker(name))


def victim_ckpt(seed):
    """Final actor ckpt of OUR trained victim (victim_hc16) for this seed."""
    pats = glob.glob(
        f"{REPO}/results/robust_victim/{ENV}/mappo/{VICTIM_EXP}/seed-{seed:05d}-*/models/actor_agent0.pt"
    )
    pats = sorted(pats, key=os.path.getmtime)
    return pats[-1] if pats else None


def victim_cmd(seed):
    name = f"{VICTIM_EXP}_seed{seed}"
    cmd = [
        PY, "-u", TRAIN,
        "--algo", "mappo", "--env", "robust_victim", "--exp_name", VICTIM_EXP,
        "--scenario", ENV, "--seed", str(seed),
        "--share_param", "True",
        "--n_rollout_threads", str(VICTIM_THREADS),
        "--num_env_steps", str(VICTIM_STEPS),
        "--eval_interval", str(EVAL_INTERVAL),
        "--use_wandb", USE_WANDB,
        "--wandb_project", f"robust_victim_{ENV}",
        "--wandb_group", VICTIM_EXP,
        "--wandb_name", f"seed{seed}",
    ]
    return name, cmd


def attack_cmd(seed, eps_tag, eps_val, algo, vpath):
    name = f"attack_{algo}_hc_eps{eps_tag}_seed{seed}"
    cmd = [
        PY, "-u", TRAIN,
        "--algo", algo, "--env", "robust_attack", "--exp_name", name,
        "--scenario", ENV,
        "--epsilon_observation", str(eps_val), "--epsilon_action", str(eps_val),
        "--model_path", vpath,
        "--num_env_steps", str(ATTACK_STEPS), "--seed", str(seed),
        "--n_rollout_threads", str(ATTACK_THREADS),
        "--share_param", "False",
        "--use_wandb", USE_WANDB,
        "--wandb_project", f"robust_attack_{ENV}",
        "--wandb_group", f"hc_eps{eps_tag}",
        "--wandb_name", f"{algo}_eps{eps_tag}_seed{seed}",
    ]
    if algo == "stage_mappo":
        cmd += ["--state_type", "FP", "--causal_critic_state", "True"]
    return name, cmd


def run_pool(jobs, conc):
    """Run (name, cmd) jobs with at most `conc` concurrent subprocesses."""
    queue = list(jobs)
    running = []  # (name, proc, logf)
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
                    open(marker(name), "w").close()  # completion marker
                    log_status(f"DONE {name} rc=0")
                else:
                    log_status(f"FAIL {name} rc={rc}")
        running = still


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    log_status("==== SERVER CAMPAIGN START ====")
    log_status(f"REPO={REPO}")
    log_status(f"PY={PY}")
    log_status(
        f"victim: threads={VICTIM_THREADS} conc={VICTIM_CONC} steps={VICTIM_STEPS} | "
        f"attack: threads={ATTACK_THREADS} conc={ATTACK_CONC} steps={ATTACK_STEPS} | wandb={USE_WANDB}"
    )

    # ---- Phase 1: victims (seed 1, 100) ----
    v_jobs = []
    for seed in SEEDS:
        name, cmd = victim_cmd(seed)
        if is_done(name):
            log_status(f"SKIP {name} (already done)")
            continue
        v_jobs.append((name, cmd))
    log_status(f"Phase 1: {len(v_jobs)} victim run(s) to train")
    run_pool(v_jobs, VICTIM_CONC)
    log_status("==== Phase 1 (victims) COMPLETE ====")

    # ---- Phase 2: attacks, ordered seed -> eps -> algo (pair runs of same eps) ----
    a_jobs = []
    for seed in SEEDS:
        vpath = victim_ckpt(seed)
        if vpath is None:
            log_status(f"ERROR no victim ckpt for seed{seed}; skipping its attacks")
            continue
        for eps_tag, eps_val in EPSES:
            for algo in ALGOS:
                name, cmd = attack_cmd(seed, eps_tag, eps_val, algo, vpath)
                if is_done(name):
                    log_status(f"SKIP {name} (already done)")
                    continue
                a_jobs.append((name, cmd))
    log_status(f"Phase 2: {len(a_jobs)} attack run(s) to train")
    run_pool(a_jobs, ATTACK_CONC)
    log_status("==== SERVER CAMPAIGN COMPLETE ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
