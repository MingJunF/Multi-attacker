#!/usr/bin/env python3
"""Nodelta (drop delta^o) attack campaign: HalfCheetah first, then Ant.

This scheduler runs the *nodelta* stage_mappo attacker (--stage_drop_delta_o True,
i.e. the obs-attacker advantage no longer carries delta^o, which was found to work
very well) together with the necessary victims and the Ant ippo baseline.

Phases (run strictly in order; concurrency = 3, --n_rollout_threads 8 each):

  Phase HC-V  : train any MISSING HalfCheetah-v4 victims for seeds {1, 10, 500, 1000}.
  Phase HC-A  : nodelta stage_mappo attacks, HalfCheetah-v4,
                seeds {1, 10, 500, 1000} x eps {0.05, 0.10, 0.15, 0.20}.
  Phase ANT-V : train any MISSING Ant-v4 victims for seeds {1, 10, 100, 1000, 10000}.
  Phase ANT-A : per (seed, eps) launch BOTH {ippo, nodelta stage_mappo}, Ant-v4,
                seeds {1, 10, 100, 1000, 10000} x eps {0.05, 0.10, 0.15, 0.20}.

Design notes
------------
* PORTABLE: REPO inferred from this file; interpreter is `sys.executable`
  (the python that runs this script -- run it from inside your conda env).
* RESUMABLE / IDEMPOTENT:
    - a victim is skipped if ANY actor_agent0.pt for that (env, seed) already exists
      (under any exp_name), so previously trained victims are reused.
    - an attack is skipped if it has a completion marker logs/campaign/<name>.done
      OR its final actor_agent0.pt already exists.
* Each victim is trained ONCE; attacks resolve their victim ckpt dynamically
  (attacker seed == victim seed).

How to run on the server (3 parallel runs supported)
----------------------------------------------------
  cd <repo-root>
  conda activate <your-env>            # env with the project + mujoco installed
  # wandb:  wandb login   |   export WANDB_API_KEY=...   |   USE_WANDB=False
  nohup python -u examples/robust_attack/run_nodelta_campaign.py \
        > logs/campaign/nodelta_campaign.log 2>&1 &
  # monitor:  tail -f logs/campaign/STATUS_nodelta.txt
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
STATUS = os.path.join(LOGDIR, "STATUS_nodelta.txt")

# --- tunables (override via environment variables) --------------------------
def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default

CONC = _int("CONC", 3)                       # server supports 3 parallel runs
THREADS = _int("THREADS", 8)                 # --n_rollout_threads per run
STEPS = _int("STEPS", 6_000_000)             # --num_env_steps for victims and attacks
EVAL_INTERVAL = _int("EVAL_INTERVAL", 25)
USE_WANDB = os.environ.get("USE_WANDB", "True")

# Phase plan ----------------------------------------------------------------
EPSES = [("005", 0.05), ("010", 0.1), ("015", 0.15), ("020", 0.2)]

HC_ENV = "HalfCheetah-v4"
HC_SEEDS = [1, 500, 1000]   # HalfCheetah seed10 intentionally excluded

ANT_ENV = "Ant-v4"
ANT_SEEDS = [1, 10, 100, 1000, 10000]

VICTIM_EXP = "victim6m"   # exp_name used when we have to TRAIN a missing victim


# --- helpers ----------------------------------------------------------------
def short(env_full):
    return "ant" if env_full.startswith("Ant") else "hc"


def log_status(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def marker(name):
    return os.path.join(LOGDIR, name + ".done")


def is_done(name):
    return os.path.exists(marker(name))


def victim_ckpt(env_full, seed):
    """Latest victim actor ckpt for (env, seed) under ANY exp_name, or None."""
    pats = glob.glob(
        f"{REPO}/results/robust_victim/{env_full}/mappo/*/seed-{seed:05d}-*/models/actor_agent0.pt"
    )
    pats = sorted(pats, key=os.path.getmtime)
    return pats[-1] if pats else None


def attack_ckpt(env_full, algo, expname, seed):
    pats = glob.glob(
        f"{REPO}/results/robust_attack/{env_full}/{algo}/{expname}/seed-{seed:05d}-*/models/actor_agent0.pt"
    )
    return bool(pats)


def victim_cmd(env_full, seed):
    name = f"{VICTIM_EXP}_{short(env_full)}_seed{seed}"
    cmd = [
        PY, "-u", TRAIN,
        "--algo", "mappo", "--env", "robust_victim", "--exp_name", VICTIM_EXP,
        "--scenario", env_full, "--seed", str(seed),
        "--share_param", "True",
        "--n_rollout_threads", str(THREADS),
        "--num_env_steps", str(STEPS),
        "--eval_interval", str(EVAL_INTERVAL),
        "--use_wandb", USE_WANDB,
        "--wandb_project", f"robust_victim_{env_full}",
        "--wandb_group", VICTIM_EXP,
        "--wandb_name", f"seed{seed}",
    ]
    return name, cmd


def attack_cmd(env_full, eps_tag, eps_val, seed, algo, vpath, nodelta):
    """algo in {ippo, stage_mappo}. nodelta only meaningful for stage_mappo."""
    suffix = "_nodelta" if (nodelta and algo == "stage_mappo") else ""
    expname = f"attack_{algo}{suffix}_{short(env_full)}_eps{eps_tag}_seed{seed}"
    cmd = [
        PY, "-u", TRAIN,
        "--algo", algo, "--env", "robust_attack", "--exp_name", expname,
        "--scenario", env_full,
        "--epsilon_observation", str(eps_val), "--epsilon_action", str(eps_val),
        "--model_path", vpath,
        "--num_env_steps", str(STEPS), "--seed", str(seed),
        "--n_rollout_threads", str(THREADS),
        "--share_param", "False",
        "--use_wandb", USE_WANDB,
        "--wandb_project", f"robust_attack_{env_full}",
        "--wandb_group", f"{short(env_full)}_eps{eps_tag}{suffix}",
        "--wandb_name", f"{algo}{suffix}_eps{eps_tag}_seed{seed}",
    ]
    if algo == "stage_mappo":
        # stage_lambda MUST stay high (0.95): it leaks into the V^o critic target
        # and (via the shared critic backbone) into V^a/adv_a, so the yaml default
        # 0.7 noticeably weakens the attacker. 0.95 reproduces the good nodelta run.
        cmd += ["--state_type", "FP", "--causal_critic_state", "True",
                "--stage_lambda", "0.95"]
        if nodelta:
            cmd += ["--stage_drop_delta_o", "True"]
    return expname, cmd


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


def victim_jobs(env_full, seeds):
    jobs = []
    for seed in seeds:
        if victim_ckpt(env_full, seed) is not None:
            log_status(f"SKIP victim {short(env_full)} seed{seed} (ckpt exists)")
            continue
        jobs.append(victim_cmd(env_full, seed))
    return jobs


def attack_jobs(env_full, seeds, algos, nodelta):
    """algos: list of algos to launch per (seed, eps). For Ant -> [ippo, stage_mappo]."""
    jobs = []
    for seed in seeds:
        vpath = victim_ckpt(env_full, seed)
        if vpath is None:
            log_status(f"ERROR no victim ckpt for {short(env_full)} seed{seed}; skipping its attacks")
            continue
        for eps_tag, eps_val in EPSES:
            for algo in algos:
                expname, cmd = attack_cmd(env_full, eps_tag, eps_val, seed, algo, vpath, nodelta)
                if is_done(expname) or attack_ckpt(env_full, algo, expname, seed):
                    log_status(f"SKIP {expname} (already done)")
                    continue
                jobs.append((expname, cmd))
    return jobs


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    log_status("==== NODELTA CAMPAIGN START ====")
    log_status(f"REPO={REPO}")
    log_status(f"PY={PY}")
    log_status(f"conc={CONC} threads={THREADS} steps={STEPS} wandb={USE_WANDB}")

    # ---- Phase HC-V: HalfCheetah victims (train only the missing ones) ----
    jobs = victim_jobs(HC_ENV, HC_SEEDS)
    log_status(f"Phase HC-V: {len(jobs)} victim run(s) to train")
    run_pool(jobs, CONC)
    log_status("==== Phase HC-V (HalfCheetah victims) COMPLETE ====")

    # ---- Phase HC-A: HalfCheetah nodelta stage_mappo attacks ----
    jobs = attack_jobs(HC_ENV, HC_SEEDS, ["stage_mappo"], nodelta=True)
    log_status(f"Phase HC-A: {len(jobs)} attack run(s) to train")
    run_pool(jobs, CONC)
    log_status("==== Phase HC-A (HalfCheetah nodelta attacks) COMPLETE ====")

    # ---- Phase ANT-V: Ant victims (train only the missing ones) ----
    jobs = victim_jobs(ANT_ENV, ANT_SEEDS)
    log_status(f"Phase ANT-V: {len(jobs)} victim run(s) to train")
    run_pool(jobs, CONC)
    log_status("==== Phase ANT-V (Ant victims) COMPLETE ====")

    # ---- Phase ANT-A: Ant ippo + nodelta stage_mappo attacks ----
    jobs = attack_jobs(ANT_ENV, ANT_SEEDS, ["ippo", "stage_mappo"], nodelta=True)
    log_status(f"Phase ANT-A: {len(jobs)} attack run(s) to train")
    run_pool(jobs, CONC)
    log_status("==== Phase ANT-A (Ant attacks) COMPLETE ====")

    log_status("==== NODELTA CAMPAIGN COMPLETE ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
