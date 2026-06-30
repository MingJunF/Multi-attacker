#!/usr/bin/env python3
"""LOCAL (this machine) runner: HalfCheetah-v4 seed10 VANILLA MAPPO attacks.

Runs the *standard* HARL MAPPO attacker (NOT stage-aware): a single shared
centralized critic, vanilla simultaneous CTDE. The critic is fed the env's
share_obs = [victim obs s, victim.act(s + delta_o), stage one-hot], i.e. victim
obs + action, exactly like the stage runs; the actors are unchanged. This is the
"standard CTDE / single shared baseline" baseline the paper contrasts against
stage_mappo.

    --algo mappo --state_type EP --causal_critic_state False  (HARL defaults)

Runs HalfCheetah-v4 seed 10 across eps {0.05, 0.10, 0.15, 0.20}, concurrency 2.

wandb:
    name  = mappo_eps{tag}_seed10
    group = hc_eps{tag}
    proj  = robust_attack_HalfCheetah-v4

Run from the repo root with an env that has torch+mujoco (e.g. the `mujoko` env):
    PYTHONPATH=$PWD nohup /home/mingjun/miniconda3/envs/mujoko/bin/python -u \
        examples/robust_attack/run_hc_seed10_mappo_local.py \
        > logs/campaign/hc_seed10_mappo_local.log 2>&1 &
    tail -f logs/campaign/STATUS_hc_seed10_mappo.txt
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
STATUS = os.path.join(LOGDIR, "STATUS_hc_seed10_mappo.txt")

ENV = "HalfCheetah-v4"
SEED = 10
EPSES = [("005", 0.05), ("010", 0.1), ("015", 0.15), ("020", 0.2)]

CONC = int(os.environ.get("CONC", 2))           # 2 parallel runs on this machine
THREADS = int(os.environ.get("THREADS", 8))
STEPS = int(os.environ.get("STEPS", 6_000_000))
USE_WANDB = os.environ.get("USE_WANDB", "True")


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
    expname = f"attack_mappo_hc_eps{eps_tag}_seed{SEED}"
    cmd = [
        PY, "-u", TRAIN,
        "--algo", "mappo", "--env", "robust_attack", "--exp_name", expname,
        "--scenario", ENV,
        "--epsilon_observation", str(eps_val), "--epsilon_action", str(eps_val),
        "--model_path", vpath,
        "--num_env_steps", str(STEPS), "--seed", str(SEED),
        "--n_rollout_threads", str(THREADS),
        "--share_param", "False",
        # Vanilla HARL MAPPO: single shared centralized critic over the env's
        # share_obs = [s, victim.act(s + delta_o), stage one-hot] (victim obs +
        # action). EP = one global state identical across agents; no stage-aware
        # causal masking. Actors unchanged.
        "--state_type", "EP", "--causal_critic_state", "False",
        "--use_wandb", USE_WANDB,
        "--wandb_project", f"robust_attack_{ENV}",
        "--wandb_group", f"hc_eps{eps_tag}",
        "--wandb_name", f"mappo_eps{eps_tag}_seed{SEED}",
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
    log_status("==== HC seed10 VANILLA-MAPPO LOCAL CAMPAIGN START ====")
    log_status(f"REPO={REPO}")
    log_status(f"PY={PY}")
    log_status(f"conc={CONC} threads={THREADS} steps={STEPS} wandb={USE_WANDB}")

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
    log_status("==== HC seed10 VANILLA-MAPPO LOCAL CAMPAIGN COMPLETE ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
