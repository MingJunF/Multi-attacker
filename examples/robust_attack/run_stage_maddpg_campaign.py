#!/usr/bin/env python3
"""STAGE-MADDPG attack campaign across the three MuJoCo maps (one script).

Off-policy Stage-Aware MADDPG attacker (the off-policy counterpart of
stage_mappo): a SINGLE shared centralized continuous Q critic with a coupled
two-stage Bellman target, sequential obs->act rollout. Launched with the same
causal FP centralized state as the stage_mappo runs:

    --algo stage_maddpg --state_type FP --causal_critic_state True --stage_lambda 0.95

Matrix (eps {0.05, 0.10, 0.15, 0.20} for every (env, seed)):
    HalfCheetah-v4 : seeds {1, 500, 1000}
    Ant-v4         : seeds {1, 10, 100, 1000, 10000}
    Hopper-v4      : seeds {1, 10, 100, 1000, 10000}

wandb (matches the stage_maddpg local run so plots can pick them up):
    name  = stage_maddpg_eps{tag}_seed{seed}
    group = {hc|ant|hopper}_eps{tag}
    proj  = robust_attack_{env}

RESUMABLE / IDEMPOTENT: an attack is skipped if its .done marker OR a final
actor_agent0.pt for (env, seed) already exists. Missing victim ckpts are skipped
with an error line (victims must already be trained).

Deployment (one script, two machines) via the TARGETS env var:

  # On the rented SERVER (Ant + Hopper), use the env that has torch+mujoco+harl:
  cd <repo-root>
  TARGETS=Ant-v4,Hopper-v4 CONC=3 nohup /venv/rl310/bin/python -u \
        examples/robust_attack/run_stage_maddpg_campaign.py \
        > logs/campaign/stage_maddpg_server.log 2>&1 &
  tail -f logs/campaign/STATUS_stage_maddpg.txt

  # On the LOCAL machine (HalfCheetah only, remaining seeds):
  TARGETS=HalfCheetah-v4 CONC=2 PYTHONPATH=$PWD nohup \
        /home/mingjun/miniconda3/envs/mujoko/bin/python -u \
        examples/robust_attack/run_stage_maddpg_campaign.py \
        > logs/campaign/stage_maddpg_hc_local.log 2>&1 &

  # No TARGETS -> runs all three envs on this machine.
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
STATUS = os.path.join(LOGDIR, "STATUS_stage_maddpg.txt")


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


CONC = _int("CONC", 3)
THREADS = _int("THREADS", 8)
STEPS = _int("STEPS", 6_000_000)
USE_WANDB = os.environ.get("USE_WANDB", "True")
STAGE_LAMBDA = os.environ.get("STAGE_LAMBDA", "0.95")

EPSES = [("005", 0.05), ("010", 0.1), ("015", 0.15), ("020", 0.2)]

# per-env seed sets: HalfCheetah has its own seed set {1, 500, 1000}
ENV_SEEDS = {
    "HalfCheetah-v4": [1, 500, 1000],
    "Ant-v4": [1, 10, 100, 1000, 10000],
    "Hopper-v4": [1, 10, 100, 1000, 10000],
}

# TARGETS env var selects which environments to run on this machine.
_targets = os.environ.get("TARGETS", "").strip()
if _targets:
    TARGETS = [e.strip() for e in _targets.split(",") if e.strip()]
else:
    TARGETS = list(ENV_SEEDS.keys())

_SHORT = {"HalfCheetah-v4": "hc", "Ant-v4": "ant", "Hopper-v4": "hopper"}


def short(env_full):
    return _SHORT.get(env_full, env_full.split("-")[0].lower())


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
    pats = glob.glob(
        f"{REPO}/results/robust_victim/{env_full}/mappo/*/seed-{seed:05d}-*/models/actor_agent0.pt"
    )
    pats = sorted(pats, key=os.path.getmtime)
    return pats[-1] if pats else None


def attack_ckpt(env_full, expname, seed):
    pats = glob.glob(
        f"{REPO}/results/robust_attack/{env_full}/stage_maddpg/{expname}/seed-{seed:05d}-*/models/actor_agent0.pt"
    )
    return bool(pats)


def attack_cmd(env_full, eps_tag, eps_val, seed, vpath):
    expname = f"attack_stage_maddpg_{short(env_full)}_eps{eps_tag}_seed{seed}"
    cmd = [
        PY, "-u", TRAIN,
        "--algo", "stage_maddpg", "--env", "robust_attack", "--exp_name", expname,
        "--scenario", env_full,
        "--epsilon_observation", str(eps_val), "--epsilon_action", str(eps_val),
        "--model_path", vpath,
        "--num_env_steps", str(STEPS), "--seed", str(seed),
        "--n_rollout_threads", str(THREADS),
        "--share_param", "False",
        # Stage-Aware MADDPG: single shared continuous Q critic with a coupled
        # two-stage target over the env's causal FP share_obs
        # (x^o = [s, 0]; x^a = [s, victim.act(s + delta_o)]). Sequential
        # obs->act rollout, identical scope to stage_mappo.
        "--state_type", "FP", "--causal_critic_state", "True",
        "--stage_lambda", STAGE_LAMBDA,
        "--use_wandb", USE_WANDB,
        "--wandb_project", f"robust_attack_{env_full}",
        "--wandb_group", f"{short(env_full)}_eps{eps_tag}",
        "--wandb_name", f"stage_maddpg_eps{eps_tag}_seed{seed}",
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


def attack_jobs(env_full, seeds):
    jobs = []
    for seed in seeds:
        vpath = victim_ckpt(env_full, seed)
        if vpath is None:
            log_status(f"ERROR no victim ckpt for {short(env_full)} seed{seed}; skipping its attacks")
            continue
        for eps_tag, eps_val in EPSES:
            expname, cmd = attack_cmd(env_full, eps_tag, eps_val, seed, vpath)
            if is_done(expname) or attack_ckpt(env_full, expname, seed):
                log_status(f"SKIP {expname} (already done)")
                continue
            jobs.append((expname, cmd))
    return jobs


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    log_status("==== STAGE-MADDPG CAMPAIGN START ====")
    log_status(f"REPO={REPO}")
    log_status(f"PY={PY}")
    log_status(f"targets={TARGETS}")
    log_status(
        f"conc={CONC} threads={THREADS} steps={STEPS} "
        f"stage_lambda={STAGE_LAMBDA} wandb={USE_WANDB}"
    )

    jobs = []
    for env_full in TARGETS:
        if env_full not in ENV_SEEDS:
            log_status(f"WARN unknown target env {env_full}; skipping")
            continue
        ejobs = attack_jobs(env_full, ENV_SEEDS[env_full])
        log_status(f"{env_full}: {len(ejobs)} attack run(s) to train")
        jobs.extend(ejobs)

    log_status(f"TOTAL {len(jobs)} attack run(s) to train")
    run_pool(jobs, CONC)
    log_status("==== STAGE-MADDPG CAMPAIGN COMPLETE ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
