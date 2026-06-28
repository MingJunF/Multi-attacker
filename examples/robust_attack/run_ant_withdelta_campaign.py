#!/usr/bin/env python3
"""WITH-delta^o attack campaign for Ant-v4 (SERVER).

Keeping delta^o was found to help (the obs-attacker advantage A^o = delta^o +
stage_lambda * A^a). This scheduler trains the Ant victims (missing ones only)
and then, per (seed, eps), launches BOTH baselines for a clean comparison:

    - ippo                       (independent PPO attacker)
    - stage_mappo  WITH delta^o  (--state_type FP --causal_critic_state True
                                  --stage_lambda 0.95, NO --stage_drop_delta_o)

This is the delta^o counterpart of run_nodelta_campaign.py. wandb run names use
NO "nodelta" suffix (stage_mappo_eps*_seed*), so they land in the "stage" column
of the comparison plots and never collide with the nodelta runs.

Phases (strict order; concurrency = 3, --n_rollout_threads 8 each):
  Phase ANT-V : train any MISSING Ant-v4 victims for seeds {1, 10, 100, 1000, 10000}.
  Phase ANT-A : per (seed, eps) launch {ippo, stage_mappo(delta^o)},
                seeds {1, 10, 100, 1000, 10000} x eps {0.05, 0.10, 0.15, 0.20}.

RESUMABLE / IDEMPOTENT: a victim is skipped if any actor_agent0.pt for (env,seed)
exists; an attack is skipped if its .done marker OR final actor_agent0.pt exists.

How to run on the server (use the env that actually has torch+mujoco+harl!):
  cd <repo-root>
  # CORRECT interpreter on the rented box: /venv/rl310/bin/python
  nohup /venv/rl310/bin/python -u \
        examples/robust_attack/run_ant_withdelta_campaign.py \
        > logs/campaign/ant_withdelta_campaign.log 2>&1 &
  tail -f logs/campaign/STATUS_ant_withdelta.txt
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
STATUS = os.path.join(LOGDIR, "STATUS_ant_withdelta.txt")


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


CONC = _int("CONC", 3)
THREADS = _int("THREADS", 8)
STEPS = _int("STEPS", 6_000_000)
EVAL_INTERVAL = _int("EVAL_INTERVAL", 25)
USE_WANDB = os.environ.get("USE_WANDB", "True")
STAGE_LAMBDA = os.environ.get("STAGE_LAMBDA", "0.95")

EPSES = [("005", 0.05), ("010", 0.1), ("015", 0.15), ("020", 0.2)]

ANT_ENV = "Ant-v4"
ANT_SEEDS = [1, 10, 100, 1000, 10000]
VICTIM_EXP = "victim6m"


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
    name = f"{VICTIM_EXP}_{short(env_full)}_seed{seed}"
    return name, cmd


def attack_cmd(env_full, eps_tag, eps_val, seed, algo, vpath):
    """algo in {ippo, stage_mappo}. stage_mappo is WITH delta^o here."""
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
        "--use_wandb", USE_WANDB,
        "--wandb_project", f"robust_attack_{env_full}",
        "--wandb_group", f"{short(env_full)}_eps{eps_tag}",
        "--wandb_name", f"{algo}_eps{eps_tag}_seed{seed}",
    ]
    if algo == "stage_mappo":
        # WITH delta^o (NO --stage_drop_delta_o). Keep stage_lambda high (0.95):
        # it leaks into the V^o critic target / shared backbone, so the yaml
        # default 0.7 would weaken the attacker.
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


def victim_jobs(env_full, seeds):
    jobs = []
    for seed in seeds:
        if victim_ckpt(env_full, seed) is not None:
            log_status(f"SKIP victim {short(env_full)} seed{seed} (ckpt exists)")
            continue
        jobs.append(victim_cmd(env_full, seed))
    return jobs


def attack_jobs(env_full, seeds, algos):
    jobs = []
    for seed in seeds:
        vpath = victim_ckpt(env_full, seed)
        if vpath is None:
            log_status(f"ERROR no victim ckpt for {short(env_full)} seed{seed}; skipping its attacks")
            continue
        for eps_tag, eps_val in EPSES:
            for algo in algos:
                expname, cmd = attack_cmd(env_full, eps_tag, eps_val, seed, algo, vpath)
                if is_done(expname) or attack_ckpt(env_full, algo, expname, seed):
                    log_status(f"SKIP {expname} (already done)")
                    continue
                jobs.append((expname, cmd))
    return jobs


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    log_status("==== ANT WITH-delta CAMPAIGN START ====")
    log_status(f"REPO={REPO}")
    log_status(f"PY={PY}")
    log_status(f"conc={CONC} threads={THREADS} steps={STEPS} stage_lambda={STAGE_LAMBDA} wandb={USE_WANDB}")

    # ---- Phase ANT-V: Ant victims (train only the missing ones) ----
    jobs = victim_jobs(ANT_ENV, ANT_SEEDS)
    log_status(f"Phase ANT-V: {len(jobs)} victim run(s) to train")
    run_pool(jobs, CONC)
    log_status("==== Phase ANT-V (Ant victims) COMPLETE ====")

    # ---- Phase ANT-A: Ant ippo + stage_mappo(delta^o) attacks ----
    jobs = attack_jobs(ANT_ENV, ANT_SEEDS, ["ippo", "stage_mappo"])
    log_status(f"Phase ANT-A: {len(jobs)} attack run(s) to train")
    run_pool(jobs, CONC)
    log_status("==== Phase ANT-A (Ant with-delta attacks) COMPLETE ====")

    log_status("==== ANT WITH-delta CAMPAIGN COMPLETE ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
