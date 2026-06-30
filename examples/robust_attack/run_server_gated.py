#!/usr/bin/env python3
"""Gated server campaign: only attack victims whose clean return clears a bar.

Goal
----
We already have 3 GOOD HalfCheetah victims (seeds 1, 10, 1000). Seeds 100 and
10000 trained poorly. This script keeps training victims on FRESH candidate
seeds (2, 20, 3, 30, ...) and, right after each victim finishes, evaluates its
clean (un-attacked) episode return. A seed is ACCEPTED only if that return is
>= VICTIM_REWARD_THRESHOLD (default 3000); otherwise it is skipped and we move
on to the next candidate -- no attacker is trained on a weak victim.

We stop once TARGET_TOTAL (default 5) good seeds exist in total, i.e. once
NEED = TARGET_TOTAL - len(PRE_QUALIFIED) more candidates pass the bar. Then the
attacker sweep (eps x algo) runs for the newly accepted seeds.

Flow
----
  for seed in CANDIDATE_SEEDS:
      train victim(seed)            # skipped if a ckpt already exists
      r = eval_clean_return(seed)   # deterministic rollout, no attack
      if r >= THRESHOLD:  accept    # else: skip, try next candidate
      stop when NEED seeds accepted
  for seed in accepted:             # attacker sweep (idempotent via .done)
      for eps in EPSES: for algo in ALGOS: train attacker(seed, eps, algo)

Tunables (env vars)
-------------------
  VICTIM_REWARD_THRESHOLD (3000)   TARGET_TOTAL (5)
  PRE_QUALIFIED  ("1,10,1000")     CANDIDATE_SEEDS ("2,20,3,30,4,40,5,50")
  EVAL_EPISODES  (10)              ATTACK_ALL (False -> only new seeds)
  VICTIM_THREADS (16)  ATTACK_THREADS (8)  ATTACK_CONC (2)
  VICTIM_STEPS (6000000)  ATTACK_STEPS (6000000)  USE_WANDB (True)

Run on the server
-----------------
  cd <repo-root> && conda activate <env>
  wandb login            # or export WANDB_API_KEY=...  |  USE_WANDB=False
  mkdir -p logs/campaign
  nohup python -u examples/robust_attack/run_server_gated.py \
        > logs/campaign/server_gated.log 2>&1 &
  tail -f logs/campaign/STATUS_gated.txt
"""
import glob
import os
import subprocess
import sys
import time

# --- portable paths ---------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable
TRAIN = os.path.join("examples", "robust_attack", "train_robust_attacker.py")
LOGDIR = os.path.join(REPO, "logs", "campaign")
STATUS = os.path.join(LOGDIR, "STATUS_gated.txt")


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _seed_list(name, default):
    raw = os.environ.get(name, default)
    return [int(x) for x in raw.replace(",", " ").split()]


# --- config -----------------------------------------------------------------
ENV = "HalfCheetah-v4"
EPSES = [("005", 0.05), ("010", 0.1), ("015", 0.15), ("020", 0.2)]
ALGOS = ["ippo", "stage_mappo"]

VICTIM_REWARD_THRESHOLD = _float("VICTIM_REWARD_THRESHOLD", 3000.0)
TARGET_TOTAL = _int("TARGET_TOTAL", 5)
PRE_QUALIFIED = _seed_list("PRE_QUALIFIED", "1,10,1000")
CANDIDATE_SEEDS = _seed_list("CANDIDATE_SEEDS", "2,20,3,30,4,40,5,50")
EVAL_EPISODES = _int("EVAL_EPISODES", 10)
ATTACK_ALL = os.environ.get("ATTACK_ALL", "False") == "True"

VICTIM_THREADS = _int("VICTIM_THREADS", 16)
ATTACK_THREADS = _int("ATTACK_THREADS", 8)
ATTACK_CONC = _int("ATTACK_CONC", 2)
VICTIM_STEPS = _int("VICTIM_STEPS", 6_000_000)
ATTACK_STEPS = _int("ATTACK_STEPS", 6_000_000)
EVAL_INTERVAL = _int("EVAL_INTERVAL", 25)
USE_WANDB = os.environ.get("USE_WANDB", "True")

VICTIM_EXP = "victim_hc16"   # keep same exp so attacker ckpt globs still match


# --- helpers ----------------------------------------------------------------
def log_status(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def marker(name):
    return os.path.join(LOGDIR, name + ".done")


def is_done(name):
    return os.path.exists(marker(name))


def victim_ckpt(seed):
    pats = glob.glob(
        f"{REPO}/results/robust_victim/{ENV}/mappo/{VICTIM_EXP}/"
        f"seed-{seed:05d}-*/models/actor_agent0.pt"
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


def run_one(name, cmd):
    """Run a single (name, cmd) to completion, streaming to its own log."""
    logpath = os.path.join(LOGDIR, name + ".log")
    with open(logpath, "w") as logf:
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        log_status(f"START {name}")
        rc = subprocess.call(
            cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT, env=env
        )
    if rc == 0:
        open(marker(name), "w").close()
        log_status(f"DONE  {name} (rc=0)")
    else:
        log_status(f"FAIL  {name} (rc={rc}) -- see {name}.log")
    return rc


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
            p = subprocess.Popen(
                cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT, env=env
            )
            running.append((name, p, logf))
            log_status(
                f"START {name} (pid {p.pid}) "
                f"[{len(queue)} queued, {len(running)} running]"
            )
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
                    log_status(f"DONE  {name} (rc=0)")
                else:
                    log_status(f"FAIL  {name} (rc={rc}) -- see {name}.log")
        running = still


def eval_clean_return(ckpt, episodes, seed=12345):
    """Mean +/- std clean (un-attacked) episode return of a trained victim ckpt.

    Loads the checkpoint as a fixed HARL victim and rolls it deterministically
    in the same robust_gymnasium task used for training (no perturbation).
    """
    import numpy as np
    from harl.envs.robust_attack.victim_env import RobustVictimEnv
    from harl.envs.robust_attack.victim_policy import make_victim_policy

    env = RobustVictimEnv({"scenario": ENV, "seed": seed})
    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    victim = make_victim_policy(
        {"type": "harl", "model_path": ckpt, "deterministic": True},
        obs_space,
        act_space,
    )
    returns = []
    for _ in range(episodes):
        victim.reset()
        obs_list, _, _ = env.reset()
        obs = obs_list[0]
        done = False
        total = 0.0
        steps = 0
        while not done and steps < 2000:
            action = victim.act(obs)
            obs_l, _, rew, dones, _, _ = env.step(np.asarray([action]))
            obs = obs_l[0]
            total += float(rew[0][0])
            done = bool(dones[0])
            steps += 1
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


# --- main -------------------------------------------------------------------
def main():
    os.makedirs(LOGDIR, exist_ok=True)
    log_status("=" * 70)
    log_status(
        f"GATED campaign | threshold={VICTIM_REWARD_THRESHOLD} "
        f"target_total={TARGET_TOTAL} pre_qualified={PRE_QUALIFIED}"
    )
    need = TARGET_TOTAL - len(PRE_QUALIFIED)
    if need <= 0:
        log_status("Already have enough qualified seeds; nothing to gate.")
        accepted = []
    else:
        log_status(f"Need {need} more good seed(s) from candidates {CANDIDATE_SEEDS}")
        accepted = []
        for seed in CANDIDATE_SEEDS:
            if len(accepted) >= need:
                break
            # 1) train victim (skip if a ckpt already exists)
            ckpt = victim_ckpt(seed)
            if ckpt is None:
                name, cmd = victim_cmd(seed)
                if not is_done(name):
                    run_one(name, cmd)
                ckpt = victim_ckpt(seed)
            else:
                log_status(f"victim seed {seed}: ckpt already exists, skip training")
            if ckpt is None:
                log_status(f"victim seed {seed}: NO ckpt produced, skipping")
                continue
            # 2) evaluate clean return and gate
            try:
                mean_r, std_r = eval_clean_return(ckpt, EVAL_EPISODES)
            except Exception as exc:  # noqa: BLE001
                log_status(f"victim seed {seed}: EVAL ERROR {exc!r}; skipping")
                continue
            verdict = "ACCEPT" if mean_r >= VICTIM_REWARD_THRESHOLD else "REJECT"
            log_status(
                f"victim seed {seed}: clean return {mean_r:.1f} +/- {std_r:.1f} "
                f"(>= {VICTIM_REWARD_THRESHOLD}? -> {verdict})"
            )
            if mean_r >= VICTIM_REWARD_THRESHOLD:
                accepted.append(seed)
                log_status(
                    f"QUALIFIED seed {seed} "
                    f"({len(accepted)}/{need} new, "
                    f"{len(PRE_QUALIFIED)+len(accepted)}/{TARGET_TOTAL} total)"
                )

    if len(accepted) < need:
        log_status(
            f"WARNING: only accepted {len(accepted)}/{need} new seeds from the "
            f"candidate pool. Add more to CANDIDATE_SEEDS and re-run."
        )

    # 3) attacker sweep
    attack_seeds = (PRE_QUALIFIED + accepted) if ATTACK_ALL else accepted
    if not attack_seeds:
        log_status("No seeds to attack. Exiting.")
        return
    log_status(f"Attacker sweep on seeds {attack_seeds} (ATTACK_ALL={ATTACK_ALL})")
    jobs = []
    for seed in attack_seeds:
        vpath = victim_ckpt(seed)
        if vpath is None:
            log_status(f"seed {seed}: no victim ckpt for attacker, skipping")
            continue
        for eps_tag, eps_val in EPSES:
            for algo in ALGOS:
                name, cmd = attack_cmd(seed, eps_tag, eps_val, algo, vpath)
                if is_done(name):
                    log_status(f"skip (done) {name}")
                    continue
                jobs.append((name, cmd))
    if jobs:
        run_pool(jobs, ATTACK_CONC)
    log_status("Gated campaign complete.")


if __name__ == "__main__":
    main()
