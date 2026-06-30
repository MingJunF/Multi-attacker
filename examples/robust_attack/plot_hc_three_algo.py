import wandb, numpy as np, re, os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

api = wandb.Api(timeout=60)
ENT = "mingjun-fan-university-of-south-australia"


def _retry(fn, tries=6, delay=4):
    """Call fn() with retries on transient wandb server errors (e.g. HTTP 500)."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # wandb HTTPError / CommError on flaky server
            last = e
            if i < tries - 1:
                time.sleep(delay * (i + 1))
    raise last


PROJ = os.environ.get("PROJ", "robust_attack_HalfCheetah-v4")
ENVNAME = os.environ.get("ENVNAME", PROJ.replace("robust_attack_", ""))
BASE = float(os.environ.get("BASE", "5000"))
runs = _retry(lambda: list(api.runs(f"{ENT}/{PROJ}")))

# strict patterns (order: nodelta before stage so it is matched first)
PATS = [
    ("stage_nodelta", re.compile(r"^stage_mappo_nodelta_eps(\d{3})_seed(\d+)(_rerun)?$")),
    ("stage",         re.compile(r"^stage_mappo_eps(\d{3})_seed(\d+)(_rerun)?$")),
    ("ippo",          re.compile(r"^ippo_eps(\d{3})_seed(\d+)(_rerun)?$")),
]
EPS = os.environ.get("EPS", "005,010,015,020").split(",")
EPSV = {"005": 0.05, "010": 0.10, "015": 0.15, "020": 0.20}
SEEDS = os.environ.get("SEEDS", "1,10,500,1000").split(",")
ERR = os.environ.get("ERR", "sem").lower()  # 'std' or 'sem'


def last20(r):
    h = _retry(lambda: r.history(keys=["attack/victim_episode_rewards"], samples=3000))
    v = h["attack/victim_episode_rewards"].dropna().values
    return (float(np.mean(v[-20:])), len(v)) if len(v) else (None, 0)


def stage_lambda_ok(r):
    """True if this stage run used stage_lambda 0.95 (or absent -> falls back
    to gae_lambda 0.95). Drops contaminated 0.7 runs."""
    try:
        cfg = _retry(lambda: r.config)
        lam = cfg["algo_args"]["algo"].get("stage_lambda", None)
    except (KeyError, TypeError):
        lam = None
    if lam is None:
        return True  # absent -> fell back to gae_lambda (0.95)
    return abs(float(lam) - 0.95) < 1e-6


best = {}  # (algo,eps,seed) -> (val, steps)
for r in runs:
    name = r.name or ""
    for algo, rx in PATS:
        m = rx.match(name)
        if not m:
            continue
        eps, seed = m.group(1), m.group(2)
        if seed not in SEEDS:
            break
        if algo in ("stage", "stage_nodelta") and not stage_lambda_ok(r):
            break  # drop stage_lambda!=0.95 (contaminated) runs
        val, n = last20(r)
        if val is None:
            break
        key = (algo, eps, seed)
        if key not in best or n > best[key][1]:
            best[key] = (val, n)
        break

# print table
for algo in ["ippo", "stage", "stage_nodelta"]:
    print(f"\n--- {algo} (victimR last20; lower=stronger) ---")
    print(f"{'eps':6}" + "".join(f"seed{s:<7}" for s in SEEDS) + "  mean(n)")
    for eps in EPS:
        vals = [best.get((algo, eps, s)) for s in SEEDS]
        cells = [f"{v[0]:<11.0f}" if v else f"{'-':<11}" for v in vals]
        common = [v[0] for v in vals if v]
        mc = f"{np.mean(common):.0f}({len(common)})" if common else "-"
        print(f"{eps:6}" + "".join(cells) + f"  {mc}")

# plot
COL = {"ippo": "#1f77b4", "stage": "#d62728", "stage_nodelta": "#2ca02c"}
LBL = {"ippo": "IPPO", "stage": "stage_mappo (with $\\delta^o$)",
       "stage_nodelta": "stage_mappo nodelta (drop $\\delta^o$)"}
plt.figure(figsize=(8, 5.5))
x = [EPSV[e] for e in EPS]
for algo in ["ippo", "stage", "stage_nodelta"]:
    means, sems, xs = [], [], []
    for e in EPS:
        vals = [best[(algo, e, s)][0] for s in SEEDS if (algo, e, s) in best]
        if not vals:
            continue
        xs.append(EPSV[e])
        means.append(np.mean(vals))
        if len(vals) > 1:
            sd = np.std(vals, ddof=1)
            sems.append(sd if ERR == "std" else sd / np.sqrt(len(vals)))
        else:
            sems.append(0.0)
        # per-seed scatter
        plt.scatter([EPSV[e]] * len(vals), vals, color=COL[algo], s=18, alpha=0.35, zorder=2)
    plt.errorbar(xs, means, yerr=sems, color=COL[algo], marker="o", lw=2,
                 capsize=4, label=LBL[algo], zorder=3)
plt.axhline(BASE, ls="--", color="gray", lw=1, alpha=0.7)
plt.text(0.052, BASE, f"no-attack baseline ~{BASE:.0f}", color="gray", va="bottom", fontsize=9)
plt.gca().invert_yaxis()  # lower (stronger attack) on top
plt.xlabel("attack budget $\\epsilon$ (obs = act)")
plt.ylabel("victim episode reward (last-20 mean)\nlower = stronger attack")
plt.title(f"{ENVNAME}: attacker comparison across $\\epsilon$\n(mean $\\pm$ " + ERR.upper() + " over seeds {" + ",".join(SEEDS) + "}; dots = per-seed)")
plt.xticks(x, [f"{v:.2f}" for v in x])
plt.legend(loc="upper right")
plt.grid(alpha=0.25)
plt.tight_layout()
OUT = os.environ.get("OUT", "examples/robust_attack/hc_three_algo_compare.png")
plt.savefig(OUT, dpi=150)
print("\nsaved:", OUT)
