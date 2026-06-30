"""Train a cooperative multi-attacker against a fixed victim in robust_gymnasium.

The attacker is a two-agent team:
    agent 0 -- observation attacker (perturbs the victim's observation)
    agent 1 -- action attacker      (tampers the victim's action)

Both agents share a cooperative reward equal to the *negative* victim reward,
so maximising the attacker return is equivalent to minimising the victim
return. The team can be optimised with any HARL on-policy algorithm:

    # Two fully independent PPO attackers (each agent has its own actor AND its
    # own critic trained only on its local observation):
    python train_robust_attacker.py --algo ippo --env robust_attack \
        --exp_name ippo

    # Shared-parameter MAPPO attackers (single centralized critic):
    python train_robust_attacker.py --algo mappo --env robust_attack \
        --exp_name mappo --share_param True

    # Heterogeneous-agent PPO (HAPPO) attackers:
    python train_robust_attacker.py --algo happo --env robust_attack \
        --exp_name happo

mappo, happo and ippo can all be launched in parallel as separate runs.

Override env settings on the command line, e.g.:
    --scenario HalfCheetah-v4 --epsilon_observation 0.2 --epsilon_action 0.05

Run from the repository root so that the ``harl`` and ``robust_gymnasium``
packages are importable.
"""

import argparse
import json
import os
import sys

# Make ``harl`` / ``robust_gymnasium`` importable even when this script is run
# directly (e.g. ``python examples/robust_attack/train_robust_attacker.py``)
# without installing the packages. The repository root is two levels up.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Train a multi-attacker (obs + action) on robust_gymnasium."
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="mappo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "mappo",
            "ma_mappo",
            "ar_mappo",
            "stage_mappo",
            "stage_maddpg",
            "ippo",
            "r_mappo",
            "mappo_potential",
            "happo_potential",
            "hatrpo_potential",
        ],
        help="MARL algorithm driving the attacker team.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="robust_attack",
        help="Environment name (keep as robust_attack).",
    )
    parser.add_argument("--exp_name", type=str, default="installtest")
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load the full config from this json file.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except Exception:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # strip the leading "--"
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)

    # Several robust_gymnasium env modules call ``get_config().parse_args()`` at
    # import time, which would otherwise try to parse this script's CLI flags
    # (e.g. --env) and crash. We have already captured everything we need above,
    # so reset argv to just the program name before importing robust_gymnasium.
    sys.argv = sys.argv[:1]

    from harl.utils.configs_tools import get_defaults_yaml_args, update_args

    if args["load_config"] != "":
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
        update_args(unparsed_dict, algo_args, env_args)

    from harl.runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
