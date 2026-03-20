from __future__ import annotations

import argparse
from pathlib import Path

import torch


def get_args():
    default_results_dir = Path(__file__).resolve().parents[1] / "results" / "critic_compare"
    parser = argparse.ArgumentParser(description="A2C with V/Q critic on RobustLunarLander-v3")
    parser.add_argument("--lr", type=float, default=7e-4, help="learning rate (default: 7e-4)")
    parser.add_argument("--eps", type=float, default=1e-5, help="RMSprop epsilon (default: 1e-5)")
    parser.add_argument("--alpha", type=float, default=0.99, help="RMSprop alpha (default: 0.99)")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor (default: 0.99)")
    parser.add_argument("--use-gae", action="store_true", default=False, help="use generalized advantage estimation")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda (default: 0.95)")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="value loss coefficient (default: 0.5)")
    parser.add_argument(
        "--critic-type",
        type=str,
        choices=("v", "q"),
        default="v",
        help="critic parameterization: state-value V(s) or action-value Q(s,a) (default: v)",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="max gradient norm (default: 0.5)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--cuda-deterministic",
        action="store_true",
        default=False,
        help="set deterministic CUDA flags when using GPU",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="number of parallel environment processes (default: 16)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="number of forward steps in A2C (default: 5)",
    )
    parser.add_argument("--log-interval", type=int, default=10, help="log interval in updates")
    parser.add_argument("--save-interval", type=int, default=100, help="save interval in updates")
    parser.add_argument("--eval-interval", type=int, default=None, help="evaluation interval in updates")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="number of episodes per checkpoint evaluation (default: 10)",
    )
    parser.add_argument(
        "--final-eval-episodes",
        type=int,
        default=100,
        help="number of episodes for the final evaluation sweep (default: 100)",
    )
    parser.add_argument(
        "--num-env-steps",
        type=int,
        default=int(10e6),
        help="number of environment steps to train (default: 1e7)",
    )
    parser.add_argument(
        "--env-name",
        default="RobustLunarLander-v3",
        help="environment to train on (default: RobustLunarLander-v3)",
    )
    parser.add_argument("--log-dir", default="/tmp/gym/", help="directory to save logs")
    parser.add_argument("--save-dir", default="./trained_models/", help="directory to save models")
    parser.add_argument(
        "--results-dir",
        default=str(default_results_dir),
        help="directory to save per-run metrics and plots",
    )
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disable CUDA")
    parser.add_argument(
        "--use-proper-time-limits",
        action="store_true",
        default=False,
        help="compute returns taking time limits into account",
    )
    parser.add_argument("--recurrent-policy", action="store_true", default=False, help="use a recurrent policy")
    parser.add_argument("--use-linear-lr-decay", action="store_true", default=False, help="linearly decay LR")
    args = parser.parse_args()

    if args.env_name != "RobustLunarLander-v3":
        raise ValueError("This extracted A2C package only supports --env-name RobustLunarLander-v3")

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
