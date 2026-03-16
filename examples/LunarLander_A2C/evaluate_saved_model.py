#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from evaluation import evaluate


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    default_results_dir = Path(__file__).resolve().parent / "results" / "critic_compare"
    parser = argparse.ArgumentParser(description="Evaluate a saved A2C-V/A2C-Q model on standard or perturbed LunarLander.")
    parser.add_argument("--model-path", default=None, help="path to a saved model checkpoint; overrides results-dir lookup")
    parser.add_argument(
        "--results-dir",
        default=str(default_results_dir),
        help="base results directory used to infer the model path when --model-path is omitted",
    )
    parser.add_argument("--critic-type", choices=("v", "q"), default="v", help="critic family when inferring model path")
    parser.add_argument("--seed", type=int, default=0, help="training seed used in the saved run directory")
    parser.add_argument(
        "--model-kind",
        choices=("best", "last"),
        default="best",
        help="which saved checkpoint to load when inferring the model path",
    )
    parser.add_argument("--env-name", default="RobustLunarLander-v3", help="environment id to evaluate")
    parser.add_argument("--eval-seed", type=int, default=123, help="seed for the evaluation environment")
    parser.add_argument("--num-processes", type=int, default=1, help="number of vectorized eval envs")
    parser.add_argument("--num-episodes", type=int, default=100, help="number of evaluation episodes")
    parser.add_argument("--log-dir", default="/tmp/gym_saved_model_eval", help="temporary monitor log directory")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disable CUDA")
    parser.add_argument("--gravity", type=float, default=-10.0, help="evaluation gravity")
    parser.add_argument("--enable-wind", type=str2bool, default=False, help="enable wind during evaluation")
    parser.add_argument("--wind-power", type=float, default=15.0, help="wind power during evaluation")
    parser.add_argument("--turbulence-power", type=float, default=1.5, help="turbulence power during evaluation")
    parser.add_argument(
        "--noise-factor",
        default="none",
        choices=("none", "state", "reward", "action"),
        help="robust noise factor for evaluation",
    )
    parser.add_argument(
        "--noise-type",
        default="none",
        choices=("none", "gauss", "shift"),
        help="robust noise type for evaluation",
    )
    parser.add_argument("--noise-mu", type=float, default=0.0, help="noise mean for evaluation")
    parser.add_argument("--noise-sigma", type=float, default=0.0, help="noise sigma for evaluation")
    parser.add_argument("--noise-shift", type=float, default=0.0, help="noise shift for evaluation")
    parser.add_argument("--output-json", default=None, help="optional path to save evaluation summary JSON")
    return parser.parse_args()


def resolve_model_path(args) -> Path:
    if args.model_path is not None:
        return Path(args.model_path).expanduser().resolve()
    return (
        Path(args.results_dir).expanduser().resolve()
        / args.critic_type
        / f"seed_{args.seed}"
        / "models"
        / f"{args.model_kind}_model.pt"
    )


def torch_load_compat(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    if args.env_name != "RobustLunarLander-v3":
        raise ValueError("Only RobustLunarLander-v3 is supported.")

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model_path = resolve_model_path(args)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch_load_compat(model_path, device)
    if isinstance(checkpoint, dict):
        actor_critic = checkpoint["actor_critic"]
        obs_rms = checkpoint.get("obs_rms")
        metadata = checkpoint.get("metadata", {})
    elif isinstance(checkpoint, (list, tuple)) and len(checkpoint) >= 2:
        actor_critic, obs_rms = checkpoint[0], checkpoint[1]
        metadata = {}
    else:
        raise ValueError(f"Unsupported checkpoint format at {model_path}")

    actor_critic.to(device)
    actor_critic.eval()

    env_kwargs = {
        "gravity": args.gravity,
        "enable_wind": args.enable_wind,
        "wind_power": args.wind_power,
        "turbulence_power": args.turbulence_power,
        "noise_factor": args.noise_factor,
        "noise_type": args.noise_type,
        "noise_mu": args.noise_mu,
        "noise_sigma": args.noise_sigma,
        "noise_shift": args.noise_shift,
    }

    eval_result = evaluate(
        actor_critic=actor_critic,
        obs_rms=obs_rms,
        env_name=args.env_name,
        seed=args.eval_seed,
        num_processes=args.num_processes,
        eval_log_dir=os.path.expanduser(args.log_dir),
        device=device,
        num_episodes=args.num_episodes,
        env_kwargs=env_kwargs,
    )

    summary = {
        "model_path": str(model_path),
        "device": str(device),
        "loaded_metadata": metadata,
        "evaluation_env": {
            "env_name": args.env_name,
            **env_kwargs,
        },
        "num_episodes": eval_result["num_episodes"],
        "mean_reward": eval_result["mean_reward"],
        "std_reward": eval_result["std_reward"],
    }

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
