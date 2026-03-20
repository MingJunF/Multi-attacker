from __future__ import annotations

import csv
import json
from pathlib import Path


TRAINING_FIELDNAMES = [
    "critic_type",
    "seed",
    "update",
    "timesteps",
    "learning_rate",
    "critic_loss",
    "action_loss",
    "dist_entropy",
    "actor_signal_variance",
    "critic_target_variance",
]

EVALUATION_FIELDNAMES = [
    "critic_type",
    "seed",
    "update",
    "timesteps",
    "stage",
    "num_episodes",
    "mean_reward",
    "std_reward",
]


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


class ExperimentLogger:
    def __init__(self, run_dir: Path, config: dict):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.training_path = self.run_dir / "training_metrics.csv"
        self.evaluation_path = self.run_dir / "evaluations.csv"
        self.final_rewards_path = self.run_dir / "final_eval_rewards.json"
        self.summary_path = self.run_dir / "summary.json"
        self.config_path = self.run_dir / "config.json"

        self._initialize_csv(self.training_path, TRAINING_FIELDNAMES)
        self._initialize_csv(self.evaluation_path, EVALUATION_FIELDNAMES)
        save_json(self.config_path, config)

    def _initialize_csv(self, path: Path, fieldnames: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

    def _append_row(self, path: Path, fieldnames: list[str], row: dict) -> None:
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerow(row)

    def append_training_metric(self, row: dict) -> None:
        self._append_row(self.training_path, TRAINING_FIELDNAMES, row)

    def append_evaluation(self, row: dict) -> None:
        self._append_row(self.evaluation_path, EVALUATION_FIELDNAMES, row)

    def save_final_eval_rewards(self, payload: dict) -> None:
        save_json(self.final_rewards_path, payload)

    def save_summary(self, payload: dict) -> None:
        save_json(self.summary_path, payload)
