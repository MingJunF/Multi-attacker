from absl import flags

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

# Logger imports are guarded so that a missing *optional* environment
# dependency (e.g. Basilisk, SMAC, GRF) does not prevent the rest of the
# environments (such as robust_attack) from being trained.
LOGGER_REGISTRY = {}


def _register_logger(name, module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        LOGGER_REGISTRY[name] = getattr(module, class_name)
    except Exception as exc:  # pragma: no cover - optional deps may be absent
        import warnings

        warnings.warn(
            f"Could not register logger for env '{name}' "
            f"({module_path}.{class_name}): {exc}"
        )


_register_logger("smac", "harl.envs.smac.smac_logger", "SMACLogger")
_register_logger("smacv2", "harl.envs.smacv2.smacv2_logger", "SMACv2Logger")
_register_logger("mamujoco", "harl.envs.mamujoco.mamujoco_logger", "MAMuJoCoLogger")
_register_logger(
    "pettingzoo_mpe",
    "harl.envs.pettingzoo_mpe.pettingzoo_mpe_logger",
    "PettingZooMPELogger",
)
_register_logger("gym", "harl.envs.gym.gym_logger", "GYMLogger")
_register_logger("football", "harl.envs.football.football_logger", "FootballLogger")
_register_logger("dexhands", "harl.envs.dexhands.dexhands_logger", "DexHandsLogger")
_register_logger("lag", "harl.envs.lag.lag_logger", "LAGLogger")
_register_logger("SatBench", "harl.envs.Bsk_wrapper", "SatBenchLogger")
_register_logger(
    "robust_attack",
    "harl.envs.robust_attack.robust_attack_logger",
    "RobustAttackLogger",
)
_register_logger(
    "robust_victim",
    "harl.envs.robust_attack.robust_attack_logger",
    "RobustVictimLogger",
)

