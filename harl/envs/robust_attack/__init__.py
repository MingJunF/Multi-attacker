"""Multi-attacker environment package for HARL + robust_gymnasium.

Exposes a cooperative two-agent (observation-attacker + action-attacker)
environment that can be trained with independent PPO, MAPPO or HAPPO to attack
a fixed victim policy inside a robust_gymnasium task.
"""

from harl.envs.robust_attack.robust_attack_env import RobustAttackEnv
from harl.envs.robust_attack.robust_attack_logger import RobustAttackLogger

__all__ = ["RobustAttackEnv", "RobustAttackLogger"]
