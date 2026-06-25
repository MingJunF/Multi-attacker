"""Single-agent victim-training environment on top of robust_gymnasium.

This wrapper exposes a robust_gymnasium task as a *single-agent* HARL
environment so that a "standard" PPO MuJoCo agent can be trained with the HARL
on-policy runner (``mappo`` / ``happo`` with one agent behaves as PPO).

Why a dedicated wrapper (instead of HARL's ``gym`` env)?
    The attacker environment (:class:`RobustAttackEnv`) builds its victim env
    with ``robust_gymnasium.make(scenario)``. To later load the trained policy
    as the victim, the policy MUST share the exact observation/action spaces of
    that same ``robust_gymnasium`` task. HARL's built-in ``gym`` wrapper uses the
    legacy ``gym`` package whose MuJoCo observation dimensions differ from
    ``robust_gymnasium`` (e.g. Ant-v4 obs = 27 here). Training the victim through
    this wrapper guarantees the spaces match.

The built-in robust_gymnasium perturbations are disabled (a clean victim is
trained); the attacker is added only afterwards in :class:`RobustAttackEnv`.
"""

import copy

import numpy as np
import robust_gymnasium as rgym
from robust_gymnasium.spaces.box import Box

from harl.envs.robust_attack.robust_attack_env import _build_disabled_robust_config


class RobustVictimEnv:
    """Single-agent HARL wrapper around a robust_gymnasium task."""

    def __init__(self, args):
        self.args = copy.deepcopy(args)
        scenario = self.args["scenario"]

        render_mode = self.args.get("render_mode", None)
        if render_mode is not None:
            self.env = rgym.make(scenario, render_mode=render_mode)
        else:
            self.env = rgym.make(scenario)

        # Maze tasks (PointMaze/AntMaze) use a plain-action ``step(action)`` API
        # and read their perturbation settings from a module-level global, unlike
        # the mujoco/fetch tasks whose ``step`` expects a ``robust_input`` dict.
        self._maze_api = scenario.startswith("PointMaze") or scenario.startswith(
            "AntMaze"
        )
        if self._maze_api:
            # Disable the maze's built-in global-args noise so the victim trains
            # on a clean environment (the attacker is added only later).
            try:
                import robust_gymnasium.envs.robust_maze.point_maze as _pm

                _pm.args.noise_factor = "disable"
            except Exception:
                pass

        # Goal-conditioned tasks expose a Dict observation
        # {observation, achieved_goal, desired_goal}. HARL's MLP policy needs a
        # flat Box, so we flatten it to concat(observation, desired_goal): this
        # keeps the per-episode goal visible to the victim (and, later, lets the
        # observation attacker tamper with the goal).
        raw_obs_space = self.env.observation_space
        self._goal_conditioned = raw_obs_space.__class__.__name__ == "Dict"
        if self._goal_conditioned:
            obs_dim = int(np.prod(raw_obs_space["observation"].shape)) + int(
                np.prod(raw_obs_space["desired_goal"].shape)
            )
            flat_obs_space = Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
        else:
            flat_obs_space = self.env.observation_space

        self.n_agents = 1
        self.share_observation_space = [flat_obs_space]
        self.observation_space = [flat_obs_space]
        self.action_space = [self.env.action_space]
        self.discrete = self.env.action_space.__class__.__name__ != "Box"

        # Clean victim training: no intrinsic robust_gymnasium noise.
        self._robust_config = _build_disabled_robust_config(
            self.args.get("background_noise")
        )
        self._robust_type = self.args.get("robust_type", "disable")
        self._seed = self.args.get("seed", None)

    def _flatten_obs(self, obs):
        """Flatten a (possibly Dict) observation to a 1-D float32 array."""
        if self._goal_conditioned:
            return np.concatenate(
                [
                    np.asarray(obs["observation"], dtype=np.float32).reshape(-1),
                    np.asarray(obs["desired_goal"], dtype=np.float32).reshape(-1),
                ]
            )
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _victim_step(self, action):
        if self._maze_api:
            # Maze tasks take a plain action and ignore the robust_input dict.
            return self.env.step(np.asarray(action, dtype=np.float32))
        robust_input = {
            "action": np.asarray(action, dtype=np.float32),
            "robust_type": self._robust_type,
            "robust_config": self._robust_config,
        }
        return self.env.step(robust_input)

    def step(self, actions):
        """return local_obs, global_state, rewards, dones, infos, available_actions"""
        if self.discrete:
            action = actions.flatten()[0]
        else:
            action = actions[0]
        obs, rew, terminated, truncated, info = self._victim_step(action)
        obs = self._flatten_obs(obs)
        done = bool(terminated or truncated)
        if done and truncated and not terminated:
            info = dict(info)
            info["bad_transition"] = True
        return [obs], [obs], [[float(rew)]], [done], [info], self.get_avail_actions()

    def reset(self):
        """Returns initial observations and states"""
        if self._seed is not None:
            obs, _ = self.env.reset(seed=self._seed)
            self._seed = None
        else:
            obs, _ = self.env.reset()
        obs = self._flatten_obs(obs)
        return [obs], [obs], self.get_avail_actions()

    def get_avail_actions(self):
        if self.discrete:
            return [[1] * self.action_space[0].n]
        return None

    def seed(self, seed):
        self._seed = seed

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
