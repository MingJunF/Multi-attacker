"""Build robosuite manipulation envs behind the robust_attack interface.

robosuite is NOT gym-registered (there is no ``rgym.make`` entry point); it is
constructed with ``suite.make`` and adapted with a ``GymWrapper``. In this
robust_gymnasium fork the underlying ``MujocoEnv.step`` expects a
``robust_input`` dict (exactly like the mujoco/fetch tasks), and
``GymWrapper.step`` forwards its ``action`` argument verbatim to that step -- so
passing the ``robust_input`` dict through ``GymWrapper.step`` works unchanged.
The wrapper exposes a flat Box observation and a Box action space, so downstream
a robosuite task behaves like any standard (non-maze) robust_gymnasium env: it
uses the ``robust_input`` step path and a plain ``reshape(-1)`` observation.

Scenario naming convention:
    robosuite/<EnvName>            e.g. ``robosuite/Lift``  (default robot Panda)
    robosuite/<EnvName>/<Robot>    e.g. ``robosuite/Door/Panda``
"""

ROBOSUITE_PREFIX = "robosuite/"


def is_robosuite_scenario(scenario):
    """True if ``scenario`` selects a robosuite task (``robosuite/<Env>``)."""
    return isinstance(scenario, str) and scenario.startswith(ROBOSUITE_PREFIX)


def make_robosuite_env(scenario, render_mode=None, horizon=500, control_freq=20):
    """Construct ``GymWrapper(suite.make(<Env>, robots=<Robot>, ...))``.

    Camera/renderer are disabled (headless) and dense reward shaping is enabled
    so the victim policy has a learnable signal. The returned object is a
    gymnasium-style env whose ``step`` accepts the ``robust_input`` dict (via
    forwarding) and returns a flat Box observation.
    """
    import robust_gymnasium.envs.robosuite as suite
    from robust_gymnasium.envs.robosuite.wrappers.gym_wrapper import GymWrapper
    from robust_gymnasium.envs.robosuite.controllers import load_controller_config

    spec = scenario[len(ROBOSUITE_PREFIX):]
    parts = [p for p in spec.split("/") if p]
    env_name = parts[0]
    robot = parts[1] if len(parts) > 1 else "Panda"

    controller_configs = load_controller_config(default_controller="OSC_POSE")
    base = suite.make(
        env_name,
        robots=robot,
        controller_configs=controller_configs,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        horizon=int(horizon),
        control_freq=int(control_freq),
    )
    return GymWrapper(base)
