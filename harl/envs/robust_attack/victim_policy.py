"""Victim policy abstractions for the multi-attacker environment.

The victim is a *fixed* (non-learning) policy that acts inside a
robust_gymnasium environment. The multi-attacker agents try to degrade the
victim's return by perturbing the observation it sees and the action it takes.

Supported victim types (selected via ``config["type"]``):
    - "random": samples uniformly from the action space (default, no checkpoint).
    - "zero"  : always outputs a zero / no-op action.
    - "harl"  : loads a trained HARL ``StochasticPolicy`` actor checkpoint.
    - "callable": a user supplied python callable ``fn(obs) -> action`` passed
                  programmatically through ``config["fn"]``.
"""

import numpy as np


class VictimPolicy:
    """Base class for a fixed victim policy."""

    def __init__(self, obs_space, action_space):
        self.obs_space = obs_space
        self.action_space = action_space

    def reset(self):
        """Reset any internal (e.g. recurrent) state at episode boundaries."""

    def act(self, obs):
        """Return the victim action for a single (possibly perturbed) observation.

        Args:
            obs: (np.ndarray) observation seen by the victim.
        Returns:
            action: (np.ndarray) victim action.
        """
        raise NotImplementedError


class RandomVictim(VictimPolicy):
    """Victim that samples actions uniformly from the action space."""

    def act(self, obs):
        return np.asarray(self.action_space.sample(), dtype=np.float32)


class ZeroVictim(VictimPolicy):
    """Victim that always outputs a zero action (no-op)."""

    def act(self, obs):
        shape = self.action_space.shape
        return np.zeros(shape, dtype=np.float32)


class CallableVictim(VictimPolicy):
    """Victim wrapping a user supplied callable ``fn(obs) -> action``."""

    def __init__(self, obs_space, action_space, fn):
        super().__init__(obs_space, action_space)
        if not callable(fn):
            raise ValueError("CallableVictim requires a callable `fn`.")
        self.fn = fn

    def act(self, obs):
        return np.asarray(self.fn(obs), dtype=np.float32)


class HARLVictim(VictimPolicy):
    """Victim that loads a trained HARL ``StochasticPolicy`` actor checkpoint.

    The checkpoint is expected to be a ``state_dict`` saved from a HARL actor
    (e.g. ``actor_agent0.pt``). The network architecture is rebuilt from
    ``model_args`` (hidden sizes etc.) before loading the weights.
    """

    def __init__(self, obs_space, action_space, model_path, model_args=None,
                 device="cpu", deterministic=True):
        super().__init__(obs_space, action_space)
        import torch
        from harl.models.policy_models.stochastic_policy import StochasticPolicy

        self.torch = torch
        self.device = torch.device(device)
        self.deterministic = deterministic

        # Sensible defaults matching HARL's on-policy config so a bare
        # checkpoint can be loaded without passing every model hyper-parameter.
        default_args = {
            "hidden_sizes": [128, 128],
            "activation_func": "relu",
            "use_feature_normalization": True,
            "initialization_method": "orthogonal_",
            "gain": 0.01,
            "use_naive_recurrent_policy": False,
            "use_recurrent_policy": False,
            "recurrent_n": 1,
            "data_chunk_length": 10,
            "use_policy_active_masks": True,
            "std_x_coef": 1.0,
            "std_y_coef": 0.5,
        }
        if model_args:
            default_args.update(model_args)
        self.model_args = default_args
        self.recurrent_n = default_args["recurrent_n"]
        self.hidden_size = default_args["hidden_sizes"][-1]

        self.actor = StochasticPolicy(
            default_args, obs_space, action_space, self.device
        )
        state_dict = torch.load(model_path, map_location=self.device)
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

        self._rnn_states = None
        self.reset()

    def reset(self):
        self._rnn_states = np.zeros(
            (1, self.recurrent_n, self.hidden_size), dtype=np.float32
        )

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        masks = np.ones((1, 1), dtype=np.float32)
        with self.torch.no_grad():
            actions, _, rnn_states = self.actor(
                obs, self._rnn_states, masks, deterministic=self.deterministic
            )
        self._rnn_states = rnn_states.detach().cpu().numpy()
        action = actions.detach().cpu().numpy()[0]
        return action.astype(np.float32)


def make_victim_policy(config, obs_space, action_space):
    """Factory that builds a victim policy from a config dictionary.

    Args:
        config: (dict) victim configuration. Key ``type`` selects the victim.
        obs_space: (gym.Space) victim observation space.
        action_space: (gym.Space) victim action space.
    Returns:
        VictimPolicy instance.
    """
    config = config or {}
    victim_type = config.get("type", "random")

    if victim_type == "random":
        return RandomVictim(obs_space, action_space)
    if victim_type == "zero":
        return ZeroVictim(obs_space, action_space)
    if victim_type == "callable":
        return CallableVictim(obs_space, action_space, config["fn"])
    if victim_type == "harl":
        return HARLVictim(
            obs_space,
            action_space,
            model_path=config["model_path"],
            model_args=config.get("model_args"),
            device=config.get("device", "cpu"),
            deterministic=config.get("deterministic", True),
        )
    raise ValueError(f"Unknown victim policy type: {victim_type}")
