"""Stochastic policy whose PPO objective only counts the first
``valid_action_dim`` action dimensions.

In the ordered obs->act attack the follower (action attacker) only emits an
``act_dim``-wide perturbation, but its reported action space is padded to
``pad_dim = max(obs_dim, act_dim)`` so the runner can stack heterogeneous
attacker actions into a single rectangular array. With a plain Gaussian policy
the padding dimensions are still sampled AND their log-prob / entropy enter the
PPO importance ratio and entropy bonus, polluting the follower's policy
gradient. ``MaskedStochasticPolicy`` keeps the padded sampling (so the env /
buffer / rollout stay rectangular) but zeroes the log-prob of the padding
dimensions and sums entropy only over the valid ones, so the padding dims carry
no gradient and never affect the ratio.

With ``action_aggregation: prod`` the importance weight is
``prod_k exp(logp_k - logp_old_k)``; zeroing the padding log-probs in BOTH the
stored (old) and re-evaluated (new) distributions makes those factors exactly
1, i.e. the ratio reduces to the product over the valid dims only.
"""
import torch

from harl.models.base.distributions import FixedNormal, DiagGaussian
from harl.models.policy_models.stochastic_policy import StochasticPolicy


class MaskedFixedNormal(FixedNormal):
    """Diagonal Gaussian whose log-prob/entropy only count the first
    ``valid_dim`` dimensions (the remaining padding dims are inert)."""

    def __init__(self, loc, scale, valid_dim, validate_args=None):
        super().__init__(loc, scale, validate_args=validate_args)
        self.valid_dim = valid_dim

    def log_probs(self, actions):
        # per-dimension log-prob, shape (..., action_dim)
        log_probs = super().log_probs(actions)
        if self.valid_dim < log_probs.shape[-1]:
            mask = torch.zeros_like(log_probs)
            mask[..., : self.valid_dim] = 1.0
            log_probs = log_probs * mask
        return log_probs

    def entropy(self):
        # base Normal entropy is per-dimension; sum only over the valid dims
        per_dim_entropy = torch.distributions.Normal.entropy(self)
        return per_dim_entropy[..., : self.valid_dim].sum(-1)


class MaskedDiagGaussian(DiagGaussian):
    """``DiagGaussian`` head that emits a :class:`MaskedFixedNormal`."""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        valid_dim,
        initialization_method="orthogonal_",
        gain=0.01,
        args=None,
    ):
        super().__init__(num_inputs, num_outputs, initialization_method, gain, args)
        self.valid_dim = valid_dim

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return MaskedFixedNormal(action_mean, action_std, self.valid_dim)


class MaskedStochasticPolicy(StochasticPolicy):
    """Stochastic policy whose PPO log-prob/entropy ignore the padding action
    dimensions beyond ``valid_action_dim``."""

    def __init__(
        self,
        args,
        obs_space,
        action_space,
        valid_action_dim,
        device=torch.device("cpu"),
    ):
        super().__init__(args, obs_space, action_space, device)
        assert action_space.__class__.__name__ == "Box", (
            "MaskedStochasticPolicy only supports Box (continuous) action spaces."
        )
        self.valid_action_dim = valid_action_dim
        action_dim = action_space.shape[0]
        if valid_action_dim < action_dim:
            # replace the Gaussian head with a masked one (shares everything
            # else: backbone, optional rnn). New parameters; rebuild the
            # optimizer at the call site.
            self.act.action_out = MaskedDiagGaussian(
                self.hidden_sizes[-1],
                action_dim,
                valid_action_dim,
                self.initialization_method,
                self.gain,
                args,
            )
        self.to(device)
