"""Return-anchored counterfactual action-value Q^o(s_t, delta_o) for the obs attacker.

Plan (a) of the stage-aware obs-advantage fix. The stage residual
``delta_o = V^a - V^o`` is a difference of two large learned values, so its
signal-to-noise ratio collapses in the small-epsilon / robust-victim regime
(see the stage diagnostics). This network instead learns the obs-stage value as
a function of the *chosen* perturbation,

    Q^o(s_t, delta_o) ~= E[ G_t | s_t, delta_o ],

regressing the o-stage lambda-return ``returns[:, :, LEADER]`` (a real-return
anchor, not a value subtraction) on ``(s_t, delta_o)``. Paired with the stage
critic's state value ``V^o(s_t)`` as a (perturbation-independent, hence
unbiased) baseline, it yields the counterfactual obs advantage

    A^o_t = Q^o(s_t, delta_o) - V^o(s_t).

Crucially Q^o does NOT take the downstream perturbation ``delta_a`` as input, so
the act attacker's response is *marginalized out by the regression* rather than
held fixed. This respects the ordered causal structure (delta_o -> x^a ->
delta_a) and keeps the baseline independent of the obs action, so the obs
policy gradient stays unbiased while gaining a strong return anchor.
"""
import torch
import torch.nn as nn

from harl.models.base.mlp import MLPBase
from harl.utils.envs_tools import check
from harl.utils.models_tools import init, get_init_method


class ObsCounterfactualQ(nn.Module):
    """MLP estimator of Q^o(s_t, delta_o) trained on the o-stage lambda-return."""

    def __init__(self, args, state_dim, action_dim, device=torch.device("cpu")):
        """
        Args:
            args: (dict) model args (hidden_sizes, activation, init, etc.).
            state_dim: (int) dimension of the clean state slice s_t.
            action_dim: (int) dimension of the obs perturbation delta_o.
            device: (torch.device) device to run on.
        """
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        input_dim = state_dim + action_dim
        self.base = MLPBase(args, [input_dim])
        init_method = get_init_method(args["initialization_method"])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.q_out = init_(nn.Linear(args["hidden_sizes"][-1], 1))
        self.optimizer = None
        self.to(device)

    def setup_optimizer(self, lr, eps, weight_decay):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, eps=eps, weight_decay=weight_decay
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q_out(self.base(x))

    @torch.no_grad()
    def get_values(self, state, action):
        state = check(state).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        return self.forward(state, action)

    def train_q(self, state, action, target, epochs, max_grad_norm):
        """Full-batch MSE regression of the lambda-return target onto (s, delta_o).

        Returns the final-epoch loss (float).
        """
        state = check(state).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        target = check(target).to(**self.tpdv)
        last_loss = 0.0
        for _ in range(epochs):
            pred = self.forward(state, action)
            loss = ((pred - target) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()
            last_loss = float(loss.item())
        return last_loss
