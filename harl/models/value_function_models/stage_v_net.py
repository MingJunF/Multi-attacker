"""Two-head V network for Stage-Aware MAPPO."""
import torch
import torch.nn as nn

from harl.models.value_function_models.v_net import VNet
from harl.utils.envs_tools import check
from harl.utils.models_tools import init, get_init_method


class StageVNet(VNet):
    """V network with two stage-specific value heads on a shared backbone.

    Stage-Aware MAPPO needs two coupled stage values:
        V^o(x^o) -- obs stage (x^o = [s, 0, 1, 0])
        V^a(x^a) -- act stage (x^a = [s, victim.act, 0, 1])
    A single critic head forces both stage targets through the same final layer
    (so they pull each other; this is part of why the single-head version
    behaves like IPPO). Here the shared backbone (and optional RNN) is reused,
    but two separate heads ``v_out_obs`` / ``v_out_act`` produce V^o and V^a.

    The active head is selected PER SAMPLE from the trailing 2-dim stage one-hot
    [obs_stage, act_stage] that the robust_attack env appends to the centralized
    state. The inactive head is gated to zero, so each head only receives
    gradient from its own stage's samples while the backbone is shared.
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super().__init__(args, cent_obs_space, device)
        init_method = get_init_method(self.initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # Two stage-specific heads. The parent's ``v_out`` is left in place but
        # unused (it simply receives no gradient).
        self.v_out_obs = init_(nn.Linear(self.hidden_sizes[-1], 1))
        self.v_out_act = init_(nn.Linear(self.hidden_sizes[-1], 1))
        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        v_obs = self.v_out_obs(critic_features)
        v_act = self.v_out_act(critic_features)
        # Stage gate from the trailing one-hot [obs_stage, act_stage].
        obs_gate = cent_obs[..., -2:-1]
        act_gate = cent_obs[..., -1:]
        values = obs_gate * v_obs + act_gate * v_act

        return values, rnn_states
