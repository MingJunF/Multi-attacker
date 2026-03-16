from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from a2c_v.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_v.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, critic_type="v"):
        super().__init__()
        if base_kwargs is None:
            base_kwargs = {}
        self.critic_type = critic_type
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        base_kwargs = dict(base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            if self.critic_type == "q":
                base_kwargs["critic_outputs"] = num_outputs
            dist_cls = Categorical
        elif action_space.__class__.__name__ == "Box":
            if self.critic_type == "q":
                raise NotImplementedError("Q-based critic is only supported for discrete actions.")
            num_outputs = action_space.shape[0]
            dist_cls = DiagGaussian
        elif action_space.__class__.__name__ == "MultiBinary":
            if self.critic_type == "q":
                raise NotImplementedError("Q-based critic is only supported for discrete actions.")
            num_outputs = action_space.shape[0]
            dist_cls = Bernoulli
        else:
            raise NotImplementedError

        base_kwargs["critic_type"] = self.critic_type
        self.base = base(obs_shape[0], **base_kwargs)
        self.dist = dist_cls(self.base.output_size, num_outputs)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def _policy_value(self, critic_output, dist):
        if self.critic_type == "q":
            return (dist.probs * critic_output).sum(dim=-1, keepdim=True)
        return critic_output

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        critic_output, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        policy_value = self._policy_value(critic_output, dist)
        return policy_value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        critic_output, actor_features, _ = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        return self._policy_value(critic_output, dist)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        critic_output, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        policy_value = self._policy_value(critic_output, dist)
        if self.critic_type == "q":
            critic_output = critic_output.gather(-1, action.long())
        return critic_output, policy_value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super().__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # Restore the flattened rollout batch to time-major form for GRU processing.
            N = hxs.size(0)
            T = int(x.size(0) / N)

            x = x.view(T, N, x.size(1))
            masks = masks.view(T, N)

            has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            has_zeros = [0] + has_zeros + [T]
            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1),
                )
                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0)
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, critic_type="v", critic_outputs=1):
        super().__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(hidden_size, critic_outputs))
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, critic_type="v", critic_outputs=1):
        super().__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2),
        )

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.critic_linear = init_(nn.Linear(hidden_size, critic_outputs))
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
