"""Stage-aware V critic.

Identical to :class:`VCritic` except that value normalization is performed
per stage. The stage of each centralized-critic sample is read from the
trailing stage one-hot that the env appends to the centralized state when
``causal_critic_state: True`` (``[..., o_flag, a_flag]``), so the o-stage value
``V^o`` and the a-stage value ``V^a`` are normalized with independent running
statistics (see :class:`StageValueNorm`).

Only the value-normalization path changes; the network, clipping, and loss are
inherited from :class:`VCritic`.
"""
import torch

from harl.algorithms.critics.v_critic import VCritic
from harl.common.stage_value_norm import StageValueNorm
from harl.utils.envs_tools import check
from harl.utils.models_tools import huber_loss, mse_loss


class StageVCritic(VCritic):
    """V critic with per-stage value normalization."""

    def __init__(self, args, cent_obs_space, num_stages=2, device=torch.device("cpu")):
        super().__init__(args, cent_obs_space, device)
        self.num_stages = num_stages

    def cal_value_loss(
        self, values, value_preds_batch, return_batch, value_normalizer, stage_ids
    ):
        """Value loss with per-stage normalization of the return targets."""
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )

        # normalize the return targets per stage (and update per-stage stats)
        normalized_return = torch.zeros_like(return_batch)
        for s in range(self.num_stages):
            row_mask = (stage_ids == s).squeeze(-1)
            if row_mask.any():
                rb_s = return_batch[row_mask]  # (M, 1) keeps the value dim
                value_normalizer[s].update(rb_s)
                normalized_return[row_mask] = value_normalizer[s].normalize(rb_s)

        error_clipped = normalized_return - value_pred_clipped
        error_original = normalized_return - values

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        return value_loss.mean()

    def update(self, sample, value_normalizer=None):
        """Update critic network with per-stage value normalization."""
        assert isinstance(value_normalizer, StageValueNorm), (
            "StageVCritic requires a StageValueNorm value normalizer."
        )
        (
            share_obs_batch,
            rnn_states_critic_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
        ) = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        # stage id from the trailing stage one-hot: [..., o_flag, a_flag]
        stage_ids = check(share_obs_batch[..., -self.num_stages :]).to(**self.tpdv)
        stage_ids = stage_ids.argmax(dim=-1, keepdim=True)

        values, _ = self.get_values(
            share_obs_batch, rnn_states_critic_batch, masks_batch
        )

        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, value_normalizer, stage_ids
        )

        self.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            import torch.nn as nn

            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
        else:
            from harl.utils.models_tools import get_grad_norm

            critic_grad_norm = get_grad_norm(self.critic.parameters())

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm
