"""Stage-aware value normalizer.

Holds one :class:`ValueNorm` per stage so the o-stage value ``V^o`` and the
a-stage value ``V^a`` maintain independent running normalization statistics.
The two stages have different return scales (the a-stage carries the reward and
the discounted next-step value, while the o-stage is a within-step bootstrap),
so sharing a single normalizer biases the critic toward whichever stage
dominates the running statistics.

This is a thin container: callers index ``self.norms[stage]`` and use the
ordinary :class:`ValueNorm` API. Being an :class:`nn.Module` with an
``nn.ModuleList`` of sub-normalizers, ``state_dict`` / ``load_state_dict`` work
out of the box for checkpointing.
"""
import torch
import torch.nn as nn

from harl.common.valuenorm import ValueNorm


class StageValueNorm(nn.Module):
    """A list of per-stage :class:`ValueNorm` normalizers."""

    def __init__(self, num_stages=2, device=torch.device("cpu")):
        super().__init__()
        self.num_stages = num_stages
        self.norms = nn.ModuleList(
            [ValueNorm(1, device=device) for _ in range(num_stages)]
        )

    def __getitem__(self, stage):
        return self.norms[stage]

    def __len__(self):
        return self.num_stages
