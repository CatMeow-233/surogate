"""Training objectives. Step I2: the masked next-token cross entropy; the KL-to-seed anchor lands in step I5."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_ce(logits: torch.Tensor, tokens: torch.Tensor, loss_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean CE of predicting tokens[:, t] from logits[:, t - 1] over the positions t >= 1 with loss_mask[:, t] = 1
    (the shard convention, docs/spec.md). Computed in fp32."""
    pred = logits[:, :-1].float()
    target = tokens[:, 1:]
    losses = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="none")
    if loss_mask is None:
        return losses.mean()
    mask = loss_mask[:, 1:].reshape(-1).to(losses.dtype)
    if float(mask.sum()) == 0:
        raise ValueError("loss_mask selects no target")
    return (losses * mask).sum() / mask.sum()
