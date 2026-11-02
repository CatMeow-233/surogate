"""Training objectives (docs/spec.md, I5): masked next-token cross entropy and the KL-to-seed anchor (MS §8.2).

Two conventions meet here. `masked_ce` takes a token sequence and predicts tokens[:, t] from logits[:, t - 1] (the
shard's loss-mask convention). The loop works on shard windows, where `targets[:, t]` is already the token that
logits[:, t] predict and `mask[:, t]` is that target's loss flag; `target_ce_sum` and `kl_sum` take that form.

Both window objectives return **sums** over the selected positions, so the loop can divide by the number of targets
of the whole optimizer step (every micro batch together) and weight every target equally. Positions with
mask = 0 are dropped before the loss is computed: their labels never enter the computation, so changing them leaves
the loss and the gradients bitwise unchanged (I5-T10). Losses are computed in fp32.
"""

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


def _select(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape != logits.shape[:-1]:
        raise ValueError(f"mask {list(mask.shape)} does not match logits {list(logits.shape[:-1])}")
    return logits[mask.to(torch.bool)].float()


def target_ce_sum(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum of CE(logits[b, t], targets[b, t]) over the positions with mask[b, t] = 1 (fp32)."""
    selected = _select(logits, mask)
    return F.cross_entropy(selected, targets[mask.to(torch.bool)], reduction="sum")


def kl_sum(logits: torch.Tensor, reference_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum over the positions with mask = 1 of KL(p_ref || p) = sum_v p_ref (log p_ref - log p), fp32 (MS §8.2:
    D_KL(p_seed || p_i)). Gradients flow into `logits` only."""
    log_p = F.log_softmax(_select(logits, mask), dim=-1)
    log_ref = F.log_softmax(_select(reference_logits.detach(), mask), dim=-1)
    return (log_ref.exp() * (log_ref - log_p)).sum()
