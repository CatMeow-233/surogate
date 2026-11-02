"""Learned-router interface and the gate function that keeps adding an expert monotone (MS §8.5; MA-11).

Training a router is deferred (docs/spec.md, "推迟项"); no P0 run uses this module. It fixes the gate form now,
because the form decides whether adding an expert disturbs the tokens that do not choose it:

- `topk_softmax_gates`: take the top k of the logits, softmax over the selected logits only, zeros elsewhere. With
  the old router's columns frozen and the old logits computed by the frozen sub-matrix in its own matmul
  (`logits_old`), the new experts' logits in another (`logits_new`), every token whose top k contains no new
  expert keeps bitwise the gates it had before: the selected values are the same numbers and the softmax runs on a
  tensor of the same shape. The I3 test checks this in fp32 and fp64 on CPU.
- Forbidden forms, kept out of the package: Switch-style raw probabilities (softmax over all E logits, then top k
  without renormalising) change every token when an expert is added; a softmax over all E + 1 logits followed by a
  gather and renormalisation is not bitwise invariant in fp32 (MA-11).

The rows of the result lie on the simplex with at most k nonzero entries; a `RoutePlan` in P0 allows k <= 2. Ties
between logits at the k-th place are broken by `torch.topk` and are not part of the invariance.
"""

from __future__ import annotations

from typing import Protocol

import torch


def topk_softmax_gates(logits_old: torch.Tensor, logits_new: torch.Tensor | None, k: int) -> torch.Tensor:
    """Gates [..., E_old + E_new] from logits_old [..., E_old] and logits_new [..., E_new] (or None)."""
    if logits_new is not None and logits_new.shape[:-1] != logits_old.shape[:-1]:
        raise ValueError(f"logits_old {list(logits_old.shape)} and logits_new {list(logits_new.shape)} disagree")
    logits = logits_old if logits_new is None else torch.cat([logits_old, logits_new], dim=-1)
    if not 1 <= k <= logits.shape[-1]:
        raise ValueError(f"k must lie in [1, {logits.shape[-1]}], got {k}")
    values, index = torch.topk(logits, k, dim=-1)
    return torch.zeros_like(logits).scatter(-1, index, torch.softmax(values, dim=-1))


class Router(Protocol):
    """A learned router: features [..., F] -> logits [..., E]. Deferred (MS §8.5: R-early reads u_0, R-late h^L);
    its parameters count towards P_active and P_update within the single path's 3,768,064 slack."""

    def __call__(self, features: torch.Tensor) -> torch.Tensor: ...
