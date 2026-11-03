"""BilingualTrunk: the bilingual seed model as the main network of the bilingual-B table (B0-B3 subword, B4 H-Net,
B5 flat bytes), reached through surogate_bilingual's public RoutedLM.forward_hidden / logits.

The trunk is a RoutedLM with one expert named "seed" and plan=None, its configuration a bilingual size profile
(surogate_bilingual.config.PROFILES; "pilot" is the 24-layer, d 512 seed geometry) plus overrides of ModelConfig
fields (max_seq, for instance: an H-Net trunk reads up to one chunk per byte). Its initialisation is RoutedLM's, so
a subword arm's model is the production seed model itself (spec S12: equal to RoutedLM.forward; gate TW-3 compares
the training loops). vocab_size is the arm's codec vocabulary. As an H-Net main network (vocab_size=None) the trunk
still holds RoutedLM's tied embedding, at the smallest size ModelConfig allows (the 5 specials and one more id) and
frozen: it is never read.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from surogate_bilingual.config import N_SPECIAL, profile
from surogate_bilingual.model.lm import RoutedLM
from torch import nn

SEED_EXPERT = "seed"
FAMILY = "swiglu"  # the seed of family I; the study's dense trunk (STUDY §9.1)


class BilingualTrunk(nn.Module):
    def __init__(
        self,
        profile_name: str = "pilot",
        overrides: Mapping[str, Any] | None = None,
        *,
        vocab_size: int | None,
        init_seed: int = 0,
        checkpointing: bool = False,
    ):
        super().__init__()
        overrides = dict(overrides or {})
        refused = sorted({"vocab_size", "family", "expert_layers"} & set(overrides))
        if refused:
            raise ValueError(f"trunk overrides may not set {refused}: the vocabulary is the arm's, the trunk dense")
        cfg = profile(profile_name, FAMILY, vocab_size=vocab_size or N_SPECIAL + 1, **overrides)
        self.lm = RoutedLM(cfg, (SEED_EXPERT,), init_seed=init_seed)
        self.cfg, self.d_model, self.vocab_size = cfg, cfg.d_model, vocab_size
        self.checkpointing = checkpointing
        if vocab_size is None:
            self.lm.backbone.emb.weight.requires_grad_(False)

    def forward_hidden(self, h: torch.Tensor) -> torch.Tensor:
        return self.lm.forward_hidden(h, None, checkpointing=self.checkpointing)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        self._has_vocabulary()
        return self.lm.backbone.emb(ids)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        self._has_vocabulary()
        return self.lm.logits(h.to(self.lm.backbone.emb.weight.dtype))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.logits(self.forward_hidden(self.embed(ids)))

    def _has_vocabulary(self) -> None:
        if self.vocab_size is None:
            raise ValueError("this trunk is an H-Net main network without a vocabulary (vocab_size=None)")

    def non_embedding_parameters(self) -> int:
        return sum(p.numel() for name, p in self.lm.named_parameters() if name != "backbone.emb.weight")
