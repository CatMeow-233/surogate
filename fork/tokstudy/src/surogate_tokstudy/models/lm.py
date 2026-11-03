"""Token LMs: a main network's embedding, its blocks and its tied head, over subword ids (SubwordLM) or byte ids
(ByteLM, id = 5 + byte, the codec's vocabulary of 261 or its `pad_to`). The two differ only in what the ids mean;
both read the main network's own embedding (A0Twin, BilingualTrunk), so a subword arm's model is the twin or the
production seed model itself. A main network is any module with d_model, vocab_size, embed(ids), forward_hidden(h)
(blocks and final norm) and logits(h) (the tied head).
"""

from __future__ import annotations

import torch
from torch import nn

from ..codecs import BYTE_VOCAB


class TokenLM(nn.Module):
    kind = "token"

    def __init__(self, main: nn.Module):
        super().__init__()
        if getattr(main, "vocab_size", None) is None:
            raise ValueError("a token LM needs a main network built with a vocabulary")
        self.main = main

    @property
    def vocab_size(self) -> int:
        return self.main.vocab_size

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Logits [B, L, V] for ids [B, L]; position t predicts id t + 1."""
        return self.main.logits(self.main.forward_hidden(self.main.embed(ids)))


class SubwordLM(TokenLM):
    kind = "subword"


class ByteLM(TokenLM):
    kind = "bytes"

    def __init__(self, main: nn.Module):
        super().__init__(main)
        if main.vocab_size < BYTE_VOCAB:
            raise ValueError(
                f"a byte LM needs at least {BYTE_VOCAB} ids (5 specials + 256 bytes), not {main.vocab_size}"
            )
