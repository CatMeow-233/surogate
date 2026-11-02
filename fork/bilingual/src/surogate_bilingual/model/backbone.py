"""The shared backbone theta_S (MS §4.1): the embedding (also the tied LM head), every attention block, every norm,
the per-layer readouts Q, b of family II, and for LP-k the frozen seed FFN of the layers that hold no expert. It
holds no expert; the bank does.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig
from ..ops.attention import CausalSelfAttention
from ..ops.ffn import SlotReadout, make_expert
from ..ops.norm import RMSNorm
from ..ops.rope import RopeCache


class SharedLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, gqa: str, seed_ffn: bool):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg, gqa)
        self.norm_ffn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.readout = SlotReadout(cfg.d_model, cfg.ffn_width) if cfg.family == "slot" else None
        self.ffn = make_expert(cfg) if seed_ffn else None  # LP-k: the frozen seed FFN of a non-expert layer


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig, gqa: str):
        super().__init__()
        shared = set(cfg.shared_ffn_layers)
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(SharedLayer(cfg, gqa, layer in shared) for layer in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.rope = RopeCache(cfg.head_dim, cfg.rope_theta)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """Logits through the tied head: the embedding matrix is the output projection."""
        return F.linear(self.final_norm(h), self.emb.weight)


def backbone_keys(cfg: ModelConfig) -> list[str]:
    """The state-dict keys of the backbone, in the naming of the packs (docs/spec.md, "pack")."""
    shared = set(cfg.shared_ffn_layers)
    keys = ["backbone.emb.weight", "backbone.final_norm.weight"]
    for layer in range(cfg.n_layers):
        prefix = f"backbone.layers.{layer}."
        keys += [prefix + k for k in ("norm_attn.weight", "norm_ffn.weight")]
        keys += [f"{prefix}attn.{p}.weight" for p in "qkvo"]
        if cfg.family == "slot":
            keys += [prefix + "readout.Q.weight", prefix + "readout.b"]
        if layer in shared:
            keys += [f"{prefix}ffn.{k}" for k in ffn_param_names(cfg)]
    return sorted(keys)


def ffn_param_names(cfg: ModelConfig) -> tuple[str, ...]:
    return ("gate.weight", "up.weight", "down.weight") if cfg.family == "swiglu" else ("R.weight", "W.weight")
