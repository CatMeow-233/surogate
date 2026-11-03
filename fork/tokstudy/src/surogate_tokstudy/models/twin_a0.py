"""A0Twin: TLC's A0 row in PyTorch, the main network of the tlc-N table (N0 subword, N1 H-Net, N2 flat bytes).

A0 is 12 x [RMSNorm -> Mamba2Mixer -> RMSNorm -> SwiGLU] at width 384, a final RMSNorm and, for a token arm, a tied
embedding / LM head (TLC's fused residual + RMSNorm computes the same sums as the pre-norm form). The geometry is
given in TLC's config.json keys (TWIN_KEYS), so a twin is built from a profile plus overrides of those keys
(`twin_config`), or from a TLC model directory (`from_tlc_model_dir`, gate TW-1: the same logits as
fork/tlc/prototypes/ref_mamba2.py, CPU, fp64). A twin built with `vocab_size=None` has no embedding: it is the H-Net
main network, which reads chunk vectors and returns hidden states (`forward_hidden`, final norm included).

Weight names follow the native loader (TLC_NAMES): blocks[i].<field> per layer, first_layer_<field> for layer 0
of A0-split, model.embed_tokens.weight and model.norm.weight. Only tied, all-Mamba rows are twins.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .mamba2 import Mamba2Geometry, Mamba2Stack

EMBEDDING_STD = 0.02  # surogate_tlc.init.EMBEDDING_STD
# The geometry of TLC's A0 (surogate_tlc.hf.default_config; a test compares the two). vocab_size is the arm's.
A0 = {
    "hidden_size": 384,
    "num_hidden_layers": 12,
    "intermediate_size": 1024,
    "rms_norm_eps": 1e-5,
    "mamba_num_heads": 12,
    "mamba_head_dim": 64,
    "ssm_state_size": 64,
    "n_groups": 1,
    "conv_kernel": 4,
    "chunk_size": 128,
}
PROFILES: dict[str, dict[str, Any]] = {
    "a0": A0,
    # CPU tests: two layers of width 32
    "tiny": {**A0, "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48, "mamba_num_heads": 4,
             "mamba_head_dim": 16, "ssm_state_size": 8, "chunk_size": 16},
}  # fmt: skip
TWIN_KEYS = frozenset(A0)
SPLIT_KEY = "tlc_split_first_layer"
EMBEDDING, FINAL_NORM = "model.embed_tokens.weight", "model.norm.weight"
# TLC per-layer field -> parameter of Mamba2Layer
TLC_NAMES = {
    "ln1_weight": "norm1.weight",
    "in_proj_weight": "mixer.in_proj.weight",
    "conv_weight": "mixer.conv_weight",
    "conv_bias": "mixer.conv_bias",
    "A_log": "mixer.A_log",
    "D_param": "mixer.D",
    "dt_bias": "mixer.dt_bias",
    "gated_norm_weight": "mixer.norm.weight",
    "out_proj_weight": "mixer.out_proj.weight",
    "ln2_weight": "norm2.weight",
    "mlp_up_weight": "mlp.up.weight",
    "mlp_down_weight": "mlp.down.weight",
}


def twin_config(profile: str = "a0", **overrides: Any) -> dict[str, Any]:
    """The twin's geometry: a profile (PROFILES) with overrides of TWIN_KEYS; anything else is refused."""
    if profile not in PROFILES:
        raise ValueError(f"unknown twin profile {profile!r}; known: {sorted(PROFILES)}")
    unknown = sorted(set(overrides) - TWIN_KEYS)
    if unknown:
        raise ValueError(f"unknown twin keys {unknown}; known: {sorted(TWIN_KEYS)}")
    return {**PROFILES[profile], **overrides}


def config_from_tlc(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The twin's geometry from a TLC config.json; refuses rows that are not twins (attention layers, untied)."""
    pattern = cfg.get("tlc_layer_pattern", "")
    if set(pattern) != {"M"} or len(pattern) != cfg.get("num_hidden_layers"):
        raise ValueError(f"A0Twin models all-Mamba rows only, not tlc_layer_pattern {pattern!r}")
    if cfg.get("tie_word_embeddings") is not True:
        raise ValueError("A0Twin models tied rows only (tie_word_embeddings = true)")
    missing = sorted(TWIN_KEYS - set(cfg))
    if missing:
        raise ValueError(f"config.json lacks {missing}")
    return {key: cfg[key] for key in TWIN_KEYS}


def geometry(cfg: Mapping[str, Any]) -> Mamba2Geometry:
    return Mamba2Geometry(
        heads=cfg["mamba_num_heads"],
        head_dim=cfg["mamba_head_dim"],
        state=cfg["ssm_state_size"],
        groups=cfg["n_groups"],
        conv_kernel=cfg["conv_kernel"],
        chunk=cfg["chunk_size"],
    )


def tlc_layer_key(layer: int, field: str, split: bool) -> str:
    """The native weight name of `field` of layer `layer` (ref_mamba2.layer_key)."""
    if split:
        return f"first_layer_{field}" if layer == 0 else f"blocks[{layer - 1}].{field}"
    return f"blocks[{layer}].{field}"


class A0Twin(nn.Module):
    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        vocab_size: int | None,
        init_seed: int | None = 0,
        backend: str = "torch",
    ):
        super().__init__()
        unknown = sorted(set(cfg) - TWIN_KEYS)
        if unknown or TWIN_KEYS - set(cfg):
            raise ValueError(f"cfg must hold exactly TWIN_KEYS (use twin_config); unknown {unknown}")
        self.cfg = dict(cfg)
        self.d_model = cfg["hidden_size"]
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, self.d_model) if vocab_size else None
        self.stack = Mamba2Stack(
            self.d_model,
            cfg["num_hidden_layers"],
            geometry(cfg),
            eps=cfg["rms_norm_eps"],
            ffn=cfg["intermediate_size"],
            backend=backend,
        )
        if init_seed is not None:
            self.reset_parameters(init_seed)

    @torch.no_grad()
    def reset_parameters(self, init_seed: int) -> None:
        generator = torch.Generator().manual_seed(int(init_seed))
        if self.embedding is not None:
            value = torch.empty(self.embedding.weight.shape).normal_(0.0, EMBEDDING_STD, generator=generator)
            self.embedding.weight.copy_(value)
        self.stack.reset_parameters(generator)

    # --- the main-network interface (models.lm, models.hnet) ------------------------------------------------------

    def forward_hidden(self, h: torch.Tensor) -> torch.Tensor:
        """Input vectors [B, L, d] -> hidden states after the blocks and the final norm."""
        return self.stack(h)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self._embedding().weight[ids]

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """The tied head: h @ E^T (h in the weight's dtype; autocast decides under it)."""
        weight = self._embedding().weight
        return F.linear(h.to(weight.dtype), weight)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.logits(self.forward_hidden(self.embed(ids)))

    def _embedding(self) -> nn.Embedding:
        if self.embedding is None:
            raise ValueError("this twin is a main network without an embedding (vocab_size=None)")
        return self.embedding

    def non_embedding_parameters(self) -> int:
        return sum(p.numel() for p in self.stack.parameters())

    # --- native weights -------------------------------------------------------------------------------------------

    def tlc_names(self, *, split: bool = False) -> dict[str, str]:
        """native name -> parameter name of this module."""
        names = {FINAL_NORM: "stack.final_norm.weight"}
        if self.embedding is not None:
            names[EMBEDDING] = "embedding.weight"
        for layer in range(len(self.stack.layers)):
            for field, local in TLC_NAMES.items():
                names[tlc_layer_key(layer, field, split)] = f"stack.layers.{layer}.{local}"
        return names

    def load_tlc_weights(self, weights: Mapping[str, torch.Tensor], *, split: bool = False) -> None:
        """Copy native weights (any dtype) into this twin; names and shapes must match exactly."""
        names = self.tlc_names(split=split)
        missing, extra = sorted(set(names) - set(weights)), sorted(set(weights) - set(names))
        if missing or extra:
            raise ValueError(f"native weights: missing {missing[:5]}, unexpected {extra[:5]}")
        own = dict(self.named_parameters())
        with torch.no_grad():
            for native, local in names.items():
                if tuple(weights[native].shape) != tuple(own[local].shape):
                    raise ValueError(
                        f"{native}: shape {list(weights[native].shape)}, expected {list(own[local].shape)}"
                    )
                own[local].copy_(weights[native].to(own[local].dtype))

    def tlc_weights(self, *, split: bool = False) -> dict[str, torch.Tensor]:
        own = dict(self.named_parameters())
        return {native: own[local].detach().clone() for native, local in self.tlc_names(split=split).items()}

    @classmethod
    def from_tlc_model_dir(cls, path: Path, *, dtype: torch.dtype = torch.float32, backend: str = "torch") -> A0Twin:
        """A twin with the weights of a TLC model directory (config.json, model.safetensors), cast to `dtype`."""
        from safetensors.torch import load_file

        path = Path(path)
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        twin = cls(config_from_tlc(cfg), vocab_size=cfg["vocab_size"], init_seed=None, backend=backend).to(dtype)
        twin.load_tlc_weights(load_file(path / "model.safetensors"), split=bool(cfg.get(SPLIT_KEY, False)))
        return twin
