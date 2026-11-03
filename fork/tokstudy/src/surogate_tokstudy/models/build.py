"""One entry point from an arm to its model: every kind of both tables is built here, on the one backend.

An arm's `model` table (arms.py) is read as

    model = {profile = "a0", overrides = {...}, hnet = {d0 = 256, encoder_layers = 2, decoder_layers = 2, ratio = 3.8}}

- `profile` and `overrides` describe the main network of the arm's table: for `twin_a0` a twin profile ("a0",
  "tiny") and TLC config.json keys (twin_a0.TWIN_KEYS); for `bilingual_trunk` a bilingual size profile ("pilot",
  "tiny", ...) and ModelConfig fields. Defaults: "a0" and "pilot", no overrides.
- `hnet` (H-Net arms only, required there) holds the fields of hnet.HNetConfig.

| kind | model | ids |
|---|---|---|
| subword | SubwordLM(main with the codec's vocabulary) | the tokenizer's |
| bytes | ByteLM(main with the byte vocabulary, 261 or the codec's pad_to) | 5 + byte, specials 0-4 |
| hnet | HNetLM(main without a vocabulary, HNetConfig) | 5 + byte, specials 0-4 |

The main network is initialised from `init_seed`, the H-Net's outer stage from a seed derived from it, so the two
never share a random stream. `flop_spec(model)` describes a built model for the pure-Python FLOP formulas
(surogate_tokstudy.flops), and `positional_limit(arm)` is the longest row its main network accepts (a Transformer
trunk's max_seq; None for Mamba).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from torch import nn

from ..arms import ArmSpec
from ..codecs import BYTE_VOCAB
from .hnet import HNetConfig, HNetLM
from .lm import ByteLM, SubwordLM

MODEL_KEYS = {"profile", "overrides", "hnet"}
DEFAULT_PROFILES = {"twin_a0": "a0", "bilingual_trunk": "pilot"}


def derived_seed(seed: int, label: str) -> int:
    """A seed for one part of a model: 63 bits of sha256("<seed>/<label>")."""
    return int.from_bytes(hashlib.sha256(f"{seed}/{label}".encode()).digest()[:8], "little") >> 1


def build_main(
    main: str,
    profile: str | None,
    overrides: Mapping[str, Any],
    *,
    vocab_size: int | None,
    init_seed: int,
    backend: str = "torch",
) -> nn.Module:
    """The table's main network; vocab_size None builds it without a token embedding (the H-Net main network).
    `backend` is the scan kernels' (kernels.py); the bilingual trunk has none."""
    if main == "twin_a0":
        from .twin_a0 import A0Twin, twin_config

        return A0Twin(twin_config(profile or DEFAULT_PROFILES[main], **overrides), vocab_size=vocab_size,
                      init_seed=init_seed, backend=backend)  # fmt: skip
    if main == "bilingual_trunk":
        from .trunk_bilingual import BilingualTrunk

        return BilingualTrunk(profile or DEFAULT_PROFILES[main], overrides, vocab_size=vocab_size, init_seed=init_seed)
    raise ValueError(f"unknown main network {main!r}; known: {sorted(DEFAULT_PROFILES)}")


def build_model(arm: ArmSpec, *, vocab_size: int | None = None, init_seed: int, backend: str = "torch") -> nn.Module:
    """The model of `arm`; `vocab_size` is the codec's (required for subword arms; bytes default to 261); `backend`
    the scan kernels' (an H-Net arm's model.hnet.backend, when given, must agree)."""
    model = dict(arm.model)
    unknown = sorted(set(model) - MODEL_KEYS)
    if unknown:
        raise ValueError(f"arm {arm.id}: unknown model keys {unknown}; known: {sorted(MODEL_KEYS)}")
    if ("hnet" in model) != (arm.kind == "hnet"):
        raise ValueError(f"arm {arm.id}: a {arm.kind} arm {'needs' if arm.kind == 'hnet' else 'has no'} model.hnet")
    profile, overrides = model.get("profile"), dict(model.get("overrides", {}))
    if arm.kind == "subword":
        if vocab_size is None:
            raise ValueError(f"arm {arm.id}: a subword arm's model needs the codec's vocab_size")
        main = build_main(arm.main, profile, overrides, vocab_size=vocab_size, init_seed=init_seed, backend=backend)
        return SubwordLM(main)
    if arm.kind == "bytes":
        main = build_main(arm.main, profile, overrides, vocab_size=vocab_size or BYTE_VOCAB, init_seed=init_seed,
                          backend=backend)  # fmt: skip
        return ByteLM(main)
    hnet = dict(model["hnet"])
    if hnet.setdefault("backend", backend) != backend:
        raise ValueError(f"arm {arm.id}: model.hnet.backend {hnet['backend']!r} is not the run's {backend!r}")
    if vocab_size is not None and hnet.setdefault("vocab_size", vocab_size) != vocab_size:
        raise ValueError(f"arm {arm.id}: model.hnet.vocab_size {hnet['vocab_size']} is not the codec's {vocab_size}")
    try:
        cfg = HNetConfig(**hnet)
    except TypeError as err:
        raise ValueError(f"arm {arm.id}: model.hnet: {err}") from None
    main = build_main(arm.main, profile, overrides, vocab_size=None, init_seed=init_seed, backend=backend)
    return HNetLM(main, cfg, init_seed=derived_seed(init_seed, "hnet-outer"))


def _main_spec(main: nn.Module) -> dict[str, Any]:
    from .trunk_bilingual import BilingualTrunk
    from .twin_a0 import A0Twin

    if isinstance(main, A0Twin):
        c = main.cfg
        return {"type": "mamba", "d": c["hidden_size"], "layers": c["num_hidden_layers"], "ffn": c["intermediate_size"],
                "heads": c["mamba_num_heads"], "head_dim": c["mamba_head_dim"], "state": c["ssm_state_size"],
                "groups": c["n_groups"], "chunk": c["chunk_size"], "kernel": c["conv_kernel"]}  # fmt: skip
    if isinstance(main, BilingualTrunk):
        c = main.cfg
        if c.family != "swiglu":
            raise ValueError(f"the FLOP formulas know the dense SwiGLU trunk, not family {c.family!r}")
        return {"type": "transformer", "d": c.d_model, "layers": c.n_layers, "q_width": c.n_heads * c.head_dim,
                "kv_width": c.n_kv_heads * c.head_dim, "ffn": c.ffn_width}  # fmt: skip
    raise ValueError(f"no FLOP spec for a main network of type {type(main).__name__}")


def _outer_spec(stack) -> dict[str, Any]:
    g = stack.geometry
    return {"type": "mamba", "d": stack.d_model, "layers": len(stack.layers), "ffn": stack.ffn, "heads": g.heads,
            "head_dim": g.head_dim, "state": g.state, "groups": g.groups, "chunk": g.chunk, "kernel": g.conv_kernel}  # fmt: skip


def flop_spec(model: nn.Module) -> dict[str, Any]:
    """The flops.py spec dict of a model built by build_model."""
    if isinstance(model, HNetLM):
        return {"kind": "hnet", "main": _main_spec(model.main), "vocab": model.cfg.vocab_size, "d0": model.cfg.d0,
                "encoder": _outer_spec(model.encoder), "decoder": _outer_spec(model.decoder)}  # fmt: skip
    if isinstance(model, (SubwordLM, ByteLM)):
        return {"kind": model.kind, "main": _main_spec(model.main), "vocab": model.vocab_size}
    raise ValueError(f"no FLOP spec for a model of type {type(model).__name__}")


def positional_limit(arm: ArmSpec) -> int | None:
    """The most positions one row may hold for this arm: the bilingual trunk's max_seq (for an H-Net arm it bounds
    the chunks, and a row may realize one chunk per byte, so it bounds the bytes too); None for the Mamba twin."""
    if arm.main == "twin_a0":
        return None
    if arm.main == "bilingual_trunk":
        from .trunk_bilingual import trunk_config

        model = dict(arm.model)
        return trunk_config(model.get("profile") or DEFAULT_PROFILES[arm.main], model.get("overrides", {})).max_seq
    raise ValueError(f"unknown main network {arm.main!r}")
