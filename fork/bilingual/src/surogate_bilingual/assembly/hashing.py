"""Content hashes of tensor states and the seed fingerprint (docs/spec.md, "内容哈希与种子指纹"; IMPL-14).

    content_hash(state) = sha256 over sorted(state) of
        name.utf8 + 0x00 + str(dtype) + 0x00 + json(list(shape)) + 0x00 + raw little-endian bytes

The hash depends only on tensor names, dtypes, shapes and values: never on containers, step counters or file
layout, so it survives a safetensors round trip. A big-endian host raises rather than hash different bytes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping

import torch

from ..canonical import sha256_json

SEED_FINGERPRINT_FORMAT = "bilingual-seed-fingerprint/1"


def tensor_bytes(t: torch.Tensor) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("content hashes are defined on little-endian hosts only")
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def content_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        t = state[name]
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"content_hash covers tensors only; {name!r} is {type(t).__name__}")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(t.dtype).encode() + b"\0")
        digest.update(json.dumps(list(t.shape)).encode() + b"\0")
        digest.update(tensor_bytes(t))
    return digest.hexdigest()


def tensor_hashes(state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    """One content hash per tensor (each over the one-entry state {name: tensor})."""
    return {name: content_hash({name: state[name]}) for name in sorted(state)}


def seed_fingerprint(config_hash: str, backbone_hash: str, seed_ffn_hash: str, tokenizer_sha256: str) -> str:
    return sha256_json(
        {
            "format": SEED_FINGERPRINT_FORMAT,
            "config_hash": config_hash,
            "backbone": backbone_hash,
            "seed_ffn": seed_ffn_hash,
            "tokenizer_sha256": tokenizer_sha256,
        }
    )
