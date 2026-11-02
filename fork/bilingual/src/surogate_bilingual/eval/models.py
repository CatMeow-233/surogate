"""The model an evaluation arm runs, and the arm presets of MS §7 (step I6).

`load_model(path)` accepts what the training and assembly steps produce:

- a seed pack directory (`bilingual-seed/1`): the seed model, one expert named "seed" (B0, B0');
- an assembly manifest (`bilingual-assembly/1`): every slot of the manifest, re-verified by hash. A one-slot
  assembly is a standalone expert (B1, B2, B12a); `fusion` may override the manifest's fusion for dual rows, so B8
  (`pre`) and B9-II (`post`) evaluate the same checkpoints;
- with `merge`, an assembly whose experts are merged into one expert "merged" before evaluation (B3a average,
  B3b task arithmetic, B3c TIES; B3-II is the family-II average). Merges act on expert FFN states only, in float64.

The tokenizer is always the seed pack's copy, so a model is never scored with another tokenizer than it was
trained with.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..assembly.assemble import model_from_seed, read_assembly
from ..assembly.packs import TOKENIZER, SeedPack, read_pack, seed_backbone_for, seed_expert_init
from ..config import FUSIONS
from ..data.tokenizer import load_tokenizer, tokenizer_sha256
from ..model.lm import RoutedLM
from ..ops import merge as merges
from .items import Routing

MERGES = ("average", "task_arithmetic", "ties")

# Arm presets: the routing an MS §7 row uses by default. Every flag of eval-tasks overrides them and the prediction
# records keep what actually ran. Single-expert arms (B0, B1, B2, B12a, merges) need no routing policy.
ARMS: dict[str, dict[str, Any]] = {
    "B0": {}, "B0'": {}, "B1": {}, "B1'": {}, "B2": {}, "B12a": {},
    "B3a": {"merge": "average"}, "B3b": {"merge": "task_arithmetic"}, "B3c": {"merge": "ties"},
    "B3-II": {"merge": "average"},
    "B4": {"route": "R_out"}, "B4-src": {"route": "R_src"}, "B4-wrong": {"route": "R_wrong"},
    "B4-rand": {"route": "R_random"}, "LP-4": {"route": "R_out"}, "T0": {"route": "R_out"},
    "B6": {"route": "R_out", "gate": "dual_const", "fusion": "output"},
    "B7": {"route": "R_out", "gate": "dual_const", "fusion": "concat"},
    "B4-II": {"route": "R_out"},
    "B9-II": {"route": "R_out", "gate": "dual_const", "fusion": "post"},
    "B8": {"route": "R_out", "gate": "dual_const", "fusion": "pre"},
    "T2": {"route": "R_out", "answer_route": "bridge"},
}  # fmt: skip


@dataclass
class Evaluated:
    model: RoutedLM
    tokenizer: Any
    tokenizer_sha256: str
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        return self.model.expert_names


def _is_seed_pack(path: Path) -> bool:
    meta = path / "pack.json"
    return meta.is_file() and json.loads(meta.read_text("utf-8")).get("format") == "bilingual-seed/1"


def load_model(
    path: Path,
    *,
    fusion: str | None = None,
    merge: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    gqa: str = "sdpa_gqa",
) -> Evaluated:
    path = Path(path)
    if path.is_dir():
        if not _is_seed_pack(path):
            raise ValueError(f"{path} is not a seed pack; evaluate an expert through an assembly manifest")
        if merge:
            raise ValueError("a merge needs an assembly of the experts to merge")
        seed = read_pack(path)
        model = model_from_seed(seed, ["seed"], gqa=gqa)
        source = {"kind": "seed", "path": str(path), "seed_fingerprint": seed.seed_fingerprint}
    else:
        assembly = read_assembly(path)
        seed = assembly.seed
        cfg = assembly.config
        chosen = fusion or assembly.fusion
        if chosen not in FUSIONS[cfg.family]:
            raise ValueError(f"fusion {chosen!r} does not belong to family {cfg.family!r}")
        experts = {name: pack.weights for name, pack in assembly.slots}
        source = {"kind": "assembly", "path": str(path), "seed_fingerprint": seed.seed_fingerprint,
                  "experts": {name: pack.expert_hash for name, pack in assembly.slots}, "fusion": chosen,
                  "fusion_override": fusion is not None and fusion != assembly.fusion}  # fmt: skip
        if merge:
            state = merged_state(seed, cfg, experts, merge)
            model = RoutedLM(cfg, ["merged"], chosen, gqa)
            model.load_backbone_state(seed_backbone_for(seed, cfg))
            model.load_expert_state("merged", state)
            source["merge"] = dict(merge)
        else:
            model = RoutedLM(cfg, list(experts), chosen, gqa)
            model.load_backbone_state(seed_backbone_for(seed, cfg))
            for name, weights in experts.items():
                model.load_expert_state(name, weights)
    tok_path = seed.path / TOKENIZER
    return Evaluated(model.to(device).eval(), load_tokenizer(tok_path), tokenizer_sha256(tok_path), source)


def merged_state(seed: SeedPack, cfg, experts: Mapping[str, Mapping[str, torch.Tensor]], merge: Mapping[str, Any]
                 ) -> dict[str, torch.Tensor]:  # fmt: skip
    method = merge.get("method")
    names = list(merge.get("experts") or ["zh", "en"])
    missing = [n for n in names if n not in experts]
    if missing:
        raise ValueError(f"merge {method}: the assembly has no experts {missing}")
    states = [experts[n] for n in names]
    if method == "average":
        return merges.weight_average(states, [1.0 / len(states)] * len(states))
    base = seed_expert_init(seed, cfg)
    lam = float(merge.get("lam", 1.0))
    if method == "task_arithmetic":
        return merges.task_arithmetic(base, states, lam)
    if method == "ties":
        return merges.ties(base, states, float(merge.get("density", 0.2)), lam)
    raise ValueError(f"unknown merge {method!r}; known: {MERGES}")


def arm_routing(arm: str | None, **overrides) -> tuple[Routing, dict[str, Any]]:
    """(routing, preset) of an arm with explicit overrides (None values keep the preset)."""
    preset = dict(ARMS.get(arm or "", {}))
    values = {"route": preset.get("route", "R_out"), "gate": preset.get("gate", "one_hot"),
              "answer_route": preset.get("answer_route")}  # fmt: skip
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return Routing(**values), preset
