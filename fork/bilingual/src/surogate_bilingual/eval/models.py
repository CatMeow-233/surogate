"""The model an evaluation arm runs, and the arm presets of MS §7 (step I6).

`load_model(path)` accepts what the training and assembly steps produce:

- a seed pack directory (`bilingual-seed/1`): the seed model, one expert named "seed" (B0, B0');
- an assembly manifest (`bilingual-assembly/1`): every slot of the manifest, re-verified by hash, built by
  `assembly.assemble.build_model` (the same construction `verify` checks). A one-slot assembly is a standalone expert
  (B1, B2, B12a); `fusion` may override the manifest's fusion for dual rows, so B8 (`pre`) and B9-II (`post`)
  evaluate the same checkpoints. T1' evaluates one assembly per direction whose target slot holds the T1' copy (the
  copy keeps its original's name, so it takes the original's slot), with `eval-tasks --direction`;
- with `merge`, an assembly whose experts are merged into one expert "merged" before evaluation (B3a average,
  B3b task arithmetic, B3c TIES; B3-II is the family-II average). Merges act on expert FFN states only, in float64.
  The coefficient comes from the one grid of DEFAULTS["eval"]["merge"] (MS §9.6).

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

from ..assembly.assemble import build_model, model_from_seed, read_assembly
from ..assembly.packs import TOKENIZER, SeedPack, read_pack, seed_backbone_for, seed_expert_init
from ..config import DEFAULTS
from ..data.tokenizer import load_tokenizer, tokenizer_sha256
from ..model.lm import RoutedLM
from ..ops import merge as merges
from ..registry.matrix import MISSING_PROTOCOL, ROW_IDS
from .items import Routing

MERGES = ("average", "task_arithmetic", "ties")
MERGE_DEFAULTS = DEFAULTS["eval"]["merge"]

# Arm presets: the routing an MS §7 row uses by default. Every flag of eval-tasks overrides them and the prediction
# records keep what actually ran. Single-expert arms (B0, B1, B2, B12a, merges) need no routing policy. The answer
# route of the T arms applies to translation items only (E4): on E1/E2/E3 a T arm routes exactly as T0 (MS §7.3,
# §8.4). T1' needs `--direction` (one assembly per direction). Matrix rows without a preset are in
# registry.matrix.MISSING_PROTOCOL and are refused.
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
    "T1'": {"route": "R_out", "needs_direction": True},
    "T2": {"route": "R_out", "answer_route": "bridge"},
    "T3'": {"route": "R_out", "answer_route": "t3"}, "T3''": {"route": "R_out", "answer_route": "t3"},
}  # fmt: skip
if set(ARMS) & set(MISSING_PROTOCOL) or set(ARMS) | set(MISSING_PROTOCOL) != set(ROW_IDS):
    raise RuntimeError("eval.models.ARMS and registry.matrix.MISSING_PROTOCOL must cover the MS §7 rows exactly once")


class ArmError(ValueError):
    pass


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
        experts = {name: pack.weights for name, pack in assembly.slots}
        source = {
            "kind": "assembly",
            "path": str(path),
            "seed_fingerprint": seed.seed_fingerprint,
            "experts": {name: pack.expert_hash for name, pack in assembly.slots},
        }
        if merge:
            state = merged_state(seed, cfg, experts, merge)
            model = RoutedLM(cfg, ["merged"], fusion or assembly.fusion, gqa)
            model.load_backbone_state(seed_backbone_for(seed, cfg))
            model.load_expert_state("merged", state)
            source["merge"] = dict(merge)
        else:
            model = build_model(assembly, gqa, fusion=fusion)
        source["fusion"] = model.fusion
        source["fusion_override"] = fusion is not None and fusion != assembly.fusion
    tok_path = seed.path / TOKENIZER
    return Evaluated(model.to(device).eval(), load_tokenizer(tok_path), tokenizer_sha256(tok_path), source)


def merged_state(
    seed: SeedPack, cfg, experts: Mapping[str, Mapping[str, torch.Tensor]], merge: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """The merged expert state. Task arithmetic and TIES need `lam` (chosen on dev from the grid, MS §9.6); TIES
    takes its density from `merge["density"]`, default DEFAULTS["eval"]["merge"]["density"]."""
    method = merge.get("method")
    names = list(merge.get("experts") or ["zh", "en"])
    missing = [n for n in names if n not in experts]
    if missing:
        raise ValueError(f"merge {method}: the assembly has no experts {missing}")
    states = [experts[n] for n in names]
    if method == "average":
        return merges.weight_average(states, [1.0 / len(states)] * len(states))
    if method not in MERGES:
        raise ValueError(f"unknown merge {method!r}; known: {MERGES}")
    if merge.get("lam") is None:
        raise ValueError(f"merge {method} needs lam, a point of the grid {MERGE_DEFAULTS['lam_grid']} chosen on dev")
    base = seed_expert_init(seed, cfg)
    lam = float(merge["lam"])
    if method == "task_arithmetic":
        return merges.task_arithmetic(base, states, lam)
    return merges.ties(base, states, float(merge.get("density", MERGE_DEFAULTS["density"])), lam)


def arm_routing(arm: str | None, **overrides) -> tuple[Routing, dict[str, Any]]:
    """(routing, preset) of an arm with explicit overrides (None values keep the preset).

    An MS §7 row ID takes its preset; a row without an evaluation protocol is refused, naming what is missing. Any
    other label must give its routing explicitly (`route`), so a typo of a row ID never falls back to plain R_out."""
    label = arm or ""
    if label in MISSING_PROTOCOL:
        raise ArmError(f"arm {label!r} has no evaluation protocol in P0: {MISSING_PROTOCOL[label]}")
    if label in ARMS:
        preset = dict(ARMS[label])
    elif overrides.get("route") is None:
        raise ArmError(
            f"arm {label!r} is not an MS §7 row id with a preset ({', '.join(ARMS)}); a free label needs its "
            "routing given explicitly (--route, and --gate / --answer-route / --fusion / --merge as needed)"
        )
    else:
        preset = {}
    values = {
        "route": preset.get("route", "R_out"),
        "gate": preset.get("gate", "one_hot"),
        "answer_route": preset.get("answer_route"),
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return Routing(**values), preset
