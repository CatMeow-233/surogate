"""Assembly manifests (`bilingual-assembly/1`), loading, and forced-route fidelity (MS §8.3; IMPL-04, MA-14).

An assembly copies no weights: the manifest names the seed pack and one expert pack per slot, with their hashes.
Paths are stored relative to the manifest's directory. `assemble` refuses, each with its own AssemblyError subclass:

- a slot name that does not match [a-z][a-z0-9_]* (InvalidSlotNameError) or that repeats (DuplicateSlotError);
- one expert in two slots, i.e. a repeated expert_hash, even under another name (PackReusedError);
- a pack whose own name differs from its slot (SlotNameMismatchError);
- a pack of another family (FamilyMismatchError), from another seed (SeedFingerprintMismatchError), with another
  config than the other experts (ConfigHashMismatchError), or whose config is not the seed's (ConfigMismatchError);
- a fusion that does not belong to the family (FusionError);
- and everything `read_pack` refuses: tampered files (HashMismatchError), missing or unexpected weight keys, an
  unknown format.

Slot names are free identifiers ("t3" for an ordinary third expert is accepted). `load_assembly` re-reads and
re-verifies every pack and loads the backbone and the experts with exact key sets.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..canonical import atomic_write_json, check_format
from ..config import DEFAULT_FUSION, FUSIONS, NAME_PATTERN, ModelConfig, config_hash
from ..model.lm import RoutedLM
from ..routing.plan import RoutePlan
from .packs import (
    AssemblyError,
    ExpertPack,
    FormatError,
    SeedPack,
    check_extends,
    code_identity,
    probe_batch,
    read_pack,
    seed_backbone_for,
    seed_expert_init,
)

MANIFEST_FORMAT = "bilingual-assembly/1"
FIDELITY_FORMAT = "bilingual-fidelity/1"


class InvalidSlotNameError(AssemblyError):
    pass


class DuplicateSlotError(AssemblyError):
    pass


class PackReusedError(AssemblyError):
    pass


class SlotNameMismatchError(AssemblyError):
    pass


class FamilyMismatchError(AssemblyError):
    pass


class SeedFingerprintMismatchError(AssemblyError):
    pass


class ConfigHashMismatchError(AssemblyError):
    pass


class FusionError(AssemblyError):
    pass


def model_from_seed(
    seed_pack: Path | SeedPack,
    names: Sequence[str],
    cfg: ModelConfig | None = None,
    fusion: str | None = None,
    gqa: str = "sdpa_gqa",
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> RoutedLM:
    """A model forked from a seed: the seed's backbone (plus, for LP-k, its FFN on the non-expert layers) and every
    named expert a copy of the seed FFN. `cfg` defaults to the seed's config; it may differ only in expert_layers."""
    seed = seed_pack if isinstance(seed_pack, SeedPack) else read_pack(Path(seed_pack))
    if not isinstance(seed, SeedPack):
        raise FormatError(f"{seed_pack} is not a seed pack")
    cfg = cfg or seed.config
    model = RoutedLM(cfg, names, fusion, gqa)
    model.load_backbone_state(seed_backbone_for(seed, cfg))
    for name in names:
        model.load_expert_state(name, seed_expert_init(seed, cfg))
    return model.to(device=device or "cpu", dtype=dtype)


def _slot_list(slots: Mapping[str, Path] | Sequence[tuple[str, Path]]) -> list[tuple[str, Path]]:
    items = list(slots.items()) if isinstance(slots, Mapping) else [tuple(s) for s in slots]
    names = [name for name, _ in items]
    bad = [n for n in names if not isinstance(n, str) or not NAME_PATTERN.fullmatch(n)]
    if bad:
        raise InvalidSlotNameError(f"slot names must match {NAME_PATTERN.pattern}: {bad}")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise DuplicateSlotError(f"slot names repeat: {repeated}")
    if not items:
        raise AssemblyError("an assembly needs at least one slot")
    return [(name, Path(path)) for name, path in items]


def _check(seed: SeedPack, packs: Sequence[tuple[str, ExpertPack]], fusion: str | None) -> tuple[ModelConfig, str]:
    hashes = [pack.expert_hash for _, pack in packs]
    reused = sorted({h for h in hashes if hashes.count(h) > 1})
    if reused:
        slots = [name for name, pack in packs if pack.expert_hash in reused]
        raise PackReusedError(f"slots {slots} hold the same expert (expert_hash {reused[0][:12]}...)")
    for name, pack in packs:
        if pack.name != name:
            raise SlotNameMismatchError(f"slot {name!r} holds the pack of expert {pack.name!r} ({pack.path})")
    for name, pack in packs:
        if pack.config.family != seed.config.family or pack.meta.get("family") != seed.config.family:
            raise FamilyMismatchError(
                f"slot {name!r} is family {pack.config.family!r}, the seed {seed.config.family!r}"
            )
    for name, pack in packs:
        if pack.seed_fingerprint != seed.seed_fingerprint:
            raise SeedFingerprintMismatchError(f"slot {name!r} was forked from another seed than {seed.path}")
    first = packs[0][1]
    for name, pack in packs:
        if pack.meta["config_hash"] != first.meta["config_hash"]:
            raise ConfigHashMismatchError(f"slot {name!r} has another config than slot {packs[0][0]!r}")
    check_extends(seed, first.config)
    fusion = fusion or DEFAULT_FUSION[first.config.family]
    if fusion not in FUSIONS[first.config.family]:
        raise FusionError(f"fusion {fusion!r} does not belong to family {first.config.family!r}")
    return first.config, fusion


def _read_expert(path: Path) -> ExpertPack:
    pack = read_pack(path)
    if not isinstance(pack, ExpertPack):
        raise FormatError(f"{path} is not an expert pack")
    return pack


def assemble(
    seed_pack: Path,
    slots: Mapping[str, Path] | Sequence[tuple[str, Path]],
    fusion: str | None,
    out: Path,
    *,
    overwrite: bool = False,
) -> Path:
    items = _slot_list(slots)
    seed = read_pack(Path(seed_pack))
    if not isinstance(seed, SeedPack):
        raise FormatError(f"{seed_pack} is not a seed pack")
    packs = [(name, _read_expert(path)) for name, path in items]
    cfg, fusion = _check(seed, packs, fusion)
    out = Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists")
    base = out.resolve().parent
    commit, dirty = code_identity()
    manifest = {
        "format": MANIFEST_FORMAT,
        "seed": {"path": os.path.relpath(Path(seed_pack).resolve(), base), "seed_fingerprint": seed.seed_fingerprint},
        "slots": [
            {"name": name, "path": os.path.relpath(pack.path.resolve(), base), "expert_hash": pack.expert_hash}
            for name, pack in packs
        ],
        "fusion": fusion,
        "config_hash": config_hash(cfg),
        "created": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_commit": commit,
        "code_dirty": dirty,
    }
    return atomic_write_json(out, manifest)


@dataclass(frozen=True)
class Assembly:
    manifest: dict[str, Any]
    seed: SeedPack
    slots: tuple[tuple[str, ExpertPack], ...]
    config: ModelConfig
    fusion: str


def read_assembly(manifest: Path) -> Assembly:
    """Read a manifest and re-verify every pack it names against it."""
    manifest = Path(manifest)
    try:
        doc = json.loads(manifest.read_text("utf-8"))
        check_format(doc, "assembly")
    except (OSError, ValueError) as exc:
        raise FormatError(f"{manifest}: {exc}") from exc
    base = manifest.resolve().parent
    items = _slot_list([(slot["name"], base / slot["path"]) for slot in doc["slots"]])
    seed = read_pack(base / doc["seed"]["path"])
    if not isinstance(seed, SeedPack) or seed.seed_fingerprint != doc["seed"]["seed_fingerprint"]:
        raise SeedFingerprintMismatchError(f"{manifest}: the seed pack is not the one the manifest recorded")
    packs = [(name, _read_expert(path)) for name, path in items]
    for slot, (name, pack) in zip(doc["slots"], packs, strict=True):
        if pack.expert_hash != slot["expert_hash"]:
            raise PackReusedError(f"{manifest}: slot {name!r} now holds another expert than the one recorded")
    cfg, fusion = _check(seed, packs, doc["fusion"])
    if config_hash(cfg) != doc["config_hash"]:
        raise ConfigHashMismatchError(f"{manifest}: config_hash differs from the packs' config")
    return Assembly(doc, seed, tuple(packs), cfg, fusion)


def build_model(assembly: Assembly, gqa: str = "sdpa_gqa") -> RoutedLM:
    names = [name for name, _ in assembly.slots]
    model = RoutedLM(assembly.config, names, assembly.fusion, gqa)
    model.load_backbone_state(seed_backbone_for(assembly.seed, assembly.config))
    for name, pack in assembly.slots:
        model.load_expert_state(name, pack.weights)
    return model


def load_assembly(
    manifest: Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32, gqa: str = "sdpa_gqa"
) -> RoutedLM:
    return build_model(read_assembly(manifest), gqa).to(device=device, dtype=dtype)


# --- fidelity ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerances:
    """Forced-route fidelity against the recorded fp32 CPU probe (MS §9.4)."""

    max_abs: float = 1e-5  # fp32 CPU; 0.0 expected
    min_argmax_agreement: float | None = None

    @staticmethod
    def bf16_cuda() -> Tolerances:
        """[建议] the bf16 CUDA tolerance of MS §9.4, to be locked before G0 runs."""
        return Tolerances(max_abs=0.05, min_argmax_agreement=0.995)


@dataclass(frozen=True)
class SlotFidelity:
    name: str
    expert_hash: str
    max_abs: float
    argmax_agreement: float
    passed: bool


@dataclass(frozen=True)
class FidelityReport:
    manifest: str
    device: str
    dtype: str
    autocast: str | None
    tolerances: Tolerances
    slots: tuple[SlotFidelity, ...]

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.slots)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["slots"] = [asdict(s) for s in self.slots]
        out.update(format=FIDELITY_FORMAT, passed=self.passed)
        return out


def verify_fidelity(
    manifest: Path,
    tol: Tolerances | None = None,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    autocast: torch.dtype | None = None,
    gqa: str = "sdpa_gqa",
) -> FidelityReport:
    """Replay every expert pack's probe under the forced route of the assembled model."""
    tol = tol or Tolerances()
    assembly = read_assembly(manifest)
    model = build_model(assembly, gqa).to(device=device, dtype=dtype).eval()
    names = model.expert_names
    results = []
    for name, pack in assembly.slots:
        probe = pack.meta["probe"]
        tokens = probe_batch(assembly.config, probe["seed"]).to(device)
        plan = RoutePlan.forced(name, names, *tokens.shape)
        device_type = torch.device(device).type
        with (
            torch.no_grad(),
            torch.autocast(device_type, dtype=autocast or torch.bfloat16, enabled=autocast is not None),
        ):
            logits = model(tokens, plan)[:, probe["positions"]].float().cpu()
        max_abs = float((logits - pack.probe).abs().max())
        agreement = float((logits.argmax(-1) == pack.probe.argmax(-1)).double().mean())
        passed = max_abs <= tol.max_abs and (tol.min_argmax_agreement is None or agreement >= tol.min_argmax_agreement)
        results.append(SlotFidelity(name, pack.expert_hash, max_abs, agreement, passed))
    return FidelityReport(
        str(manifest), str(device), str(dtype).removeprefix("torch."), None if autocast is None else str(autocast),
        tol, tuple(results),
    )  # fmt: skip


# --- command line --------------------------------------------------------------------------------------------------


def _slot_arg(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"--slot takes NAME=PACK, got {text!r}")
    return name, Path(path)


def main(argv: Sequence[str]) -> int:
    """`assemble --seed PACK --slot NAME=PACK ... --out assembly.json [--fusion F] [--dry-run]`"""
    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual assemble", description="write an assembly manifest")
    ap.add_argument("--seed", type=Path, required=True)
    ap.add_argument("--slot", type=_slot_arg, action="append", required=True, help="NAME=PACK, in slot order")
    ap.add_argument("--fusion", choices=[f for fs in FUSIONS.values() for f in fs])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="check every pack; write the manifest to a temp dir only")
    args = ap.parse_args(list(argv))
    try:
        if args.dry_run:
            with tempfile.TemporaryDirectory() as tmp:
                path = assemble(args.seed, args.slot, args.fusion, Path(tmp) / "assembly.json")
                print(path.read_text("utf-8"), end="")
        else:
            print(assemble(args.seed, args.slot, args.fusion, args.out))
    except (AssemblyError, FileExistsError) as exc:
        print(f"assemble: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


DTYPES = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16}


def verify_main(argv: Sequence[str]) -> int:
    """`verify MANIFEST [--device D] [--dtype T] [--autocast bf16] [--max-abs X] [--out report.json] [--dry-run]`"""
    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual verify", description="forced-route fidelity")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    ap.add_argument("--autocast", choices=["none", "bf16"], default="none")
    ap.add_argument("--max-abs", type=float, help="default 1e-5, or 0.05 with --autocast bf16 (MS §9.4)")
    ap.add_argument("--min-argmax", type=float, help="default none, or 0.995 with --autocast bf16")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="re-verify the manifest and its packs; run nothing")
    args = ap.parse_args(list(argv))
    try:
        if args.dry_run:
            read_assembly(args.manifest)
            print(f"verify: {args.manifest}: manifest and packs verified (dry run, no forward)")
            return 0
        base = Tolerances.bf16_cuda() if args.autocast == "bf16" else Tolerances()
        tol = Tolerances(
            base.max_abs if args.max_abs is None else args.max_abs,
            base.min_argmax_agreement if args.min_argmax is None else args.min_argmax,
        )
        autocast = torch.bfloat16 if args.autocast == "bf16" else None
        report = verify_fidelity(args.manifest, tol, device=args.device, dtype=DTYPES[args.dtype], autocast=autocast)
    except AssemblyError as exc:
        print(f"verify: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    doc = report.to_dict()
    if args.out:
        atomic_write_json(args.out, doc)
    print(json.dumps(doc, indent=1, sort_keys=True))
    return 0 if report.passed else 1
