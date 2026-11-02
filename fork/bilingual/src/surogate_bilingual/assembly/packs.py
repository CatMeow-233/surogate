"""Seed and expert packs (`bilingual-seed/1`, `bilingual-expert/1`; docs/spec.md, "pack").

A pack is an immutable directory written atomically:

    <pack>/pack.json             sidecar: config, hashes, training metadata, probe description
           weights.safetensors   seed: backbone.* + ffn.* (the seed FFN, every layer); expert: ffn.* (expert layers)
           probe.safetensors     "logits" [2, P, V] fp32 of the standalone model at the probe positions
           tokenizer.json        seed only: a copy, its sha256 recorded

The probe batch is drawn from `torch.Generator().manual_seed(probe.seed)`, shape [2, max_seq], uniform over the
non-special ids. The probe forward runs the standalone model (the seed itself, or the seed backbone plus this one
expert) on the CPU in fp32, whatever device the pack was trained on, so a CPU fp32 replay expects a difference of
exactly 0.0 (MS §9.4).

`read_pack` checks, in this order: the format, the config hash, every recorded file hash, the weight key set, then
the content hashes (and the seed fingerprint). Every failure is a distinct subclass of `AssemblyError`.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from ..canonical import FORMAT_PATTERN, atomic_directory, check_format, sha256_file
from ..config import N_SPECIAL, NAME_PATTERN, ModelConfig, config_hash
from ..model.backbone import backbone_keys, ffn_param_names
from ..model.bank import expert_keys
from ..registry.env import code_identity
from .hashing import content_hash, seed_fingerprint

SIDECAR = "pack.json"
WEIGHTS = "weights.safetensors"
PROBE = "probe.safetensors"
TOKENIZER = "tokenizer.json"
PROBE_SEED = 0
PROBE_BATCH = 2


class AssemblyError(ValueError):
    """Base of every pack and assembly failure."""


class FormatError(AssemblyError):
    """Unknown format or major version, or a malformed sidecar."""


class HashMismatchError(AssemblyError):
    """A recorded file hash, content hash, config hash or seed fingerprint does not match what is on disk."""


class MissingKeyError(AssemblyError):
    """The weights file lacks keys the pack's config requires."""


class UnexpectedKeyError(AssemblyError):
    """The weights file holds keys the pack's config does not define."""


class ConfigMismatchError(AssemblyError):
    """An expert's config does not extend its seed's config."""


class BackboneMismatchError(AssemblyError):
    """The model's frozen parts differ from the seed they were forked from."""


@dataclass(frozen=True)
class SeedPack:
    path: Path
    meta: dict[str, Any]
    config: ModelConfig
    weights: dict[str, torch.Tensor]
    probe: torch.Tensor

    @property
    def seed_fingerprint(self) -> str:
        return self.meta["seed_fingerprint"]

    def backbone_state(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.weights.items() if k.startswith("backbone.")}

    def seed_ffn_state(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.weights.items() if k.startswith("ffn.")}


@dataclass(frozen=True)
class ExpertPack:
    path: Path
    meta: dict[str, Any]
    config: ModelConfig
    weights: dict[str, torch.Tensor]
    probe: torch.Tensor

    @property
    def name(self) -> str:
        return self.meta["name"]

    @property
    def expert_hash(self) -> str:
        return self.meta["hashes"]["expert"]

    @property
    def seed_fingerprint(self) -> str:
        return self.meta["seed_fingerprint"]


# --- probe ---------------------------------------------------------------------------------------------------------


def probe_positions(max_seq: int) -> list[int]:
    return sorted({0, 1, max_seq // 2, max_seq - 1})


def probe_batch(cfg: ModelConfig, seed: int = PROBE_SEED) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randint(N_SPECIAL, cfg.vocab_size, (PROBE_BATCH, cfg.max_seq), generator=generator)


def standalone_probe(
    cfg: ModelConfig,
    backbone: Mapping[str, torch.Tensor],
    expert: Mapping[str, torch.Tensor],
    name: str,
    gqa: str = "sdpa_gqa",
    seed: int = PROBE_SEED,
) -> torch.Tensor:
    """Logits [2, P, V] (fp32, CPU) of the single-expert model backbone + expert on the probe batch."""
    from ..model.lm import RoutedLM

    model = RoutedLM(cfg, [name], gqa=gqa)
    model.load_backbone_state(backbone)
    model.load_expert_state(name, expert)
    model.eval()
    with torch.no_grad():
        logits = model(probe_batch(cfg, seed))
    return logits[:, probe_positions(cfg.max_seq)].float().contiguous()


def _probe_meta(cfg: ModelConfig, plan: str) -> dict[str, Any]:
    return {
        "seed": PROBE_SEED,
        "batch": PROBE_BATCH,
        "seq": cfg.max_seq,
        "positions": probe_positions(cfg.max_seq),
        "plan": plan,
        "dtype": "float32",
        "device": "cpu",
    }


# --- writing -------------------------------------------------------------------------------------------------------


def _created() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cpu(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().to("cpu").contiguous() for k, v in state.items()}


def _common_meta(model, train_meta: Mapping[str, Any]) -> dict[str, Any]:
    commit, dirty = code_identity()
    return {
        "family": model.cfg.family,
        "config": model.cfg.to_dict(),
        "config_hash": config_hash(model.cfg),
        "expert_layers": list(model.cfg.layers_with_experts),
        "train": dict(train_meta),
        "code_commit": train_meta.get("code_commit", commit),
        "code_dirty": train_meta.get("code_dirty", dirty),
        "created": _created(),
    }


def _write(out: Path, meta: dict[str, Any], weights: dict[str, torch.Tensor], probe: torch.Tensor, tokenizer=None):
    with atomic_directory(Path(out)) as tmp:
        save_file(weights, str(tmp / WEIGHTS))
        save_file({"logits": probe}, str(tmp / PROBE))
        meta["hashes"]["weights_file"] = sha256_file(tmp / WEIGHTS)
        meta["hashes"]["probe_file"] = sha256_file(tmp / PROBE)
        if tokenizer is not None:
            shutil.copyfile(tokenizer, tmp / TOKENIZER)
            meta["hashes"]["tokenizer_file"] = sha256_file(tmp / TOKENIZER)
        (tmp / SIDECAR).write_text(json.dumps(meta, sort_keys=True, indent=1, allow_nan=False) + "\n", "utf-8")
    return Path(out)


def write_seed_pack(out: Path, model, tokenizer_path: Path, train_meta: Mapping[str, Any]) -> Path:
    if model.expert_names != ("seed",) or model.cfg.expert_layers is not None:
        raise ConfigMismatchError("a seed pack comes from a seed model: one expert named 'seed' on every layer")
    backbone, ffn = _cpu(model.backbone_state()), _cpu(model.seed_ffn_state())
    tokenizer_sha256 = sha256_file(Path(tokenizer_path))
    meta = _common_meta(model, train_meta)
    hashes = {"backbone": content_hash(backbone), "seed_ffn": content_hash(ffn)}
    meta.update(
        format="bilingual-seed/1",
        hashes=hashes,
        tokenizer_sha256=tokenizer_sha256,
        seed_fingerprint=seed_fingerprint(
            meta["config_hash"], hashes["backbone"], hashes["seed_ffn"], tokenizer_sha256
        ),
        probe=_probe_meta(model.cfg, "single(seed)"),
    )
    probe = standalone_probe(model.cfg, backbone, ffn, "seed", model.gqa)
    return _write(out, meta, {**backbone, **ffn}, probe, tokenizer=Path(tokenizer_path))


def seed_backbone_for(seed: SeedPack, cfg: ModelConfig) -> dict[str, torch.Tensor]:
    """The backbone a model of config `cfg` forked from `seed` has: theta_S of the seed, plus (LP-k) the seed FFN of
    the layers that hold no expert."""
    check_extends(seed, cfg)
    state = dict(seed.backbone_state())
    ffn = seed.seed_ffn_state()
    for layer in cfg.shared_ffn_layers:
        for p in ffn_param_names(cfg):
            state[f"backbone.layers.{layer}.ffn.{p}"] = ffn[f"ffn.{layer}.{p}"]
    return state


def seed_expert_init(seed: SeedPack, cfg: ModelConfig) -> dict[str, torch.Tensor]:
    """An expert as forked from the seed: a copy of the seed FFN on the expert layers."""
    check_extends(seed, cfg)
    ffn = seed.seed_ffn_state()
    return {key: ffn[key].clone() for key in expert_keys(cfg)}


def check_extends(seed: SeedPack, cfg: ModelConfig) -> None:
    """A config forked from a seed may differ from it in expert_layers (LP-k), and a family-I config with experts
    on every layer also in ffn_width (B12a's width-2f expert): neither changes theta_S."""
    base = cfg.replace(expert_layers=None, ffn_width=seed.config.ffn_width)
    wider = cfg.ffn_width != seed.config.ffn_width
    if base != seed.config or (wider and (cfg.family != "swiglu" or cfg.expert_layers is not None)):
        raise ConfigMismatchError(
            f"config {config_hash(cfg)[:12]} is not the seed's config ({seed.meta['config_hash'][:12]}) with expert"
            " layers chosen; only expert_layers (and, for a family-I expert on every layer, ffn_width) may differ"
        )


def write_expert_pack(
    out: Path,
    model,
    name: str,
    seed_pack: Path,
    train_meta: Mapping[str, Any],
    depends_on: Sequence[Path | tuple[str, Path]] = (),
) -> Path:
    """`depends_on` entries are pack paths or (role, path) pairs; each is recorded as {name, expert_hash, path}
    (and its role), so a Bridge (c) or T1' pack names the exact frozen experts it was trained against (IMPL-13);
    the path is relative to the new pack's directory."""
    seed = read_pack(Path(seed_pack))
    if not isinstance(seed, SeedPack):
        raise FormatError(f"{seed_pack} is not a seed pack")
    if not NAME_PATTERN.fullmatch(name) or name not in model.expert_names:
        raise FormatError(f"{name!r} is not an expert of the model {model.expert_names}")
    backbone = _cpu(model.backbone_state())
    if content_hash(backbone) != content_hash(seed_backbone_for(seed, model.cfg)):
        raise BackboneMismatchError(f"the model's backbone differs from the seed pack {seed_pack}")
    expert = _cpu(model.expert_state(name))
    dependencies = []
    for entry in depends_on:
        role, path = entry if isinstance(entry, tuple) else (None, entry)
        dep = read_pack(Path(path))
        if not isinstance(dep, ExpertPack):
            raise FormatError(f"depends_on entry {path} is not an expert pack")
        rel = os.path.relpath(Path(path).resolve(), Path(out).resolve())  # relative to this pack: relocatable
        record = {"name": dep.name, "expert_hash": dep.expert_hash, "path": rel}
        dependencies.append(record if role is None else {**record, "role": role})
    meta = _common_meta(model, train_meta)
    meta.update(
        format="bilingual-expert/1",
        name=name,
        seed_fingerprint=seed.seed_fingerprint,
        seed_config_hash=seed.meta["config_hash"],
        hashes={"expert": content_hash(expert)},
        depends_on=dependencies,
        probe=_probe_meta(model.cfg, f"single({name})"),
    )
    probe = standalone_probe(model.cfg, backbone, expert, name, model.gqa)
    return _write(out, meta, expert, probe)


# --- reading -------------------------------------------------------------------------------------------------------


def _check_keys(got: set[str], expected: set[str], path: Path) -> None:
    missing, unexpected = sorted(expected - got), sorted(got - expected)
    if missing:
        raise MissingKeyError(f"{path / WEIGHTS} lacks {len(missing)} keys, e.g. {missing[:3]}")
    if unexpected:
        raise UnexpectedKeyError(f"{path / WEIGHTS} holds {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")


def read_pack(path: Path) -> SeedPack | ExpertPack:
    path = Path(path)
    try:
        meta = json.loads((path / SIDECAR).read_text("utf-8"))
        match = FORMAT_PATTERN.fullmatch(str(meta.get("format", "")))
        kind = match.group(1) if match else None
        if kind not in ("seed", "expert"):
            raise ValueError(f"{path / SIDECAR}: not a seed or expert pack (format {meta.get('format')!r})")
        check_format(meta, kind)
        cfg = ModelConfig.from_dict(meta["config"])
        hashes = meta["hashes"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FormatError(f"{path}: {exc}") from exc
    if config_hash(cfg) != meta.get("config_hash"):
        raise HashMismatchError(f"{path / SIDECAR}: config_hash does not match its config")
    files = [(WEIGHTS, "weights_file"), (PROBE, "probe_file")] + (
        [(TOKENIZER, "tokenizer_file")] if kind == "seed" else []
    )
    for filename, key in files:
        if not (path / filename).is_file() or sha256_file(path / filename) != hashes.get(key):
            raise HashMismatchError(f"{path / filename} does not match the sha256 recorded in {SIDECAR}")
    weights = load_file(str(path / WEIGHTS))
    probe = load_file(str(path / PROBE))["logits"]
    if kind == "seed":
        if cfg.expert_layers is not None:
            raise FormatError(f"{path}: a seed pack holds the seed FFN of every layer")
        _check_keys(set(weights), set(backbone_keys(cfg)) | set(expert_keys(cfg)), path)
        pack = SeedPack(path, meta, cfg, weights, probe)
        backbone_hash, ffn_hash = content_hash(pack.backbone_state()), content_hash(pack.seed_ffn_state())
        if (backbone_hash, ffn_hash) != (hashes.get("backbone"), hashes.get("seed_ffn")):
            raise HashMismatchError(f"{path}: backbone or seed FFN content differs from its recorded hash")
        if meta.get("tokenizer_sha256") != hashes["tokenizer_file"]:
            raise HashMismatchError(f"{path}: tokenizer_sha256 differs from the tokenizer file's hash")
        expected = seed_fingerprint(meta["config_hash"], backbone_hash, ffn_hash, meta["tokenizer_sha256"])
        if meta.get("seed_fingerprint") != expected:
            raise HashMismatchError(f"{path}: seed_fingerprint does not match the pack's content")
    else:
        if not NAME_PATTERN.fullmatch(str(meta.get("name", ""))):
            raise FormatError(f"{path}: expert name {meta.get('name')!r} does not match {NAME_PATTERN.pattern}")
        _check_keys(set(weights), set(expert_keys(cfg)), path)
        pack = ExpertPack(path, meta, cfg, weights, probe)
        if content_hash(weights) != hashes.get("expert"):
            raise HashMismatchError(f"{path}: expert content differs from its recorded hash")
    expected_shape = [meta["probe"]["batch"], len(meta["probe"]["positions"]), cfg.vocab_size]
    if list(probe.shape) != expected_shape or probe.dtype != torch.float32:
        raise FormatError(
            f"{path / PROBE}: logits {probe.dtype} {list(probe.shape)}, expected float32 {expected_shape}"
        )
    return pack
