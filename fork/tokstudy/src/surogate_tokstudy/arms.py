"""Arm registries (`tokstudy-arms/1`, `arms/<project>.toml`) and the codec of each arm.

A registry holds the arms of exactly one table:

    format = "tokstudy-arms/1"
    table = "tlc-N"                      # or "bilingual-B"

    [[arm]]
    id = "N0"
    kind = "subword"                     # subword | bytes | hnet
    tokenizer = {path = "tok/tokenizer.json", sha256 = "<64 hex>"}   # subword only; path relative to the file
    model = {profile = "a0"}             # profile, overrides, hnet: read by models.build
    seeds = [0, 1, 2]
    stages = ["screen", "decision"]

The table fixes the project, hence the special ids (TLC or bilingual ids 0-4) and the main network (`twin_a0` for
tlc-N, `bilingual_trunk` for bilingual-B); an arm may restate `specials` and `main`, which must then agree. Loading
checks the sha256 of every tokenizer and its special ids, and refuses duplicate ids, a second table, unknown keys
and kinds, and a tokenizer on a bytes or H-Net arm. arm_sha256 is the sha256 of the canonical JSON of the arm with
its tokenizer reduced to the sha256: moving a file does not change an arm, changing its bytes does.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .canonical import TokstudyError, check_format, check_sha256_value, sha256_file, sha256_json
from .codecs import PROJECT_SPECIALS, ByteCodec, Codec, SubwordCodec

TABLES = {"tlc-N": ("tlc", "twin_a0"), "bilingual-B": ("bilingual", "bilingual_trunk")}
KINDS = ("subword", "bytes", "hnet")
ARM_KEYS = {"id", "kind", "tokenizer", "specials", "model", "main", "seeds", "stages"}


@dataclass(frozen=True)
class TokRef:
    path: Path
    sha256: str


@dataclass(frozen=True)
class ArmSpec:
    id: str
    table: str
    kind: Literal["subword", "bytes", "hnet"]
    tokenizer: TokRef | None
    specials: Mapping[str, int]
    model: Mapping[str, Any]
    main: str
    seeds: tuple[int, ...]
    stages: tuple[str, ...]

    @property
    def project(self) -> str:
        return TABLES[self.table][0]


def arm_sha256(arm: ArmSpec) -> str:
    body = asdict(arm)
    body["tokenizer"] = None if arm.tokenizer is None else arm.tokenizer.sha256
    return sha256_json(body)


def _arm(table: str, entry: Mapping[str, Any], base: Path) -> ArmSpec:
    where = f"arm {entry.get('id')!r}"
    unknown = sorted(set(entry) - ARM_KEYS)
    if unknown:
        raise TokstudyError(f"{where}: unknown keys {unknown}")
    if not isinstance(entry.get("id"), str) or not entry["id"]:
        raise TokstudyError(f"{where}: id must be a non-empty string")
    kind = entry.get("kind")
    if kind not in KINDS:
        raise TokstudyError(f"{where}: kind must be one of {KINDS}, got {kind!r}")
    project, main = TABLES[table]
    if entry.get("main", main) != main:
        raise TokstudyError(f"{where}: table {table} trains on {main!r}, not {entry['main']!r}")
    if entry.get("specials", project) != project:
        raise TokstudyError(f"{where}: table {table} uses the {project} special ids, not {entry['specials']!r}")
    tokenizer = entry.get("tokenizer")
    if (kind == "subword") != (tokenizer is not None):
        raise TokstudyError(f"{where}: a {kind} arm {'needs' if kind == 'subword' else 'has no'} tokenizer")
    ref = None
    if tokenizer is not None:
        if not isinstance(tokenizer, Mapping) or set(tokenizer) != {"path", "sha256"}:
            raise TokstudyError(f"{where}: tokenizer = {{path, sha256}}")
        ref = TokRef((base / tokenizer["path"]).resolve(), check_sha256_value(tokenizer["sha256"], where))
        if not ref.path.is_file():
            raise TokstudyError(f"{where}: {ref.path} does not exist")
        found = sha256_file(ref.path)
        if found != ref.sha256:
            raise TokstudyError(f"{where}: {ref.path} has sha256 {found}, the registry says {ref.sha256}")
    seeds = entry.get("seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise TokstudyError(f"{where}: seeds must be a non-empty list of distinct integers")
    if not all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds):
        raise TokstudyError(f"{where}: seeds must be non-negative integers")
    stages = entry.get("stages")
    if not isinstance(stages, list) or not stages or not all(isinstance(s, str) and s for s in stages):
        raise TokstudyError(f"{where}: stages must be a non-empty list of names")
    model = entry.get("model", {})
    if not isinstance(model, Mapping):
        raise TokstudyError(f"{where}: model must be a table")
    return ArmSpec(entry["id"], table, kind, ref, dict(PROJECT_SPECIALS[project]), dict(model), main,
                   tuple(seeds), tuple(stages))  # fmt: skip


def load_arms(path: Path) -> list[ArmSpec]:
    """The arms of one registry, every tokenizer sha256 checked."""
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise TokstudyError(f"{path}: cannot read an arm registry: {err}") from None
    check_format(data, "arms")
    unknown = sorted(set(data) - {"format", "table", "arm"})
    if unknown:
        raise TokstudyError(f"{path}: unknown top-level keys {unknown}")
    table = data.get("table")
    if table not in TABLES:
        raise TokstudyError(f"{path}: table must be one of {sorted(TABLES)}, got {table!r}")
    entries = data.get("arm", [])
    if not entries:
        raise TokstudyError(f"{path}: no [[arm]]")
    if any(isinstance(e, Mapping) and e.get("table", table) != table for e in entries):
        raise TokstudyError(f"{path}: arms of more than one table; a registry holds one table ({table})")
    arms = [_arm(table, {k: v for k, v in e.items() if k != "table"}, path.parent) for e in entries]
    ids = [a.id for a in arms]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise TokstudyError(f"{path}: duplicate arm ids {duplicates}")
    return arms


def codec_for(arm: ArmSpec, *, check: bool = False) -> Codec | None:
    """The arm's codec: its tokenizer (special ids checked), the byte codec, or None for an H-Net arm (bytes in,
    learned chunks inside: it has no token-level intrinsic metrics)."""
    if arm.kind == "subword":
        return SubwordCodec.from_tokenizer_json(arm.tokenizer.path, arm.tokenizer.sha256, arm.specials, name=arm.id,
                                                check=check)  # fmt: skip
    if arm.kind == "bytes":
        return ByteCodec(arm.specials, name=arm.id, check=check)
    return None


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_tokstudy arms", description="load and check a registry")
    parser.add_argument("registry", type=Path)
    args = parser.parse_args(list(argv))
    try:
        arms = load_arms(args.registry)
        for arm in arms:
            codec = codec_for(arm)
            vocab = "-" if codec is None else codec.vocab_size
            print(f"{arm.id}\t{arm.table}\t{arm.kind}\tvocab {vocab}\tseeds {len(arm.seeds)}\t{arm_sha256(arm)}")
    except TokstudyError as err:
        print(f"arms: {err}", file=sys.stderr)
        return 1
    return 0
