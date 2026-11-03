"""Writers of the export directory: JSONL parts in each project's corpus format, the `[[source]]` spec fragment, the
SPDX attribution list and the provenance manifest.

- `parts/<name>-NNNNN.jsonl`: one canonical JSON object per line (UTF-8, sorted keys, no ASCII escaping), at most
  `part_bytes` (256 MiB) per part. A part rolls over before a line that would overflow it, so parts depend only on
  the order of the documents. The text is the source's bytes, decoded and nothing else: no newline, Unicode or
  whitespace normalisation of any kind.
- TLC lines: code `{"group": repo_name, "id": uuid, "meta": {...}, "text"}`; English `{"id", "meta", "text"}`
  without a group. `surogate_tlc.data.corpus` reads `text` and `group` and ignores the rest.
- Bilingual lines: `{"id": "<file stem>:<rg>:<row>", "meta": {"upstream_source", "score", "unit", ...}, "text"}`;
  parallel lines `{"id", "pair_id", "segments": [{"lang": "en", ...}, {"lang": "zh", ...}], "meta"}`. Only keys the
  bilingual document reader knows; never `group`, which it refuses. Ultra-FineWeb zh gets one part series per
  upstream source (`ufw-zh-tele`, `ufw-zh-cci3`, ...), so each becomes its own fragment entry with its own licence.
- `spec_fragment.toml`: `[[source]]` tables to paste into the project's spec, with the exact part paths (relative to
  the export directory), url, pinned revision, licence, `downloaded` and `filters` (every applied filter, in order,
  then `corpora export <plan_sha256>`). Bilingual entries carry `licence_verified = false`.
- `attribution.jsonl` (code exports): one line per repository, `{"repo", "commit_ids", "spdx", "files", "bytes"}`.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import CorporaError, atomic_write_bytes, canonical_json, sha256_file
from .filters import TEXT, TEXTS
from .profiles import Profile
from .sources import SourceDef

PART_BYTES = 256 << 20
PARTS = "parts"
NAME = re.compile(r"[a-z][a-z0-9_-]*")


@dataclass(frozen=True)
class PartInfo:
    path: str  # relative to the export directory
    sha256: str
    bytes: int
    docs: int

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes, "docs": self.docs}


def unit_ref(path: str, rg: int, row: int) -> str:
    return f"{path}#{rg}:{row}"


def file_stem(path: str) -> str:
    return path.rpartition("/")[2].removesuffix(".parquet")


def finite(value: Any) -> float | None:
    """A score as a JSON number: Ultra-FineWeb stores decimal strings; NaN, infinities and unparsable values become
    null (canonical JSON has no NaN)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


class PartSeries:
    """`parts/<name>-NNNNN.jsonl`, appended line by line; `state()` and `restore()` make it resumable."""

    def __init__(self, out_dir: Path, name: str, max_bytes: int):
        if not NAME.fullmatch(name):
            raise CorporaError(f"part name {name!r} must match {NAME.pattern}")
        self.out_dir = Path(out_dir)
        self.name = name
        self.max_bytes = max_bytes
        self.parts: list[list[Any]] = []  # [relative path, bytes, docs]
        self.handle = None

    def _path(self, index: int) -> str:
        return f"{PARTS}/{self.name}-{index:05d}.jsonl"

    def _open_last(self) -> None:
        (self.out_dir / PARTS).mkdir(parents=True, exist_ok=True)
        self.handle = open(self.out_dir / self.parts[-1][0], "ab")

    def write(self, line: bytes) -> None:
        if not self.parts or (self.parts[-1][1] and self.parts[-1][1] + len(line) > self.max_bytes):
            self.close()
            self.parts.append([self._path(len(self.parts)), 0, 0])
        if self.handle is None:
            self._open_last()
        self.handle.write(line)
        self.parts[-1][1] += len(line)
        self.parts[-1][2] += 1

    def sync(self) -> None:
        if self.handle is not None:
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def close(self) -> None:
        if self.handle is not None:
            self.sync()
            self.handle.close()
            self.handle = None

    def state(self) -> list[list[Any]]:
        return [list(p) for p in self.parts]

    def restore(self, state: Sequence[Sequence[Any]]) -> None:
        """Cut the series back to `state` (a finished unit's): truncate the recorded parts to their recorded size."""
        self.close()
        self.parts = [list(p) for p in state]
        for path, size, _ in self.parts:
            target = self.out_dir / path
            if not target.is_file() or target.stat().st_size < size:
                raise CorporaError(f"{target}: missing or shorter than the done list records; start a new directory")
            with open(target, "r+b") as f:
                f.truncate(size)


class Writer:
    """Base writer: turns kept rows into lines of named part series."""

    project = ""

    def __init__(self, out_dir: Path, source: SourceDef, profile: Profile, *, part_bytes: int = PART_BYTES):
        self.out_dir = Path(out_dir)
        self.source = source
        self.profile = profile
        self.part_bytes = part_bytes
        self.series: dict[str, PartSeries] = {}

    def series_name(self, row: Mapping[str, Any]) -> str:
        return self.source.name

    def line(self, row: Mapping[str, Any], path: str, rg: int, index: int, meta: Mapping[str, Any]) -> dict:
        raise NotImplementedError

    def observe(self, obj: Mapping[str, Any]) -> None:
        """Called for every line written (and for every line restored on resume)."""

    def write(self, row: Mapping[str, Any], path: str, rg: int, index: int, meta: Mapping[str, Any]) -> None:
        obj = self.line(row, path, rg, index, meta)
        name = self.series_name(row)
        if name not in self.series:
            self.series[name] = PartSeries(self.out_dir, name, self.part_bytes)
        self.series[name].write(canonical_json(obj) + b"\n")
        self.observe(obj)

    def state(self) -> dict[str, list[list[Any]]]:
        return {name: s.state() for name, s in sorted(self.series.items())}

    def restore(self, state: Mapping[str, Sequence[Sequence[Any]]]) -> None:
        """Cut every series back to `state` (the last finished unit's) and delete this writer's part files that
        `state` does not list: what an interrupted unit had written."""
        self.series = {}
        for name, parts in state.items():
            self.series[name] = PartSeries(self.out_dir, name, self.part_bytes)
            self.series[name].restore(parts)
        listed = {p[0] for parts in state.values() for p in parts}
        directory = self.out_dir / PARTS
        for path in sorted(directory.glob("*.jsonl")) if directory.is_dir() else []:
            if f"{PARTS}/{path.name}" not in listed and self.owns(path.name):
                path.unlink()

    def restored_lines(self) -> Iterator[dict[str, Any]]:
        """Every line written so far, in order, each passed to `observe` (rebuilds the attribution on resume)."""
        for name in sorted(self.series):
            for path, _, _ in self.series[name].parts:
                with open(self.out_dir / path, "rb") as f:
                    for raw in f:
                        obj = json.loads(raw)
                        self.observe(obj)
                        yield obj

    def owns(self, filename: str) -> bool:
        base = re.escape(self.source.name)
        return re.fullmatch(rf"{base}(-[a-z0-9-]+)?-\d{{5}}\.jsonl", filename) is not None

    def texts(self, obj: Mapping[str, Any]) -> dict[str, str]:
        """The text columns of a written line, as the dedup filter hashes them."""
        return {self.source.content[0]: obj["text"]}

    def sync(self) -> None:
        for s in self.series.values():
            s.sync()

    def close(self) -> dict[str, list[PartInfo]]:
        out = {}
        for name, s in sorted(self.series.items()):
            s.close()
            out[name] = [PartInfo(p, sha256_file(self.out_dir / p), b, d) for p, b, d in s.parts]
        return out

    # --- fragments ----------------------------------------------------------------------------------------------

    def licence_of(self, name: str) -> str:
        raise NotImplementedError

    def fragment_entry(self, name: str, parts: Sequence[PartInfo], filters: Sequence[str], downloaded: str) -> dict:
        raise NotImplementedError


class TlcWriter(Writer):
    project = "tlc"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attribution: dict[str, dict[str, Any]] = {}

    def line(self, row, path, rg, index, meta):
        unit = unit_ref(path, rg, index)
        text = row[TEXT]
        if self.profile.kind == "code":
            info: dict[str, Any] = {"path": row.get("relative_path"), "unit": unit}
            if "source_meta.commit_id" in row:
                info["commit"] = row["source_meta.commit_id"]
                info["spdx"] = list(row.get("source_meta.file_meta.detected_licenses") or [])
            if "algo_rel_score" in row:
                info["algo_rel_score"] = finite(row["algo_rel_score"])
            obj = {"id": row["uuid"], "meta": {**info, **meta}, "text": text}
            if isinstance(row.get("repo_name"), str):
                obj["group"] = row["repo_name"]
            return obj
        info = {"unit": unit, "score": finite(row.get("score"))}
        return {"id": f"{file_stem(path)}:{rg}:{index}", "meta": {**info, **meta}, "text": text}

    def observe(self, obj):
        if self.profile.kind != "code" or "group" not in obj:
            return
        entry = self.attribution.setdefault(obj["group"], {"commit_ids": set(), "spdx": set(), "files": 0, "bytes": 0})
        if obj["meta"].get("commit"):
            entry["commit_ids"].add(obj["meta"]["commit"])
        entry["spdx"].update(obj["meta"].get("spdx") or [])
        entry["files"] += 1
        entry["bytes"] += len(obj["text"].encode("utf-8"))

    def attribution_lines(self) -> bytes:
        lines = [
            canonical_json(
                {
                    "repo": repo,
                    "commit_ids": sorted(e["commit_ids"]),
                    "spdx": sorted(e["spdx"]),
                    "files": e["files"],
                    "bytes": e["bytes"],
                }
            )
            + b"\n"
            for repo, e in sorted(self.attribution.items())
        ]
        return b"".join(lines)

    def licence_of(self, name):
        return self.source.licence.tlc_license + self.profile.licence_note

    def fragment_entry(self, name, parts, filters, downloaded):
        return {
            "name": name,
            "kind": self.profile.kind,
            "files": [p.path for p in parts],
            "url": f"https://huggingface.co/datasets/{self.source.repo}",
            "revision": self.source.revision,
            "license": self.licence_of(name),
            "downloaded": downloaded,
            "filters": list(filters),
        }


class BilingualWriter(Writer):
    project = "bilingual"

    def series_name(self, row):
        if self.source.upstream_column:
            return f"{self.source.name}-{slug(str(row[self.source.upstream_column]))}"
        return self.source.name

    def upstream_of(self, name: str) -> str:
        if not self.source.upstream_column:
            return "*"
        for upstream in self.source.licence.spdx:
            if f"{self.source.name}-{slug(upstream)}" == name:
                return upstream
        raise CorporaError(f"{name}: an upstream source the registry has no licence id for")

    def line(self, row, path, rg, index, meta):
        info = {
            "unit": unit_ref(path, rg, index),
            "upstream_source": row.get(self.source.upstream_column or "source"),
            "score": finite(row.get("score")),
        }
        return {"id": f"{file_stem(path)}:{rg}:{index}", "meta": {**info, **meta}, "text": row[TEXT]}

    def licence_of(self, name):
        spdx = self.source.licence.spdx
        upstream = self.upstream_of(name)
        if upstream not in spdx:
            raise CorporaError(f"{name}: the registry has no bilingual licence id for {self.source.id} ({upstream})")
        return spdx[upstream]

    def fragment_entry(self, name, parts, filters, downloaded):
        return {
            "name": name,
            "kind": self.profile.kind,
            "files": [p.path for p in parts],
            "url": f"https://huggingface.co/datasets/{self.source.repo}",
            "revision": self.source.revision,
            "licence": self.licence_of(name),
            "licence_verified": False,
            "downloaded": downloaded,
            "filters": list(filters),
        }


class ParallelWriter(BilingualWriter):
    """Source segment first (en for UN PC, whose original language is not recorded); the bilingual Bridge layout
    decides the direction later."""

    def line(self, row, path, rg, index, meta):
        texts = row[TEXTS]
        key = f"{self.source.name}-{file_stem(path)}:{rg}:{index}"
        segments = [{"lang": c.rpartition(".")[2], "text": texts[c]} for c in self.source.content]
        return {"id": key, "pair_id": key, "segments": segments, "meta": {"unit": unit_ref(path, rg, index), **meta}}

    def texts(self, obj):
        by_lang = {s["lang"]: s["text"] for s in obj["segments"]}
        return {c: by_lang[c.rpartition(".")[2]] for c in self.source.content}


WRITER_CLASSES = {"tlc": TlcWriter, "bilingual": BilingualWriter, "parallel": ParallelWriter}


def make_writer(out_dir: Path, source: SourceDef, profile: Profile, *, part_bytes: int = PART_BYTES) -> Writer:
    return WRITER_CLASSES[profile.writer](out_dir, source, profile, part_bytes=part_bytes)


# --- the fragment ---------------------------------------------------------------------------------------------------


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # a TOML basic string accepts every JSON escape json.dumps writes
    if isinstance(value, list):
        if not value:
            return "[]"
        return "[\n" + "".join(f"    {toml_value(v)},\n" for v in value) + "]"
    raise CorporaError(f"cannot write {value!r} as TOML")


def fragment_text(entries: Sequence[Mapping[str, Any]], plan_sha256: str) -> str:
    lines = [
        f"# [[source]] tables written by `python -m surogate_corpora fetch` (corpora export {plan_sha256}).",
        "# File paths are relative to the export directory: keep the spec next to it, or adjust them when pasting.",
    ]
    for entry in entries:
        lines += ["", "[[source]]"] + [f"{k} = {toml_value(v)}" for k, v in entry.items()]
    return "\n".join(lines) + "\n"


def write_fragment(out_dir: Path, entries: Sequence[Mapping[str, Any]], plan_sha256: str) -> Path:
    return atomic_write_bytes(Path(out_dir) / "spec_fragment.toml", fragment_text(entries, plan_sha256).encode())


def write_attribution(out_dir: Path, data: bytes) -> Path:
    return atomic_write_bytes(Path(out_dir) / "attribution.jsonl", data)
