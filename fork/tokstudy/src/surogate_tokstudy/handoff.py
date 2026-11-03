"""Tokenizer hand-off between the projects and the study (spec §1.3): the projects build and own every tokenizer;
the study reads each `tokenizer.json` by path and sha256 and checks what it was trained on.

**In: project -> registry** (`arms REGISTRY --tlc SPEC | --bilingual MANIFEST [--pin] [--record FILE]`). For every
subword arm of the registry, the project's own record next to its `tokenizer.json` is read and checked against the
corpus the table's stream is built from, so no arm can be scored with a tokenizer that saw other text or the
held-out split:

| project | record next to the file | checks |
|---|---|---|
| TLC | `tokenizer_manifest.json` (data.tokenizer / data.freqtrim) | `tokenizer_sha256` is the file's; `corpus.spec_sha256` and `corpus.decontam` are the given spec's (the stream's `--tlc`); `corpus.trained_on` is the train split; no normalizer |
| bilingual | `<stem>.meta.json` (`bilingual-tokenizer/1`) | `tokenizer_sha256` is the file's; `split` is `seed_train`; `manifest_sha256` is the given derived manifest's (the stream's `--bilingual`); normalization `none` |

`--pin` writes each subword arm's sha256 into the registry (a placeholder "unpinned", or an older value) by
rewriting the one line `tokenizer = {path = "...", sha256 = "..."}` of that arm, then re-loads the registry with
every sha256 checked; comments and layout stay. Without `--pin` a sha256 that differs from the file is refused.
`--record` writes `tokstudy-handoff/1`: the registry's sha256, the corpus identity and, per subword arm, the
tokenizer's sha256 and the project's recipe (kind, pre-tokenizer or regex, vocabulary size, teacher for freqtrim).

**Out: verdict -> project.** The study never writes into a project. The winning arm's tokenizer is a file the project
built; freezing it is the project's step (TLC: the tokenizer directory every row's `init --tokenizer` and shards use;
bilingual: the `run.tokenizer` of every row from G1' on, whose sha256 enters every run id and seed fingerprint). The
record names the sha256 the project's frozen file must have (docs/runbook.md, 冻结).
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

from .arms import PLACEHOLDER, TABLES, load_arms
from .canonical import TokstudyError, atomic_write_bytes, atomic_write_json, check_format, sha256_file
from .docs import _derived_manifest, tlc_source

HANDOFF_FORMAT = "tokstudy-handoff/1"
TLC_RECORD = "tokenizer_manifest.json"
BILINGUAL_RECORD = "bilingual-tokenizer/1"
TOKENIZER_LINE = re.compile(
    r'^(?P<head>\s*tokenizer\s*=\s*\{\s*path\s*=\s*"(?P<path>[^"\\]*)"\s*,\s*sha256\s*=\s*")(?P<sha>[^"\\]*)'
    r'(?P<tail>"\s*\}\s*(?:#.*)?)$'
)


def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise TokstudyError(f"{path}: cannot read {what}: {err}") from None
    if not isinstance(doc, dict):
        raise TokstudyError(f"{path}: {what} is a JSON object")
    return doc


def tlc_provenance(path: Path, spec: Path) -> dict[str, Any]:
    """What TLC's own record says of tokenizer.json at `path`, checked against corpus spec `spec`."""
    record_path = Path(path).with_name(TLC_RECORD)
    record = _read_json(record_path, "a TLC tokenizer manifest")
    found = sha256_file(path)
    if record.get("tokenizer_sha256") != found:
        raise TokstudyError(f"{record_path}: records tokenizer sha256 {record.get('tokenizer_sha256')}, "
                            f"the file has {found}")  # fmt: skip
    corpus = record.get("corpus") or {}
    source = tlc_source(spec, "train")
    if corpus.get("spec_sha256") != source["spec_sha256"]:
        raise TokstudyError(f"{path}: trained on the corpus spec with sha256 {corpus.get('spec_sha256')}, not on {spec} "
                            f"({source['spec_sha256']}): the stream reads another corpus")  # fmt: skip
    if corpus.get("decontam") != source["decontam"]:
        raise TokstudyError(f"{path}: trained under another decontamination report than {spec} names")
    if corpus.get("trained_on") != "train split":
        raise TokstudyError(f"{path}: trained on {corpus.get('trained_on')!r}, not the train split")
    if record.get("normalizer") is not None:
        raise TokstudyError(f"{path}: has a normalizer; the study scores raw bytes")
    teacher = record.get("teacher")
    return {
        "project": "tlc",
        "kind": record.get("kind", "bpe"),
        "pretokenizer": record.get("pretokenizer"),
        "regex": record.get("regex"),
        "vocab_size": record.get("vocab_size"),
        "teacher": None if teacher is None else {k: teacher.get(k) for k in ("repo", "revision", "file_sha256")},
        "record": record_path.name,
        "record_sha256": sha256_file(record_path),
        "corpus": {"spec_sha256": corpus["spec_sha256"], "decontam": corpus["decontam"], "split": "train"},
    }


def bilingual_provenance(path: Path, manifest: Path) -> dict[str, Any]:
    """What the bilingual project's tokenizer.meta.json says of `path`, checked against derived manifest `manifest`."""
    path = Path(path)
    record_path = path.with_name(path.stem + ".meta.json")
    record = _read_json(record_path, "a bilingual tokenizer record")
    if record.get("format") != BILINGUAL_RECORD:
        raise TokstudyError(f"{record_path}: not a {BILINGUAL_RECORD} document")
    found = sha256_file(path)
    if record.get("tokenizer_sha256") != found:
        raise TokstudyError(f"{record_path}: records tokenizer sha256 {record.get('tokenizer_sha256')}, "
                            f"the file has {found}")  # fmt: skip
    derived = _derived_manifest(manifest)
    if record.get("split") != "seed_train":
        raise TokstudyError(f"{path}: trained on split {record.get('split')!r}, not seed_train (B-O11)")
    if record.get("manifest_sha256") != derived.get("manifest_sha256"):
        raise TokstudyError(f"{path}: trained on the manifest {record.get('manifest_sha256')}, not on {manifest} "
                            f"({derived.get('manifest_sha256')}): the stream reads another corpus")  # fmt: skip
    if record.get("normalization") != "none":
        raise TokstudyError(f"{path}: normalization {record.get('normalization')!r}; the study scores raw bytes")
    return {
        "project": "bilingual",
        "kind": "bbpe",
        "pretokenizer": record.get("pretokenizer"),
        "vocab_size": record.get("vocab_size"),
        "record": record_path.name,
        "record_sha256": sha256_file(record_path),
        "corpus": {"manifest_sha256": derived["manifest_sha256"], "split": "seed_train"},
    }


def _registry(path: Path) -> tuple[str, list[dict[str, Any]], str]:
    try:
        text = Path(path).read_text(encoding="utf-8")
        data = tomllib.loads(text)
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise TokstudyError(f"{path}: cannot read an arm registry: {err}") from None
    check_format(data, "arms")
    if data.get("table") not in TABLES:
        raise TokstudyError(f"{path}: table must be one of {sorted(TABLES)}, got {data.get('table')!r}")
    return data["table"], [a for a in data.get("arm", []) if a.get("kind") == "subword"], text


def handoff(registry: Path, *, tlc: Path | None = None, bilingual: Path | None = None, pin: bool = False,
            record: Path | None = None) -> dict[str, Any]:  # fmt: skip
    """Check (and with `pin`, write) the sha256 and the provenance of every subword arm's tokenizer."""
    registry = Path(registry)
    table, subword, text = _registry(registry)
    project = TABLES[table][0]
    corpus = {"tlc": tlc, "bilingual": bilingual}
    if corpus[project] is None or corpus["bilingual" if project == "tlc" else "tlc"] is not None:
        flag = "--tlc SPEC" if project == "tlc" else "--bilingual MANIFEST"
        raise TokstudyError(f"table {table} is a {project} table: give {flag}, the corpus its stream reads")
    arms, new_sha = [], {}
    for entry in subword:
        tok = entry.get("tokenizer")
        if not isinstance(tok, dict) or set(tok) != {"path", "sha256"}:
            raise TokstudyError(f"arm {entry.get('id')!r}: tokenizer = {{path, sha256}}")
        path = (registry.parent / tok["path"]).resolve()
        if not path.is_file():
            raise TokstudyError(f"arm {entry['id']}: {path} does not exist (build it with the project's tokenizer "
                                "command first)")  # fmt: skip
        found = sha256_file(path)
        if not pin and tok["sha256"] != found:
            what = "is not pinned yet" if tok["sha256"] == PLACEHOLDER else f"is pinned to {tok['sha256']}"
            raise TokstudyError(f"arm {entry['id']}: {tok['path']} {what}, the file has sha256 {found} (--pin)")
        provenance = tlc_provenance(path, tlc) if project == "tlc" else bilingual_provenance(path, bilingual)
        if new_sha.setdefault(tok["path"], found) != found:
            raise TokstudyError(f"{tok['path']}: changed while it was read")
        arms.append({"id": entry["id"], "tokenizer": {"path": tok["path"], "sha256": found, "previous": tok["sha256"]},
                     "provenance": provenance})  # fmt: skip
    changed = [a["id"] for a in arms if a["tokenizer"]["previous"] != a["tokenizer"]["sha256"]]
    if pin and changed:
        lines = text.splitlines(keepends=True)
        hits = 0
        for i, line in enumerate(lines):
            m = TOKENIZER_LINE.match(line.rstrip("\r\n"))
            if m and m.group("path") in new_sha:
                end = line[len(line.rstrip("\r\n")) :]
                lines[i] = m.group("head") + new_sha[m.group("path")] + m.group("tail") + end
                hits += 1
        if hits != len(subword):
            raise TokstudyError(f"{registry}: write every subword arm's tokenizer on one line as "
                                'tokenizer = {path = "...", sha256 = "..."} (found {hits} of {len(subword)})')  # fmt: skip
        atomic_write_bytes(registry, "".join(lines).encode("utf-8"))
    loaded = load_arms(registry)  # every sha256 checked, as every later command does
    if project == "tlc":
        source = tlc_source(tlc, "train")
    else:
        source = {"manifest_sha256": _derived_manifest(bilingual)["manifest_sha256"], "split": "seed_train"}
    doc = {
        "format": HANDOFF_FORMAT,
        "table": table,
        "registry": registry.name,
        "registry_sha256": sha256_file(registry),
        "corpus": source,
        "arms": arms,
        "other_arms": [a.id for a in loaded if a.kind != "subword"],
        "pinned": changed if pin else [],
    }
    if record is not None:
        atomic_write_json(record, doc)
    return doc
