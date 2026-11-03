"""Canonical JSON, sha256 helpers, versioned formats and atomic writes. Knows no schema.

The same conventions as fork/bilingual, in this package's own copy (neither package imports the other): canonical
JSON is UTF-8 with sorted keys, separators (",", ":") and no NaN or infinity; every recorded hash is the lowercase
hex sha256 of such bytes or of a file; every document carries `format: "corpora-<name>/<major>"` and readers refuse
another name or an unknown major. Every write is atomic: the bytes go to `<target>.tmp-<pid>-<rand>` in the target's
directory, are fsynced (the directory too), and `os.replace` puts them in place.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

TEMP_MARK = ".tmp-"
FORMAT_PATTERN = re.compile(r"corpora-([a-z][a-z0-9-]*)/(\d+)")


class CorporaError(ValueError):
    """A refusal: the input, the remote or the state on disk is not what the contract says. The command line prints
    the message and exits 1."""


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_format(doc: Mapping[str, Any], name: str, majors: tuple[int, ...] = (1,)) -> int:
    """The major version of a `corpora-<name>/<major>` document; refuses another name or an unknown major."""
    match = FORMAT_PATTERN.fullmatch(str(doc.get("format", "")))
    if not match or match.group(1) != name:
        raise CorporaError(f"expected a corpora-{name} document, found format {doc.get('format')!r}")
    major = int(match.group(2))
    if major not in majors:
        raise CorporaError(f"corpora-{name}/{major} is not a known version (known: {list(majors)})")
    return major


def with_sha256(doc: Mapping[str, Any], field: str) -> dict[str, Any]:
    """`doc` plus `field` = sha256 of the canonical JSON of `doc` without that field."""
    body = {k: v for k, v in doc.items() if k != field}
    return {**body, field: sha256_json(body)}


def check_sha256(doc: Mapping[str, Any], field: str) -> None:
    body = {k: v for k, v in doc.items() if k != field}
    if sha256_json(body) != doc.get(field):
        raise CorporaError(f"{field} does not match the document's content (edited by hand?)")


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(target: Path, data: bytes) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.name}{TEMP_MARK}{os.getpid()}-{secrets.token_hex(4)}")
    try:
        with open(temp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    fsync_dir(target.parent)
    return target


def json_text(obj: Any) -> str:
    """Human-readable JSON (sorted keys, one-space indent, no NaN): deterministic for equal objects. Hashes are taken
    over canonical_json, not over these bytes."""
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def atomic_write_json(target: Path, obj: Any) -> Path:
    return atomic_write_bytes(target, json_text(obj).encode("utf-8"))


def read_json(path: Path, name: str) -> dict[str, Any]:
    """A `corpora-<name>/1` document from disk; refuses anything else."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise CorporaError(f"{path}: cannot read a corpora-{name} document: {err}") from None
    if not isinstance(doc, dict):
        raise CorporaError(f"{path}: a corpora-{name} document is a JSON object")
    check_format(doc, name)
    return doc
