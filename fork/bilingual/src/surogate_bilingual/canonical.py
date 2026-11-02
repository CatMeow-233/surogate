"""Canonical JSON, sha256 helpers and atomic writes (docs/spec.md, "数据契约摘要"). Knows no schema.

Canonical JSON is UTF-8 with sorted keys, separators (",", ":") and no NaN or infinity; every recorded hash is the
lowercase hex sha256 of such bytes or of a file. Every write is atomic: the bytes go to `<target>.tmp-<pid>-<rand>`
in the target's directory, are fsynced (the directory too), and `os.replace` puts them in place. A crashed write
leaves only a `*.tmp-*` entry, which `remove_stale_temps` deletes on the next start.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

TEMP_MARK = ".tmp-"
FORMAT_PATTERN = re.compile(r"bilingual-([a-z][a-z0-9-]*)/(\d+)")


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


def check_format(doc: Mapping[str, Any], kind: str, majors: tuple[int, ...] = (1,)) -> int:
    """The major version of a `bilingual-<kind>/<major>` document; refuses another kind or an unknown major."""
    match = FORMAT_PATTERN.fullmatch(str(doc.get("format", "")))
    if not match or match.group(1) != kind:
        raise ValueError(f"expected a bilingual-{kind} document, found format {doc.get('format')!r}")
    major = int(match.group(2))
    if major not in majors:
        raise ValueError(f"bilingual-{kind}/{major} is not a known version (known: {list(majors)})")
    return major


def _temp_name(target: Path) -> Path:
    return target.with_name(f"{target.name}{TEMP_MARK}{os.getpid()}-{secrets.token_hex(4)}")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(target: Path, data: bytes) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = _temp_name(target)
    try:
        with open(temp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    _fsync_dir(target.parent)
    return target


def atomic_write_json(target: Path, obj: Any, *, indent: int | None = 1) -> Path:
    """Human-readable JSON (sorted keys, no NaN) written atomically; hashes are taken over canonical_json, not
    over these bytes."""
    text = json.dumps(obj, sort_keys=True, indent=indent, ensure_ascii=False, allow_nan=False) + "\n"
    return atomic_write_bytes(target, text.encode("utf-8"))


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_file():
            with open(path, "rb") as f:
                os.fsync(f.fileno())
        else:
            _fsync_dir(path)
    _fsync_dir(root)


@contextlib.contextmanager
def atomic_directory(target: Path) -> Iterator[Path]:
    """Yield an empty temporary directory next to `target`; on a clean exit it is fsynced and renamed onto
    `target`, which must not exist yet (directories written this way are immutable). On an error it is removed."""
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"{target} exists; directories written atomically are never overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = _temp_name(target)
    temp.mkdir()
    try:
        yield temp
        _fsync_tree(temp)
        os.replace(temp, target)
        _fsync_dir(target.parent)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def remove_stale_temps(directory: Path) -> list[Path]:
    """Delete the `*.tmp-*` leftovers of crashed atomic writes directly inside `directory`; return what went."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    removed = []
    for path in sorted(directory.iterdir()):
        if TEMP_MARK in path.name:
            shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
            removed.append(path)
    return removed
