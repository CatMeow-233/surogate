"""The corpus manifest (`bilingual-manifest/1`) built from a TOML source spec (docs/spec.md, "语料 manifest 与划分").

A spec has one `[[source]]` table per source: `name`, `kind` (zh | en | parallel | codeswitch | synthetic),
`files` (globs relative to the corpus root), `url`, `revision`, `licence` (an SPDX-style id; required),
`licence_verified` (default false), `downloaded` (a date) and `filters`. The manifest records each source and, per
file, its path relative to the root, sha256, byte count and document count; files are listed in path order, so the
manifest does not depend on how the file system lists them. `manifest_sha256` is the sha256 of the canonical JSON
of the manifest without that field, so it is stable across runs. A file may also carry the split it holds
(`split`), which the split materializer sets (data.split); the tokenizer trains only on files of `seed_train`.

A licence that is not verified is never silently accepted: `unverified_licences` lists those sources, and every
data card copies the list (MS §6.1: unverified subsets are not redistributed).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..canonical import atomic_write_json, check_format, sha256_file, sha256_json
from .docs import CORPUS_SUFFIXES, KINDS, Doc, DocError, iter_file_docs

FORMAT = "bilingual-manifest/1"
SOURCE_FIELDS = ("name", "kind", "files", "url", "revision", "licence", "licence_verified", "downloaded", "filters")
SOURCE_NAME = re.compile(r"[a-z][a-z0-9_-]*")
LICENCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]*(-only|-or-later)?|LicenseRef-[A-Za-z0-9.-]+")


class ManifestError(ValueError):
    pass


def _source(entry: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(entry) - set(SOURCE_FIELDS)
    if unknown:
        raise ManifestError(f"source {entry.get('name')!r}: unknown fields {sorted(unknown)}")
    name = entry.get("name")
    if not isinstance(name, str) or not SOURCE_NAME.fullmatch(name):
        raise ManifestError(f"source name must match {SOURCE_NAME.pattern}: {name!r}")
    if entry.get("kind") not in KINDS:
        raise ManifestError(f"source {name}: kind must be one of {KINDS}: {entry.get('kind')!r}")
    licence = entry.get("licence")
    if not isinstance(licence, str) or not licence.strip():
        raise ManifestError(f"source {name}: `licence` is required (an SPDX-style id such as CC-BY-SA-4.0)")
    if not LICENCE_ID.fullmatch(licence):
        raise ManifestError(f"source {name}: licence {licence!r} is not an SPDX-style id")
    verified = entry.get("licence_verified", False)
    if not isinstance(verified, bool):
        raise ManifestError(f"source {name}: licence_verified must be true or false")
    files = entry.get("files")
    if isinstance(files, str):
        files = [files]
    if not files or not all(isinstance(g, str) and g for g in files):
        raise ManifestError(f"source {name}: `files` must list one or more globs")
    downloaded = entry.get("downloaded")
    return {
        "name": name,
        "kind": entry["kind"],
        "files": list(files),
        "url": entry.get("url"),
        "revision": entry.get("revision"),
        "licence": licence,
        "licence_verified": verified,
        "downloaded": None if downloaded is None else str(downloaded),
        "filters": entry.get("filters"),
    }


def load_spec(spec: Path) -> list[dict[str, Any]]:
    with open(spec, "rb") as f:
        data = tomllib.load(f)
    unknown = set(data) - {"source"}
    if unknown:
        raise ManifestError(f"{spec}: unknown top-level keys {sorted(unknown)}; a spec holds [[source]] tables")
    sources = [_source(entry) for entry in data.get("source", [])]
    if not sources:
        raise ManifestError(f"{spec}: no [[source]] tables")
    names = [s["name"] for s in sources]
    if len(set(names)) != len(names):
        raise ManifestError(f"{spec}: duplicate source names")
    return sources


def _match(root: Path, globs: Sequence[str]) -> list[Path]:
    found = set()
    for pattern in globs:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ManifestError(f"file globs are relative to the corpus root: {pattern!r}")
        found.update(p for p in root.glob(pattern) if p.is_file())
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())


def file_entry(path: Path, root: Path, source: Mapping[str, Any], split: str | None = None) -> dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    if path.suffix not in CORPUS_SUFFIXES:
        raise ManifestError(f"{rel}: corpus files are {CORPUS_SUFFIXES}")
    n_docs = sum(1 for _ in iter_file_docs(path, source=source["name"], kind=source["kind"], rel=rel))
    return {
        "path": rel,
        "source": source["name"],
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "n_docs": n_docs,
        "split": split,
    }


def finish(doc: dict[str, Any]) -> dict[str, Any]:
    """Sort the file list, check paths and keys are unique, and set manifest_sha256."""
    doc["files"] = sorted(doc["files"], key=lambda f: f["path"])
    paths = [f["path"] for f in doc["files"]]
    if len(set(paths)) != len(paths):
        raise ManifestError("a file is matched by two sources")
    doc["unverified_licences"] = sorted(s["name"] for s in doc["sources"] if not s["licence_verified"])
    body = {k: v for k, v in doc.items() if k != "manifest_sha256"}
    doc["manifest_sha256"] = sha256_json(body)
    return doc


def manifest_doc(spec: Path, root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    sources = load_spec(spec)
    files = []
    for source in sources:
        matched = _match(root, source["files"])
        if not matched:
            raise ManifestError(f"source {source['name']}: no file matches {source['files']} under {root}")
        files.extend(file_entry(p, root, source) for p in matched)
    return finish({"format": FORMAT, "sources": sources, "files": files, "derived_from": None})


def _with_root(doc: Mapping[str, Any], root: Path, out: Path) -> dict[str, Any]:
    """The manifest as written: `root` (outside the hash) is stored relative to the manifest's directory."""
    return {**doc, "root": os.path.relpath(Path(root).resolve(), Path(out).resolve().parent)}


def build_manifest(spec: Path, root: Path, out: Path) -> Path:
    """Hash and count every file of the corpus under `root` that `spec` describes and write the manifest to `out`.
    File paths are relative to `root`; refuses duplicate document keys before anything is written."""
    doc = manifest_doc(spec, root)
    doc_keys(load_manifest_doc(doc, Path(root).resolve()))
    atomic_write_json(Path(out), _with_root(doc, root, out))
    return Path(out)


class Manifest:
    """A loaded manifest: the JSON document plus the root its paths are relative to."""

    def __init__(self, doc: Mapping[str, Any], root: Path):
        check_format(doc, "manifest")
        body = {k: v for k, v in doc.items() if k not in ("manifest_sha256", "root")}
        if sha256_json(body) != doc.get("manifest_sha256"):
            raise ManifestError("manifest_sha256 does not match the manifest's content")
        self.doc = dict(doc)
        self.root = Path(root)
        self.sources = {s["name"]: s for s in doc["sources"]}

    @property
    def sha256(self) -> str:
        return self.doc["manifest_sha256"]

    def files(self, *, split: str | None = None, kinds: Sequence[str] | None = None) -> list[dict[str, Any]]:
        out = []
        for entry in self.doc["files"]:
            if split is not None and entry.get("split") != split:
                continue
            if kinds is not None and self.sources[entry["source"]]["kind"] not in kinds:
                continue
            out.append(entry)
        return out

    def verify_file(self, entry: Mapping[str, Any]) -> Path:
        path = self.root / entry["path"]
        if not path.is_file():
            raise ManifestError(f"{entry['path']}: listed in the manifest but missing under {self.root}")
        if sha256_file(path) != entry["sha256"]:
            raise ManifestError(f"{entry['path']}: sha256 differs from the manifest")
        return path

    def entry_for(self, path: Path) -> dict[str, Any] | None:
        try:
            rel = Path(path).resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return None
        return next((f for f in self.doc["files"] if f["path"] == rel), None)

    def iter_docs(self, *, split: str | None = None, kinds: Sequence[str] | None = None, verify: bool = True):
        for entry in self.files(split=split, kinds=kinds):
            path = self.verify_file(entry) if verify else self.root / entry["path"]
            source = self.sources[entry["source"]]
            yield from iter_file_docs(path, source=source["name"], kind=source["kind"], rel=entry["path"])


def load_manifest_doc(doc: Mapping[str, Any], root: Path) -> Manifest:
    return Manifest(doc, root)


def load_manifest(path: Path, root: Path | None = None) -> Manifest:
    """Read a manifest; its file paths resolve against `root`, else the `root` it records (relative to the
    manifest's directory), else that directory."""
    import json

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    here = Path(path).resolve().parent
    base = Path(root) if root is not None else (here / doc["root"]).resolve() if doc.get("root") else here
    return Manifest(doc, base)


def doc_keys(manifest: Manifest) -> list[str]:
    """Every document key, in manifest order; refuses duplicates (keys address documents in split files)."""
    keys, seen = [], set()
    for doc in manifest.iter_docs():
        if doc.key in seen:
            raise ManifestError(f"duplicate document key {doc.key!r}")
        seen.add(doc.key)
        keys.append(doc.key)
    return keys


def iter_docs(manifest: Manifest, **kw) -> Iterator[Doc]:
    return manifest.iter_docs(**kw)


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual manifest", description="build the corpus manifest")
    ap.add_argument("--spec", required=True, type=Path, help="TOML with [[source]] tables")
    ap.add_argument("--root", required=True, type=Path, help="corpus root the file globs are relative to")
    ap.add_argument("--out", type=Path, help="where to write bilingual-manifest/1 JSON")
    ap.add_argument("--dry-run", action="store_true", help="hash and count every file; write nothing")
    args = ap.parse_args(list(argv))
    if not args.dry_run and args.out is None:
        ap.error("--out is required unless --dry-run")
    try:
        doc = manifest_doc(args.spec, args.root)
        doc_keys(load_manifest_doc(doc, args.root.resolve()))
    except (ManifestError, DocError) as err:
        print(f"manifest: {err}", file=sys.stderr)
        return 1
    if not args.dry_run:
        atomic_write_json(args.out, _with_root(doc, args.root, args.out))
    n_docs = sum(f["n_docs"] for f in doc["files"])
    print(f"manifest: {len(doc['sources'])} sources, {len(doc['files'])} files, {n_docs} documents, "
          f"manifest_sha256 {doc['manifest_sha256']}" + (" (dry run)" if args.dry_run else f" -> {args.out}"))  # fmt: skip
    for name in doc["unverified_licences"]:
        print(f"manifest: warning: the licence of source {name!r} is not verified (licence_verified = false)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
