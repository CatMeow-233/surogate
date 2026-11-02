"""Document-level splits (`bilingual-split/1`) and the per-split corpus files (SCI-10).

The seven sets of MS §6.1 are pairwise disjoint: D_seed, D_zh, D_en, D_parallel, D_calib, D_dev and D_test, named
`seed_train`, `zh_train`, `en_train`, `parallel_train`, `calib`, `dev` and `test`. The unit is a cluster, not a
line: documents that share any of the following keys are merged (union-find) and always land in one split:

- the sha256 of the text after NFKC, casefold and whitespace collapse (near-duplicates);
- `pair_id` (both sides of a parallel pair);
- `entity_id` (every synthetic document about one entity).

Assignment. Within each kind, clusters are ordered by sha256(seed, cluster key) and dealt out in that order to fill
the splits' quotas (fraction x the kind's document count). Every split therefore lands within one cluster's size of
its target, and the same seed gives a byte-identical split file. [建议] this trades the per-document stability of
hash bucketing (a document's split depending on its key alone) for exact fractions: a changed corpus is a new
manifest, and its split is recomputed from (manifest, seed).

`write_split_corpora` writes each split's documents to `<out>/<split>/<source>.jsonl` and a derived manifest whose
files carry their split; decontamination removals are applied there.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..canonical import atomic_write_bytes, atomic_write_json, canonical_json, check_format, sha256_bytes, sha256_json
from ..config import DEFAULTS
from .docs import Doc, doc_line
from .manifest import FORMAT as MANIFEST_FORMAT
from .manifest import Manifest, file_entry, finish, load_manifest

FORMAT = "bilingual-split/1"
SPLITS = ("seed_train", "zh_train", "en_train", "parallel_train", "calib", "dev", "test")
TRAIN_SPLITS = ("seed_train", "zh_train", "en_train", "parallel_train")
HELDOUT_SPLITS = ("calib", "dev", "test")
CLUSTER_RULE = "union-find over sha256(nfkc+casefold+collapsed-whitespace text), pair_id, entity_id"


class SplitError(ValueError):
    pass


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def cluster_keys(doc: Doc) -> list[str]:
    keys = ["text:" + sha256_bytes(normalized_text(doc.text).encode("utf-8"))]
    if doc.pair_id is not None:
        keys.append(f"pair:{doc.pair_id}")
    if doc.entity_id is not None:
        keys.append(f"entity:{doc.entity_id}")
    return keys


def clusters(docs: Sequence[Doc]) -> list[list[int]]:
    """Connected components of documents sharing a cluster key, each sorted, ordered by first member."""
    parent = list(range(len(docs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[str, int] = {}
    for i, doc in enumerate(docs):
        for key in cluster_keys(doc):
            if key in owner:
                a, b = find(i), find(owner[key])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                owner[key] = i
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(docs)):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def _fractions_by_kind(fractions: Mapping[str, Any], kinds: Iterable[str]) -> dict[str, dict[str, float]]:
    flat = all(isinstance(v, (int, float)) for v in fractions.values())
    table = {kind: dict(fractions) for kind in kinds} if flat else {k: dict(v) for k, v in fractions.items()}
    for kind in kinds:
        if kind not in table:
            raise SplitError(f"no split fractions for documents of kind {kind!r}")
    for kind, fr in table.items():
        unknown = set(fr) - set(SPLITS)
        if unknown:
            raise SplitError(f"{kind}: unknown splits {sorted(unknown)}; the splits are {SPLITS}")
        if any(v < 0 for v in fr.values()) or not math.isclose(sum(fr.values()), 1.0, abs_tol=1e-9):
            raise SplitError(f"{kind}: split fractions must be non-negative and sum to 1, got {fr}")
    return {k: {s: float(v) for s, v in fr.items()} for k, fr in table.items()}


def assign_splits(docs: Iterable[Doc], fractions: Mapping[str, Any], seed: int) -> dict[str, list[str]]:
    """{split: sorted doc keys} for every split in SPLITS (empty lists included). `fractions` is either
    {split: fraction} for every document, or {kind: {split: fraction}}."""
    docs = list(docs)
    keys = [d.key for d in docs]
    if len(set(keys)) != len(keys):
        raise SplitError("document keys must be unique")
    table = _fractions_by_kind(fractions, {d.kind for d in docs})
    by_kind: dict[str, list[list[int]]] = defaultdict(list)
    for group in clusters(docs):
        kinds = {docs[i].kind for i in group}
        kind = docs[group[0]].kind
        allowed = {s for s, f in table[kind].items() if f > 0}
        for other in kinds - {kind}:
            if not allowed <= {s for s, f in table[other].items() if f > 0}:
                raise SplitError(f"a cluster mixes kinds {sorted(kinds)} whose allowed splits differ: "
                                 f"{[docs[i].key for i in group]}")  # fmt: skip
        by_kind[kind].append(group)
    out: dict[str, list[str]] = {s: [] for s in SPLITS}
    for kind, groups in sorted(by_kind.items()):
        order = sorted(groups, key=lambda g: sha256_json([seed, min(keys[i] for i in g)]))
        total = sum(len(g) for g in groups)
        names = [s for s in SPLITS if table[kind].get(s, 0.0) > 0]
        bounds, acc = [], 0.0
        for name in names:
            acc += table[kind][name]
            bounds.append(acc * total)
        filled, k = 0, 0
        for group in order:
            while k < len(names) - 1 and filled >= bounds[k]:
                k += 1
            # a cluster goes where its first document falls; a big cluster can overrun the bound by its size
            out[names[k]].extend(keys[i] for i in group)
            filled += len(group)
    return {s: sorted(v) for s, v in out.items()}


def split_doc(docs: Iterable[Doc], fractions: Mapping[str, Any], seed: int, manifest_sha256: str | None) -> dict:
    docs = list(docs)
    splits = assign_splits(docs, fractions, seed)
    return {
        "format": FORMAT,
        "seed": int(seed),
        "fractions": json.loads(canonical_json(fractions)),
        "cluster_rule": CLUSTER_RULE,
        "manifest_sha256": manifest_sha256,
        "counts": {s: len(v) for s, v in splits.items()},
        "splits": splits,
    }


def split_sha256(doc: Mapping[str, Any]) -> str:
    return sha256_json(doc)


def write_split(out: Path, doc: Mapping[str, Any]) -> Path:
    check_disjoint(doc["splits"])
    return atomic_write_json(Path(out), doc)


def load_split(path: Path) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    check_format(doc, "split")
    check_disjoint(doc["splits"])
    return doc


def check_disjoint(splits: Mapping[str, Sequence[str]]) -> None:
    seen: dict[str, str] = {}
    for name, keys in splits.items():
        for key in keys:
            if key in seen and seen[key] != name:
                raise SplitError(f"document {key!r} is in both {seen[key]} and {name}")
            seen[key] = name


def default_fractions() -> dict[str, dict[str, float]]:
    return {k: dict(v) for k, v in DEFAULTS["data"]["split_fractions"].items()}


def write_split_corpora(manifest: Manifest, split: Mapping[str, Any], out: Path, exclude: Iterable[str] = ()) -> Path:
    """Write every split's documents to `<out>/<split>/<source>.jsonl` (canonical JSON lines, in manifest order) and
    `<out>/manifest.json`, a derived manifest whose files carry their split. Documents in `exclude` (keys removed by
    decontamination) are left out of the training splits only; held-out splits are never filtered."""
    if split.get("manifest_sha256") not in (None, manifest.sha256):
        raise SplitError("the split file was computed from another manifest")
    exclude = set(exclude)
    where = {key: name for name, keys in split["splits"].items() for key in keys}
    lines: dict[tuple[str, str], list[bytes]] = defaultdict(list)
    dropped = 0
    for doc in manifest.iter_docs():
        name = where.get(doc.key)
        if name is None:
            raise SplitError(f"document {doc.key!r} is in the manifest but in no split")
        if doc.key in exclude and name in TRAIN_SPLITS:
            dropped += 1
            continue
        lines[(name, doc.source)].append(doc_line(doc))
    out = Path(out)
    files = []
    for (name, source), chunk in sorted(lines.items()):
        path = out / name / f"{source}.jsonl"
        atomic_write_bytes(path, b"".join(chunk))
        files.append(file_entry(path, out, {"name": source, "kind": manifest.sources[source]["kind"]}, name))
    derived = finish(
        {
            "format": MANIFEST_FORMAT,
            "sources": list(manifest.doc["sources"]),
            "files": files,
            "derived_from": {
                "manifest_sha256": manifest.sha256,
                "split_sha256": split_sha256(split),
                "excluded": sorted(exclude & set(where)),
                "dropped_from_training": dropped,
            },
        }
    )
    atomic_write_json(out / "manifest.json", {**derived, "root": "."})
    return out / "manifest.json"


def _load_fractions(path: Path | None) -> dict[str, Any]:
    if path is None:
        return default_fractions()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data.get("split_fractions", data)


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual split", description="document-level splits")
    sub = ap.add_subparsers(dest="action", required=True)
    a = sub.add_parser("assign", help="assign every document of a manifest to one of the seven splits")
    a.add_argument("--manifest", required=True, type=Path)
    a.add_argument("--seed", required=True, type=int)
    a.add_argument("--fractions", type=Path, help="TOML {kind: {split: fraction}}; default DEFAULTS[data]")
    a.add_argument("--out", type=Path)
    a.add_argument("--dry-run", action="store_true")
    m = sub.add_parser("materialize", help="write per-split corpus files and a derived manifest")
    m.add_argument("--manifest", required=True, type=Path)
    m.add_argument("--split", required=True, type=Path, help="bilingual-split/1 file")
    m.add_argument("--exclude", type=Path, help="bilingual-overlap/1 report whose flagged documents are dropped")
    m.add_argument("--out", type=Path)
    m.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(list(argv))
    if not args.dry_run and args.out is None:
        ap.error("--out is required unless --dry-run")
    manifest = load_manifest(args.manifest)
    if args.action == "assign":
        doc = split_doc(manifest.iter_docs(), _load_fractions(args.fractions), args.seed, manifest.sha256)
        if not args.dry_run:
            write_split(args.out, doc)
        print(
            "split: " + ", ".join(f"{k} {v}" for k, v in doc["counts"].items()) + (" (dry run)" if args.dry_run else "")
        )
        return 0
    split = load_split(args.split)
    exclude = []
    if args.exclude is not None:
        report = json.loads(args.exclude.read_text(encoding="utf-8"))
        check_format(report, "overlap")
        exclude = list(report["flagged"])
    if args.dry_run:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            write_split_corpora(manifest, split, Path(tmp), exclude)
        print("split: materialize checked (dry run)")
        return 0
    path = write_split_corpora(manifest, split, args.out, exclude)
    print(f"split: wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
