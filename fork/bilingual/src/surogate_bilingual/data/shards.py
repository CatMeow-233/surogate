"""Shards (`bilingual-shard/1`; IMPL-07, IMPL-03): packed tokens with document offsets, per-token routes and the
loss mask, under a header that pins the tokenizer, the manifest and the split.

```
<shard>/
  shard.json        header
  tokens.bin        little-endian uint16 if the vocabulary has <= 65535 ids, else uint32
  doc_offsets.bin   little-endian uint64, n_docs + 1 offsets into tokens
  routes.bin        uint8 per token: index into route_names; 255 = unset
  loss_mask.bin     uint8 per token: 1 iff the token is a prediction target (logits at t-1 predict tokens[t])
```

Encoding a document: a text segment is encoded as user text (a literal "<|en|>" stays text), a tag segment is
its special id, and `<eos>` ends every document. A token's route is its segment's route, else the document's, else
the shard's default, else unset; `routes[t]` is the expert that computes position t. The loss flag works the same
way; `<eos>` takes the route and the loss flag of the document's last segment.

The header also records `tag_counts` (occurrences of each control-tag id) and `mixed_window_fraction`: the share
of all length-`seq` windows (every start position, which is what a uniformly random window start sees) that hold
at least one zh token and at least one en token, by segment language. v1.0 packed all of zh.txt before all of
en.txt and got 2.6e-7 at seq 512 (SCI-02); a declared floor can make the writer refuse a shard below it, and a
declared minimum per tag can make it refuse a shard with too few tags.

`ShardReader` maps the files read-only and refuses a shard whose header or file hashes do not match, whose
tokenizer is not the run's, or whose dtype does not fit the run's vocabulary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..canonical import atomic_directory, atomic_write_json, check_format, sha256_file
from ..config import DEFAULTS
from .docs import CONTROL_TAGS, Doc

FORMAT = "bilingual-shard/1"
FILES = ("tokens.bin", "doc_offsets.bin", "routes.bin", "loss_mask.bin")
UNSET_ROUTE = 255
LANG_CODE = {None: 0, "zh": 1, "en": 2}


class ShardError(ValueError):
    pass


def token_dtype(vocab_size: int) -> str:
    return "uint16" if vocab_size <= 65535 else "uint32"


def _np_dtype(name: str) -> np.dtype:
    return {"uint16": np.dtype("<u2"), "uint32": np.dtype("<u4")}[name]


def encode_doc(doc: Doc, tok, route_names: Sequence[str], default_route: str | None = None):
    """(ids, routes, loss, langs) of one document, `<eos>` included."""
    from .tokenizer import encode_user_text

    index = {name: i for i, name in enumerate(route_names)}
    ids: list[int] = []
    routes: list[int] = []
    loss: list[int] = []
    langs: list[int] = []

    def route_id(name: str | None) -> int:
        name = name or doc.route or default_route
        if name is None:
            return UNSET_ROUTE
        if name not in index:
            raise ShardError(f"{doc.key}: route {name!r} is not one of {tuple(route_names)}")
        return index[name]

    last_route, last_loss = route_id(None), 1
    for seg in doc.segments:
        piece = [tok.token_to_id(seg.tag)] if seg.tag is not None else encode_user_text(tok, seg.text)
        r = route_id(seg.route)
        ids.extend(piece)
        routes.extend([r] * len(piece))
        loss.extend([int(seg.loss)] * len(piece))
        langs.extend([LANG_CODE[seg.lang if seg.tag is None else None]] * len(piece))
        last_route, last_loss = r, int(seg.loss)
    ids.append(tok.token_to_id("<eos>"))
    routes.append(last_route)
    loss.append(last_loss)
    langs.append(0)
    return ids, routes, loss, langs


def mixed_window_fraction(langs: np.ndarray, seq: int, chunk: int = 1 << 24) -> float | None:
    """Share of the windows [s, s + seq), s = 0 .. n - seq, holding both a zh (1) and an en (2) token. Computed
    from the end in chunks with the next-occurrence index of each language, so memory stays O(chunk)."""
    n = len(langs)
    if n < seq:
        return None
    big = np.int64(np.iinfo(np.int64).max // 2)
    next_zh, next_en = big, big
    mixed = 0
    starts_end = n - seq + 1  # starts s < starts_end
    for hi in range(n, 0, -chunk):
        lo = max(0, hi - chunk)
        block = np.asarray(langs[lo:hi])
        pos = np.arange(lo, hi, dtype=np.int64)
        nz = np.where(block == 1, pos, big)[::-1]
        ne = np.where(block == 2, pos, big)[::-1]
        nz = np.minimum.accumulate(np.minimum(nz, next_zh))[::-1]
        ne = np.minimum.accumulate(np.minimum(ne, next_en))[::-1]
        next_zh, next_en = nz[0], ne[0]
        valid = pos < starts_end
        mixed += int(np.count_nonzero(valid & (nz < pos + seq) & (ne < pos + seq)))
    return mixed / starts_end


def write_shard(
    out: Path,
    docs: Iterable[Doc],
    tok,
    route_names: Sequence[str],
    split: str,
    manifest_sha256: str | None,
    *,
    tokenizer_sha256: str,
    split_sha256: str | None = None,
    default_route: str | None = None,
    seq: int | None = None,
    min_tag_counts: Mapping[str, int] | None = None,
    min_mixed_window_fraction: float | None = None,
    inputs: Sequence[Mapping[str, Any]] = (),
    extra: Mapping[str, Any] | None = None,
) -> Path:
    route_names = list(route_names)
    if len(route_names) >= UNSET_ROUTE or len(set(route_names)) != len(route_names):
        raise ShardError("route_names must be unique and fewer than 255")
    seq = int(seq or DEFAULTS["train"]["seq"])
    vocab = tok.get_vocab_size()
    dtype = token_dtype(vocab)
    tag_ids = {tag: tok.token_to_id(tag) for tag in CONTROL_TAGS}
    out = Path(out)
    n_tokens, offsets, lang_parts = 0, [0], []
    tag_counts = {tag: 0 for tag in CONTROL_TAGS}
    with atomic_directory(out) as tmp:
        handles = {name: open(tmp / name, "wb") for name in FILES}
        try:
            for doc in docs:
                ids, routes, loss, langs = encode_doc(doc, tok, route_names, default_route)
                arr = np.asarray(ids, dtype=np.int64)
                if arr.max() >= vocab or arr.min() < 0:
                    raise ShardError(f"{doc.key}: token id outside the vocabulary")
                for tag, tid in tag_ids.items():
                    tag_counts[tag] += int(np.count_nonzero(arr == tid))
                arr.astype(_np_dtype(dtype)).tofile(handles["tokens.bin"])
                np.asarray(routes, dtype=np.uint8).tofile(handles["routes.bin"])
                np.asarray(loss, dtype=np.uint8).tofile(handles["loss_mask.bin"])
                lang_parts.append(np.asarray(langs, dtype=np.uint8))
                n_tokens += len(ids)
                offsets.append(n_tokens)
            np.asarray(offsets, dtype="<u8").tofile(handles["doc_offsets.bin"])
        finally:
            for handle in handles.values():
                handle.close()
        if n_tokens == 0:
            raise ShardError("a shard needs at least one document")
        langs_all = np.concatenate(lang_parts)
        mixed = mixed_window_fraction(langs_all, seq)
        for tag, minimum in (min_tag_counts or {}).items():
            if tag not in tag_counts:
                raise ShardError(f"unknown tag {tag!r} in min_tag_counts")
            if tag_counts[tag] < minimum:
                raise ShardError(f"tag {tag} occurs {tag_counts[tag]} times, below the declared minimum {minimum}")
        if min_mixed_window_fraction is not None and (mixed is None or mixed < min_mixed_window_fraction):
            raise ShardError(
                f"mixed-window fraction {mixed} at seq {seq} is below the declared floor {min_mixed_window_fraction}"
            )
        header = {
            "format": FORMAT,
            "split": split,
            "dtype": dtype,
            "vocab_size": vocab,
            "tokenizer_sha256": tokenizer_sha256,
            "manifest_sha256": manifest_sha256,
            "split_sha256": split_sha256,
            "route_names": route_names,
            "n_tokens": n_tokens,
            "n_docs": len(offsets) - 1,
            "files": {name: sha256_file(tmp / name) for name in FILES},
            "tag_counts": tag_counts,
            "min_tag_counts": dict(min_tag_counts or {}),
            "mixed_window_fraction": {"seq": seq, "value": mixed, "floor": min_mixed_window_fraction},
            "eos_between_docs": True,
            "inputs": [dict(i) for i in inputs],
            "extra": dict(extra or {}),
        }
        atomic_write_json(tmp / "shard.json", header)
    return out


class ShardReader:
    """A shard opened read-only. `tokenizer_sha256` and `vocab_size` are the run's; pass None only to inspect a
    shard outside a run (the check is then skipped and says so in `checks`)."""

    def __init__(self, path: Path, *, tokenizer_sha256: str | None, vocab_size: int | None, verify: bool = True):
        self.path = Path(path)
        self.header = json.loads((self.path / "shard.json").read_text(encoding="utf-8"))
        check_format(self.header, "shard")
        h = self.header
        self.checks = {"files": verify, "tokenizer": tokenizer_sha256 is not None, "vocab": vocab_size is not None}
        if tokenizer_sha256 is not None and h["tokenizer_sha256"] != tokenizer_sha256:
            raise ShardError(
                f"{self.path}: built with tokenizer {h['tokenizer_sha256'][:12]}…, "
                f"the run uses {tokenizer_sha256[:12]}…"
            )
        if h["dtype"] not in ("uint16", "uint32"):
            raise ShardError(f"{self.path}: unknown token dtype {h['dtype']!r}")
        if vocab_size is not None and h["dtype"] != token_dtype(vocab_size):
            raise ShardError(
                f"{self.path}: dtype {h['dtype']} does not fit a vocabulary of {vocab_size} ids "
                f"(expected {token_dtype(vocab_size)})"
            )
        if verify:
            for name in FILES:
                if sha256_file(self.path / name) != h["files"][name]:
                    raise ShardError(f"{self.path / name}: sha256 differs from the shard header")
        n = h["n_tokens"]
        self.tokens = np.memmap(self.path / "tokens.bin", dtype=_np_dtype(h["dtype"]), mode="r")
        self.doc_offsets = np.memmap(self.path / "doc_offsets.bin", dtype="<u8", mode="r")
        self.routes = np.memmap(self.path / "routes.bin", dtype=np.uint8, mode="r")
        self.loss_mask = np.memmap(self.path / "loss_mask.bin", dtype=np.uint8, mode="r")
        sizes = (len(self.tokens), len(self.routes), len(self.loss_mask), len(self.doc_offsets))
        if sizes != (n, n, n, h["n_docs"] + 1) or int(self.doc_offsets[-1]) != n:
            raise ShardError(f"{self.path}: file sizes {sizes} do not match the header")
        if vocab_size is not None and verify and int(self.tokens.max()) >= vocab_size:
            raise ShardError(f"{self.path}: token ids exceed the vocabulary of {vocab_size}")

    @property
    def route_names(self) -> list[str]:
        return list(self.header["route_names"])

    def __len__(self) -> int:
        return int(self.header["n_tokens"])

    def windows(self, starts: Sequence[int], seq: int) -> dict[str, np.ndarray]:
        """Training windows sliced in lockstep: inputs tokens[s : s+seq], targets tokens[s+1 : s+seq+1], the
        routes of the input positions, and the loss mask of the target positions."""
        starts = np.asarray(starts, dtype=np.int64)
        if len(starts) and (starts.min() < 0 or starts.max() + seq + 1 > len(self)):
            raise ShardError(f"a window start lies outside 0..{len(self) - seq - 1}")
        idx = starts[:, None] + np.arange(seq + 1, dtype=np.int64)[None, :]
        tokens = np.asarray(self.tokens[idx], dtype=np.int64)
        return {
            "inputs": tokens[:, :-1],
            "targets": tokens[:, 1:],
            "routes": np.asarray(self.routes[idx[:, :-1]], dtype=np.int64),
            "loss_mask": np.asarray(self.loss_mask[idx[:, 1:]], dtype=np.uint8),
        }


def main(argv: Sequence[str]) -> int:
    from .manifest import load_manifest
    from .tokenizer import load_tokenizer, tokenizer_sha256

    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual shard", description="write a shard")
    ap.add_argument("--manifest", required=True, type=Path, help="a derived manifest whose files carry their split")
    ap.add_argument(
        "--split",
        required=True,
        help="a split, or several joined by commas (zh_train,en_train for the D_zh u D_en shard of B2 and B12a)",
    )
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--route-names", default="zh,en,bridge,seed")
    ap.add_argument("--route", help="default route of every token (e.g. seed for D_seed, zh for D_zh)")
    ap.add_argument("--add", nargs="*", type=Path, default=[], help="extra document files (e.g. Track S components)")
    ap.add_argument("--exclude", type=Path, help="bilingual-overlap/1 report; its flagged documents are skipped")
    ap.add_argument("--seed-mix", action="store_true", help="build the D_seed mixture (data.mixture) first")
    ap.add_argument("--mix-seed", type=int, default=0)
    ap.add_argument(
        "--bridge",
        action="store_true",
        help="parallel documents in the Bridge layout (data.parallel): source on its language's route "
        "with loss 0, target on the bridge route; for T2, T3' and T1'",
    )
    ap.add_argument("--direction", choices=["zh>en", "en>zh"], help="with --bridge: keep one direction (T1')")
    ap.add_argument("--unpair-seed", type=int, help="with --bridge: break every pair with this seed (T3')")
    ap.add_argument("--seq", type=int, default=DEFAULTS["train"]["seq"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="read and encode everything; write into a temp dir")
    args = ap.parse_args(list(argv))
    if not args.dry_run and args.out is None:
        ap.error("--out is required unless --dry-run")
    if (args.direction or args.unpair_seed is not None) and not args.bridge:
        ap.error("--direction and --unpair-seed apply to --bridge shards")
    if args.bridge and (args.seed_mix or args.route):
        ap.error("--bridge sets every route itself; it takes neither --seed-mix nor --route")
    try:
        return _run(
            args, load_manifest(args.manifest), load_tokenizer(args.tokenizer), tokenizer_sha256(args.tokenizer)
        )
    except ValueError as err:  # ManifestError, DocError, ShardError, MixtureError, TokenizerError
        print(f"shard: {err}", file=sys.stderr)
        return 1


def _run(args, manifest, tok, tok_sha: str) -> int:
    from .docs import iter_file_docs

    splits = [part for part in args.split.split(",") if part]
    if not splits or len(set(splits)) != len(splits):
        raise ShardError(f"--split names one split or several distinct ones: {args.split!r}")
    docs = [doc for split in splits for doc in manifest.iter_docs(split=split)]
    inputs = [{"path": e["path"], "sha256": e["sha256"]} for split in splits for e in manifest.files(split=split)]
    for path in args.add:
        docs.extend(iter_file_docs(path, source=None, kind=None))
        inputs.append({"path": str(path), "sha256": sha256_file(path)})
    if args.exclude is not None:
        from .decontam import load_report

        flagged = load_report(args.exclude)["flagged"]
        docs = [d for d in docs if d.key not in flagged]
    kwargs: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    if args.bridge:
        from .parallel import bridge_layout, select_direction, unpair

        if args.direction:
            docs = select_direction(docs, args.direction)
        if args.unpair_seed is not None:
            docs = unpair(docs, args.unpair_seed)
        docs = [bridge_layout(d) for d in docs]
        extra["bridge_layout"] = {"direction": args.direction, "unpair_seed": args.unpair_seed}
    if args.seed_mix:
        from .mixture import build_seed_mixture, pools_from_docs

        docs, card = build_seed_mixture(pools_from_docs(docs), tok, seed=args.mix_seed, sources=manifest.doc["sources"])
        kwargs["min_tag_counts"] = card["min_tag_counts"]
        kwargs["min_mixed_window_fraction"] = DEFAULTS["data"]["min_mixed_window_fraction"]
        extra["seed_mix"] = card
    derived = manifest.doc.get("derived_from") or {}
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "shard" if args.dry_run else args.out
        write_shard(
            target,
            docs,
            tok,
            args.route_names.split(","),
            "+".join(splits),
            manifest.sha256,
            tokenizer_sha256=tok_sha,
            split_sha256=derived.get("split_sha256"),
            default_route=args.route,
            seq=args.seq,
            inputs=inputs,
            extra=extra,
            **kwargs,
        )
        header = json.loads((target / "shard.json").read_text(encoding="utf-8"))
    print(
        f"shard: {header['n_docs']} documents, {header['n_tokens']} tokens, tags {header['tag_counts']}, "
        f"mixed-window fraction {header['mixed_window_fraction']['value']}"
        + (" (dry run)" if args.dry_run else f" -> {args.out}")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
