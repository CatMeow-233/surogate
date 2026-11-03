"""Evaluation: exact cross-arm bits per byte on a common byte prefix (spec §5.5, §6.1-6.2; STUDY M1-M4).

**Evalset** (`tokstudy-evalset/1`, `eval prefix`). A table's held-out documents (TLC val, bilingual dev, or framed
sample files), each cut to `common_prefix_bytes`: the largest k <= K_eval text bytes, at a character boundary, such
that every arm of the table's registry can hold truncate(doc, k) in one pass (the bilingual trunk's max_seq for its
token, byte and H-Net arms, the H-Net bounded at one chunk per byte; the Mamba twin has no limit). The prefix is
computed once from the registry and stored with the documents (`docs.jsonl`) and the arms it was computed for
(id and arm sha256 of every arm). `eval score` refuses an evalset whose arm set differs from the run's current
registry (an arm added, dropped or changed: the prefix depends on every arm's codec and positional limit, so the
prefix is recomputed with `eval prefix` and the older NLL files are not mixed with the new ones; the verdict refuses
NLL files of different evalsets in one table). Bilingual documents that mix languages are left out: BPB is
compared within one language. Documents of generated sources (TLC `generated = true`, bilingual kind "synthetic")
are left out unless `--include-generated` is given; the evalset's source records which sources were excluded.
`--max-docs` takes, per domain, the documents of lowest rank sha256("<seed>\\0<key>").

**Scoring** (`eval score`, `tokstudy-nll/1`). One forward pass per document (batched with right padding; causal, so
padding never reaches a real position). A document is its segments encoded one by one (specials as their ids); the
targets are all positions after the first. Per document the line holds the text bytes, the summed NLL in nats of
the targets whose token comes from a text segment (`nll_nats`, `n_targets`), and, apart, the NLL of the `<eos>`
targets (`eos_nats`, `n_eos`, the sensitivity variant) and of the other specials (`special_nats`, `n_special`).
Under every codec the text tokens' bytes add up to the document's text bytes (checked), and the first position must
be a special, so that every text byte is a target.

    BPB(domain) = sum nll_nats / (ln 2 * sum bytes) over the domain's documents (byte-weighted)

The NLL format is neutral: the header names the backend ("tokstudy-torch"), so a native writer can produce the same
lines for its own table without the tables mixing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import registry
from .arms import ArmSpec, arm_sha256, load_arms
from .canonical import TokstudyError, atomic_write_bytes, atomic_write_json, canonical_json, read_json, sha256_file
from .codecs import ByteCodec, Codec, utf8
from .docs import (
    MIXED,
    EvalDoc,
    bilingual_documents,
    bilingual_source,
    encode_doc,
    frame_plain,
    parse_docs_arg,
    plain_documents,
    tlc_documents,
    tlc_source,
    truncate,
)
from .stream import rank_key

EVALSET_FORMAT = "tokstudy-evalset/1"
NLL_FORMAT = "tokstudy-nll/1"
EVALSET_FILE, EVALSET_DOCS = "evalset.json", "docs.jsonl"
EOS_NAMES = ("<|eos|>", "<eos>")
LN2 = math.log(2.0)


@dataclass(frozen=True)
class DocNLL:
    key: str
    domain: str
    bytes: int
    nll_nats: float
    n_targets: int
    eos_nats: float = 0.0
    n_eos: int = 0
    special_nats: float = 0.0
    n_special: int = 0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# --- the common prefix -----------------------------------------------------------------------------------------------


def arm_codec(arm: ArmSpec) -> Codec:
    from .arms import codec_for

    codec = codec_for(arm)
    return codec if codec is not None else ByteCodec(arm.specials, name=arm.id)


def arm_limits(arms: Sequence[ArmSpec]) -> dict[str, int | None]:
    from .models.build import positional_limit

    return {arm.id: positional_limit(arm) for arm in arms}


def _char_boundaries(doc: EvalDoc) -> np.ndarray:
    """Every text byte offset of `doc` that is a character boundary, ascending (0 and the total included)."""
    lengths = [len(utf8(ch)) for text in doc.texts() for ch in text]
    return np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)]) if lengths else np.zeros(1, dtype=np.int64)


def common_prefix_bytes(
    doc: EvalDoc,
    arms: Sequence[ArmSpec],
    *,
    k_eval: int,
    codecs: Mapping[str, Codec] | None = None,
    limits: Mapping[str, int | None] | None = None,
) -> int:
    """max k <= k_eval at a character boundary such that every arm's encoding of truncate(doc, k) fits its limit
    (0 when not even the empty prefix fits). Token counts need not grow with k, so the scan goes down from the top
    and the first fit is the maximum."""
    codecs = dict(codecs) if codecs is not None else {a.id: arm_codec(a) for a in arms}
    limits = dict(limits) if limits is not None else arm_limits(arms)
    bounded = [a for a in arms if limits.get(a.id) is not None]
    boundaries = _char_boundaries(doc)
    candidates = boundaries[boundaries <= k_eval][::-1]
    for k in candidates:
        cut = truncate(doc, int(k))
        if all(len(encode_doc(codecs[a.id], cut).ids) <= limits[a.id] for a in bounded):
            return int(k)
    return 0


def build_evalset(
    docs: Iterable[EvalDoc],
    arms: Sequence[ArmSpec],
    out: Path,
    *,
    k_eval: int,
    source: Mapping[str, Any],
    registry_sha256: str | None = None,
    max_docs: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Cut every document to the table's common prefix and write evalset.json and docs.jsonl into `out`."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise TokstudyError(f"{out} is not empty")
    tables = {a.table for a in arms}
    if len(tables) != 1:
        raise TokstudyError(f"an evalset serves one table, got arms of {sorted(tables)}")
    codecs = {a.id: arm_codec(a) for a in arms}
    limits = arm_limits(arms)
    chosen: dict[str, list[tuple[str, EvalDoc]]] = {}
    dropped: Counter = Counter()
    for doc in docs:
        if doc.domain == MIXED:
            dropped["mixed language"] += 1
            continue
        chosen.setdefault(doc.domain, []).append((rank_key(seed, doc.key), doc))
    lines, counts = [], {k: Counter() for k in ("docs", "bytes", "full_bytes")}
    keys = set()
    for domain in sorted(chosen):
        items = sorted(chosen[domain], key=lambda item: item[0])
        if max_docs is not None:
            dropped["over max_docs"] += max(0, len(items) - max_docs)
            items = items[:max_docs]
        for _, doc in items:
            if doc.key in keys:
                raise TokstudyError(f"document key {doc.key!r} occurs twice")
            keys.add(doc.key)
            k = common_prefix_bytes(doc, arms, k_eval=k_eval, codecs=codecs, limits=limits)
            if k <= 0:
                dropped["no prefix fits every arm"] += 1
                continue
            cut = truncate(doc, k)
            if not cut.segs or cut.segs[0].special is None:
                raise TokstudyError(f"{doc.key}: an evaluation document starts with a special (frame it)")
            lines.append(canonical_json({**cut.to_json(), "full_bytes": doc.text_bytes()}) + b"\n")
            counts["docs"][domain] += 1
            counts["bytes"][domain] += k
            counts["full_bytes"][domain] += doc.text_bytes()
    if not lines:
        raise TokstudyError("no evaluation document survives")
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out / EVALSET_DOCS, b"".join(lines))
    manifest = {
        "format": EVALSET_FORMAT,
        "table": next(iter(tables)),
        "source": dict(source),
        "k_eval": k_eval,
        "arms": [{"id": a.id, "arm_sha256": arm_sha256(a), "limit": limits[a.id]} for a in arms],
        "registry_sha256": registry_sha256,
        "seed": seed,
        "max_docs": max_docs,
        "prefix_rule": "max k <= K_eval at a character boundary that every arm holds in one pass",
        "docs_file": EVALSET_DOCS,
        "docs_sha256": sha256_file(out / EVALSET_DOCS),
        "by_domain": {d: {k: counts[k][d] for k in counts} for d in sorted(counts["docs"])},
        "dropped": dict(sorted(dropped.items())),
    }
    atomic_write_json(out / EVALSET_FILE, manifest)
    return manifest


def open_evalset(path: Path) -> tuple[dict[str, Any], str, list[EvalDoc]]:
    """(manifest, its sha256, documents); the documents file is checked against the manifest."""
    path = Path(path)
    manifest = read_json(path / EVALSET_FILE, "evalset")
    found = sha256_file(path / manifest["docs_file"])
    if found != manifest["docs_sha256"]:
        raise TokstudyError(f"{path}: {manifest['docs_file']} has sha256 {found}, the evalset says otherwise")
    docs = []
    with (path / manifest["docs_file"]).open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            record.pop("full_bytes", None)
            docs.append(EvalDoc.from_json(record))
    return manifest, sha256_file(path / EVALSET_FILE), docs


# --- scoring ---------------------------------------------------------------------------------------------------------


def score_documents(
    model, codec: Codec, docs: Sequence[EvalDoc], *, batch: int = 8, precision: str = "fp32", device=None
) -> list[DocNLL]:
    """Per-document NLLs of `docs` (already cut to their prefix) under `model`, in the order of `docs`."""
    import torch
    from torch.nn import functional as F

    from . import devices
    from .models.hnet import HNetLM

    dev = device or next(model.parameters()).device
    encoded = []
    eos_ids = {codec.special_ids[n] for n in EOS_NAMES if n in codec.special_ids}
    for doc in docs:
        enc = encode_doc(codec, doc)
        if len(enc.ids) < 2 or not enc.special[0]:
            raise TokstudyError(f"{doc.key}: needs a special first position and at least one target")
        if enc.text_bytes != doc.text_bytes():
            raise TokstudyError(
                f"{codec.name}: the tokens of {doc.key} cover {enc.text_bytes} of {doc.text_bytes()} text bytes"
            )
        encoded.append(enc)
    order = sorted(range(len(docs)), key=lambda i: len(encoded[i].ids))
    results: dict[int, DocNLL] = {}
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(order), batch):
            index = order[start : start + batch]
            width = max(len(encoded[i].ids) for i in index)
            ids = np.zeros((len(index), width), dtype=np.int64)
            forced = np.zeros((len(index), width), dtype=bool)
            valid = np.zeros((len(index), width), dtype=bool)
            for r, i in enumerate(index):
                n = len(encoded[i].ids)
                ids[r, :n], forced[r, :n], valid[r, :n] = encoded[i].ids, encoded[i].forced(), True
            x, f, v = (torch.from_numpy(a).to(dev) for a in (ids, forced, valid))
            with torch.no_grad(), devices.autocast(dev, precision):
                logits = model(x, f, v).logits if isinstance(model, HNetLM) else model(x)
            head = logits[:, :-1]
            head = head if head.dtype == torch.float64 else head.float()
            nll = F.cross_entropy(head.flatten(0, 1), x[:, 1:].flatten(), reduction="none").view(len(index), -1)
            nll = nll.double().cpu().numpy()
            for r, i in enumerate(index):
                enc, n = encoded[i], len(encoded[i].ids)
                row, targets = nll[r, : n - 1], enc.ids[1:]
                special = enc.special[1:]
                eos = special & np.isin(targets, list(eos_ids))
                results[i] = DocNLL(
                    docs[i].key,
                    docs[i].domain,
                    enc.text_bytes,
                    float(row[~special].sum()),
                    int((~special).sum()),
                    float(row[eos].sum()),
                    int(eos.sum()),
                    float(row[special & ~eos].sum()),
                    int((special & ~eos).sum()),
                )
    finally:
        model.train(was_training)
    return [results[i] for i in range(len(docs))]


def bpb(rows: Sequence[DocNLL], domain: str, *, with_eos: bool = False) -> float:
    """Byte-weighted bits per byte of one domain (with the <eos> targets' NLL added for the sensitivity view)."""
    picked = [r for r in rows if r.domain == domain]
    n_bytes = sum(r.bytes for r in picked)
    if not n_bytes:
        raise TokstudyError(f"no bytes of domain {domain!r}")
    nats = sum(r.nll_nats + (r.eos_nats if with_eos else 0.0) for r in picked)
    return nats / (LN2 * n_bytes)


def summary(rows: Sequence[DocNLL]) -> dict[str, Any]:
    domains = sorted({r.domain for r in rows})
    return {
        d: {
            "bpb": bpb(rows, d),
            "bpb_with_eos": bpb(rows, d, with_eos=True),
            "docs": sum(1 for r in rows if r.domain == d),
            "bytes": sum(r.bytes for r in rows if r.domain == d),
            "targets": sum(r.n_targets for r in rows if r.domain == d),
        }
        for d in domains
    }


def write_nll(path: Path, header: Mapping[str, Any], rows: Sequence[DocNLL]) -> Path:
    lines = [canonical_json({"format": NLL_FORMAT, **header})] + [canonical_json(r.to_json()) for r in rows]
    return atomic_write_bytes(Path(path), b"\n".join(lines) + b"\n")


def read_nll(path: Path) -> tuple[dict[str, Any], list[DocNLL]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise TokstudyError(f"{path}: empty")
    header = json.loads(lines[0])
    if header.get("format") != NLL_FORMAT:
        raise TokstudyError(f"{path}: not a {NLL_FORMAT} file")
    return header, [DocNLL(**json.loads(line)) for line in lines[1:] if line.strip()]


def evaluate_point(
    run_dir: Path,
    endpoint: float,
    evalset: Path,
    out: Path,
    *,
    batch: int = 8,
    device: str | None = None,
    arms: Path | None = None,
    prereg: Path | None = None,
    stream: Path | None = None,
) -> dict[str, Any]:
    """Score a run's budget point (final.pt at 1, its decay branch's below) on an evalset; writes tokstudy-nll/1."""
    from . import train

    manifest, evalset_sha, docs = open_evalset(evalset)
    s = train.reopen(
        run_dir, arms=arms, stream=stream, prereg=prereg, device=device, verify_stream=False, same_code=False
    )
    if manifest["table"] != s.arm.table:
        raise TokstudyError(f"the evalset is table {manifest['table']}'s, the run table {s.arm.table}'s")
    listed = {a["id"]: a["arm_sha256"] for a in manifest["arms"]}
    if listed.get(s.arm.id) != arm_sha256(s.arm):
        raise TokstudyError(
            f"arm {s.arm.id} (sha256 {arm_sha256(s.arm)}) is not an arm the evalset's common prefix "
            "was computed for; rebuild the evalset for the table's current arms"
        )
    current = {a.id: arm_sha256(a) for a in load_arms(Path(s.record["arm"]["registry"]["path"]))}
    if listed != current:
        changed = sorted(
            set(listed) ^ set(current) | {i for i in set(listed) & set(current) if listed[i] != current[i]}
        )
        raise TokstudyError(
            f"the evalset's common prefix was computed for another arm set than the table's registry now holds "
            f"(differs in {changed}); rebuild it with `eval prefix` and rescore every arm on the new evalset"
        )
    if manifest["k_eval"] != s.prereg.k_eval:
        raise TokstudyError(f"the evalset has K_eval {manifest['k_eval']}, the prereg {s.prereg.k_eval}")
    path = train.point_checkpoint(run_dir, endpoint)
    payload = train.load_checkpoint(path, s.record["run_id"], s.device)
    s.model.load_state_dict(payload["model"])
    rows = score_documents(s.model, s.codec, docs, batch=batch, precision=s.config.precision, device=s.device)
    header = {
        "table": s.arm.table,
        "backend": train.BACKEND,
        "arm": s.arm.id,
        "arm_sha256": arm_sha256(s.arm),
        "run_id": s.record["run_id"],
        "seed": s.seed,
        "point": endpoint,
        "checkpoint": path.name,
        "checkpoint_sha256": sha256_file(path),
        "step": payload["step"],
        "flops": payload["flops"],
        "progress": payload["flops"] / s.prereg.f_target,
        "k_eval": manifest["k_eval"],
        "evalset_sha256": evalset_sha,
        "precision": s.config.precision,
        "run_code": s.record["code"],
        "eval_code": registry.code_identity(),
    }
    write_nll(out, header, rows)
    return {"nll": str(out), **header, "summary": summary(rows)}


# --- command line ----------------------------------------------------------------------------------------------------


def _docs(args) -> tuple[list[EvalDoc], dict[str, Any]]:
    given = [x for x in (args.tlc, args.bilingual) if x is not None] + (["plain"] if args.docs else [])
    if len(given) != 1:
        raise TokstudyError("give exactly one of --tlc SPEC, --bilingual MANIFEST or --docs DOMAIN=PATH")
    gen = args.include_generated
    if args.tlc is not None:
        split = args.split or "val"
        return list(tlc_documents(args.tlc, split, generated=gen)), tlc_source(args.tlc, split, generated=gen)
    if args.bilingual is not None:
        split = args.split or "dev"
        docs = list(bilingual_documents(args.bilingual, split, generated=gen))
        return docs, bilingual_source(args.bilingual, split, generated=gen)
    docs, files = [], []
    for arg in args.docs:
        domain, path = parse_docs_arg(arg)
        docs += [frame_plain(d, args.project) for d in plain_documents(path, domain)]
        files.append({"domain": domain, "file": Path(path).name, "sha256": sha256_file(path)})
    return docs, {"project": args.project, "plain": files}


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_tokstudy eval", description="common-prefix BPB")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prefix", help="build a table's evalset: held-out documents cut to the common byte prefix")
    p.add_argument("--arms", type=Path, required=True, help="the table's arm registry")
    p.add_argument("--prereg", type=Path, required=True, help="the table's prereg (K_eval)")
    p.add_argument("--tlc", type=Path, help="a TLC corpus spec (split --split, default val)")
    p.add_argument("--bilingual", type=Path, help="a bilingual derived manifest (split --split, default dev)")
    p.add_argument("--docs", action="append", default=[], help="DOMAIN=PATH sample files (framed per --project)")
    p.add_argument("--project", choices=["tlc", "bilingual"], help="framing of --docs")
    p.add_argument("--split")
    p.add_argument("--max-docs", type=int, help="per domain, lowest rank first")
    p.add_argument("--seed", type=int, default=0, help="rank seed of --max-docs")
    p.add_argument(
        "--include-generated",
        action="store_true",
        help="keep the documents of generated sources (TLC generated = true, bilingual synthetic); default: left out",
    )
    p.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score", help="score one budget point of a run on an evalset (tokstudy-nll/1)")
    s.add_argument("run_dir", type=Path)
    s.add_argument("--point", type=float, default=1.0, help="budget point: 1 (final.pt) or a branch's")
    s.add_argument("--evalset", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--device", default="auto")
    s.add_argument("--arms", type=Path)
    s.add_argument("--prereg", type=Path)
    s.add_argument("--stream", type=Path)
    args = parser.parse_args(list(argv))
    try:
        if args.action == "prefix":
            from .prereg import load_prereg

            prereg = load_prereg(args.prereg)
            arms = load_arms(args.arms)
            if args.docs and not args.project:
                raise TokstudyError("--docs needs --project (the framing)")
            docs, source = _docs(args)
            result = build_evalset(
                docs,
                arms,
                args.out,
                k_eval=prereg.k_eval,
                source=source,
                registry_sha256=sha256_file(args.arms),
                max_docs=args.max_docs,
                seed=args.seed,
            )
            print(json.dumps({"out": str(args.out), "by_domain": result["by_domain"], "dropped": result["dropped"]}))
        else:
            result = evaluate_point(
                args.run_dir,
                args.point,
                args.evalset,
                args.out,
                batch=args.batch,
                device=args.device,
                arms=args.arms,
                prereg=args.prereg,
                stream=args.stream,
            )
            print(json.dumps({"nll": result["nll"], "summary": result["summary"]}))
    except (TokstudyError, ValueError) as err:
        print(f"eval: {err}", file=sys.stderr)
        return 1
    return 0
