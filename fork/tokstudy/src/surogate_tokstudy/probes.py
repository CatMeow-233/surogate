"""Probes: candidate scoring of continuations at a token boundary (spec §5.5, §6.3; STUDY M7), byte-normalised, with
the coverage of every arm, for the bilingual Track S rule (STUDY §9.2(3)) and the secondary readouts.

**Items** (`tokstudy-probe-items/1`, one JSON object per line). A probe item is a context, the text `stem` the
candidates continue, and the candidates:

    {"format": "tokstudy-probe-items/1", "key": "...", "domain": "en", "lang": "en" | null,
     "context": [{"special": "<eos>"}, {"text": "...", "lang": "en"}, ...],
     "stem": "Question: ...", "candidates": [" Paris", " Rome"], "gold": 0}

A candidate's document is `context + [stem + candidate]`: the stem and the candidate are one text segment (as the
bilingual training documents join question and answer), every segment is encoded on its own, so tokens never merge
across the context's specials. `items_from_bilingual` builds these items from Track S items (`bilingual-item/1`,
read with the project's own surogate_bilingual.eval.items.load_items): <eos>, the prompt segments before the
question, the question as the stem, `answer_sep + candidate` as each candidate, the answer language as the domain;
only the ranked families (the items with candidates).

**Scorable** (`scorable`). An item enters a comparison only if, under the arm's codec, the encoding of stem +
candidate starts with the encoding of the stem and is longer, for every candidate: the candidate then starts at a
token boundary, its tokens carry exactly its bytes, and its NLL is a sum over its own tokens. A merge across the
boundary makes the item unscorable for that arm (reason recorded). Byte and H-Net arms (the byte codec) are always
scorable. The verdict compares arms only on items scorable for every arm of the table; `coverage` is per arm.

**Scores** (`score_items`, `tokstudy-probe/1`). Per candidate the summed NLL (nats) of its tokens and its UTF-8
bytes; ranking uses the summed log-likelihood when every candidate has the same byte length, else the log-likelihood
per byte (`length_norm = "bytes"`; the token count never enters). `gold_logprob_per_byte` (the gold candidate's
log-likelihood over its bytes) is the continuous readout the verdict compares (M7: Track S answer log-prob,
length unit bytes).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import TokstudyError, atomic_write_bytes, canonical_json, sha256_file, sha256_json
from .codecs import ByteCodec, Codec, utf8
from .docs import BILINGUAL_EOS, Seg, check_domain, encode_doc

ITEMS_FORMAT = "tokstudy-probe-items/1"
PROBE_FORMAT = "tokstudy-probe/1"
LENGTH_NORM = "bytes"


@dataclass(frozen=True)
class ProbeItem:
    key: str
    domain: str
    context: tuple[Seg, ...]
    stem: str
    candidates: tuple[str, ...]
    gold: int
    lang: str | None = None

    def __post_init__(self):
        check_domain(self.domain)
        if len(self.candidates) < 2:
            raise TokstudyError(f"{self.key}: a probe item has at least two candidates")
        if not 0 <= self.gold < len(self.candidates):
            raise TokstudyError(f"{self.key}: gold {self.gold} is not a candidate index")
        if any(not c for c in self.candidates):
            raise TokstudyError(f"{self.key}: an empty candidate has no bytes to score")
        if not self.context or self.context[0].special is None:
            raise TokstudyError(f"{self.key}: the context starts with a special (the document's framing)")

    def to_json(self) -> dict[str, Any]:
        return {
            "format": ITEMS_FORMAT,
            "key": self.key,
            "domain": self.domain,
            "lang": self.lang,
            "context": [s.to_json() for s in self.context],
            "stem": self.stem,
            "candidates": list(self.candidates),
            "gold": self.gold,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ProbeItem:
        if data.get("format") != ITEMS_FORMAT:
            raise TokstudyError(f"not a {ITEMS_FORMAT} line: {str(data)[:80]}")
        try:
            return cls(
                str(data["key"]),
                data["domain"],
                tuple(Seg.from_json(s) for s in data["context"]),
                str(data["stem"]),
                tuple(str(c) for c in data["candidates"]),
                int(data["gold"]),
                data.get("lang"),
            )
        except (KeyError, TypeError, ValueError) as err:
            if isinstance(err, TokstudyError):
                raise
            raise TokstudyError(f"not a probe item: {err!r}") from None

    def document(self, candidate: str | None) -> tuple[Seg, ...]:
        """The segments of the context with the stem (candidate None) or with stem + candidate."""
        text = self.stem + (candidate or "")
        return (*self.context, Seg(text=text, lang=self.lang)) if text else tuple(self.context)


def read_items(path: Path) -> list[ProbeItem]:
    items = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                items.append(ProbeItem.from_json(json.loads(line)))
            except (ValueError, TokstudyError) as err:
                raise TokstudyError(f"{path}:{number}: {err}") from None
    keys = [i.key for i in items]
    if len(set(keys)) != len(keys):
        raise TokstudyError(f"{path}: duplicate item keys")
    return items


def write_items(path: Path, items: Iterable[ProbeItem]) -> Path:
    return atomic_write_bytes(Path(path), b"".join(canonical_json(i.to_json()) + b"\n" for i in items))


def items_sha256(items: Sequence[ProbeItem]) -> str:
    return sha256_json([i.to_json() for i in items])


def items_from_bilingual(paths: Sequence[Path], *, split: str | None = "dev") -> list[ProbeItem]:
    """The ranked Track S items of `paths` (bilingual-item/1, the project's loader) as probe items (module docstring)."""
    from surogate_bilingual.eval.items import load_items

    try:
        raw, _ = load_items([Path(p) for p in paths], split=split)
    except (OSError, ValueError) as err:
        raise TokstudyError(f"cannot read the Track S items: {err}") from None
    out = []
    for item in raw:
        if not item.get("candidates"):
            continue  # a generated family (E4): no candidates to rank
        prompt = []
        for seg in item["segments"]:
            if seg.get("role") == "answer" or seg.get("text") is None:
                break
            prompt.append(seg)
        if not prompt:
            raise TokstudyError(f"{item['id']}: no prompt segment")
        sep = item.get("answer_sep", "")
        context = (Seg(BILINGUAL_EOS), *(Seg(text=s["text"], lang=s["lang"]) for s in prompt[:-1]))
        out.append(
            ProbeItem(
                str(item["id"]),
                item["answer_lang"],
                context,
                prompt[-1]["text"],
                tuple(sep + c for c in item["candidates"]),
                int(item["gold_index"]),
                prompt[-1]["lang"],
            )
        )
    if not out:
        raise TokstudyError("no ranked Track S item in the given files")
    return out


# --- scorability and scoring -----------------------------------------------------------------------------------------


def scorable(codec: Codec, stem: str, candidate: str) -> tuple[bool, str | None]:
    """(True, None) when stem + candidate encodes as the stem's tokens followed by at least one more token."""
    if isinstance(codec, ByteCodec):
        return True, None
    head, _ = codec.encode_text(stem) if stem else (np.zeros(0, dtype=np.int64), None)
    whole, _ = codec.encode_text(stem + candidate)
    if len(whole) <= len(head) or not np.array_equal(whole[: len(head)], head):
        return False, "a token merges across the boundary between the stem and a candidate"
    return True, None


@dataclass(frozen=True)
class ProbeRow:
    key: str
    domain: str
    scorable: bool
    reason: str | None
    candidate_nll: tuple[float, ...] = ()
    candidate_bytes: tuple[int, ...] = ()
    norm: str | None = None
    pred: int | None = None
    correct: bool | None = None
    gold_logprob_per_byte: float | None = None

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["candidate_nll"], out["candidate_bytes"] = list(self.candidate_nll), list(self.candidate_bytes)
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ProbeRow:
        data = dict(data)
        data["candidate_nll"] = tuple(data.get("candidate_nll", ()))
        data["candidate_bytes"] = tuple(data.get("candidate_bytes", ()))
        return cls(**data)


def rank(nll: Sequence[float], n_bytes: Sequence[int]) -> tuple[str, int]:
    """(norm, predicted index): summed log-likelihood when every candidate has the same byte length, else per byte;
    ties go to the lowest index."""
    norm = "sum" if len(set(n_bytes)) == 1 else "per_byte"
    scores = [-x if norm == "sum" else -x / b for x, b in zip(nll, n_bytes, strict=True)]
    return norm, max(range(len(scores)), key=lambda i: (scores[i], -i))


def score_items(
    model, codec: Codec, items: Sequence[ProbeItem], *, batch: int = 8, precision: str = "fp32", device=None
) -> list[ProbeRow]:
    """One ProbeRow per item, in the order of `items` (module docstring)."""
    from .evaluate import target_nll

    encoded, where, rows = [], [], {}
    for index, item in enumerate(items):
        reasons = {scorable(codec, item.stem, c)[1] for c in item.candidates} - {None}
        if reasons:
            rows[index] = ProbeRow(item.key, item.domain, False, sorted(reasons)[0])
            continue
        prefix = len(encode_doc(codec, item.document(None)).ids)
        for c, candidate in enumerate(item.candidates):
            enc = encode_doc(codec, item.document(candidate))
            if prefix < 1 or int(enc.token_bytes[prefix:].sum()) != len(utf8(candidate)):
                raise TokstudyError(f"{codec.name}: the tokens of {item.key} candidate {c} miss its bytes")
            encoded.append(enc)
            where.append((index, c, prefix))
    nll = target_nll(model, encoded, batch=batch, precision=precision, device=device)
    sums: dict[int, dict[int, float]] = {}
    for (index, c, prefix), row in zip(where, nll, strict=True):
        sums.setdefault(index, {})[c] = float(row[prefix - 1 :].sum())  # targets prefix .. n-1 are the candidate's
    for index, by_candidate in sums.items():
        item = items[index]
        values = tuple(by_candidate[c] for c in range(len(item.candidates)))
        n_bytes = tuple(len(utf8(c)) for c in item.candidates)
        norm, pred = rank(values, n_bytes)
        rows[index] = ProbeRow(
            item.key,
            item.domain,
            True,
            None,
            values,
            n_bytes,
            norm,
            pred,
            pred == item.gold,
            -values[item.gold] / n_bytes[item.gold],
        )
    return [rows[i] for i in range(len(items))]


def coverage(rows: Sequence[ProbeRow]) -> dict[str, Any]:
    by_domain: dict[str, list[int]] = {}
    for r in rows:
        by_domain.setdefault(r.domain, [0, 0])
        by_domain[r.domain][0] += 1
        by_domain[r.domain][1] += int(r.scorable)
    return {d: {"items": n, "scorable": k, "share": k / n} for d, (n, k) in sorted(by_domain.items())}


def write_probe(path: Path, header: Mapping[str, Any], rows: Sequence[ProbeRow]) -> Path:
    lines = [canonical_json({"format": PROBE_FORMAT, **header})] + [canonical_json(r.to_json()) for r in rows]
    return atomic_write_bytes(Path(path), b"\n".join(lines) + b"\n")


def read_probe(path: Path) -> tuple[dict[str, Any], list[ProbeRow]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise TokstudyError(f"{path}: empty")
    header = json.loads(lines[0])
    if header.get("format") != PROBE_FORMAT:
        raise TokstudyError(f"{path}: not a {PROBE_FORMAT} file")
    return header, [ProbeRow.from_json(json.loads(line)) for line in lines[1:] if line.strip()]


def probe_point(
    run_dir: Path,
    endpoint: float,
    items: Sequence[ProbeItem],
    out: Path,
    *,
    source: Mapping[str, Any],
    batch: int = 8,
    device: str | None = None,
    arms: Path | None = None,
    prereg: Path | None = None,
    stream: Path | None = None,
) -> dict[str, Any]:
    """Score a run's budget point on probe items; writes tokstudy-probe/1."""
    from . import registry, train
    from .arms import arm_sha256

    s = train.reopen(
        run_dir, arms=arms, stream=stream, prereg=prereg, device=device, verify_stream=False, same_code=False
    )
    path = train.point_checkpoint(run_dir, endpoint)
    payload = train.load_checkpoint(path, s.record["run_id"], s.device)
    s.model.load_state_dict(payload["model"])
    rows = score_items(s.model, s.codec, items, batch=batch, precision=s.config.precision, device=s.device)
    header = {
        "table": s.arm.table,
        "backend": train.BACKEND,
        "arm": s.arm.id,
        "arm_kind": s.arm.kind,
        "arm_sha256": arm_sha256(s.arm),
        "run_id": s.record["run_id"],
        "seed": s.seed,
        "point": endpoint,
        "checkpoint": path.name,
        "checkpoint_sha256": sha256_file(path),
        "items_sha256": items_sha256(items),
        "source": dict(source),
        "length_norm": LENGTH_NORM,
        "prereg_sha256": s.prereg.sha256,
        "run_prereg_sha256": registry.read_run(run_dir)["prereg"]["sha256"],
        "coverage": coverage(rows),
        "eval_code": registry.code_identity(),
    }
    write_probe(out, header, rows)
    return {"probe": str(out), **header}


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_tokstudy probe", description="probe candidate scores")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--point", type=float, default=1.0, help="budget point: 1 (final.pt) or a branch's")
    parser.add_argument("--items", type=Path, action="append", default=[], help="tokstudy-probe-items/1 files")
    parser.add_argument(
        "--bilingual-items", type=Path, action="append", default=[], help="Track S bilingual-item/1 files (ranked)"
    )
    parser.add_argument("--split", default="dev", help="the Track S split of --bilingual-items")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--arms", type=Path)
    parser.add_argument("--prereg", type=Path)
    parser.add_argument("--stream", type=Path)
    args = parser.parse_args(list(argv))
    try:
        if bool(args.items) == bool(args.bilingual_items):
            raise TokstudyError("give --items or --bilingual-items")
        if args.items:
            items = [i for path in args.items for i in read_items(path)]
            source = {"items": [{"file": p.name, "sha256": sha256_file(p)} for p in args.items]}
        else:
            items = items_from_bilingual(args.bilingual_items, split=args.split)
            source = {
                "bilingual_items": [{"file": p.name, "sha256": sha256_file(p)} for p in args.bilingual_items],
                "split": args.split,
            }
        result = probe_point(
            args.run_dir,
            args.point,
            items,
            args.out,
            source=source,
            batch=args.batch,
            device=args.device,
            arms=args.arms,
            prereg=args.prereg,
            stream=args.stream,
        )
    except (TokstudyError, ValueError) as err:
        print(f"probe: {err}", file=sys.stderr)
        return 1
    print(json.dumps({"probe": result["probe"], "coverage": result["coverage"]}))
    return 0
