"""Teacher-forced NLL, bits per byte, the 2x2 matrix, the unigram baseline and the after-switch spike (MS §9.1,
§8.3; SCI-10, MA-08) and the `eval-nll` command.

- `token_nll` scores every token of a sequence given all earlier ones; a sequence longer than `max_seq` runs in
  windows of max_seq that advance by `stride`, each new target scored once with at least max_seq - stride tokens of
  context, and every window uses the slice of the full-history plan (MA-09).
- BPB = nats / (ln 2 * UTF-8 bytes). A document is `<eos>` + its text segments (the training convention, IMPL-15);
  its targets are the text tokens, its bytes the UTF-8 bytes of the text. BPB is compared only within one
  evaluation language, i.e. one column of the 2x2 matrix (a Han character is 3 bytes; bytes are not information).
- The 2x2 matrix: rows are experts run on the forced route (or the single expert of a seed / one-slot model),
  columns the evaluation languages zh and en. The matrix carries its labels.
- The unigram baseline (G1): add-one smoothed token frequencies of a training shard [建议].
- `spike_after_switch(k)`: the mean NLL of the first k target tokens after each route switch, pooled over switches;
  a switch's tokens stop at the next switch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..canonical import atomic_write_json, sha256_file
from ..config import DEFAULTS
from ..data.docs import LANGS, Doc, iter_file_docs
from ..data.tokenizer import encode_user_text
from ..routing.plan import RoutePlan

NLL_FORMAT = "bilingual-nll/1"
LN2 = math.log(2.0)
MATRIX_LABEL = ("rows: the expert on the forced route (a single-expert model's only expert); columns: the evaluation "
                "language; BPB = nats / (ln 2 * UTF-8 bytes), compared only within a column (SCI-10)")  # fmt: skip


def bpb(nats: float, n_bytes: int) -> float:
    if n_bytes <= 0:
        raise ValueError("BPB needs a positive byte count")
    return float(nats) / (LN2 * int(n_bytes))


@torch.no_grad()
def token_nll(model, ids: Sequence[int], plan: RoutePlan | None = None, *, stride: int | None = None,
              autocast: bool = False) -> np.ndarray:  # fmt: skip
    """NLL in nats of ids[t] given ids[:t], for t = 1 .. len(ids) - 1 (index t - 1 of the result)."""
    ids = list(map(int, ids))
    n, width = len(ids), model.cfg.max_seq
    if n < 2:
        return np.zeros(0)
    if plan is not None and plan.gates.shape[1] != n:
        raise ValueError(f"the plan covers {plan.gates.shape[1]} positions, the sequence {n}")
    stride = stride or max(1, width // 2)
    if not 0 < stride <= width:
        raise ValueError(f"stride must lie in 1..max_seq = {width}")
    device = next(model.parameters()).device
    out = np.full(n - 1, np.nan)
    start, done = 0, 1  # targets < done are scored
    while True:
        end = min(start + width, n)
        x = torch.tensor([ids[start:end]], device=device)
        window = None if plan is None else RoutePlan(plan.names, plan.gates[:, start:end].to(device), plan.policy)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
            logp = model(x, window).float().log_softmax(-1)[0].cpu()
        targets = torch.arange(max(done, start + 1), end)
        if targets.numel():
            out[targets.numpy() - 1] = (-logp[targets - 1 - start, torch.tensor(ids)[targets]]).double().numpy()
        done = end
        if end == n:
            break
        start += stride
    return out


def doc_tokens(tok, doc: Doc) -> tuple[list[int], int, list[str | None]]:
    """(<eos> + text ids, UTF-8 bytes of the text, the language of every position)."""
    ids, langs, n_bytes = [int(tok.token_to_id("<eos>"))], [None], 0
    for seg in doc.segments:
        if seg.text is None:
            continue
        piece = encode_user_text(tok, seg.text)
        ids += piece
        langs += [seg.lang] * len(piece)
        n_bytes += len(seg.text.encode("utf-8"))
    langs[0] = langs[1] if len(langs) > 1 else None
    return ids, n_bytes, langs


def doc_language(doc: Doc) -> str | None:
    """The one evaluation language of a document, or None when it mixes languages or has none."""
    langs = doc.langs or ((doc.kind,) if doc.kind in LANGS else ())
    return langs[0] if len(langs) == 1 else None


PlanFor = Callable[[int], RoutePlan | None]


def nll_matrix(model, tok, docs: Sequence[Doc], rows: Mapping[str, PlanFor], *, stride: int | None = None,
               autocast: bool = False) -> dict[str, Any]:  # fmt: skip
    """The labelled matrix of pooled BPB (and nats per token) per row and evaluation language."""
    cols = list(LANGS)
    encoded = []
    skipped = 0
    for doc in docs:
        lang = doc_language(doc)
        if lang is None:
            skipped += 1
            continue
        ids, n_bytes, _ = doc_tokens(tok, doc)
        if len(ids) > 1 and n_bytes > 0:
            encoded.append((lang, ids, n_bytes))
    tables = {k: [[0.0] * len(cols) for _ in rows] for k in ("nats", "tokens", "bytes", "docs")}
    for r, (_, plan_for) in enumerate(rows.items()):
        for lang, ids, n_bytes in encoded:
            c = cols.index(lang)
            nll = token_nll(model, ids, plan_for(len(ids)), stride=stride, autocast=autocast)
            tables["nats"][r][c] += float(nll.sum())
            tables["tokens"][r][c] += len(nll)
            tables["bytes"][r][c] += n_bytes
            tables["docs"][r][c] += 1
    bpb_t = [[bpb(tables["nats"][r][c], tables["bytes"][r][c]) if tables["bytes"][r][c] else None
              for c in range(len(cols))] for r in range(len(rows))]  # fmt: skip
    npt = [[tables["nats"][r][c] / tables["tokens"][r][c] if tables["tokens"][r][c] else None
            for c in range(len(cols))] for r in range(len(rows))]  # fmt: skip
    return {"label": MATRIX_LABEL, "rows": list(rows), "cols": cols, "bpb": bpb_t, "nats_per_token": npt,
            "tokens": tables["tokens"], "bytes": tables["bytes"], "docs": tables["docs"],
            "skipped_mixed_docs": skipped}  # fmt: skip


def unigram_bpb(counts: np.ndarray, tok, docs: Sequence[Doc]) -> dict[str, Any]:
    """[建议] the G1 baseline: add-one smoothed unigram token probabilities from training-shard counts."""
    counts = np.asarray(counts, dtype=np.float64)
    logp = np.log((counts + 1.0) / (counts.sum() + counts.size))
    out = {}
    for lang in LANGS:
        nats = n_bytes = 0.0
        for doc in docs:
            if doc_language(doc) != lang:
                continue
            ids, b, _ = doc_tokens(tok, doc)
            nats -= float(logp[np.asarray(ids[1:], dtype=np.int64)].sum())
            n_bytes += b
        out[lang] = bpb(nats, int(n_bytes)) if n_bytes else None
    return {"rule": "add-one smoothed unigram of the training shard [建议]", "bpb": out}


def spike_after_switch(nll: np.ndarray, routes: Sequence, mask: Sequence[bool] | None = None, k: int = 16) -> dict:
    """Mean NLL of the first k target tokens after each route switch (MS §9.1, RQ3a). `nll[t]` and `routes[t]` are
    per position (nll NaN where t is not a target); a switch is a position whose route differs from the previous
    one; a switch's tokens stop at the next switch."""
    nll = np.asarray(nll, dtype=np.float64)
    if len(routes) != nll.size:
        raise ValueError("nll and routes must cover the same positions")
    mask = np.isfinite(nll) if mask is None else np.asarray(mask, dtype=bool) & np.isfinite(nll)
    switches = [t for t in range(1, len(routes)) if routes[t] != routes[t - 1]]
    bounds = switches[1:] + [len(routes)] if switches else []
    taken, per = [], []
    for s, stop in zip(switches, bounds, strict=True):
        positions = [t for t in range(s, stop) if mask[t]][:k]
        taken += positions
        per.append({"at": s, "n": len(positions), "mean": float(nll[positions].mean()) if positions else None})
    return {"k": k, "n_switches": len(switches), "n_tokens": len(taken),
            "mean": float(nll[taken].mean()) if taken else None, "per_switch": per}  # fmt: skip


# --- command line ----------------------------------------------------------------------------------------------------


def _docs(paths: Sequence[Path], limit: int | None) -> tuple[list[Doc], dict[str, str]]:
    docs, files = [], {}
    for path in paths:
        files[str(path)] = sha256_file(path)
        docs += list(iter_file_docs(path, source=None, kind=None))
    return (docs[:limit] if limit else docs), files


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual eval-nll", description="NLL, BPB, 2x2 matrix")
    ap.add_argument("--model", type=Path, required=True, help="a seed pack directory or an assembly manifest")
    ap.add_argument("--forced", help="experts to run on the forced route, comma separated (default: all)")
    ap.add_argument("--docs", type=Path, nargs="+", required=True, help="dev documents (data.docs JSON lines)")
    ap.add_argument("--unigram-shard", type=Path, help="a training shard for the unigram baseline (G1)")
    ap.add_argument("--stride", type=int)
    ap.add_argument("--limit-docs", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--autocast", choices=["none", "bf16"], default=DEFAULTS["eval"]["autocast"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="load and verify the model and the documents; run nothing")
    args = ap.parse_args(list(argv))
    from .models import load_model
    from .score import _device

    try:
        ev = load_model(args.model, device=_device(args.device))
        docs, files = _docs(args.docs, args.limit_docs)
        names = ev.names
        forced = args.forced.split(",") if args.forced else list(names)
        unknown = [n for n in forced if n not in names]
        if unknown:
            raise ValueError(f"the model has no experts {unknown}: {names}")
        rows = {name: (lambda n, name=name: None if len(names) == 1 else RoutePlan.forced(name, names, 1, n))
                for name in forced}  # fmt: skip
        doc = {"format": NLL_FORMAT, "model": ev.source, "tokenizer_sha256": ev.tokenizer_sha256, "docs": files,
               "n_docs": len(docs), "rows": forced, "autocast": args.autocast}  # fmt: skip
        if args.dry_run:
            print(json.dumps({**doc, "dry_run": True}, indent=1, sort_keys=True))
            return 0
        doc["matrix"] = nll_matrix(ev.model, ev.tokenizer, docs, rows, stride=args.stride,
                                   autocast=args.autocast == "bf16")  # fmt: skip
        if args.unigram_shard:
            from ..data.shards import ShardReader

            shard = ShardReader(args.unigram_shard, tokenizer_sha256=ev.tokenizer_sha256,
                                vocab_size=ev.model.cfg.vocab_size)  # fmt: skip
            counts = np.bincount(np.asarray(shard.tokens, dtype=np.int64), minlength=ev.model.cfg.vocab_size)
            doc["unigram"] = {"shard": str(args.unigram_shard), **unigram_bpb(counts, ev.tokenizer, docs)}
    except (OSError, ValueError, KeyError) as exc:
        print(f"eval-nll: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.out:
        atomic_write_json(args.out, doc)
    print(json.dumps(doc, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
