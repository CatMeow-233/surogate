"""The switch-cost decomposition of MS §8.3 (RQ3a; MA-08) and the `switch` command.

On the same token ids, teacher-forced, the mean NLL of the target segment (the last answer) under

- M: the causal mixed route (R_out by segment);
- R_new: every token on the target's route (a re-prefill; by H1 this is exactly the standalone new-route model);
- R_old: every token on the route the sequence switched away from (the last different route before the target);
- L_ref (when a translated history is given): the history already in the target language, all on the new route.

history-state mismatch cost = L(M) - L(R_new); the cost of the new expert reading the other language's history =
L(R_new) - L_ref. On an item without a switch, M is the forced plan of its only route, so M equals R_new exactly.
E4 items are scored on their references: source, instruction and the reference translation (and, for switchback,
the back instruction and the back reference). The spike over the first k target tokens after each switch is
reported for M.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..canonical import atomic_write_json
from ..config import DEFAULTS
from ..data.tokenizer import encode_user_text
from ..routing.plan import RoutePlan
from ..routing.policies import plan_from_segments, segment_routes
from .items import Seg, bos, load_items
from .nll import spike_after_switch, token_nll

SWITCH_FORMAT = "bilingual-switch/1"


def switch_segments(tok, item: Mapping[str, Any]) -> list[Seg]:
    """The teacher-forced segments of an E4 item: <eos>, source, instruction, reference (+ back instruction and
    back reference); the last segment is the target."""
    if item["family"] != "E4":
        raise ValueError(f"{item['id']}: the switch decomposition runs on E4 items")
    texts = [s for s in item["segments"] if s.get("text") is not None]
    nl = tuple(encode_user_text(tok, "\n"))
    segs = [Seg(texts[0]["lang"], "source", tuple(encode_user_text(tok, texts[0]["text"])))]
    for k, stage in enumerate(item["stages"]):
        instr = texts[1 + k]
        segs.append(Seg(instr["lang"], "instruction", tuple(encode_user_text(tok, instr["text"]))))
        ref = tuple(encode_user_text(tok, stage["reference"]))
        last = k == len(item["stages"]) - 1
        segs.append(Seg(stage["answer_lang"], "answer", ref if last else ref + nl))
    return [bos(tok, segs[0].lang), *segs]


def decompose_segments(
    model, segs: Sequence[Seg], *, reference_history: Sequence[Seg] | None = None, k: int = 16
) -> dict[str, Any]:
    """M, R_new, R_old (and L_ref) of the last segment of `segs` (see the module docstring)."""
    names = model.expert_names
    if len(names) < 2:
        raise ValueError(f"the decomposition needs a model with at least two experts, not {names}")
    meta = [(s.lang, len(s.ids), s.role) for s in segs]
    routes = segment_routes(meta, "R_out")
    new = routes[-1]
    old = next((r for r in reversed(routes[:-1]) if r != new), new)
    ids = [t for s in segs for t in s.ids]
    n_target = len(segs[-1].ids)
    first = len(ids) - n_target  # position of the first target token

    def mean_nll(plan: RoutePlan, sequence=ids, start=first) -> tuple[float, np.ndarray]:
        nll = token_nll(model, sequence, plan)
        return float(nll[start - 1 :].mean()), nll

    m_plan = plan_from_segments(meta, names, "R_out")
    m, m_nll = mean_nll(m_plan)
    r_new, _ = mean_nll(RoutePlan.forced(new, names, 1, len(ids)))
    r_old, _ = mean_nll(RoutePlan.forced(old, names, 1, len(ids)))
    out = {
        "M": m,
        "R_new": r_new,
        "R_old": r_old,
        "L_ref": None,
        "new_route": new,
        "old_route": old,
        "switched": old != new,
        "n_target_tokens": n_target,
        "mismatch_cost": m - r_new,
    }
    if reference_history is not None:
        ref_ids = [t for s in reference_history for t in s.ids] + list(segs[-1].ids)
        out["L_ref"], _ = mean_nll(RoutePlan.forced(new, names, 1, len(ref_ids)), ref_ids, len(ref_ids) - n_target)
        out["reading_cost"] = r_new - out["L_ref"]
    per_pos = np.concatenate([[np.nan], m_nll])
    mask = np.zeros(len(ids), dtype=bool)
    mask[first:] = True
    route_of = [r for r, (_, n, _) in zip(routes, meta, strict=True) for _ in range(n)]
    out["spike_M"] = spike_after_switch(per_pos, route_of, mask, k)
    return out


def decompose_switch(model, tok, item: Mapping[str, Any], *, k: int = 16) -> dict[str, Any]:
    return {
        "item_id": item["id"],
        "direction": item["direction"],
        "variant": item["variant"],
        **decompose_segments(model, switch_segments(tok, item), k=k),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[f"{r['variant']}/{r['direction']}"].append(r)
    out = {}
    for key, group in sorted(groups.items()):
        spikes = [r["spike_M"]["mean"] for r in group if r["spike_M"]["mean"] is not None]
        out[key] = {
            "n": len(group),
            **{f"mean_{m}": float(np.mean([r[m] for r in group])) for m in ("M", "R_new", "R_old", "mismatch_cost")},
            "mean_spike_M": float(np.mean(spikes)) if spikes else None,
        }
    return out


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual switch", description="switch-cost decomposition")
    ap.add_argument("--model", type=Path, required=True, help="an assembly manifest with at least two experts")
    ap.add_argument("--items", type=Path, nargs="+", required=True, help="E4 bilingual-item/1 files")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--k", type=int, default=DEFAULTS["eval"]["switch_k"], help="after-switch spike (MS §9.1)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="load and verify the model and the items; run nothing")
    args = ap.parse_args(list(argv))
    from .items import items_sha256
    from .models import load_model
    from .score import _device

    try:
        ev = load_model(args.model, device=_device(args.device))
        items, files = load_items(args.items, family="E4", split=args.split, limit=args.limit)
        if any(i["split"] == "test" for i in items) and not args.allow_test:
            raise ValueError("test items are evaluated only after the preregistration lock (MS §9.7): --allow-test")
        if len(ev.names) < 2:
            raise ValueError(f"the decomposition needs at least two experts; {args.model} has {ev.names}")
        doc = {
            "format": SWITCH_FORMAT,
            "model": ev.source,
            "item_files": files,
            "items_sha256": items_sha256(items),
            "n_items": len(items),
            "k": args.k,
        }
        if args.dry_run:
            print(json.dumps({**doc, "dry_run": True}, indent=1, sort_keys=True))
            return 0
        rows = [decompose_switch(ev.model, ev.tokenizer, item, k=args.k) for item in items]
        doc.update(summary=summarize(rows), items=rows)
    except (OSError, ValueError, KeyError) as exc:
        print(f"switch: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.out:
        atomic_write_json(args.out, doc)
    print(json.dumps(doc["summary"], indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
