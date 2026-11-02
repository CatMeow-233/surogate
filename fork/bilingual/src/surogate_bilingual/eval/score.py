"""Candidate log-likelihood ranking, generation scoring and the `eval-tasks` command (MS §9.1; SCI-11, IMPL-15).

Ranked families (E1, E2, E3-par, E3-ctx): the prompt is a shared prefix; for each candidate only its own tokens are
scored, log p(c_j | prompt, c_<j) summed (`sum`) or averaged per token (`mean`). `auto` (the registered rule) is
`sum` when every candidate has the same token length, else `mean`. The predicted index is the argmax (the lowest
index on a tie). Candidate sets hold the other-language form of the gold, so ranking also measures language choice;
`lang_ok` is False exactly when that distractor wins.

Generated family (E4): batched greedy decoding (`eval.generate`), then exact slot scoring per stage: the name as a
whole word; every attribute through the joint leftmost-longest match over all attribute lexicons
(`data.synth.values.find_attributes`, so "陶艺家" is not read as the hobby "陶艺"); years and other numbers also
through `extract_number`, whose boundaries are digit look-arounds (`(?<!\\d)43(?!\\d)`): `\\b` fails on
"答案是43" because CJK characters are word characters, and substring anchors accept "143", "430", "2025". The
output must also be in the answer language (`lang_ok`). There are no substring anchors anywhere.

Records (`bilingual-pred/1`): one per item and arm. An item the arm cannot score (an expert the routing needs is
not in the model; R_src on an item without a single knowledge source; ...) gets a `skip_reason` and is never
counted as correct. Every record carries `items_sha256`, the hash of the evaluated item list.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..canonical import atomic_write_bytes, atomic_write_json, canonical_json
from ..config import DEFAULTS
from ..data.docs import is_han, is_latin_letter
from ..data.synth.values import ATTRIBUTES, find_attributes, lexicon
from ..data.tokenizer import decode, encode_user_text
from ..routing.plan import RoutePlan
from .generate import greedy_batch
from .items import (
    GENERATED_FAMILIES,
    RANKED_FAMILIES,
    Routing,
    Seg,
    SkipItem,
    bos,
    candidate_ids,
    cluster_of,
    extend_plan,
    item_plan,
    items_sha256,
    load_items,
    prompt_segments,
)

PRED_FORMAT = "bilingual-pred/1"
NORMS = ("sum", "mean", "auto")
LANG_SHARE = DEFAULTS["eval"]["lang_share"]
MERGE = DEFAULTS["eval"]["merge"]
MAX_NEW = DEFAULTS["eval"]["max_new"]


# --- ranking ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateScores:
    logprobs: list[float]  # summed log p of each candidate's tokens
    n_tokens: list[int]
    norm: str  # the rule actually used: "sum" or "mean"
    scores: list[float]
    pred_index: int


def resolve_norm(norm: str, lengths: Sequence[int]) -> str:
    if norm not in NORMS:
        raise ValueError(f"length normalisation must be one of {NORMS}, got {norm!r}")
    if norm == "auto":
        return "sum" if len(set(lengths)) == 1 else "mean"
    return norm


@torch.no_grad()
def candidate_logprobs(
    model,
    prompt_ids: Sequence[int],
    plan: RoutePlan | None,
    candidates: Sequence[Sequence[int]],
    *,
    batch: int = 16,
    pad: int = 1,
) -> list[float]:
    """sum_j log p(c_j | prompt, c_<j) for every candidate. `plan` covers at least the prompt and the longest
    candidate ([1, P + L_max, E]) or is None; each row runs on its last `max_seq` tokens with its plan's slice."""
    prompt = list(map(int, prompt_ids))
    if not prompt:
        raise ValueError("ranking needs a non-empty prompt (the first candidate token is predicted from it)")
    if any(len(c) == 0 for c in candidates):
        raise ValueError("every candidate needs at least one token")
    max_seq = model.cfg.max_seq
    device = next(model.parameters()).device
    out: list[float] = []
    for first in range(0, len(candidates), batch):
        chunk = [list(map(int, c)) for c in candidates[first : first + batch]]
        full = [prompt + c for c in chunk]
        starts = [max(0, len(h) - max_seq) for h in full]
        if any(s >= len(prompt) for s in starts):
            raise ValueError(f"a candidate is too long for max_seq = {max_seq}")
        width = max(len(h) - s for h, s in zip(full, starts, strict=True))
        tokens = torch.full((len(full), width), pad, dtype=torch.long)
        rows = []
        for i, (h, s) in enumerate(zip(full, starts, strict=True)):
            tokens[i, : len(h) - s] = torch.tensor(h[s:])
            if plan is not None:
                g = plan.gates[0, s : len(h)]
                rows.append(torch.cat([g, g[-1:].expand(width - g.shape[0], -1)]) if g.shape[0] < width else g)
        window = None if plan is None else RoutePlan(plan.names, torch.stack(rows).to(device), plan.policy)
        logp = model(tokens.to(device), window).float().log_softmax(-1).cpu()
        for i, (c, s) in enumerate(zip(chunk, starts, strict=True)):
            pos = torch.arange(len(prompt) - s - 1, len(prompt) - s - 1 + len(c))
            out.append(float(logp[i, pos, torch.tensor(c)].sum()))
    return out


def rank_candidates(model, prompt_ids, plan, candidates_ids, norm: str = "auto") -> CandidateScores:
    lengths = [len(c) for c in candidates_ids]
    used = resolve_norm(norm, lengths)
    logprobs = candidate_logprobs(model, prompt_ids, plan, candidates_ids)
    scores = logprobs if used == "sum" else [lp / n for lp, n in zip(logprobs, lengths, strict=True)]
    best = max(range(len(scores)), key=lambda i: (scores[i], -i))
    return CandidateScores(logprobs, lengths, used, scores, best)


# --- generation scoring ------------------------------------------------------------------------------------------------


def extract_number(text: str, gold: str | Sequence[str]) -> bool | None:
    """True when every gold number occurs in `text` delimited by non-digits; None ("not applicable") for an empty
    gold list, never True (IMPL-15)."""
    golds = [gold] if isinstance(gold, str) else list(gold)
    if not golds:
        return None
    for g in golds:
        if not re.fullmatch(r"[0-9]+", g):
            raise ValueError(f"gold numbers are ASCII digit strings, got {g!r}")
    return all(re.search(rf"(?<![0-9０-９]){g}(?![0-9０-９])", text) is not None for g in golds)


def lang_ok(text: str, lang: str, ignore: Sequence[str] = ()) -> bool:
    """[建议] the output is in `lang`: after removing `ignore` (names are Latin script in both languages), at least
    LANG_SHARE of its Han + Latin letters belong to the language's script, and there is at least one."""
    for s in ignore:
        if s:
            text = text.replace(s, " ")
    han = sum(1 for ch in text if is_han(ch))
    latin = sum(1 for ch in text if is_latin_letter(ch))
    if han + latin == 0:
        return False
    own = han if lang == "zh" else latin
    return own / (han + latin) >= LANG_SHARE


_LEXICONS = {attr: lexicon(attr) for attr in ATTRIBUTES}


def score_slots(text: str, slots: Mapping[str, str], lang: str) -> dict[str, bool]:
    """Exact slot matches of one generated stage: the name as a whole Latin word, every attribute by the joint
    lexicon match, numbers additionally by digit boundaries."""
    found = find_attributes(text, lang, _LEXICONS)
    out = {}
    for slot, value in slots.items():
        if slot == "name":
            out[slot] = re.search(rf"(?<![A-Za-z]){re.escape(value)}(?![A-Za-z])", text) is not None
            continue
        if slot not in _LEXICONS:
            raise ValueError(f"unknown E4 slot {slot!r}")
        index = _LEXICONS[slot].index_of(value, lang)
        ok = index in found[slot]
        digits = re.findall(r"[0-9]+", value)
        if digits:
            ok = ok and bool(extract_number(text, digits))
        out[slot] = ok
    return out


# --- evaluating items ------------------------------------------------------------------------------------------------


@dataclass
class EvalContext:
    model: Any
    tok: Any
    names: tuple[str, ...]
    routing: Routing
    arm: str
    replicate: int
    items_sha256: str
    norm: str = "auto"
    max_new: int = MAX_NEW
    autocast: bool = False

    @property
    def eos(self) -> int:
        return int(self.tok.token_to_id("<eos>"))

    def autocast_ctx(self):
        device = next(self.model.parameters()).device.type
        return torch.autocast(device, dtype=torch.bfloat16, enabled=self.autocast)


def _record(ctx: EvalContext, item: Mapping[str, Any], **fields) -> dict[str, Any]:
    route, gate = ctx.routing.label(item) if len(ctx.names) > 1 else ("single", "one_hot")
    base = {
        "format": PRED_FORMAT,
        "item_id": item["id"],
        "family": item["family"],
        "variant": item["variant"],
        "direction": item["direction"],
        "split": item["split"],
        "arm": ctx.arm,
        "replicate": ctx.replicate,
        "route_policy": route,
        "gate_policy": gate,
        "norm": None,
        "candidate_logprobs": None,
        "candidate_tokens": None,
        "pred_index": None,
        "gold_index": item.get("gold_index"),
        "correct": None,
        "lang_ok": None,
        "output": None,
        "route_trace": None,
        "skip_reason": None,
        "items_sha256": ctx.items_sha256,
        "cluster": cluster_of(item),
    }
    base.update(fields)
    return base


def rank_item(ctx: EvalContext, item: Mapping[str, Any]) -> dict[str, Any]:
    segs = prompt_segments(ctx.tok, item)
    q_text = next(s["text"] for s in reversed(item["segments"]) if s.get("text") is not None)
    cands, merged = [], False
    for c in item["candidates"]:
        ids, boundary = candidate_ids(ctx.tok, q_text, item.get("answer_sep", ""), c)
        cands.append(ids)
        merged |= boundary
    prompt = [t for s in segs for t in s.ids]
    try:
        plan = item_plan(item, segs, max(len(c) for c in cands), item["answer_lang"], ctx.names, ctx.routing)
    except SkipItem as exc:
        return _record(ctx, item, skip_reason=str(exc))
    with ctx.autocast_ctx():
        result = rank_candidates(ctx.model, prompt, plan, cands, ctx.norm)
    distractor = item.get("distractor_index")
    gold = item["gold_index"]
    margin = result.scores[gold] - max(x for i, x in enumerate(result.scores) if i != gold)  # MS §9.1 secondary
    return _record(
        ctx,
        item,
        norm=result.norm,
        candidate_logprobs=result.logprobs,
        candidate_tokens=result.n_tokens,
        pred_index=result.pred_index,
        correct=result.pred_index == gold,
        margin=margin,
        lang_ok=None if distractor is None else result.pred_index != distractor,
        boundary_merge=merged,
    )


def _stage_plan_fn(ctx: EvalContext, item, segs: Sequence[Seg], answer_lang: str):
    prompt_len = sum(len(s.ids) for s in segs)

    def fn(_i: int, ids: list[int]):
        plan = item_plan(item, segs, max(1, len(ids) - prompt_len), answer_lang, ctx.names, ctx.routing)
        return extend_plan(plan, len(ids))

    return fn


def newline_ids(tok) -> frozenset[int]:
    """Every vocabulary id whose text holds a newline: an E4 answer is one line (data.synth.build renders the
    target, then a newline and the back instruction in switchback documents), so a stage stops there."""
    return frozenset(i for i in range(tok.get_vocab_size()) if "\n" in tok.decode([i], skip_special_tokens=False))


def generate_items(ctx: EvalContext, items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """E4: greedy generation stage by stage, batched across items, then exact slot scoring. A stage ends at <eos>, at
    a newline (the training documents end an answer there) or after max_new tokens."""
    records: dict[str, dict[str, Any]] = {}
    state = {}
    for item in items:
        segs = prompt_segments(ctx.tok, item)
        try:
            item_plan(item, segs, 1, item["stages"][0]["answer_lang"], ctx.names, ctx.routing)
        except SkipItem as exc:
            records[item["id"]] = _record(ctx, item, skip_reason=str(exc))
            continue
        state[item["id"]] = {"item": item, "segs": list(segs), "stages": []}
    newline = encode_user_text(ctx.tok, "\n")
    stops = newline_ids(ctx.tok) if state else frozenset()
    n_stages = max((len(s["item"]["stages"]) for s in state.values()), default=0)
    for k in range(n_stages):
        active = [s for s in state.values() if k < len(s["item"]["stages"])]
        if k > 0:  # the next instruction follows the previous answer after a newline (data.synth.tasks)
            for s in active:
                back = [seg for seg in s["item"]["segments"] if seg["role"] == "instruction"][k]
                prev = s["stages"][-1]
                lang = s["item"]["stages"][k - 1]["answer_lang"]
                s["segs"] += [
                    Seg(lang, "answer", tuple(prev["ids"]) + tuple(newline)),
                    Seg(back["lang"], "instruction", tuple(encode_user_text(ctx.tok, back["text"]))),
                ]
        fns = [_stage_plan_fn(ctx, s["item"], s["segs"], s["item"]["stages"][k]["answer_lang"]) for s in active]
        prompts = [[t for seg in s["segs"] for t in seg.ids] for s in active]
        with ctx.autocast_ctx():
            gens = greedy_batch(
                ctx.model,
                prompts,
                lambda i, ids, fns=fns: fns[i](i, ids),
                ctx.max_new,
                ctx.eos,
                stop_ids=stops,
            )
        for s, gen in zip(active, gens, strict=True):
            stage = s["item"]["stages"][k]
            text = decode(ctx.tok, gen.tokens)
            slots = score_slots(text, stage["slots"], stage["answer_lang"])
            ok = lang_ok(text, stage["answer_lang"], ignore=[stage["slots"].get("name", "")])
            s["stages"].append(
                {
                    "answer_lang": stage["answer_lang"],
                    "output": text,
                    "ids": gen.tokens,
                    "stop": gen.stop,
                    "slots": slots,
                    "lang_ok": ok,
                    "route_trace": gen.route_trace,
                }
            )
    for key, s in state.items():
        stages = s["stages"]
        n = sum(len(st["slots"]) for st in stages)
        hit = sum(sum(st["slots"].values()) if st["lang_ok"] else 0 for st in stages)
        records[key] = _record(
            ctx,
            s["item"],
            output=stages[-1]["output"],
            route_trace=stages[-1]["route_trace"],
            lang_ok=all(st["lang_ok"] for st in stages),
            slot_acc=hit / n if n else None,
            correct=all(st["lang_ok"] and all(st["slots"].values()) for st in stages),
            stages=[{k: v for k, v in st.items() if k not in ("ids", "route_trace")} for st in stages],
        )
    return [records[item["id"]] for item in items]


def evaluate(ctx: EvalContext, items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    ranked = [i for i in items if i["family"] in RANKED_FAMILIES]
    generated = [i for i in items if i["family"] in GENERATED_FAMILIES]
    unknown = [i["id"] for i in items if i["family"] not in RANKED_FAMILIES + GENERATED_FAMILIES]
    if unknown:
        raise ValueError(f"items of unknown families: {unknown[:3]}")
    with torch.no_grad():
        by_id = {i["id"]: rank_item(ctx, i) for i in ranked}
        by_id.update({r["item_id"]: r for r in generate_items(ctx, generated)})
    for item in items:
        out.append(by_id[item["id"]])
    return out


def generate_prompts(ctx: EvalContext, prompts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A fixed generation set (G1): free prompts {id, lang, text}; the output and its language are recorded."""
    segs = [
        [bos(ctx.tok, p["lang"]), Seg(p["lang"], "question", tuple(encode_user_text(ctx.tok, p["text"])))]
        for p in prompts
    ]
    fake = [
        {
            "id": p["id"],
            "family": "gen",
            "variant": "free",
            "direction": p["lang"],
            "split": "fixed",
            "segments": [],
            "template_ids": {},
        }
        for p in prompts
    ]
    fns = [_stage_plan_fn(ctx, f, s, p["lang"]) for f, s, p in zip(fake, segs, prompts, strict=True)]
    with torch.no_grad(), ctx.autocast_ctx():
        gens = greedy_batch(
            ctx.model,
            [[t for seg in s for t in seg.ids] for s in segs],
            lambda i, ids: fns[i](i, ids),
            ctx.max_new,
            ctx.eos,
        )
    out = []
    for f, p, gen in zip(fake, prompts, gens, strict=True):
        text = decode(ctx.tok, gen.tokens)
        out.append(
            _record(
                ctx,
                f,
                output=text,
                lang_ok=lang_ok(text, p["lang"]),
                route_trace=gen.route_trace,
                gold_index=None,
                stop=gen.stop,
            )
        )
    return out


# --- command line --------------------------------------------------------------------------------------------------


def _pair(text: str) -> tuple[float, float]:
    a, _, b = text.partition(",")
    return float(a), float(b)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m surogate_bilingual eval-tasks",
        description="candidate ranking and generation scoring on Track S items",
    )
    ap.add_argument("--model", type=Path, required=True, help="a seed pack directory or an assembly manifest")
    ap.add_argument("--arm", required=True, help="the arm label (an MS §7 row id sets the routing preset)")
    ap.add_argument("--row", help="the matrix row the results belong to (default: the arm)")
    ap.add_argument("--replicate", type=int, default=0)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--items", type=Path, nargs="+", help="bilingual-item/1 files")
    src.add_argument("--prompts", type=Path, help="a fixed generation set: JSON lines {id, lang, text}")
    ap.add_argument("--family")
    ap.add_argument("--variant")
    ap.add_argument("--direction", help="keep one direction, e.g. zh>en (T1': one assembly per direction)")
    ap.add_argument("--split", default="dev", help="the only split evaluated (default dev)")
    ap.add_argument("--allow-test", action="store_true", help="evaluate test items (only after the MS §9.7 lock)")
    ap.add_argument("--route", choices=["R_out", "R_src", "R_wrong", "R_random"])
    ap.add_argument("--gate", choices=["one_hot", "dual_const", "dual_window"])
    ap.add_argument("--g", type=_pair, help="dual gates, e.g. 0.5,0.5")
    ap.add_argument("--window", type=int)
    ap.add_argument(
        "--answer-route", help="send the answer segments of translation items (E4) to this expert (T2: bridge)"
    )
    ap.add_argument("--fusion", help="override the assembly's fusion (B8 pre / B9-II post on one assembly)")
    ap.add_argument("--merge", choices=["average", "task_arithmetic", "ties"])
    ap.add_argument(
        "--lam",
        type=float,
        help=f"task arithmetic / TIES coefficient, required for them: a point of {MERGE['lam_grid']} chosen on dev",
    )
    ap.add_argument("--off-grid", action="store_true", help="accept a --lam outside the grid (recorded)")
    ap.add_argument("--density", type=float, default=MERGE["density"], help="TIES density")
    ap.add_argument("--merge-experts", default="zh,en")
    ap.add_argument("--norm", choices=NORMS)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--limit", type=int, help="evaluate only the first N items after the filters")
    ap.add_argument("--rng-seed", type=int, default=0, help="R_random seed (per item: (seed, item id))")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--autocast", choices=["none", "bf16"], default=DEFAULTS["eval"]["autocast"])
    ap.add_argument("--n-boot", type=int, default=DEFAULTS["eval"]["n_boot"])
    ap.add_argument("--out", type=Path, required=True, help="a directory: predictions.jsonl, results.json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="load and verify the model and the items; run nothing")
    return ap


def run(args) -> dict[str, Any]:
    from ..devices import resolve_device
    from ..registry.env import code_identity
    from .models import arm_routing, load_model
    from .report import results_document, results_from_predictions

    preset_routing, preset = arm_routing(
        args.arm,
        route=args.route,
        gate=args.gate,
        g=args.g,
        window=args.window,
        answer_route=args.answer_route,
        rng_seed=args.rng_seed,
    )
    if preset.get("needs_direction") and not args.direction:
        raise ValueError(
            f"arm {args.arm} evaluates one assembly per direction (the T1' copy in its target slot): give --direction"
        )
    merge_method = args.merge or preset.get("merge")
    merge = None
    if merge_method:
        merge = {"method": merge_method, "experts": args.merge_experts.split(",")}
        if merge_method != "average":
            if args.lam is None:
                raise ValueError(f"--merge {merge_method} needs --lam, a point of {MERGE['lam_grid']} chosen on dev")
            on_grid = any(abs(args.lam - x) <= 1e-12 for x in MERGE["lam_grid"])
            if not on_grid and not args.off_grid:
                raise ValueError(f"--lam {args.lam} is not a point of the grid {MERGE['lam_grid']} (--off-grid)")
            merge.update(lam=args.lam, grid=list(MERGE["lam_grid"]), on_grid=on_grid)
            if merge_method == "ties":
                merge["density"] = args.density
    ev = load_model(
        args.model, fusion=args.fusion or preset.get("fusion"), merge=merge, device=resolve_device(args.device)
    )
    if args.items:
        items, files = load_items(
            args.items,
            family=args.family,
            variant=args.variant,
            direction=args.direction,
            split=args.split,
            limit=args.limit,
        )
        if any(i["split"] == "test" for i in items) and not args.allow_test:
            raise ValueError("test items are evaluated only after the preregistration lock (MS §9.7): --allow-test")
    else:
        files = {str(args.prompts): None}
        items = [json.loads(line) for line in args.prompts.read_text("utf-8").splitlines() if line.strip()]
        items = items[: args.limit] if args.limit else items
    sha = items_sha256(items)
    norm = args.norm or DEFAULTS["eval"]["length_norm"]
    ctx = EvalContext(
        ev.model,
        ev.tokenizer,
        ev.names,
        preset_routing,
        args.arm,
        args.replicate,
        sha,
        norm,
        args.max_new,
        args.autocast == "bf16",
    )
    commit, dirty = code_identity()
    meta = {
        "model": ev.source,
        "tokenizer_sha256": ev.tokenizer_sha256,
        "item_files": files,
        "items_sha256": sha,
        "n_items": len(items),
        "arm": args.arm,
        "row": args.row or args.arm,
        "replicate": args.replicate,
        "routing": dict(zip(("route_policy", "gate_policy"), preset_routing.label(), strict=True)),
        "norm": norm,
        "max_new": args.max_new,
        "autocast": args.autocast,
        "split": args.split,
        "direction": args.direction,
        "allow_test": args.allow_test,
        "code_commit": commit,
        "code_dirty": dirty,
    }
    if args.dry_run:
        return {"dry_run": True, **meta, "experts": list(ev.names)}
    preds = generate_prompts(ctx, items) if args.prompts else evaluate(ctx, items)
    rows = [] if args.prompts else results_from_predictions(preds, row=args.row or args.arm, n_boot=args.n_boot)
    out = Path(args.out)
    if (out / "predictions.jsonl").exists() and not args.overwrite:
        raise FileExistsError(f"{out}/predictions.jsonl exists (--overwrite to replace it)")
    atomic_write_bytes(out / "predictions.jsonl", b"".join(canonical_json(p) + b"\n" for p in preds))
    atomic_write_json(out / "results.json", results_document(rows, **meta))
    skipped = sum(1 for p in preds if p.get("skip_reason"))
    return {**meta, "skipped": skipped, "results": [r for r in rows if r["metric"].count("/") == 2]}


def main(argv: Sequence[str]) -> int:
    args = build_parser().parse_args(list(argv))
    try:
        summary = run(args)
    except (OSError, ValueError, KeyError) as exc:  # AssemblyError is a ValueError
        print(f"eval-tasks: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=1, sort_keys=True, ensure_ascii=False, default=_json_default))
    return 0


def _json_default(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    return str(x)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
