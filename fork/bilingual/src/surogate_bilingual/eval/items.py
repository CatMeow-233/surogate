"""Track S items (`bilingual-item/1`) as token segments and route plans for evaluation (MS §6.2, §8.5, §9.1).

Tokenization follows the training documents (data.synth.build): every prompt segment is encoded as user text on its
own (segments end with "\\n", where the byte-level pre-tokenizer always splits), and the answer of a ranked item is
`answer_sep + candidate` appended to the question segment, as in the training QA documents where question and
answer form one segment. `candidate_ids` encodes question + answer jointly and keeps the shared question prefix;
if a tokenizer merged across that boundary (no frame in examples/track_s.toml lets it), the candidate is encoded on
its own and the record says so (`boundary_merge`).

Plans (`item_plan`) come from segment metadata through `routing.policies`: the prompt segments and one answer
segment of the longest candidate (or the generation budget) get their routes from the routing policy before the
answer starts (route causality, MS §5.4); then the gate policy of a dual arm applies. `answer_route` sends the answer
segment to a named expert (the Bridge arm T2 under variant (c): source on the source expert, target on the Bridge).
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..canonical import canonical_json, check_format, sha256_bytes, sha256_file, sha256_json
from ..data.tokenizer import encode_user_text
from ..routing.plan import PlanError, RoutePlan
from ..routing.policies import ROUTE_POLICIES, apply_gate_policy, plan_from_segments, segment_routes

RANKED_FAMILIES = ("E1", "E2", "E3par", "E3ctx")
GENERATED_FAMILIES = ("E4",)


class ItemError(ValueError):
    pass


def load_items(
    paths: Sequence[Path],
    *,
    family: str | None = None,
    variant: str | None = None,
    split: str | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """(items in file order after the filters, {path: sha256}); refuses lines that are not `bilingual-item/1`."""
    items, files = [], {}
    for path in paths:
        path = Path(path)
        files[str(path)] = sha256_file(path)
        for number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            try:
                check_format(item, "item")
            except ValueError as exc:
                raise ItemError(f"{path}:{number}: {exc}") from None
            if family and item["family"] != family or variant and item["variant"] != variant:
                continue
            if split and item["split"] != split:
                continue
            items.append(item)
    if limit is not None:
        items = items[:limit]
    return items, files


def items_sha256(items: Iterable[Mapping[str, Any]]) -> str:
    """The hash of the evaluated item list, in order (MS §9.1: every run hashes its item list)."""
    return sha256_json([sha256_bytes(canonical_json(dict(item))) for item in items])


def cluster_of(item: Mapping[str, Any]) -> str:
    """The bootstrap cluster of an item (MS §9.2: templates): its question template, else all its templates."""
    tids = item.get("template_ids") or {}
    if "question" in tids:
        return f"{item['family']}:{tids['question']}"
    return f"{item['family']}:" + "|".join(f"{k}={v}" for k, v in sorted(tids.items()))


@dataclass(frozen=True)
class Seg:
    lang: str
    role: str
    ids: tuple[int, ...]


def bos(tok, lang: str) -> Seg:
    """The <eos> every prompt starts with: in training every document but a shard's first follows an <eos>, so
    evaluation uses the same convention (IMPL-15). It takes the route of the first segment's language."""
    return Seg(lang, "bos", (int(tok.token_to_id("<eos>")),))


def prompt_segments(tok, item: Mapping[str, Any]) -> list[Seg]:
    """<eos>, then the text segments before the first answer, each encoded on its own."""
    out = []
    for seg in item["segments"]:
        if seg["role"] == "answer" or seg.get("text") is None:
            break
        out.append(Seg(seg["lang"], seg["role"], tuple(encode_user_text(tok, seg["text"]))))
    if not out:
        raise ItemError(f"{item['id']}: no prompt segment")
    return [bos(tok, out[0].lang), *out]


def candidate_ids(tok, question_text: str, answer_sep: str, candidate: str) -> tuple[list[int], bool]:
    """(candidate ids, boundary_merge): the ids `answer_sep + candidate` adds after the question segment."""
    prefix = encode_user_text(tok, question_text)
    joint = encode_user_text(tok, question_text + answer_sep + candidate)
    if joint[: len(prefix)] == prefix and len(joint) > len(prefix):
        return joint[len(prefix) :], False
    return encode_user_text(tok, answer_sep + candidate), True


def item_rng(seed: int, item_id: str) -> np.random.Generator:
    """The R_random generator of one item: a function of (seed, item id), so arms and reruns agree."""
    return np.random.default_rng([int(seed), zlib.crc32(item_id.encode("utf-8"))])


@dataclass(frozen=True)
class Routing:
    """How an arm routes: a routing policy (None for a single-expert model), a gate policy and its parameters."""

    route: str | None = "R_out"
    gate: str = "one_hot"
    g: tuple[float, float] = (0.5, 0.5)
    window: int = 32
    answer_route: str | None = None
    rng_seed: int = 0

    def label(self) -> tuple[str, str]:
        gate = (
            self.gate
            if self.gate == "one_hot"
            else f"{self.gate}({self.g[0]:g},{self.g[1]:g}"
            + (f",w={self.window})" if self.gate == "dual_window" else ")")
        )
        route = self.route or "single"
        if self.answer_route:
            route += f"+answer>{self.answer_route}"
        return route, gate


class SkipItem(Exception):
    """An item the arm cannot score; the reason goes into the prediction record (IMPL-15)."""


def item_plan(
    item: Mapping[str, Any],
    segs: Sequence[Seg],
    answer_len: int,
    answer_lang: str,
    names: Sequence[str],
    routing: Routing,
) -> RoutePlan | None:
    """The [1, P + answer_len, E] plan of a prompt followed by one answer segment; None for a single-expert model."""
    if len(names) == 1:
        return None
    if routing.route not in ROUTE_POLICIES:
        raise SkipItem(f"a model with experts {tuple(names)} needs a routing policy, got {routing.route!r}")
    meta = [(s.lang, len(s.ids), s.role) for s in segs] + [(answer_lang, int(answer_len), "answer")]
    source = item.get("knowledge_source")
    if routing.route == "R_src" and source is None:
        why = ("its facts live in two partitions (knowledge_sources), R_src is undefined"
               if item.get("knowledge_sources") else "it has no knowledge_source")  # fmt: skip
        raise SkipItem(f"R_src cannot route {item['id']}: {why}")

    def rng():
        return item_rng(routing.rng_seed, item["id"]) if routing.route == "R_random" else None

    wanted = set(segment_routes(meta, routing.route, knowledge_source=source, rng=rng()))
    if routing.answer_route:
        wanted.add(routing.answer_route)
    missing = sorted(wanted - set(names))
    if missing:
        raise SkipItem(f"missing expert {missing} (the model has {tuple(names)})")
    plan = plan_from_segments(meta, names, routing.route, knowledge_source=source, rng=rng())
    if routing.answer_route:  # every answer segment (earlier stages included) and the new answer
        gates = plan.gates.clone()
        target = list(names).index(routing.answer_route)
        start = 0
        for _lang, n, role in meta:
            if role == "answer":
                gates[:, start : start + n] = 0.0
                gates[:, start : start + n, target] = 1.0
            start += n
        plan = RoutePlan(plan.names, gates, f"{plan.policy}+answer>{routing.answer_route}")
    if routing.gate != "one_hot":
        try:
            plan = apply_gate_policy(plan, routing.gate, routing.g, routing.window)
        except PlanError as exc:
            raise SkipItem(f"gate policy {routing.gate}: {exc}") from None
    plan.validate(names)
    return plan


def extend_plan(plan: RoutePlan | None, length: int) -> RoutePlan | None:
    """The plan cut or extended (by repeating its last row) to `length` positions."""
    if plan is None:
        return None
    t = plan.gates.shape[1]
    if length <= t:
        return plan.slice(0, length)
    tail = plan.gates[:, -1:].expand(-1, length - t, -1)
    return RoutePlan(plan.names, torch.cat([plan.gates, tail], dim=1), plan.policy)
