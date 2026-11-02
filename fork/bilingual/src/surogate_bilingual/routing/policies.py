"""Route plans from segment metadata (MS §8.5) and the gate policies of the dual arms (B-O5).

A sequence is a list of segments `(lang, n_tokens, role)`; roles follow the Track S items (`fact`, `rule`,
`question`, `answer`, ...). Every policy assigns token t from the metadata of its own segment and the segments
before it, so the plan is causal: nothing after t decides t's route.

Routing policies (one-hot plans):

- `R_out` (default): each segment takes its own language; the `answer` segment takes the language of the most
  recent `question` segment, which is known before the answer starts (MS §6.4).
- `R_src`: every token goes to `knowledge_source`, the expert whose partition holds the facts. A leaking oracle,
  reported as an upper bound only. [假设] routing the whole item (prompt and answer) to the source is this
  implementation's reading of "route by knowledge source"; the owner can narrow it to the answer segment.
- `R_wrong`: the R_out route with zh and en swapped, on every segment.
- `R_random`: each segment draws zh or en uniformly from `rng` (one draw per segment, in segment order).

Gate policies (`apply_gate_policy`) turn a one-hot plan over a language pair into the plan a dual arm uses:
`one_hot` keeps it, `dual_const` gives every token the gates g on the pair, and `dual_window` gives g only to the
`window` tokens that start at each route switch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .plan import PlanError, RoutePlan

ROUTE_POLICIES = ("R_out", "R_src", "R_wrong", "R_random")
GATE_POLICIES = ("one_hot", "dual_const", "dual_window")
PAIR = ("zh", "en")


def _swap(lang: str, pair: tuple[str, str]) -> str:
    if lang not in pair:
        raise PlanError(f"R_wrong swaps {pair[0]} and {pair[1]}; a segment is routed to {lang!r}")
    return pair[1] if lang == pair[0] else pair[0]


def segment_routes(
    segments: Sequence[tuple[str, int, str]],
    policy: str,
    *,
    knowledge_source: str | None = None,
    rng: np.random.Generator | None = None,
    pair: tuple[str, str] = PAIR,
) -> list[str]:
    """The expert name of each segment under a routing policy."""
    if policy not in ROUTE_POLICIES:
        raise PlanError(f"unknown routing policy {policy!r}; known: {ROUTE_POLICIES}")
    if policy == "R_src":
        if knowledge_source is None:
            raise PlanError("R_src needs the item's knowledge_source (parametric items only)")
        return [knowledge_source] * len(segments)
    if policy == "R_random":
        if rng is None:
            raise PlanError("R_random needs an explicit numpy Generator")
        return [pair[int(rng.integers(2))] for _ in segments]
    routes, question = [], None
    for lang, _, role in segments:
        route = question if role == "answer" and question is not None else lang
        if role == "question":
            question = lang
        routes.append(route)
    if policy == "R_wrong":
        routes = [_swap(r, pair) for r in routes]
    return routes


def plan_from_segments(
    segments: Sequence[tuple[str, int, str]],
    names: Sequence[str],
    policy: str,
    *,
    knowledge_source: str | None = None,
    rng: np.random.Generator | None = None,
) -> RoutePlan:
    """A [1, T, E] one-hot plan, T = the sum of the segment lengths."""
    for lang, n, role in segments:
        if not isinstance(n, int) or n < 0:
            raise PlanError(f"segment ({lang!r}, {n!r}, {role!r}) needs a non-negative token count")
    routes = segment_routes(segments, policy, knowledge_source=knowledge_source, rng=rng)
    unknown = sorted(set(routes) - set(names))
    if unknown:
        raise PlanError(f"policy {policy} routes to {unknown}, which are not in {tuple(names)}")
    index = {name: i for i, name in enumerate(names)}
    ids = [index[route] for route, (_, n, _) in zip(routes, segments, strict=True) for _ in range(n)]
    plan = RoutePlan.one_hot(torch.tensor([ids], dtype=torch.long).reshape(1, -1), names, policy)
    plan.validate()
    return plan


def apply_gate_policy(
    plan: RoutePlan,
    policy: str,
    g: tuple[float, float] = (0.5, 0.5),
    window: int = 32,
    pair: tuple[str, str] = PAIR,
) -> RoutePlan:
    """The gates of a dual arm over `pair`, from a one-hot plan whose every token is on one of the pair."""
    if policy not in GATE_POLICIES:
        raise PlanError(f"unknown gate policy {policy!r}; known: {GATE_POLICIES}")
    if policy == "one_hot":
        if not bool(plan.is_one_hot().all()):
            raise PlanError("gate policy one_hot needs a one-hot plan")
        return plan
    g0, g1 = (float(x) for x in g)
    if g0 <= 0 or g1 <= 0 or abs(g0 + g1 - 1.0) > 1e-12:
        raise PlanError(f"dual gates must be two positive weights summing to 1, got {g}")
    if window <= 0:
        raise PlanError(f"window must be positive, got {window}")
    a, b = (list(plan.names).index(p) if p in plan.names else -1 for p in pair)
    if a < 0 or b < 0:
        raise PlanError(f"the plan has no expert pair {pair}: {plan.names}")
    ids = plan.route_ids()
    if not bool(((ids == a) | (ids == b)).all()):
        raise PlanError(f"dual gate policies apply to plans routed within {pair} only")
    dual = torch.zeros_like(plan.gates)
    dual[..., a], dual[..., b] = g0, g1
    if policy == "dual_const":
        mask = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
    else:
        mask = torch.zeros(ids.shape, dtype=torch.bool, device=ids.device)
        switches = torch.nonzero(ids[:, 1:] != ids[:, :-1])
        for row, col in switches.tolist():
            mask[row, col + 1 : col + 1 + window] = True
    gates = torch.where(mask[..., None], dual, plan.gates)
    label = f"{plan.policy}+{policy}({g0:g},{g1:g}" + (f",w={window})" if policy == "dual_window" else ")")
    out = RoutePlan(plan.names, gates, label)
    out.validate()
    return out
