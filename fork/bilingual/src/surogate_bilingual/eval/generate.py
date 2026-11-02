"""Batched greedy decoding with full-history route plans and route traces (MS §8.3; MA-09, IMPL-15).

No KV cache (deferred, IMPL-17): every step recomputes the window of the last `max_seq` tokens. The route plan of a
row is built over its WHOLE history by `plan_fn(row, ids)` and the window takes the matching slice, so a tag or a
segment boundary that scrolled out of the window still decides the route (the v1.0 bug recomputed routes on the
truncated window and fell back to zh). Rows of a batch are right-padded: under causal attention a padded position
never reaches a real one, and each row reads its logits at its own last real position, so batched decoding equals
per-row decoding up to floating-point summation order.

`Generation.route_trace` is the plan of the final sequence (prompt plus generated tokens) as expert names (one-hot
rows) or gate lists (dual rows); it is part of the decoding state, not recomputed afterwards on a window.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from ..routing.plan import RoutePlan

PlanFn = Callable[[int, list[int]], RoutePlan | None]


@dataclass
class Generation:
    prompt: list[int]
    tokens: list[int]  # generated ids, the stopping <eos> left out
    stop: str  # "eos" | "max_new"
    route_trace: list | None  # per position of prompt + tokens: an expert name, or the gate row of a dual token
    step_logits: list[torch.Tensor] | None = None  # [V] fp32 rows, one per generated token, when kept


def trace_of(plan: RoutePlan | None) -> list | None:
    if plan is None:
        return None
    gates = plan.gates[0]
    one_hot = plan.is_one_hot()[0]
    out = []
    for t in range(gates.shape[0]):
        if bool(one_hot[t]):
            out.append(plan.names[int(gates[t].argmax())])
        else:
            out.append([float(x) for x in gates[t]])
    return out


def _window_plans(plans: Sequence[RoutePlan | None], starts: Sequence[int], ends: Sequence[int], width: int):
    if plans[0] is None:
        return None
    rows = []
    for plan, s, e in zip(plans, starts, ends, strict=True):
        g = plan.gates[0, s:e]
        if e - s < width:
            g = torch.cat([g, g[-1:].expand(width - (e - s), -1)], dim=0)
        rows.append(g)
    return RoutePlan(plans[0].names, torch.stack(rows), plans[0].policy)


@torch.no_grad()
def window_logits(model, histories: Sequence[list[int]], plans: Sequence[RoutePlan | None], pad: int) -> torch.Tensor:
    """fp32 logits [B, V] of the next token after each history, each on its last `max_seq` tokens."""
    max_seq = model.cfg.max_seq
    starts = [max(0, len(h) - max_seq) for h in histories]
    ends = [len(h) for h in histories]
    width = max(e - s for s, e in zip(starts, ends, strict=True))
    device = next(model.parameters()).device
    tokens = torch.full((len(histories), width), pad, dtype=torch.long)
    for i, (h, s) in enumerate(zip(histories, starts, strict=True)):
        tokens[i, : len(h) - s] = torch.tensor(h[s:], dtype=torch.long)
    plan = _window_plans(plans, starts, ends, width)
    if plan is not None:
        plan = RoutePlan(plan.names, plan.gates.to(device), plan.policy)
    logits = model(tokens.to(device), plan)
    last = torch.tensor([e - s - 1 for s, e in zip(starts, ends, strict=True)], device=device)
    return logits[torch.arange(len(histories), device=device), last].float().cpu()


def greedy_batch(
    model,
    prompts: Sequence[Sequence[int]],
    plan_fn: PlanFn,
    max_new: int,
    eos: int,
    *,
    keep_logits: bool = False,
    batch_size: int = 32,
) -> list[Generation]:
    """Greedy continuation of each prompt until <eos> or `max_new` tokens. `plan_fn(i, ids)` returns the [1, len(ids),
    E] plan of row i over its full history (None for a single-expert model)."""
    histories = [list(map(int, p)) for p in prompts]
    if any(not h for h in histories):
        raise ValueError("every prompt needs at least one token")
    new: list[list[int]] = [[] for _ in histories]
    logits_kept: list[list[torch.Tensor]] = [[] for _ in histories]
    stop = ["max_new"] * len(histories)
    active = [i for i in range(len(histories))]
    for _ in range(int(max_new)):
        if not active:
            break
        still = []
        for start in range(0, len(active), batch_size):
            rows = active[start : start + batch_size]
            plans = [plan_fn(i, histories[i]) for i in rows]
            for i, plan in zip(rows, plans, strict=True):
                if plan is not None and tuple(plan.gates.shape[:2]) != (1, len(histories[i])):
                    raise ValueError(f"plan_fn({i}) covers {list(plan.gates.shape[:2])}, not [1, {len(histories[i])}]")
            out = window_logits(model, [histories[i] for i in rows], plans, eos)
            for i, row in zip(rows, out, strict=True):
                token = int(row.argmax())
                if keep_logits:
                    logits_kept[i].append(row.clone())
                if token == eos:
                    stop[i] = "eos"
                    continue
                histories[i].append(token)
                new[i].append(token)
                still.append(i)
        active = sorted(still)
    out = []
    for i, prompt in enumerate(prompts):
        plan = plan_fn(i, histories[i])
        out.append(
            Generation(list(map(int, prompt)), new[i], stop[i], trace_of(plan), logits_kept[i] if keep_logits else None)
        )
    return out
