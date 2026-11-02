"""The budget rows of MS §5.1, recomputed from module trees on the meta device (`bilingual-budget/1`).

For every row the code builds the row's model on the meta device (no memory) and counts:

- P_total with `p_total` over the model plus any per-layer extra modules of the row;
- P_update with `p_update` over an AdamW built on the row's trainable set (one expert, or the extra modules);
- P_active by running the forward under the `p_active` recorder with the row's plan (`method = "executed"`). Rows
  whose extra modules have no forward before P1 (the B10-ch block router, the B11 adapters, the router of the 2B
  extrapolation) add those modules' parameters to the executed count, or count the selected channel blocks, by
  declaration (`method = "structural"`);
- weight MACs per token with `macs_per_token`, and the same at T = 1024.

`within_*` compare with the 100M budgets; `declared_in_budget` is whether MS §4.6 puts the row on the hard-budget
line. Dual rows are on the capability line and are reported over budget, never asserted within it (MA-12).

Command line: `python -m surogate_bilingual audit [--profile pilot] [--out budget.json] [--dry-run]`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from torch import nn

from ..canonical import atomic_write_json
from ..config import PROFILES, ModelConfig, profile
from ..model.lm import RoutedLM
from ..routing.plan import RoutePlan
from ..routing.policies import apply_gate_policy
from ..train.recipes import ExpertRecipe
from .flops import macs_per_token
from .params import p_active, p_total, p_update

BUDGET_FORMAT = "bilingual-budget/1"
ACTIVE_BUDGET = 100_000_000
UPDATE_BUDGET = 100_000_000
AUDIT_T = 8  # tokens of the executed meta forward
BLOCK = 64  # B10-ch channel block width
BLOCK_ROUTER_RANK = 8
TWO_B = {"vocab_size": 65536, "ffn_width": 1328, "experts": 40}  # MS §5.2 reconstruction


@dataclass(frozen=True)
class BudgetRow:
    id: str
    label: str
    family: str
    P_total: int
    P_active: int
    P_update: int | None
    weight_macs: int
    macs_T1024: int
    within_active_budget: bool
    within_update_budget: bool | None
    declared_in_budget: bool
    method: str
    standalone_P_total: int | None = None
    P_active_dual: int | None = None
    note: str = ""


def _model(cfg: ModelConfig, names: Sequence[str], fusion: str | None = None) -> RoutedLM:
    return RoutedLM(cfg, names, fusion, device="meta")


def _plan(model: RoutedLM, kind: str) -> RoutePlan:
    names = model.expert_names
    plan = RoutePlan.forced(names[0], names, 1, AUDIT_T)
    if kind == "dual":
        plan = apply_gate_policy(plan, "dual_const", pair=(names[0], names[1]))
    return plan


def _active(model: RoutedLM, kind: str) -> int:
    tokens = torch.zeros(1, AUDIT_T, dtype=torch.long, device="meta")
    return p_active(model, tokens, _plan(model, kind))


def _update(model: RoutedLM, extra: nn.Module | None = None) -> int:
    params = (
        list(extra.parameters())
        if extra is not None
        else ExpertRecipe(model.expert_names[0]).freeze_plan(model).parameters()
    )
    return p_update(torch.optim.AdamW(params, lr=1e-3))


def _per_layer(cfg: ModelConfig, make) -> nn.ModuleList:
    with torch.device("meta"):
        return nn.ModuleList(make() for _ in range(cfg.n_layers))


def _linear_chain(*dims: int):
    return lambda: nn.Sequential(*(nn.Linear(a, b, bias=False) for a, b in zip(dims, dims[1:], strict=False)))


def _row(rid, label, cfg, total, active, update, kind, declared, method, *, extra_macs=0, **more) -> BudgetRow:
    return BudgetRow(
        id=rid,
        label=label,
        family=cfg.family,
        P_total=total,
        P_active=active,
        P_update=update,
        weight_macs=macs_per_token(cfg, kind, extra_per_layer=extra_macs),
        macs_T1024=macs_per_token(cfg, kind, 1024, extra_per_layer=extra_macs),
        within_active_budget=active <= ACTIVE_BUDGET,
        within_update_budget=None if update is None else update <= UPDATE_BUDGET,
        declared_in_budget=declared,
        method=method,
        **more,
    )


def _bank_rows(rid, label, cfg, names, kind, declared, **more) -> BudgetRow:
    model = _model(cfg, names)
    return _row(
        rid, label, cfg, p_total(model), _active(model, kind), _update(model), kind, declared, "executed", **more
    )


def budget_rows(profile_name: str = "pilot") -> list[BudgetRow]:
    cfg1, cfg2 = profile(profile_name, "swiglu"), profile(profile_name, "slot")
    L, d, f = cfg1.n_layers, cfg1.d_model, cfg1.ffn_width
    zh_en = ["zh", "en"]
    rows = [
        _bank_rows("B4", "B1/B4 family I Top-1 (ZH+EN bank)", cfg1, zh_en, "single", True),
        _bank_rows("B6/B7", "B6/B7 family I dual", cfg1, zh_en, "dual", False),
        _bank_rows("ZH+EN+Bridge", "ZH+EN+Bridge Top-1", cfg1, zh_en + ["bridge"], "single", True),
        _bank_rows("31-paths", "31 paths Top-1 (no router)", cfg1, [f"p{i:02d}" for i in range(31)], "single", True),
    ]
    half = cfg1.replace(ffn_width=f // 2)
    rows.append(
        _bank_rows(
            "B10-I", "B10-I Top-2 of two f/2 experts", half, zh_en, "dual", True,
            standalone_P_total=p_total(_model(half, ["zh"])), note="fidelity reference: the standalone half model",
        )
    )  # fmt: skip

    bank = _model(cfg1, zh_en)
    shared = p_total(bank.backbone)
    path = p_total(bank.bank.experts["zh"])
    if f % BLOCK == 0:
        blocks = f // BLOCK
        router = _per_layer(cfg1, _linear_chain(d, BLOCK_ROUTER_RANK, 2 * blocks))
        selected = (
            L * blocks * BLOCK * 3 * d
        )  # `blocks` of the 2 * blocks channel blocks, each 64 rows/cols of gate, up, down
        rows.append(
            _row(
                "B10-ch", f"B10-ch {blocks} of {2 * blocks} blocks (rank-{BLOCK_ROUTER_RANK} block router)", cfg1,
                p_total(bank) + p_total(router), shared + selected + p_total(router), path, "single", True,
                "structural", extra_macs=BLOCK_ROUTER_RANK * (d + 2 * blocks),
                note="channel selection and the block router run in P1; counted by declaration",
            )
        )  # fmt: skip
    single = shared + path
    single_active = rows[0].P_active  # B4: the executed Top-1 path of this bank
    ranks = [128]
    if single < ACTIVE_BUDGET:
        ranks.insert(0, math.floor((ACTIVE_BUDGET - single) / (2 * d * L)))
    for r in ranks:
        adapters = _per_layer(cfg1, _linear_chain(d, r, d))
        rows.append(
            _row(
                f"B11-r{r}", f"B11 Top-1 + zero-init adapter r={r}", cfg1, p_total(bank) + p_total(adapters),
                single_active + p_total(adapters), _update(bank, adapters), "single", True, "structural",
                extra_macs=2 * d * r, note="adapters run in P1 (repair ladder); counted by declaration",
            )
        )  # fmt: skip
    for k in (2, 4, 6):
        if 2 * k >= L:
            continue
        lp = cfg1.replace(expert_layers=tuple(range(k)) + tuple(range(L - k, L)))
        model = _model(lp, zh_en)
        rows.append(
            _row(
                f"LP-{k}", f"LP-{k} layer-partial Top-1", lp, p_total(model), _active(model, "single"),
                _update(model), "single", True, "executed", P_active_dual=_active(model, "dual"),
            )
        )  # fmt: skip
    # B10-II halves the slots, rounding up: n = 395 at the pilot geometry (B-O4).
    for n, tag, declared_dual in ((cfg2.slot_width, "II", False), ((cfg2.slot_width + 1) // 2, "B10-II", True)):
        cfg = cfg2.replace(slot_width=n)
        model = _model(cfg, zh_en)
        stand = p_total(_model(cfg, ["zh"]))
        rows.append(_row(f"{tag}-single", f"{tag} n={n} Top-1", cfg, p_total(model), _active(model, "single"),
                         _update(model), "single", True, "executed"))  # fmt: skip
        rows.append(_row(f"{tag}-dual", f"{tag} n={n} dual (B8 pre / B9 post)", cfg, p_total(model),
                         _active(model, "dual"), _update(model), "dual", declared_dual, "executed",
                         standalone_P_total=stand))  # fmt: skip
    if d < f:
        cfg = cfg2.replace(slot_width=d)
        model = _model(cfg, zh_en)
        note = "sensitivity row of budget_v11.json; not in the MS §5.1 table"
        rows.append(_row(f"II-n{d}-single", f"II n={d} Top-1", cfg, p_total(model), _active(model, "single"),
                         _update(model), "single", True, "executed", note=note))  # fmt: skip
        rows.append(_row(f"II-n{d}-dual", f"II n={d} dual", cfg, p_total(model), _active(model, "dual"),
                         _update(model), "dual", False, "executed", note=note))  # fmt: skip
    if profile_name == "pilot":
        big = cfg1.replace(vocab_size=TWO_B["vocab_size"], ffn_width=TWO_B["ffn_width"])
        model = _model(big, [f"e{i:02d}" for i in range(TWO_B["experts"])])
        router = _per_layer(big, _linear_chain(d, TWO_B["experts"]))
        rows.append(
            _row(
                "2B", f"2B extrapolation (V {big.vocab_size}, {TWO_B['experts']} FFNs of f={big.ffn_width} per layer, "
                f"{d}x{TWO_B['experts']} router per layer)", big, p_total(model) + p_total(router),
                _active(model, "single") + p_total(router), None, "single", True, "structural",
                extra_macs=d * TWO_B["experts"],
                note="the v1.0 §5.2 organisation; P_update undefined (not whole-path experts); router by declaration",
            )
        )  # fmt: skip
    return rows


def report(profile_name: str = "pilot") -> dict:
    rows = budget_rows(profile_name)
    by_id = {r.id: r for r in rows}
    cfg1, cfg2 = profile(profile_name, "swiglu"), profile(profile_name, "slot")
    bank1, bank2 = _model(cfg1, ["zh", "en"]), _model(cfg2, ["zh", "en"])
    shared, path = p_total(bank1.backbone), p_total(bank1.bank.experts["zh"])
    b4, dual1 = by_id["B4"], by_id["B6/B7"]
    single2, dual2 = by_id["II-single"], by_id["II-dual"]
    return {
        "format": BUDGET_FORMAT,
        "profile": profile_name,
        "budgets": {"active": ACTIVE_BUDGET, "update": UPDATE_BUDGET},
        "totals": {
            "shared": shared,
            "path": path,
            "single": shared + path,
            "zh_en": shared + 2 * path,
            "slack_top1": ACTIVE_BUDGET - shared - path,
            "shared_II": p_total(bank2.backbone),
            "readouts_II": p_total(bank2.backbone) - shared,
            "path_II": p_total(bank2.bank.experts["zh"]),
        },
        "ratios": {
            "dual_I_weight": dual1.weight_macs / b4.weight_macs,
            "dual_I_T1024": dual1.macs_T1024 / b4.macs_T1024,
            "dual_II_vs_own_single": dual2.weight_macs / single2.weight_macs,
            "dual_II_vs_B4": dual2.weight_macs / b4.weight_macs,
            "dual_II_vs_B4_T1024": dual2.macs_T1024 / b4.macs_T1024,
        },
        "rows": [asdict(r) for r in rows],
    }


def markdown(doc: dict) -> str:
    def num(x):
        return "—" if x is None else f"{x:,}"

    lines = [
        f"budget ({doc['profile']} profile; P = parameters, MACs per token)",
        "",
        "| row | P_total | P_active | P_update | <=100M active | weight MACs | MACs T=1024 | declared | method |",
        "|---|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for r in doc["rows"]:
        lines.append(
            f"| {r['id']} | {num(r['P_total'])} | {num(r['P_active'])} | {num(r['P_update'])} | "
            f"{'yes' if r['within_active_budget'] else 'no'} | {num(r['weight_macs'])} | {num(r['macs_T1024'])} | "
            f"{'in budget' if r['declared_in_budget'] else 'over budget'} | {r['method']} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual audit", description="budget and parameter report")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="pilot")
    ap.add_argument("--out", help="write the bilingual-budget/1 JSON here")
    ap.add_argument("--json", action="store_true", help="print JSON instead of the markdown table")
    ap.add_argument("--dry-run", action="store_true", help="compute and print; write nothing")
    args = ap.parse_args(list(argv))
    doc = report(args.profile)
    if args.out and not args.dry_run:
        atomic_write_json(args.out, doc)
    print(json.dumps(doc, indent=1, sort_keys=True) if args.json else markdown(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
