"""Parameter, MAC and budget accounting (docs/spec.md, I2-A12 to I2-A15; IMPL-21, MA-05, MA-12).

The exact numbers are MS §5.1 and the reviewer's closed-form `budget_v11.json`; here the code recounts them from
module trees on the meta device and from executed forwards.
"""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from surogate_bilingual.__main__ import main as cli
from surogate_bilingual.audit.flops import counted_flops, linear_flops, macs_per_token
from surogate_bilingual.audit.params import p_active, p_total, p_update
from surogate_bilingual.config import profile
from surogate_bilingual.model.lm import RoutedLM
from surogate_bilingual.ops.reads import declare_partial_read
from surogate_bilingual.routing.plan import RoutePlan
from surogate_bilingual.routing.policies import apply_gate_policy

GOLDEN = Path(__file__).resolve().parent / "golden" / "budget_pilot.json"


# --- I2-A12: exact budgets ---------------------------------------------------------------------------------------


def test_i2_a12_pilot_budgets_are_exact(pilot):
    doc, rows = pilot
    totals = doc["totals"]
    assert totals["shared"] == 32_530_944
    assert totals["path"] == 63_700_992
    assert totals["single"] == rows["B4"]["P_active"] == 96_231_936
    assert totals["zh_en"] == rows["B4"]["P_total"] == 159_932_928
    assert totals["slack_top1"] == 3_768_064
    assert rows["ZH+EN+Bridge"]["P_total"] == 223_633_920
    assert rows["31-paths"]["P_total"] == 2_007_261_696
    assert (rows["II-single"]["P_active"], rows["II-dual"]["P_active"], rows["II-dual"]["P_update"]) == (
        96_222_720,
        138_639_360,
        42_416_640,
    )
    assert (rows["B10-II-dual"]["P_active"], rows["B10-II-dual"]["standalone_P_total"]) == (96_276_480, 75_041_280)
    assert (rows["B10-I"]["P_active"], rows["B10-I"]["standalone_P_total"]) == (96_231_936, 64_381_440)
    assert rows["B10-ch"]["P_active"] == 96_340_608
    assert (rows["B11-r128"]["P_active"], rows["B11-r153"]["P_active"]) == (99_377_664, 99_992_064)
    assert (rows["LP-4"]["P_update"], rows["LP-4"]["P_total"]) == (21_233_664, 117_465_600)
    assert (rows["2B"]["P_total"], rows["2B"]["P_active"]) == (2_008_015_360, 98_755_072)


# budget_v11.json (closed form, CPU arithmetic), row by row: P_total, P_active, P_update, weight MACs, MACs at T=1024.
BUDGET_V11 = {
    "B4": (159932928, 96231936, 63700992, 96206848, 108789760),
    "B6/B7": (159932928, 159932928, 63700992, 159907840, 172490752),
    "ZH+EN+Bridge": (223633920, 96231936, 63700992, 96206848, 108789760),
    "31-paths": (2007261696, 96231936, 63700992, 96206848, 108789760),
    "B10-I": (96231936, 96231936, 31850496, 96206848, 108789760),
    "B10-ch": (160041600, 96340608, 63700992, 96206848, 108789760),
    "B11-r153": (163693056, 99992064, 3760128, 99966976, 112549888),
    "B11-r128": (163078656, 99377664, 3145728, 99352576, 111935488),
    "LP-2": (106848768, 96231936, 10616832, 96206848, 108789760),
    "LP-4": (117465600, 96231936, 21233664, 96206848, 108789760),
    "LP-6": (128082432, 96231936, 31850496, 96206848, 108789760),
    "II-single": (138639360, 96222720, 42416640, 96156160, 108739072),
    "II-dual": (138639360, 138639360, 42416640, 138572800, 151155712),
    "B10-II-single": (96276480, 75041280, 21235200, 74974720, 87557632),
    "B10-II-dual": (96276480, 96276480, 21235200, 96209920, 108792832),
    "II-n512-single": (108856320, 81331200, 27525120, 81264640, 93847552),
    "II-n512-dual": (108856320, 108856320, 27525120, 108789760, 121372672),
    # B12a (CR1): one 2f-wide expert; P_total = shared + 2 paths, all of it active, 2 paths updated
    "B12a": (159932928, 159932928, 127401984, 159907840, 172490752),
}
B10_CH_ROUTER_MACS = 24 * 8 * (512 + 54)  # budget_v11.json leaves the rank-8 block router out of B10-ch's MACs


def test_i2_a12_every_row_matches_the_closed_form(pilot):
    _, rows = pilot
    for rid, (total, active, update, macs, macs_1024) in BUDGET_V11.items():
        row = rows[rid]
        assert (row["P_total"], row["P_active"], row["P_update"]) == (total, active, update), rid
        extra = B10_CH_ROUTER_MACS if rid == "B10-ch" else 0
        assert (row["weight_macs"], row["macs_T1024"]) == (macs + extra, macs_1024 + extra), rid
    assert rows["LP-4"]["P_active_dual"] == 117_465_600
    assert set(rows) == set(BUDGET_V11) | {"2B"}


def test_i2_a12_only_dual_rows_and_b12a_exceed_the_active_budget(pilot):
    _, rows = pilot
    over = {rid for rid, row in rows.items() if not row["within_active_budget"]}
    assert over == {"B6/B7", "II-dual", "II-n512-dual", "B12a"}  # the dual rows and the B12a ceiling
    assert over == {rid for rid, row in rows.items() if not row["declared_in_budget"]}
    for rid, row in rows.items():
        if row["declared_in_budget"]:  # only rows declared in budget are asserted within it (MA-12)
            assert row["within_active_budget"] and row["within_update_budget"] in (True, None), rid
    assert {rid for rid, row in rows.items() if rid != "2B" and not row["within_update_budget"]} == {"B12a"}


# --- I2-A13: executed P_active -----------------------------------------------------------------------------------


def smoke_bank(family, names=("zh", "en", "bridge")):
    return RoutedLM(profile("smoke", family), names).eval()


def test_i2_a13_executed_p_active_counts_the_path_and_the_tied_head_once():
    model = smoke_bank("swiglu")
    cfg = model.cfg
    shared, path = p_total(model.backbone), p_total(model.bank.experts["zh"])
    attn = 2 * cfg.d_model * cfg.d_model + 2 * cfg.d_model * cfg.n_kv_heads * cfg.head_dim
    assert shared == cfg.vocab_size * cfg.d_model + cfg.n_layers * (attn + 2 * cfg.d_model) + cfg.d_model  # head once
    x = torch.randint(5, cfg.vocab_size, (2, 9))
    names = model.expert_names
    assert p_active(model, x, RoutePlan.forced("zh", names, 2, 9)) == shared + path
    routes = torch.tensor([[0, 1] * 4 + [0], [1] * 9])
    mixed = RoutePlan.one_hot(routes, names, "zh/en")
    assert p_active(model, x, mixed, per_token=False) == shared + 2 * path  # the union; bridge is never read
    assert p_active(model, x, mixed) == shared + path  # what one token reads
    assert p_total(model) == shared + 3 * path


@pytest.mark.parametrize("fusion", ["pre", "post"])
def test_i2_a13_slot_dual_plan_reads_the_shared_readouts_and_both_experts(fusion):
    model = RoutedLM(profile("smoke", "slot"), ("zh", "en", "bridge"), fusion).eval()
    shared_ii, expert = p_total(model.backbone), p_total(model.bank.experts["zh"])
    x = torch.randint(5, model.cfg.vocab_size, (1, 8))
    dual = apply_gate_policy(RoutePlan.forced("zh", model.expert_names, 1, 8), "dual_const")
    assert p_active(model, x, dual) == shared_ii + 2 * expert
    assert p_active(model, x, RoutePlan.forced("en", model.expert_names, 1, 8)) == shared_ii + expert


def test_i2_a13_operators_declare_partial_reads():
    class Sliced(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(4, 6))
            self.v = nn.Parameter(torch.ones(6))
            self.unused = nn.Parameter(torch.ones(100))

        def forward(self, tokens, plan):
            half = self.w[:, :3]  # the recorder sees the whole Parameter go into the slicing call
            declare_partial_read(self.w, half.numel())
            return tokens.float()[:, :4] @ half + self.v[:3]

    module = Sliced()
    assert p_active(module, torch.ones(2, 4), None) == 12 + 6
    assert p_total(module) == 24 + 6 + 100
    declare_partial_read(module.w, 1)  # outside an audit: no effect, no error


def test_p_update_counts_unique_storages():
    w = nn.Parameter(torch.ones(3, 4))
    alias = nn.Parameter(w.detach())
    assert p_update(torch.optim.SGD([w, alias, nn.Parameter(torch.ones(5))], lr=0.1)) == 12 + 5


# --- I2-A14: MACs ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["swiglu", "slot"])
@pytest.mark.parametrize("kind", ["single", "dual"])
def test_i2_a14_flop_counter_equals_twice_the_analytic_macs_on_smoke(family, kind):
    model = smoke_bank(family)
    x = torch.randint(5, model.cfg.vocab_size, (2, 8))
    plan = RoutePlan.forced("zh", model.expert_names, 2, 8)
    if kind == "dual":
        plan = apply_gate_policy(plan, "dual_const")
    flops = counted_flops(model, x, plan)
    assert set(flops) <= {"aten.mm", "aten.addmm"}, flops  # CPU: SDPA's fused kernel is not counted
    assert linear_flops(model, x, plan) == 2 * macs_per_token(model.cfg, kind) * x.numel()


def test_i2_a14_pilot_meta_model_single_path():
    cfg = profile("pilot", "swiglu")
    model = RoutedLM(cfg, ["zh", "en"], device="meta")
    x = torch.zeros(1, 8, dtype=torch.long, device="meta")
    plan = RoutePlan.forced("zh", model.expert_names, 1, 8)
    assert macs_per_token(cfg, "single") == 96_206_848
    flops = counted_flops(model, x, plan)
    assert set(flops) <= {"aten.mm", "aten.addmm", "aten.bmm"}, flops  # meta: SDPA lowers to bmm
    assert linear_flops(model, x, plan) == 2 * 96_206_848 * 8
    ratio = macs_per_token(cfg, "dual") / macs_per_token(cfg, "single")
    assert round(ratio, 4) == 1.6621
    assert macs_per_token(cfg, "single", 1024) == 108_789_760


# --- I2-A15: golden report ---------------------------------------------------------------------------------------


def test_i2_a15_budget_report_equals_the_golden_file(pilot):
    doc, _ = pilot
    assert doc == json.loads(GOLDEN.read_text("utf-8")), "regenerate with: audit --profile pilot --out (see README)"


def test_audit_command_prints_and_writes(tmp_path, capsys):
    assert cli(["audit", "--profile", "tiny", "--dry-run", "--out", str(tmp_path / "b.json")]) == 0
    assert "| B4 |" in capsys.readouterr().out and not (tmp_path / "b.json").exists()
    assert cli(["audit", "--profile", "tiny", "--out", str(tmp_path / "b.json")]) == 0
    doc = json.loads((tmp_path / "b.json").read_text("utf-8"))
    assert doc["format"] == "bilingual-budget/1" and doc["profile"] == "tiny"
