"""The owner's v1.0 verify_merge_math.py, run as it is, and its checks 7, 8 and 11 ported and strengthened
(docs/spec.md, I3 port test, I3-C12, I3-C13; MA-12). Checks 1-6 are ported in test_fusion.py, check 9 in
test_router_monotone.py; check 10 is I2-A11 in test_freeze.py.

These are tensor-algebra and planted-data checks, like the original: not language-model results.
"""

import json

import pytest
import torch

F64 = torch.float64
EXACT = 1e-12


@pytest.fixture(scope="module")
def v10_out(verify_merge_math, tmp_path_factory):
    """Run `main` of the owner's file once and return the JSON it writes."""
    out = tmp_path_factory.mktemp("verify_merge_math") / "out.json"
    verify_merge_math.main(out)
    return json.loads(out.read_text("utf-8"))


def test_i3_port_verify_merge_math_main_reports_every_assertion_passed(v10_out):
    assert v10_out["all_assertions_passed"] is True
    assert (v10_out["dtype"], v10_out["device"]) == ("float64", "cpu")
    assert v10_out["status"] == "mathematical and synthetic numerical checks only"


# --- I3-C12: check 7, the rank-constrained ridge -----------------------------------------------------------------


def rank_constrained_ridge(X: torch.Tensor, Y: torch.Tensor, lam: float, rank: int):
    """argmin over rank(Delta) <= r of ||Delta X - (Y - X)||_F^2 + lam ||Delta||_F^2 (v1.0 `ridge_rank_delta`):
    with H = X X^T + lam I, the problem is a rank-r approximation of T = (Y - X) X^T H^(-1/2) (Eckart-Young), and
    Delta = [T]_r H^(-1/2). Returns Delta and the singular values of T."""
    d = X.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(X @ X.T + lam * torch.eye(d, dtype=X.dtype))
    h_inv_sqrt = (eigenvectors * eigenvalues.rsqrt()) @ eigenvectors.T
    u, s, vh = torch.linalg.svd((Y - X) @ X.T @ h_inv_sqrt, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank] @ h_inv_sqrt, s


@pytest.fixture(scope="module")
def planted():
    """v1.0 check 7's planted data: a rank-3 Delta, anisotropic inputs, small noise."""
    g = torch.Generator().manual_seed(20261102)
    d = 20
    true = (torch.randn(d, 3, generator=g, dtype=F64) @ torch.randn(3, d, generator=g, dtype=F64)) * 0.03
    xt = torch.randn(d, 256, generator=g, dtype=F64) * torch.linspace(0.6, 1.7, d, dtype=F64)[:, None]
    yt = xt + true @ xt + 0.005 * torch.randn(d, 256, generator=g, dtype=F64)
    xv = torch.randn(d, 128, generator=g, dtype=F64)
    return xt, yt, xv, xv + true @ xv, 0.1


def test_i3_c12_rank_constrained_ridge_is_optimal_on_planted_data(planted, verify_merge_math):
    xt, yt, xv, yv, lam = planted
    d = xt.shape[0]

    def objective(delta):
        return float((delta @ xt - (yt - xt)).square().sum() + lam * delta.square().sum())

    full, s = rank_constrained_ridge(xt, yt, lam, d)
    closed = torch.linalg.solve(xt @ xt.T + lam * torch.eye(d, dtype=F64), xt @ (yt - xt).T).T
    assert float((full - closed).abs().max()) <= 1e-10  # the unconstrained optimum is the ridge solution
    assert float(((full @ xt - (yt - xt)) @ xt.T + lam * full).abs().max()) < 1e-8  # v1.0's stationarity
    mse = {}
    for r in (0, 1, 2, 3, 5, d):
        delta, _ = rank_constrained_ridge(xt, yt, lam, r)
        assert torch.linalg.matrix_rank(delta) <= r
        assert abs(objective(delta) - objective(full) - float(s[r:].square().sum())) < 1e-8  # the spectral tail
        ref, _ = verify_merge_math.ridge_rank_delta(xt, yt, lam, r)  # the owner's implementation agrees
        assert float((delta - ref).abs().max()) <= EXACT
        mse[r] = float((xv + delta @ xv - yv).square().mean())
    assert mse[3] < mse[0]  # v1.0: the planted rank recovers held-out structure


def test_i3_c12_no_other_rank_r_matrix_does_better(planted):
    """Eckart-Young in the whitened space: random rank-r perturbations of the optimum never lower the objective."""
    xt, yt, _, _, lam = planted
    d, r = xt.shape[0], 3
    best, _ = rank_constrained_ridge(xt, yt, lam, r)

    def objective(delta):
        return float((delta @ xt - (yt - xt)).square().sum() + lam * delta.square().sum())

    g = torch.Generator().manual_seed(1)
    u, s, vh = torch.linalg.svd(best)
    for _ in range(50):
        a = u[:, :r] + 0.05 * torch.randn(d, r, generator=g, dtype=F64)
        b = (s[:r, None] * vh[:r]) + 0.05 * torch.randn(r, d, generator=g, dtype=F64)
        assert objective(a @ b) >= objective(best) - 1e-12


# --- I3-C12: check 8, the JS decomposition -----------------------------------------------------------------------


def kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * (a.log() - b.log())).sum(-1)


@pytest.mark.parametrize("vocab", [2, 7])
def test_i3_c12_a_context_free_predictor_pays_at_least_the_js_divergence(vocab):
    """For a predictor that emits one distribution r for both languages because its input does not reveal the
    language (a context-free, language-undecidable context), the mean teacher KL is
    (1/2) KL(p||r) + (1/2) KL(q||r) = JS(p, q) + KL(m||r) with m = (p + q)/2, so it is at least JS > 0 and reaches it
    at r = m. The bound says nothing about a predictor that sees the language: that one can match both teachers."""
    g = torch.Generator().manual_seed(vocab)
    if vocab == 2:  # v1.0's pair
        p, q = torch.tensor([0.9, 0.1], dtype=F64), torch.tensor([0.2, 0.8], dtype=F64)
    else:
        p, q = ((2 * torch.randn(vocab, generator=g, dtype=F64)).softmax(-1) for _ in range(2))
    m = (p + q) / 2
    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    assert float(js) > 0
    for _ in range(20):
        r = (3 * torch.randn(vocab, generator=g, dtype=F64)).softmax(-1)
        risk = 0.5 * kl(p, r) + 0.5 * kl(q, r)
        assert abs(float(risk - js - kl(m, r))) <= EXACT
        assert float(risk) >= float(js)
    assert float(kl(m, m)) == 0.0  # the minimum, JS, is reached at r = m
    assert float(0.5 * kl(p, p) + 0.5 * kl(q, q)) == 0.0  # a language-aware predictor (r = p, then r = q)


# --- I3-C13: check 11, budget rows with flags --------------------------------------------------------------------


def test_i3_c13_budget_rows_carry_flags_and_only_declared_rows_are_asserted_within(pilot):
    doc, rows = pilot
    for row in doc["rows"]:
        assert isinstance(row["within_active_budget"], bool) and isinstance(row["declared_in_budget"], bool)
        assert row["within_update_budget"] in (True, False, None)
        if row["declared_in_budget"]:
            assert row["within_active_budget"] and row["within_update_budget"] is not False, row["id"]
    over = {row["id"] for row in doc["rows"] if not row["declared_in_budget"]}
    # the capability line: reported over budget, never asserted within it, and the report still completes
    assert over == {"B6/B7", "II-dual", "II-n512-dual", "B12a"}
    assert not any(rows[i]["within_active_budget"] for i in over)


def test_i3_c13_v10_check_11_numbers_are_the_code_counts(v10_out, pilot):
    """v1.0 check 11 computed shared and path parameters from a formula and asserted every row within 100M; the
    code counts the same numbers from module trees, and its rows of the same layouts agree."""
    doc, rows = pilot
    budgets = {b["experts"]: b for b in v10_out["parameter_budgets"]}
    assert all(
        b["shared"] == doc["totals"]["shared"] and b["per_expert"] == doc["totals"]["path"] for b in budgets.values()
    )
    assert all(b["expert_training_updated"] == rows["B4"]["P_update"] for b in budgets.values())
    assert budgets[1]["router"] == 0 and budgets[1]["total"] == doc["totals"]["single"]
    for experts, row in ((2, "B4"), (3, "ZH+EN+Bridge"), (31, "31-paths")):
        assert budgets[experts]["total"] - budgets[experts]["router"] == rows[row]["P_total"]
    # v1.0's two-expert layout selecting one path of channels with a rank-8 block router is the B10-ch row
    assert budgets[2]["active_selecting_one_path_worth_of_channels"] == rows["B10-ch"]["P_active"] == 96_340_608
    assert budgets[2]["total"] == rows["B10-ch"]["P_total"]
