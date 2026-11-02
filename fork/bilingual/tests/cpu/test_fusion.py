"""Fusion operators against fp64 oracles (docs/spec.md, I3-C1 to I3-C9 and I3-C14; MA-02, MA-03, MA-04, MA-06,
MA-07, NOV-13, MA-10).

Every identity here is known mathematics [已知数学事实] checked numerically, not an experimental result. Tests run in
float64 with the tolerance of v1.0's verify_merge_math.py (TOL = 1e-10) or the tighter EXACT = 1e-12 the spec names
for that check; negative controls assert a difference above 1e-3 (or the bound the spec gives).
"""

import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode

from surogate_bilingual.audit.params import p_active, p_total
from surogate_bilingual.config import profile
from surogate_bilingual.model.lm import RoutedLM
from surogate_bilingual.ops.ffn import SlotExpert, SlotReadout, SwiGLU, standalone
from surogate_bilingual.ops.fusion import (
    ZeroInitAdapter,
    channel_select,
    concat_forward,
    cross_link_lowrank,
    cross_link_params,
    mix_outputs,
    slot_post,
    slot_pre,
    stack_swiglu,
)
from surogate_bilingual.ops.merge import permute_channels, weight_average
from surogate_bilingual.routing.plan import RoutePlan

F64 = torch.float64
TOL = 1e-10  # verify_merge_math.TOL
EXACT = 1e-12
NEGATIVE = 1e-3
D, M, N = 16, 24, 31  # v1.0's check geometry: model width, expert width, tokens
SLOTS = 11  # family-II slot width n of the operator tests
REFERENCE = Path(__file__).resolve().parents[2] / "reference"


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).detach().abs().max())


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).detach().norm() / b.detach().norm())


def swiglu_from(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> SwiGLU:
    expert = SwiGLU(gate.shape[1], gate.shape[0]).to(gate.dtype)
    with torch.no_grad():
        expert.gate.weight.copy_(gate)
        expert.up.weight.copy_(up)
        expert.down.weight.copy_(down)
    return expert


def swiglu(seed: int, m: int = M, dtype=F64) -> SwiGLU:
    g = gen(seed)
    return swiglu_from(
        torch.randn(m, D, generator=g, dtype=dtype) / D**0.5,
        torch.randn(m, D, generator=g, dtype=dtype) / D**0.5,
        torch.randn(D, m, generator=g, dtype=dtype) / m**0.5,
    )


def slot_pair(seed: int = 0, expert_cls=SlotExpert, readout_cls=SlotReadout):
    """Two family-II experts (R: n x d, W: k x n) and a shared readout (Q: d x k, b: k) with b far from zero."""
    g = gen(seed)
    experts = []
    for _ in range(2):
        e = expert_cls(D, SLOTS, M).to(F64)
        with torch.no_grad():
            e.R.weight.copy_(torch.randn(SLOTS, D, generator=g, dtype=F64) / D**0.5)
            e.W.weight.copy_(torch.randn(M, SLOTS, generator=g, dtype=F64) / SLOTS**0.5)
        experts.append(e)
    readout = readout_cls(D, M).to(F64)
    with torch.no_grad():
        readout.Q.weight.copy_(torch.randn(D, M, generator=g, dtype=F64) / M**0.5)
        readout.b.copy_(torch.randn(M, generator=g, dtype=F64))
    return experts, readout


def inputs(n: int = N, seed: int = 100, dtype=F64) -> torch.Tensor:
    return torch.randn(n, D, generator=gen(seed), dtype=dtype)


def token_gates(n: int, seed: int, dtype=F64) -> torch.Tensor:
    """Token-dependent rows on the simplex, both entries in [0.1, 0.9]."""
    w = torch.rand(n, 1, generator=gen(seed), dtype=dtype) * 0.8 + 0.1
    return torch.cat([w, 1 - w], dim=1)


def const_gates(n: int, g0: float, dtype=F64) -> torch.Tensor:
    return torch.tensor([[g0, 1.0 - g0]], dtype=dtype).expand(n, 2)


def rms(y: torch.Tensor) -> torch.Tensor:
    return y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6)


# --- I3-C1: concat = Top-2, and the conditions it needs (MA-07) --------------------------------------------------


def test_i3_c1_concat_with_constant_g_folded_into_down_is_the_weighted_top2():
    """v1.0 check 2: a static concat whose down columns carry the constant g is the weighted Top-2."""
    ez, ee, x = swiglu(0), swiglu(1), inputs()
    g0 = 0.37
    wide = swiglu_from(
        torch.cat([ez.gate.weight, ee.gate.weight]),
        torch.cat([ez.up.weight, ee.up.weight]),
        torch.cat([g0 * ez.down.weight, (1 - g0) * ee.down.weight], dim=1),
    )
    assert err(wide(x), mix_outputs([ez(x), ee(x)], const_gates(N, g0))) <= TOL


def test_i3_c1_block_gated_concat_equals_mix_outputs_under_token_gates():
    ez, ee, x = swiglu(0), swiglu(1), inputs()
    g = token_gates(N, 2)
    assert err(concat_forward(x, stack_swiglu([ez, ee]), g), mix_outputs([ez(x), ee(x)], g)) <= EXACT


def test_i3_c1_negative_controls_break_the_identity():
    ez, ee, x = swiglu(0), swiglu(1), inputs()
    experts, stacked, g = (ez, ee), stack_swiglu([ez, ee]), token_gates(N, 2)
    top2 = mix_outputs([ez(x), ee(x)], g)
    # a static concat (the mean gate folded into W_down) is not the token-gated Top-2
    static = concat_forward(x, stacked, const_gates(N, float(g[:, 0].mean())))
    assert err(static, top2) > NEGATIVE
    # a down bias must be gated with its block; summed ungated it breaks the identity
    c = [torch.randn(D, generator=gen(5 + i), dtype=F64) for i in range(2)]
    top2_bias = mix_outputs([ez(x) + c[0], ee(x) + c[1]], g)
    gated_bias = concat_forward(x, stacked, g) + g[:, :1] * c[0] + g[:, 1:] * c[1]
    assert err(gated_bias, top2_bias) <= EXACT
    assert err(concat_forward(x, stacked, g) + c[0] + c[1], top2_bias) > NEGATIVE
    # the gate on the input side: SwiGLU is not homogeneous
    input_side = sum(e(g[:, i : i + 1] * x) for i, e in enumerate(experts))
    assert err(input_side, top2) > NEGATIVE
    # a per-expert post-norm cannot be expressed by one concat, normalised or not
    top2_post = mix_outputs([rms(ez(x)), rms(ee(x))], g)
    assert err(concat_forward(x, stacked, g), top2_post) > NEGATIVE
    assert err(rms(concat_forward(x, stacked, g)), top2_post) > NEGATIVE


def test_i3_c1_per_expert_pre_norm_gain_folds_into_the_weight_columns():
    """Experts with their own pre-norm gain gamma_i on a shared RMS statistic: gamma_i folds into the columns of
    G_i and U_i, and the concat of the folded experts is the Top-2."""
    ez, ee, x = swiglu(0), swiglu(1), inputs()
    g = token_gates(N, 2)
    gamma = [1 + 0.3 * torch.randn(D, generator=gen(7 + i), dtype=F64) for i in range(2)]
    n = rms(x)
    top2 = mix_outputs([ez(gamma[0] * n), ee(gamma[1] * n)], g)
    folded = [swiglu_from(e.gate.weight * gm, e.up.weight * gm, e.down.weight) for e, gm in zip((ez, ee), gamma)]
    assert err(concat_forward(n, stack_swiglu(folded), g), top2) <= EXACT


def test_i3_c1_concat_equals_mix_in_fp32_within_tolerance():
    """In fp32 the identity holds to a tolerance, not bitwise (MS §4.2 condition 6)."""
    ez, ee = swiglu(0, dtype=torch.float32), swiglu(1, dtype=torch.float32)
    x, g = inputs(dtype=torch.float32), token_gates(N, 2, dtype=torch.float32)
    assert err(concat_forward(x, stack_swiglu([ez, ee]), g), mix_outputs([ez(x), ee(x)], g)) <= 1e-5


# --- I3-C2: permutation ------------------------------------------------------------------------------------------


def test_i3_c2_hidden_channel_permutation_preserves_the_expert():
    """v1.0 check 3, through `merge.permute_channels` (what re-basin applies)."""
    ee, x = swiglu(1), inputs()
    perm = torch.randperm(M, generator=gen(3)).numpy()
    state = {f"ffn.0.{k}": v for k, v in ee.state_dict().items()}
    moved = permute_channels(state, 0, perm)
    permuted = swiglu_from(moved["ffn.0.gate.weight"], moved["ffn.0.up.weight"], moved["ffn.0.down.weight"])
    assert not torch.equal(permuted.gate.weight, ee.gate.weight)
    assert err(permuted(x), ee(x)) <= TOL


def test_i3_c2_slot_permutation_preserves_a_family_ii_expert():
    (expert, _), readout = slot_pair()
    u = inputs()
    perm = torch.randperm(SLOTS, generator=gen(4)).numpy()
    moved = permute_channels({f"ffn.0.{k}": v for k, v in expert.state_dict().items()}, 0, perm)
    permuted = SlotExpert(D, SLOTS, M).to(F64)
    permuted.load_state_dict({k.removeprefix("ffn.0."): v for k, v in moved.items()})
    assert err(standalone(permuted, readout, u), standalone(expert, readout, u)) <= TOL


# --- I3-C3: cross links are a reparameterised concat (MA-06) -----------------------------------------------------


def links(f: int, r: int, seed: int, scale: float = 0.01):
    g = gen(seed)
    return (
        torch.randn(2, 2, 2, f, r, generator=g, dtype=F64) * scale,
        torch.randn(2, 2, 2, f, r, generator=g, dtype=F64),
    )


def full_rank_links(seed: int, scale: float = 0.01):
    """v1.0 check 4's full-rank C (random f x f) as A = C, B = I."""
    C = torch.randn(2, 2, 2, M, M, generator=gen(seed), dtype=F64) * scale
    return C, torch.eye(M, dtype=F64).expand(2, 2, 2, M, M)


def test_i3_c3_zero_links_equal_the_concat_and_one_hot_endpoints_are_exact():
    """v1.0 check 4a and 4b."""
    experts, x = [swiglu(0), swiglu(1)], inputs()
    A, B = full_rank_links(9)
    for g in (const_gates(N, 0.37), token_gates(N, 2)):
        concat = concat_forward(x, stack_swiglu(experts), g)
        assert err(cross_link_lowrank(x, experts, g, torch.zeros_like(A), B), concat) <= TOL
        assert err(cross_link_lowrank(x, experts, g, A, B), concat) > 1e-6  # links change the dual path
    for i, expert in enumerate(experts):
        one_hot = torch.zeros(N, 2, dtype=F64)
        one_hot[:, i] = 1
        assert torch.equal(cross_link_lowrank(x, experts, one_hot, A, B), expert(x))


def reparameterised(experts, g0: float, delta):
    """Expert i with G_i + g_j dG_ij and U_i + g_j dU_ij (constant g), for delta(p, i, j, W_j) -> f x d."""
    g = (g0, 1 - g0)
    out = []
    for i, e in enumerate(experts):
        j = 1 - i
        gate = e.gate.weight + g[j] * delta(0, i, j, experts[j].gate.weight)
        up = e.up.weight + g[j] * delta(1, i, j, experts[j].up.weight)
        out.append(swiglu_from(gate, up, e.down.weight))
    return out


def test_i3_c3_constant_g_cross_links_are_concat_with_reparameterised_experts():
    experts, x, g0 = [swiglu(0), swiglu(1)], inputs(), 0.37
    A, B = full_rank_links(9, scale=0.1)
    C = A @ B.transpose(-1, -2)
    moved = reparameterised(experts, g0, lambda p, i, j, W: C[p, i, j] @ W)
    linked = cross_link_lowrank(x, experts, const_gates(N, g0), A, B)
    assert err(linked, concat_forward(x, stack_swiglu(moved), const_gates(N, g0))) <= EXACT


def test_i3_c3_rank_r_cross_links_are_a_gated_lora():
    """C = A B^T of rank r on G_i is the LoRA dG_i = g_j A (B^T G_j): down factor B^T G_j (r x d), up factor A."""
    experts, x, g0 = [swiglu(0), swiglu(1)], inputs(), 0.37
    A, B = links(M, 3, seed=10, scale=0.3)
    lora = reparameterised(experts, g0, lambda p, i, j, W: A[p, i, j] @ (B[p, i, j].T @ W))
    linked = cross_link_lowrank(x, experts, const_gates(N, g0), A, B)
    assert err(linked, concat_forward(x, stack_swiglu(lora), const_gates(N, g0))) <= EXACT
    assert err(linked, concat_forward(x, stack_swiglu(experts), const_gates(N, g0))) > NEGATIVE


def test_i3_c3_cross_link_parameter_counts():
    cfg = profile("pilot", "swiglu")
    f, L, d = cfg.ffn_width, cfg.n_layers, cfg.d_model
    assert cross_link_params(f, L) == 24 * 2 * 2 * f * f == 286_654_464
    assert round(cross_link_params(f, L) / 96_231_936, 2) == 2.98  # MA-06: 2.98 single paths
    assert 2 * L * 2 * f * d == 84_934_656  # fine-tuning both experts' gate and up directly costs this
    r = 8
    assert cross_link_params(f, L, rank=r) == L * 2 * 2 * (2 * f * r)  # 2 f r per projection, direction, layer
    assert cross_link_params(f, 1, rank=1) // (2 * 2) == 2 * f == 3456  # a plain LoRA on G_i costs (f + d) r
    assert f + d == 2240
    A, B = links(M, 3, seed=0)  # the operator's factors hold exactly that many off-diagonal entries per layer
    off = sum(A[:, i, j].numel() + B[:, i, j].numel() for i in range(2) for j in range(2) if i != j)
    assert off == cross_link_params(M, 1, rank=3)


# --- I3-C4: pre/post, the bias convention and the curvature bound (MA-03, MA-04, NOV-13) -------------------------


@pytest.mark.parametrize("i", [0, 1])
def test_i3_c4_one_hot_endpoints_give_the_standalone_expert_exactly(i):
    experts, readout = slot_pair()
    u = inputs()
    g = torch.zeros(N, 2, dtype=F64)
    g[:, i] = 1
    alone = standalone(experts[i], readout, u)
    assert err(slot_pre(u, experts, readout, g), alone) == 0.0
    assert err(slot_post(u, experts, readout, g), alone) == 0.0


@pytest.mark.parametrize(
    "family,fusion", [("swiglu", "output"), ("swiglu", "concat"), ("slot", "pre"), ("slot", "post")]
)
def test_i3_c4_identical_experts_under_simplex_gates_reproduce_the_seed_logits(family, fusion):
    """MA-04's G0 test on the model: both slots hold the seed's FFN; ten random simplex gate tensors (token-dependent
    rows) give the seed's logits."""
    names = ("zh", "en")
    model = RoutedLM(profile("tiny", family), names, fusion, dtype=F64, init_seed=3).eval()
    model.bank.experts["en"].load_state_dict(model.bank.experts["zh"].state_dict())
    if family == "slot":  # b = 0 at initialisation; a nonzero b is what g = (1, 1) would count twice
        with torch.no_grad():
            for layer in model.backbone.layers:
                layer.readout.b.copy_(torch.randn(layer.readout.b.shape, generator=gen(2), dtype=F64))
    B, T = 2, model.cfg.max_seq
    x = torch.randint(5, model.cfg.vocab_size, (B, T), generator=gen(1))
    with torch.no_grad():
        seed_logits = model(x, RoutePlan.forced("zh", names, B, T))
        for draw in range(10):
            w = torch.rand(B, T, 1, generator=gen(20 + draw), dtype=F64) * 0.98 + 0.01
            plan = RoutePlan(names, torch.cat([w, 1 - w], dim=-1), "test-dual")
            plan.validate(names, (B, T))
            assert err(model(x, plan), seed_logits) <= EXACT, draw


def test_i3_c4_identical_experts_under_unnormalised_gates_do_not_reproduce_the_seed():
    """The documented negative of MA-04: with g = (1, 1) none of the four fusions gives the seed; post and output
    double it, pre counts b once against two pre-activations."""
    expert, x = swiglu(0), inputs()
    (slot, _), readout = slot_pair()
    ones = torch.ones(N, 2, dtype=F64)
    seed_i, seed_ii = expert(x), standalone(slot, readout, x)
    fused = {
        "output": (mix_outputs([expert(x), expert(x)], ones), seed_i),
        "concat": (concat_forward(x, stack_swiglu([expert, expert]), ones), seed_i),
        "pre": (slot_pre(x, [slot, slot], readout, ones), seed_ii),
        "post": (slot_post(x, [slot, slot], readout, ones), seed_ii),
    }
    for name, (out, seed) in fused.items():
        assert rel(out, seed) > 0.1, name
    assert err(fused["post"][0], 2 * seed_ii) <= EXACT and err(fused["output"][0], 2 * seed_i) <= EXACT
    for k in range(10):  # on the simplex the same pairs reproduce the seed
        g = const_gates(N, float(torch.rand((), generator=gen(40 + k), dtype=F64)))
        assert err(slot_pre(x, [slot, slot], readout, g), seed_ii) <= EXACT
        assert err(slot_post(x, [slot, slot], readout, g), seed_ii) <= EXACT
        assert err(concat_forward(x, stack_swiglu([expert, expert]), g), seed_i) <= EXACT


def silu2(x: torch.Tensor) -> torch.Tensor:
    """SiLU''(x) = s (1 - s) (2 + x (1 - 2 s)), s = sigmoid(x)."""
    s = torch.sigmoid(x)
    return s * (1 - s) * (2 + x * (1 - 2 * s))


GRID = torch.arange(-100_000, 100_001, dtype=F64) * 1e-4  # [-10, 10], step 1e-4, x = 0 exactly on the grid


def test_i3_c4_silu_second_derivative_closed_form_matches_autograd():
    x = torch.linspace(-6, 6, 97, dtype=F64, requires_grad=True)
    (d1,) = torch.autograd.grad(F.silu(x).sum(), x, create_graph=True)
    (d2,) = torch.autograd.grad(d1.sum(), x)
    assert err(d2, silu2(x.detach())) <= EXACT


def test_i3_c4_silu_curvature_constant_and_convexity_interval():
    curv = silu2(GRID)
    peak = int(curv.abs().argmax())
    assert abs(float(curv.abs().max()) - 0.5) <= 1e-6 and float(GRID[peak]) == 0.0  # sup|SiLU''| = 0.5 at 0
    flips = torch.nonzero(torch.sign(curv[1:]) != torch.sign(curv[:-1])).flatten()
    roots = [float(GRID[i] + GRID[i + 1]) / 2 for i in flips]
    assert len(roots) == 2 and all(abs(abs(r) - 2.3994) <= 1e-3 for r in roots) and roots[0] < 0 < roots[1]
    assert float(silu2(torch.tensor(3.0, dtype=F64))) < 0  # concave beyond |x| = 2.40
    assert round(float(curv.min()), 3) == -0.037  # MA-03's minimum


def test_i3_c4_tanh_tight_curvature_constant():
    t = torch.tanh(GRID)
    assert abs(float((-2 * t * (1 - t * t)).abs().max()) - 4 / (3 * math.sqrt(3))) <= 1e-6  # v1.0 used 2


def test_i3_c4_silu_jensen_bound_holds_and_is_attained():
    """|phi(g1 a1 + g2 a2) - g1 phi(a1) - g2 phi(a2)| <= (1/4) g1 g2 (a1 - a2)^2 on the simplex (MA-03)."""
    g = gen(11)
    a1, a2 = torch.randn(100_000, generator=g, dtype=F64) * 3, torch.randn(100_000, generator=g, dtype=F64) * 3
    w = torch.rand(100_000, generator=g, dtype=F64)
    gap = (F.silu(w * a1 + (1 - w) * a2) - w * F.silu(a1) - (1 - w) * F.silu(a2)).abs()
    bound = 0.25 * w * (1 - w) * (a1 - a2).square()
    assert bool((gap <= bound + 1e-13).all())  # 1e-13: rounding of silu at |a| <= 15
    for delta in (1e-2, 1e-3):  # attained near a = +-delta with g = (1/2, 1/2)
        a = torch.tensor([delta, -delta], dtype=F64)
        gap = abs(float(F.silu(a.mean()) - F.silu(a).mean()))
        assert gap / (0.25 * 0.25 * (2 * delta) ** 2) >= 0.99


class LinearReadout(SlotReadout):
    """Q (a + b): the readout with a linear phi."""

    def activate(self, a: torch.Tensor) -> torch.Tensor:
        return a + self.b


def test_i3_c4_unnormalised_gates_add_a_first_order_bias_term():
    """With linear phi and g = (1, 1): pre = Q(a1 + a2 + b), post = Q(a1 + a2 + 2b), so post - pre = Q b, a
    first-order term no curvature bound covers (NOV-13). On the simplex b enters once and post = pre."""
    experts, readout = slot_pair(readout_cls=LinearReadout)
    u = inputs()
    ones = torch.ones(N, 2, dtype=F64)
    diff = slot_post(u, experts, readout, ones) - slot_pre(u, experts, readout, ones)
    assert err(diff, readout.Q(readout.b).expand(N, D)) <= EXACT
    g = token_gates(N, 2)
    assert err(slot_post(u, experts, readout, g), slot_pre(u, experts, readout, g)) <= EXACT


# --- I3-C5: linear P collapses only for constant g (MA-02) -------------------------------------------------------


def test_i3_c5_input_duplication_alone_collapses_to_a_summed_linear_map():
    """v1.0 check 1: [W_z, W_e] [x; x] = (W_z + W_e) x, so duplicating the input adds nothing without a nonlinearity."""
    g, x = gen(17), inputs()
    Wz, We = torch.randn(7, D, generator=g, dtype=F64), torch.randn(7, D, generator=g, dtype=F64)
    assert err(torch.cat([Wz, We], dim=1) @ torch.cat([x.T, x.T]), (Wz + We) @ x.T) <= TOL


class LinearSlotExpert(SlotExpert):
    """z = R u: the family-II expert with a linear P (B8-lin)."""

    def pre_activation(self, u: torch.Tensor) -> torch.Tensor:
        return self.W(self.R(u))


def slot_state(expert: SlotExpert) -> dict:
    return {f"ffn.0.{k}": v for k, v in expert.state_dict().items()}


def test_i3_c5_linear_p_pre_with_constant_g_is_one_mlp_with_the_product_merged_first_layer():
    experts, readout = slot_pair(expert_cls=LinearSlotExpert)
    u, g0 = inputs(), 0.37
    (W1, R1), (W2, R2) = [(e.W.weight, e.R.weight) for e in experts]
    first = g0 * W1 @ R1 + (1 - g0) * W2 @ R2  # sum_i g_i W_i R_i, [k, d]
    assert err(slot_pre(u, experts, readout, const_gates(N, g0)), readout(F.linear(u, first))) <= EXACT


def test_i3_c5_per_tensor_merge_differs_from_the_product_merge_by_the_cross_term():
    """B3-II averages R and W separately (`weight_average`); B8-lin with constant g merges the products. Their
    difference is the cross term: product - per-tensor = g1 g2 (W1 - W2)(R1 - R2). (The English spec writes
    "per-tensor minus product" for the same right-hand side; that sign is wrong, see the I3 landing record.)"""
    experts, _ = slot_pair(expert_cls=LinearSlotExpert)
    g0 = 0.37
    (W1, R1), (W2, R2) = [(e.W.weight, e.R.weight) for e in experts]
    merged = weight_average([slot_state(e) for e in experts], [g0, 1 - g0])
    per_tensor = merged["ffn.0.W.weight"] @ merged["ffn.0.R.weight"]
    product = g0 * W1 @ R1 + (1 - g0) * W2 @ R2
    cross = g0 * (1 - g0) * (W1 - W2) @ (R1 - R2)
    assert err(product - per_tensor, cross) <= EXACT
    assert err(per_tensor - product, cross) > NEGATIVE


def test_i3_c5_token_dependent_gates_leave_a_residual_no_fixed_first_layer_removes():
    experts, readout = slot_pair(expert_cls=LinearSlotExpert)
    u = torch.randn(512, D, generator=gen(12), dtype=F64)
    w = torch.rand(512, 1, generator=gen(13), dtype=F64)
    g = torch.cat([w, 1 - w], dim=1)
    (W1, R1), (W2, R2) = [(e.W.weight, e.R.weight) for e in experts]
    target = w * (u @ (W1 @ R1).T) + (1 - w) * (u @ (W2 @ R2).T)  # the mixed pre-activation of slot_pre
    assert err(slot_pre(u, experts, readout, g), readout(target)) <= EXACT
    fixed = torch.linalg.lstsq(u, target).solution  # best fixed first layer, [d, k]
    assert rel(u @ fixed, target) > 0.1


# --- I3-C6: B8 and B9 cost the same; post reads Q once -----------------------------------------------------------


@pytest.mark.parametrize("profile_name", ["tiny", "pilot"])
def test_i3_c6_b8_and_b9_have_equal_parameter_counts(profile_name):
    cfg = profile(profile_name, "slot")
    device = "meta" if profile_name == "pilot" else "cpu"
    models = {f: RoutedLM(cfg, ("zh", "en"), f, device=device) for f in ("pre", "post")}
    x = torch.zeros(1, 8, dtype=torch.long, device=device)
    w = torch.full((1, 8, 1), 0.5)
    dual = RoutePlan(("zh", "en"), torch.cat([w, 1 - w], dim=-1), "dual_const(0.5,0.5)")
    totals = {f: p_total(m) for f, m in models.items()}
    active = {f: p_active(m, x, dual) for f, m in models.items()}
    assert totals["pre"] == totals["post"] and active["pre"] == active["post"]
    if profile_name == "pilot":
        assert active["pre"] == 138_639_360  # MS §5.1, family II dual


class CountCalls(TorchFunctionMode):
    """Counts torch function calls that receive `weight`."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight, self.calls = weight, 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if any(a is self.weight for a in (*args, *kwargs.values())):
            self.calls += 1
        return func(*args, **kwargs)


def test_i3_c6_slot_post_uses_one_q_matmul_equal_to_the_per_branch_sum():
    experts, readout = slot_pair()
    u, g = inputs(), token_gates(N, 2)
    with CountCalls(readout.Q.weight) as counter:
        post = slot_post(u, experts, readout, g)
    assert counter.calls == 1
    with CountCalls(readout.Q.weight) as counter:
        slot_pre(u, experts, readout, g)
    assert counter.calls == 1
    per_branch = sum(g[:, i : i + 1] * readout.Q(readout.activate(e.pre_activation(u))) for i, e in enumerate(experts))
    assert err(post, per_branch) <= EXACT


# --- I3-C7: channel selection ------------------------------------------------------------------------------------


class Shapes(TorchFunctionMode):
    """Records the shape of every tensor a torch function returns."""

    def __init__(self):
        super().__init__()
        self.shapes: list[tuple[int, ...]] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor):
            self.shapes.append(tuple(out.shape))
        return out


def full_width_intermediates(shapes, rows: int, width: int) -> list[tuple[int, ...]]:
    return [s for s in shapes if len(s) == 2 and s[0] == rows and s[1] >= width]


HALF = [(0, torch.arange(M // 2)), (1, torch.arange(M // 2, M))]


def test_i3_c7_channel_select_endpoints_recover_each_expert():
    experts, x = [swiglu(0), swiglu(1)], inputs()
    for i, expert in enumerate(experts):
        assert err(channel_select(x, experts, [(i, torch.arange(M))]), expert(x)) <= TOL


def test_i3_c7_half_and_half_is_the_explicit_selected_network_without_full_width_intermediates():
    """v1.0 check 6. D = 16 < M = 24, so an [N, >= M] tensor can only be a full-width intermediate."""
    (ez, ee), x = (swiglu(0), swiglu(1)), inputs()
    h = M // 2
    explicit = swiglu_from(
        torch.cat([ez.gate.weight[:h], ee.gate.weight[h:]]),
        torch.cat([ez.up.weight[:h], ee.up.weight[h:]]),
        torch.cat([ez.down.weight[:, :h], ee.down.weight[:, h:]], dim=1),
    )
    with Shapes() as hook:
        out = channel_select(x, [ez, ee], HALF)
    assert err(out, explicit(x)) <= TOL
    assert hook.shapes and full_width_intermediates(hook.shapes, N, M) == []
    with Shapes() as hook:  # the hook catches "compute the whole expert, then mask"
        mask = torch.zeros(M, dtype=F64)
        mask[:h] = 1
        ez.down(ez.hidden(x) * mask)
    assert full_width_intermediates(hook.shapes, N, M)


class Selected(nn.Module):
    def __init__(self, experts, selection):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.selection = selection

    def forward(self, x, plan):
        return channel_select(x, list(self.experts), self.selection)


def test_i3_c7_channel_select_declares_what_it_reads_to_the_audit():
    module = Selected([swiglu(0), swiglu(1)], HALF)
    assert p_active(module, inputs(), None) == 3 * D * M  # v1.0 check 6: selected_ffn_scalar_weights
    assert p_total(module) == 2 * 3 * D * M  # stored_two_experts_scalar_weights


def test_i3_c7_b10_ch_channel_count_is_what_channel_select_declares():
    """The B10-ch budget row counts 27 of 54 blocks of 64 channels per layer by declaration; at the pilot geometry
    (meta device) channel_select on such a selection declares exactly that count."""
    cfg = profile("pilot", "swiglu")
    d, f, block = cfg.d_model, cfg.ffn_width, 64
    with torch.device("meta"):
        experts = [SwiGLU(d, f), SwiGLU(d, f)]
    per_expert = f // block  # 27 blocks each, 54 in the bank
    ids = [torch.arange(14 * block), torch.arange(14 * block, per_expert * block)]  # 14 + 13 = 27 blocks
    module = Selected(experts, [(0, ids[0]), (1, ids[1])])
    x = torch.zeros(8, d, device="meta")
    selected = 27 * block * 3 * d
    assert p_active(module, x, None) == selected
    assert cfg.n_layers * selected == 96_340_608 - 32_530_944 - 24 * 8 * (512 + 54)  # B10-ch P_active - shared - router


def test_i3_c7_channel_select_refuses_bad_selections():
    experts, x = [swiglu(0), swiglu(1)], inputs()
    for selection in ([], [(0, torch.arange(3)), (0, torch.arange(3, 6))], [(2, torch.arange(3))],
                      [(0, torch.tensor([1, 1]))], [(0, torch.tensor([M]))]):  # fmt: skip
        with pytest.raises(ValueError):
            channel_select(x, experts, selection)


# --- I3-C8: the B10-I identity -----------------------------------------------------------------------------------


def test_i3_c8_two_half_width_experts_concatenated_are_one_dense_swiglu():
    halves, x, g0 = [swiglu(0, m=M // 2), swiglu(1, m=M // 2)], inputs(), 0.37
    dense = swiglu_from(
        torch.cat([e.gate.weight for e in halves]),
        torch.cat([e.up.weight for e in halves]),
        torch.cat([g0 * halves[0].down.weight, (1 - g0) * halves[1].down.weight], dim=1),
    )
    assert dense.gate.weight.shape[0] == M
    assert err(concat_forward(x, stack_swiglu(halves), const_gates(N, g0)), dense(x)) <= EXACT


# --- I3-C9: the B11 adapter --------------------------------------------------------------------------------------


def test_i3_c9_zero_init_adapter_adds_exactly_zero_and_trains_its_up_projection():
    adapter = ZeroInitAdapter(D, 4, dtype=F64)
    h = inputs()
    delta = adapter(h)
    assert torch.equal(h + delta, h) and not bool(delta.any())
    (delta * torch.randn(N, D, generator=gen(14), dtype=F64)).sum().backward()
    assert float(adapter.up.weight.grad.abs().max()) > 0
    assert not bool(adapter.down.weight.grad.any())  # zero at initialisation: it passes through U = 0
    assert p_total(adapter) == 2 * D * 4


# --- I3-C14: the identities of v1.0 compare.py (恒等式检查, never results; MA-10, SCI-17, IMPL-16) ----------------


@pytest.mark.identity_check
@pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")
def test_i3_c14_identity_check_compare_run_holds_the_test_compare_assertions(compare_v10):
    """恒等式检查：v1.0 test_compare.py 的三条断言，按原样移植。hard_top1 与 oracle 是同一个表达式，校准从
    损失 0、梯度 0 出发，所以这几个数按构造成立，不是结果（MA-10）。"""
    r = compare_v10.run(0, 10)
    assert r["2n_top2_max_abs_diff"] < 1e-5
    assert r["hard_top1"] == 0
    assert r["wrong_route"] > 0
    assert r["hard_top1"] == r["oracle"] == r["single_route_bias_calibration"] == 0.0


@pytest.mark.identity_check
def test_i3_c14_identity_check_tanh_block_concat_is_the_mean_of_the_outputs(compare_v10):
    """恒等式检查：compare.block_concat（tanh MLP，down 列减半）等于两个专家输出的平均，fp64 下 <= 1e-12。"""
    torch.manual_seed(0)
    e0, e1 = compare_v10.expert().double(), compare_v10.expert().double()
    x = torch.randn(512, 12, generator=gen(15), dtype=F64)
    with torch.no_grad():
        assert err(compare_v10.block_concat(e0, e1, x), (e0(x) + e1(x)) / 2) <= EXACT


@pytest.mark.identity_check
def test_i3_c14_identity_check_top2_average_against_the_oracle_is_a_quarter_of_the_disagreement():
    """恒等式检查：oracle 为前半取 z、后半取 en 时，输出平均对 oracle 的 MSE 恒等于 mean((z − en)²)/4，这是
    输出平均的固有性质，不是缺陷，也不是结果（MA-10 的核验者限定）。"""
    g = gen(16)
    z, en = torch.randn(512, 12, generator=g, dtype=F64), torch.randn(512, 12, generator=g, dtype=F64)
    oracle = torch.cat([z[:256], en[256:]])
    mse = float(((z + en) / 2 - oracle).square().mean())
    assert abs(mse - float((z - en).square().mean()) / 4) <= EXACT


@pytest.mark.identity_check
def test_i3_c14_identity_check_smoke_results_hold_only_identities():
    """恒等式检查：v1.0 的 smoke_results.json 只含按构造成立的数（scope 自述为合成单元对照）。"""
    doc = json.loads((REFERENCE / "smoke_results.json").read_text("utf-8"))
    assert doc["scope"] == "synthetic unit comparison, not language model result"
    (row,) = doc["results"]
    assert row["hard_top1"] == row["oracle"] == row["single_route_bias_calibration"] == 0.0
    assert row["2n_top2_max_abs_diff"] < 1e-5 and abs(row["top2_average"] - row["2n_concat"]) < 1e-6
