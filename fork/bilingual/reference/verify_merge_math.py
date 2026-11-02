#!/usr/bin/env python3
"""CPU/float64 checks for expert expansion and merging algebra.

These are tensor-algebra and planted-linear-data checks, NOT language-model,
translation, reasoning, training-throughput, or RTX 5090 benchmarks.
Run: python verify_merge_math.py --output verification_results.json
Dependency: torch (tested environment is recorded in the JSON).
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import torch
import torch.nn.functional as F

DTYPE = torch.float64
SEED = 20261102
TOL = 1e-10

@dataclass
class Expert:
    gate: torch.Tensor  # [m, d]
    up: torch.Tensor    # [m, d]
    down: torch.Tensor  # [d, m]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.down @ (F.silu(self.gate @ x) * (self.up @ x))


def random_expert(d: int, m: int) -> Expert:
    return Expert(torch.randn(m, d, dtype=DTYPE) / d**0.5,
                  torch.randn(m, d, dtype=DTYPE) / d**0.5,
                  torch.randn(d, m, dtype=DTYPE) / m**0.5)


def err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def require_close(a: torch.Tensor, b: torch.Tensor, name: str) -> float:
    e = err(a, b)
    if e > TOL:
        raise AssertionError(f'{name}: error {e:.4e} exceeds {TOL:.1e}')
    return e


def cross_mix(x: torch.Tensor, experts: list[Expert], g: torch.Tensor,
              cg: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    """Explicit active-set execution; inactive experts are not evaluated.

    cg[i,j], cu[i,j] map j's preactivations into i's. Diagonal entries unused.
    This tests the dense cross-connection operator, not a fast sparse kernel.
    """
    active = [i for i in range(len(experts)) if float(g[i]) != 0.0]
    aa = {i: experts[i].gate @ x for i in active}
    bb = {i: experts[i].up @ x for i in active}
    out = torch.zeros(experts[0].down.shape[0], x.shape[1], dtype=x.dtype)
    for i in active:
        a, b = aa[i], bb[i]
        for j in active:
            if i != j:
                a = a + g[j] * (cg[i, j] @ aa[j])
                b = b + g[j] * (cu[i, j] @ bb[j])
        out = out + g[i] * (experts[i].down @ (F.silu(a) * b))
    return out


def sparse_channels(x: torch.Tensor, experts: list[Expert],
                    selected: list[tuple[int, list[int]]]) -> torch.Tensor:
    """Gather weights BEFORE projections; never compute a full expert then mask."""
    out = torch.zeros(experts[0].down.shape[0], x.shape[1], dtype=x.dtype)
    for expert_id, ids in selected:
        idx = torch.tensor(ids, dtype=torch.long)
        e = experts[expert_id]
        a = e.gate.index_select(0, idx) @ x
        b = e.up.index_select(0, idx) @ x
        out = out + e.down.index_select(1, idx) @ (F.silu(a) * b)
    return out


def ridge_rank_delta(X: torch.Tensor, Y: torch.Tensor, lam: float,
                     rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve min_rank(Delta)<=r ||Delta X - (Y-X)||_F² + lam||Delta||_F².

    Uses eigendecomposition/SVD; avoids an explicit inverse in deployment.
    Returns Delta and all singular values of the whitened regression target.
    """
    d = X.shape[0]
    if X.shape != Y.shape or lam <= 0 or not 0 <= rank <= d:
        raise ValueError('Need equal [d,N] shapes, lambda>0 and 0<=rank<=d')
    H = X @ X.T + lam * torch.eye(d, dtype=X.dtype)
    eigenvalues, eigenvectors = torch.linalg.eigh(H)
    H_inv_sqrt = (eigenvectors * eigenvalues.rsqrt()) @ eigenvectors.T
    target = (Y - X) @ X.T @ H_inv_sqrt
    u, s, vh = torch.linalg.svd(target, full_matrices=False)
    truncated = (u[:, :rank] * s[:rank]) @ vh[:rank]
    return truncated @ H_inv_sqrt, s


def main(output: Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    d, m, N = 16, 24, 31
    X = torch.randn(d, N, dtype=DTYPE)
    ez, ee = random_expert(d, m), random_expert(d, m)
    result: dict[str, object] = {
        'status': 'mathematical and synthetic numerical checks only',
        'seed': SEED, 'dtype': 'float64', 'device': 'cpu',
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda_available_in_this_environment': torch.cuda.is_available(),
        'checks': {},
    }
    checks = result['checks']
    assert isinstance(checks, dict)

    # 1: input duplication alone collapses to a summed linear weight matrix.
    Wz, We = torch.randn(7, d, dtype=DTYPE), torch.randn(7, d, dtype=DTYPE)
    checks['linear_input_expansion_collapses'] = require_close(
        torch.cat([Wz, We], dim=1) @ torch.cat([X, X], dim=0),
        (Wz + We) @ X, 'linear folding')

    # 2: exact concatenation = weighted sum of complete SwiGLU outputs.
    g = torch.tensor([0.37, 0.63], dtype=DTYPE)
    wide = Expert(torch.cat([ez.gate, ee.gate]), torch.cat([ez.up, ee.up]),
                  torch.cat([g[0] * ez.down, g[1] * ee.down], dim=1))
    checks['swiglu_concatenation_equals_weighted_sum'] = require_close(
        wide(X), g[0] * ez(X) + g[1] * ee(X), 'SwiGLU concatenation')

    # 3: arbitrary hidden-channel permutation preserves an individual expert.
    permutation = torch.randperm(m)
    permuted = Expert(ee.gate[permutation], ee.up[permutation], ee.down[:, permutation])
    checks['permutation_preserves_function'] = require_close(ee(X), permuted(X), 'permutation')

    # 4: new cross-connections have exact endpoint preservation.
    cg, cu = torch.randn(2, 2, m, m, dtype=DTYPE) * .01, torch.randn(2, 2, m, m, dtype=DTYPE) * .01
    zero = torch.zeros_like(cg)
    checks['zero_cross_connections_equals_moe'] = require_close(
        cross_mix(X, [ez, ee], g, zero, zero), wide(X), 'zero cross-links')
    endpoints = []
    for i, e in enumerate([ez, ee]):
        onehot = torch.eye(2, dtype=DTYPE)[i]
        endpoints.append(require_close(cross_mix(X, [ez, ee], onehot, cg, cu),
                                       e(X), 'one-hot endpoint'))
    checks['nonzero_cross_connections_preserve_onehot_endpoints'] = endpoints
    difference = err(cross_mix(X, [ez, ee], g, cg, cu), wide(X))
    assert difference > 1e-6
    checks['cross_connections_change_dual_path_max_difference'] = difference

    # 5: pre/post activation coincide at endpoints, not generally inside.
    a, b = torch.randn(m, N, dtype=DTYPE), torch.randn(m, N, dtype=DTYPE)
    for alpha in (0.0, 1.0):
        require_close(torch.tanh(alpha*a + (1-alpha)*b),
                      alpha*torch.tanh(a) + (1-alpha)*torch.tanh(b), 'pre/post endpoint')
    alpha = .37
    gap = (torch.tanh(alpha*a + (1-alpha)*b)
           - alpha*torch.tanh(a) - (1-alpha)*torch.tanh(b)).abs()
    # tanh has |phi''|<=2, so the interpolation error bound is alpha(1-alpha)(a-b)^2.
    bound = alpha * (1-alpha) * (a-b).square()
    assert bool(torch.all(gap <= bound + TOL))
    checks['pre_post_jensen_gap_max'] = float(gap.max())
    checks['pre_post_curvature_bound_holds'] = True

    # 6: channel routing can select m total channels from a bank of 2m.
    checks['sparse_all_zh_recovers_zh'] = require_close(
        sparse_channels(X, [ez, ee], [(0, list(range(m)))]), ez(X), 'ZH channel endpoint')
    checks['sparse_all_en_recovers_en'] = require_close(
        sparse_channels(X, [ez, ee], [(1, list(range(m)))]), ee(X), 'EN channel endpoint')
    selected = [(0, list(range(m//2))), (1, list(range(m//2, m)))]
    masked_wide = Expert(torch.cat([ez.gate[:m//2], ee.gate[m//2:]]),
                         torch.cat([ez.up[:m//2], ee.up[m//2:]]),
                         torch.cat([ez.down[:, :m//2], ee.down[:, m//2:]], dim=1))
    checks['mixed_channels_equals_explicit_selected_network'] = require_close(
        sparse_channels(X, [ez, ee], selected), masked_wide(X), 'sparse selected network')
    checks['selected_ffn_scalar_weights'] = 3*d*m
    checks['stored_two_experts_scalar_weights'] = 2*3*d*m

    # 7: closed-form ridge fit and its rank-constrained optimum on planted data.
    dd, train_n, test_n = 20, 256, 128
    true_delta = (torch.randn(dd, 3, dtype=DTYPE) @ torch.randn(3, dd, dtype=DTYPE)) * .03
    xt = torch.randn(dd, train_n, dtype=DTYPE) * torch.linspace(.6, 1.7, dd, dtype=DTYPE)[:, None]
    yt = xt + true_delta @ xt + .005*torch.randn_like(xt)
    xv = torch.randn(dd, test_n, dtype=DTYPE)
    yv = xv + true_delta @ xv
    lam = .1
    full_delta, singular_values = ridge_rank_delta(xt, yt, lam, dd)
    gradient_half = (full_delta @ xt - (yt-xt)) @ xt.T + lam*full_delta
    checks['ridge_stationarity_max_abs'] = float(gradient_half.abs().max())
    assert float(gradient_half.abs().max()) < 1e-8
    def objective(delta: torch.Tensor) -> torch.Tensor:
        return (delta @ xt - (yt-xt)).square().sum() + lam*delta.square().sum()
    full_obj = objective(full_delta)
    rows = []
    for r in [0, 1, 2, 3, 5, dd]:
        delta, s = ridge_rank_delta(xt, yt, lam, r)
        spectral_tail = s[r:].square().sum()
        gap_error = abs(float(objective(delta)-full_obj-spectral_tail))
        assert gap_error < 1e-8
        rows.append({'rank': r, 'objective': float(objective(delta)),
                     'test_mse': float((xv + delta @ xv-yv).square().mean()),
                     'spectral_tail_identity_error': gap_error})
    checks['rank_constrained_ridge_planted_data'] = rows
    assert rows[3]['test_mse'] < rows[0]['test_mse']

    # 8: lossless unconditional merging is impossible when target distributions differ.
    p = torch.tensor([.9, .1], dtype=DTYPE)
    q = torch.tensor([.2, .8], dtype=DTYPE)
    mixture = .5*(p+q)
    def kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a*(a.log()-b.log())).sum()
    js = .5*kl(p, mixture)+.5*kl(q, mixture)
    r = torch.tensor([.4,.6], dtype=DTYPE)
    risk = .5*kl(p, r)+.5*kl(q, r)
    checks['js_decomposition_error'] = require_close(risk, js+kl(mixture, r), 'JS decomposition')
    checks['unconditional_minimum_teacher_kl_nats'] = float(js)
    assert float(js) > 0

    # 9: adding an expert changes ordinary softmax even if old expert weights freeze.
    logits2 = torch.tensor([.3,-.2], dtype=DTYPE)
    logits3 = torch.cat([logits2, torch.tensor([-.4], dtype=DTYPE)])
    old = logits2.softmax(0)
    new = logits3.softmax(0)
    masked = logits3.clone(); masked[-1] = -torch.inf
    checks['masked_new_expert_preserves_old_gate'] = require_close(old, masked.softmax(0)[:2], 'router mask')
    checks['unmasked_new_expert_gate_difference'] = err(old, new[:2])
    assert err(old, new[:2]) > .01

    # 10: a frozen expert still transmits input gradients; only permitted leaf is updated.
    x_grad = X.clone().requires_grad_(True)
    frozen_tensors = [ez.gate.clone(), ez.up.clone(), ez.down.clone()]
    frozen_copy = [t.clone() for t in frozen_tensors]
    coeff = torch.nn.Parameter(torch.tensor(.2, dtype=DTYPE))
    frozen_expert = Expert(*frozen_tensors)
    optimizer = torch.optim.AdamW([coeff], lr=.01)
    loss = ((1+coeff)*frozen_expert(x_grad)).square().mean()
    loss.backward()
    assert x_grad.grad is not None and float(x_grad.grad.norm()) > 0
    assert all(t.grad is None for t in frozen_tensors)
    optimizer.step()
    assert all(torch.equal(a,b) for a,b in zip(frozen_tensors, frozen_copy))
    checks['frozen_weights_unchanged_input_gradient_nonzero'] = True

    # 11: parameter accounting for the explicitly stated bias-free architecture.
    L, width, ffn, vocab, kv_ratio = 24, 512, 1728, 32768, .25
    shared = vocab*width + int(L*2*width**2*(1+kv_ratio)) + (2*L+1)*width
    path = 3*L*width*ffn
    block_size, router_rank = 64, 8
    qblocks = ffn//block_size
    budgets = []
    for E in [1, 2, 3, 31]:
        router = 0 if E==1 else L*router_rank*(width+E*qblocks)
        budget = {'experts': E, 'shared': shared, 'per_expert': path,
                  'router': router, 'total': shared+E*path+router,
                  'active_selecting_one_path_worth_of_channels': shared+path+router,
                  'expert_training_updated': path}
        assert budget['active_selecting_one_path_worth_of_channels'] <= 100_000_000
        budgets.append(budget)
    result['parameter_budgets'] = budgets
    result['all_assertions_passed'] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('verification_results.json'))
    main(parser.parse_args().output)
