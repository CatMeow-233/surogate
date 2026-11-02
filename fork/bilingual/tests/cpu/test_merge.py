"""Same-initialisation merges and Re-Basin (docs/spec.md, I3-C10; NOV-6, IMPL-19). fp64; TOL 1e-12 unless stated.

These test that the B3 baselines compute what their definitions say. Whether any of them works on real experts is
an experiment [待实验].
"""

import itertools

import numpy as np
import pytest
import torch

from surogate_bilingual.ops import merge
from surogate_bilingual.ops.merge import (
    EQUIVALENT_TO_AVERAGING,
    NON_IDENTITY,
    NUMPY_LAP_MAX,
    alignment_label,
    channel_similarity,
    dare,
    permute_channels,
    rebasin_align,
    rebasin_permutations,
    solve_lap,
    task_arithmetic,
    ties,
    weight_average,
)

F64 = torch.float64
EXACT = 1e-12
D, F_, LAYERS = 8, 12, (0, 1)


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def swiglu_state(seed: int, d: int = D, f: int = F_, layers=LAYERS) -> dict[str, torch.Tensor]:
    g = gen(seed)
    shapes = {"gate.weight": (f, d), "up.weight": (f, d), "down.weight": (d, f)}
    return {f"ffn.{layer}.{k}": torch.randn(s, generator=g, dtype=F64) for layer in layers for k, s in shapes.items()}


def slot_state(seed: int, d: int = D, n: int = 7, k: int = F_, layers=LAYERS) -> dict[str, torch.Tensor]:
    g = gen(seed)
    shapes = {"R.weight": (n, d), "W.weight": (k, n)}
    return {f"ffn.{layer}.{p}": torch.randn(s, generator=g, dtype=F64) for layer in layers for p, s in shapes.items()}


def maxdiff(a: dict, b: dict) -> float:
    assert sorted(a) == sorted(b)
    return max(float((a[k] - b[k]).abs().max()) for k in a)


def equal(a: dict, b: dict) -> bool:
    return sorted(a) == sorted(b) and all(torch.equal(a[k], b[k]) for k in a)


# --- weight average, task arithmetic, TIES, DARE -----------------------------------------------------------------


def test_i3_c10_weight_average_of_identical_experts_returns_the_expert():
    a = swiglu_state(0)
    assert equal(weight_average([a, dict(a)], [0.5, 0.5]), a)
    assert maxdiff(weight_average([a, a, a], [0.2, 0.3, 0.5]), a) <= EXACT
    fp32 = {k: v.float() for k, v in a.items()}
    out = weight_average([fp32, fp32], [0.3, 0.7])
    assert all(v.dtype == torch.float32 for v in out.values()) and maxdiff(out, fp32) == 0.0


def test_i3_c10_weight_average_is_the_weighted_sum():
    a, b = swiglu_state(0), swiglu_state(1)
    out = weight_average([a, b], [0.37, 0.63])
    assert maxdiff(out, {k: 0.37 * a[k] + 0.63 * b[k] for k in a}) <= EXACT


def test_i3_c10_task_arithmetic_endpoints():
    seed, a, b = swiglu_state(0), swiglu_state(1), swiglu_state(2)
    assert equal(task_arithmetic(seed, [a], 0.0), seed)
    assert maxdiff(task_arithmetic(seed, [a], 1.0), a) <= EXACT
    two = task_arithmetic(seed, [a, b], 0.5)
    assert maxdiff(two, {k: seed[k] + 0.5 * (a[k] - seed[k] + b[k] - seed[k]) for k in seed}) <= EXACT
    assert maxdiff(two, weight_average([a, b], [0.5, 0.5])) <= EXACT  # lambda = 1/E is the plain average


def test_i3_c10_ties_with_full_density_and_one_expert_returns_the_expert():
    seed, a = swiglu_state(0), swiglu_state(1)
    assert maxdiff(ties(seed, [a], 1.0, 1.0), a) <= EXACT


def test_i3_c10_ties_trims_elects_a_sign_and_averages_the_agreeing_values():
    """A hand-worked case: density 0.5 keeps the two largest |tau| of each expert; the elected sign is that of the
    sum of the kept values; the merge averages the kept values with that sign."""
    seed = {"ffn.0.w": torch.zeros(4, dtype=F64)}
    a = {"ffn.0.w": torch.tensor([3.0, -1.0, 0.5, 2.0], dtype=F64)}
    b = {"ffn.0.w": torch.tensor([-1.0, -2.0, 0.4, 2.0], dtype=F64)}
    # kept: a -> [3, 0, 0, 2], b -> [0, -2, 0, 2]; sign of the sum [3, -2, 0, 4]; agreeing means [3, -2, 0, 2]
    assert torch.equal(ties(seed, [a, b], 0.5, 1.0)["ffn.0.w"], torch.tensor([3.0, -2.0, 0.0, 2.0], dtype=F64))
    assert torch.equal(ties(seed, [a, b], 0.5, 0.5)["ffn.0.w"], torch.tensor([1.5, -1.0, 0.0, 1.0], dtype=F64))


def test_i3_c10_dare_without_drops_is_task_arithmetic():
    seed, a, b = swiglu_state(0), swiglu_state(1), swiglu_state(2)
    assert equal(dare(seed, [a, b], 0.0, rng_seed=7, lam=0.5), task_arithmetic(seed, [a, b], 0.5))


def test_i3_c10_dare_is_deterministic_and_independent_of_key_order():
    seed, a = swiglu_state(0), swiglu_state(1)
    first = dare(seed, [a], 0.5, rng_seed=7, lam=1.0)
    assert equal(first, dare(seed, [a], 0.5, rng_seed=7, lam=1.0))
    reordered = dict(reversed(list(a.items())))
    assert equal(first, dare(dict(reversed(list(seed.items()))), [reordered], 0.5, rng_seed=7, lam=1.0))
    assert not equal(first, dare(seed, [a], 0.5, rng_seed=8, lam=1.0))


def test_i3_c10_dare_rescales_kept_entries_and_is_unbiased():
    """Each entry of one draw is 0 or tau / (1 - p); over 2000 seeds the keep rate is within 3 SE of 1 - p and the
    mean merge within 3 SE of task arithmetic, entry by entry (the draws are fixed, so the check is deterministic)."""
    seed, a = swiglu_state(0, d=3, f=2, layers=(0,)), swiglu_state(1, d=3, f=2, layers=(0,))
    p, draws = 0.4, 2000
    tau = {k: a[k] - seed[k] for k in a}
    samples = {k: [] for k in a}
    for s in range(draws):
        out = dare(seed, [a], p, rng_seed=s, lam=1.0)
        for k in a:
            delta = out[k] - seed[k]
            kept = delta != 0
            assert torch.allclose(delta[kept], tau[k][kept] / (1 - p), rtol=0, atol=EXACT)
            samples[k].append(delta)
    ta = task_arithmetic(seed, [a], 1.0)
    keep = torch.cat([(torch.stack(v) != 0).flatten() for v in samples.values()]).double()
    assert abs(float(keep.mean()) - (1 - p)) <= 3 * float(keep.std()) / len(keep) ** 0.5
    for k, v in samples.items():
        stack = torch.stack(v)
        se = stack.std(0) / draws**0.5
        assert bool(((stack.mean(0) - (ta[k] - seed[k])).abs() <= 3 * se).all()), k


def test_i3_c10_merges_refuse_inconsistent_inputs():
    a, b = swiglu_state(0), swiglu_state(1)
    with pytest.raises(ValueError, match="weights"):
        weight_average([a, b], [0.6, 0.6])
    with pytest.raises(ValueError, match="weights"):
        weight_average([a, b], [1.5, -0.5])
    with pytest.raises(ValueError, match="different keys"):
        weight_average([a, {k: v for k, v in b.items() if k != "ffn.0.up.weight"}], [0.5, 0.5])
    with pytest.raises(ValueError, match="state 1"):
        task_arithmetic(a, [{**b, "ffn.0.up.weight": b["ffn.0.up.weight"].float()}], 1.0)
    with pytest.raises(ValueError, match="backbone"):
        weight_average([{"backbone.emb.weight": torch.ones(2)}] * 2, [0.5, 0.5])
    with pytest.raises(ValueError):
        ties(a, [b], 0.0, 1.0)
    with pytest.raises(ValueError):
        dare(a, [b], 1.0, rng_seed=0, lam=1.0)


# --- Re-Basin ----------------------------------------------------------------------------------------------------


def permuted_copy(state: dict, seed: int) -> tuple[dict, dict[int, np.ndarray]]:
    """`state` with a random permutation of each layer's channels; returns it and each layer's permutation."""
    rng = np.random.default_rng(seed)
    out, perms = dict(state), {}
    for layer in LAYERS:
        perms[layer] = rng.permutation(channel_similarity(state, state, layer).shape[0])  # f, or n for family II
        out = permute_channels(out, layer, perms[layer])
    return out, perms


@pytest.mark.parametrize("make", [swiglu_state, slot_state])
def test_i3_c10_rebasin_recovers_a_planted_permutation(make):
    a = make(0)
    b, planted = permuted_copy(a, seed=1)
    assert not equal(a, b)
    found = rebasin_permutations(a, b, LAYERS)
    for layer in LAYERS:
        assert np.array_equal(found[layer], np.argsort(planted[layer]))  # the inverse of the planted permutation
    aligned, share = rebasin_align(a, b, LAYERS)
    assert equal(aligned, a)
    assert maxdiff(weight_average([a, aligned], [0.5, 0.5]), a) <= EXACT
    fixed = sum(int((found[i] == np.arange(len(found[i]))).sum()) for i in LAYERS)
    assert share == fixed / sum(len(found[i]) for i in LAYERS) < 1.0
    assert alignment_label(share) == NON_IDENTITY


@pytest.mark.parametrize("make", [swiglu_state, slot_state])
def test_i3_c10_rebasin_of_identical_experts_is_the_identity_and_equivalent_to_averaging(make):
    a = make(0)
    aligned, share = rebasin_align(a, dict(a), LAYERS)
    assert share == 1.0 and alignment_label(share) == EQUIVALENT_TO_AVERAGING
    assert equal(aligned, a)
    assert all(np.array_equal(p, np.arange(len(p))) for p in rebasin_permutations(a, a, LAYERS).values())


def test_i3_c10_channel_similarity_is_the_sum_of_three_gram_products():
    a, b = swiglu_state(0), swiglu_state(1)
    G = lambda s, k: s[f"ffn.1.{k}.weight"]  # noqa: E731
    expected = G(a, "gate") @ G(b, "gate").T + G(a, "up") @ G(b, "up").T + G(a, "down").T @ G(b, "down")
    assert float((channel_similarity(a, b, 1) - expected).abs().max()) <= EXACT


def brute_force(score: np.ndarray) -> tuple[float, tuple[int, ...]]:
    n = score.shape[0]
    return max((score[np.arange(n), list(p)].sum(), p) for p in itertools.permutations(range(n)))


def test_i3_c10_numpy_hungarian_equals_brute_force():
    rng = np.random.default_rng(0)
    for trial in range(200):
        n = 1 + trial % 6
        score = rng.normal(size=(n, n))
        perm = solve_lap(score, "numpy")
        best, best_perm = brute_force(score)
        assert np.array_equal(perm, np.array(best_perm)), trial
        assert abs(score[np.arange(n), perm].sum() - best) <= EXACT
    for trial in range(50):  # integer scores: ties, so only the optimal value is unique
        n = 2 + trial % 5
        score = rng.integers(0, 3, size=(n, n)).astype(float)
        perm = solve_lap(score, "numpy")
        assert sorted(perm.tolist()) == list(range(n))
        assert score[np.arange(n), perm].sum() == brute_force(score)[0], trial


def test_i3_c10_numpy_hungarian_finds_a_planted_assignment_at_its_size_limit():
    rng = np.random.default_rng(1)
    n = NUMPY_LAP_MAX
    planted = rng.permutation(n)
    score = rng.normal(size=(n, n))
    score[np.arange(n), planted] += 10.0  # every planted entry beats any row's alternatives by far
    assert np.array_equal(solve_lap(score, "numpy"), planted)


def test_i3_c10_numpy_hungarian_equals_scipy():
    pytest.importorskip("scipy", reason="scipy is not installed here (the CPU lane); fork/.venv-bilingual runs this")
    rng = np.random.default_rng(2)
    for trial in range(20):
        score = rng.normal(size=(64, 64))
        assert np.array_equal(solve_lap(score, "numpy"), solve_lap(score, "scipy")), trial


def test_i3_c10_solve_lap_refuses_large_numpy_problems_and_bad_input(monkeypatch):
    big = np.zeros((NUMPY_LAP_MAX + 1, NUMPY_LAP_MAX + 1))
    with pytest.raises(ValueError, match="scipy"):
        solve_lap(big, "numpy")
    monkeypatch.setattr(merge, "linear_sum_assignment", None)  # as on the CPU lane
    with pytest.raises(ValueError, match="scipy"):
        solve_lap(big, "auto")
    with pytest.raises(RuntimeError, match="scipy"):
        solve_lap(np.zeros((3, 3)), "scipy")
    assert np.array_equal(solve_lap(np.eye(3), "auto"), np.arange(3))
    for bad in (np.zeros((2, 3)), np.array([[0.0, np.nan], [0.0, 0.0]]), np.zeros(3)):
        with pytest.raises(ValueError):
            solve_lap(bad)
    with pytest.raises(ValueError):
        solve_lap(np.eye(2), "hungarian")
