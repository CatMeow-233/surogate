"""Same-initialisation merges of expert FFN states: the B3 baselines (MS §7.1, §7.2; NOV-6, IMPL-19).

Every merge acts on expert states only, mappings `ffn.{layer}.{param}` -> tensor as an expert pack stores them
(`model.bank.expert_keys`). The backbone is shared and identical by construction, so it is never merged; a
`backbone.*` key is refused. All states of one merge must have the same keys, shapes and dtypes. The arithmetic runs
in float64 and the result is cast back to the inputs' dtype.

- `weight_average(states, weights)`: sum_i w_i theta_i with w on the simplex (B3a; B3-II with w = g).
- `task_arithmetic(seed, experts, lam)`: theta_seed + lam * sum_i tau_i, tau_i = theta_i - theta_seed (B3b).
- `ties(seed, experts, density, lam)`: TIES-Merging (Yadav et al. 2023, arXiv:2306.01708): keep the top `density`
  share of each tau_i by magnitude, elect a sign per element from the sum of the kept values, average the kept values
  that agree with it (B3c). Trimming is per tensor, as in common implementations; the paper trims the whole task
  vector at once [建议: a global variant is not implemented].
- `dare(seed, experts, drop_p, rng_seed, lam)`: DARE (Yu et al. 2024): drop each element of each tau_i with
  probability p, rescale the rest by 1 / (1 - p), then task arithmetic (B3d). Each tensor's mask comes from its own
  generator, seeded from (rng_seed, expert index, key), so masks do not depend on dict order.
- `channel_similarity`, `solve_lap`, `permute_channels`, `rebasin_permutations`, `rebasin_align`,
  `alignment_label`: Git Re-Basin for a frozen residual basis (B3-rb). Only the hidden channels of each layer can be
  permuted, so alignment is one exact linear assignment per layer on G_a G_b^T + U_a U_b^T + D_a^T D_b, solved by
  Hungarian, not by row-wise argmax. When every layer's solution is the identity the result is labelled
  "equivalent to averaging" (NOV-6). Family II permutes its n slots: rows of R, columns of W (Q and b are shared).

`solve_lap` uses scipy when it is installed (`fork/.venv-bilingual` has it; the CPU lane does not) and a numpy
Hungarian otherwise, which refuses n > 256: the pilot's n = 1728 needs scipy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import torch

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # the CPU lane has no scipy
    linear_sum_assignment = None

NUMPY_LAP_MAX = 256
SIMPLEX_ATOL = 1e-9
EQUIVALENT_TO_AVERAGING = "equivalent to averaging"
NON_IDENTITY = "aligned (non-identity permutation)"

# Expert parameters with a permutable hidden dimension, and which dimension it is.
PERMUTABLE = {"gate.weight": 0, "up.weight": 0, "down.weight": 1, "R.weight": 0, "W.weight": 1}

State = Mapping[str, torch.Tensor]


def _check_states(states: Sequence[State]) -> list[str]:
    if not states:
        raise ValueError("a merge needs at least one state")
    keys = sorted(states[0])
    if not keys:
        raise ValueError("a merge needs a non-empty state")
    shared = [k for k in keys if k.startswith("backbone.")]
    if shared:
        raise ValueError(f"merges act on expert FFN states only; the backbone is shared, got {shared[:3]}")
    for i, state in enumerate(states[1:], 1):
        if sorted(state) != keys:
            raise ValueError(f"state {i} has different keys: {sorted(set(state) ^ set(keys))[:5]}")
        for k in keys:
            a, b = states[0][k], state[k]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"{k}: state {i} is {b.dtype} {list(b.shape)}, state 0 is {a.dtype} {list(a.shape)}")
    return keys


def _f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float64)


def weight_average(states: Sequence[State], weights: Sequence[float]) -> dict[str, torch.Tensor]:
    keys = _check_states(states)
    w = [float(x) for x in weights]
    if len(w) != len(states) or min(w) < 0 or abs(sum(w) - 1.0) > SIMPLEX_ATOL:
        raise ValueError(f"weights must be {len(states)} non-negative numbers summing to 1, got {w}")
    out = {}
    for k in keys:
        acc = sum(wi * _f64(s[k]) for wi, s in zip(w, states, strict=True))
        out[k] = acc.to(states[0][k].dtype)
    return out


def _deltas(seed: State, experts: Sequence[State]) -> tuple[list[str], list[dict[str, torch.Tensor]]]:
    if not experts:
        raise ValueError("a merge needs at least one expert")
    keys = _check_states([seed, *experts])
    return keys, [{k: _f64(e[k]) - _f64(seed[k]) for k in keys} for e in experts]


def task_arithmetic(seed: State, experts: Sequence[State], lam: float) -> dict[str, torch.Tensor]:
    keys, taus = _deltas(seed, experts)
    return {k: (_f64(seed[k]) + float(lam) * sum(t[k] for t in taus)).to(seed[k].dtype) for k in keys}


def _trim(tau: torch.Tensor, density: float) -> torch.Tensor:
    """tau with all but its ceil(density * n) largest-magnitude entries set to 0 (stable order on ties)."""
    flat = tau.reshape(-1)
    k = min(flat.numel(), int(np.ceil(density * flat.numel())))
    keep = torch.argsort(flat.abs(), descending=True, stable=True)[:k]
    out = torch.zeros_like(flat)
    out[keep] = flat[keep]
    return out.reshape(tau.shape)


def ties(seed: State, experts: Sequence[State], density: float, lam: float) -> dict[str, torch.Tensor]:
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must lie in (0, 1], got {density}")
    keys, taus = _deltas(seed, experts)
    out = {}
    for k in keys:
        kept = torch.stack([_trim(t[k], density) for t in taus])  # [n_experts, ...]
        sign = torch.sign(kept.sum(0))
        agree = (torch.sign(kept) == sign) & (kept != 0)
        merged = (kept * agree).sum(0) / agree.sum(0).clamp(min=1)
        out[k] = (_f64(seed[k]) + float(lam) * merged).to(seed[k].dtype)
    return out


def _mask_seed(rng_seed: int, expert: int, key: str) -> int:
    digest = hashlib.sha256(f"bilingual-dare/1:{int(rng_seed)}:{expert}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def dare(seed: State, experts: Sequence[State], drop_p: float, rng_seed: int, lam: float) -> dict[str, torch.Tensor]:
    if not 0.0 <= drop_p < 1.0:
        raise ValueError(f"drop_p must lie in [0, 1), got {drop_p}")
    keys, taus = _deltas(seed, experts)
    out = {}
    for k in keys:
        total = torch.zeros_like(taus[0][k])
        for i, t in enumerate(taus):
            g = torch.Generator().manual_seed(_mask_seed(rng_seed, i, k))
            keep = torch.rand(t[k].shape, generator=g, dtype=torch.float64) >= drop_p
            total = total + torch.where(keep, t[k] / (1.0 - drop_p), torch.zeros_like(t[k]))
        out[k] = (_f64(seed[k]) + float(lam) * total).to(seed[k].dtype)
    return out


# --- Re-Basin --------------------------------------------------------------------------------------------------


def _layer_params(state: State, layer: int) -> dict[str, int]:
    prefix = f"ffn.{layer}."
    params = {k: PERMUTABLE[k[len(prefix) :]] for k in state if k.startswith(prefix) and k[len(prefix) :] in PERMUTABLE}
    if not params:
        raise ValueError(f"no permutable parameters of layer {layer} (keys ffn.{layer}.<{'|'.join(PERMUTABLE)}>)")
    return params


def channel_similarity(a: State, b: State, layer: int) -> torch.Tensor:
    """S[p, q] = <channel p of a, channel q of b> summed over the layer's parameters, in float64: for SwiGLU
    G_a G_b^T + U_a U_b^T + D_a^T D_b, [f, f]; for family II R_a R_b^T + W_a^T W_b, [n, n]."""
    params = _layer_params(a, layer)
    if set(params) != set(_layer_params(b, layer)):
        raise ValueError(f"layer {layer}: the two states hold different parameters")
    score = None
    for key, dim in sorted(params.items()):
        ma, mb = _f64(a[key]).movedim(dim, 0).flatten(1), _f64(b[key]).movedim(dim, 0).flatten(1)
        term = ma @ mb.T
        score = term if score is None else score + term
    return score


def _hungarian_max(score: np.ndarray) -> np.ndarray:
    """Exact maximum-score assignment of a square matrix: the O(n^3) Hungarian method with potentials (shortest
    augmenting paths), vectorised over columns. Returns perm with perm[row] = column."""
    cost = -score
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    owner = np.zeros(n + 1, dtype=np.int64)  # owner[j] = row (1-based) assigned to column j; column 0 is the root
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        owner[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = owner[j0]
            free = ~used[1:]
            reduced = cost[i0 - 1] - u[i0] - v[1:]
            better = free & (reduced < minv[1:])
            minv[1:][better] = reduced[better]
            way[1:][better] = j0
            candidates = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(candidates)) + 1
            delta = candidates[j1 - 1]
            done = np.nonzero(used)[0]
            u[owner[done]] += delta
            v[done] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
    perm = np.empty(n, dtype=np.int64)
    perm[owner[1:] - 1] = np.arange(n)
    return perm


def solve_lap(score: np.ndarray, solver: Literal["auto", "scipy", "numpy"] = "auto") -> np.ndarray:
    """The permutation maximising sum_p score[p, perm[p]] of a square, finite score matrix."""
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 2 or score.shape[0] != score.shape[1] or not np.isfinite(score).all():
        raise ValueError(f"solve_lap needs a finite square matrix, got shape {score.shape}")
    if solver not in ("auto", "scipy", "numpy"):
        raise ValueError(f"solver must be 'auto', 'scipy' or 'numpy': {solver!r}")
    if solver == "scipy" or (solver == "auto" and linear_sum_assignment is not None):
        if linear_sum_assignment is None:
            raise RuntimeError("solver='scipy' needs scipy, which is not installed (fork/.venv-bilingual has it)")
        rows, cols = linear_sum_assignment(score, maximize=True)
        perm = np.empty(score.shape[0], dtype=np.int64)
        perm[rows] = cols
        return perm
    n = score.shape[0]
    if n > NUMPY_LAP_MAX:
        raise ValueError(
            f"the numpy Hungarian solver is limited to n <= {NUMPY_LAP_MAX} (got n = {n}); install scipy "
            "(fork/.venv-bilingual has it) and use solver='scipy' or 'auto'"
        )
    return _hungarian_max(score)


def permute_channels(state: State, layer: int, perm) -> dict[str, torch.Tensor]:
    """A copy of `state` whose layer-`layer` hidden channels are reordered: new channel p is old channel perm[p].
    The expert's function is unchanged (v1.0 check 3)."""
    params = _layer_params(state, layer)
    out = dict(state)
    for key, dim in params.items():
        index = torch.as_tensor(np.asarray(perm), dtype=torch.long, device=state[key].device)
        if sorted(index.tolist()) != list(range(state[key].shape[dim])):
            raise ValueError(f"{key}: perm is not a permutation of {state[key].shape[dim]} channels")
        out[key] = state[key].index_select(dim, index)
    return out


def rebasin_permutations(
    ref: State, other: State, layers: Sequence[int], solver: Literal["auto", "scipy", "numpy"] = "auto"
) -> dict[int, np.ndarray]:
    """Per layer, the permutation of `other`'s channels that best matches `ref`: perm[p] is the channel of `other`
    placed at position p."""
    return {layer: solve_lap(channel_similarity(ref, other, layer).numpy(), solver) for layer in layers}


def rebasin_align(
    ref: State, other: State, layers: Sequence[int], solver: Literal["auto", "scipy", "numpy"] = "auto"
) -> tuple[dict[str, torch.Tensor], float]:
    """(`other` with its channels aligned to `ref` in each of `layers`, the share of channels left in place). The
    second value is 1.0 exactly when every layer's assignment is the identity; then aligned averaging equals plain
    averaging (`alignment_label`)."""
    _check_states([ref, other])
    perms = rebasin_permutations(ref, other, layers, solver)
    aligned = dict(other)
    for layer, perm in perms.items():
        aligned = permute_channels(aligned, layer, perm)
    total = sum(len(p) for p in perms.values())
    fixed = sum(int((p == np.arange(len(p))).sum()) for p in perms.values())
    return aligned, (fixed / total if total else 1.0)


def alignment_label(identity_share: float) -> str:
    """How a B3-rb row is reported: an identity alignment is plain averaging, not a separate baseline (NOV-6)."""
    return EQUIVALENT_TO_AVERAGING if identity_share == 1.0 else NON_IDENTITY
