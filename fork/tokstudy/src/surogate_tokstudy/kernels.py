"""Sequence kernels of the study's Mamba-2 and H-Net layers: chunked SSD, the H-Net smoothing EMA, the depthwise
causal conv. Imports only torch; `mamba_ssm` only inside the "triton" route, guarded.

Shapes (b batch, T or L positions, H heads of P channels, G groups of N states; head h reads group h // (H / G)):

- ssd(x [b,T,H,P], dt [b,T,H], A [H], B and C [b,T,G,N]) -> y [b,T,H,P]: the Mamba-2 recurrence
  h_t = exp(dt_t A) h_{t-1} + dt_t x_t B_t^T, y_t = h_t C_t, per head, h_0 = `initial_state` (default 0). dt is
  taken as given (the mixer applies softplus, dt_bias and the clamp); D, the gate and the norm are the mixer's.
  "torch" is the chunked SSD form of Dao & Gu 2024 (ssd_minimal_discrete, with groups and an initial state);
  ssd_reference is the recurrence itself, one position after another. Both evaluate in fp32 (fp64 for fp64
  inputs) with autocast off, as the Triton kernel accumulates in fp32, and return x's dtype.
- ema_scan(z [b,L,d], P [b,L]) -> zbar [b,L,d]: zbar_k = P_k z_k + (1 - P_k) zbar_{k-1}, zbar_0 = 0 (H-Net eq. 5).
  "torch" is a chunked form without logarithms (P = 1 makes the decay 0, whose log has no gradient): segment products
  of the decays come from a masked cumprod, within chunks and between chunks. ema_scan_reference is the loop.
- causal_conv(x [b,T,D], weight [D,K], bias [D]): depthwise conv, output t reads inputs t-K+1..t (weight[:, K-1-k]
  for t-k), before the activation.

Backend "triton" calls mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined (signature read from the
mamba-ssm 2.3.2.post1 sdist [verified]) and is an error, never a silent fallback, when that import fails or the
tensors are not on CUDA. The EMA rides on the same kernel as the H-Net code does: dt = -log(1 - P), x = z / dt,
A = -1, B = P, C = 1 (ema_via_ssd); that needs 0 < P < 1, which the H-Net dechunk guarantees by clamping. Importing
that module runs mamba_ssm/__init__.py, which imports the CUDA extension selective_scan_cuda unguarded
[verified, 2.3.2.post1 source], so a MAMBA_SKIP_CUDA_BUILD install cannot import it (README.md, 模型与内核).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

BACKENDS = ("torch", "triton")
EMA_CHUNK = 64


def _check_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")


def compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """fp64 stays fp64; everything else (bf16, fp16, fp32) is evaluated in fp32."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def _no_autocast(device: torch.device):
    return torch.autocast(device_type=device.type, enabled=False)


def _check_ssd_shapes(x, dt, A, B, C, initial_state) -> None:
    if x.dim() != 4:
        raise ValueError(f"x must be [b, T, H, P], got {list(x.shape)}")
    b, T, H, P = x.shape
    if tuple(dt.shape) != (b, T, H) or tuple(A.shape) != (H,):
        raise ValueError(f"dt must be [b, T, H] = {[b, T, H]} and A [H], got {list(dt.shape)} and {list(A.shape)}")
    if B.dim() != 4 or tuple(B.shape[:2]) != (b, T) or B.shape != C.shape:
        raise ValueError(f"B and C must be [b, T, G, N] with b, T = {b}, {T}, got {list(B.shape)}, {list(C.shape)}")
    if H % B.shape[2]:
        raise ValueError(f"{H} heads do not split into {B.shape[2]} groups")
    if initial_state is not None and tuple(initial_state.shape) != (b, H, P, B.shape[3]):
        raise ValueError(f"initial_state must be [b, H, P, N], got {list(initial_state.shape)}")


def segsum(a: torch.Tensor) -> torch.Tensor:
    """a [..., l] -> [..., l, l]: a[j+1] + ... + a[i] for j <= i, -inf above the diagonal (so exp gives 0 there and no
    exp of a positive sum reaches the backward)."""
    l = a.shape[-1]  # noqa: E741
    ones = torch.ones(l, l, dtype=torch.bool, device=a.device)
    sums = a[..., :, None].expand(*a.shape, l).masked_fill(~ones.tril(-1), 0).cumsum(-2)
    return sums.masked_fill(~ones.tril(0), -torch.inf)


def ssd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk: int,
    backend: str = "torch",
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """The chunked SSD (module docstring). Returns y, or (y, final state [b, H, P, N]) with return_final_state."""
    _check_backend(backend)
    _check_ssd_shapes(x, dt, A, B, C, initial_state)
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    if backend == "triton":
        return _ssd_triton(x, dt, A, B, C, chunk, initial_state, return_final_state)
    with _no_autocast(x.device):
        y, state = _ssd_torch(x, dt, A, B, C, chunk, initial_state)
    return (y, state) if return_final_state else y


def _ssd_torch(x, dt, A, B, C, chunk, initial_state):
    out_dtype, ct = x.dtype, compute_dtype(x.dtype)
    x, dt, A, B, C = (t.to(ct) for t in (x, dt, A, B, C))
    b, T, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    E = H // G  # heads per group
    pad = -T % chunk
    c = (T + pad) // chunk
    # padding has no input and no decay, so the final state is that of the last real position
    X = F.pad(x * dt[..., None], (0, 0, 0, 0, 0, pad)).reshape(b, c, chunk, G, E, P)
    a = F.pad(dt * A, (0, 0, 0, pad)).reshape(b, c, chunk, G, E).permute(0, 3, 4, 1, 2)  # b g e c l
    Bc, Cc = (F.pad(t, (0, 0, 0, 0, 0, pad)).reshape(b, c, chunk, G, N) for t in (B, C))
    a_cum = a.cumsum(-1)
    # 1. within each chunk: the quadratic (attention-like) form
    CB = torch.einsum("bclgn,bcsgn->bgcls", Cc, Bc)
    y_diag = torch.einsum("bgcls,bgecls,bcsgep->bclgep", CB, torch.exp(segsum(a)), X)
    # 2. the state each chunk's inputs leave at its end
    to_end = torch.exp(a_cum[..., -1:] - a_cum)
    states = torch.einsum("bcsgn,bgecs,bcsgep->bcgepn", Bc, to_end, X)
    # 3. states carried between chunks, the initial state first
    if initial_state is None:
        first = states.new_zeros(b, 1, G, E, P, N)
    else:
        first = initial_state.to(ct).reshape(b, 1, G, E, P, N)
    states = torch.cat([first, states], 1)
    between = torch.exp(segsum(F.pad(a_cum[..., -1], (1, 0))))  # b g e (c+1) (c+1)
    carried = torch.einsum("bgezc,bcgepn->bzgepn", between, states)
    # 4. the carried state read out at every position of the next chunk
    y_off = torch.einsum("bclgn,bcgepn,bgecl->bclgep", Cc, carried[:, :-1], torch.exp(a_cum))
    y = (y_diag + y_off).reshape(b, c * chunk, H, P)[:, :T]
    return y.to(out_dtype), carried[:, -1].reshape(b, H, P, N)


def ssd_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """The recurrence, one position after another (the oracle of `ssd`)."""
    _check_ssd_shapes(x, dt, A, B, C, initial_state)
    out_dtype, ct = x.dtype, compute_dtype(x.dtype)
    x, dt, A, B, C = (t.to(ct) for t in (x, dt, A, B, C))
    b, T, H, P = x.shape
    E = H // B.shape[2]
    Bh, Ch = (t.repeat_interleave(E, dim=2) for t in (B, C))  # b T H N
    h = x.new_zeros(b, H, P, B.shape[3]) if initial_state is None else initial_state.to(ct)
    ys = []
    for t in range(T):
        decay = torch.exp(dt[:, t] * A)[..., None, None]
        h = decay * h + (dt[:, t, :, None] * x[:, t])[..., None] * Bh[:, t, :, None, :]
        ys.append((h * Ch[:, t, :, None, :]).sum(-1))
    y = torch.stack(ys, 1).to(out_dtype)
    return (y, h) if return_final_state else y


def _ssd_triton(x, dt, A, B, C, chunk, initial_state, return_final_state):
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except ImportError as err:
        raise RuntimeError(
            "backend 'triton' needs mamba_ssm.ops.triton.ssd_combined, which does not import here "
            f"({type(err).__name__}: {err}); use backend 'torch' and record it (fork/tokstudy/README.md, G0-ts)"
        ) from err
    if not x.is_cuda:
        raise RuntimeError("backend 'triton' runs on CUDA tensors only; use backend 'torch' on the CPU")
    out = mamba_chunk_scan_combined(
        x, dt, A, B, C, chunk, initial_states=initial_state, return_final_states=return_final_state
    )
    return out


def _check_ema_shapes(z: torch.Tensor, P: torch.Tensor) -> None:
    if z.dim() != 3 or tuple(P.shape) != tuple(z.shape[:2]):
        raise ValueError(f"z must be [b, L, d] and P [b, L], got {list(z.shape)} and {list(P.shape)}")


def ema_scan(z: torch.Tensor, P: torch.Tensor, *, backend: str = "torch", chunk: int = EMA_CHUNK) -> torch.Tensor:
    """zbar_k = P_k z_k + (1 - P_k) zbar_{k-1}, zbar_0 = 0 (module docstring); P in [0, 1] ("torch") or (0, 1)."""
    _check_backend(backend)
    _check_ema_shapes(z, P)
    if backend == "triton":
        return ema_via_ssd(z, P, backend="triton")
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    out_dtype, ct = z.dtype, compute_dtype(z.dtype)
    with _no_autocast(z.device):
        z, P = z.to(ct), P.to(ct)
        b, L, d = z.shape
        pad = -L % chunk
        c = (L + pad) // chunk
        # padding: P = 0, so no input and a decay of 1
        decay = F.pad(1 - P, (0, pad), value=1.0).reshape(b, c, chunk)
        u = F.pad(P[..., None] * z, (0, 0, 0, pad)).reshape(b, c, chunk, d)
        inside = _segment_products(decay)  # b c l l
        y = torch.einsum("bcij,bcjd->bcid", inside, u)  # the EMA of each chunk from a zero state
        from_start = decay.cumprod(-1)  # b c l: product of the decays from the chunk start to i
        totals = from_start[..., -1]  # b c
        # end state after chunk k: sum over k' <= k of (product of totals k'+1..k) * y[k', last]
        ends = torch.einsum("bkj,bjd->bkd", _segment_products(totals), y[:, :, -1])
        entering = F.pad(ends, (0, 0, 1, 0))[:, :-1]  # the state entering chunk k
        out = y + from_start[..., None] * entering[:, :, None, :]
        return out.reshape(b, c * chunk, d)[:, :L].to(out_dtype)


def _segment_products(a: torch.Tensor) -> torch.Tensor:
    """a [..., l] -> [..., l, l]: a[j+1] * ... * a[i] for j <= i (1 on the diagonal), 0 above it. A masked cumprod,
    so a zero decay needs no logarithm."""
    l = a.shape[-1]  # noqa: E741
    ones = torch.ones(l, l, dtype=torch.bool, device=a.device)
    products = a[..., :, None].expand(*a.shape, l).masked_fill(~ones.tril(-1), 1.0).cumprod(-2)
    return products.masked_fill(~ones.tril(0), 0.0)


def ema_scan_reference(z: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """The EMA loop (the oracle of `ema_scan`)."""
    _check_ema_shapes(z, P)
    out_dtype, ct = z.dtype, compute_dtype(z.dtype)
    z, P = z.to(ct), P.to(ct)
    state = z.new_zeros(z.shape[0], z.shape[2])
    out = []
    for k in range(z.shape[1]):
        state = P[:, k, None] * z[:, k] + (1 - P[:, k, None]) * state
        out.append(state)
    return torch.stack(out, 1).to(out_dtype)


def ema_head_dim(d: int) -> int:
    """Head width of the EMA on the SSD kernel: the largest of 64, 32, 16, 8 dividing d, else d."""
    return next((p for p in (64, 32, 16, 8) if d % p == 0), d)


def ema_via_ssd(z: torch.Tensor, P: torch.Tensor, *, backend: str, chunk: int = 256) -> torch.Tensor:
    """The EMA as an SSD with one state: dt = -log(1 - P), x = z / dt, A = -1, B = P, C = 1, so that
    exp(dt A) = 1 - P and dt x B = P z. Needs 0 < P < 1. Runs on either backend; the CPU tests check the mapping
    through backend "torch"."""
    _check_ema_shapes(z, P)
    b, L, d = z.shape
    p = ema_head_dim(d)
    H = d // p
    ct = compute_dtype(z.dtype)
    with _no_autocast(z.device):
        Pc = P.to(ct)
        dt = -torch.log1p(-Pc)  # b L
        x = (z.to(ct) / dt[..., None]).reshape(b, L, H, p)
        A = -torch.ones(H, dtype=ct, device=z.device)
        B = Pc[..., None, None]  # b L 1 1
        C = torch.ones_like(B)
        y = ssd(x, dt[..., None].expand(b, L, H), A, B, C, chunk=chunk, backend=backend)
    return y.reshape(b, L, d).to(z.dtype)


def causal_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Depthwise causal conv of x [b, T, D] with weight [D, K] and bias [D] (module docstring)."""
    if x.dim() != 3 or weight.dim() != 2 or weight.shape[0] != x.shape[2]:
        raise ValueError(f"x must be [b, T, D] and weight [D, K], got {list(x.shape)} and {list(weight.shape)}")
    T, D, K = x.shape[1], x.shape[2], weight.shape[1]
    out = F.conv1d(x.transpose(1, 2), weight[:, None, :], bias, padding=K - 1, groups=D)
    return out[..., :T].transpose(1, 2)
