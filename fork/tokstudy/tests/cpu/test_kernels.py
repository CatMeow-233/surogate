"""Kernels (surogate_tokstudy.kernels, spec step S11): the chunked SSD against the recurrence, the chunked EMA
against its loop, gradients by finite differences, the EMA-on-SSD mapping of the Triton route, the conv taps, the
compute dtype, and refusals. fp64 on the CPU throughout, random inputs from fixed seeds.
"""

import pytest
import torch
from surogate_tokstudy import kernels

F64 = torch.float64


@pytest.fixture(autouse=True)
def deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


def ssd_inputs(b, T, H, P, G, N, seed, dtype=F64):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, T, H, P, generator=g, dtype=dtype)
    dt = torch.rand(b, T, H, generator=g, dtype=dtype) * 0.9 + 0.05
    A = -(torch.rand(H, generator=g, dtype=dtype) * 3 + 0.1)
    B = torch.randn(b, T, G, N, generator=g, dtype=dtype)
    C = torch.randn(b, T, G, N, generator=g, dtype=dtype)
    h0 = torch.randn(b, H, P, N, generator=g, dtype=dtype)
    return x, dt, A, B, C, h0


@pytest.mark.parametrize("chunk", [4, 128])
@pytest.mark.parametrize("T", [1, 8, 37, 256])
@pytest.mark.parametrize("initial", [False, True])
def test_chunked_ssd_equals_the_recurrence(chunk, T, initial):
    """4 heads in 2 groups (head h reads group h // 2), T a multiple of the chunk or not, with an initial state."""
    x, dt, A, B, C, h0 = ssd_inputs(2, T, 4, 3, 2, 5, seed=T + chunk)
    h0 = h0 if initial else None
    y, state = kernels.ssd(x, dt, A, B, C, chunk=chunk, initial_state=h0, return_final_state=True)
    y_ref, state_ref = kernels.ssd_reference(x, dt, A, B, C, initial_state=h0, return_final_state=True)
    assert y.dtype == F64 and y.shape == x.shape and state.shape == (2, 4, 3, 5)
    assert float((y - y_ref).abs().max()) <= 1e-10
    assert float((state - state_ref).abs().max()) <= 1e-10
    assert float(y_ref.abs().max()) > 0.5  # the comparison is not between two near-zero tensors


def test_ssd_groups_are_read_by_head_blocks():
    """With G = 2, heads 0-1 read group 0 and heads 2-3 group 1: changing group 1 changes only heads 2-3."""
    x, dt, A, B, C, _ = ssd_inputs(1, 9, 4, 2, 2, 3, seed=1)
    y = kernels.ssd(x, dt, A, B, C, chunk=4)
    B2 = B.clone()
    B2[:, :, 1] += 1.0
    y2 = kernels.ssd(x, dt, A, B2, C, chunk=4)
    assert torch.equal(y[:, :, :2], y2[:, :, :2]) and not torch.allclose(y[:, :, 2:], y2[:, :, 2:])


def test_ssd_chunks_carry_state_like_one_pass():
    """Two calls with the state carried equal one call over the concatenation (the decode / piecewise form)."""
    x, dt, A, B, C, _ = ssd_inputs(2, 40, 2, 3, 1, 4, seed=2)
    full = kernels.ssd(x, dt, A, B, C, chunk=8)
    first, state = kernels.ssd(x[:, :17], dt[:, :17], A, B[:, :17], C[:, :17], chunk=8, return_final_state=True)
    second = kernels.ssd(x[:, 17:], dt[:, 17:], A, B[:, 17:], C[:, 17:], chunk=8, initial_state=state)
    assert float((torch.cat([first, second], 1) - full).abs().max()) <= 1e-12


@pytest.mark.parametrize("chunk", [1, 4, 64])
def test_chunked_ema_equals_the_loop(chunk):
    """P in [0, 1] including exact 1 (a forced boundary: no decay term) and exact 0; L not a multiple of chunk."""
    g = torch.Generator().manual_seed(chunk)
    z = torch.randn(3, 150, 5, generator=g, dtype=F64)
    P = torch.rand(3, 150, generator=g, dtype=F64)
    P[:, 0] = 1.0
    P[1, 40] = 1.0
    P[2, 63:66] = 1.0
    P[0, 10:20] = 0.0
    out = kernels.ema_scan(z, P, chunk=chunk)
    ref = kernels.ema_scan_reference(z, P)
    assert float((out - ref).abs().max()) <= 1e-12
    # P = 1 restarts the average: position 40 of row 1 is z itself
    assert torch.equal(ref[1, 40], z[1, 40])


def test_ema_on_the_ssd_kernel_is_the_ema():
    """The Triton route's mapping (dt = -log(1-P), x = z/dt, A = -1, B = P, C = 1), run through the torch SSD."""
    g = torch.Generator().manual_seed(3)
    z = torch.randn(2, 70, 48, generator=g, dtype=F64)  # 48 = 3 heads of 16
    P = torch.rand(2, 70, generator=g, dtype=F64).clamp(1e-4, 1 - 1e-4)
    assert kernels.ema_head_dim(48) == 16 and kernels.ema_head_dim(30) == 30
    out = kernels.ema_via_ssd(z, P, backend="torch", chunk=16)
    assert float((out - kernels.ema_scan_reference(z, P)).abs().max()) <= 1e-10


def test_gradients_match_finite_differences():
    x, dt, A, B, C, h0 = (t.requires_grad_() for t in ssd_inputs(1, 6, 2, 2, 1, 2, seed=4))
    assert torch.autograd.gradcheck(lambda *a: kernels.ssd(*a, chunk=4), (x, dt, A, B, C))
    assert torch.autograd.gradcheck(
        lambda *a: kernels.ssd(*a[:5], chunk=4, initial_state=a[5], return_final_state=True), (x, dt, A, B, C, h0)
    )
    g = torch.Generator().manual_seed(5)
    z = torch.randn(2, 7, 3, generator=g, dtype=F64, requires_grad=True)
    P = torch.rand(2, 7, generator=g, dtype=F64) * 0.8 + 0.1
    P[0, 3] = 1.0  # a zero decay: the masked cumprod needs no logarithm
    P.requires_grad_()
    assert torch.autograd.gradcheck(lambda z, P: kernels.ema_scan(z, P, chunk=3), (z, P))
    u = torch.randn(2, 9, 4, generator=g, dtype=F64, requires_grad=True)
    w = torch.randn(4, 3, generator=g, dtype=F64, requires_grad=True)
    bias = torch.randn(4, generator=g, dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(kernels.causal_conv, (u, w, bias))


def test_causal_conv_reads_the_last_k_inputs():
    g = torch.Generator().manual_seed(6)
    x, w, bias = (
        torch.randn(2, 11, 3, generator=g, dtype=F64),
        torch.randn(3, 4, generator=g, dtype=F64),
        torch.randn(3, dtype=F64),
    )
    out = kernels.causal_conv(x, w, bias)
    for t in range(11):
        expected = bias + sum(x[:, t - k] * w[:, 3 - k] for k in range(4) if t - k >= 0)
        assert float((out[:, t] - expected).abs().max()) <= 1e-14, t


def test_low_precision_inputs_are_evaluated_in_fp32():
    """bf16 inputs: the result is the fp32 evaluation of the same (rounded) values, returned in bf16."""
    x, dt, A, B, C, _ = ssd_inputs(1, 20, 2, 4, 1, 3, seed=7, dtype=torch.float32)
    low = [t.to(torch.bfloat16) for t in (x, dt, A, B, C)]
    y = kernels.ssd(*low, chunk=8)
    assert y.dtype == torch.bfloat16
    assert torch.equal(y, kernels.ssd(*(t.float() for t in low), chunk=8).to(torch.bfloat16))
    assert kernels.compute_dtype(torch.bfloat16) == torch.float32 and kernels.compute_dtype(F64) == F64


def test_refusals():
    x, dt, A, B, C, _ = ssd_inputs(1, 5, 2, 2, 1, 2, seed=8)
    with pytest.raises(ValueError, match="backend must be one of"):
        kernels.ssd(x, dt, A, B, C, chunk=4, backend="cuda")
    with pytest.raises(ValueError, match="dt must be"):
        kernels.ssd(x, dt[:, :-1], A, B, C, chunk=4)
    with pytest.raises(ValueError, match="do not split"):
        kernels.ssd(x, dt, A, B.expand(1, 5, 3, 2), C.expand(1, 5, 3, 2), chunk=4)
    with pytest.raises(ValueError, match="z must be"):
        kernels.ema_scan(torch.zeros(2, 3), torch.zeros(2, 3))


def test_the_triton_backend_is_an_error_never_a_fallback():
    """Here (CPU tensors; mamba_ssm absent or present) backend "triton" raises and names the remedy."""
    x, dt, A, B, C, _ = ssd_inputs(1, 5, 2, 2, 1, 2, seed=9, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="backend 'triton'"):
        kernels.ssd(x, dt, A, B, C, chunk=4, backend="triton")
    with pytest.raises(RuntimeError, match="backend 'triton'"):
        kernels.ema_scan(torch.zeros(1, 4, 8), torch.full((1, 4), 0.5), backend="triton")
