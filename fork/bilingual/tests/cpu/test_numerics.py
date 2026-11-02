"""Operator numerics (docs/spec.md, I2-A6; MA-13, IMPL-10): RoPE against a complex-exponential reference with its
tables in the promoted dtype, RMSNorm against the closed form, and the two GQA paths against each other.
"""

import math

import pytest
import torch

from surogate_bilingual.config import profile
from surogate_bilingual.ops.attention import CausalSelfAttention, sdpa_supports_gqa
from surogate_bilingual.ops.norm import RMSNorm
from surogate_bilingual.ops.rope import RopeCache, apply_rope


def complex_rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    """(x_{2i} + i x_{2i+1}) * exp(i * pos * theta^(-2i/hd)), computed in complex128 from Python floats."""
    B, H, T, hd = x.shape
    xc = torch.complex(x[..., ::2].double(), x[..., 1::2].double())
    angles = torch.tensor(
        [[pos * theta ** (-2 * i / hd) for i in range(hd // 2)] for pos in range(T)], dtype=torch.float64
    )
    rotated = xc * torch.polar(torch.ones_like(angles), angles)
    return torch.stack((rotated.real, rotated.imag), dim=-1).flatten(-2)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-6), (torch.float64, 1e-12)])
def test_i2_a6_rope_matches_the_complex_reference(dtype, tol):
    x = torch.randn(2, 4, 128, 16, generator=torch.Generator().manual_seed(0), dtype=dtype)
    cache = RopeCache(16, 10000.0)
    cos, sin = cache.get(128, x.device, dtype)
    assert cos.dtype == sin.dtype == dtype  # tables in the promoted dtype, not float32 (MA-13)
    out = apply_rope(x, cos, sin)
    assert out.dtype == dtype
    assert float((out.double() - complex_rope(x, 10000.0)).abs().max()) <= tol


def test_i2_a6_rope_tables_are_promoted_and_cached():
    cache = RopeCache(8, 10000.0)
    cos, _ = cache.get(5, torch.device("cpu"), torch.bfloat16)
    assert cos.dtype == torch.float32  # bf16 activations rotate in fp32
    assert cache.get(5, torch.device("cpu"), torch.bfloat16)[0] is cos  # cached per (T, device, dtype)
    assert cache.get(6, torch.device("cpu"), torch.bfloat16)[0] is not cos
    x = torch.randn(1, 1, 5, 8, dtype=torch.bfloat16)
    assert apply_rope(x, *cache.get(5, x.device, x.dtype)).dtype == torch.bfloat16


def test_i2_a6_rmsnorm_matches_the_closed_form_in_fp64():
    norm = RMSNorm(64, 1e-6).double()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(64, dtype=torch.float64))
    x = torch.randn(3, 7, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(1)) * 3
    expected = torch.empty_like(x)
    for idx in torch.cartesian_prod(torch.arange(3), torch.arange(7)).tolist():
        row = x[idx[0], idx[1]].tolist()
        scale = 1.0 / math.sqrt(sum(v * v for v in row) / len(row) + 1e-6)
        expected[idx[0], idx[1]] = torch.tensor(row, dtype=torch.float64) * scale * norm.weight.detach()
    with torch.no_grad():
        assert float((norm(x) - expected).abs().max()) <= 1e-14


def test_i2_a6_gqa_paths_agree_in_fp32():
    if not sdpa_supports_gqa():
        pytest.skip(f"torch {torch.__version__} has no enable_gqa in scaled_dot_product_attention")
    cfg = profile("smoke", "swiglu")
    assert cfg.n_kv_heads < cfg.n_heads
    gqa = CausalSelfAttention(cfg, "sdpa_gqa")
    repeat = CausalSelfAttention(cfg, "repeat")
    repeat.load_state_dict(gqa.state_dict())
    x = torch.randn(2, 128, cfg.d_model, generator=torch.Generator().manual_seed(2))
    rope = RopeCache(cfg.head_dim, cfg.rope_theta)
    with torch.no_grad():
        assert float((gqa(x, rope) - repeat(x, rope)).abs().max()) <= 1e-6
