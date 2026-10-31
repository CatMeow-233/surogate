"""Ledger topic T24 (NVFP4 Four Over Six quantization), runnable without a GPU.

Upstream's 4/6 kernels (csrc/src/recipes/nvfp4/kernels/quant_nvfp4_4o6_sm100.cu at 282f9901, unchanged at
488e5f00) have two defects; the fork's kernels in fork/native/src/nvfp4_4o6.cu replace them by default:

- SR: the stochastic-rounding kernel (wgrad operand dout^T of every quantized dense matmul in full fine-tuning)
  adds a fixed [-0.25, 0.25) dither before round-to-nearest, which is unbiased only where the E2M1 spacing is 0.5,
  and encodes with the unrounded 6/amax or 4/amax while the GEMM decodes with the rounded E4M3 scale.
- RN: the round-to-nearest kernel (forward operands, wgrad inp^T) has the same encode/decode mismatch, and
  decodes blocks whose scale is below its 1/128 clamp at 1/128 (inflated).

What runs here:

1. A numpy emulation of upstream's kernels and of the fork's, operation by operation in fp32 (the rcp.approx of
   the kernels is modelled as the correctly rounded reciprocal, which it matches to 1 ulp, and the multiply-adds
   nvcc contracts are fused). It measures the bias exactly (expectation over all 256 values of an 8-bit field)
   and over many seeds: upstream's per-value bias reaches 0.25 of the spacing on [2, 4) and 0.38 on [4, 6), its
   tensor-level bias does not average out over seeds, the fork's stays below spacing / 512 and decays as
   1/sqrt(seeds).
2. fork/native/tests/host/nvfp4_4o6_check.cpp: the header the fork kernels call (surogate_fork/nvfp4_4o6_arith.h),
   compiled with the host compiler and checked exhaustively (and against cuda_fp8.h and curand when the CUDA
   headers are found: CUDA_HOME, CUDA_PATH or /usr/local/cuda). Skipped without a C++17 compiler.
3. The header and the emulation compute the same values, request by request.
4. Source pins: the upstream launchers dispatch to the fork kernels behind the switches, the fork kernels call the
   checked arithmetic, upstream's kernels (the legacy path the emulation models) are unedited, and the
   documented switch names are the built ones.

The GPU behaviour (the kernels compute what the emulation computes) is unverified here:
fork/native/tests/test-fork-nvfp4-4o6.cu and fork/docs/upstream-errata.md give the RTX 5090 checks.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[4]
KERNELS = REPO / "csrc/src/recipes/nvfp4/kernels"
NATIVE = REPO / "fork/native"
HOST_CHECK = NATIVE / "tests/host/nvfp4_4o6_check.cpp"

F32 = np.float32
U32 = np.uint64(0xFFFFFFFF)
SR_KEY1 = 0x346F3653  # surogate_fork::nvfp4::kSrKey1

HOST_CHECK_NAMES = [
    "philox_known_answers",
    "sr_bytes_follow_the_documented_layout",
    "sr_byte_marginals_are_uniform",
    "e4m3_values_round_trip",
    "e4m3_rn_matches_an_independent_reference",
    "no_saturation_code_is_the_smallest_code_above_the_slack",
    "clamp_keeps_scales_in_the_finite_e4m3_range",
    "sr_returns_grid_values_and_keeps_grid_values",
    "sr_expectation_is_within_spacing_over_512",
    "sr_bias_averages_to_zero_on_the_half_step_lattice",
    "negative_control_upstream_dither_is_biased",
]
CUDA_HEADER_CHECK_NAMES = ["e4m3_rn_matches_cuda_fp8_on_every_float", "philox_matches_curand"]


# ============================================================================================ emulation
def f32(x):
    return np.asarray(x, dtype=np.float32)


def rcp(x):
    """rcp.approx.ftz.f32, modelled as the correctly rounded reciprocal (the instruction is within 1 ulp)."""
    return (F32(1.0) / f32(x)).astype(np.float32)


def fma(a, b, c):
    """fp32 a*b+c with one rounding (nvcc contracts these): the product of two fp32 values is exact in float64."""
    return (f32(a).astype(np.float64) * f32(b).astype(np.float64) + f32(c).astype(np.float64)).astype(np.float32)


def e2m1_rn(y):
    """cvt.rn.satfinite.e2m1x2.f32: nearest of 0, .5, 1, 1.5, 2, 3, 4, 6, ties to the even code (0, 1, 2, 4)."""
    y = f32(y)
    a = np.abs(y)
    mag = np.select(
        [a <= 0.25, a < 0.75, a <= 1.25, a < 1.75, a <= 2.5, a < 3.5, a <= 5.0], [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0], 6.0
    ).astype(np.float32)
    return np.copysign(mag, y)


def spacing(y):
    a = np.abs(np.asarray(y, dtype=np.float64))
    return np.where(a < 2, 0.5, np.where(a < 4, 1.0, 2.0))


def pow2(e):
    return np.ldexp(F32(1.0), np.asarray(e)).astype(np.float32)


def e4m3_value(code):
    code = np.asarray(code).astype(np.int64)
    e, m = (code >> 3) & 0xF, code & 7
    return np.where(e == 0, m * F32(2.0**-9), (8 + m) * pow2(np.maximum(e, 1) - 10)).astype(np.float32)


def e4m3_code_rn(x):
    """surogate_fork::nvfp4::e4m3_code_rn (= __nv_cvt_float_to_fp8(x, SATFINITE, E4M3) for x >= 0)."""
    x = f32(x)
    out = np.zeros(x.shape, dtype=np.int64)
    pos = x > 0
    xp = x[pos]
    e = ((xp.view(np.uint32) >> 23) & 0xFF).astype(np.int64) - 127
    k_sub = np.rint((xp * F32(512.0)).astype(np.float32)).astype(np.int64)
    k_nor = np.rint((xp * pow2(3 - np.clip(e, -6, 8))).astype(np.float32)).astype(np.int64)
    code = np.where(e < -6, k_sub, ((e + 7) << 3) + (k_nor - 8))
    out[pos] = np.where(xp >= 448.0, 0x7E, np.minimum(code, 0x7E))
    return out


def clamp_block_scale(x, lo=2.0**-9, hi=448.0):
    return np.minimum(np.maximum(f32(x), F32(lo)), F32(hi)).astype(np.float32)


def e4m3_code_no_saturation(x):
    code = e4m3_code_rn(x)
    return np.where(e4m3_value(code) < (f32(x) * F32(1.0 - 2.0**-16)).astype(np.float32), code + 1, code)


def sr_parts(y):
    """|y| clamped to 6 as (t, lo, k, s): t = |y| / s, lo = floor(t), k = round_even(256 (t - lo))."""
    a = np.minimum(np.abs(f32(y)), F32(6.0))
    s = np.where(a < 2, F32(0.5), np.where(a < 4, F32(1.0), F32(2.0))).astype(np.float32)
    t = (a * np.where(a < 2, F32(2.0), np.where(a < 4, F32(1.0), F32(0.5)))).astype(np.float32)
    lo = np.floor(t)
    k = np.rint(((t - lo).astype(np.float32) * F32(256.0)).astype(np.float32))
    return lo, k, s


def e2m1_stochastic(y, byte):
    """surogate_fork::nvfp4::e2m1_stochastic: SRC few-bit stochastic rounding (arXiv:2504.20634)."""
    lo, k, s = sr_parts(y)
    return np.copysign(((lo + (np.asarray(byte) < k)).astype(np.float32) * s).astype(np.float32), f32(y))


def e2m1_stochastic_expectation(y):
    lo, k, s = sr_parts(y)
    return np.copysign((lo + k / 256.0) * s, f32(y).astype(np.float64))


def upstream_noise(byte):
    """(float(b) / 256 - 0.5) * 0.5 (quantize_block_4o6_stochastic), exact in fp32."""
    return ((f32(byte) / F32(256.0) - F32(0.5)) * F32(0.5)).astype(np.float32)


def bytes_of_words(words):
    """Element i of a block takes byte i % 4 of word i / 4 (both kernels)."""
    shifts = np.uint64(8) * (np.arange(16, dtype=np.uint64) % np.uint64(4))
    return ((words[:, np.arange(16) // 4] >> shifts) & np.uint64(0xFF)).astype(np.int64)


def lcg_bytes(seed, nblocks):
    """Upstream: state = seed ^ (block * 1103515245 + 12345), then four steps state = state * 1103515245 + 12345."""
    a, c = np.uint64(1103515245), np.uint64(12345)
    state = (np.uint64(seed) ^ ((np.arange(nblocks, dtype=np.uint64) * a + c) & U32)) & U32
    words = []
    for _ in range(4):
        state = (state * a + c) & U32
        words.append(state)
    return bytes_of_words(np.stack(words, axis=1))


def philox4x32_10(c0, c1, c2, c3, k0, k1):
    c = list(np.broadcast_arrays(*[np.asarray(v, dtype=np.uint64) & U32 for v in (c0, c1, c2, c3)]))
    k0, k1 = np.uint64(k0) & U32, np.uint64(k1) & U32
    for _ in range(10):
        p0, p1 = np.uint64(0xD2511F53) * c[0], np.uint64(0xCD9E8D57) * c[2]
        c = [(p1 >> np.uint64(32)) ^ c[1] ^ k0, p1 & U32, (p0 >> np.uint64(32)) ^ c[3] ^ k1, p0 & U32]
        k0, k1 = (k0 + np.uint64(0x9E3779B9)) & U32, (k1 + np.uint64(0xBB67AE85)) & U32
    return np.stack(c, axis=-1)


def philox_bytes(seed, nblocks, generation=0):
    """surogate_fork::nvfp4::sr_block_random: counter (block, generation, 0, 0), key (seed, kSrKey1)."""
    return bytes_of_words(philox4x32_10(np.arange(nblocks, dtype=np.uint64), generation, 0, 0, seed, SR_KEY1))


def global_encode_scale(x):
    g = F32(np.max(np.abs(x)))
    return g, (F32(F32(1536.0) * rcp(max(g, F32(1e-12)))) if g > 0 else F32(1.0))


def select_four(vals, enc6, enc4, recon6, recon4):
    """The MSE 4-vs-6 choice of both upstream kernels and the fork's: per candidate, RN codes of vals * enc and the
    error sum((dq * recon - v)^2), each multiply-add contracted, two groups of 8 summed."""
    err = []
    for enc, recon in ((enc6, recon6), (enc4, recon4)):
        total = np.zeros(vals.shape[0], np.float32)
        for g in range(2):
            part = np.zeros(vals.shape[0], np.float32)
            for i in range(8):
                v = vals[:, g * 8 + i]
                diff = fma(e2m1_rn((v * enc).astype(np.float32)), recon, -v)
                part = fma(diff, diff, part)
            total = (total + part).astype(np.float32)
        err.append(total)
    return err[1] < err[0]


def unrounded_choice(vals):
    """Upstream's scales (6/amax, 4/amax, amax/6, amax/4) and its choice; returns amax, use4, recon of the choice."""
    amax = np.max(np.abs(vals), axis=1).astype(np.float32)
    inv = rcp(np.maximum(amax, F32(1e-12)))
    recon6, recon4 = (amax * rcp(F32(6.0))).astype(np.float32), (amax * rcp(F32(4.0))).astype(np.float32)
    use4 = select_four(vals, (F32(6.0) * inv).astype(np.float32), (F32(4.0) * inv).astype(np.float32), recon6, recon4)
    return amax, use4, np.where(use4, recon4, recon6), np.where(use4, F32(4.0) * inv, F32(6.0) * inv).astype(np.float32)


def upstream_scale_code(recon, genc):
    """Upstream: __nv_cvt_float_to_fp8(min(max(recon * genc, 1/128), 480), SATFINITE, E4M3)."""
    return e4m3_code_rn(np.minimum(np.maximum((recon * genc).astype(np.float32), F32(1.0 / 128.0)), F32(480.0)))


def blocks(x):
    return f32(x).reshape(-1, 16)


def finish(x, amax, q, code, g):
    zero = amax == 0
    q = np.where(zero[:, None], 0.0, q)
    code = np.where(zero, 0, code)
    return q.reshape(x.shape), code.reshape(x.shape[0], -1), g


def sr_upstream(x, seed):
    """quantize_4o6_stochastic_cutlass_kernel<MSE>: (E2M1 values [M, K], E4M3 codes [M, K/16], global amax)."""
    vals = blocks(x)
    g, genc = global_encode_scale(x)
    amax, use4, recon, enc = unrounded_choice(vals)
    q = e2m1_rn(fma(vals, enc[:, None], upstream_noise(lcg_bytes(seed, vals.shape[0]))))
    return finish(x, amax, q, upstream_scale_code(recon, genc), g)


def sr_fork_scale(vals, genc, stored_scale=True):
    amax, use4, recon, enc_unrounded = unrounded_choice(vals)
    if not stored_scale:  # ablation: fork rounding, upstream scale
        return amax, upstream_scale_code(recon, genc), enc_unrounded
    code = e4m3_code_no_saturation(clamp_block_scale((recon * genc).astype(np.float32)))
    return amax, code, (genc / e4m3_value(code)).astype(np.float32)


def sr_fork(x, seed, generation=0):
    """surogate_fork::nvfp4 quantize_4o6_sr_kernel<MSE>."""
    vals = blocks(x)
    g, genc = global_encode_scale(x)
    amax, code, enc = sr_fork_scale(vals, genc)
    q = e2m1_stochastic((vals * enc[:, None]).astype(np.float32), philox_bytes(seed, vals.shape[0], generation))
    return finish(x, amax, q, code, g)


def dequantize(q, code, g):
    """What the CUTLASS GEMM multiplies: code value * block scale * amax / 1536 (alpha = amax_a amax_b / 1536^2)."""
    return q * np.repeat(e4m3_value(code).astype(np.float64), 16, axis=1) * (float(g) / 1536.0)


def sr_upstream_expectation(x):
    """E over the 256 dither values (the LCG byte of an element is uniform over seeds)."""
    vals = blocks(x)
    g, genc = global_encode_scale(x)
    amax, use4, recon, enc = unrounded_choice(vals)
    eq = sum(e2m1_rn(fma(vals, enc[:, None], upstream_noise(np.full(vals.shape, b)))) for b in range(256)) / 256.0
    q, code, g = finish(x, amax, eq, upstream_scale_code(recon, genc), g)
    return dequantize(q, code, g)


def sr_fork_expectation(x, stored_scale=True):
    vals = blocks(x)
    g, genc = global_encode_scale(x)
    amax, code, enc = sr_fork_scale(vals, genc, stored_scale)
    q, code, g = finish(x, amax, e2m1_stochastic_expectation((vals * enc[:, None]).astype(np.float32)), code, g)
    return dequantize(q, code, g)


def rn_upstream(x):
    vals = blocks(x)
    g, genc = global_encode_scale(x)
    amax, use4, recon, enc = unrounded_choice(vals)
    return finish(x, amax, e2m1_rn((vals * enc[:, None]).astype(np.float32)), upstream_scale_code(recon, genc), g)


def rn_fork(x):
    """quantize_4o6_rn_kernel<MSE>: each candidate encoded with, and judged on, its stored scale."""
    vals = blocks(x)
    g, genc = global_encode_scale(x)
    amax = np.max(np.abs(vals), axis=1).astype(np.float32)
    cand = []
    for c in (6.0, 4.0):
        code = e4m3_code_rn(clamp_block_scale(((amax * rcp(F32(c))).astype(np.float32) * genc).astype(np.float32)))
        sf = e4m3_value(code)
        cand.append((code, (genc / sf).astype(np.float32), (sf / genc).astype(np.float32)))
    use4 = select_four(vals, cand[0][1], cand[1][1], cand[0][2], cand[1][2])
    enc = np.where(use4, cand[1][1], cand[0][1])
    q = e2m1_rn((vals * enc[:, None]).astype(np.float32))
    return finish(x, amax, q, np.where(use4, cand[1][0], cand[0][0]), g)


def to_bf16(x):
    b = f32(x).view(np.uint32).astype(np.uint64)
    b = ((b + ((b >> np.uint64(16)) & np.uint64(1)) + np.uint64(0x7FFF)) >> np.uint64(16)) << np.uint64(16)
    return b.astype(np.uint32).view(np.float32)


def dout_like(seed, rows=32, cols=4096, kind="gauss"):
    """A dout^T-shaped BF16 tensor: rows = output channels with log-normal scales, columns = tokens."""
    rng = np.random.default_rng(seed)
    x = rng.standard_t(3, (rows, cols)) if kind == "t3" else rng.standard_normal((rows, cols))
    if kind == "outliers":  # 0.2% of the tokens 3e4 x larger: most blocks sit below upstream's 1/128 clamp
        x[:, rng.random(cols) < 0.002] *= 3e4
    return to_bf16((x * np.exp(rng.normal(0.0, 1.0, size=(rows, 1)))).astype(np.float32))


def relative(a, x):
    return float(np.linalg.norm(a - x) / np.linalg.norm(x))


# ===================================================================================== scalar rounding
BYTES = np.arange(256)


def upstream_expectation(y):
    y = f32(y)
    return np.mean([e2m1_rn((y + upstream_noise(b)).astype(np.float32)) for b in BYTES], axis=0)


def fork_expectation_by_enumeration(y):
    y = f32(y)
    return np.mean([e2m1_stochastic(y, b) for b in BYTES], axis=0)


def test_upstream_dither_is_biased_where_the_spacing_exceeds_one_half():
    e = upstream_expectation([2.4, 2.2, 4.6, 2.8, 5.4, 0.5, 1.5])
    assert e[0] == 2.0 + 76 / 256  # P(up) = 0.297 instead of 0.4
    assert e[1] == 2.0 and e[2] == 4.0  # never round up (should with p = 0.2 and 0.3)
    assert e[3] == 3.0 and e[4] == 6.0  # always round up
    assert e[5] == 0.5 - 0.5 / 256 and e[6] == 1.5 - 0.5 / 256  # grid values move with p = 1/256 (RNE ties)
    y = np.linspace(0.0, 6.0, 60001)[1:-1].astype(np.float32)
    bias = upstream_expectation(y) - y
    for lo, hi, worst, mean_abs in ((2, 4, 0.25, 0.125), (4, 6, 0.75, 0.375)):
        band = (y >= lo) & (y < hi)
        assert np.abs(bias[band]).max() >= worst  # 25% and 38% of the spacing
        assert abs(np.abs(bias[band]).mean() - mean_abs) < 0.01
    low = y < 2
    assert abs(bias[low].mean() + 1 / 1024) < 1e-4  # [0, 2): the dither has mean -1/1024 (SRFF)


def test_fork_rounding_is_unbiased_on_the_whole_grid():
    y = np.concatenate([np.linspace(-6.0, 6.0, 120001), [2.0, 4.0, 6.0, -2.0, -4.0, np.nextafter(F32(2), F32(0))]])
    y = y.astype(np.float32)
    exact = e2m1_stochastic_expectation(y)
    assert np.array_equal(exact, fork_expectation_by_enumeration(y))  # P(up) = k / 256
    assert np.abs((exact - y) / spacing(y)).max() <= 1 / 512
    ties = ((np.arange(6 * 512) + 0.5) / 512).astype(np.float32)  # fractions exactly on the 8-bit RTNE ties
    assert abs((e2m1_stochastic_expectation(ties) - ties).mean()) < 1e-6
    grid = np.array([0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
    for b in BYTES:
        assert np.array_equal(e2m1_stochastic(grid, b), grid) and np.array_equal(e2m1_stochastic(-grid, b), -grid)


def test_naive_spacing_scaled_dither_with_round_to_nearest_stays_biased():
    """Why the fork floors: widening the dither to +-spacing/2 and keeping RN is still biased at 2 and 4."""
    y = np.linspace(2.0, 6.0, 4001)[:-1].astype(np.float32)
    s = spacing(y).astype(np.float32)
    e = np.mean([e2m1_rn((y + (F32(b) / 256 - F32(0.5)) * s).astype(np.float32)) for b in BYTES], axis=0)
    assert np.abs(e - y).max() > 0.1


# ===================================================================================== tensor level
@pytest.mark.parametrize("kind", ["gauss", "t3", "outliers"])
def test_tensor_expectation(kind):
    """Exact expectation over the SR field of a 32 x 4096 dout^T-like tensor, through the whole kernel."""
    x = dout_like(11, kind=kind).astype(np.float64)
    upstream = relative(sr_upstream_expectation(x), x)
    fork = relative(sr_fork_expectation(x), x)
    assert upstream > 2e-2  # ~5e-2 Gaussian, ~4e-2 t(3), ~3e-2 outliers
    assert fork < 1e-3  # ~4e-4: the 1/512 quantization of the probability
    if kind == "gauss":
        # The rounding fix alone leaves the encode/decode scale mismatch (stored/exact scale RMS 2.6%).
        assert relative(sr_fork_expectation(x, stored_scale=False), x) > 2e-2


def test_seed_average_converges_for_the_fork_and_plateaus_upstream():
    x = dout_like(12, rows=16, cols=2048).astype(np.float64)
    errors = {}
    for name, kernel in (("upstream", sr_upstream), ("fork", sr_fork)):
        total, out = np.zeros_like(x), []
        for n in range(64):
            total += dequantize(*kernel(x, 1000 + 7919 * n))
            if n + 1 in (4, 16, 64):
                out.append(relative(total / (n + 1), x))
        errors[name] = out
    up, fk = errors["upstream"], errors["fork"]
    assert up[2] > 4e-2 and up[1] / up[2] < 1.3  # bias: flat
    assert fk[0] / fk[1] > 1.6 and fk[1] / fk[2] > 1.6  # noise: halves per 4x seeds
    assert fk[2] < 0.03


def test_round_to_nearest_with_the_stored_scale():
    for kind, lower in (("gauss", 0.97), ("t3", 0.985), ("outliers", 0.95)):
        x = dout_like(14, kind=kind).astype(np.float64)
        upstream, fork = relative(dequantize(*rn_upstream(x)), x), relative(dequantize(*rn_fork(x)), x)
        assert fork < upstream * lower, (kind, upstream, fork)  # Gaussian 9.0e-2 -> 8.7e-2
    # Blocks whose decode scale is below 1/128: upstream decodes them inflated, the fork at their own scale.
    x = dout_like(15, rows=16, kind="outliers")
    vals = blocks(x).astype(np.float64)
    g, _ = global_encode_scale(x)
    small = (np.abs(vals).max(axis=1) * 256.0 / float(g) < 1 / 128) & (np.abs(vals).max(axis=1) > 0)
    assert small.mean() > 0.5
    for kernel, low, high in ((rn_upstream, 1.5, None), (rn_fork, 0.97, 1.03)):
        deq = dequantize(*kernel(x)).reshape(-1, 16)[small]
        ratio = float((vals[small] * deq).sum() / (vals[small] ** 2).sum())
        assert ratio > low and (high is None or ratio < high), (kernel.__name__, ratio)


# ===================================================================================== host program
def _compiler():
    for name in (os.environ.get("CXX"), "c++", "g++", "clang++"):
        if name and shutil.which(name):
            return shutil.which(name)
    return None


def _cuda_include():
    for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if root and (Path(root) / "include/cuda_fp8.h").is_file() and (Path(root) / "include/curand_philox4x32_x.h").is_file():
            return Path(root) / "include"
    return None


@pytest.fixture(scope="module")
def host_binary(tmp_path_factory):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("the T24 host checks need a C++17 compiler (CXX, c++, g++ or clang++)")
    out = tmp_path_factory.mktemp("nvfp4_4o6") / "nvfp4_4o6_check"
    flags = ["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-I", str(NATIVE / "include")]
    cuda = _cuda_include()
    if cuda is not None:
        flags += ["-DFORK_CUDA_HEADERS", "-isystem", str(cuda)]
    result = subprocess.run([compiler, *flags, str(HOST_CHECK), "-o", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-4000:]
    return out, cuda is not None


@pytest.fixture(scope="module")
def host_checks(host_binary):
    binary, _ = host_binary
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=110)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return {line.split()[1].rstrip(":"): line for line in lines}, result.returncode


def test_the_host_program_runs_exactly_the_listed_checks(host_checks, host_binary):
    checks, _ = host_checks
    expected = HOST_CHECK_NAMES + (CUDA_HEADER_CHECK_NAMES if host_binary[1] else [])
    assert list(checks) == expected


@pytest.mark.parametrize("name", HOST_CHECK_NAMES + CUDA_HEADER_CHECK_NAMES)
def test_host_check(host_checks, host_binary, name):
    if name in CUDA_HEADER_CHECK_NAMES and not host_binary[1]:
        pytest.skip("no CUDA headers (CUDA_HOME, CUDA_PATH or /usr/local/cuda with cuda_fp8.h and curand)")
    checks, _ = host_checks
    assert checks[name].startswith("ok "), checks[name]


def test_the_header_computes_what_the_emulation_computes(host_binary, tmp_path):
    binary, _ = host_binary
    rng = np.random.default_rng(16)
    scales = np.concatenate([np.exp2(rng.uniform(-12, 9.5, 4000)), e4m3_value(np.arange(127)), [384.00006, 256.00003]])
    scales = scales.astype(np.float32)
    ys = np.concatenate([rng.uniform(-6.5, 6.5, 6000), np.arange(-6 * 512, 6 * 512) / 512.0]).astype(np.float32)
    bs = rng.integers(0, 256, ys.size)
    counters = rng.integers(0, 2**32, (300, 6), dtype=np.uint64)
    lines = [f"e4m3 {float(v).hex()}" for v in scales] + [f"nosat {float(v).hex()}" for v in scales]
    lines += [f"sr {float(y).hex()} {b}" for y, b in zip(ys, bs)]
    lines += ["philox " + " ".join(str(int(w)) for w in row) for row in counters]
    requests = tmp_path / "requests.txt"
    requests.write_text("\n".join(lines) + "\n")
    result = subprocess.run([str(binary), "dump", str(requests)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    out = result.stdout.splitlines()
    n = scales.size
    assert [int(v) for v in out[:n]] == e4m3_code_rn(scales).tolist()
    assert [int(v) for v in out[n : 2 * n]] == e4m3_code_no_saturation(clamp_block_scale(scales)).tolist()
    sr = np.array([float.fromhex(v) for v in out[2 * n : 2 * n + ys.size]])
    assert np.array_equal(sr, e2m1_stochastic(ys, bs).astype(np.float64))
    words = np.array([[int(w) for w in v.split()] for v in out[2 * n + ys.size :]], dtype=np.uint64)
    want = np.stack([philox4x32_10(*[np.uint64(w) for w in row]) for row in counters])
    assert np.array_equal(words, want)


# ===================================================================================== source pins
def body(path: Path, signature: str) -> str:
    """The brace-balanced body of the first definition matching `signature` (a regex), whitespace collapsed."""
    text = path.read_text(encoding="utf-8")
    match = re.search(signature + r"[^;{]*\{", text)
    assert match, f"{path.relative_to(REPO)}: no definition matching {signature!r}"
    depth, start = 1, match.end()
    for end in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            return " ".join(text[start:end].split())
    raise AssertionError(f"{path.relative_to(REPO)}: unbalanced braces after {signature!r}")


def collapse(text: str) -> str:
    return " ".join(text.split())


RN_DISPATCH = collapse(
    """if (surogate_fork::nvfp4::four_over_six_rn_fixed()) {
        surogate_fork::nvfp4::quantize_4o6(out_fp4, block_scales, global_amax, in, M, K, error_metric, stream);
        CUDA_CHECK(cudaGetLastError()); return; }"""
)
SR_DISPATCH = collapse(
    """if (surogate_fork::nvfp4::four_over_six_sr_fixed()) {
        surogate_fork::nvfp4::quantize_4o6_stochastic( out_fp4, block_scales, global_amax, in, M, K, error_metric,
        seed, stream); CUDA_CHECK(cudaGetLastError()); return; }"""
)


@pytest.mark.parametrize(
    "launcher, dispatch, computes_amax",
    [
        ("quantize_nvfp4_4o6_cutlass_auto_scale", RN_DISPATCH, True),
        ("quantize_nvfp4_4o6_cutlass_from_amax", RN_DISPATCH, False),
        ("quantize_nvfp4_4o6_stochastic_cutlass_auto_scale", SR_DISPATCH, True),
    ],
)
def test_the_upstream_launchers_dispatch_to_the_fork_kernels(launcher, dispatch, computes_amax):
    code = body(KERNELS / "quant_nvfp4_4o6.cu", rf"void {launcher}\(")
    assert code.count(dispatch) == 1
    assert code.index("is_supported()") < code.index(dispatch) < code.index("switch (error_metric)")
    if computes_amax:
        assert code.index("abs_max(global_amax, in, (long)M * K, dp, stream);") < code.index(dispatch)
    source = (KERNELS / "quant_nvfp4_4o6.cu").read_text(encoding="utf-8")
    assert '#include "surogate_fork/nvfp4_4o6.h"' in source


def test_every_4o6_entry_point_goes_through_a_dispatching_launcher():
    """The tensor wrappers call the pointer launchers; no other upstream code calls the sm100 kernels directly."""
    for wrapper, launcher in (
        ("quantize_nvfp4_4o6_cutlass", "quantize_nvfp4_4o6_cutlass_auto_scale("),
        ("quantize_nvfp4_4o6_stochastic_cutlass", "quantize_nvfp4_4o6_stochastic_cutlass_auto_scale("),
    ):
        assert launcher in body(KERNELS / "quant_nvfp4_4o6.cu", rf"void {wrapper}\(Tensor& out_fp4")
    callers = [
        str(path.relative_to(REPO))
        for root in (REPO / "csrc/src", NATIVE)
        for path in root.rglob("*")
        if path.suffix in {".cu", ".cuh", ".cpp", ".h"} and "nvfp4_4o6_sm100::quantize_4o6" in path.read_text(errors="replace")
    ]
    assert callers == ["csrc/src/recipes/nvfp4/kernels/quant_nvfp4_4o6.cu"]


def test_the_fork_kernels_run_the_checked_arithmetic():
    src = NATIVE / "src/nvfp4_4o6.cu"
    sr = body(src, r"void\s+quantize_block_sr\(")
    assert "select_four<M>(vals, 6.0f * inv, 4.0f * inv, recon6, recon4, codes6, codes4)" in sr
    assert "scale_code = e4m3_code_no_saturation(clamp_block_scale((four ? recon4 : recon6) * global_encode));" in sr
    assert "const float encode = global_encode / e4m3_value(scale_code);" in sr
    assert "q[i] = e2m1_stochastic(vals[g * 8 + i] * encode, sr_byte(random, g * 8 + i));" in sr
    kernel = body(src, r"__global__ void quantize_4o6_sr_kernel\(")
    assert "sr_block_random(static_cast<uint32_t>(block), 0u, seed)" in kernel
    rn = body(src, r"void\s+quantize_block_rn\(")
    for c in ("6", "4"):
        assert f"code{c} = e4m3_code_rn(clamp_block_scale(amax * rcp_approx_ftz({c}.0f) * global_encode));" in rn
    assert "vals, global_encode / sf6, global_encode / sf4, sf6 / global_encode, sf4 / global_encode" in rn
    text = src.read_text(encoding="utf-8")
    assert "1103515245" not in text and "/ 256.0f - 0.5f" not in text  # no LCG, no fixed dither


def test_upstream_kernels_are_what_the_emulation_models():
    """The legacy path (the *_LEGACY switches) is upstream's kernel file, unedited; these are the lines modelled."""
    text = collapse((KERNELS / "quant_nvfp4_4o6_sm100.cu").read_text(encoding="utf-8"))
    for line in (
        "float noise = (float(rand_bits) / 256.0f - 0.5f) * 0.5f; group[i] = scaled + noise;",
        "uint32_t rand_bits = (random[g * 2 + (i / 4)] >> ((i % 4) * 8)) & 0xFF;",
        "uint32_t state = seed ^ (block_idx * 1103515245 + 12345);",
        "state = state * 1103515245 + 12345; random[i] = state;",
        "decode_scale = fminf(fmaxf(decode_scale, 1.0f / 128.0f), 480.0f);",
        "float encode_scale_6 = 6.0f * rcp_approx_ftz(fmaxf(block_amax, 1e-12f));",
        "float recon_scale_4 = block_amax * rcp_approx_ftz(4.0f);",
        "bool use_scale_4 = (err_4 < err_6);",
        "float decode_scale = selected_recon_scale * global_encode_scale;",
    ):
        assert line in text, line
    assert text.count("decode_scale = fminf(fmaxf(decode_scale, 1.0f / 128.0f), 480.0f);") == 2  # RN and SR



def test_the_fork_source_and_gpu_test_are_built():
    cmake = (NATIVE / "cmake/fork_native.cmake").read_text(encoding="utf-8")
    configure = re.search(r"function\(surogate_fork_configure_common target\)(.*?)endfunction\(\)", cmake, re.S)
    assert configure and 'target_sources(${target} PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/src/nvfp4_4o6.cu")' in configure.group(1)
    attach = re.search(r"function\(_surogate_fork_attach_tests\)(.*?)endfunction\(\)", cmake, re.S)
    assert attach and '"${SUROGATE_FORK_NATIVE_DIR}/tests/test-fork-nvfp4-4o6.cu"' in attach.group(1)
    gpu = (NATIVE / "tests/test-fork-nvfp4-4o6.cu").read_text(encoding="utf-8")
    for call in (
        "quantize_nvfp4_4o6_stochastic_cutlass_auto_scale(",
        "quantize_nvfp4_4o6_cutlass_auto_scale(",
    ):
        assert call in gpu, call


def test_the_documented_switches_are_the_built_ones():
    built = set()
    for path in (NATIVE / "src/nvfp4_4o6.cu",):
        built |= set(re.findall(r'"(SUROGATE_NVFP4_[A-Z0-9_]+)"', path.read_text(encoding="utf-8")))
    assert built == {"SUROGATE_NVFP4_4O6_SR_LEGACY", "SUROGATE_NVFP4_4O6_RN_LEGACY"}
    ledger = (REPO / "fork/patches/ledger.toml").read_text(encoding="utf-8")
    errata = (REPO / "fork/docs/upstream-errata.md").read_text(encoding="utf-8")
    for name in built:
        assert f"{name}=1" in ledger and f"{name}=1" in errata, name
