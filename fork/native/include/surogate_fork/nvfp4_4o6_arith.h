// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0

// Scalar arithmetic of the fork's NVFP4 Four Over Six kernels (fork/native/src/nvfp4_4o6.cu, ledger
// T24): the E4M3 block-scale codes, the stochastic rounding onto the E2M1 grid and the Philox4x32-10 generator.
// Everything here is exact or integer arithmetic with one rounding per operation and no a*b+c pattern that nvcc
// could contract, so the host build (fork/native/tests/host/nvfp4_4o6_check.cpp, run by
// fork/patches/tests/native/test_nvfp4_4o6.py) computes bit for bit what the kernels compute. The header needs
// no CUDA header on the host.
//
// Formats. E2M1 (the FP4 element) holds 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives: the spacing is 0.5 on
// [0, 2), 1 on [2, 4) and 2 on [4, 6]. E4M3 (the per-16-value block scale, OCP FP8) holds 0, the subnormals
// m * 2^-9 (m = 1..7) and the normals (8 + m) * 2^(e - 10) (e = 1..15, m = 0..7, 0x7F excluded), up to 448.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define SUROGATE_FORK_HD __host__ __device__ __forceinline__
#else
#define SUROGATE_FORK_HD inline
#endif

namespace surogate_fork::nvfp4 {

/// E4M3 code of 448, the largest finite value (0x7F is NaN).
constexpr uint8_t kE4M3MaxCode = 0x7E;
/// Smallest positive E4M3 value, 2^-9. The fork kernels clamp block decode scales to [2^-9, 448]; upstream
/// clamped to [1/128, 480], which decoded blocks whose scale was below 1/128 at 1/128 (inflated).
constexpr float kE4M3Min = 0.001953125f;
constexpr float kE4M3Max = 448.0f;
/// A stored SR block scale may be this much smaller than the exact decode scale (relative 2^-16) before the
/// next code up is used: absorbs the few-ulp fp32 error of the exact scale (a block holding the global amax
/// computes 384 or 256 up to rounding), clips the block amax by at most 6 * 2^-16 in scaled units.
constexpr float kNoSaturationSlack = 0.9999847412109375f;  // 1 - 2^-16, exact in fp32

SUROGATE_FORK_HD uint32_t float_bits(float x) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(x);
#else
    uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
#endif
}

SUROGATE_FORK_HD float bits_float(uint32_t u) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(u);
#else
    float x;
    std::memcpy(&x, &u, sizeof x);
    return x;
#endif
}

/// 2^e for e in [-126, 127], exact.
SUROGATE_FORK_HD float pow2(int e) {
    return bits_float(static_cast<uint32_t>(127 + e) << 23);
}

/// Round half to even (the default rounding mode on the host; cvt.rni on the device).
SUROGATE_FORK_HD float round_even(float x) {
#if defined(__CUDA_ARCH__)
    return rintf(x);
#else
    return std::rint(x);
#endif
}

/// Value of an E4M3 code; the sign bit is ignored (block scales are non-negative).
SUROGATE_FORK_HD float e4m3_value(uint8_t code) {
    const int e = (code >> 3) & 0xF;
    const int m = code & 7;
    if (e == 0) return static_cast<float>(m) * 0.001953125f;  // m * 2^-9
    return static_cast<float>(8 + m) * pow2(e - 10);
}

/// Round-to-nearest-even E4M3 code of x >= 0 (or +inf), saturating to 448: for these inputs, what
/// __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3) returns (the host check compares every float in
/// [2^-11, 512] when the CUDA headers are present). Not for NaN or negative inputs: the callers clamp first.
SUROGATE_FORK_HD uint8_t e4m3_code_rn(float x) {
    if (!(x > 0.0f)) return 0;
    if (x >= kE4M3Max) return kE4M3MaxCode;
    const int e = static_cast<int>((float_bits(x) >> 23) & 0xFF) - 127;  // floor(log2 x) for normal x
    if (e < -6) {
        // Subnormal E4M3 range: spacing 2^-9; x * 512 is exact. 8 (x rounds up to 2^-6) is also the code of 2^-6.
        return static_cast<uint8_t>(round_even(x * 512.0f));
    }
    const int k = static_cast<int>(round_even(x * pow2(3 - e)));  // x / 2^(e-3) in [8, 16), exact; k in [8, 16]
    const int code = ((e + 7) << 3) + (k - 8);                    // k == 16 carries into the next exponent
    return static_cast<uint8_t>(code > kE4M3MaxCode ? kE4M3MaxCode : code);
}

/// Block decode scale clamped to the finite positive E4M3 range [2^-9, 448]; NaN maps to 2^-9.
SUROGATE_FORK_HD float clamp_block_scale(float x) {
    return fminf(fmaxf(x, kE4M3Min), kE4M3Max);
}

/// Stored scale for a stochastically rounded block: the smallest E4M3 code whose value is at least
/// x * (1 - 2^-16), for x in [2^-9, 448]. Encoding with it never pushes a block value above 6 by more than the
/// slack, so SR never saturates (a saturated value has expectation 6, not its own value). 256 and 384, the
/// scales of a block holding the global amax, stay exact.
SUROGATE_FORK_HD uint8_t e4m3_code_no_saturation(float x) {
    uint8_t code = e4m3_code_rn(x);
    if (e4m3_value(code) < x * kNoSaturationSlack) ++code;  // RN went down by more than the slack: next code up
    return code;
}

/// Stochastic rounding of a scaled value y onto the E2M1 grid, sign kept, with the 8-bit field `byte`
/// (uniform on 0..255). |y| is clamped to 6 first. Below that, with s the grid spacing at |y| and
/// t = |y| / s = lo + f (lo an integer, 0 <= f < 1), the result is (lo + 1) * s if byte < k and lo * s
/// otherwise, where k = round_even(256 f): the "SRC" few-bit rounding of arXiv:2504.20634, whose
/// probability k/256 is within 1/512 of f, so |E[q] - y| <= s/512 for every y, the error averages to zero
/// over f, grid values are exact (k = 0) and k = 256 always rounds up. Upstream's kernel added a fixed
/// [-0.25, 0.25) dither and rounded to nearest, which is unbiased only where the spacing is 0.5.
/// The returned value is exactly representable in E2M1, so cvt.rn.satfinite.e2m1x2.f32 encodes it exactly.
SUROGATE_FORK_HD float e2m1_stochastic(float y, uint32_t byte) {
    const float a = fminf(fabsf(y), 6.0f);  // NaN -> 6 (fminf returns the number)
    const float s = a < 2.0f ? 0.5f : (a < 4.0f ? 1.0f : 2.0f);
    const float t = a * (a < 2.0f ? 2.0f : (a < 4.0f ? 1.0f : 0.5f));  // a / s, exact
    const float lo = floorf(t);
    const int k = static_cast<int>(round_even((t - lo) * 256.0f));  // t - lo and * 256 are exact
    const float q = (lo + (static_cast<int>(byte) < k ? 1.0f : 0.0f)) * s;
    return copysignf(q, y);
}

/// Philox4x32-10 (Salmon et al., SC'11; the generator behind curand_Philox4x32_10): 128 random bits for the
/// counter (c0, c1, c2, c3) under the key (k0, k1). The host check runs the Random123 known-answer vectors and,
/// with the CUDA headers present, compares it with curand_Philox4x32_10.
struct Philox4x32 {
    uint32_t v[4];
};

SUROGATE_FORK_HD void philox_mulhilo(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
#if defined(__CUDA_ARCH__)
    hi = __umulhi(a, b);
    lo = a * b;
#else
    const uint64_t p = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    hi = static_cast<uint32_t>(p >> 32);
    lo = static_cast<uint32_t>(p);
#endif
}

SUROGATE_FORK_HD Philox4x32 philox4x32_10(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0, uint32_t k1) {
    for (int round = 0; round < 10; ++round) {
        uint32_t hi0, lo0, hi1, lo1;
        philox_mulhilo(0xD2511F53u, c0, hi0, lo0);
        philox_mulhilo(0xCD9E8D57u, c2, hi1, lo1);
        const uint32_t n0 = hi1 ^ c1 ^ k0;
        const uint32_t n2 = hi0 ^ c3 ^ k1;
        c0 = n0;
        c1 = lo1;
        c2 = n2;
        c3 = lo0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return Philox4x32{{c0, c1, c2, c3}};
}

/// Second key word of the SR kernel's Philox stream (the first is the caller's seed). Arbitrary; it keeps the
/// stream apart from other Philox users of the same seed.
constexpr uint32_t kSrKey1 = 0x346F3653u;  // "S6o4"

/// The 16 random bytes of one 16-value block: element i takes byte i % 4 of word i / 4. counter =
/// (block index, generation, 0, 0), key = (seed, kSrKey1); the kernel passes generation 0.
SUROGATE_FORK_HD Philox4x32 sr_block_random(uint32_t block_index, uint32_t generation, uint32_t seed) {
    return philox4x32_10(block_index, generation, 0u, 0u, seed, kSrKey1);
}

SUROGATE_FORK_HD uint32_t sr_byte(const Philox4x32& r, int i) {
    return (r.v[i >> 2] >> ((i & 3) * 8)) & 0xFFu;
}

}  // namespace surogate_fork::nvfp4
