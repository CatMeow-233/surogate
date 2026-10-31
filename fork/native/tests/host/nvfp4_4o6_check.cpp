// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// CPU checks of the arithmetic the fork's Four Over Six kernels run (fork ledger T24):
// fork/native/include/surogate_fork/nvfp4_4o6_arith.h, the header fork/native/src/nvfp4_4o6.cu calls, compiled
// here with the host compiler. Built and run by fork/patches/tests/native/test_nvfp4_4o6.py.
//
//   nvfp4_4o6_check            run every check; prints "ok <name>" or "FAIL <name>: <why>", exit status = failures
//   nvfp4_4o6_check dump FILE  evaluate the requests in FILE (one per line, hex floats) for the Python emulator:
//                                e4m3 X      -> e4m3_code_rn(X)
//                                nosat X     -> e4m3_code_no_saturation(clamp_block_scale(X))
//                                sr Y BYTE   -> e2m1_stochastic(Y, BYTE) as a hex float
//                                philox C0 C1 C2 C3 K0 K1 -> four hex words
//
// With -DFORK_CUDA_HEADERS (the CUDA toolkit's include directory on the path) two more checks compare the header
// with NVIDIA's host implementations: __nv_cvt_float_to_fp8 (cuda_fp8.h) on every float in [2^-11, 512], and
// curand_Philox4x32_10 (curand_philox4x32_x.h, compiled for the host).

#include <cinttypes>
#include <cstdarg>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "surogate_fork/nvfp4_4o6_arith.h"

#if defined(FORK_CUDA_HEADERS)
#include <cuda_fp8.h>
#include <vector_types.h>
#define QUALIFIERS static inline
#include <curand_philox4x32_x.h>
#endif

namespace q4 = surogate_fork::nvfp4;

namespace {

int g_failures = 0;

void check(const char* name, const std::function<std::string()>& body) {
    const std::string why = body();
    if (why.empty()) {
        std::printf("ok %s\n", name);
    } else {
        std::printf("FAIL %s: %s\n", name, why.c_str());
        ++g_failures;
    }
    std::fflush(stdout);
}

std::string fmt(const char* format, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* format, ...) {
    char buf[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buf, sizeof buf, format, args);
    va_end(args);
    return buf;
}

float next_up(float x) {
    return std::nextafter(x, INFINITY);
}

const double kGrid[] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

double spacing(double a) {
    return a < 2.0 ? 0.5 : (a < 4.0 ? 1.0 : 2.0);
}

bool on_grid(float q) {
    for (double g : kGrid)
        if (std::fabs(q) == g) return true;
    return false;
}

//! cvt.rn.satfinite.e2m1x2.f32 (round to nearest, ties to the even code: 0, 1, 2, 4), for the upstream control.
double e2m1_rn(double y) {
    const double a = std::fmin(std::fabs(y), 6.0);
    int hi = 0;
    while (hi < 7 && kGrid[hi] < a)
        ++hi;
    if (kGrid[hi] == a || hi == 0) return std::copysign(kGrid[hi], y);
    const double lo = kGrid[hi - 1], up = kGrid[hi];
    double mag = (up - a < a - lo) ? up : lo;
    if (up - a == a - lo) mag = (hi % 2 == 0) ? up : lo;
    return std::copysign(mag, y);
}

//! E over the 256 bytes of the fork rounding.
double expect_fixed(float y) {
    double sum = 0.0;
    for (uint32_t b = 0; b < 256; ++b)
        sum += q4::e2m1_stochastic(y, b);
    return sum / 256.0;
}

//! E over the 256 bytes of upstream's rounding: RN(y + (b / 256 - 0.5) * 0.5), the sum rounded to fp32.
double expect_upstream(float y) {
    double sum = 0.0;
    for (uint32_t b = 0; b < 256; ++b) {
        const float noise = (static_cast<float>(b) / 256.0f - 0.5f) * 0.5f;
        sum += e2m1_rn(static_cast<float>(y + noise));
    }
    return sum / 256.0;
}

//! Independent E4M3 reference: nearest of the 127 finite non-negative values, ties to the even code, satfinite.
uint8_t e4m3_reference(float x) {
    if (!(x > 0.0f)) return 0;
    int best = 0;
    for (int c = 1; c <= q4::kE4M3MaxCode; ++c) {
        const double dc = std::fabs(q4::e4m3_value(static_cast<uint8_t>(c)) - static_cast<double>(x));
        const double db = std::fabs(q4::e4m3_value(static_cast<uint8_t>(best)) - static_cast<double>(x));
        if (dc < db || (dc == db && (c % 2) == 0)) best = c;
    }
    return static_cast<uint8_t>(best);
}

std::string philox_known_answers() {
    // Random123 kat_vectors, philox4x32_10.
    struct Kat {
        uint32_t c[4], k[2], r[4];
    } kats[] = {
        {{0, 0, 0, 0}, {0, 0}, {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}},
        {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu},
         {0xffffffffu, 0xffffffffu},
         {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}},
        {{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u},
         {0xa4093822u, 0x299f31d0u},
         {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}},
    };
    for (const Kat& kat : kats) {
        const q4::Philox4x32 r = q4::philox4x32_10(kat.c[0], kat.c[1], kat.c[2], kat.c[3], kat.k[0], kat.k[1]);
        for (int i = 0; i < 4; ++i)
            if (r.v[i] != kat.r[i])
                return fmt("counter %08x...: word %d is %08x, Random123 says %08x", kat.c[0], i, r.v[i], kat.r[i]);
    }
    return {};
}

std::string sr_bytes_follow_the_documented_layout() {
    const q4::Philox4x32 r{{0x03020100u, 0x07060504u, 0x0b0a0908u, 0x0f0e0d0cu}};
    for (int i = 0; i < 16; ++i)
        if (q4::sr_byte(r, i) != static_cast<uint32_t>(i)) return fmt("element %d takes byte %u", i, q4::sr_byte(r, i));
    const q4::Philox4x32 a = q4::sr_block_random(5, 0, 42), b = q4::sr_block_random(5, 1, 42),
                         c = q4::sr_block_random(6, 0, 42), d = q4::sr_block_random(5, 0, 43);
    const q4::Philox4x32 want = q4::philox4x32_10(5, 0, 0, 0, 42, q4::kSrKey1);
    if (std::memcmp(&a, &want, sizeof a) != 0) return "sr_block_random is not philox(block, generation, 0, 0; seed, key1)";
    if (!std::memcmp(&a, &b, sizeof a) || !std::memcmp(&a, &c, sizeof a) || !std::memcmp(&a, &d, sizeof a))
        return "block, generation and seed do not all change the stream";
    return {};
}

std::string sr_byte_marginals_are_uniform() {
    // 2^16 blocks x 16 bytes; chi^2 with 255 degrees of freedom (mean 255, sd 22.6): accept below 400.
    std::vector<uint64_t> hist(256, 0);
    for (uint32_t block = 0; block < (1u << 16); ++block) {
        const q4::Philox4x32 r = q4::sr_block_random(block, 3, 0x9E3779BAu);
        for (int i = 0; i < 16; ++i)
            ++hist[q4::sr_byte(r, i)];
    }
    const double expect = (1u << 16) * 16.0 / 256.0;
    double chi2 = 0.0;
    for (uint64_t h : hist)
        chi2 += (h - expect) * (h - expect) / expect;
    return chi2 < 400.0 ? std::string{} : fmt("chi^2 = %.1f over 255 degrees of freedom", chi2);
}

std::string e4m3_values_round_trip() {
    if (q4::e4m3_value(0x7E) != 448.0f || q4::e4m3_value(0x01) != q4::kE4M3Min || q4::e4m3_value(0x08) != 0.015625f)
        return "448, 2^-9 or 2^-6 decode wrongly";
    for (int c = 0; c <= q4::kE4M3MaxCode; ++c) {
        const float v = q4::e4m3_value(static_cast<uint8_t>(c));
        if (q4::e4m3_code_rn(v) != c) return fmt("code %#x (%a) re-encodes as %#x", c, v, q4::e4m3_code_rn(v));
        if (c > 0 && !(v > q4::e4m3_value(static_cast<uint8_t>(c - 1)))) return fmt("code %#x is not increasing", c);
    }
    return {};
}

std::string e4m3_rn_matches_an_independent_reference() {
    // Every E4M3 value, its neighbours, the midpoints and their neighbours, then 2^20 log-uniform floats.
    std::vector<float> xs = {0.0f, 1e-30f, 0.0009765625f, 448.0f, 463.99997f, 464.0f, 480.0f, 1e6f, INFINITY};
    for (int c = 0; c < q4::kE4M3MaxCode; ++c) {
        const float lo = q4::e4m3_value(static_cast<uint8_t>(c)), hi = q4::e4m3_value(static_cast<uint8_t>(c + 1));
        const float mid = static_cast<float>((static_cast<double>(lo) + hi) / 2);
        for (float x : {lo, hi, mid})
            for (float y : {std::nextafter(x, 0.0f), x, next_up(x)})
                xs.push_back(y);
    }
    uint64_t state = 0x853c49e6748fea9bULL;
    for (int i = 0; i < (1 << 20); ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const double u = static_cast<double>(state >> 11) * 0x1p-53;
        xs.push_back(static_cast<float>(std::exp2(-12.0 + 21.0 * u)));
    }
    for (float x : xs) {
        const uint8_t got = q4::e4m3_code_rn(x), want = e4m3_reference(x);
        if (got != want) return fmt("x=%a: %#x, nearest is %#x", x, got, want);
    }
    return {};
}

std::string no_saturation_code_is_the_smallest_code_above_the_slack() {
    if (q4::e4m3_code_no_saturation(384.0f) != q4::e4m3_code_rn(384.0f) ||
        q4::e4m3_code_no_saturation(256.0f) != q4::e4m3_code_rn(256.0f))
        return "256 or 384 do not stay exact";
    if (q4::e4m3_value(q4::e4m3_code_no_saturation(384.00006f)) != 384.0f)
        return "a few-ulp excess over 384 moved to the next code";
    uint64_t state = 7;
    for (int i = 0; i < (1 << 20); ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const double u = static_cast<double>(state >> 11) * 0x1p-53;
        const float x = q4::clamp_block_scale(static_cast<float>(std::exp2(-9.0 + 17.807 * u)));
        const uint8_t code = q4::e4m3_code_no_saturation(x);
        const float floor_x = x * q4::kNoSaturationSlack;
        if (q4::e4m3_value(code) < floor_x) return fmt("x=%a: code %#x decodes below x(1-2^-16)", x, code);
        if (code > 1 && q4::e4m3_value(static_cast<uint8_t>(code - 1)) >= floor_x)
            return fmt("x=%a: code %#x is not the smallest", x, code);
        // The block amax, encoded as x / value * 6 in the kernel's units, stays at or below 6 up to the slack.
        if (6.0 * x / q4::e4m3_value(code) > 6.0 / q4::kNoSaturationSlack) return fmt("x=%a saturates", x);
    }
    return {};
}

std::string clamp_keeps_scales_in_the_finite_e4m3_range() {
    if (q4::clamp_block_scale(0.0f) != q4::kE4M3Min || q4::clamp_block_scale(1e-20f) != q4::kE4M3Min ||
        q4::clamp_block_scale(480.0f) != 448.0f || q4::clamp_block_scale(NAN) != q4::kE4M3Min)
        return "0, tiny, 480 or NaN clamp wrongly";
    if (q4::e4m3_code_rn(q4::clamp_block_scale(1e-20f)) != 1) return "the smallest scale is not code 1";
    return {};
}

std::string sr_returns_grid_values_and_keeps_grid_values() {
    for (double g : kGrid)
        for (uint32_t b = 0; b < 256; ++b) {
            const float y = static_cast<float>(g);
            if (q4::e2m1_stochastic(y, b) != y || q4::e2m1_stochastic(-y, b) != -y)
                return fmt("grid value %g moved with byte %u", g, b);
        }
    for (float y : {6.0f, 6.5f, 1e9f, INFINITY})
        for (uint32_t b : {0u, 128u, 255u})
            if (q4::e2m1_stochastic(y, b) != 6.0f || q4::e2m1_stochastic(-y, b) != -6.0f)
                return fmt("%a does not saturate to 6", y);
    for (int i = 0; i < 6 * 4096; ++i) {
        const float y = static_cast<float>(i) / 4096.0f;
        for (uint32_t b = 0; b < 256; b += 17) {
            const float q = q4::e2m1_stochastic(y, b);
            if (!on_grid(q)) return fmt("y=%a byte %u gives %a, not an E2M1 value", y, b, q);
            const double s = spacing(y);
            if (std::fabs(q - y) >= s) return fmt("y=%a byte %u gives %a, not a neighbour", y, b, q);
            if (q4::e2m1_stochastic(-y, b) != -q) return fmt("y=%a: sign not symmetric", y);
        }
    }
    return {};
}

std::string sr_expectation_is_within_spacing_over_512() {
    // 2^21 points on [0, 6) (signs are symmetric by the previous check) plus the binade edges.
    double worst = 0.0;
    float at = 0.0f;
    std::vector<float> ys;
    for (int i = 0; i < (1 << 21); ++i)
        ys.push_back(static_cast<float>(6.0 * (i + 0.5) / (1 << 21)));
    for (float edge : {2.0f, 4.0f, 6.0f})
        ys.insert(ys.end(), {std::nextafter(edge, 0.0f), edge});
    for (float y : ys) {
        const double rel = std::fabs(expect_fixed(y) - y) / spacing(y);
        if (rel > worst) worst = rel, at = y;
    }
    return worst <= 1.0 / 512.0 + 1e-12 ? std::string{} : fmt("|E-y|/spacing = %.3e at %a (bound 1/512)", worst, at);
}

std::string sr_bias_averages_to_zero_on_the_half_step_lattice() {
    // y = (j + 1/2) / 512: the fraction sits exactly on RTNE ties of the 8-bit probability, where the plain
    // (b + 0.5) / 256 < f rule (SRF) keeps a -3.3e-4 bias.
    double sum = 0.0;
    int n = 0;
    for (int j = 0; j < 6 * 512; ++j) {
        const float y = static_cast<float>((j + 0.5) / 512.0);
        sum += expect_fixed(y) - y;
        ++n;
    }
    return std::fabs(sum / n) < 1e-6 ? std::string{} : fmt("mean bias %.3e", sum / n);
}

std::string negative_control_upstream_dither_is_biased() {
    // The rule this replaces: 2.4 rounds up with p = 76/256 instead of 0.4; 2.2 and 4.6 never round up.
    const double e24 = expect_upstream(2.4f), e22 = expect_upstream(2.2f), e46 = expect_upstream(4.6f);
    if (std::fabs(e24 - (2.0 + 76.0 / 256.0)) > 1e-12) return fmt("E[2.4] = %.6f", e24);
    if (e22 != 2.0 || e46 != 4.0) return fmt("E[2.2] = %.6f, E[4.6] = %.6f", e22, e46);
    if (std::fabs(expect_fixed(2.4f) - 2.4) > 1.0 / 512.0) return "the fork rounding is biased at 2.4";
    return {};
}

#if defined(FORK_CUDA_HEADERS)
std::string e4m3_rn_matches_cuda_fp8_on_every_float() {
    // Every float in [2^-11, 512], and the specials; the kernels clamp to [2^-9, 448] before converting.
    uint32_t bad = 0, first = 0;
    for (uint32_t u = 0x3A000000u; u <= 0x44000000u; ++u) {
        const float x = q4::bits_float(u);
        const auto want = static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
        if (q4::e4m3_code_rn(x) != want && bad++ == 0) first = u;
    }
    for (float x : {0.0f, 1e-30f, 1e6f, INFINITY}) {
        const auto want = static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
        if (q4::e4m3_code_rn(x) != want && bad++ == 0) first = q4::float_bits(x);
    }
    return bad == 0 ? std::string{} : fmt("%u floats differ, first %a", bad, q4::bits_float(first));
}

std::string philox_matches_curand() {
    uint64_t state = 99;
    auto next = [&] {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(state >> 32);
    };
    for (int i = 0; i < 100000; ++i) {
        const uint4 c = make_uint4(next(), next(), next(), next());
        const uint2 k = make_uint2(next(), next());
        const uint4 want = curand_Philox4x32_10(c, k);
        const q4::Philox4x32 got = q4::philox4x32_10(c.x, c.y, c.z, c.w, k.x, k.y);
        if (got.v[0] != want.x || got.v[1] != want.y || got.v[2] != want.z || got.v[3] != want.w)
            return fmt("counter %08x %08x %08x %08x differs", c.x, c.y, c.z, c.w);
    }
    return {};
}
#endif

int dump(const char* path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", path);
        return 2;
    }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream words(line);
        std::string op;
        words >> op;
        if (op == "e4m3" || op == "nosat" || op == "sr") {
            std::string xs;
            words >> xs;
            const float x = std::strtof(xs.c_str(), nullptr);
            if (op == "e4m3") {
                std::printf("%u\n", q4::e4m3_code_rn(x));
            } else if (op == "nosat") {
                std::printf("%u\n", q4::e4m3_code_no_saturation(q4::clamp_block_scale(x)));
            } else {
                uint32_t byte = 0;
                words >> byte;
                std::printf("%a\n", static_cast<double>(q4::e2m1_stochastic(x, byte)));
            }
        } else if (op == "philox") {
            uint32_t v[6];
            for (uint32_t& w : v)
                words >> w;
            const q4::Philox4x32 r = q4::philox4x32_10(v[0], v[1], v[2], v[3], v[4], v[5]);
            std::printf("%u %u %u %u\n", r.v[0], r.v[1], r.v[2], r.v[3]);
        } else {
            std::fprintf(stderr, "unknown request: %s\n", line.c_str());
            return 2;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "dump") == 0) return dump(argv[2]);
    check("philox_known_answers", philox_known_answers);
    check("sr_bytes_follow_the_documented_layout", sr_bytes_follow_the_documented_layout);
    check("sr_byte_marginals_are_uniform", sr_byte_marginals_are_uniform);
    check("e4m3_values_round_trip", e4m3_values_round_trip);
    check("e4m3_rn_matches_an_independent_reference", e4m3_rn_matches_an_independent_reference);
    check("no_saturation_code_is_the_smallest_code_above_the_slack",
          no_saturation_code_is_the_smallest_code_above_the_slack);
    check("clamp_keeps_scales_in_the_finite_e4m3_range", clamp_keeps_scales_in_the_finite_e4m3_range);
    check("sr_returns_grid_values_and_keeps_grid_values", sr_returns_grid_values_and_keeps_grid_values);
    check("sr_expectation_is_within_spacing_over_512", sr_expectation_is_within_spacing_over_512);
    check("sr_bias_averages_to_zero_on_the_half_step_lattice", sr_bias_averages_to_zero_on_the_half_step_lattice);
    check("negative_control_upstream_dither_is_biased", negative_control_upstream_dither_is_biased);
#if defined(FORK_CUDA_HEADERS)
    check("e4m3_rn_matches_cuda_fp8_on_every_float", e4m3_rn_matches_cuda_fp8_on_every_float);
    check("philox_matches_curand", philox_matches_curand);
#endif
    return g_failures;
}
