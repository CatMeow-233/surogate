// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// Host loops of the cpu_training step (fork ledgers T14 and T15); surogate_fork/cpu_step_kernels.h states the
// contract. fork/native/cmake/cpu_step_kernels.cmake compiles this file with -fno-math-errno -ffp-contract=off,
// into surogate-common and into the CPU contract test alike.

#include "surogate_fork/cpu_step_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "surogate_fork/omp_team.h"

// x86-64-v4 (AVX-512 F/BW/DQ/VL) widens the BF16 loops to 64-byte vectors, which AVX-512F alone does not; it also
// has FMA, which -ffp-contract=off keeps out (without the flag GCC 13 fuses 57 multiply-adds in this file). The
// clones need GCC's ifunc support (glibc on x86-64 Linux); elsewhere only the baseline loop is built, which GCC
// still vectorizes with SSE2.
// The benchmark defines SUROGATE_FORK_SIMD_CLONES empty to build one ISA at a time (-march), so that it can check
// each clone's loop against the upstream one on a host whose resolver would pick only one of them.
#ifndef SUROGATE_FORK_SIMD_CLONES
#if defined(__x86_64__) && defined(__linux__) && defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 12
#define SUROGATE_FORK_SIMD_CLONES __attribute__((target_clones("arch=x86-64-v4", "avx2", "default")))
#else
#define SUROGATE_FORK_SIMD_CLONES
#endif
#endif

namespace surogate_fork {

namespace {

bool env_flag(const char* name) {
    const char* value = std::getenv(name);
    return value && std::string_view(value) == "1";
}

inline float bf16_to_f32(std::uint16_t h) {
    const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// __float2bfloat16 on the host (cuda_bf16.hpp): NaN -> 0x7fff, otherwise round to nearest even. Adding 0x7fff
// plus the kept LSB carries into bit 16 exactly when the dropped half is above 0x8000, or equal to it with an odd
// LSB; finite values up to the largest one and infinities cannot overflow the 32-bit sum.
inline std::uint16_t f32_to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    const std::uint32_t rounded = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    return (u & 0x7fffffffu) > 0x7f800000u ? std::uint16_t{0x7fff} : static_cast<std::uint16_t>(rounded);
}

SUROGATE_FORK_SIMD_CLONES
void add_bf16_serial(std::uint16_t* __restrict__ dst, const std::uint16_t* __restrict__ src, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = f32_to_bf16(bf16_to_f32(dst[i]) + bf16_to_f32(src[i]));
}

SUROGATE_FORK_SIMD_CLONES
void add_f32_serial(float* __restrict__ dst, const float* __restrict__ src, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        dst[i] += src[i];
}

// The loop bodies below are cpu_adamw.cpp's, expression for expression.
SUROGATE_FORK_SIMD_CLONES
void adamw_f32_serial(float* __restrict__ p,
                      const float* __restrict__ g,
                      float* __restrict__ m,
                      float* __restrict__ v,
                      std::size_t n,
                      const AdamWStep s) {
    for (std::size_t i = 0; i < n; ++i) {
        const float gi = g[i] * s.grad_scale;
        m[i] = s.beta1 * m[i] + (1.0f - s.beta1) * gi;
        v[i] = s.beta2 * v[i] + (1.0f - s.beta2) * gi * gi;
        const float m_hat = m[i] / s.bc1;
        const float v_hat = v[i] / s.bc2;
        p[i] = p[i] * s.wd_factor - s.lr * m_hat / (std::sqrt(v_hat) + s.eps);
    }
}

SUROGATE_FORK_SIMD_CLONES
void adamw_bf16_grad_serial(float* __restrict__ p,
                            const std::uint16_t* __restrict__ g,
                            float* __restrict__ m,
                            float* __restrict__ v,
                            std::size_t n,
                            const AdamWStep s) {
    for (std::size_t i = 0; i < n; ++i) {
        const float gi = bf16_to_f32(g[i]) * s.grad_scale;
        m[i] = s.beta1 * m[i] + (1.0f - s.beta1) * gi;
        v[i] = s.beta2 * v[i] + (1.0f - s.beta2) * gi * gi;
        const float m_hat = m[i] / s.bc1;
        const float v_hat = v[i] / s.bc2;
        p[i] = p[i] * s.wd_factor - s.lr * m_hat / (std::sqrt(v_hat) + s.eps);
    }
}

SUROGATE_FORK_SIMD_CLONES
void adamw_bf16_param_serial(std::uint16_t* __restrict__ p,
                             const std::uint16_t* __restrict__ g,
                             float* __restrict__ m,
                             float* __restrict__ v,
                             std::size_t n,
                             const AdamWStep s) {
    for (std::size_t i = 0; i < n; ++i) {
        float pi = bf16_to_f32(p[i]);
        const float gi = bf16_to_f32(g[i]) * s.grad_scale;
        m[i] = s.beta1 * m[i] + (1.0f - s.beta1) * gi;
        v[i] = s.beta2 * v[i] + (1.0f - s.beta2) * gi * gi;
        const float m_hat = m[i] / s.bc1;
        const float v_hat = v[i] / s.bc2;
        pi = pi * s.wd_factor - s.lr * m_hat / (std::sqrt(v_hat) + s.eps);
        p[i] = f32_to_bf16(pi);
    }
}

// Chunks start on multiples of kChunkElems elements (256 bytes of FP32, 128 of BF16), so no two team threads
// write one cache line.
constexpr std::size_t kChunkElems = 64;

// Runs body(begin, end) over [0, n): on the calling thread for a team of 1, otherwise one contiguous chunk per
// team thread.
template <class Body>
void run_chunks(std::size_t n, int threads, const Body& body) {
    if (threads <= 1) {
        body(std::size_t{0}, n);
        return;
    }
#ifdef _OPENMP
    const std::size_t units = (n + kChunkElems - 1) / kChunkElems;
#pragma omp parallel num_threads(threads)
    {
        const auto t = static_cast<std::size_t>(omp_get_thread_num());
        const auto k = static_cast<std::size_t>(omp_get_num_threads());
        const std::size_t begin = std::min(n, units * t / k * kChunkElems);
        const std::size_t end = std::min(n, units * (t + 1) / k * kChunkElems);
        if (begin < end) body(begin, end);
    }
#else
    body(std::size_t{0}, n);
#endif
}

}  // namespace

bool host_grad_accum_enabled() {
    static const bool on = env_flag("SUROGATE_FORK_HOST_GRAD_ACCUM");
    return on;
}

bool grad_d2h_stream_enabled() {
    static const bool on = env_flag("SUROGATE_FORK_GRAD_D2H_STREAM");
    return on;
}

bool cpu_adamw_simd_enabled() {
    static const bool on = env_flag("SUROGATE_FORK_CPU_ADAMW_SIMD");
    return on;
}

int host_grad_add_bf16(void* dst, const void* src, std::size_t n, const void* serial_group) {
    auto* d = static_cast<std::uint16_t*>(dst);
    const auto* s = static_cast<const std::uint16_t*>(src);
    const OmpTeam team(n, serial_group);
    run_chunks(n, team.threads, [&](std::size_t b, std::size_t e) { add_bf16_serial(d + b, s + b, e - b); });
    return team.threads;
}

int host_grad_add_f32(float* dst, const float* src, std::size_t n, const void* serial_group) {
    const OmpTeam team(n, serial_group);
    run_chunks(n, team.threads, [&](std::size_t b, std::size_t e) { add_f32_serial(dst + b, src + b, e - b); });
    return team.threads;
}

int adamw_step_f32(float* param, const float* grad, float* m, float* v, std::size_t n, const AdamWStep& s) {
    const OmpTeam team(n);
    run_chunks(n, team.threads, [&](std::size_t b, std::size_t e) {
        adamw_f32_serial(param + b, grad + b, m + b, v + b, e - b, s);
    });
    return team.threads;
}

int adamw_step_bf16_grad(float* param, const void* grad_bf16, float* m, float* v, std::size_t n, const AdamWStep& s) {
    const auto* g = static_cast<const std::uint16_t*>(grad_bf16);
    const OmpTeam team(n);
    run_chunks(n, team.threads, [&](std::size_t b, std::size_t e) {
        adamw_bf16_grad_serial(param + b, g + b, m + b, v + b, e - b, s);
    });
    return team.threads;
}

int adamw_step_bf16_param(void* param_bf16,
                          const void* grad_bf16,
                          float* m,
                          float* v,
                          std::size_t n,
                          const AdamWStep& s) {
    auto* p = static_cast<std::uint16_t*>(param_bf16);
    const auto* g = static_cast<const std::uint16_t*>(grad_bf16);
    const OmpTeam team(n);
    run_chunks(n, team.threads, [&](std::size_t b, std::size_t e) {
        adamw_bf16_param_serial(p + b, g + b, m + b, v + b, e - b, s);
    });
    return team.threads;
}

}  // namespace surogate_fork
