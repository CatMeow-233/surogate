// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// Host loops of the cpu_training step, vectorized and run on OmpTeam teams (fork ledgers T14 and T15). Each is
// opt-in and gives the same bits as the upstream loop it replaces, for every input, team size and runtime ISA
// (clone), except that a NaN in an FP32 result may carry another NaN's payload: an operation with two NaN operands
// returns one of them, IEEE 754 does not say which, and GCC commutes the operands of + and * (BF16 results map
// every NaN to 0x7fff, so they match outright). For the AdamW loops this holds only where the build's baseline ISA
// has no fused multiply-add (upstream's x86-64 wheel): elsewhere (aarch64, or x86-64 built with FMA in CXXFLAGS)
// upstream's cpu_adamw.cpp fuses multiply-adds, and SUROGATE_FORK_CPU_ADAMW_SIMD=1 is refused
// (cpu_adamw_simd_supported()). The additions multiply nothing and match on every ISA. The loops replaced:
//   - host_grad_add_*: DslGradStore::accumulate_staged_layer, which adds each staged layer of a micro-step after
//     the first into the CPU gradients inside a CUDA host function (SUROGATE_FORK_HOST_GRAD_ACCUM=1, T14);
//   - adamw_step_*: the element-wise loops of cpu_adamw.cpp's cpu_adamw_step, cpu_adamw_step_bf16 and
//     cpu_adamw_step_bf16_param (SUROGATE_FORK_CPU_ADAMW_SIMD=1, T15).
//
// Why the bits match:
//   - every element is computed by one thread with the upstream expression, in the upstream order of
//     operations, so neither the team size nor the chunking can change it;
//   - fork/native/src/cpu_step_kernels.cpp is compiled with -fno-math-errno, which lets std::sqrt vectorize
//     (IEEE sqrt is correctly rounded in every width), and -ffp-contract=off, so no clone fuses a multiply and an
//     add into an FMA that the upstream build (x86-64 baseline ISA, no FMA) does not use;
//   - where the baseline ISA has FMA, upstream's build fuses and this file cannot match it, hence the refusal;
//   - BF16 conversions use the bit operations of cuda_bf16.hpp's host path: widening is a shift, narrowing rounds
//     to nearest even and maps every NaN to 0x7fff, written without the branch that keeps GCC from vectorizing
//     the upstream loop.
// The serial inner kernels carry target_clones (x86-64-v4, AVX2, baseline) and each team thread calls one on its
// own chunk; a clone around the parallel region would not reach the outlined region's body.
//
// The switches are read once per process from the environment: the variable set to exactly "1" turns the path
// on; anything else, or unset, keeps upstream's loop.
#pragma once

#include <cstddef>

namespace surogate_fork {

// SUROGATE_FORK_HOST_GRAD_ACCUM=1 (T14).
bool host_grad_accum_enabled();
// SUROGATE_FORK_GRAD_D2H_STREAM=1 (T14): DslGradStore copies block gradients and runs the host additions on a
// stream of its own instead of the weight-prefetch side stream.
bool grad_d2h_stream_enabled();
// SUROGATE_FORK_CPU_ADAMW_SIMD=1 (T15), and cpu_adamw_simd_supported(); otherwise the switch is ignored, with a note
// on stderr. AsyncStaleAdamW (the Dispatch-PP async optimizer surface) runs cpu_adamw_step too, so it follows.
bool cpu_adamw_simd_enabled();
// Whether the AdamW loops can give upstream's bits in this build: false where the baseline ISA has a fused
// multiply-add (see the top of this file).
bool cpu_adamw_simd_supported();

// dst[i] = bf16(float(dst[i]) + float(src[i])) for BF16 `dst` and `src`, as accumulate_staged_layer computes it.
// The team is OmpTeam(n, serial_group): the CUDA host function passes its DslGradStore, the key its rank thread
// set with set_omp_serial_group (omp_team.h), so the two do not halve each other's teams. Returns the team size.
int host_grad_add_bf16(void* dst, const void* src, std::size_t n, const void* serial_group);
// dst[i] += src[i] for FP32 `dst` and `src`. Returns the team size.
int host_grad_add_f32(float* dst, const float* src, std::size_t n, const void* serial_group);

// The scalars of one AdamW step as cpu_adamw.cpp computes them before its loop.
struct AdamWStep {
    float lr;
    float beta1;
    float beta2;
    float bc1;  // 1 - beta1^step
    float bc2;  // 1 - beta2^step
    float eps;
    float wd_factor;  // 1 - lr * weight_decay
    float grad_scale;
};

// cpu_adamw_step: FP32 master, FP32 gradient. The team is OmpTeam(n), the calling thread's as in cpu_adamw.cpp
// (T5). Each returns the team size.
int adamw_step_f32(float* param, const float* grad, float* m, float* v, std::size_t n, const AdamWStep& s);
// cpu_adamw_step_bf16: FP32 master, BF16 gradient.
int adamw_step_bf16_grad(float* param, const void* grad_bf16, float* m, float* v, std::size_t n, const AdamWStep& s);
// cpu_adamw_step_bf16_param: BF16 master, BF16 gradient.
int adamw_step_bf16_param(void* param_bf16,
                          const void* grad_bf16,
                          float* m,
                          float* v,
                          std::size_t n,
                          const AdamWStep& s);

}  // namespace surogate_fork
