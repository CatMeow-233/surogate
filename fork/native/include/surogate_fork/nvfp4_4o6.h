// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0

// The fork's Four Over Six NVFP4 quantization kernels (fork/native/src/nvfp4_4o6.cu), which the upstream
// launchers in csrc/src/recipes/nvfp4/kernels/quant_nvfp4_4o6.cu call instead of the kernels in
// quant_nvfp4_4o6_sm100.cu unless the matching *_LEGACY switch is set. Both take the tensor's global amax (the
// launcher computes it first) and write the same outputs as upstream's kernels: packed E2M1 codes [M, K/2],
// E4M3 block scales in CUTLASS's Sm1xxBlkScaledConfig layout, decoded by the GEMM as code * scale * amax / 1536.
//
// T24, SR: quantize_4o6_stochastic (the wgrad dout^T operand): stochastic rounding that is unbiased on the whole
//   E2M1 grid, an E4M3 scale rounded up so that no value saturates, encoding with the stored scale, and one
//   Philox4x32-10 draw per 16 values. Upstream (SUROGATE_NVFP4_4O6_SR_LEGACY=1) dithered by a fixed +-0.25 before
//   round-to-nearest and encoded with the unrounded scale: biased wherever the spacing is not 0.5 and by the
//   per-block scale rounding.
// T24, RN: quantize_4o6 (forward operands and the wgrad inp^T operand): both 4/6 candidates encoded with their
//   stored E4M3 scale and compared on what the GEMM decodes, scales clamped to [2^-9, 448]. Upstream
//   (SUROGATE_NVFP4_4O6_RN_LEGACY=1) encoded with 6/amax or 4/amax but decoded with the rounded scale.
// T24, graphs (sr_graph_generation.cuh): captured SR launches draw fresh noise on every graph replay.
//
// The switches are read once per process, at the first call.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace recipes {
enum class FourOverSixErrorMetric;
}  // namespace recipes

namespace surogate_fork::nvfp4 {

/// False when SUROGATE_NVFP4_4O6_RN_LEGACY=1 (upstream's RN kernel runs).
bool four_over_six_rn_fixed();

/// False when SUROGATE_NVFP4_4O6_SR_LEGACY=1 (upstream's SR kernel runs).
bool four_over_six_sr_fixed();

void quantize_4o6(uint8_t* out_fp4,
                  uint8_t* block_scales,
                  const float* global_amax,
                  const nv_bfloat16* in,
                  int M,
                  int K,
                  recipes::FourOverSixErrorMetric error_metric,
                  cudaStream_t stream);

void quantize_4o6_stochastic(uint8_t* out_fp4,
                             uint8_t* block_scales,
                             const float* global_amax,
                             const nv_bfloat16* in,
                             int M,
                             int K,
                             recipes::FourOverSixErrorMetric error_metric,
                             unsigned int seed,
                             cudaStream_t stream);

}  // namespace surogate_fork::nvfp4
