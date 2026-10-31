// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0

// Four Over Six NVFP4 quantization (arXiv:2512.02010) with the fork's defect fixes (ledger T24: stochastic rounding,
// round to nearest). Compiled into
// surogate-common by fork/native/cmake/fork_native.cmake and called from the upstream launchers in
// csrc/src/recipes/nvfp4/kernels/quant_nvfp4_4o6.cu. Thread layout, the 4-vs-6 error metrics, the CUTLASS
// scale layout and the PTX conversions are upstream's (quant_nvfp4_4o6_sm100.cu, 282f9901); the scale and
// rounding arithmetic is in surogate_fork/nvfp4_4o6_arith.h, which the host check runs. See nvfp4_4o6.h.

#include "surogate_fork/nvfp4_4o6.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

// CUTLASS_ARCH_MMA_SM100_SUPPORTED, the guard upstream's 4/6 kernels use (nvcc 12.8 or newer).
#include "cutlass/arch/config.h"
#include "recipes/nvfp4/nvfp4_recipe.h"
#include "surogate_fork/nvfp4_4o6_arith.h"

namespace surogate_fork::nvfp4 {

namespace {

bool env_is_one(const char* name) {
    const char* value = std::getenv(name);
    return value && std::string_view(value) == "1";
}

}  // namespace

bool four_over_six_rn_fixed() {
    static const bool fixed = !env_is_one("SUROGATE_NVFP4_4O6_RN_LEGACY");
    return fixed;
}

bool four_over_six_sr_fixed() {
    static const bool fixed = !env_is_one("SUROGATE_NVFP4_4O6_SR_LEGACY");
    return fixed;
}

}  // namespace surogate_fork::nvfp4

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace surogate_fork::nvfp4 {

namespace {

constexpr int kBlockSize = 16;
constexpr int kThreads = 256;
constexpr float k4o6TensorScale = 1536.0f;  // 384 * 4: block scales up to 384 (max=4) and 256 (max=6)

enum class Metric { MSE, L1, AbsMax };

__device__ __forceinline__ float rcp_approx_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

// cvt.rn.satfinite.e2m1x2.f32 of 8 values and their decoded values (upstream's helper).
__device__ __forceinline__ void e2m1x8_with_dequant(const float (&x)[8], uint32_t& codes, float (&dq)[8]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t d0, d1, d2, d3;
    asm volatile("{\n"
                 ".reg .b8 byte0, byte1, byte2, byte3;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte0, %6, %5;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte1, %8, %7;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte2, %10, %9;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte3, %12, %11;\n"
                 "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
                 "cvt.rn.f16x2.e2m1x2 %1, byte0;\n"
                 "cvt.rn.f16x2.e2m1x2 %2, byte1;\n"
                 "cvt.rn.f16x2.e2m1x2 %3, byte2;\n"
                 "cvt.rn.f16x2.e2m1x2 %4, byte3;\n"
                 "}"
                 : "=r"(codes), "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
    const uint32_t d[4] = {d0, d1, d2, d3};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        dq[2 * i] = __half2float(__ushort_as_half(static_cast<unsigned short>(d[i] & 0xFFFF)));
        dq[2 * i + 1] = __half2float(__ushort_as_half(static_cast<unsigned short>(d[i] >> 16)));
    }
#else
    codes = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        dq[i] = 0.0f;
#endif
}

// cvt.rn.satfinite.e2m1x2.f32 of 8 values; exact for values already on the E2M1 grid.
__device__ __forceinline__ uint32_t e2m1x8(const float (&x)[8]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t codes;
    asm volatile("{\n"
                 ".reg .b8 byte0, byte1, byte2, byte3;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
                 "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
                 "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
                 "}"
                 : "=r"(codes)
                 : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
    return codes;
#else
    return 0;
#endif
}

template <Metric M>
__device__ __forceinline__ float block_error(const float (&x)[8], const float (&dq)[8], float recon) {
    float err = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float diff = dq[i] * recon - x[i];
        if constexpr (M == Metric::MSE) {
            err += diff * diff;
        } else if constexpr (M == Metric::L1) {
            err += fabsf(diff);
        } else {
            err = fmaxf(err, fabsf(diff));
        }
    }
    return err;
}

// RN codes of the block under both candidates and whether the max=4 candidate has the lower error, the
// reconstruction of each candidate being dq * recon. Upstream's selection, factored out.
template <Metric M>
__device__ __forceinline__ bool select_four(const float (&vals)[16],
                                            float enc6,
                                            float enc4,
                                            float recon6,
                                            float recon4,
                                            uint32_t (&codes6)[2],
                                            uint32_t (&codes4)[2]) {
    float err6 = 0.0f, err4 = 0.0f;
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        float group[8], s6[8], s4[8], dq6[8], dq4[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            group[i] = vals[g * 8 + i];
            s6[i] = group[i] * enc6;
            s4[i] = group[i] * enc4;
        }
        e2m1x8_with_dequant(s6, codes6[g], dq6);
        e2m1x8_with_dequant(s4, codes4[g], dq4);
        err6 += block_error<M>(group, dq6, recon6);
        err4 += block_error<M>(group, dq4, recon4);
    }
    return err4 < err6;
}

__device__ __forceinline__ void unpack_codes(const uint32_t (&codes)[2], uint8_t (&out)[8]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[i] = (codes[0] >> (i * 8)) & 0xFF;
        out[i + 4] = (codes[1] >> (i * 8)) & 0xFF;
    }
}

__device__ __forceinline__ float block_amax(const float (&vals)[16]) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        amax = fmaxf(amax, fabsf(vals[i]));
    return amax;
}

// T24, RN: each candidate is encoded with the scale the GEMM decodes it with, and judged on that decode.
template <Metric M>
__device__ __forceinline__ void
quantize_block_rn(const float (&vals)[16], float global_encode, uint8_t (&out)[8], uint8_t& scale_code) {
    const float amax = block_amax(vals);
    if (amax == 0.0f) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
            out[i] = 0;
        scale_code = 0;
        return;
    }
    const uint8_t code6 = e4m3_code_rn(clamp_block_scale(amax * rcp_approx_ftz(6.0f) * global_encode));
    const uint8_t code4 = e4m3_code_rn(clamp_block_scale(amax * rcp_approx_ftz(4.0f) * global_encode));
    const float sf6 = e4m3_value(code6), sf4 = e4m3_value(code4);
    uint32_t codes6[2], codes4[2];
    const bool four = select_four<M>(
        vals, global_encode / sf6, global_encode / sf4, sf6 / global_encode, sf4 / global_encode, codes6, codes4);
    unpack_codes(four ? codes4 : codes6, out);
    scale_code = four ? code4 : code6;
}

// T24, SR: upstream's 4-vs-6 choice, then the no-saturation scale, the stored-scale encode and SRC rounding.
template <Metric M>
__device__ __forceinline__ void quantize_block_sr(const float (&vals)[16],
                                                  float global_encode,
                                                  const Philox4x32& random,
                                                  uint8_t (&out)[8],
                                                  uint8_t& scale_code) {
    const float amax = block_amax(vals);
    if (amax == 0.0f) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
            out[i] = 0;
        scale_code = 0;
        return;
    }
    const float inv = rcp_approx_ftz(fmaxf(amax, 1e-12f));
    const float recon6 = amax * rcp_approx_ftz(6.0f), recon4 = amax * rcp_approx_ftz(4.0f);
    uint32_t codes6[2], codes4[2];
    const bool four = select_four<M>(vals, 6.0f * inv, 4.0f * inv, recon6, recon4, codes6, codes4);

    scale_code = e4m3_code_no_saturation(clamp_block_scale((four ? recon4 : recon6) * global_encode));
    const float encode = global_encode / e4m3_value(scale_code);
    uint32_t codes[2];
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        float q[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            q[i] = e2m1_stochastic(vals[g * 8 + i] * encode, sr_byte(random, g * 8 + i));
        codes[g] = e2m1x8(q);
    }
    unpack_codes(codes, out);
}

// Upstream's cutlass_scale_index: offset of (row, scale_col) in Sm1xxBlkScaledConfig's SfKMajorAtom layout.
__device__ __forceinline__ size_t cutlass_scale_index(int row, int scale_col, int num_scale_cols) {
    const int atoms_per_row = (num_scale_cols + 3) / 4;
    const size_t atom = static_cast<size_t>(row / 128) * atoms_per_row + scale_col / 4;
    const int r = row % 128;
    return atom * 512 + (r % 32) * 16 + (r / 32) * 4 + scale_col % 4;
}

__device__ __forceinline__ float global_encode_scale(const float* global_amax) {
    const float g = *global_amax;
    return g > 0.0f ? k4o6TensorScale * rcp_approx_ftz(fmaxf(g, 1e-12f)) : 1.0f;
}

__device__ __forceinline__ void load_block(const nv_bfloat16* in, int row, int col_start, int K, float (&vals)[16]) {
    const nv_bfloat16* src = in + static_cast<size_t>(row) * K;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int col = col_start + i;
        vals[i] = col < K ? __bfloat162float(src[col]) : 0.0f;
    }
}

__device__ __forceinline__ void store_block(uint8_t* out_fp4, int row, int col_start, int K, const uint8_t (&codes)[8]) {
    uint8_t* dst = out_fp4 + static_cast<size_t>(row) * (K / 2);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int col = col_start / 2 + i;
        if (col < K / 2) dst[col] = codes[i];
    }
}

template <Metric M>
__global__ void quantize_4o6_rn_kernel(uint8_t* __restrict__ out_fp4,
                                       uint8_t* __restrict__ block_scales,
                                       const float* __restrict__ global_amax,
                                       const nv_bfloat16* __restrict__ in,
                                       int rows,
                                       int K,
                                       int num_scale_cols) {
    const int block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= rows * num_scale_cols) return;
    const int row = block / num_scale_cols, scale_col = block % num_scale_cols;
    float vals[16];
    load_block(in, row, scale_col * kBlockSize, K, vals);
    uint8_t codes[8], scale;
    quantize_block_rn<M>(vals, global_encode_scale(global_amax), codes, scale);
    store_block(out_fp4, row, scale_col * kBlockSize, K, codes);
    block_scales[cutlass_scale_index(row, scale_col, num_scale_cols)] = scale;
}

template <Metric M>
__global__ void quantize_4o6_sr_kernel(uint8_t* __restrict__ out_fp4,
                                       uint8_t* __restrict__ block_scales,
                                       const float* __restrict__ global_amax,
                                       const nv_bfloat16* __restrict__ in,
                                       int rows,
                                       int K,
                                       int num_scale_cols,
                                       uint32_t seed) {
    const int block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= rows * num_scale_cols) return;
    const int row = block / num_scale_cols, scale_col = block % num_scale_cols;
    float vals[16];
    load_block(in, row, scale_col * kBlockSize, K, vals);
    const Philox4x32 random = sr_block_random(static_cast<uint32_t>(block), 0u, seed);
    uint8_t codes[8], scale;
    quantize_block_sr<M>(vals, global_encode_scale(global_amax), random, codes, scale);
    store_block(out_fp4, row, scale_col * kBlockSize, K, codes);
    block_scales[cutlass_scale_index(row, scale_col, num_scale_cols)] = scale;
}

Metric metric_of(recipes::FourOverSixErrorMetric metric) {
    switch (metric) {
        case recipes::FourOverSixErrorMetric::L1:
            return Metric::L1;
        case recipes::FourOverSixErrorMetric::AbsMax:
            return Metric::AbsMax;
        case recipes::FourOverSixErrorMetric::MSE:
        default:
            return Metric::MSE;
    }
}

int grid_for(int rows, int num_scale_cols) {
    return (rows * num_scale_cols + kThreads - 1) / kThreads;
}

}  // namespace

void quantize_4o6(uint8_t* out_fp4,
                  uint8_t* block_scales,
                  const float* global_amax,
                  const nv_bfloat16* in,
                  int M,
                  int K,
                  recipes::FourOverSixErrorMetric error_metric,
                  cudaStream_t stream) {
    const int cols = (K + kBlockSize - 1) / kBlockSize;
    const int grid = grid_for(M, cols);
    if (grid == 0) return;
    switch (metric_of(error_metric)) {
        case Metric::MSE:
            quantize_4o6_rn_kernel<Metric::MSE>
                <<<grid, kThreads, 0, stream>>>(out_fp4, block_scales, global_amax, in, M, K, cols);
            break;
        case Metric::L1:
            quantize_4o6_rn_kernel<Metric::L1>
                <<<grid, kThreads, 0, stream>>>(out_fp4, block_scales, global_amax, in, M, K, cols);
            break;
        case Metric::AbsMax:
            quantize_4o6_rn_kernel<Metric::AbsMax>
                <<<grid, kThreads, 0, stream>>>(out_fp4, block_scales, global_amax, in, M, K, cols);
            break;
    }
}

void quantize_4o6_stochastic(uint8_t* out_fp4,
                             uint8_t* block_scales,
                             const float* global_amax,
                             const nv_bfloat16* in,
                             int M,
                             int K,
                             recipes::FourOverSixErrorMetric error_metric,
                             unsigned int seed,
                             cudaStream_t stream) {
    const int cols = (K + kBlockSize - 1) / kBlockSize;
    const int grid = grid_for(M, cols);
    if (grid == 0) return;
    switch (metric_of(error_metric)) {
        case Metric::MSE:
            quantize_4o6_sr_kernel<Metric::MSE>
                <<<grid, kThreads, 0, stream>>>(out_fp4, block_scales, global_amax, in, M, K, cols, seed);
            break;
        case Metric::L1:
            quantize_4o6_sr_kernel<Metric::L1>
                <<<grid, kThreads, 0, stream>>>(out_fp4, block_scales, global_amax, in, M, K, cols, seed);
            break;
        case Metric::AbsMax:
            quantize_4o6_sr_kernel<Metric::AbsMax>
                <<<grid, kThreads, 0, stream>>>(out_fp4, block_scales, global_amax, in, M, K, cols, seed);
            break;
    }
}

}  // namespace surogate_fork::nvfp4

#else  // !CUTLASS_ARCH_MMA_SM100_SUPPORTED: upstream's launchers refuse first (nvfp4_4o6_sm100::is_supported).

namespace surogate_fork::nvfp4 {

void quantize_4o6(uint8_t*, uint8_t*, const float*, const nv_bfloat16*, int, int, recipes::FourOverSixErrorMetric,
                  cudaStream_t) {}

void quantize_4o6_stochastic(uint8_t*, uint8_t*, const float*, const nv_bfloat16*, int, int,
                             recipes::FourOverSixErrorMetric, unsigned int, cudaStream_t) {}

}  // namespace surogate_fork::nvfp4

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED
