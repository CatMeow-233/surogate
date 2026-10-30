// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0

// Base expert weight gradients (fork ledger T3), moved out of CompiledExecutor::backward_moe_base_weights
// (csrc/src/runtime/ops/moe_grouped_gemm.cpp, upstream 282f9901) so that a test can call it. Compiled into
// surogate-common by fork/native/cmake/fork_native.cmake. Apart from the policy-gated zero-fill (upstream:
// `if (!accumulate) fill_zero(...)`), the code is upstream's.

#include "surogate_fork/moe_weight_grad.h"

#include <cstddef>
#include <stdexcept>

#include "surogate_fork/moe_weight_grad_policy.h"
#include "utilities/dtype.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"

namespace dsl {

void moe_base_weight_grad(Tensor& dw,
                          const Tensor& dout,
                          const Tensor& input,
                          const int* offsets,
                          int experts,
                          bool accumulate,
                          cublasHandle_t cublas,
                          cudaStream_t stream,
                          const std::string& name) {
    const int rows = dout.Sizes[1], cols = input.Sizes[1];
    if (dw.Rank != 3 || dw.Sizes[0] != experts || dw.Sizes[1] != rows || dw.Sizes[2] != cols)
        throw std::runtime_error("Expert weight gradient shape mismatch: " + name);
    auto dtype = [](ETensorDType type) {
        if (type == ETensorDType::BF16) return CUDA_R_16BF;
        if (type == ETensorDType::FP32) return CUDA_R_32F;
        throw std::runtime_error("Expert weight gradients require BF16/FP32 tensors");
    };
    if (moe_weight_grad_needs_zero(offsets, experts, accumulate)) fill_zero(dw, stream);
    const float alpha = 1.f, beta = accumulate ? 1.f : 0.f;
    CUBLAS_CHECK(cublasSetStream(cublas, stream));
    for (int e = 0; e < experts; ++e) {
        const int count = offsets[e + 1] - offsets[e];
        if (count <= 0) continue;
        // Row-major dW = dY^T X; cuBLAS sees dW^T = X^T dY.
        CUBLAS_CHECK(
            cublasGemmEx(cublas,
                         CUBLAS_OP_N,
                         CUBLAS_OP_T,
                         cols,
                         rows,
                         count,
                         &alpha,
                         input.Data + static_cast<std::size_t>(offsets[e]) * cols * get_dtype_size(input.DType),
                         dtype(input.DType),
                         cols,
                         dout.Data + static_cast<std::size_t>(offsets[e]) * rows * get_dtype_size(dout.DType),
                         dtype(dout.DType),
                         rows,
                         &beta,
                         dw.Data + static_cast<std::size_t>(e) * rows * cols * get_dtype_size(dw.DType),
                         dtype(dw.DType),
                         cols,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

}  // namespace dsl
