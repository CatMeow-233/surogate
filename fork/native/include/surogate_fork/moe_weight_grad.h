// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <string>

struct Tensor;

namespace dsl {

// Base expert weight gradients of a grouped MoE projection (fork ledger T3). This is the body of
// CompiledExecutor::backward_moe_base_weights from the shape check onwards; it lives here so that
// fork/native/tests/test-fork-moe-grouped-gemm.cu runs the code that production runs.
//
// For each expert e with tokens t in [offsets[e], offsets[e + 1]), in row-major terms:
//   dw[e] = sum over t of dout[t]^T input[t]        accumulate == false
//   dw[e] += sum over t of dout[t]^T input[t]       accumulate == true
// dout is [tokens, rows], input is [tokens, cols] and dw is [experts, rows, cols]; each may be BF16 or
// FP32 (BF16 operands with an FP32 dw is the usual full fine-tuning case). offsets are host offsets
// into the rows of dout and input. An expert without tokens gets zeros, or keeps its gradient when
// accumulating. Runs on `stream`, and sets it on `cublas`.
//
// The whole of dw is zero-filled only when moe_weight_grad_needs_zero() says so: every expert with
// tokens gets one beta=0 GEMM that writes its complete rows x cols block, so a step where every
// expert has tokens overwrites dw without reading it. Any change to the GEMM loop must keep that
// property or change the policy; the [wgrad-zero] test checks it with a NaN-filled dw.
//
// Throws std::runtime_error when dw is not [experts, rows, cols] (`name` names dw in the message),
// before anything is written, and when an expert with tokens has an operand that is neither BF16
// nor FP32.
void moe_base_weight_grad(Tensor& dw,
                          const Tensor& dout,
                          const Tensor& input,
                          const int* offsets,
                          int experts,
                          bool accumulate,
                          cublasHandle_t cublas,
                          cudaStream_t stream,
                          const std::string& name);

}  // namespace dsl
