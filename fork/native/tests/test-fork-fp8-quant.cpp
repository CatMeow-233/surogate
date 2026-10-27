// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

// FP8 backward consumes cached quantized weights (fork ledger T4). Built into integration-tests
// by fork/native/cmake/fork_native.cmake.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernels.h"
#include "recipes/fp8_hybrid/fp8_hybrid_recipe.h"
#include "runtime/core/fp8_scaling_state.h"
#include "runtime/training/model.h"
#include "utilities/allocator.h"

TEST_CASE("FP8 backward consumes cached weights without requantizing them", "[fp8][recipe][cuda]") {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        SKIP("No CUDA device available");
    }
    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    cudaDeviceProp properties{};
    REQUIRE(cudaGetDeviceProperties(&properties, 0) == cudaSuccess);
    if (properties.major < 8 || (properties.major == 8 && properties.minor < 9)) {
        SKIP("FP8 matmul requires SM89 or newer");
    }

    constexpr int M = 32;
    constexpr int K = 64;
    for (const int N : {64, 128}) {
        for (const auto weight_dtype : {ETensorDType::BF16, ETensorDType::FP32}) {
            INFO("N=" << N << " K=" << K << " weight_dtype=" << static_cast<int>(weight_dtype));
            auto allocator = std::make_shared<TensorAllocator>();
            IRunState rs(nullptr, 1, M, allocator);
            auto allocate = [&](ETensorDType dtype, const char* name, std::vector<long> shape) {
                return allocator->allocate(dtype, name, shape);
            };
            auto allocate_quant = [&](ETensorDType dtype, const char* name, std::vector<long> shape) {
                Tensor result = allocate(dtype, name, shape);
                result.Stats = allocate(ETensorDType::FP32, name, {Tensor::STATS_FLOATS}).get<float>();
                return result;
            };
            Tensor scratch = allocate(ETensorDType::BYTE, "recipe_scratch", {1 << 20});
            Tensor inp = allocate(ETensorDType::BF16, "recipe_input", {M, K});
            Tensor dout = allocate(ETensorDType::BF16, "recipe_dout", {M, N});
            Tensor weight = allocate(weight_dtype, "recipe_weight", {N, K});
            Tensor dinp = allocate(ETensorDType::BF16, "recipe_dinp", {M, K});
            Tensor dweight = allocate(ETensorDType::BF16, "recipe_dweight", {N, K});
            Tensor grad_quant = allocate_quant(ETensorDType::FP8_E5M2, "recipe_grad_quant", {M, N});
            Tensor weight_quant = allocate_quant(ETensorDType::FP8_E4M3, "recipe_weight_quant", {N, K});
            Tensor weight_quant_t = allocate_quant(ETensorDType::FP8_E4M3, "recipe_weight_quant_t", {K, N});

            auto upload = [&](Tensor& tensor) {
                // Binary fractions are represented exactly by both source dtypes.
                std::vector<float> values(tensor.nelem());
                for (std::size_t i = 0; i < values.size(); ++i)
                    values[i] = (static_cast<int>((i * 17 + 3) % 31) - 15) * 0.0625f;
                auto copy = [&](const void* source) {
                    REQUIRE(
                        cudaMemcpyAsync(tensor.Data, source, tensor.bytes(), cudaMemcpyHostToDevice, rs.MainStream) ==
                        cudaSuccess);
                    REQUIRE(cudaStreamSynchronize(rs.MainStream) == cudaSuccess);
                };
                if (tensor.DType == ETensorDType::BF16) {
                    std::vector<nv_bfloat16> bf16(values.begin(), values.end());
                    copy(bf16.data());
                } else {
                    copy(values.data());
                }
            };
            upload(inp);
            upload(dout);
            upload(weight);
            abs_max(weight_quant.abs_max(), weight, weight.nelem(), rs.DeviceProp, rs.MainStream);
            quantize_with_abs_max(weight_quant,
                                  weight_quant.scale(),
                                  weight,
                                  weight_quant.abs_max(),
                                  weight.nelem(),
                                  rs.DeviceProp,
                                  rs.MainStream);
            transpose(weight_quant_t, weight_quant, N, K, rs.MainStream);
            REQUIRE(cudaMemcpyAsync(weight_quant_t.scale(),
                                    weight_quant.scale(),
                                    sizeof(float),
                                    cudaMemcpyDeviceToDevice,
                                    rs.MainStream) == cudaSuccess);

            modules::MatmulContext ctx;
            ctx.inp = &inp;
            ctx.weight = &weight;
            ctx.dout = &dout;
            ctx.dinp = &dinp;
            ctx.dweight = &dweight;
            ctx.dout_quant = &grad_quant;
            ctx.B = 1;
            ctx.T = M;
            ctx.C_in = K;
            ctx.C_out = N;
            ctx.run_state = &rs;
            ctx.stream = rs.MainStream;
            ctx.op_caps = dsl::OpCapabilities{dsl::OpCapabilityFp8Eligible};
            ctx.matmul_caps = dsl::MatmulCapabilities{dsl::MatmulCapabilityFp8BackwardEligible};
            recipes::FP8HybridRecipe recipe;
            auto read = [&](const Tensor& tensor) {
                REQUIRE(cudaStreamSynchronize(rs.MainStream) == cudaSuccess);
                std::vector<std::uint16_t> values(tensor.nelem());
                REQUIRE(cudaMemcpy(values.data(), tensor.Data, tensor.bytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
                return values;
            };
            auto run = [&](const Tensor* cache, bool skip_weight_grad, std::size_t scratch_bytes) {
                rs.Stack = DeviceMemoryStack(scratch.Data, scratch_bytes, rs.DeviceId);
                ctx.cached_weight = cache;
                ctx.skip_weight_grad = skip_weight_grad;
                recipe.backward_matmul(ctx);
                return read(dinp);
            };

            const auto expected_dinp = run(nullptr, false, scratch.bytes());
            const auto expected_dweight = read(dweight);
            // Square caches retain the existing W^T interpretation; rectangular ordinary
            // caches exercise the branch which still needs a transpose.
            if (N != K) {
                REQUIRE(run(&weight_quant, false, scratch.bytes()) == expected_dinp);
                REQUIRE(read(dweight) == expected_dweight);
            }
            REQUIRE(run(&weight_quant_t, false, scratch.bytes()) == expected_dinp);
            REQUIRE(read(dweight) == expected_dweight);
            // A frozen cached dgrad needs no recipe scratch. The old path tried to
            // allocate/quantize an unused full weight here and exhausted this budget.
            ctx.dweight = nullptr;
            REQUIRE(run(&weight_quant_t, true, 0) == expected_dinp);
            REQUIRE(rs.Stack.max_utilization() == 0);

            ctx.weight = &weight_quant;
            REQUIRE(run(nullptr, true, scratch.bytes()) == expected_dinp);
            REQUIRE(run(&weight_quant_t, true, 0) == expected_dinp);

            Tensor invalid_cache = weight_quant_t;
            invalid_cache.Stats = nullptr;
            ctx.weight = &weight;
            REQUIRE_THROWS_WITH(run(&invalid_cache, true, scratch.bytes()),
                                N == K ? "FP8HybridRecipe::backward_matmul: cached FP8 weight missing scale Stats"
                                       : "FP8HybridRecipe::backward_matmul: cached FP8 W^T missing scale Stats");
            Tensor invalid_weight = weight_quant;
            invalid_weight.Stats = nullptr;
            ctx.weight = &invalid_weight;
            REQUIRE_THROWS_WITH(run(&weight_quant_t, true, scratch.bytes()),
                                "FP8HybridRecipe::backward_matmul: FP8 weight missing scale Stats");
        }
    }
}
