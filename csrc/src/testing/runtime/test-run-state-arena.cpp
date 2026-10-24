// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kernels/kernels.h"
#include "runtime/dsl/dsl_run_state.h"
#include "runtime/executor/graph_executor_utils.h"

TEST_CASE("Empty named tensor reads preserve their requests without GPU work", "[dispatch-pp][host-copy]") {
    REQUIRE(dsl::copy_named_tensors_to_host({}, nullptr).empty());
    const auto result = dsl::copy_named_tensors_to_host({{"", nullptr}, {"missing", nullptr}}, nullptr);
    REQUIRE(result.size() == 2);
    CHECK(result[0].first.empty());
    CHECK(result[1].first == "missing");
    CHECK(result[0].second.empty());
    CHECK(result[1].second.empty());
}

TEST_CASE("Batched named tensor reads preserve exact bytes and order", "[dispatch-pp][host-copy][gpu]") {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) SKIP("CUDA device required");

    TensorAllocator allocator;
    auto first = allocator.allocate(ETensorDType::INT32, "first_boundary", {4});
    auto second = allocator.allocate(ETensorDType::BYTE, "second_boundary", {7});
    const std::vector<std::int32_t> first_values{0, -1, 1234567, -7654321};
    const std::vector<unsigned char> second_values{0, 1, 127, 128, 254, 255, 17};
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(first.Data, first_values.data(), first.bytes(), cudaMemcpyHostToDevice, stream) ==
            cudaSuccess);
    REQUIRE(cudaMemcpyAsync(second.Data, second_values.data(), second.bytes(), cudaMemcpyHostToDevice, stream) ==
            cudaSuccess);
    // The reads must order after the uploads on this stream and return completed
    // host data without the caller issuing another completion fence.
    auto zero = Tensor::from_pointer(first.Data, first.Device, ETensorDType::BYTE, std::vector<long>{0});
    auto no_storage = Tensor::from_pointer(nullptr, first.Device, ETensorDType::BYTE, std::vector<long>{4});
    const auto result = dsl::copy_named_tensors_to_host({{"second", &second},
                                                         {"missing", nullptr},
                                                         {"first", &first},
                                                         {"second", &second},
                                                         {"", &first},
                                                         {"zero", &zero},
                                                         {"no_storage", &no_storage}},
                                                        stream);
    const std::vector<std::string> names{"second", "missing", "first", "second", "", "zero", "no_storage"};
    REQUIRE(result.size() == names.size());
    for (std::size_t i = 0; i < names.size(); ++i)
        CHECK(result[i].first == names[i]);
    REQUIRE(result[0].second.size() == second_values.size());
    CHECK(std::memcmp(result[0].second.data(), second_values.data(), second_values.size()) == 0);
    CHECK(result[0].second == result[3].second);
    REQUIRE(result[2].second.size() == first_values.size() * sizeof(std::int32_t));
    CHECK(std::memcmp(result[2].second.data(), first_values.data(), result[2].second.size()) == 0);
    for (const auto i : {1, 4, 5, 6})
        CHECK(result[i].second.empty());

    // Refuse before submitting copies or fencing a capturing stream. A valid
    // capture afterwards proves that rejection did not invalidate the stream.
    REQUIRE(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
    REQUIRE_THROWS_AS(dsl::copy_named_tensors_to_host({{"first", &first}}, stream), std::logic_error);
    cudaGraph_t graph = nullptr;
    REQUIRE(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
    REQUIRE(cudaGraphDestroy(graph) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("Persistent scratch migration preserves alignment and embedding gradients", "[arena][embedding][gpu]") {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) SKIP("CUDA device required");

    constexpr int C = 1024, V = 16;
    for (const int T : {7, 16}) {
        INFO("sequence length=" << T);
        auto allocator = std::make_shared<TensorAllocator>();
        PretrainedConfig config;
        config.HiddenSize = C;
        config.IntermediateSize = 2048;
        config.VocabSize = V;
        config.NumQueryHeads = 8;
        config.NumKeyValHeads = 2;
        config.NumLayers = 1;
        config.MaxPositionEmbeddings = 32;

        auto requirements = dsl::RuntimeRunStateRequirements::embedding();
        requirements.encoder_backward_scratch = true;
        dsl::DslRunState state(config, {}, RuntimeOptions{}, 1, T, allocator,
                              false, false, 1 << 20, nullptr, nullptr, requirements);
        auto& scratch = state.scratch();
        // The production scale buffer is eight bytes. The odd-length loss buffer
        // also exercises padding beyond the usual multiple-of-16 training shapes.
        scratch.matmul_scales = allocator->allocate(ETensorDType::FP32, "scales", {2});
        scratch.cross_entropy_dloss = allocator->allocate(ETensorDType::FP32, "dloss", {T});
        const std::vector<float> scales{0.5f, 2.0f}, dloss(T, 0.25f);
        REQUIRE(cudaMemcpy(scratch.matmul_scales.Data, scales.data(), scales.size() * sizeof(float),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(scratch.cross_entropy_dloss.Data, dloss.data(), dloss.size() * sizeof(float),
                           cudaMemcpyHostToDevice) == cudaSuccess);

        const auto bytes = state.non_graph_persistent_extras_bytes();
        auto arena = allocator->allocate(ETensorDType::BYTE, "extras", {static_cast<long>(bytes + 256)});
        REQUIRE(cudaMemset(arena.Data, 0x7b, arena.bytes()) == cudaSuccess);
        state.rebind_non_graph_persistent_to_arena(arena.Data, bytes, state.MainStream);
        std::byte* end = arena.Data;
        for (const Tensor* tensor : {&scratch.matmul_scales, &scratch.cross_entropy_dloss,
                                    &scratch.encoder_bwd_scratch}) {
            REQUIRE(reinterpret_cast<std::uintptr_t>(tensor->Data) % 256 == 0);
            REQUIRE(tensor->Data >= end);
            end = tensor->Data + tensor->bytes();
            REQUIRE(end <= arena.Data + bytes);
        }
        for (const auto& [tensor, expected] :
             {std::pair{&scratch.matmul_scales, scales}, std::pair{&scratch.cross_entropy_dloss, dloss}}) {
            std::vector<float> actual(expected.size());
            REQUIRE(cudaMemcpy(actual.data(), tensor->Data, tensor->bytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
            REQUIRE(actual == expected);
        }

        auto inputs = allocator->allocate(ETensorDType::INT32, "tokens", {1, T});
        auto upstream = allocator->allocate(ETensorDType::BF16, "upstream", {1, T, C});
        auto gradient = allocator->allocate(ETensorDType::BF16, "gradient", {V, C});
        std::vector<int> tokens(T);
        std::vector<nv_bfloat16> values(T * C), expected(V * C, nv_bfloat16(0.0f));
        for (int t = 0; t < T; ++t) {
            tokens[t] = (t * 3) % 5;
            for (int c = 0; c < C; ++c) {
                values[t * C + c] = nv_bfloat16((t % 3 + c % 7 - 3) * 0.0625f);
                const auto i = tokens[t] * C + c;
                expected[i] = nv_bfloat16(float(expected[i]) + float(values[t * C + c]));
            }
        }
        REQUIRE(cudaMemcpy(inputs.Data, tokens.data(), inputs.bytes(), cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(upstream.Data, values.data(), upstream.bytes(), cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemset(gradient.Data, 0, gradient.bytes()) == cudaSuccess);
        for (int step = 1; step <= 2; ++step) {
            encoder_backward(gradient, scratch.encoder_bwd_scratch, upstream, inputs, 1, T, C, 42, state.MainStream);
            REQUIRE(cudaStreamSynchronize(state.MainStream) == cudaSuccess);
            std::vector<nv_bfloat16> actual(V * C);
            REQUIRE(cudaMemcpy(actual.data(), gradient.Data, gradient.bytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
            for (int i = 0; i < V * C; ++i) REQUIRE(float(actual[i]) == step * float(expected[i]));
        }
        std::vector<unsigned char> guard(256);
        REQUIRE(cudaMemcpy(guard.data(), arena.Data + bytes, guard.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(std::all_of(guard.begin(), guard.end(), [](auto value) { return value == 0x7b; }));
    }
}

TEST_CASE("A captured embedding backward reads the tokens of each replay", "[embedding][gpu]") {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) SKIP("CUDA device required");

    // Two micro-steps in one graph, as a full training step is captured: each copies its tokens from
    // its own pinned buffer into the shared input buffer, then runs the embedding backward.
    constexpr int C = 1024, V = 16, T = 64, kMicroSteps = 2;
    auto allocator = std::make_shared<TensorAllocator>();
    auto scratch =
        allocator->allocate(ETensorDType::BYTE, "scratch", {static_cast<long>(encoder_backward_scratch_bytes(T))});
    auto inputs = allocator->allocate(ETensorDType::INT32, "tokens", {1, T});
    auto gradient = allocator->allocate(ETensorDType::BF16, "gradient", {V, C});
    std::vector<Tensor> host_tokens, upstream;
    std::vector<std::vector<float>> values(kMicroSteps, std::vector<float>(T * C));
    for (int j = 0; j < kMicroSteps; ++j) {
        host_tokens.push_back(allocator->allocate(ETensorDType::INT32, "host_tokens", EAllocationType::PINNED, {1, T}));
        upstream.push_back(allocator->allocate(ETensorDType::BF16, "upstream", {1, T, C}));
        std::vector<nv_bfloat16> bf16(T * C);
        for (int i = 0; i < T * C; ++i) {
            values[j][i] = ((i / C + 2 * j) % 3 + i % 7 - 3) * 0.0625f;
            bf16[i] = nv_bfloat16(values[j][i]);
        }
        REQUIRE(cudaMemcpy(upstream[j].Data, bf16.data(), upstream[j].bytes(), cudaMemcpyHostToDevice) == cudaSuccess);
    }

    std::mt19937 rng(7);
    auto fill_tokens = [&]() {
        for (auto& tokens : host_tokens) {
            for (int t = 0; t < T; ++t)
                tokens.get<int>()[t] = static_cast<int>(rng() % V);
        }
    };
    // Each micro-step adds its sum to the gradient, which holds bf16: the values are multiples of
    // 1/16 small enough that every sum is exact.
    auto expected_gradient = [&]() {
        std::vector<float> expected(V * C, 0.0f);
        for (int j = 0; j < kMicroSteps; ++j) {
            std::vector<float> step(V * C, 0.0f);
            for (int t = 0; t < T; ++t) {
                for (int c = 0; c < C; ++c)
                    step[host_tokens[j].get<int>()[t] * C + c] += values[j][t * C + c];
            }
            for (int i = 0; i < V * C; ++i)
                expected[i] = float(nv_bfloat16(expected[i] + step[i]));
        }
        return expected;
    };

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    fill_tokens();
    cudaGraph_t graph = nullptr;
    REQUIRE(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
    for (int j = 0; j < kMicroSteps; ++j) {
        REQUIRE(cudaMemcpyAsync(inputs.Data, host_tokens[j].Data, inputs.bytes(), cudaMemcpyHostToDevice, stream) ==
                cudaSuccess);
        encoder_backward(gradient, scratch, upstream[j], inputs, 1, T, C, 0, stream);
    }
    REQUIRE(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
    cudaGraphExec_t exec = nullptr;
    REQUIRE(cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess);

    // The first replay runs the captured tokens; the later ones run tokens written after the capture.
    for (int replay = 0; replay < 3; ++replay) {
        INFO("replay " << replay);
        if (replay > 0) fill_tokens();
        REQUIRE(cudaMemsetAsync(gradient.Data, 0, gradient.bytes(), stream) == cudaSuccess);
        REQUIRE(cudaGraphLaunch(exec, stream) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        std::vector<nv_bfloat16> actual(V * C);
        REQUIRE(cudaMemcpy(actual.data(), gradient.Data, gradient.bytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
        const auto expected = expected_gradient();
        int wrong = 0;
        for (int i = 0; i < V * C; ++i)
            wrong += float(actual[i]) != expected[i];
        REQUIRE(wrong == 0);
    }
    REQUIRE(cudaGraphExecDestroy(exec) == cudaSuccess);
    REQUIRE(cudaGraphDestroy(graph) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}
