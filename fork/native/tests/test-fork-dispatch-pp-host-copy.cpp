// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

// Batched host reads of named tensors for dispatch-PP boundaries (fork ledger T1).
// Built into dsl-run-state-tests by fork/native/cmake/fork_native.cmake.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/executor/graph_executor_utils.h"
#include "utilities/allocator.h"

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
