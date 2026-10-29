// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

// A tied lm_head is the embedding in full training too (fork ledger T10): the parameter store, or the weight manager
// when there is one, gives the head the embedding's storage, checkpoints hold it once, and under cpu_training both
// names stream one host gradient. Built into dsl-run-state-tests by fork/native/cmake/fork_native.cmake.
#include <catch2/catch_test_macros.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "config/pretrained_config.h"
#include "runtime/dsl/dsl_grad_store.h"
#include "runtime/dsl/dsl_param_store.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/dsl/ir.h"
#include "runtime/lora/lora_config.h"
#include "runtime/optimizers/cpu_adamw.h"
#include "runtime/training/runtime_options.h"
#include "utilities/allocator.h"

namespace {

constexpr long kVocab = 64;
constexpr long kWidth = 32;

void require_cuda() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) SKIP("CUDA device required");
}

dsl::TensorInfo param(std::vector<long> shape) {
    dsl::TensorInfo info;
    for (long d : shape)
        info.shape.push_back(dsl::Dim::concrete(d));
    info.dtype = ETensorDType::BF16;
    info.is_param = true;
    return info;
}

// The model-level parameters of a decoder (the names every DSL model compiles to) plus one layer weight, so that
// cpu_training has a layer gradient to stream.
dsl::Graph graph(long head_width = kWidth, bool with_embedding = true) {
    dsl::Graph g;
    if (with_embedding) g.params.emplace("embedding", param({kVocab, kWidth}));
    g.params.emplace("lm_head", param({kVocab, head_width}));
    g.params.emplace("final_norm", param({kWidth}));
    g.params.emplace("blocks[0].mlp_down_weight", param({kWidth, kWidth}));
    return g;
}

PretrainedConfig config(bool tied) {
    PretrainedConfig c;
    c.TiedWordEmbeddings = tied;
    c.DType = ETensorDType::BF16;
    c.NumLayers = 1;
    c.HiddenSize = kWidth;
    c.VocabSize = kVocab;
    return c;
}

modules::ModularLoRAConfig lora() {
    modules::ModularLoRAConfig c;
    c.rank = 8;
    return c;
}

std::set<std::string> saved(ITensorContainer& container) {
    std::set<std::string> names;
    container.iterate_tensors([&](std::string name, const TensorShard&) { names.insert(std::move(name)); });
    return names;
}

const std::set<std::string> kWithoutHead{"blocks[0].mlp_down_weight", "embedding", "final_norm"};

}  // namespace

TEST_CASE("Full training gives a tied head the embedding's storage", "[tied-head][gpu]") {
    require_cuda();
    auto allocator = std::make_shared<TensorAllocator>();
    dsl::Module module;
    const dsl::Graph g = graph();
    RuntimeOptions options;
    const PretrainedConfig cfg = config(/*tied=*/true);

    dsl::DslParamStore store(module, g, options, cfg, allocator);
    REQUIRE(store.is_storage_alias("lm_head"));
    CHECK(store.is_trainable("lm_head"));  // it keeps its gradient key, which is the embedding's buffer
    CHECK(store.get("lm_head").Data == store.get("embedding").Data);
    CHECK(saved(store) == kWithoutHead);  // a checkpoint holds the matrix once

    dsl::DslGradStore grads(store, allocator, false, EAllocationType::PINNED, 1, cfg.TiedWordEmbeddings);
    bool accumulate = false;
    Tensor* head_grad = grads.get_param_grad("lm_head", accumulate);
    Tensor* embedding_grad = grads.get_param_grad("embedding", accumulate);
    REQUIRE(head_grad);
    REQUIRE(embedding_grad);
    CHECK(head_grad->Data == embedding_grad->Data);
}

TEST_CASE("A frozen LoRA base keeps its tied alias and an untied head its own storage", "[tied-head][gpu]") {
    require_cuda();
    auto allocator = std::make_shared<TensorAllocator>();
    dsl::Module module;
    const dsl::Graph g = graph();
    RuntimeOptions options;
    const auto adapter = lora();

    dsl::DslParamStore frozen(module, g, options, config(true), allocator, &adapter);
    CHECK(frozen.is_storage_alias("lm_head"));
    CHECK_FALSE(frozen.is_trainable("lm_head"));
    CHECK(frozen.get("lm_head").Data == frozen.get("embedding").Data);

    dsl::DslParamStore untied(module, g, options, config(false), allocator);
    CHECK_FALSE(untied.is_storage_alias("lm_head"));
    CHECK(untied.get("lm_head").Data != untied.get("embedding").Data);
    CHECK(saved(untied).count("lm_head") == 1);
}

TEST_CASE("A tied head that cannot be the embedding is refused at setup", "[tied-head][gpu]") {
    require_cuda();
    auto allocator = std::make_shared<TensorAllocator>();
    dsl::Module module;
    RuntimeOptions options;
    const auto adapter = lora();
    // Another shape: neither full training nor LoRA may alias it.
    CHECK_THROWS_AS(dsl::DslParamStore(module, graph(kWidth / 2), options, config(true), allocator),
                    std::runtime_error);
    CHECK_THROWS_AS(dsl::DslParamStore(module, graph(kWidth / 2), options, config(true), allocator, &adapter),
                    std::runtime_error);
    // No 'embedding' parameter to share: full training would leave the head at its initial value.
    CHECK_THROWS_AS(
        dsl::DslParamStore(module, graph(kWidth, /*with_embedding=*/false), options, config(true), allocator),
        std::runtime_error);
}

TEST_CASE("The weight manager gives a tied head the embedding's master and work copy", "[tied-head][gpu]") {
    require_cuda();
    struct Case {
        const char* name;
        std::optional<ETensorDType> master_dtype;
        bool offload_master;
        bool cpu_training;
    };
    for (const Case& c : {Case{"fp32 masters", ETensorDType::FP32, false, false},
                          Case{"offload_master", std::nullopt, true, false},
                          Case{"cpu_training", ETensorDType::FP32, true, true}}) {
        INFO(c.name);
        auto allocator = std::make_shared<TensorAllocator>();
        dsl::Module module;
        const dsl::Graph g = graph();
        RuntimeOptions options;
        options.MasterDType = c.master_dtype;
        options.OffloadMaster = c.offload_master;
        options.CpuTraining = c.cpu_training;
        const PretrainedConfig cfg = config(true);

        dsl::DslParamStore store(module, g, options, cfg, allocator, nullptr, nullptr, /*use_weight_manager=*/true);
        dsl::DslWeightManager manager(module, g, options, cfg, allocator);
        REQUIRE_NOTHROW(store.set_weight_manager(&manager));
        REQUIRE(manager.is_tied_alias("lm_head"));
        CHECK_FALSE(manager.is_tied_alias("embedding"));
        CHECK(manager.get_master("lm_head").Data == manager.get_master("embedding").Data);
        CHECK(manager.get("lm_head").Data == manager.get("embedding").Data);
        CHECK(store.get("lm_head").Data == store.get("embedding").Data);
        CHECK(saved(manager) == kWithoutHead);
        CHECK(saved(store) == kWithoutHead);
    }
}

TEST_CASE("A LoRA weight manager keeps two frozen copies, and a mismatch with the store is refused",
          "[tied-head][gpu]") {
    require_cuda();
    auto allocator = std::make_shared<TensorAllocator>();
    dsl::Module module;
    const dsl::Graph g = graph();
    RuntimeOptions options;
    options.MasterDType = ETensorDType::FP32;
    const PretrainedConfig cfg = config(true);
    const auto adapter = lora();

    dsl::DslParamStore frozen(module, g, options, cfg, allocator, &adapter, nullptr, true);
    dsl::DslWeightManager frozen_manager(module, g, options, cfg, allocator, &adapter);
    REQUIRE_NOTHROW(frozen.set_weight_manager(&frozen_manager));
    CHECK_FALSE(frozen_manager.is_tied_alias("lm_head"));
    CHECK(frozen_manager.get_master("lm_head").Data != frozen_manager.get_master("embedding").Data);

    // A full-training store wired to a manager that keeps two heads would read the stale one.
    dsl::DslParamStore full(module, g, options, cfg, allocator, nullptr, nullptr, true);
    CHECK_THROWS_AS(full.set_weight_manager(&frozen_manager), std::logic_error);
}

TEST_CASE("cpu_training streams a tied gradient through one host copy, counted once", "[tied-head][gpu]") {
    require_cuda();
    auto allocator = std::make_shared<TensorAllocator>();
    dsl::Module module;
    const dsl::Graph g = graph();
    RuntimeOptions options;
    const PretrainedConfig cfg = config(true);
    dsl::DslParamStore store(module, g, options, cfg, allocator);

    dsl::DslGradStore grads(store,
                            allocator,
                            /*offload_grads=*/true,
                            EAllocationType::PINNED,
                            1,
                            true,
                            std::nullopt,
                            /*cpu_training=*/true);
    dsl::DslGradStoreConfig grad_config;
    grad_config.num_layers = cfg.NumLayers;
    grads.configure(grad_config);
    grads.enable_streaming(store);

    const Tensor& head = grads.get_cpu_grad("lm_head");
    const Tensor& embedding = grads.get_cpu_grad("embedding");
    REQUIRE(head.Data == embedding.Data);
    CHECK(grads.get_cpu_grad("final_norm").Data != embedding.Data);

    for (const auto& [name, host] : grads.get_cpu_grads_map())
        std::memset(host.Data, 0, host.bytes());
    auto* values = reinterpret_cast<nv_bfloat16*>(embedding.Data);
    for (std::size_t i = 0; i < embedding.nelem(); ++i)
        values[i] = __float2bfloat16(1.0f);
    // The device norm skips a gradient pointer it has seen; the host norm now sees one pointer as well.
    CHECK(optimizers::cpu_gradient_norm_squared(grads.get_cpu_grads_map(), grads.param_names()) ==
          static_cast<double>(embedding.nelem()));
}
