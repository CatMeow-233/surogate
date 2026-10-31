// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// CPU-only checks of fork ledger T21 (checkpoints of the cpu_training AdamW moments). Built and run by
// fork/patches/tests/native/test_cpu_adamw_checkpoint.py with the host compiler against the CUDA headers and
// libcudart, from the production sources: csrc/src/utilities/safetensors.cpp (SafeTensorWriter/Reader),
// cu_file_fallback.cpp and cu_file_common.cpp (the reader's pread path for host targets), tensor.cpp, dtype.cpp,
// utils.cpp and fork/native/src/cpu_adamw_checkpoint.cpp. No GPU is needed: on a host without a driver every CUDA
// call fails, so a check that passes made none (upstream's writer threw cudaErrorInsufficientDriver at its
// cudaMemcpy for a host tensor). Each check prints "ok <name>" or "FAIL <name>: <why>"; the exit status is the
// number of failures.
//
// What AdamW8BitOptimizer does with this module (layout at the first step or at a resume, the containers it
// returns, begin_load/finish_load around the reads) is pinned in the Python test; the GPU behaviour is checked on
// a 5090 (fork/docs/upstream-errata.md).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "surogate_fork/cpu_adamw_checkpoint.h"
#include "surogate_fork/pageable_host.h"
#include "utilities/safetensors.h"
#include "utilities/tensor.h"
#include "utilities/tensor_container.h"

namespace ck = surogate_fork::cpu_adamw_checkpoint;

namespace {

int g_failures = 0;
std::string g_dir;

void report(const char* name, bool ok, const std::string& why = {}) {
    if (ok) {
        std::printf("ok %s\n", name);
    } else {
        std::printf("FAIL %s: %s\n", name, why.c_str());
        ++g_failures;
    }
    std::fflush(stdout);
}

//! Runs `body`; an exception fails the check.
void check(const char* name, const std::function<std::string()>& body) {
    try {
        const std::string why = body();
        report(name, why.empty(), why);
    } catch (const std::exception& e) {
        report(name, false, std::string("threw: ") + e.what());
    }
}

//! The message `body` throws, or "" when it does not throw.
std::string thrown(const std::function<void()>& body) {
    try {
        body();
    } catch (const std::exception& e) {
        return e.what() + std::string(" ");  // never empty
    }
    return {};
}

bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

struct EnvGuard {
    std::string name;
    explicit EnvGuard(const char* n, const char* value)
        : name(n) {
        if (value) {
            setenv(n, value, 1);
        } else {
            unsetenv(n);
        }
    }
    ~EnvGuard() {
        unsetenv(name.c_str());
    }
};

//! A host master: storage plus the Tensor the weight manager would hand out.
struct Master {
    std::vector<float> storage;
    Tensor tensor;
    Master(std::vector<long> shape) {
        long n = 1;
        for (long d : shape)
            n *= d;
        storage.assign(static_cast<std::size_t>(n), 0.0f);
        tensor = Tensor::from_pointer(reinterpret_cast<std::byte*>(storage.data()), -1, ETensorDType::FP32, shape);
    }
};

//! A model's trainable parameters in param_names() order. lm_head shares the embedding's master (tied, T10).
struct Model {
    Master embedding{{11, 4}};
    Master q{{4, 4}};
    Master norm{{4}};
    Master extra{{3, 2}};
    std::vector<ck::Param> params(bool with_extra = false) const {
        std::vector<ck::Param> p = {{"embedding", &embedding.tensor},
                                    {"blocks[0].q_weight", &q.tensor},
                                    {"blocks[0].ln1_weight", &norm.tensor},
                                    {"lm_head", &embedding.tensor}};
        if (with_extra) p.push_back({"blocks[0].extra", &extra.tensor});
        return p;
    }
};

//! Upstream's step_cpu_streaming layout before T21 (names with the T10 dedupe, offsets in that order).
std::vector<std::pair<std::string, std::size_t>> upstream_layout(const std::vector<ck::Param>& params) {
    std::vector<std::string> names;
    std::unordered_set<const std::byte*> stepped;
    for (const auto& p : params) {
        const Tensor& master = *p.master;
        if (master.Data && !stepped.insert(master.Data).second) continue;
        names.push_back(p.name);
    }
    std::vector<std::pair<std::string, std::size_t>> out;
    std::size_t offset = 0;
    for (const auto& name : names) {
        for (const auto& p : params) {
            if (p.name == name) {
                out.emplace_back(name, offset);
                offset += p.master->nelem();
                break;
            }
        }
    }
    return out;
}

//! Distinct bit patterns, special values included, so a misplaced or converted element shows.
void fill(std::vector<float>& data, std::uint32_t seed) {
    for (std::size_t i = 0; i < data.size(); ++i) {
        std::uint32_t bits = seed * 2654435761u + static_cast<std::uint32_t>(i) * 40503u;
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value)) value = static_cast<float>(i) * 0.25f - 1.0f;
        data[i] = value;
    }
    if (data.size() > 3) {
        data[0] = -0.0f;
        data[1] = 1e-45f;  // denormal
        data[2] = std::nanf("0x1234");
    }
}

nlohmann::json header_of(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::uint64_t size = 0;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::string text(size, '\0');
    file.read(text.data(), static_cast<std::streamsize>(size));
    auto header = nlohmann::json::parse(text);
    header.erase("__metadata__");
    return header;
}

//! A container of arbitrary named host tensors, for files the fork container would not write.
struct Raw final : ITensorContainer {
    std::map<std::string, Tensor> tensors;
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override {
        for (auto& [name, tensor] : tensors)
            callback(name, TensorShard(tensor));
    }
};

//! A saved run: a state over the model's params, filled, and written to <dir>/<tag>.adam.{m,v}.safetensors.
struct Saved {
    optimizers::CPUAdamWState state;
    ck::CpuAdamWCheckpoint checkpoint;
    std::string m_file, v_file;
    Saved(const Model& model, const std::string& tag, bool with_extra = false) {
        checkpoint.initialize(state, model.params(with_extra));
        fill(state.m, 1);
        fill(state.v, 2);
        m_file = g_dir + "/" + tag + ".adam.m.safetensors";
        v_file = g_dir + "/" + tag + ".adam.v.safetensors";
        write_safetensors(m_file, *checkpoint.momentum());
        write_safetensors(v_file, *checkpoint.variance());
    }
};

//! What load_checkpoint does for the optimizer files after T21, against a fresh optimizer state.
void resume(ck::CpuAdamWCheckpoint& checkpoint, const std::string& m_file, const std::string& v_file) {
    checkpoint.begin_load();
    ck::require_complete(m_file, *checkpoint.momentum());
    load_safetensors(m_file, *checkpoint.momentum(), false);
    ck::require_complete(v_file, *checkpoint.variance());
    load_safetensors(v_file, *checkpoint.variance(), false);
    checkpoint.finish_load();
}

}  // namespace

int run_checks();

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <scratch directory>\n", argv[0]);
        return 2;
    }
    g_dir = argv[1];
    try {
        return run_checks();
    } catch (const std::exception& e) {
        report("setup", false, std::string("threw: ") + e.what());
        return g_failures;
    }
}

int run_checks() {
    unsetenv(ck::kSaveStateEnv);
    unsetenv(ck::kAllowMissingStateEnv);
    const Model model;

    check("layout_matches_upstream_step_order_and_steps_a_tied_master_once", [&] {
        const auto slots = ck::build_layout(model.params());
        const auto reference = upstream_layout(model.params());
        if (slots.size() != reference.size()) return std::string("slot count differs from upstream's names");
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (slots[i].name != reference[i].first || slots[i].offset != reference[i].second) {
                return "slot " + std::to_string(i) + " is " + slots[i].name + "@" + std::to_string(slots[i].offset);
            }
            if (slots[i].n != static_cast<std::size_t>(model.params()[i].master->nelem())) {
                return "slot " + slots[i].name + " has the wrong size";
            }
        }
        for (const auto& slot : slots) {
            if (slot.name == "lm_head") return std::string("the tied lm_head got a slot of its own");
        }
        if (slots[0].shape != std::vector<long>{11, 4}) return std::string("embedding shape lost");
        return std::string();
    });

    check("initialize_allocates_zeroed_moments_once", [&] {
        optimizers::CPUAdamWState state;
        ck::CpuAdamWCheckpoint checkpoint;
        if (checkpoint.active()) return std::string("active before initialize");
        checkpoint.initialize(state, model.params());
        const std::size_t total = 11 * 4 + 16 + 4;
        if (!checkpoint.active() || !state.initialized || state.total_params != total || state.m.size() != total ||
            state.v.size() != total) {
            return std::string("state not sized to the layout");
        }
        for (std::size_t i = 0; i < total; ++i) {
            if (state.m[i] != 0.0f || state.v[i] != 0.0f) return std::string("moments not zero-filled");
        }
        state.m[0] = 3.0f;
        checkpoint.initialize(state, model.params(true));  // a second call changes nothing
        if (state.m.size() != total || state.m[0] != 3.0f) return std::string("second initialize reallocated");
        return std::string();
    });

    check("pageable_host_memory_is_recognized", [&] {
        std::vector<float> host(4);
        if (!surogate_fork::is_pageable_host(host.data(), -1)) return std::string("std::vector not pageable");
        if (surogate_fork::is_pageable_host(host.data(), 0)) return std::string("Device 0 taken for host memory");
        return std::string();
    });

    check("save_writes_one_fp32_tensor_per_master_and_the_format_tag", [&] {
        Saved saved(model, "layout");
        for (const auto& [file, prefix] : {std::pair{saved.m_file, std::string(ck::kMomentPrefix)},
                                           std::pair{saved.v_file, std::string(ck::kVariancePrefix)}}) {
            const auto header = header_of(file);
            const std::map<std::string, std::vector<long>> expected = {{prefix + "embedding", {11, 4}},
                                                                       {prefix + "blocks[0].q_weight", {4, 4}},
                                                                       {prefix + "blocks[0].ln1_weight", {4}},
                                                                       {ck::kFormatTensor, {1}}};
            if (header.size() != expected.size()) return file + " has " + std::to_string(header.size()) + " entries";
            for (const auto& [name, shape] : expected) {
                if (!header.contains(name)) return file + " lacks " + name;
                const std::string dtype = name == ck::kFormatTensor ? "I32" : "F32";
                if (header[name]["dtype"] != dtype) return name + " is " + header[name]["dtype"].dump();
                if (header[name]["shape"].get<std::vector<long>>() != shape) return name + " has another shape";
            }
        }
        return std::string();
    });

    check("resume_round_trips_the_moments_bit_for_bit", [&] {
        Saved saved(model, "roundtrip");
        optimizers::CPUAdamWState state;
        ck::CpuAdamWCheckpoint checkpoint;
        checkpoint.initialize(state, model.params());
        resume(checkpoint, saved.m_file, saved.v_file);
        if (state.m.size() != saved.state.m.size()) return std::string("size differs");
        if (std::memcmp(state.m.data(), saved.state.m.data(), state.m.size() * sizeof(float)) != 0) {
            return std::string("m differs");
        }
        if (std::memcmp(state.v.data(), saved.state.v.data(), state.v.size() * sizeof(float)) != 0) {
            return std::string("v differs");
        }
        // The restored state saves again, unchanged (a second resume reads the same moments).
        const std::string again = g_dir + "/roundtrip.again.adam.m.safetensors";
        write_safetensors(again, *checkpoint.momentum());
        if (header_of(again) != header_of(saved.m_file)) return std::string("re-save writes another header");
        return std::string();
    });

    // A cpu_training checkpoint written before T21: adam.m/adam.v with no tensors (no step ever called init_state).
    Raw nothing;
    const std::string empty_file = g_dir + "/legacy-empty.safetensors";
    write_safetensors(empty_file, nothing);
    // One written after an earlier upstream resume: the unused 8-bit device state's names and nothing else.
    std::vector<std::byte> junk(64, std::byte{0});
    std::vector<std::uint16_t> junk_scales(2, 0);
    Raw device_state;
    device_state.tensors.emplace("adamw8bit.state1",
                                 Tensor::from_pointer(junk.data(), -1, ETensorDType::BYTE, std::array<long, 1>{64}));
    device_state.tensors.emplace("adamw8bit.scales1",
                                 Tensor::from_pointer(reinterpret_cast<std::byte*>(junk_scales.data()),
                                                      -1,
                                                      ETensorDType::FP16,
                                                      std::array<long, 1>{2}));
    const std::string junk_file = g_dir + "/legacy-8bit.safetensors";
    write_safetensors(junk_file, device_state);

    check("negative_control_upstream_reader_skips_the_missing_moments_silently", [&] {
        optimizers::CPUAdamWState state;
        ck::CpuAdamWCheckpoint checkpoint;
        checkpoint.initialize(state, model.params());
        state.m.assign(state.m.size(), 42.0f);
        checkpoint.begin_load();
        const std::string error = thrown([&] { load_safetensors(empty_file, *checkpoint.momentum(), false); });
        if (!error.empty()) return "the reader threw: " + error;
        if (state.m[0] != 42.0f) return std::string("the reader wrote into the moments");
        return std::string();  // the defect T21 guards against: nothing read, nothing said
    });

    for (const auto& [name, file] : {std::pair{"legacy_header_only_file_is_refused", empty_file},
                                     std::pair{"legacy_file_with_unused_8bit_state_is_refused", junk_file}}) {
        check(name, [&, file = file] {
            optimizers::CPUAdamWState state;
            ck::CpuAdamWCheckpoint checkpoint;
            checkpoint.initialize(state, model.params());
            checkpoint.begin_load();
            const std::string error = thrown([&] { ck::require_complete(file, *checkpoint.momentum()); });
            for (const char* part : {"lacks 4 of the 4 CPU AdamW tensors",
                                     "cpu_adamw.format",
                                     "holds no CPU AdamW state",
                                     "SUROGATE_FORK_CPU_ADAMW_ALLOW_MISSING_STATE=1"}) {
                if (!contains(error, part)) return "message lacks '" + std::string(part) + "': " + error;
            }
            return std::string();
        });
    }

    check("legacy_file_resumes_with_zero_moments_when_opted_in", [&] {
        EnvGuard allow(ck::kAllowMissingStateEnv, "1");
        optimizers::CPUAdamWState state;
        ck::CpuAdamWCheckpoint checkpoint;
        checkpoint.initialize(state, model.params());
        resume(checkpoint, empty_file, junk_file);
        for (std::size_t i = 0; i < state.m.size(); ++i) {
            if (state.m[i] != 0.0f || state.v[i] != 0.0f) return std::string("moments not zero");
        }
        // After the opted-in resume the state saves as v1.
        const std::string out = g_dir + "/opted-in.adam.m.safetensors";
        write_safetensors(out, *checkpoint.momentum());
        if (!header_of(out).contains(ck::kFormatTensor)) return std::string("re-save lacks the format tag");
        return std::string();
    });

    check("allow_missing_is_off_for_0_and_empty", [&] {
        for (const char* value : {"0", ""}) {
            EnvGuard allow(ck::kAllowMissingStateEnv, value);
            if (ck::missing_state_allowed()) return "allowed with '" + std::string(value) + "'";
        }
        return std::string();
    });

    check("finish_load_refuses_files_read_without_require_complete", [&] {
        optimizers::CPUAdamWState state;
        ck::CpuAdamWCheckpoint checkpoint;
        checkpoint.initialize(state, model.params());
        checkpoint.begin_load();
        load_safetensors(empty_file, *checkpoint.momentum(), false);
        load_safetensors(empty_file, *checkpoint.variance(), false);
        const std::string error = thrown([&] { checkpoint.finish_load(); });
        if (!contains(error, "has no cpu_adamw.format")) return "not refused: " + error;
        return std::string();
    });

    check("checkpoint_of_another_parameter_set_is_refused_by_name", [&] {
        Saved saved(model, "fewer");  // without blocks[0].extra
        optimizers::CPUAdamWState state;
        ck::CpuAdamWCheckpoint checkpoint;
        checkpoint.initialize(state, model.params(/*with_extra=*/true));
        checkpoint.begin_load();
        const std::string error = thrown([&] { ck::require_complete(saved.m_file, *checkpoint.momentum()); });
        if (!contains(error, "lacks 1 of the 5") || !contains(error, "cpu_adamw.m.blocks[0].extra") ||
            !contains(error, "covers other parameters")) {
            return "unexpected: " + error;
        }
        return std::string();
    });

    check("another_format_version_is_refused_even_when_opted_in", [&] {
        optimizers::CPUAdamWState saved_state;
        ck::CpuAdamWCheckpoint saved;
        saved.initialize(saved_state, model.params());
        saved.momentum()->set_format(2);
        saved.variance()->set_format(2);
        const std::string m_file = g_dir + "/v2.adam.m.safetensors", v_file = g_dir + "/v2.adam.v.safetensors";
        write_safetensors(m_file, *saved.momentum());
        write_safetensors(v_file, *saved.variance());
        EnvGuard allow(ck::kAllowMissingStateEnv, "1");
        optimizers::CPUAdamWState state;
        ck::CpuAdamWCheckpoint checkpoint;
        checkpoint.initialize(state, model.params());
        const std::string error = thrown([&] { resume(checkpoint, m_file, v_file); });
        if (!contains(error, "format version 2")) return "not refused: " + error;
        return std::string();
    });

    check("shape_and_dtype_changes_are_refused_by_the_reader", [&] {
        optimizers::CPUAdamWState state;
        ck::CpuAdamWCheckpoint checkpoint;
        checkpoint.initialize(state, model.params());
        std::vector<float> other(12);
        std::vector<std::uint16_t> bf16(44);
        for (const auto& [tag, tensor] : {std::pair{"shape",
                                                    Tensor::from_pointer(reinterpret_cast<std::byte*>(other.data()),
                                                                         -1,
                                                                         ETensorDType::FP32,
                                                                         std::array<long, 2>{3, 4})},
                                          std::pair{"dtype",
                                                    Tensor::from_pointer(reinterpret_cast<std::byte*>(bf16.data()),
                                                                         -1,
                                                                         ETensorDType::BF16,
                                                                         std::array<long, 2>{11, 4})}}) {
            Raw file;
            file.tensors.emplace("cpu_adamw.m.embedding", tensor);
            const std::string path = g_dir + "/bad-" + std::string(tag) + ".safetensors";
            write_safetensors(path, file);
            checkpoint.begin_load();
            const std::string error = thrown([&] { load_safetensors(path, *checkpoint.momentum(), false); });
            if (error.empty()) return std::string("a ") + tag + " change loaded";
        }
        return std::string();
    });

    check("save_opt_out_writes_no_moments_but_a_load_still_reads_them", [&] {
        Saved saved(model, "optout-source");
        EnvGuard off(ck::kSaveStateEnv, "0");
        const std::string out = g_dir + "/optout.adam.m.safetensors";
        write_safetensors(out, *saved.checkpoint.momentum());
        if (!header_of(out).empty()) return std::string("moments written despite the opt-out");
        optimizers::CPUAdamWState state;
        ck::CpuAdamWCheckpoint checkpoint;
        checkpoint.initialize(state, model.params());
        resume(checkpoint, saved.m_file, saved.v_file);
        if (std::memcmp(state.m.data(), saved.state.m.data(), state.m.size() * sizeof(float)) != 0) {
            return std::string("the load skipped the moments");
        }
        return std::string();
    });

    check("other_containers_are_not_checked", [&] {
        Raw gpu_state;  // e.g. the GPU path's 8-bit containers: upstream behaviour
        std::vector<std::byte> state1(8, std::byte{0});
        gpu_state.tensors.emplace("adamw8bit.state1",
                                  Tensor::from_pointer(state1.data(), -1, ETensorDType::BYTE, std::array<long, 1>{8}));
        ck::require_complete(empty_file, gpu_state);
        return std::string();
    });

    std::printf("%d failure(s)\n", g_failures);
    return g_failures;
}
