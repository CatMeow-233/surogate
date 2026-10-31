"""Ledger topic T26 (the MoE FP8 weight cache refuses trainable expert weights), runnable without CUDA.

With SUROGATE_FP8_MOE_WEIGHT_CACHE set, CompiledExecutor::attach_moe_fp8_cache gives an MoE grouped GEMM a
per-weight FP8 image that FP8HybridRecipe fills once (`initialized`) and that only a QLoRA selective refresh resets
(invalidate_moe_fp8_cache). The dense FP8/FP4 caches refuse trainable weights (graph_executor_weight_cache.cpp); this
one did not, so an expert weight the optimizer updates would keep being multiplied by its first FP8 image. Today the
only op that fills or reads the cache is the generic moe_grouped_gemm (Nemotron-H experts_up), whose weight gets no
gradient, so the visible effect is limited to decoupled weight decay (the cache keeps W0 while the master shrinks);
the guard makes the cache safe for any trainable expert weight.

The probe compiles attach_moe_fp8_cache's body, parse_mib_env, moe_fp8_cache_entry_bytes and MoEFP8WeightCacheEntry
as they stand in the sources, against host stubs for Tensor, the allocator and DslParamStore, and checks which weights
get a cache entry. On upstream's sources (488e5f00) the trainable-weight tests fail (the entry is allocated and
attached). The GPU behaviour is unverified here; the errata section for T26 gives the 5090 check.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
COMPILED_OPS = REPO / "csrc/src/runtime/executor/compiled_ops.cpp"
INTERNAL_H = REPO / "csrc/src/runtime/executor/graph_executor_internal.h"


def strip_comments(text: str) -> str:
    out, i = [], 0
    while i < len(text):
        two = text[i : i + 2]
        if two == "//":
            j = text.find("\n", i)
            i = len(text) if j < 0 else j
            continue
        if two == "/*":
            i = text.index("*/", i) + 2
            out.append(" ")
            continue
        c = text[i]
        if c in "\"'":
            j = i + 1
            while text[j] != c:
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def definition(path: Path, signature: str) -> tuple[str, str]:
    """(header through the opening brace, body) of the first definition matching `signature`, comments removed."""
    text = strip_comments(path.read_text(encoding="utf-8"))
    match = re.search(signature + r"[^;{]*\{", text)
    assert match, f"{path.relative_to(REPO)}: no definition matching {signature!r}"
    depth, start = 1, match.end()
    for end in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            return text[match.start() : start], text[start:end]
    raise AssertionError(f"{path.relative_to(REPO)}: unbalanced braces after {signature!r}")


def free_function(signature: str) -> str:
    head, body = definition(COMPILED_OPS, signature)
    return head + body + "}\n"


def cache_entry_struct() -> str:
    text = strip_comments(INTERNAL_H.read_text(encoding="utf-8"))
    match = re.search(r"struct MoEFP8WeightCacheEntry\s*\{[^}]*\};", text)
    assert match, "graph_executor_internal.h: no MoEFP8WeightCacheEntry"
    return match.group(0)


PROBE = r"""
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

enum class ETensorDType { FP8_E4M3, FP32 };
enum class EAllocationType { ON_DEVICE };
inline std::size_t get_dtype_size(ETensorDType t) { return t == ETensorDType::FP32 ? 4 : 1; }
struct Tensor {
    void* Data = nullptr;
    std::size_t nbytes = 0;
    std::size_t bytes() const { return nbytes; }
};
struct TensorAllocator {
    int calls = 0;
    std::vector<std::vector<char>> blocks;
    Tensor allocate(ETensorDType t, const char*, EAllocationType, const std::vector<long>& shape) {
        ++calls;
        std::size_t n = get_dtype_size(t);
        for (long d : shape) n *= static_cast<std::size_t>(d);
        blocks.emplace_back(n);
        return Tensor{blocks.back().data(), n};
    }
};
struct RunState { TensorAllocator* Allocator = nullptr; };
struct DslParamStore {
    std::map<std::string, bool> trainable;
    bool has(const std::string& n) const { return trainable.count(n) != 0; }
    bool is_trainable(const std::string& n) const { auto it = trainable.find(n); return it != trainable.end() && it->second; }
    bool work_is_transient(const std::string&) const { return false; }
};
namespace modules {
struct MoeMatmulContext {
    int num_experts = 0, N = 0, K = 0;
    Tensor* cached_moe_weights_e4m3 = nullptr;
    Tensor* cached_moe_weight_amax = nullptr;
    Tensor* cached_moe_weight_scales = nullptr;
    bool* cached_moe_weights_initialized = nullptr;
};
}

@ENTRY@
@PARSE_MIB@
@ENTRY_BYTES@

struct CompiledExecutor {
    DslParamStore& mWeights;
    RunState& mRunState;
    std::unordered_map<std::string, MoEFP8WeightCacheEntry>* mMoEFP8Cache = nullptr;
    std::size_t mMoEFP8CacheBytes = 0;
    std::optional<std::size_t> mMoEFP8CacheBudgetBytes;
    void attach_moe_fp8_cache(modules::MoeMatmulContext& ctx, const std::string& weight_name) {
        @BODY@
    }
};

// argv: weight name, "trainable" | "frozen" | "absent"; prints "<attached> <cache entries> <allocations>"
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    TensorAllocator alloc;
    RunState rs{&alloc};
    DslParamStore params;
    if (std::strcmp(argv[2], "absent") != 0) params.trainable[argv[1]] = std::strcmp(argv[2], "trainable") == 0;
    std::unordered_map<std::string, MoEFP8WeightCacheEntry> cache;
    CompiledExecutor ex{params, rs, &cache};
    modules::MoeMatmulContext ctx;
    ctx.num_experts = 4; ctx.N = 64; ctx.K = 32;
    ex.attach_moe_fp8_cache(ctx, argv[1]);
    ex.attach_moe_fp8_cache(ctx, argv[1]);  // forward, then the backward of the same op
    std::printf("%d %zu %d\n", ctx.cached_moe_weights_e4m3 != nullptr, cache.size(), alloc.calls);
    return 0;
}
"""


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        pytest.skip("no C++ compiler")
    out = tmp_path_factory.mktemp("t26")
    _, body = definition(COMPILED_OPS, r"void CompiledExecutor::attach_moe_fp8_cache\(")
    src = out / "probe.cpp"
    src.write_text(
        PROBE.replace("@ENTRY@", cache_entry_struct())
        .replace("@PARSE_MIB@", free_function(r"std::size_t parse_mib_env\("))
        .replace("@ENTRY_BYTES@", free_function(r"std::size_t moe_fp8_cache_entry_bytes\("))
        .replace("@BODY@", body),
        encoding="utf-8",
    )
    exe = out / "probe"
    result = subprocess.run([cxx, "-std=c++20", "-O0", str(src), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, f"probe does not compile:\n{result.stderr}"
    return exe


def attach(exe: Path, name: str, kind: str, env_value: str | None = "1") -> tuple[bool, int, int]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SUROGATE_FP8_MOE")}
    if env_value is not None:
        env["SUROGATE_FP8_MOE_WEIGHT_CACHE"] = env_value
    result = subprocess.run([str(exe), name, kind], capture_output=True, text=True, env=env, check=True)
    attached, entries, allocations = result.stdout.split()
    return attached == "1", int(entries), int(allocations)


@pytest.mark.parametrize("name", ["blocks[0].experts_up", "blocks[3].experts_gate_up", "blocks[3].experts_down"])
def test_a_trainable_expert_weight_gets_no_fp8_image(probe, name):
    # Full fine-tuning: the optimizer updates the weight, so a quantize-once image would go stale after one step.
    assert attach(probe, name, "trainable") == (False, 0, 0)


@pytest.mark.parametrize("name", ["blocks[0].experts_up", "blocks[3].experts_gate_up"])
def test_a_frozen_expert_weight_keeps_the_cache(probe, name):
    # LoRA/QLoRA: experts are frozen; the cache is allocated once (three tensors) and reused by the backward.
    assert attach(probe, name, "frozen") == (True, 1, 3)


def test_names_outside_the_parameter_store_behave_as_before(probe):
    assert attach(probe, "llep_merged_experts", "absent") == (True, 1, 3)


@pytest.mark.parametrize("env_value", [None, "0"])
def test_the_cache_stays_opt_in(probe, env_value):
    assert attach(probe, "blocks[0].experts_up", "frozen", env_value) == (False, 0, 0)


def test_the_guard_runs_before_anything_is_allocated():
    _, body = definition(COMPILED_OPS, r"void CompiledExecutor::attach_moe_fp8_cache\(")
    flat = " ".join(body.split())
    guard = flat.find("if (mWeights.is_trainable(weight_name)) { return; }")
    assert guard >= 0, "attach_moe_fp8_cache: no trainable-weight guard"
    assert guard < flat.find("mMoEFP8Cache->find(weight_name)") < flat.find("Allocator->allocate(")
