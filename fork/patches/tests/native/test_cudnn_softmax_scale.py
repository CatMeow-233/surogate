"""Ledger topic T22 (the cuDNN SDPA backend applies the op's softmax_scale), runnable without CUDA.

A flash_attention op carries softmax_scale: 0 means the kernel default 1/sqrt(Hs), anything else is to be used as is
(graph_compiler.h). mem_eff, flash_varlen, kvprefix and custom honour it; upstream's cuDNN backend neither checked it
in supports() nor passed it on, and attention_cudnn.cpp fed cuDNN 1/sqrt(HS) in both directions. Gemma3-27B (head_dim
128, query_pre_attn_scalar 168) therefore trained its ten global layers at 128**-0.5 instead of 168**-0.5 whenever a
micro-batch had no in-row document boundary (cu_seqlens null: unpacked SFT, one document per row, eval, the decode
prefill), and at the declared scale on packed micro-batches, which go to flash_varlen.

The probe compiles, as they stand in the sources, CuDNNAttention (backend_cudnn.cpp), the BF16 cuDNN wrappers
(attention_cudnn.cpp), the Tensor and FP32 overloads (attention_custom.cu) and AttentionParams / AttentionBackend /
the priorities (attention_backend.h), against host stubs for CUDA, cuDNN and the cuDNN frontend graph. The stub
graph records the attn_scale value each execute() receives through the variant pack, which is what cuDNN multiplies
Q K^T by. The tests feed it the flash_attention ops the real DSL emits for the Gemma3-27B text geometry and require
the recorded scale to be the op's, forward and backward. On upstream's sources (488e5f00) the backend probe compiles
and the Gemma3-27B test fails with 0x1.6a09e6p-4 (1/sqrt(128)) recorded for 0x1.3c0ca4p-4 (168**-0.5); the FP32
probe does not compile there (no scale parameter). The kernels themselves are GPU-unverified here; the errata section
for T22 in fork/docs/upstream-errata.md gives the 5090 check.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[4]
ATTN = REPO / "csrc/src/runtime/attention"
BACKEND_H = ATTN / "attention_backend.h"
BACKEND_CUDNN = ATTN / "backend_cudnn.cpp"
WRAPPER_CUDNN = ATTN / "attention_cudnn.cpp"
WRAPPER_CUSTOM = ATTN / "attention_custom.cu"
FLASH_ATTENTION = ATTN / "flash_attention.cpp"

# Gemma3-27B text geometry (google/gemma-3-27b-* text_config), as a text-only Gemma3ForCausalLM export.
GEMMA3_27B_TEXT = {
    "architectures": ["Gemma3ForCausalLM"],
    "model_type": "gemma3_text",
    "hidden_size": 5376,
    "num_attention_heads": 32,
    "num_key_value_heads": 16,
    "head_dim": 128,
    "intermediate_size": 21504,
    "num_hidden_layers": 62,
    "query_pre_attn_scalar": 168,
    "sliding_window": 1024,
    "_sliding_window_pattern": 6,
    "rope_theta": 1000000.0,
    "rope_local_base_freq": 10000.0,
    "rms_norm_eps": 1e-06,
    "max_position_embeddings": 131072,
    "vocab_size": 262208,
    "hidden_activation": "gelu_pytorch_tanh",
}


def definition(path: Path, signature: str, terminator: str = "") -> str:
    """The first definition matching `signature` (a regex) through its closing brace; comments and literals skipped."""
    text = path.read_text(encoding="utf-8")
    match = re.search(signature + r"[^;{]*\{", text)
    assert match, f"{path.relative_to(REPO)}: no definition matching {signature!r}"
    depth, i = 1, match.end()
    while i < len(text):
        two = text[i : i + 2]
        if two == "//":
            i = text.index("\n", i)
            continue
        if two == "/*":
            i = text.index("*/", i) + 2
            continue
        c = text[i]
        if c in "\"'":
            j = i + 1
            while text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        depth += {"{": 1, "}": -1}.get(c, 0)
        if depth == 0:
            return text[match.start() : i + 1] + terminator
        i += 1
    raise AssertionError(f"{path.relative_to(REPO)}: unbalanced braces after {signature!r}")


def body(path: Path, signature: str) -> str:
    """The definition matching `signature`, whitespace collapsed."""
    return " ".join(definition(path, signature).split())


def line_matching(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, f"{path.relative_to(REPO)}: no line matching {pattern!r}"
    return match.group(0)


PRELUDE = r"""
#include <math.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ---- CUDA / cuDNN host stubs ---------------------------------------------------------------------------------------
using cudaStream_t = void*;
using cudnnHandle_t = struct cudnnContext*;
using cublasHandle_t = struct cublasContext*;
struct nv_bfloat16 { unsigned short x; };
enum cudaError_t { cudaSuccess = 0 };
enum cudnnStatus_t { CUDNN_STATUS_SUCCESS = 0 };
static cudaError_t cudaGetLastError() { return cudaSuccess; }
static cudnnStatus_t cudnnSetStream(cudnnHandle_t, cudaStream_t) { return CUDNN_STATUS_SUCCESS; }
#define CUDA_CHECK(x) do { if ((x) != cudaSuccess) std::abort(); } while (0)
#define cuDNNCheck(x) do { if ((x) != CUDNN_STATUS_SUCCESS) std::abort(); } while (0)
#define checkCudnnFE(x) do { if (!(x).is_good()) std::abort(); } while (0)

// Every softmax scale a kernel would apply, in call order: "<where>=<hex float>".
static std::vector<std::string> g_events;
static void record(const char* where, float scale) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "%s=%a", where, static_cast<double>(scale));
    g_events.emplace_back(buf);
}

enum class ETensorDType { FP32, BF16, BYTE };
struct Tensor {
    ETensorDType DType = ETensorDType::BF16;
    std::byte* Data = nullptr;
    long batch_start = 0;
    template <typename T> T* get() { return reinterpret_cast<T*>(Data); }
    template <typename T> const T* get() const { return reinterpret_cast<const T*>(Data); }
};
static Tensor slice(const Tensor& src, int, long start, long) {
    Tensor t = src;
    t.batch_start = start;
    return t;
}
template <class T> constexpr T div_exact(T a, T b) {
    if (a % b != 0) throw std::runtime_error("div_exact");
    return a / b;
}
std::size_t cudnn_get_workspace_size(int, int, int, int, int, cudnnHandle_t) { return 1; }

// ---- cuDNN frontend graph stub: execute() records the pass-by-value attn_scale -------------------------------------
struct FeStatus { bool is_good() const { return true; } };
struct FakeGraph {
    const char* dir;
    FeStatus execute(cudnnHandle_t, std::unordered_map<int64_t, void*>& pack, std::byte*);
};
"""

GRAPH_STUBS = r"""
FeStatus FakeGraph::execute(cudnnHandle_t, std::unordered_map<int64_t, void*>& pack, std::byte*) {
    auto it = pack.find(Attn_scale_UID);
    if (it == pack.end() || it->second == nullptr) { std::puts("FATAL attn_scale missing from the variant pack"); std::exit(3); }
    record(dir, *static_cast<const float*>(it->second));
    return {};
}
static std::shared_ptr<FakeGraph> lookup_cache_or_build_graph_fwd(int, int, int, int, int, int, cudnnHandle_t) {
    return std::make_shared<FakeGraph>(FakeGraph{"fwd"});
}
static std::shared_ptr<FakeGraph> lookup_cache_or_build_graph_bwd(int, int, int, int, int, cudnnHandle_t) {
    return std::make_shared<FakeGraph>(FakeGraph{"bwd"});
}

// ---- in-tree kernel launchers the FP32 overloads call --------------------------------------------------------------
template <class F>
cudaError_t attention_gpu_forward(F*, float*, float scale, const F*, int, int, int, int, int, int, cudaStream_t) {
    record("gpu_fwd", scale);
    return cudaSuccess;
}
template <class F>
cudaError_t attention_gpu_backward(F*, const float*, float scale, const F*, const F*, const F*, int, int, int, int, int,
                                   int, cudaStream_t) {
    record("gpu_bwd", scale);
    return cudaSuccess;
}
"""

BACKEND_MAIN = r"""
// stdin, one case per line: B T Hq Hkv Hs window causal scale(hex) varlen chunks
// stdout: "priorities cudnn=<p> mem_eff=<p> flash_varlen=<p> kvprefix=<p>", then per case
// "supports=<0|1> <events...>"
int main() {
    std::printf("priorities cudnn=%d mem_eff=%d flash_varlen=%d kvprefix=%d\n", dsl::attention_priority::kCuDNN,
                dsl::attention_priority::kMemEff, dsl::attention_priority::kFlashVarlen,
                dsl::attention_priority::kKvPrefix);
    static std::byte storage[64];
    char scale_text[64];
    int B, T, Hq, Hkv, Hs, window, causal, varlen, chunks;
    while (std::scanf("%d %d %d %d %d %d %d %63s %d %d", &B, &T, &Hq, &Hkv, &Hs, &window, &causal, scale_text, &varlen,
                      &chunks) == 10) {
        dsl::AttentionParams p;
        p.B = B; p.T = T; p.Hq = Hq; p.Hkv = Hkv; p.Hs = Hs;
        p.window_size = window;
        p.causal = causal != 0;
        p.softmax_scale = std::strtof(scale_text, nullptr);
        p.dtype = ETensorDType::BF16;
        for (Tensor* t : {&p.qkv, &p.out, &p.lse, &p.d_out, &p.d_qkv}) t->Data = storage;
        p.lse.DType = ETensorDType::FP32;
        p.cudnn_workspace.Data = storage;
        p.cudnn_workspace.DType = ETensorDType::BYTE;
        p.cudnn_handle = reinterpret_cast<cudnnHandle_t>(storage);
        static const std::int32_t cu[2] = {0, 1};
        p.cu_seqlens = varlen ? cu : nullptr;
        p.attn_bwd_chunks = chunks;
        dsl::probe::CuDNNAttention backend;
        const bool supported = backend.supports(p);
        g_events.clear();
        if (supported) {
            backend.forward(p);
            backend.backward(p);
        }
        std::printf("supports=%d", supported ? 1 : 0);
        for (const auto& e : g_events) std::printf(" %s", e.c_str());
        std::printf("\n");
    }
    return 0;
}
"""

FP32_MAIN = r"""
// stdin, one case per line: B T Hq Hkv Hs scale(hex); calls the Tensor overloads with FP32 tensors directly
// (CuDNNAttention::supports() takes BF16 only, so these are reached only by direct callers).
int main() {
    static std::byte storage[64];
    char scale_text[64];
    int B, T, Hq, Hkv, Hs;
    while (std::scanf("%d %d %d %d %d %63s", &B, &T, &Hq, &Hkv, &Hs, scale_text) == 6) {
        const float scale = std::strtof(scale_text, nullptr);
        Tensor out, stats, inp, dout, dqkv, ws;
        for (Tensor* t : {&out, &stats, &inp, &dout, &dqkv}) { t->Data = storage; t->DType = ETensorDType::FP32; }
        ws.Data = storage;
        ws.DType = ETensorDType::BYTE;
        g_events.clear();
        attention_forward_cudnn(out, stats, inp, ws, nullptr, B, T, Hq, Hkv, Hs, nullptr, scale);
        attention_backward_cudnn(dqkv, stats, out, dout, inp, ws, nullptr, B, T, Hq, Hkv, Hs, nullptr, scale);
        for (const auto& e : g_events) std::printf(" %s", e.c_str());
        std::printf("\n");
    }
    return 0;
}
"""


def probe_source(main: str) -> str:
    uids = definition(WRAPPER_CUDNN, r"enum UIDs", ";")
    parts = [
        PRELUDE,
        uids,
        GRAPH_STUBS,
        "// ---- attention_cudnn.cpp: BF16 cuDNN wrappers ----",
        definition(WRAPPER_CUDNN, r"void attention_forward_cudnn\(\s*nv_bfloat16\s*\*"),
        definition(WRAPPER_CUDNN, r"void attention_backward_cudnn\(\s*nv_bfloat16\s*\*"),
        "// ---- attention_custom.cu: FP32 and Tensor overloads ----",
        definition(WRAPPER_CUSTOM, r"void attention_forward_cudnn\(\s*float\s*\*"),
        definition(WRAPPER_CUSTOM, r"void attention_backward_cudnn\(\s*float\s*\*"),
        definition(WRAPPER_CUSTOM, r"void attention_forward_cudnn\(\s*Tensor\s*&"),
        definition(WRAPPER_CUSTOM, r"void attention_backward_cudnn\(\s*Tensor\s*&"),
        "namespace dsl {",
        "class DslRunState;",
        "// ---- attention_backend.h ----",
        definition(BACKEND_H, r"namespace attention_priority"),
        definition(BACKEND_H, r"struct AttentionParams", ";"),
        definition(BACKEND_H, r"class AttentionBackend(?=\s*\{)", ";"),
        "namespace probe {",
        "// ---- backend_cudnn.cpp ----",
        line_matching(BACKEND_CUDNN, r"^constexpr int kCuDNNMaxHeadDim = \d+;$"),
        definition(BACKEND_CUDNN, r"class CuDNNAttention final : public AttentionBackend", ";"),
        "}  // namespace probe",
        "}  // namespace dsl",
        main,
    ]
    return "\n\n".join(parts)


def build(tmp_path_factory, name: str, main: str) -> Path:
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("no C++ compiler (g++, c++ or clang++) to build the cuDNN scale probe")
    work = tmp_path_factory.mktemp(name)
    source = work / f"{name}.cpp"
    source.write_text(probe_source(main), encoding="utf-8")
    binary = work / name
    built = subprocess.run(
        [compiler, "-std=c++20", "-O1", "-Wall", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert built.returncode == 0, built.stderr[-4000:]
    return binary


@pytest.fixture(scope="module")
def backend_probe(tmp_path_factory) -> Path:
    return build(tmp_path_factory, "cudnn_scale_backend_probe", BACKEND_MAIN)


@pytest.fixture(scope="module")
def fp32_probe(tmp_path_factory) -> Path:
    return build(tmp_path_factory, "cudnn_scale_fp32_probe", FP32_MAIN)


def hexf(value: float) -> str:
    return float(np.float32(value)).hex()


def default_scale(hs: int) -> float:
    """1.f / sqrtf(HS), the kernel default, in float32."""
    return float(np.float32(1.0) / np.sqrt(np.float32(hs)))


def run_backend(probe: Path, cases: list[tuple]) -> tuple[dict, list[tuple[bool, list[tuple[str, float]]]]]:
    """cases: (B, T, Hq, Hkv, Hs, window, causal, scale, varlen, chunks). Returns the priorities and, per case,
    (supported, [(where, scale), ...])."""
    lines = [" ".join(str(v) if i != 7 else hexf(v) for i, v in enumerate(case)) for case in cases]
    out = subprocess.run([str(probe)], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    rows = out.stdout.splitlines()
    head = rows[0].split()
    assert head[0] == "priorities"
    priorities = {k: int(v) for k, v in (item.split("=") for item in head[1:])}
    results = []
    for row in rows[1:]:
        fields = row.split()
        supported = fields[0] == "supports=1"
        events = [(where, float.fromhex(value)) for where, value in (f.split("=") for f in fields[1:])]
        results.append((supported, events))
    assert len(results) == len(cases)
    return priorities, results


def gemma3_27b_attention_ops() -> list[dict]:
    sys.path.insert(0, str(REPO))
    from surogate.dsl import models as _models  # noqa: F401 - registers the HF configs
    from surogate.dsl.py_compiler import compile_model_for_hf

    ir = json.loads(compile_model_for_hf("Gemma3ForCausalLM", GEMMA3_27B_TEXT))
    assert ir["success"], ir.get("errors")
    ops = [o for o in ir["modules"][0]["forward"]["operations"] if o["kernel_type"] == "flash_attention"]
    assert len(ops) == GEMMA3_27B_TEXT["num_hidden_layers"]
    return ops


def case_for(op: dict, *, varlen: int, B: int = 1, T: int = 256, chunks: int = 1) -> tuple:
    attrs = op["attrs"]
    cfg = GEMMA3_27B_TEXT
    return (
        B,
        T,
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
        cfg["head_dim"],
        int(attrs.get("window_size") or 0),
        1 if attrs.get("causal", True) else 0,
        float(attrs.get("softmax_scale") or 0.0),
        varlen,
        chunks,
    )


def test_gemma3_27b_global_layers_take_cudnn_with_the_declared_scale(backend_probe):
    """The defect: the ten global (window 0) layers of Gemma3-27B select cuDNN on an unpacked micro-batch, and
    cuDNN must multiply Q K^T by the op's 168**-0.5 in the forward and in the backward, not by 1/sqrt(128)."""
    ops = gemma3_27b_attention_ops()
    priorities, results = run_backend(backend_probe, [case_for(op, varlen=0) for op in ops])
    # cuDNN outranks every backend that could also take these ops (kvprefix runs only in chunked-sequence mode).
    assert priorities["cudnn"] > priorities["mem_eff"] > priorities["flash_varlen"]
    declared = float(np.float32(168.0**-0.5))
    assert declared != default_scale(128)
    taken = []
    for op, (supported, events) in zip(ops, results):
        if not supported:
            assert int(op["attrs"].get("window_size") or 0) > 0, op.get("id")
            continue
        taken.append(op)
        assert float(np.float32(op["attrs"]["softmax_scale"])) == declared
        assert events == [("fwd", declared), ("bwd", declared)], (
            f"cuDNN applied {[e[1] for e in events]} for the declared {declared} (1/sqrt(128) = {default_scale(128)})"
        )
    assert len(taken) == 10  # layers 5, 11, ..., 59: every sixth layer is global


def test_packed_micro_batches_leave_cudnn(backend_probe):
    """With in-row document boundaries (cu_seqlens set) cuDNN refuses and flash_varlen / mem_eff run, which already
    honour softmax_scale: before T22 one run mixed both scales depending on the batch contents."""
    ops = gemma3_27b_attention_ops()
    _, results = run_backend(backend_probe, [case_for(op, varlen=1) for op in ops])
    assert not any(supported for supported, _ in results)


@pytest.mark.parametrize("hs", range(8, 129, 8))
def test_an_unset_scale_keeps_one_over_sqrt_head_dim(backend_probe, hs):
    """softmax_scale 0 (every model that does not declare one) keeps the kernel default, bit for bit what upstream
    computed: its forward wrote 1.0 / sqrtf(HS) (a double quotient rounded to float), its backward 1.f / sqrtf(HS)."""
    _, [(supported, events)] = run_backend(backend_probe, [(2, 64, 8, 2, hs, 0, 1, 0.0, 0, 1)])
    assert supported
    upstream_fwd = float(np.float32(np.float64(1.0) / np.float64(np.sqrt(np.float32(hs)))))
    assert upstream_fwd == default_scale(hs)
    assert events == [("fwd", default_scale(hs)), ("bwd", default_scale(hs))]


def test_an_explicit_scale_is_used_verbatim(backend_probe):
    """Gemma4 passes 1.0 (its QK-norm already scales); a config with a full-attention layer at Hs <= 128 would have
    been run at 1/sqrt(Hs). The published Gemma4 head dims (256/512) never reach cuDNN; this is the synthetic case."""
    _, results = run_backend(backend_probe, [(1, 128, 8, 8, 128, 0, 1, 1.0, 0, 1), (1, 128, 8, 4, 64, 0, 1, 0.3, 0, 1)])
    assert results[0] == (True, [("fwd", 1.0), ("bwd", 1.0)])
    assert results[1] == (True, [("fwd", float(np.float32(0.3))), ("bwd", float(np.float32(0.3)))])


def test_every_chunk_of_a_chunked_backward_uses_the_scale(backend_probe):
    """RuntimeOptions::AttBwdChunks > 1 splits the cuDNN backward over the batch; each slice gets the same scale."""
    scale = float(np.float32(168.0**-0.5))
    _, [(supported, events)] = run_backend(backend_probe, [(4, 64, 32, 16, 128, 0, 1, scale, 0, 2)])
    assert supported
    assert events == [("fwd", scale), ("bwd", scale), ("bwd", scale)]


def test_the_fp32_tensor_overloads_forward_the_scale(fp32_probe):
    """The FP32 branch of the Tensor overloads runs the in-tree kernel; it takes the same scale argument."""
    scale = float(np.float32(168.0**-0.5))
    lines = [f"1 64 4 2 128 {hexf(scale)}", f"1 64 4 2 64 {hexf(0.0)}"]
    out = subprocess.run([str(fp32_probe)], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stdout + out.stderr
    rows = [[(w, float.fromhex(v)) for w, v in (f.split("=") for f in row.split())] for row in out.stdout.splitlines()]
    assert rows == [
        [("gpu_fwd", scale), ("gpu_bwd", scale)],
        [("gpu_fwd", default_scale(64)), ("gpu_bwd", default_scale(64))],
    ]


def test_the_dispatch_hands_the_op_scale_to_both_directions():
    """What the backend relies on: the forward and backward dispatch copy the op's softmax_scale into the params,
    the backward op inherits the attribute from the forward, and the compiled op reads it from the IR."""
    forward = body(FLASH_ATTENTION, r"void CompiledExecutor::dispatch_flash_attention\(")
    backward = body(FLASH_ATTENTION, r"void CompiledExecutor::dispatch_flash_attention_backward\(")
    for code in (forward, backward):
        assert code.count("params.softmax_scale = op.attrs.softmax_scale;") == 1
    text = FLASH_ATTENTION.read_text(encoding="utf-8")
    assert re.search(r'copy_attrs\(fwd\.attrs, \{[^}]*"softmax_scale"', text)
    compiler = (REPO / "csrc/src/runtime/dsl/graph_compiler.cpp").read_text(encoding="utf-8")
    assert 'find_attr(op.attrs, "softmax_scale")' in compiler


def test_the_cudnn_graphs_take_the_scale_at_execute_time():
    """The cached cuDNN graphs are keyed by shape only, so a per-op scale is correct only while attn_scale stays a
    pass-by-value tensor read from the variant pack at every execute(), in both graph builders."""
    for signature in (r"auto lookup_cache_or_build_graph_fwd\(", r"auto lookup_cache_or_build_graph_bwd\("):
        code = body(WRAPPER_CUDNN, signature)
        scale = code[code.index('.set_name("attn_scale")') :]
        scale = scale[: scale.index(";")]
        assert ".set_is_pass_by_value(true)" in scale and ".set_uid(Attn_scale_UID)" in scale
        assert "set_attn_scale(attn_scale)" in code
        assert "softmax_scale" not in code and "scale" not in code[code.index("auto key") : code.index("auto it")]


# The GPU check (errata T22) reads the attention dumps flash_attention.cpp writes for one layer and decides which
# scale the kernel applied; the tests below run that checker on dumps of an independent kernel (torch SDPA) and pin
# the dump names and format it reads.
DUMP_CHECK = Path(__file__).with_name("t22_attention_dump_check.py")


def dump_check():
    spec = importlib.util.spec_from_file_location("t22_attention_dump_check", DUMP_CHECK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_kernel_dumps(directory: Path, scale: float, *, backward: bool, layer: int = 5) -> None:
    """What a bf16 kernel applying `scale` would leave in SUROGATE_DEBUG_DUMP_DIR, in the native layouts:
    qkv [B, T, Hq + 2 Hkv, Hs], out / d_out [B, T, Hq, Hs], lse [B, Hq, T]; q/k RMS-normalised with gain 2."""
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    B, T, hq, hkv, hs = 1, 128, 4, 2, 128
    gen = torch.Generator().manual_seed(0)

    def bf16(x):
        return x.to(torch.bfloat16).float()

    def gain2(x):
        return 2 * x / x.pow(2).mean(-1, keepdim=True).sqrt()

    q = bf16(gain2(torch.randn(B, hq, T, hs, generator=gen)))
    k = bf16(gain2(torch.randn(B, hkv, T, hs, generator=gen)))
    v = bf16(torch.randn(B, hkv, T, hs, generator=gen))
    d_out = bf16(torch.randn(B, hq, T, hs, generator=gen))
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
    out = F.scaled_dot_product_attention(*leaves, is_causal=True, scale=scale, enable_gqa=True)
    scores = scale * (q @ k.repeat_interleave(hq // hkv, 1).transpose(-1, -2))
    lse = torch.logsumexp(scores.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf")), -1)
    qkv = torch.cat([q, k, v], 1).transpose(1, 2)
    tensors = {"qkv": qkv, "out": bf16(out.detach()).transpose(1, 2), "lse": lse}
    if backward:
        out.backward(d_out)
        d_qkv = torch.cat([t.grad for t in leaves], 1).transpose(1, 2)
        tensors.update(qkv_bwd=qkv, d_out=d_out.transpose(1, 2), d_qkv=bf16(d_qkv))
    for name, tensor in tensors.items():
        stem = directory / f"attn_live.layer{layer}.{name}"
        data = tensor.detach().float().contiguous().numpy()
        data.tofile(f"{stem}.bin")
        Path(f"{stem}.json").write_text(json.dumps({"name": stem.name, "dtype": "float32", "shape": list(data.shape)}))


@pytest.mark.parametrize("applied", ["declared", "default"])
def test_the_gpu_dump_check_names_the_scale_the_kernel_applied(tmp_path, capsys, applied):
    """The fixed kernel (168**-0.5) passes, the unfixed one (1/sqrt(128)) fails, in out, lse and d_qkv alike."""
    declared = 168.0**-0.5
    write_kernel_dumps(tmp_path, declared if applied == "declared" else 128.0**-0.5, backward=True)
    check = dump_check()
    hs, default, rows = check.compare(tmp_path, 5, 4, 2, declared)
    assert (hs, default) == (128, pytest.approx(128.0**-0.5, rel=1e-15))
    assert [r[0] for r in rows] == ["out", "lse", "d_qkv"]
    assert [check.verdict(*r[:3]) for r in rows] == [applied] * 3
    status = check.main([str(tmp_path), "--layer", "5", "--hq", "4", "--hkv", "2", "--scale", repr(declared)])
    assert status == (0 if applied == "declared" else 1), capsys.readouterr().out


def test_the_gpu_dump_check_handles_forward_only_and_missing_dumps(tmp_path):
    """trainer.validate leaves only the forward dumps; a directory without them is an error, not a pass."""
    check = dump_check()
    args = ["--layer", "5", "--hq", "4", "--hkv", "2", "--scale", repr(168.0**-0.5)]
    assert check.main([str(tmp_path), *args]) == 2
    write_kernel_dumps(tmp_path, 168.0**-0.5, backward=False)
    assert [r[0] for r in check.compare(tmp_path, 5, 4, 2, 168.0**-0.5)[2]] == ["out", "lse"]
    assert check.main([str(tmp_path), *args]) == 0


def test_the_gpu_dump_check_reads_what_flash_attention_writes():
    """The dump names, the env switches and the float32 .bin + {"shape": [...]} .json format the checker relies on,
    and that the forward dump follows the kernel call and d_qkv the backward call."""
    text = FLASH_ATTENTION.read_text(encoding="utf-8")
    assert 'std::getenv("SUROGATE_DEBUG_ATTENTION_DUMP_LAYER")' in text
    writer = body(FLASH_ATTENTION, r"void debug_dump_runtime_tensor\(")
    assert 'std::string(dump_dir) + "/" + safe + ".bin"' in writer and 'safe + ".json"' in writer
    assert "sizeof(float), nelem, bin_f" in writer and '\\"shape\\": [' in writer
    forward = body(FLASH_ATTENTION, r"void CompiledExecutor::dispatch_flash_attention\(")
    backward = body(FLASH_ATTENTION, r"void CompiledExecutor::dispatch_flash_attention_backward\(")
    for name in ("qkv", "out", "lse"):
        dump = f'debug_dump_runtime_tensor(fmt::format("attn_live.layer{{}}.{name}", layer_idx), {name}, dump_dir);'
        assert forward.index("backend.forward(params);") < forward.index(dump)
    for name, var in (("d_out", "d_out"), ("qkv_bwd", "qkv")):
        assert f'fmt::format("attn_live.layer{{}}.{name}", layer_idx), {var}, dump_dir);' in backward
    d_qkv = 'fmt::format("attn_live.layer{}.d_qkv", layer_idx), d_qkv,'
    assert backward.index("backend.backward(params);") < backward.index(d_qkv)
