"""Ledger topic T27 (FP8 producer pre-quantization computes the amax it quantizes with), runnable without CUDA.

fused_residual_rmsnorm (LN1/LN2 output), swiglu and matmul_swiglu (SwiGLU output) can quantize their output into the
shared FP8 forward buffer and mark it ready, so the consuming matmul skips its own quantization
(FP8HybridRecipe::forward_matmul with inp_quant_ready). The producer kernels run with abs_max_ptr == nullptr, yet
upstream's blocks called quantize_with_abs_max with the buffer's abs_max slot: an amax that nothing had computed for
that tensor (another layer's value, or the slot's initial contents). The blocks are gated by
`has_fp8_forward() && !has_fp8_delayed_scaling()`, which no configuration satisfies today (both reduce to fp8-hybrid,
and fp8-hybrid always creates the FP8 scaling state), so the fix changes no current run; it makes the path correct for
whoever opens the gate (roadmap train-native #29).

The probe compiles the three pre-quantization blocks as they stand in the sources, gate forced open, against host
stubs whose abs_max computes max|x| and whose quantize_with_abs_max records the amax it reads, with the buffer's slot
holding a stale value. On upstream's sources (488e5f00) the recorded amax is the stale one and the tests fail. A last
test pins that the two FP8/BF16 recipe headers whose JIT branches have the same problem stay unused. The kernels are
GPU-unverified here; the errata section for T27 gives the 5090 check.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
OPS = REPO / "csrc/src/runtime/ops"
GATE = "if (mRecipe && mRecipe->uses_fp8_forward() && mRunState.has_fp8_forward()"
SITES = {
    "fused_residual_rmsnorm": (OPS / "fused_residual_rmsnorm.cpp", r"void CompiledExecutor::dispatch_fused_residual_rmsnorm\("),
    "swiglu": (OPS / "swiglu.cpp", r"void CompiledExecutor::dispatch_swiglu\("),
    "matmul_swiglu": (OPS / "matmul_swiglu.cpp", r"void CompiledExecutor::dispatch_matmul_swiglu\("),
}


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


def balanced(text: str, open_brace: int) -> int:
    """Index just past the brace matching the one at `open_brace`."""
    depth = 0
    for i in range(open_brace, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return i + 1
    raise AssertionError("unbalanced braces")


def prequant_block(site: str) -> str:
    """The pre-quantization if-statement of the site's dispatch function, comments removed."""
    path, signature = SITES[site]
    text = strip_comments(path.read_text(encoding="utf-8"))
    fn = re.search(signature + r"[^;{]*\{", text)
    assert fn, f"{path.name}: no definition matching {signature!r}"
    body = text[fn.end() - 1 : balanced(text, fn.end() - 1)]
    flat = " ".join(body.split())
    assert flat.count(GATE) == 1, f"{path.name}: expected one pre-quantization block"
    start = flat.index(GATE)
    end = balanced(flat, flat.index("{", start))
    return flat[start:end]


PROBE = r"""
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <vector>

using cudaStream_t = void*;
struct cudaDeviceProp {};
struct Tensor {
    void* Data = nullptr;
    float* Stats = nullptr;
    long n = 0;
    float* abs_max() { return Stats; }
    float* scale() { return Stats ? Stats + 4 : nullptr; }
};
inline Tensor view_tensor(const Tensor& t, std::initializer_list<long>) { return t; }

static float g_quantized_with = -1.f;  // the amax the last quantize_with_abs_max read
static int g_abs_max_calls = 0;
void abs_max(float* out, const Tensor& in, long n, const cudaDeviceProp&, cudaStream_t) {
    ++g_abs_max_calls;
    float m = 0.f;
    const float* x = static_cast<const float*>(in.Data);
    for (long i = 0; i < n; ++i) m = std::fmax(m, std::fabs(x[i]));
    *out = m;
}
void quantize_with_abs_max(Tensor& out, float* scale_ptr, const Tensor&, const float* amax, long, const cudaDeviceProp&,
                           cudaStream_t) {
    g_quantized_with = *amax;
    *scale_ptr = *amax / 448.f;  // fp8_scale_from_amax's inverse scale for E4M3
    (void)out;
}

namespace modules {
struct FP8ForwardQuantActivations { Tensor ln1, ln2, att, swiglu; };
}
struct DslRunState {
    enum FP8BufferReady : uint8_t { FP8Ready_None = 0, FP8Ready_LN1 = 1, FP8Ready_LN2 = 2, FP8Ready_Att = 4, FP8Ready_SwiGLU = 8 };
    modules::FP8ForwardQuantActivations quants;
    uint8_t ready = 0;
    cudaDeviceProp DeviceProp;
    cudaStream_t MainStream = nullptr;
    bool has_fp8_forward() const { return true; }
    bool has_fp8_delayed_scaling() const { return false; }  // the gate forced open
    modules::FP8ForwardQuantActivations& fp8_forward_quants() { return quants; }
    void set_fp8_buffer_ready(FP8BufferReady f) { ready |= f; }
};
struct Recipe { bool uses_fp8_forward() const { return true; } };
struct Config { long HiddenSize = 0; };

struct CompiledExecutor {
    Recipe* mRecipe;
    DslRunState& mRunState;
    long mB, mT;
    Config mConfig;
    void fused_residual_rmsnorm(Tensor& y, bool declared_ln1, bool declared_ln2, bool is_ln2_fwd, int fwd_layer_idx) {
        @fused_residual_rmsnorm@
    }
    void swiglu(Tensor& out, long B, long T, long D) {
        @swiglu@
    }
    void matmul_swiglu(Tensor& out_3d, long D) {
        @matmul_swiglu@
    }
};

int main() {
    const long B = 2, T = 3, C = 8, D = 8;
    std::vector<float> act(B * T * C);
    for (long i = 0; i < B * T * C; ++i) act[i] = 0.01f * static_cast<float>((i * 37) % 101) - 0.5f;
    act[13] = -3.5f;  // this tensor's amax
    Tensor y{act.data(), nullptr, B * T * C};
    float stats[3][8] = {};
    Recipe recipe;
    DslRunState rs;
    Tensor* bufs[3] = {&rs.quants.ln1, &rs.quants.ln2, &rs.quants.swiglu};
    static std::vector<unsigned char> fp8(B * T * C);
    for (int b = 0; b < 3; ++b) {
        bufs[b]->Data = fp8.data();
        bufs[b]->Stats = stats[b];
    }
    CompiledExecutor ex{&recipe, rs, B, T, Config{C}};
    auto report = [&](const char* what, Tensor& buf, uint8_t flag) {
        std::printf("%s %g %g %d %d\n", what, g_quantized_with, buf.Stats[0], (rs.ready & flag) != 0, g_abs_max_calls);
        g_quantized_with = -1.f;
        g_abs_max_calls = 0;
        rs.ready = 0;
    };
    // The slots hold what an earlier layer (or nothing) left there.
    stats[0][0] = 0.25f;
    ex.fused_residual_rmsnorm(y, true, false, false, 0);
    report("ln1", rs.quants.ln1, DslRunState::FP8Ready_LN1);
    stats[1][0] = 0.f;
    ex.fused_residual_rmsnorm(y, false, true, true, 0);
    report("ln2", rs.quants.ln2, DslRunState::FP8Ready_LN2);
    stats[2][0] = 100.f;
    ex.swiglu(y, B, T, D);
    report("swiglu", rs.quants.swiglu, DslRunState::FP8Ready_SwiGLU);
    stats[2][0] = 0.f;
    ex.matmul_swiglu(y, D);
    report("matmul_swiglu", rs.quants.swiglu, DslRunState::FP8Ready_SwiGLU);
    return 0;
}
"""


@pytest.fixture(scope="module")
def probe_output(tmp_path_factory):
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        pytest.skip("no C++ compiler")
    out = tmp_path_factory.mktemp("t27")
    src_text = PROBE
    for site in SITES:
        src_text = src_text.replace(f"@{site}@", prequant_block(site))
    src = out / "probe.cpp"
    src.write_text(src_text, encoding="utf-8")
    exe = out / "probe"
    result = subprocess.run([cxx, "-std=c++20", "-O0", str(src), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, f"probe does not compile:\n{result.stderr}"
    run = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    rows = {}
    for line in run.stdout.splitlines():
        what, used, slot, ready, calls = line.split()
        rows[what] = (float(used), float(slot), ready == "1", int(calls))
    return rows


@pytest.mark.parametrize("what", ["ln1", "ln2", "swiglu", "matmul_swiglu"])
def test_pre_quantization_uses_the_amax_of_the_tensor_it_quantizes(probe_output, what):
    used, slot, ready, calls = probe_output[what]
    assert ready, f"{what}: the block did not run (gate or buffer checks changed?)"
    assert used == 3.5, f"{what}: quantized with amax {used}, the tensor's is 3.5 (stale slot value)"
    assert slot == 3.5 and calls == 1, f"{what}: slot {slot}, abs_max calls {calls}"


@pytest.mark.parametrize("site", sorted(SITES))
def test_the_amax_is_computed_right_before_the_quantization(site):
    block = prequant_block(site)
    amax = [m.start() for m in re.finditer(r"(?<![\w.>])abs_max\(fp8_buf", block)]
    quant = [m.start() for m in re.finditer(r"(?<![\w.>])quantize_with_abs_max\(", block)]
    assert len(quant) == 1, f"{site}: expected one quantize_with_abs_max call"
    assert len(amax) == 1 and amax[0] < quant[0], f"{site}: no abs_max(...) of the FP8 buffer before quantize_with_abs_max"


def test_the_recipe_headers_with_the_same_jit_gap_stay_unused():
    # fp8_matmul.h's JIT branch and bf16_matmul.h quantize with an abs_max nothing computed; no TU includes them.
    # If a sync starts including one, that branch needs the same abs_max call before quantize_with_abs_max.
    names = ("fp8_matmul.h", "bf16_matmul.h")
    users = []
    for path in (REPO / "csrc").rglob("*"):
        if path.suffix in {".cpp", ".cu", ".h", ".cuh", ".hpp"} and path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            users += [f"{path.relative_to(REPO)} includes {n}" for n in names if re.search(rf'#\s*include\s*[<"][^>"]*{re.escape(n)}[>"]', text)]
    assert not users, users
