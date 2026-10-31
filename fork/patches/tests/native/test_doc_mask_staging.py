"""Ledger topics T17 and T18 (cap-and-pad document masking of the full-step CUDA graph), runnable without CUDA.

A full-step graph captured with packed documents gives each micro-step a slice of cap+1 cu_seqlens offsets; every
captured forward copies its slice to the device, and train_step_graphed rewrites the slices before each launch.
The probe below compiles GraphExecutor's doc-masking functions and CausalLMExecutionProfile's compute_doc_masking /
apply_doc_masking as they stand in the sources (cut out of graph_executor.cpp and causal_lm_execution_profile.cpp),
against host stubs for CUDA: a captured memcpy is recorded, not run, and runs again at every replay, as in a CUDA
graph. Around them it replays train_step_graphed's doc-mask steps with the fork functions py_train.cpp calls
(surogate_fork/doc_mask_staging.h). For every micro-step it prints the geometry the captured attention reads at
replay and the geometry eager mode (compute_doc_masking without a graph) runs. The tests turn both into the set of
(query, key) pairs attended, for FlashAttention-2 varlen (a document's queries are computed up to max_seqlen) and for
mem_eff (Hs > 256: up to min(max_seqlen, T) queries per document), and require them to be equal.

Two negative controls run the same code with upstream's choices put back: the {0, B*T} synthesis for a step without
in-row resets after a capture with documents (C2), and clear_doc_masking for such a micro-step during the capture
(the companion defect: that micro-step is captured dense). Both must show the defect, so the harness can see it.
The GPU half (the kernels themselves) is unverified here; fork/docs/upstream-errata.md gives the 5090 checks.
"""

import os
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
EXECUTOR = REPO / "csrc/src/runtime/executor/graph_executor.cpp"
PROFILE = REPO / "csrc/src/runtime/executor/causal_lm_execution_profile.cpp"
PY_TRAIN = REPO / "csrc/src/binding/py_train.cpp"
FORK_INCLUDE = REPO / "fork/native/include"


def definition(path: Path, signature: str) -> str:
    """The first definition matching `signature` (a regex), through its closing brace; comments and literals skipped."""
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
            return text[match.start() : i + 1]
        i += 1
    raise AssertionError(f"{path.relative_to(REPO)}: unbalanced braces after {signature!r}")


PRELUDE = r"""
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "surogate_fork/doc_mask_staging.h"

// ---- CUDA host stubs: a captured memcpy is recorded and replayed, never run at capture time ----
using cudaStream_t = void*;
enum cudaError_t { cudaSuccess = 0 };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone = 0, cudaStreamCaptureStatusActive = 1 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1 };
#define CUDA_CHECK(x) do { if ((x) != cudaSuccess) std::abort(); } while (0)

struct Node {
    bool copy = false;
    const void* src = nullptr;
    void* dst = nullptr;
    std::size_t bytes = 0;
    // attention node: the doc-masking context baked at capture
    int ms = 0;
    const char* pass = "";
    const std::int32_t* gpu = nullptr;
    int num_docs = 0, max_seqlen = 0, total_q = 0;
};
static bool g_capturing = false;
static std::vector<Node>* g_graph = nullptr;

cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* status) {
    *status = g_capturing ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
    return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind, cudaStream_t) {
    if (g_capturing) {
        Node n;
        n.copy = true;
        n.src = src;
        n.dst = dst;
        n.bytes = bytes;
        g_graph->push_back(n);
    } else {
        std::memcpy(dst, src, bytes);
    }
    return cudaSuccess;
}
cudaError_t cudaMalloc(std::int32_t** p, std::size_t bytes) {
    if (g_capturing) { std::puts("FATAL cudaMalloc during capture"); std::exit(3); }
    *p = static_cast<std::int32_t*>(std::calloc(bytes, 1));
    return cudaSuccess;
}
cudaError_t cudaFree(void* p) {
    if (g_capturing) { std::puts("FATAL cudaFree during capture"); std::exit(3); }
    std::free(p);
    return cudaSuccess;
}

struct RuntimeOptions { bool DocMasking = true; };
namespace modules {
struct RopeStub { bool is_multimodal() const { return false; } };
struct ModelConfig { RopeStub Rope; };
}  // namespace modules
struct Tensor {
    void* Data = nullptr;
    int Device = -1;
    long Sizes[4] = {0, 0, 0, 0};
    long nelem() const { return Sizes[0] * Sizes[1]; }
};

namespace dsl {

struct CausalLMDocMaskingInfo {
    std::vector<std::int32_t> cu_seqlens;
    int num_docs = 0;
    int max_seqlen = 0;
    int total_q = 0;
};

class IGraphExecutor {
public:
    virtual ~IGraphExecutor() = default;
    virtual void set_doc_masking(const std::int32_t*, int, int, int, int = 0) = 0;
    virtual void clear_doc_masking() = 0;
    virtual bool set_row_doc_masking_for_capture(int, int, int) { return false; }
};

struct CompiledStub {
    const std::int32_t* gpu = nullptr;
    const std::int32_t* cpu = nullptr;
    int num_docs = 0, max_seqlen = 0, total_q = 0;
    void set_doc_masking_context(std::int32_t* g, const std::int32_t* c, int n, int m, int t) {
        gpu = g; cpu = c; num_docs = n; max_seqlen = m; total_q = t;
    }
    void clear_doc_masking_context() { gpu = nullptr; cpu = nullptr; num_docs = max_seqlen = total_q = 0; }
};
struct RunStateStub { cudaStream_t MainStream = nullptr; };

class GraphExecutor : public IGraphExecutor {
public:
    void set_doc_masking(const std::int32_t* cu_seqlens_cpu, int num_docs, int max_seqlen, int total_q,
                         int micro_step = 0) override;
    void clear_doc_masking() override;
    bool set_row_doc_masking_for_capture(int B, int T, int micro_step) override;
    void enable_doc_masking_pad_to_max(int max_num_docs, int num_micro_steps);
    void stage_cu_seqlens_for_micro_step(int micro_step, const std::int32_t* cu_seqlens_cpu, int count, int total_q);
    void reset_cuda_graphs() { mForwardGraph = nullptr; }
    void set_doc_masking_exact_outside_capture(const std::int32_t* cu_seqlens_cpu, int num_docs, int max_seqlen,
                                               int total_q);

    RunStateStub mRunState;
    std::unique_ptr<CompiledStub> mCompiledExecutor = std::make_unique<CompiledStub>();
    void* mForwardGraph = nullptr;
    std::int32_t* mCuSeqlensGpu = nullptr;
    std::vector<std::int32_t> mCuSeqlensCpu;
    int mCuSeqlensCount = 0;
    int mDocMaskingNumDocs = 0;
    int mDocMaskingMaxSeqlen = 0;
    int mDocMaskingTotalQ = 0;
    int mCapturedNumDocs = 0;
    int mCapturedMaxSeqlen = 0;
    int mDocMaskingPadMaxDocs = 0;
    int mDocMaskingPadMicroSteps = 1;
    std::vector<std::int32_t> mForkExactCuSeqlensCpu;
    std::int32_t* mForkExactCuSeqlensGpu = nullptr;
    int mForkExactCuSeqlensCount = 0;
};

class CausalLMExecutionProfile {
public:
    std::optional<CausalLMDocMaskingInfo> compute_doc_masking(const std::int32_t* position_ids, int B, int T,
                                                              bool mrope = false) const;
    bool apply_doc_masking(IGraphExecutor& executor, const RuntimeOptions& options,
                           const modules::ModelConfig& config, Tensor inputs, Tensor position_ids,
                           int micro_step = 0) const;
};

namespace {
@@STREAM_IS_CAPTURING@@
}  // namespace

@@PRODUCTION@@

}  // namespace dsl
"""

DRIVER = r"""
using namespace dsl;

// Upstream's apply_doc_masking choice for a micro-step without in-row resets during the capture: clear.
struct NoRowRoute final : IGraphExecutor {
    GraphExecutor* ex;
    explicit NoRowRoute(GraphExecutor* e) : ex(e) {}
    void set_doc_masking(const std::int32_t* c, int n, int m, int t, int ms) override { ex->set_doc_masking(c, n, m, t, ms); }
    void clear_doc_masking() override { ex->clear_doc_masking(); }
};

static std::string join(const std::int32_t* p, int n) {
    std::ostringstream o;
    for (int i = 0; i < n; ++i) o << (i ? "," : "") << p[i];
    return o.str();
}
static int round_up_pow2_int(int v) { int p = 1; while (p < v) p <<= 1; return p; }

struct Trainer {
    int B = 0, T = 0, GA = 0, env_cap = 0;
    std::string mode = "fixed";  // fixed | upstream_c2 | no_row_route
    GraphExecutor ex;
    std::unique_ptr<NoRowRoute> no_row;
    CausalLMExecutionProfile profile;
    RuntimeOptions opts;
    modules::ModelConfig cfg;
    bool captured = false;
    int captured_doc_cap = 0;
    std::vector<Node> graph;
    int step = 0;

    IGraphExecutor& api() {
        if (mode == "no_row_route") {
            if (!no_row) no_row = std::make_unique<NoRowRoute>(&ex);
            return *no_row;
        }
        return ex;
    }
    void emit(const char* tag, int ms, const char* pass, const std::int32_t* gpu, int n, int m, int t) {
        if (!gpu) { std::printf("%s step=%d ms=%d pass=%s dense\n", tag, step, ms, pass); return; }
        std::printf("%s step=%d ms=%d pass=%s varlen n=%d max=%d total=%d cu=%s\n", tag, step, ms, pass, n, m, t,
                    join(gpu, n + 1).c_str());
    }
    void attention(int ms, const char* pass, const char* tag) {
        const auto& c = *ex.mCompiledExecutor;
        if (g_capturing) {
            Node node;
            node.ms = ms; node.pass = pass; node.gpu = c.gpu;
            node.num_docs = c.num_docs; node.max_seqlen = c.max_seqlen; node.total_q = c.total_q;
            graph.push_back(node);
        } else {
            emit(tag, ms, pass, c.gpu, c.num_docs, c.max_seqlen, c.total_q);
        }
    }
    // DslModel::forward + backward for one micro-step: apply_doc_masking, attention forward and backward, clear.
    void fwd_bwd(std::vector<std::int32_t>& pos, int j, const char* tag) {
        Tensor in; in.Sizes[0] = B; in.Sizes[1] = T;
        Tensor p; p.Data = pos.data(); p.Device = -1; p.Sizes[0] = B; p.Sizes[1] = T;
        const bool active = profile.apply_doc_masking(api(), opts, cfg, in, p, j);
        attention(j, "fwd", tag);
        attention(j, "bwd", tag);
        if (active) api().clear_doc_masking();
    }
    // train_step_graphed's doc-mask steps (py_train.cpp), with the fork's functions.
    void train_step(std::vector<std::vector<std::int32_t>>& pos) {
        std::vector<std::vector<std::int32_t>> per_ms(GA);
        bool has = false;
        int cur_max = 0;
        for (int j = 0; j < GA; ++j) {
            auto docs = surogate_fork::doc_mask::scan_micro_step_documents(pos[j].data(), B, T);
            has = has || docs.has_in_row_boundaries;
            per_ms[j] = std::move(docs.cu_seqlens);
            cur_max = std::max(cur_max, static_cast<int>(per_ms[j].size()) - 1);
        }
        int max_num_docs = env_cap > 0 ? env_cap : 64;
        if (has && cur_max > max_num_docs) {
            if (env_cap > 0) { std::printf("THROW cap exceeded\n"); return; }
            max_num_docs = round_up_pow2_int(cur_max);
        }
        if (!captured) {  // warmup: eager forward/backward of every micro-step
            for (int j = 0; j < GA; ++j) fwd_bwd(pos[j], j, "warmup");
        }
        if (captured) {
            if (mode == "upstream_c2") {
                if (has && (captured_doc_cap <= 0 || cur_max > captured_doc_cap)) {
                    captured = false; captured_doc_cap = 0;
                } else if (!has && captured_doc_cap > 0) {
                    has = true;
                    for (auto& cu : per_ms) cu.assign({0, B * T});
                }
            } else {
                const auto plan = surogate_fork::doc_mask::plan_captured_step(captured_doc_cap, has, cur_max);
                if (plan.recapture) { captured = false; captured_doc_cap = 0; }
                else has = plan.stage_slices;
            }
        }
        if (!captured) {
            if (has) {
                const int cap = std::min(max_num_docs, round_up_pow2_int(cur_max));
                ex.enable_doc_masking_pad_to_max(cap, GA);
                captured_doc_cap = cap;
                for (int j = 0; j < GA; ++j) ex.stage_cu_seqlens_for_micro_step(j, per_ms[j].data(), (int)per_ms[j].size(), B * T);
            } else if (mode != "upstream_c2") {
                ex.enable_doc_masking_pad_to_max(0, GA);
            }
            graph.clear();
            g_graph = &graph;
            g_capturing = true;
            for (int j = 0; j < GA; ++j) fwd_bwd(pos[j], j, "capture");
            g_capturing = false;
            captured = true;
            std::printf("captured step=%d cap=%d\n", step, captured_doc_cap);
        }
        if (has) {
            for (int j = 0; j < GA; ++j) ex.stage_cu_seqlens_for_micro_step(j, per_ms[j].data(), (int)per_ms[j].size(), B * T);
        }
        for (const auto& node : graph) {  // cudaGraphLaunch
            if (node.copy) { std::memcpy(node.dst, node.src, node.bytes); continue; }
            emit("graph", node.ms, node.pass, node.gpu, node.num_docs, node.max_seqlen, node.total_q);
        }
        // Eager reference: the same micro-steps without a graph (use_cuda_graphs: false).
        GraphExecutor eager;
        for (int j = 0; j < GA; ++j) {
            Tensor in; in.Sizes[0] = B; in.Sizes[1] = T;
            Tensor p; p.Data = pos[j].data(); p.Device = -1; p.Sizes[0] = B; p.Sizes[1] = T;
            const bool active = profile.apply_doc_masking(eager, opts, cfg, in, p, j);
            const auto& c = *eager.mCompiledExecutor;
            emit("eager", j, "fwd", c.gpu, c.num_docs, c.max_seqlen, c.total_q);
            if (active) eager.clear_doc_masking();
        }
        ++step;
    }
    // DslModel::validate between steps: eager, pad mode left as the last capture set it.
    void validate(std::vector<std::int32_t>& pos, int micro_step) {
        const std::vector<std::int32_t> before = ex.mCuSeqlensCpu;
        const void* before_data = ex.mCuSeqlensCpu.data();
        Tensor in; in.Sizes[0] = B; in.Sizes[1] = T;
        Tensor p; p.Data = pos.data(); p.Device = -1; p.Sizes[0] = B; p.Sizes[1] = T;
        bool active = false;
        try {
            active = profile.apply_doc_masking(ex, opts, cfg, in, p, micro_step);
        } catch (const std::exception& e) {
            std::printf("eval_throw %s\n", e.what());
            return;
        }
        const auto& c = *ex.mCompiledExecutor;
        emit("evalattn", micro_step, "fwd", c.gpu, c.num_docs, c.max_seqlen, c.total_q);
        if (active) ex.clear_doc_masking();
        GraphExecutor eager;
        const bool eactive = profile.apply_doc_masking(eager, opts, cfg, in, p, micro_step);
        const auto& e = *eager.mCompiledExecutor;
        emit("evaleager", micro_step, "fwd", e.gpu, e.num_docs, e.max_seqlen, e.total_q);
        if (eactive) eager.clear_doc_masking();
        // The slices are what captured copies read: the eval may rewrite the slice it was given, never the storage.
        std::printf("eval_slices_storage_kept=%d\n", before_data == ex.mCuSeqlensCpu.data() &&
                                                         before.size() == ex.mCuSeqlensCpu.size() ? 1 : 0);
    }
};

static std::vector<std::int32_t> read_ints(std::istream& in, long n) {
    std::vector<std::int32_t> v(static_cast<std::size_t>(n));
    for (auto& x : v) in >> x;
    return v;
}

int main() {
    Trainer tr;
    std::string cmd;
    while (std::cin >> cmd) {
        if (cmd == "mode") { std::cin >> tr.mode; }
        else if (cmd == "dims") { std::cin >> tr.B >> tr.T >> tr.GA >> tr.env_cap; }
        else if (cmd == "step") {
            std::vector<std::vector<std::int32_t>> pos(tr.GA);
            for (auto& p : pos) p = read_ints(std::cin, (long)tr.B * tr.T);
            tr.train_step(pos);
        } else if (cmd == "eval") {
            int ms; std::cin >> ms;
            auto p = read_ints(std::cin, (long)tr.B * tr.T);
            tr.validate(p, ms);
        } else if (cmd == "scan") {
            auto p = read_ints(std::cin, (long)tr.B * tr.T);
            const auto docs = surogate_fork::doc_mask::scan_micro_step_documents(p.data(), tr.B, tr.T);
            const auto info = tr.profile.compute_doc_masking(p.data(), tr.B, tr.T, false);
            std::printf("scan has=%d cu=%s ref=%s\n", docs.has_in_row_boundaries ? 1 : 0,
                        join(docs.cu_seqlens.data(), (int)docs.cu_seqlens.size()).c_str(),
                        info ? join(info->cu_seqlens.data(), (int)info->cu_seqlens.size()).c_str() : "none");
        } else if (cmd == "padoff") {
            GraphExecutor ex;
            ex.enable_doc_masking_pad_to_max(4, 2);
            const void* data = ex.mCuSeqlensCpu.data();
            const std::size_t cap = ex.mCuSeqlensCpu.capacity(), size = ex.mCuSeqlensCpu.size();
            ex.enable_doc_masking_pad_to_max(0, 2);
            std::printf("padoff pad=%d ms=%d storage_kept=%d\n", ex.mDocMaskingPadMaxDocs, ex.mDocMaskingPadMicroSteps,
                        data == ex.mCuSeqlensCpu.data() && cap == ex.mCuSeqlensCpu.capacity() &&
                                size == ex.mCuSeqlensCpu.size() ? 1 : 0);
        } else if (cmd == "stage_overflow") {
            GraphExecutor ex;
            ex.enable_doc_masking_pad_to_max(2, 1);
            const std::int32_t cu[] = {0, 2, 4, 6, 8};
            for (int capturing = 0; capturing <= 1; ++capturing) {
                std::vector<Node> scratch;
                g_graph = &scratch;
                g_capturing = capturing == 1;
                try {
                    ex.stage_cu_seqlens_for_micro_step(0, cu, 5, 8);
                    std::printf("stage_overflow capturing=%d threw=0\n", capturing);
                } catch (const std::exception& e) {
                    std::printf("stage_overflow capturing=%d threw=1 %s\n", capturing, e.what());
                }
                g_capturing = false;
            }
        } else if (cmd == "exact_buffers") {
            // A captured graph holds mCuSeqlensGpu (sized at enable) and the slices; an eager overflow,
            // even one larger than that buffer, runs through buffers of its own.
            GraphExecutor ex;
            ex.enable_doc_masking_pad_to_max(2, 1);
            const std::int32_t* shared = ex.mCuSeqlensGpu;
            const int shared_count = ex.mCuSeqlensCount;
            const std::int32_t* slices = ex.mCuSeqlensCpu.data();
            std::vector<std::int32_t> big;
            for (int i = 0; i <= 12; ++i) big.push_back(i * 2);  // 12 documents over 24 tokens: 13 offsets
            ex.set_doc_masking(big.data(), 12, 2, 24, 0);
            const auto& c = *ex.mCompiledExecutor;
            std::printf("exact_buffers shared_kept=%d own=%d n=%d max=%d total=%d cu=%s\n",
                        shared == ex.mCuSeqlensGpu && shared_count == ex.mCuSeqlensCount &&
                                slices == ex.mCuSeqlensCpu.data() ? 1 : 0,
                        c.gpu != nullptr && c.gpu != ex.mCuSeqlensGpu ? 1 : 0, c.num_docs, c.max_seqlen, c.total_q,
                        join(c.gpu, c.num_docs + 1).c_str());
            std::printf("enable_sized shared_count=%d\n", shared_count);
        }
    }
    return 0;
}
"""


def probe_source() -> str:
    production = "\n\n".join(
        [
            definition(EXECUTOR, r"void GraphExecutor::enable_doc_masking_pad_to_max\("),
            definition(EXECUTOR, r"void GraphExecutor::stage_cu_seqlens_for_micro_step\("),
            definition(EXECUTOR, r"void GraphExecutor::set_doc_masking\("),
            definition(EXECUTOR, r"void GraphExecutor::clear_doc_masking\("),
            definition(EXECUTOR, r"void GraphExecutor::set_doc_masking_exact_outside_capture\("),
            definition(EXECUTOR, r"bool GraphExecutor::set_row_doc_masking_for_capture\("),
            definition(PROFILE, r"std::optional<CausalLMDocMaskingInfo>\s*CausalLMExecutionProfile::compute_doc_masking\("),
            definition(PROFILE, r"bool CausalLMExecutionProfile::apply_doc_masking\("),
        ]
    )
    capturing = definition(EXECUTOR, r"inline bool stream_is_capturing\(")
    return PRELUDE.replace("@@STREAM_IS_CAPTURING@@", capturing).replace("@@PRODUCTION@@", production) + DRIVER


@pytest.fixture(scope="module")
def probe(tmp_path_factory) -> Path:
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("no C++ compiler (g++, c++ or clang++) to build the doc-mask probe")
    work = tmp_path_factory.mktemp("doc_mask_staging")
    source, binary = work / "probe.cpp", work / "probe"
    source.write_text(probe_source(), encoding="utf-8")
    build = subprocess.run(
        [compiler, "-std=c++20", "-O1", f"-I{FORK_INCLUDE}", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert build.returncode == 0, f"doc-mask probe does not compile:\n{build.stderr[-4000:]}"
    return binary


# ---- scenarios --------------------------------------------------------------------------------------------------


def rows_from_docs(doc_lengths, B, T, start=0):
    """Pack documents (lengths) into B rows of T without truncation, as tokenize.py's packer does: a document that
    does not fit continues in the next row with its positions counting on. `start` is the position the first token
    continues from (a document already running before this micro-step)."""
    pos, p, remaining = [], start, list(doc_lengths)
    left = remaining.pop(0) if remaining else 0
    while len(pos) < B * T:
        if left == 0:
            left, p = (remaining.pop(0) if remaining else B * T), 0
        pos.append(p)
        p += 1
        left -= 1
    return pos


def long_document(B, T, start):
    return list(range(start, start + B * T))


def run(probe, B, T, GA, steps, mode="fixed", env_cap=0, evals=()):
    """steps: list of GA position lists per step; evals: {after_step: (micro_step, positions)}."""
    lines = [f"mode {mode}", f"dims {B} {T} {GA} {env_cap}"]
    evals = dict(evals)
    for s, pos_ms in enumerate(steps):
        lines.append("step " + " ".join(" ".join(map(str, p)) for p in pos_ms))
        if s in evals:
            ms, pos = evals[s]
            lines.append(f"eval {ms} " + " ".join(map(str, pos)))
    out = subprocess.run([str(probe)], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout.splitlines()


def parse(line):
    head, *fields = line.split()
    rec = {"tag": head}
    for f in fields:
        if "=" in f:
            k, v = f.split("=", 1)
            rec[k] = v
        else:
            rec["kind"] = f
    return rec


def coverage(rec, T, backend):
    """Varlen attention: the (query, key) pairs computed and the queries written. FA2 runs ceil(max_seqlen / kBlockM)
    query blocks per document (block size 1 here); mem_eff sizes its grid from min(max_seqlen, T)."""
    total = int(rec["total"])
    cu = [int(x) for x in rec["cu"].split(",")]
    n, max_seqlen = int(rec["n"]), int(rec["max"])
    limit = max_seqlen if backend == "fa2" else min(max_seqlen, T)
    pairs, written = set(), set()
    for d in range(n):
        s, e = cu[d], cu[d + 1]
        for q in range(s, min(e, s + limit)):
            written.add(q)
            pairs.update((q, k) for k in range(s, q + 1))
    assert cu[n] <= total
    return frozenset(pairs), frozenset(written)


def dense_coverage(B, T):
    pairs = {(b * T + t, b * T + k) for b in range(B) for t in range(T) for k in range(t + 1)}
    return frozenset(pairs), frozenset(range(B * T))


def cov(rec, B, T, backend):
    return dense_coverage(B, T) if rec["kind"] == "dense" else coverage(rec, T, backend)


def mismatches(lines, B, T):
    """Micro-steps whose replayed attention differs from eager, per backend: [(backend, step, ms, pass)]."""
    eager = {(r["step"], r["ms"]): r for r in map(parse, lines) if r["tag"] == "eager"}
    bad = []
    for r in map(parse, lines):
        if r["tag"] != "graph":
            continue
        ref = eager[(r["step"], r["ms"])]
        for backend in ("fa2", "mem_eff"):
            if cov(r, B, T, backend) != cov(ref, B, T, backend):
                bad.append((backend, int(r["step"]), int(r["ms"]), r["pass"]))
    return bad


B, T, GA = 2, 8, 2
SHORT = rows_from_docs([3, 5, 2, 6], B, T)  # 4 documents over the two rows: in-row resets
SHORT_B = rows_from_docs([5, 3, 4, 4], B, T)


@pytest.mark.parametrize("mode", ["fixed", "upstream_c2"])
def test_c2_boundary_free_step_after_doc_capture(probe, mode):
    steps = [[SHORT, SHORT_B], [long_document(B, T, 100), long_document(B, T, 116)]]
    lines = run(probe, B, T, GA, steps, mode=mode)
    assert "captured step=0 cap=4" in lines, lines
    assert sum(line.startswith("captured") for line in lines) == 1, "step 1 must replay the step-0 graph"
    bad = mismatches(lines, B, T)
    if mode == "fixed":
        assert bad == [], bad
        replay = [parse(x) for x in lines if x.startswith("graph step=1")]
        assert all(r["cu"].startswith("0,8,16,16") for r in replay), replay  # per-row documents, padded
    else:
        # Negative control: upstream's {0, B*T} attends across rows on FA2 and leaves rows unwritten on mem_eff.
        assert {(b, s) for b, s, _, _ in bad} == {("fa2", 1), ("mem_eff", 1)}, bad
        fa2 = [parse(x) for x in lines if x.startswith("graph step=1")]
        pairs, written = coverage(fa2[0], T, "fa2")
        assert (T, T - 1) in pairs  # row 1's first query attends to row 0's last key
        _, written_mem_eff = coverage(fa2[0], T, "mem_eff")
        assert written_mem_eff == frozenset(range(T))  # rows 1..B-1 get no attention output


@pytest.mark.parametrize("mode", ["fixed", "no_row_route"])
def test_boundary_free_micro_step_during_capture_stays_varlen(probe, mode):
    # Step 0 captures with documents in ms0 and none in ms1; step 1 brings documents into ms1 (within the cap).
    steps = [[SHORT, long_document(B, T, 40)], [SHORT_B, SHORT]]
    lines = run(probe, B, T, GA, steps, mode=mode)
    assert sum(line.startswith("captured") for line in lines) == 1
    bad = mismatches(lines, B, T)
    if mode == "fixed":
        assert bad == [], bad
        cap_ms1 = [parse(x) for x in lines if x.startswith("graph step=0 ms=1")]
        assert cap_ms1 and all(r["kind"] == "varlen" for r in cap_ms1), cap_ms1
    else:
        # Negative control: upstream captured ms1 dense, so step 1's documents in ms1 attend across each other.
        assert {(s, ms) for _, s, ms, _ in bad} == {(1, 1)}, bad


def test_eval_with_more_documents_than_the_cap_runs_exact_geometry(probe):
    # Capture with at most 4 documents per micro-step (cap 4), then validate a batch of 8 documents.
    eval_pos = rows_from_docs([2] * 8, B, T)
    steps = [[SHORT, SHORT_B], [SHORT_B, SHORT]]
    lines = run(probe, B, T, GA, steps, evals={0: (5, eval_pos)})
    assert "captured step=0 cap=4" in lines
    attn = parse(next(x for x in lines if x.startswith("evalattn")))
    ref = parse(next(x for x in lines if x.startswith("evaleager")))
    for backend in ("fa2", "mem_eff"):
        assert cov(attn, B, T, backend) == cov(ref, B, T, backend), (backend, attn, ref)
    assert attn["n"] == "8" and attn["cu"] == "0,2,4,6,8,10,12,14,16"
    assert "eval_slices_storage_kept=1" in lines
    assert mismatches(lines, B, T) == [], "the eval disturbed the captured graph"


def test_eval_within_the_cap_keeps_the_padded_path(probe):
    eval_pos = rows_from_docs([4, 4, 4, 4], B, T)
    lines = run(probe, B, T, GA, [[SHORT, SHORT_B], [SHORT_B, SHORT]], evals={0: (0, eval_pos)})
    attn = parse(next(x for x in lines if x.startswith("evalattn")))
    assert attn["n"] == "4" and attn["max"] == str(B * T)  # upstream's padded geometry (num_docs = cap)
    ref = parse(next(x for x in lines if x.startswith("evaleager")))
    for backend in ("fa2", "mem_eff"):
        assert cov(attn, B, T, backend) == cov(ref, B, T, backend)
    assert mismatches(lines, B, T) == []


def random_micro_step(rng, B, T):
    """Mostly packed documents, sometimes a row run interior to a long document (no in-row reset)."""
    if rng.random() < 0.35:
        return long_document(B, T, rng.randrange(0, 500))
    lengths, left = [], B * T + rng.randrange(0, T)
    while left > 0:
        n = rng.choice([1, 2, 3, 5, 7, T, 2 * T, 3 * T])
        lengths.append(n)
        left -= n
    return rows_from_docs(lengths, B, T, start=rng.choice([0, 0, rng.randrange(1, 50)]))


@pytest.mark.parametrize("seed", range(40))
def test_random_steps_replay_what_eager_runs(probe, seed):
    rng = random.Random(seed)
    # B*T a power of two, as in real runs: a cap rounds up to a power of two and upstream sizes the device
    # buffer for total_q + 1 offsets, so a cap above B*T would need a cudaMalloc under the capture.
    b, t, ga = rng.choice([(1, 8, 2), (2, 8, 2), (2, 4, 3), (4, 4, 2), (4, 4, 1), (1, 16, 3)])
    steps = [[random_micro_step(rng, b, t) for _ in range(ga)] for _ in range(6)]
    lines = run(probe, b, t, ga, steps)
    assert mismatches(lines, b, t) == []


def test_the_fork_scan_is_compute_doc_masking(probe):
    rng = random.Random(7)
    lines = ["mode fixed", f"dims {B} {T} {GA} 0"]
    for _ in range(200):
        lines.append("scan " + " ".join(map(str, random_micro_step(rng, B, T))))
    out = subprocess.run([str(probe)], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=60)
    for rec in map(parse, out.stdout.splitlines()):
        if rec["has"] == "1":
            assert rec["cu"] == rec["ref"], rec
        else:
            assert rec["ref"] == "none" and rec["cu"] == ",".join(str(b * T) for b in range(B + 1)), rec


def probe_lines(probe, command):
    out = subprocess.run([str(probe)], input=command + "\n", capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout.splitlines()


def test_pad_mode_turns_off_without_touching_the_slices(probe):
    assert probe_lines(probe, "padoff") == ["padoff pad=0 ms=1 storage_kept=1"]


def test_staging_more_documents_than_the_cap_throws_outside_a_capture(probe):
    lines = probe_lines(probe, "stage_overflow")
    assert lines[0].startswith("stage_overflow capturing=0 threw=1 ") and "exceed the captured cap of 2" in lines[0]
    assert lines[1] == "stage_overflow capturing=1 threw=0"


def test_exact_geometry_leaves_the_buffers_a_graph_holds_alone(probe):
    exact, sized = probe_lines(probe, "exact_buffers")
    assert exact == "exact_buffers shared_kept=1 own=1 n=12 max=2 total=24 cu=" + ",".join(str(2 * i) for i in range(13))
    assert sized == "enable_sized shared_count=3"  # cap + 1 offsets, allocated before any capture


def test_first_documents_after_a_capture_without_them(probe):
    # Step 0 has no in-row reset anywhere (a long-document corpus), so its graph is captured dense and no eager
    # call ever allocates the device cu_seqlens buffer; step 1 brings documents and recaptures without a warmup.
    # Upstream then allocated that buffer with cudaMalloc under the open capture (the stub exits on it).
    steps = [[long_document(B, T, 0), long_document(B, T, 16)], [SHORT, SHORT_B], [long_document(B, T, 7), SHORT]]
    lines = run(probe, B, T, GA, steps)
    assert [x for x in lines if x.startswith("captured")] == ["captured step=0 cap=0", "captured step=1 cap=4"]
    assert mismatches(lines, B, T) == []


def test_py_train_calls_the_fork_plan():
    """train_step_graphed takes its scan and replay plan from the fork header and no longer synthesizes {0, B*T}."""
    body = definition(PY_TRAIN, r"std::pair<float, float> MultiGPUPyTrainer::train_step_graphed\(")
    flat = " ".join(body.split())
    assert "surogate_fork::doc_mask::scan_micro_step_documents(pos, B, T_step)" in flat
    assert "surogate_fork::doc_mask::plan_captured_step( gs.captured_doc_cap, has_doc_boundaries, current_max_num_docs)" in flat
    assert "cu.assign({0, total_q})" not in flat and "cu.assign({0, B * T})" not in flat
    assert "dsl_model->enable_doc_masking_pad_to_max(0, micro_steps);" in flat
    assert '#include "surogate_fork/doc_mask_staging.h"' in PY_TRAIN.read_text(encoding="utf-8")
