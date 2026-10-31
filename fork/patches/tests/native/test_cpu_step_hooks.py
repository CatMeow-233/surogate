"""Source pins of ledger topics T13-T15 (the host side of the cpu_training step), runnable without CUDA.

T13 drops the step-start memset of every host gradient (DslGradStore::start_micro_step) for HostGradOverwrites:
micro-step 0's D2H copies are recorded and only the buffers they miss are zeroed, before their first reader. T14
adds an opt-in vectorized, team-parallel host addition (SUROGATE_FORK_HOST_GRAD_ACCUM=1) and an opt-in D2H stream of
the grad store's own (SUROGATE_FORK_GRAD_D2H_STREAM=1). T15 adds an opt-in vectorized CPU AdamW
(SUROGATE_FORK_CPU_ADAMW_SIMD=1). The CPU contract test (`fork/dev test openmp`, cpu-step-contract) checks the
behaviour; these pins fail when a sync drops or reshapes a hunk, or when upstream changes code the topics rely on: who
reads and writes the host gradients, and the upstream loops the fork kernels must reproduce bit for bit.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
GRAD_STORE = REPO / "csrc/src/runtime/dsl/dsl_grad_store.cpp"
GRAD_STORE_H = REPO / "csrc/src/runtime/dsl/dsl_grad_store.h"
CPU_ADAMW = REPO / "csrc/src/runtime/optimizers/cpu_adamw.cpp"
KERNELS = REPO / "fork/native/src/cpu_step_kernels.cpp"
KERNELS_H = REPO / "fork/native/include/surogate_fork/cpu_step_kernels.h"
KERNELS_CMAKE = REPO / "fork/native/cmake/cpu_step_kernels.cmake"
FORK_NATIVE = REPO / "fork/native/cmake/fork_native.cmake"
STEPBENCH = REPO / "fork/native/bench/cpu-optimizer/stepbench.cpp"
PERFORMANCE = REPO / "fork/docs/performance.md"


def body(path: Path, signature: str) -> str:
    """The brace-balanced body of the first definition matching `signature` (a regex), whitespace collapsed."""
    text = path.read_text(encoding="utf-8")
    match = re.search(signature + r"[^;{]*\{", text)
    assert match, f"{path.relative_to(REPO)}: no definition matching {signature!r}"
    depth, start = 1, match.end()
    for end in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            return " ".join(text[start:end].split())
    raise AssertionError(f"{path.relative_to(REPO)}: unbalanced braces after {signature!r}")


def block_after(code: str, opener: str) -> str:
    """The brace-balanced block that follows the first `opener` (which ends with "{") in collapsed code."""
    start = code.index(opener) + len(opener)
    depth = 1
    for end in range(start, len(code)):
        depth += {"{": 1, "}": -1}.get(code[end], 0)
        if depth == 0:
            return code[start:end].strip()
    raise AssertionError(f"unbalanced braces after {opener!r}")


LOOP = "for (std::size_t i = 0; i < n; ++i) {"

# ---- T13 ------------------------------------------------------------------------------------------------------


def test_micro_step_zero_records_instead_of_zeroing():
    code = body(GRAD_STORE, r"void DslGradStore::start_micro_step\(")
    assert "std::memset" not in code, "the step-start memset of the host gradients is back: drop it again (T13)"
    first = block_after(code, "if (!mAccumulate) {")
    assert "if (mStreamGrads) { mHostOverwrites->begin_step();" in first
    later = block_after(code, "} else if (mStreamGrads) {")
    assert (
        later.startswith("// CPU gradients that micro-step 0 did not overwrite")
        and "mHostOverwrites->zero_rest();" in later
    )


# Each copy into a host gradient calls before_copy() with the copy's destination and size before it is enqueued:
# a copy over part of a buffer must zero it first, as upstream's memset did (host_grad_overwrites.h).
HOST_COPY = "CUDA_CHECK(cudaMemcpyAsync(cpu_it->second.Data, grad_it->second.Data, grad_it->second.bytes(),"
RECORDED_HOST_COPY = (
    "mHostOverwrites->before_copy(cpu_it->second.Data, grad_it->second.bytes()); // fork ledger T13 " + HOST_COPY
)


def test_every_copy_into_a_host_gradient_calls_before_copy_first():
    """zero_rest leaves only buffers no copy wrote; a copy enqueued before before_copy could be zeroed under it."""
    layer = body(GRAD_STORE, r"void DslGradStore::offload_layer_grads\(")
    first = block_after(layer, "if (mMicroStep == 0) {")
    assert first.count(HOST_COPY) == first.count(RECORDED_HOST_COPY) == 1
    non_block = body(GRAD_STORE, r"void DslGradStore::offload_non_block_grads\(")
    # micro-step 0's copy and the later micro-steps' copy of the GPU-accumulated value
    assert non_block.count(HOST_COPY) == non_block.count(RECORDED_HOST_COPY) == 2
    # The only copies into host gradients in the grad store are these three.
    whole = " ".join(GRAD_STORE.read_text(encoding="utf-8").split())
    assert whole.count("cudaMemcpyAsync(cpu_it->second.Data,") == whole.count(RECORDED_HOST_COPY) == 3
    assert "overwritten(" not in whole, "the recorder's call after the copy is back: call before_copy() first"


def test_the_optimizer_reads_host_gradients_only_after_wait_all_offloads():
    """zero_rest runs in wait_all_offloads for one micro-step; a reader that skipped it would see stale buffers."""
    wait = body(GRAD_STORE, r"void DslGradStore::wait_all_offloads\(")
    assert wait.index("cudaStreamSynchronize(stream)") < wait.index("mHostOverwrites->zero_rest();")
    readers = []
    for path in (REPO / "csrc/src").rglob("*"):
        if path.suffix not in {".cpp", ".cu", ".h", ".cuh"} or path.name.startswith("dsl_grad_store."):
            continue
        if "third_party" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "get_cpu_grads_map" in text or "get_cpu_grad(" in text:
            readers.append(path.relative_to(REPO).as_posix())
    assert readers == ["csrc/src/runtime/optimizers/adamw_8bit_optimizer.cpp"], readers
    step = body(REPO / readers[0], r"void AdamW8BitOptimizer::step_cpu_streaming\(")
    assert step.index("wait_all_offloads(stream)") < step.index("get_cpu_grads_map()") < step.index("get_cpu_grad(")


def test_every_host_gradient_is_registered():
    alloc = body(GRAD_STORE, r"void DslGradStore::allocate_streaming_buffers\(")
    assert (
        "mHostOverwrites = std::make_shared<surogate_fork::HostGradOverwrites>(); "
        "for (const auto& [name, host] : mCpuGrads) { mHostOverwrites->add(host.Data, host.bytes()); }"
    ) in alloc


# ---- T14 ------------------------------------------------------------------------------------------------------

UPSTREAM_BF16_ADD = (
    "for (std::size_t i = 0; i < task.nelem; ++i) { float val = static_cast<float>(dst[i]) + "
    "static_cast<float>(src[i]); dst[i] = static_cast<nv_bfloat16>(val); }"
)


def test_the_host_function_dispatches_before_the_upstream_loop():
    code = body(GRAD_STORE, r"void DslGradStore::accumulate_staged_layer\(")
    fork = block_after(code, "if (surogate_fork::host_grad_accum_enabled()) {")
    assert "surogate_fork::host_grad_add_f32(" in fork and "surogate_fork::host_grad_add_bf16(" in fork
    assert fork.count("task.serial_group") == 2 and fork.endswith("return;")
    assert code.index("host_grad_accum_enabled()") < code.index(UPSTREAM_BF16_ADD), "upstream's loop changed"


def test_the_benchmark_times_upstreams_loop():
    """stepbench.cpp's copy of the BF16 branch is what the T14 rows of performance.md compare against."""
    bench = body(STEPBENCH, r"void upstream_accumulate_bf16\(")
    assert UPSTREAM_BF16_ADD.replace("task.nelem", "nelem") in bench


def test_tasks_carry_their_store_and_the_rank_thread_joins_its_group():
    header = " ".join(GRAD_STORE_H.read_text(encoding="utf-8").split())
    assert "const void* serial_group = nullptr;" in header
    alloc = body(GRAD_STORE, r"void DslGradStore::allocate_streaming_buffers\(")
    assert "cpu_it->second.DType, this});" in alloc
    start = body(GRAD_STORE, r"void DslGradStore::start_micro_step\(")
    assert "if (surogate_fork::host_grad_accum_enabled()) surogate_fork::set_omp_serial_group(this);" in start


def test_the_d2h_stream_replaces_the_callers_copy_stream():
    events = body(GRAD_STORE, r"void DslGradStore::create_streaming_events\(")
    assert "if (surogate_fork::grad_d2h_stream_enabled() && !mForkD2HStream) {" in events
    assert "cudaStreamCreate(&mForkD2HStream)" in events
    layer = body(GRAD_STORE, r"void DslGradStore::offload_layer_grads\(")
    swap = "if (mForkD2HStream) copy_stream = mForkD2HStream;"
    assert layer.index(swap) < layer.index("cudaStreamWaitEvent(copy_stream, slot.compute_done, 0)")


# ---- T15 ------------------------------------------------------------------------------------------------------

ADAMW = {
    # upstream function -> (fork entry point, fork serial kernel, the call's leading arguments)
    "cpu_adamw_step": ("adamw_step_f32", "adamw_f32_serial", "(p, g, m, v, n,"),
    "cpu_adamw_step_bf16": ("adamw_step_bf16_grad", "adamw_bf16_grad_serial", "(p, grad_bf16, m, v, n,"),
    "cpu_adamw_step_bf16_param": (
        "adamw_step_bf16_param",
        "adamw_bf16_param_serial",
        "(param_bf16, grad_bf16, m, v, n,",
    ),
}
STEP_FIELDS = "{lr, beta1, beta2, bc1, bc2, eps, wd_factor, grad_scale}"


def upstream_as_fork(loop: str) -> str:
    """An upstream AdamW loop body in the fork kernel's spelling: the scalars live in `s`, BF16 goes by bits."""
    loop = re.sub(r"\b(lr|beta1|beta2|bc1|bc2|eps|wd_factor|grad_scale)\b", r"s.\1", loop)
    loop = loop.replace("static_cast<float>(p[i])", "bf16_to_f32(p[i])")
    loop = loop.replace("static_cast<float>(g[i])", "bf16_to_f32(g[i])")
    return loop.replace("p[i] = static_cast<nv_bfloat16>(pi);", "p[i] = f32_to_bf16(pi);")


def test_each_adamw_step_dispatches_with_its_own_scalars():
    for function, (entry, _, args) in ADAMW.items():
        code = body(CPU_ADAMW, rf"void {function}\(")
        fork = block_after(code, "if (surogate_fork::cpu_adamw_simd_enabled()) {")
        assert fork == f"surogate_fork::{entry}{args} {STEP_FIELDS}); return;", (function, fork)
        assert (
            code.index("const float wd_factor") < code.index("cpu_adamw_simd_enabled()") < code.index("OmpTeam team(n)")
        )


def test_the_step_fields_are_declared_in_the_order_cpu_adamw_fills_them():
    header = KERNELS_H.read_text(encoding="utf-8")
    struct = re.search(r"struct AdamWStep \{(.*?)\};", header, re.S).group(1)
    fields = re.findall(r"float (\w+);", struct)
    assert "{" + ", ".join(fields) + "}" == STEP_FIELDS


def test_the_fork_kernels_compute_the_upstream_expressions():
    """When upstream changes its AdamW arithmetic (for example to filter NaNs), the fork kernel must follow."""
    for function, (_, kernel, _) in ADAMW.items():
        upstream = block_after(body(CPU_ADAMW, rf"void {function}\("), LOOP)
        fork = block_after(body(KERNELS, rf"void {kernel}\("), LOOP)
        assert upstream_as_fork(upstream) == fork, (function, upstream_as_fork(upstream), fork)


# ---- build and documentation ----------------------------------------------------------------------------------


def test_the_kernels_build_with_their_flags_in_surogate_common():
    cmake = KERNELS_CMAKE.read_text(encoding="utf-8")
    assert 'COMPILE_OPTIONS "-fno-math-errno;-ffp-contract=off"' in cmake
    common = re.search(
        r"function\(surogate_fork_configure_common target\)(.*?)endfunction\(\)",
        FORK_NATIVE.read_text(encoding="utf-8"),
        re.S,
    )
    assert "surogate_fork_add_cpu_step_kernels(${target})" in common.group(1)


def test_the_contract_references_compile_like_upstream():
    """At -O0 the references cannot fuse multiply-adds, so an FMA-baseline build would hide T15's mismatch."""
    cmake = (REPO / "fork/native/openmp-contract/CMakeLists.txt").read_text(encoding="utf-8")
    assert "set_source_files_properties(cpu_step_check.cpp PROPERTIES COMPILE_OPTIONS -O3)" in cmake
    source = KERNELS.read_text(encoding="utf-8")
    assert "if (cpu_adamw_simd_supported()) return true;" in source


def test_the_topic_is_documented():
    doc = PERFORMANCE.read_text(encoding="utf-8")
    assert "HostGradOverwrites" in doc and "## cpu_training 主机端步" in doc


def test_the_switches_are_documented():
    doc = PERFORMANCE.read_text(encoding="utf-8")
    source = KERNELS.read_text(encoding="utf-8")
    for switch in ("SUROGATE_FORK_HOST_GRAD_ACCUM", "SUROGATE_FORK_GRAD_D2H_STREAM", "SUROGATE_FORK_CPU_ADAMW_SIMD"):
        assert f'env_flag("{switch}")' in source
        assert f"`{switch}=1`" in doc, switch
