"""Source pins of ledger topic T13 (the host side of the cpu_training step), runnable without CUDA.

T13 drops the step-start memset of every host gradient (DslGradStore::start_micro_step) for HostGradOverwrites:
micro-step 0's D2H copies are recorded and only the buffers they miss are zeroed, before their first reader. The CPU
contract test (`fork/dev test openmp`, cpu-step-contract) checks the behaviour; these pins fail when a sync drops or
reshapes a hunk, or when upstream changes code the topic relies on: who reads and writes the host gradients.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
GRAD_STORE = REPO / "csrc/src/runtime/dsl/dsl_grad_store.cpp"
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


def test_every_micro_step_zero_copy_into_a_host_gradient_is_recorded():
    """Each D2H into a host gradient at micro-step 0 marks its buffer, so zero_rest leaves only uncovered ones."""
    layer = body(GRAD_STORE, r"void DslGradStore::offload_layer_grads\(")
    first = block_after(layer, "if (mMicroStep == 0) {")
    assert first.count("cudaMemcpyAsync(cpu_it->second.Data,") == 1
    assert first.count("mHostOverwrites->overwritten(cpu_it->second.Data, grad_it->second.bytes());") == 1
    non_block = body(GRAD_STORE, r"void DslGradStore::offload_non_block_grads\(")
    first = block_after(non_block, "if (mMicroStep == 0) {")
    assert first.count("cudaMemcpyAsync(cpu_it->second.Data,") == 1
    assert first.count("mHostOverwrites->overwritten(cpu_it->second.Data, grad_it->second.bytes());") == 1
    # The only copies into host gradients in the grad store: the two above and the non-block copy of later
    # micro-steps, which overwrites a buffer micro-step 0 has already zeroed or covered.
    assert GRAD_STORE.read_text(encoding="utf-8").count("cudaMemcpyAsync(cpu_it->second.Data,") == 3


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


# ---- build and documentation ----------------------------------------------------------------------------------


def test_the_topic_is_documented():
    doc = PERFORMANCE.read_text(encoding="utf-8")
    assert "HostGradOverwrites" in doc and "## cpu_training 主机端步" in doc
