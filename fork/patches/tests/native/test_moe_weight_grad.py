"""Source pins of ledger topic T3 (conditional zero-fill of MoE expert weight gradients), runnable without CUDA.

Production skips the dW zero-fill when every expert has tokens, which is correct only while the per-expert GEMMs
overwrite each routed expert's whole block. fork/native/tests/test-fork-moe-grouped-gemm.cu checks that on a GPU by
calling dsl::moe_base_weight_grad (fork/native/src/moe_weight_grad.cpp), the function production calls. These pins
fail on a CPU-only machine when a sync puts a GEMM loop or zero-fill back into the upstream function, drops the
call, or stops compiling the fork source into the library, any of which would leave the GPU test checking code
that production no longer runs. The last pin checks that the Catch2 filters the fork docs give for rerunning T3
select every case of that test.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OPS = REPO / "csrc/src/runtime/ops"
NATIVE = REPO / "fork/native"


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


def test_production_writes_expert_weight_gradients_only_through_the_tested_function():
    code = body(OPS / "moe_grouped_gemm.cpp", r"void CompiledExecutor::backward_moe_base_weights\(")
    call = (
        "moe_base_weight_grad(dw, dout, input, offsets, experts, accumulate, mRunState.cublas_handle(), "
        "mRunState.MainStream, op.outputs[1].name);"
    )
    assert code.count(call) == 1
    assert code.index("Tensor& dw = ensure_output_tensor(op.outputs[1]);") < code.index(call)
    assert code.index(call) < code.index("store_tensor(op.outputs[1], dw);")
    for write in ("cublasGemmEx", "cublasLt", "fill_zero", "dw.Data", "moe_weight_grad_needs_zero"):
        assert write not in code, f"backward_moe_base_weights uses {write} outside dsl::moe_base_weight_grad"
    source = (OPS / "moe_grouped_gemm.cpp").read_text(encoding="utf-8")
    assert '#include "surogate_fork/moe_weight_grad.h"' in source


def test_the_fork_source_is_part_of_the_training_library():
    cmake = (NATIVE / "cmake/fork_native.cmake").read_text(encoding="utf-8")
    match = re.search(r"function\(surogate_fork_configure_common target\)(.*?)endfunction\(\)", cmake, re.S)
    assert match, "fork_native.cmake: no surogate_fork_configure_common()"
    hook = match.group(1)
    assert 'target_sources(${target} PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/src/moe_weight_grad.cpp")' in hook
    csrc = (REPO / "csrc/CMakeLists.txt").read_text(encoding="utf-8")
    assert "surogate_fork_configure_common(surogate-common)" in csrc


def test_the_gpu_test_calls_the_production_function():
    test = (NATIVE / "tests/test-fork-moe-grouped-gemm.cu").read_text(encoding="utf-8")
    assert "dsl::moe_base_weight_grad(" in test
    assert "moe_grouped_gemm_weight_grad" not in test
    assert "[wgrad-zero]" in test


def test_the_documented_reruns_select_every_t3_case():
    # Catch2 tag filters: ',' is OR, adjacent tags are AND. A filter such as '[moe][gemm][wgrad-zero]' would skip
    # the host-only cases (policy truth table, shape check), and a case without [wgrad-zero] would be skipped by all.
    test = (NATIVE / "tests/test-fork-moe-grouped-gemm.cu").read_text(encoding="utf-8")
    cases = re.findall(r'TEST_CASE\(\s*"([^"]*)"\s*,\s*"([^"]*)"', test)
    assert cases and len(cases) == test.count("TEST_CASE("), "a TEST_CASE without a literal name and tag string"
    filters = []
    for doc in ("fork/docs/performance.md", "fork/native/README.md"):
        text = (REPO / doc).read_text(encoding="utf-8")
        filters += [(doc, f) for f in re.findall(r"integration-tests '([^']*)'", text) if "[wgrad-zero]" in f]
    assert any(doc == "fork/docs/performance.md" for doc, _ in filters), "performance.md: no T3 rerun command"
    for doc, expr in filters:
        terms = [re.findall(r"\[[^\]]+\]", term) for term in expr.split(",")]
        for name, tags in cases:
            assert "[wgrad-zero]" in tags, f"{name!r} lacks [wgrad-zero]"
            assert any(all(t in tags for t in term) for term in terms), f"{doc}: '{expr}' skips {name!r}"
