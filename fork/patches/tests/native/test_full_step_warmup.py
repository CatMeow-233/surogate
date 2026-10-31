"""Ledger topic T20 (the full-step warmup runs once for a model that never captures), runnable without CUDA.

train_step_graphed runs one discarded eager forward+backward over every micro-step before it captures the full-step
graph. A model with capture-unsafe ops (MoE, EP) never captures, so upstream ran that warmup on every step. The
decision is surogate_fork::full_step_warmup::run_warmup (fork/native/include/surogate_fork/full_step_warmup.h); the
probe compiles it and replays step sequences, and the pins check that train_step_graphed feeds it, keeps the stack
shrink inside the warmup and keeps the post-warmup capture-unsafe check as the fallback. The speed-up and the
bit-identical losses are GPU-unverified; fork/docs/performance.md gives the 5090 check.
"""

import itertools
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
PY_TRAIN = REPO / "csrc/src/binding/py_train.cpp"
PY_TRAIN_H = REPO / "csrc/src/binding/py_train.h"
EXECUTOR = REPO / "csrc/src/runtime/executor/graph_executor.cpp"
MODEL_EXECUTION = REPO / "csrc/src/runtime/dsl/dsl_model_execution.cpp"
FORK_INCLUDE = REPO / "fork/native/include"
FIELDS = ("captured", "skip_env", "force_env", "capture_unsafe", "expert_parallel", "warmed_up")

PROBE = r"""
#include <cstdio>
#include "surogate_fork/full_step_warmup.h"
int main() {
    for (int bits = 0; bits < 64; ++bits) {
        surogate_fork::full_step_warmup::Inputs in;
        in.captured = bits & 1; in.skip_env = bits & 2; in.force_env = bits & 4;
        in.capture_unsafe = bits & 8; in.expert_parallel = bits & 16; in.warmed_up = bits & 32;
        std::printf("%d %d\n", bits, surogate_fork::full_step_warmup::run_warmup(in) ? 1 : 0);
    }
}
"""


def body(path: Path, signature: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(signature + r"[^;{]*\{", text)
    assert match, f"{path.relative_to(REPO)}: no definition matching {signature!r}"
    depth = 1
    for end in range(match.end(), len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            return " ".join(text[match.start() : end + 1].split())
    raise AssertionError(f"{path.relative_to(REPO)}: unbalanced braces after {signature!r}")


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("no C++ compiler to build the warmup probe")
    work = tmp_path_factory.mktemp("full_step_warmup")
    (work / "probe.cpp").write_text(PROBE, encoding="utf-8")
    build = subprocess.run([compiler, "-std=c++17", f"-I{FORK_INCLUDE}", str(work / "probe.cpp"), "-o", str(work / "probe")],
                           capture_output=True, text=True, timeout=60)
    assert build.returncode == 0, build.stderr[-3000:]
    out = subprocess.run([str(work / "probe")], capture_output=True, text=True, timeout=10, check=True).stdout
    decisions = {}
    for line in out.splitlines():
        bits, run = map(int, line.split())
        decisions[tuple(bool(bits >> i & 1) for i in range(len(FIELDS)))] = bool(run)
    return decisions


def test_the_decision_is_upstreams_except_for_a_warmed_up_capture_unsafe_model_without_ep(table):
    for key in itertools.product((False, True), repeat=len(FIELDS)):
        c = dict(zip(FIELDS, key))
        upstream = not c["captured"] and not c["skip_env"]
        skipped = c["capture_unsafe"] and not c["expert_parallel"] and c["warmed_up"] and not c["force_env"]
        assert table[key] == (upstream and not skipped), c


def steps(table, *, capture_unsafe, expert_parallel=False, force=False, skip=False, n=5):
    """Warmups over n steps of train_step_graphed: the flag is set by a warmup, and a capture-safe model captures."""
    ran, captured, warmed = [], False, False
    for _ in range(n):
        run = table[(captured, skip, force, capture_unsafe, expert_parallel, warmed)]
        ran.append(run)
        warmed = warmed or run
        captured = captured or not capture_unsafe
    return ran


def test_step_sequences(table):
    assert steps(table, capture_unsafe=False) == [True, False, False, False, False]  # capture-safe: as upstream
    assert steps(table, capture_unsafe=True) == [True, False, False, False, False]  # MoE: once, with the stack shrink
    assert steps(table, capture_unsafe=True, expert_parallel=True) == [True] * 5  # EP: upstream (sticky plan ages)
    assert steps(table, capture_unsafe=True, force=True) == [True] * 5  # SUROGATE_FORK_FORCE_GRAPH_WARMUP=1
    assert steps(table, capture_unsafe=True, skip=True) == [False] * 5  # SUROGATE_DSL_GRAPH_SKIP_WARMUP=1


def test_train_step_graphed_feeds_the_decision():
    step = body(PY_TRAIN, r"std::pair<float, float> MultiGPUPyTrainer::train_step_graphed\(")
    for line in (
        "warmup_in.captured = gs.captured;",
        'warmup_in.skip_env = env_enabled("SUROGATE_DSL_GRAPH_SKIP_WARMUP");',
        'warmup_in.force_env = env_enabled("SUROGATE_FORK_FORCE_GRAPH_WARMUP");',
        "warmup_in.capture_unsafe = dsl_model->has_capture_unsafe_ops();",
        "warmup_in.expert_parallel = mOptions.EPSize > 1;",
        "warmup_in.warmed_up = gs.fork_warmup_done;",
        "const bool warmup_full_graph = surogate_fork::full_step_warmup::run_warmup(warmup_in);",
    ):
        assert line in step, line
    warmup = step[step.index("if (warmup_full_graph) {") :]
    flag, shrink, fallback = (warmup.index(s) for s in (
        "gs.fork_warmup_done = true;", "shrink_stack_to_high_water_mark(", "if (dsl_model->has_capture_unsafe_ops()) {"))
    assert warmup.index("dsl_model->set_rng_state(rng_state);") < flag < shrink < fallback
    # reset_capture (shape changes) keeps the flag: the warmup's one job for these models is the one-shot shrink.
    reset = body(PY_TRAIN_H, r"void reset_capture\(")
    assert "fork_warmup_done" not in reset
    assert "bool fork_warmup_done = false;" in PY_TRAIN_H.read_text(encoding="utf-8")


def test_capture_unsafety_is_known_before_the_first_step():
    """has_capture_unsafe_ops reads the compiled forward graph, which allocate_run_state compiles."""
    assert "return graph_has_capture_unsafe_ops(mCompiledForward.get());" in body(
        EXECUTOR, r"bool GraphExecutor::has_capture_unsafe_ops\(")
    assert "exec->ensure_graphs_compiled(B, T);" in body(MODEL_EXECUTION, r"void DslModel::allocate_run_state\(")
