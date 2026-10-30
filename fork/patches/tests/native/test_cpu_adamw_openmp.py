"""Source pins of ledger topic T5 (OpenMP for the CPU optimizer), runnable without CUDA.

T5 links OpenMP into surogate-common, which turns on the `#pragma omp` regions of upstream's cpu_adamw.cpp. Each
region now runs a surogate_fork::OmpTeam (fork/native/include/surogate_fork/omp_team.h): sized from the calling
thread's CPU mask, serial below a measured threshold, and the gradient norm sums in a fixed block order. The
openmp-contract test (`fork/dev test openmp`) checks the header's behaviour. These pins fail when a sync drops or
reshapes the cpu_adamw.cpp hunks, or when upstream adds a region that would take libgomp's default team again.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CPU_ADAMW = REPO / "csrc/src/runtime/optimizers/cpu_adamw.cpp"
HEADER = REPO / "fork/native/include/surogate_fork/omp_team.h"
PERFORMANCE = REPO / "fork/docs/performance.md"

TEAM = "const surogate_fork::OmpTeam team(n);"
PRAGMA = "#pragma omp parallel for schedule(static) num_threads(team.threads) if (team.threads > 1)"


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


def test_cpu_adamw_includes_the_team_header():
    assert '#include "surogate_fork/omp_team.h"' in CPU_ADAMW.read_text(encoding="utf-8")


def test_every_region_runs_a_sized_team():
    """Each element-wise loop sizes its team on the line before the pragma; no region keeps the default team."""
    lines = [line.strip() for line in CPU_ADAMW.read_text(encoding="utf-8").splitlines()]
    pragmas = [i for i, line in enumerate(lines) if line.startswith("#pragma omp")]
    assert len(pragmas) == 5, "cpu_adamw.cpp gained or lost an OpenMP region: give it an OmpTeam, update this pin"
    for i in pragmas:
        assert lines[i] == PRAGMA, lines[i]
        assert lines[i - 1] == TEAM, lines[i - 1]


def test_each_element_wise_kernel_has_one_region():
    for function in (
        "cpu_adamw_step",
        "cpu_adamw_step_bf16",
        "cpu_adamw_step_bf16_param",
        "cpu_vector_add_f32",
        "cpu_vector_add_bf16_to_f32",
    ):
        code = body(CPU_ADAMW, rf"void {function}\(")
        assert code.count(TEAM) == 1 and code.count(PRAGMA) == 1, function


def test_the_norm_sums_in_a_fixed_order():
    """An OpenMP reduction would make the clipping norm depend on the team size, which now differs per rank.

    The norm also keeps upstream's pointer dedupe: T10 shares one host gradient between a tied embedding and
    lm_head, and only this line keeps cpu_training from counting it twice (fork/docs/sync.md, T5).
    """
    norm = body(CPU_ADAMW, r"double cpu_gradient_norm_squared\(")
    assert norm.count("surogate_fork::ordered_block_sum(info.nelem, [d](std::size_t i) {") == 2
    assert norm.count("std::isfinite(") == 2
    assert "if (seen.count(it->second.Data)) continue; seen.insert(it->second.Data);" in norm
    assert "#pragma omp" not in norm
    assert "reduction(" not in CPU_ADAMW.read_text(encoding="utf-8")


def test_the_documented_threshold_and_block_are_the_built_ones():
    """fork/docs/performance.md records the measurement behind the threshold and the order the block fixes."""
    header = HEADER.read_text(encoding="utf-8")
    threshold = re.search(r"#define SUROGATE_FORK_OMP_MIN_PARALLEL_ELEMS (\d+)", header).group(1)
    block = re.search(r"kOrderedSumBlock = (\d+);", header).group(1)
    doc = PERFORMANCE.read_text(encoding="utf-8")
    assert f"`kOmpMinParallelElems` = {threshold}" in doc
    assert f"`kOrderedSumBlock` = {block}" in doc
