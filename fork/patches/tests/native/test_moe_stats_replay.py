"""Source pins of ledger topic T19 (MoE routing statistics counted once per routed chunk), runnable without CUDA.

With recompute on (the default), the backward replays each layer's forward ops (replay_layer_forward sets
mInReplay), MoETopK included. Its statistics block added every layer to moe_stats_device a second time: the summed
aux_loss and z_loss read 2x, while the per-layer averages divide by a slot-4 layer count that doubled too and stayed
the same. Chunked training (sequence_chunks > 1) routes every chunk twice more: train_step_graphed's phase A (the KV
sweep) and phase B (the re-forward before the backward) both run MoETopK, and only micro index 0 resets the buffer,
so every micro-step after the first counted twice even with recompute off. The fix skips the block in replay and in
the KV sweep (the flag upstream already uses to skip the loss there). The kernels are GPU code, so these pins check
the guard and the upstream code it relies on, and a model of the pinned schedule counts what Python reads; the GPU
check is in fork/docs/upstream-errata.md. The roadmap's other claim (the full-step warmup also double counts) is
wrong and is pinned here as well: micro-step 0 of the real pass resets the buffer after the warmup.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
MOE_TOPK = REPO / "csrc/src/runtime/ops/moe_topk.cpp"
REPLAY = REPO / "csrc/src/runtime/executor/compiled_ops_execute.cpp"
EXECUTOR = REPO / "csrc/src/runtime/executor/graph_executor.cpp"
RUN_STATE = REPO / "csrc/src/runtime/dsl/dsl_run_state.cpp"
PY_TRAIN = REPO / "csrc/src/binding/py_train.cpp"
COMPILED_OPS = REPO / "csrc/src/runtime/executor/compiled_ops.h"
FLASH_ATTENTION = REPO / "csrc/src/runtime/attention/flash_attention.cpp"
MODEL_H = REPO / "csrc/src/runtime/training/model.h"


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


def stats_guard() -> str:
    """The condition under which dispatch_moe_topk skips its statistics block ('' when it never does)."""
    topk = body(MOE_TOPK, r"void CompiledExecutor::dispatch_moe_topk\(")
    assert topk.count("moe_stats_device()") == 1, "a second statistics site: guard it too"
    match = re.search(
        r"if \(float\* stats = (?:\((.*?)\) \? nullptr : |(.*?) \? nullptr : )?mRunState\.moe_stats_device\(\)\) \{",
        topk,
    )
    assert match, "the statistics block no longer starts with `if (float* stats = ...moe_stats_device()) {`"
    return match.group(1) or match.group(2) or ""


def chunked_schedule() -> str:
    """train_step_graphed's chunked-sequence branch (seq_chunks > 1), whitespace-normalised."""
    step = body(PY_TRAIN, r"std::pair<float, float> MultiGPUPyTrainer::train_step_graphed\(")
    start = step.find("if (seq_chunks > 1) { // ---")
    end = step.find("} else for (int j = 0; j < micro_steps; ++j) {", start)
    assert -1 < start < end, "train_step_graphed's chunked schedule moved: re-read it and update these pins"
    return step[start:end]


def test_the_routing_stats_skip_the_recompute_replay_and_the_kv_sweep():
    topk = body(MOE_TOPK, r"void CompiledExecutor::dispatch_moe_topk\(")
    assert (
        "if (float* stats = (mInReplay || sequence_chunk_kv_sweep()) ? nullptr : mRunState.moe_stats_device()) {"
        in topk
    )
    assert stats_guard() == "mInReplay || sequence_chunk_kv_sweep()"


def test_replay_runs_the_layer_forward_ops_in_replay_mode():
    replay = body(REPLAY, r"void CompiledExecutor::replay_layer_forward\(")
    start, dispatch, end = (
        replay.find("mInReplay = true;"),
        replay.find("op.fn(*this, op,"),
        replay.rfind("mInReplay = false;"),
    )
    assert -1 < start < dispatch < end, (
        "replay_layer_forward no longer dispatches its ops between mInReplay = true/false"
    )
    # Loss ops are skipped in replay; MoETopK is not, which is what made the statistics count twice.
    assert "CompiledOpType::CrossEntropyLoss" in replay and "CompiledOpType::MoETopK" not in replay


def test_the_kv_sweep_flag_marks_phase_a_of_the_chunked_schedule_only():
    # The flag is set per chunk from the pack set_sequence_chunk receives, and reset by a pack without it.
    assert (
        "bool sequence_chunk_kv_sweep() const { return sequence_chunk_active() && mChunkPack.kv_sweep; }"
        in " ".join(COMPILED_OPS.read_text(encoding="utf-8").split())
    )
    assert "bool kv_sweep = false;" in MODEL_H.read_text(encoding="utf-8")
    setter = body(FLASH_ATTENTION, r"void CompiledExecutor::set_sequence_chunk\(")
    assert "if (pack) { mChunkPack = *pack; } else { mChunkPack = IModel::ChunkPackMeta{}; }" in setter
    # Phase A: kv_sweep pack, forward at micro index micro_base + c. Phase B: a fresh pack without it, forward and
    # backward at micro_base + (eff_chunks - 1 - c), so micro-step 0's first phase-B forward is index 0 (reset).
    sched = chunked_schedule()
    order = [
        "auto pack = build_chunk_pack(",
        "pack.kv_sweep = true;",
        "ctx.Model->set_sequence_chunk(c, seq_chunks, &pack);",
        "ctx.Model->forward(in_v, pos_v, *ctx.Communicator, micro_base + c);",
        "ctx.Model->zero_sequence_chunk_dkv();",
        "const int micro_eff = micro_base + (eff_chunks - 1 - c);",
        "auto pack = build_chunk_pack(",
        "pack.reuse_ep = true;",
        "ctx.Model->set_sequence_chunk(c, seq_chunks, &pack);",
        "ctx.Model->forward(in_v, pos_v, *ctx.Communicator, micro_eff);",
        "ctx.Model->backward(in_v, tgt_v, *ctx.Communicator, accum_eff, micro_eff);",
        "micro_base += eff_chunks;",
    ]
    at = 0
    for needle in order:
        found = sched.find(needle, at)
        assert found >= 0, f"chunked schedule: {needle!r} missing or out of order"
        at = found + len(needle)
    assert sched.count("kv_sweep") == 1 and sched.count("ctx.Model->forward(") == 2


def test_step_chunked_does_not_mark_its_kv_sweep():
    # step() (the multimodal loop's entry) has its own chunked schedule: phase A builds its pack with
    # chunk_pack_meta_from_positions and never sets kv_sweep, so the gate does not cover it and its phase A still
    # counts for micro-steps > 0. The errata says so; when this fails, upstream marks that sweep: update both.
    chunked = body(PY_TRAIN, r"void MultiGPUPyTrainer::step_chunked\(")
    assert (
        "chunk_pack_meta_from_positions(" in chunked
        and "forward_no_save(in, pos, *ctx.Communicator, micro_eff);" in chunked
    )
    assert "kv_sweep" not in chunked
    assert "kv_sweep" not in body(PY_TRAIN, r"IModel::ChunkPackMeta chunk_pack_meta_from_positions\(")


def reported_counts(chunks_per_micro: list[int], guard: str, recompute: bool) -> dict[tuple[int, int], int]:
    """How often each (micro-step, chunk) routing is in moe_stats_device when Python reads it after the step.

    train_step_graphed's chunked schedule as pinned above: forward at micro index 0 runs begin_step, which resets
    the buffer; the recompute replay re-runs the chunk's layers during its backward."""
    replay_counts, sweep_counts = "mInReplay" not in guard, "sequence_chunk_kv_sweep()" not in guard
    counts: dict[tuple[int, int], int] = {}

    def forward(micro: int, key: tuple[int, int], kv_sweep: bool) -> None:
        if micro == 0:
            counts.clear()
        if sweep_counts or not kv_sweep:
            counts[key] = counts.get(key, 0) + 1

    micro_base = 0
    for j, eff in enumerate(chunks_per_micro):
        for c in range(eff):
            forward(micro_base + c, (j, c), kv_sweep=True)
        for c in reversed(range(eff)):
            forward(micro_base + (eff - 1 - c), (j, c), kv_sweep=False)
            if recompute and replay_counts:
                counts[(j, c)] += 1
        micro_base += eff
    return counts


@pytest.mark.parametrize(
    "chunks_per_micro",
    [[1], [3], [2, 2], [4, 1, 3], [1, 1, 1, 1]],
    ids=lambda chunks: "chunks" + "-".join(map(str, chunks)),
)
@pytest.mark.parametrize("recompute", [False, True])
def test_every_routed_chunk_is_counted_once(chunks_per_micro, recompute):
    counts = reported_counts(chunks_per_micro, stats_guard(), recompute)
    expected = {(j, c): 1 for j, eff in enumerate(chunks_per_micro) for c in range(eff)}
    assert counts == expected


def test_the_model_shows_the_double_counts_without_the_guards():
    chunks = [3, 2, 2]
    first, later = [(0, c) for c in range(3)], [(j, c) for j in (1, 2) for c in range(2)]
    t19_replay_only = reported_counts(chunks, "mInReplay", recompute=False)
    assert all(t19_replay_only[k] == 1 for k in first) and all(t19_replay_only[k] == 2 for k in later)
    upstream = reported_counts(chunks, "", recompute=True)
    assert all(upstream[k] == 2 for k in first) and all(upstream[k] == 3 for k in later)
    assert reported_counts([1, 1], "", recompute=True) == {(0, 0): 2, (1, 0): 3}


def test_the_reported_sums_and_averages_use_slot_four():
    stats = body(RUN_STATE, r"IRunState::MoEStats DslRunState::get_moe_stats\(")
    assert "stats.aux_loss = mMoEStatsHost[0];" in stats and "stats.z_loss = mMoEStatsHost[1];" in stats
    assert "const int num_layers = static_cast<int>(mMoEStatsHost[4]);" in stats
    for field, slot in (("expert_utilization", 2), ("load_imbalance", 3), ("router_entropy", 9)):
        assert f"stats.{field} = mMoEStatsHost[{slot}] / num_layers;" in stats


def test_micro_step_zero_resets_the_stats_so_the_warmup_does_not_count():
    forward = body(EXECUTOR, r"ExecutionResult GraphExecutor::execute_forward\(")
    assert "if (request.micro_step == 0) { begin_step(); }" in forward
    assert "rs.reset_moe_stats();" in body(EXECUTOR, r"void GraphExecutor::begin_step\(")
