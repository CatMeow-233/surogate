"""Source pins of ledger topic T19 (MoE routing statistics counted once under recompute), runnable without CUDA.

With recompute on (the default), the backward replays each layer's forward ops (replay_layer_forward sets
mInReplay), MoETopK included. Its statistics block added every layer to moe_stats_device a second time: the summed
aux_loss and z_loss read 2x, while the per-layer averages divide by a slot-4 layer count that doubled too and stayed
the same. The fix skips the block in replay. The kernels are GPU code, so these pins check the guard and the upstream
code it relies on; the GPU check is in fork/docs/upstream-errata.md. The roadmap's other half of the claim (the
full-step warmup also double counts) is wrong and is pinned here as well: micro-step 0 of the real pass resets the
buffer after the warmup.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
MOE_TOPK = REPO / "csrc/src/runtime/ops/moe_topk.cpp"
REPLAY = REPO / "csrc/src/runtime/executor/compiled_ops_execute.cpp"
EXECUTOR = REPO / "csrc/src/runtime/executor/graph_executor.cpp"
RUN_STATE = REPO / "csrc/src/runtime/dsl/dsl_run_state.cpp"


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


def test_the_routing_stats_skip_the_recompute_replay():
    topk = body(MOE_TOPK, r"void CompiledExecutor::dispatch_moe_topk\(")
    assert topk.count("moe_stats_device()") == 1, "a second statistics site: guard it too"
    assert "if (float* stats = mInReplay ? nullptr : mRunState.moe_stats_device()) {" in topk


def test_replay_runs_the_layer_forward_ops_in_replay_mode():
    replay = body(REPLAY, r"void CompiledExecutor::replay_layer_forward\(")
    start, dispatch, end = replay.find("mInReplay = true;"), replay.find("op.fn(*this, op,"), replay.rfind("mInReplay = false;")
    assert -1 < start < dispatch < end, "replay_layer_forward no longer dispatches its ops between mInReplay = true/false"
    # Loss ops are skipped in replay; MoETopK is not, which is what made the statistics count twice.
    assert "CompiledOpType::CrossEntropyLoss" in replay and "CompiledOpType::MoETopK" not in replay


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
