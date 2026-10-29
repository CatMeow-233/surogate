"""Source pins of ledger topic T10 (tied lm_head in full training), runnable without CUDA.

The fix edits upstream C++; the native suite (fork/native/tests/test-fork-tied-lm-head.cpp) and the TLC gate G2 check
the behaviour on a GPU. These pins fail on a CPU-only machine when a sync drops or reshapes a hunk, or when upstream
changes the code the fix relies on: the optimizers updating a shared gradient once, and the lm_head gradient being
the embedding's. Each pin reads one function body, so moved lines do not matter.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
DSL = REPO / "csrc/src/runtime/dsl"
OPTIMIZERS = REPO / "csrc/src/runtime/optimizers"


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


def statement(code: str, target: str) -> str:
    """The right-hand side of the assignment to `target` in `code`."""
    match = re.search(re.escape(target) + r"\s*=\s*([^;]*);", code)
    assert match, f"no assignment to {target}"
    return match.group(1)


@pytest.fixture(scope="module")
def param_store() -> str:
    return body(DSL / "dsl_param_store.cpp", r"DslParamStore::DslParamStore\(")


def test_full_training_aliases_a_tied_head(param_store):
    """The alias no longer needs a frozen base: only the weight-manager case still depends on freeze_base."""
    alias = statement(param_store, "entry.storage_alias")
    for term in (
        "config.TiedWordEmbeddings",
        'name == "lm_head"',
        'graph.params.contains("embedding")',
        "!entry.external",
    ):
        assert term in alias
    assert "(!mUsesWeightManager || !freeze_base)" in alias
    assert "freeze_base" not in alias.replace("(!mUsesWeightManager || !freeze_base)", "")
    assert "head = embedding;" in param_store
    assert "head.Rank != embedding.Rank || head.DType != embedding.DType" in param_store


def test_an_unshareable_trainable_head_is_refused(param_store):
    refusal = param_store[param_store.index("if (config.TiedWordEmbeddings && !freeze_base)") :]
    assert '{"lm_head", "lm_head_weight"}' in refusal
    assert "it->second.trainable && !it->second.storage_alias" in refusal
    assert "throw std::runtime_error" in refusal


def test_the_store_and_the_weight_manager_agree():
    wiring = body(DSL / "dsl_param_store.cpp", r"void DslParamStore::set_weight_manager\(")
    assert 'manager->is_tied_alias("lm_head")' in wiring and "throw std::logic_error" in wiring
    header = (DSL / "dsl_param_store.h").read_text(encoding="utf-8")
    assert "void set_weight_manager(DslWeightManager* manager);" in header


def test_checkpoints_hold_a_tied_head_once():
    store = body(DSL / "dsl_param_store.cpp", r"void DslParamStore::iterate_tensors\(")
    assert "if (it->second.storage_alias) { continue;" in store
    manager = body(DSL / "dsl_weight_manager.cpp", r"void DslWeightManager::iterate_tensors\(")
    assert "if (it->second.tied_alias) continue;" in manager


def test_the_weight_manager_shares_master_and_work():
    source = DSL / "dsl_weight_manager.cpp"
    constructor = body(source, r"DslWeightManager::DslWeightManager\(")
    assert (
        statement(constructor, "mConfig.tie_lm_head")
        == "config.TiedWordEmbeddings && !(lora_config && lora_config->enabled())"
    )
    allocate = body(source, r"void DslWeightManager::allocate_weights\(")
    assert 'if (tie_head && name == "lm_head") { entry.tied_alias = true;' in allocate
    for line in (
        "head.master = embedding.master;",
        "head.master_sharded = embedding.master_sharded;",
        "head.work = embedding.work;",
    ):
        assert line in allocate
    # The alias is made before cpu_training points both names at the shared staging buffer.
    assert allocate.index("head.work = embedding.work;") < allocate.index("embedding_lm_head_work_shared")


def test_exports_write_a_tied_head_once():
    source = DSL / "dsl_model_weights.cpp"
    export = body(source, r"void DslModel::export_weights\(")
    assert "if (!mParams->is_storage_alias(name)) names.push_back(name);" in export
    assert export.count("for (const auto& name : names)") == 2
    assert "for (const auto& name : mParams->param_names())" in export  # only the loop that builds `names`
    assert export.count("mParams->param_names()") == 1
    load = body(source, r"void DslModel::import_weights\(")
    assert "mParams->is_external(name) || (mParams->has(name) && mParams->is_storage_alias(name))" in load
    assert "if (mParams->is_external(name) || mParams->is_storage_alias(name)) { continue;" in load
    init = body(source, r"void DslModel::init_weights\(")
    assert "if (mParams->is_external(name) || mParams->is_storage_alias(name)) { continue;" in init


def test_cpu_training_steps_a_shared_master_once_and_counts_its_gradient_once():
    streaming = body(OPTIMIZERS / "adamw_8bit_optimizer.cpp", r"void AdamW8BitOptimizer::step_cpu_streaming\(")
    assert "if (master.Data && !stepped.insert(master.Data).second) continue;" in streaming
    assert streaming.count("for (const auto& name : names)") == 2
    assert "for (const auto& name : model.mGrads->param_names())" in streaming  # only the loop that builds `names`
    host = body(DSL / "dsl_grad_store.cpp", r"void DslGradStore::allocate_streaming_buffers\(")
    assert "mCpuGrads.emplace(name, shared->second);" in host
    assert "if (gpu_grad.Data) host_of_device.emplace(gpu_grad.Data, host);" in host


def test_upstream_still_sums_the_tied_gradient_and_updates_it_once():
    """What T10 relies on: the lm_head gradient is the embedding's buffer, and each GPU optimizer, and the device
    gradient norm, visit a gradient pointer once. Without the skip, the shared matrix would take two steps."""
    grads = body(DSL / "dsl_grad_store.cpp", r"DslGradStore::DslGradStore\(")
    assert "mGrads.emplace(name, emb_it->second);" in grads
    adamw8 = body(OPTIMIZERS / "adamw_8bit_optimizer.cpp", r"void AdamW8BitOptimizer::init_state\(")
    assert "!seen_grads.insert(static_cast<void*>(grad->Data)).second" in adamw8
    for name, optimizer in (("adamw_optimizer.cpp", "AdamWOptimizer"), ("normuon_optimizer.cpp", "NorMuonOptimizer")):
        for step in ("step", "step_graph"):
            update = body(OPTIMIZERS / name, rf"void {optimizer}::{step}\(")
            assert "if (seen_grad_ptrs.count(grad->Data) > 0)" in update, (name, step)
    norm = body(DSL / "dsl_model.cpp", r"void DslModel::calculate_gradient_norm\(")
    assert "if (!seen_ptrs.insert(grad.Data).second) continue;" in norm
