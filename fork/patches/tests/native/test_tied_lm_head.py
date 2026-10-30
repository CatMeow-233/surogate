"""Source pins of ledger topic T10 (tied lm_head in full training), runnable without CUDA.

The fix edits upstream C++ and the Gemma 3 declaration; the native suite (fork/native/tests/test-fork-tied-lm-head.cpp)
and the TLC gate G2 check the behaviour on a GPU. These pins fail on a CPU-only machine when a sync drops or reshapes a
hunk, or when upstream changes the code the fix relies on: the optimizers and both gradient norms handling a shared
gradient once, the lm_head gradient being the embedding's, a missing tie_word_embeddings key resolving to the
declaration, and a checkpoint without lm_head.weight loading the embedding into a head that is not aliased. Each pin
reads one function body, so moved lines do not matter.
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
DSL = REPO / "csrc/src/runtime/dsl"
OPTIMIZERS = REPO / "csrc/src/runtime/optimizers"

#: google/gemma-3-270m-it's published config.json, copied from the record in upstream's
#: tests/serve/test_gemma3_convert.py. No test checks that record against the published file:
#: tests/test_gemma3_dsl.py compares its own 17-key subset (GEMMA3_270M_CONFIG) key by key, and only when the gated
#: checkpoint is cached, and no upstream test checks that the tie key is absent. That absence (HF's
#: Gemma3TextConfig defaults tie_word_embeddings to True, so the official files omit it) is what the Gemma 3
#: declaration exists for; test_the_published_gemma3_configs_still_omit_the_tie checks it when the checkpoints are
#: cached, and is skipped otherwise (always in fork CI, which runs offline).
GEMMA3_270M_IT = {
    "architectures": ["Gemma3ForCausalLM"],
    "model_type": "gemma3_text",
    "hidden_activation": "gelu_pytorch_tanh",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_logit_softcapping": None,
    "final_logit_softcapping": None,
    "use_bidirectional_attention": False,
    "rms_norm_eps": 1e-6,
    "rope_scaling": None,
    "rope_theta": 1000000.0,
    "rope_local_base_freq": 10000.0,
    "query_pre_attn_scalar": 256,
    "max_position_embeddings": 32768,
    "sliding_window": 512,
    "_sliding_window_pattern": 6,
    "layer_types": ["full_attention" if layer in (5, 11, 17) else "sliding_attention" for layer in range(18)],
    "num_hidden_layers": 18,
    "hidden_size": 640,
    "intermediate_size": 2048,
    "vocab_size": 262144,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 256,
}

#: google/embeddinggemma-300m's published config.json (Gemma3TextModel), copied from tests/test_gemma3_dsl.py's
#: EMBEDDINGGEMMA_CONFIG. That test compares its record with the published file key by key (all keys but
#: architectures) when the checkpoint is cached; it does not check that the tie key is absent.
EMBEDDINGGEMMA_300M = {
    "architectures": ["Gemma3TextModel"],
    "model_type": "gemma3_text",
    "_sliding_window_pattern": 6,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 768,
    "intermediate_size": 1152,
    "max_position_embeddings": 2048,
    "num_attention_heads": 3,
    "num_hidden_layers": 24,
    "num_key_value_heads": 1,
    "query_pre_attn_scalar": 256,
    "rms_norm_eps": 1e-06,
    "rope_local_base_freq": 10000.0,
    "rope_theta": 1000000.0,
    "sliding_window": 512,
    "use_bidirectional_attention": True,
    "vocab_size": 262144,
}


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
    # The norm is taken over those host copies, so the shared one is counted once by cpu_adamw.cpp's pointer
    # dedupe (pinned in test_upstream_still_sums_the_tied_gradient_and_updates_it_once).
    assert "cpu_gradient_norm_squared(model.mGrads->get_cpu_grads_map(), model.mGrads->param_names())" in streaming


def test_upstream_still_sums_the_tied_gradient_and_updates_it_once():
    """What T10 relies on: the lm_head gradient is the embedding's buffer, and each GPU optimizer, the device
    gradient norm and the cpu_training gradient norm visit a gradient pointer once. Without the skip, the shared
    matrix would take two steps, or its gradient would count twice in the norm that clipping uses."""
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
    # cpu_training: T10 gives lm_head the embedding's host gradient copy (allocate_streaming_buffers); upstream's
    # CPU norm skips a host pointer it has seen, which upstream itself never exercised (its host copies were distinct).
    cpu_norm = body(OPTIMIZERS / "cpu_adamw.cpp", r"double cpu_gradient_norm_squared\(")
    assert "if (seen.count(it->second.Data)) continue; seen.insert(it->second.Data);" in cpu_norm
    assert cpu_norm.count("for (const auto& name : param_names)") == 1


def test_a_tied_export_still_loads_where_the_head_has_its_own_storage():
    """T10's exports and weight shards have no lm_head.weight, as HF writes a tied model. Full training aliases the
    head and skips it on import, but QLoRA and LoRA with the weight manager keep a separate head and fill it through
    the loader's tied fallback, which upstream added for HF checkpoints and T10 relies on, unedited."""
    load = body(DSL / "dsl_weight_loader.cpp", r"bool DslWeightLoader::load_direct\(")
    fallback = load[load.index('if (mConfig.TiedWordEmbeddings && hf_name == "lm_head.weight") {') :]
    assert 'if (const auto* emb_spec = find_mapping_spec("embedding", emb_layer_idx)) {' in fallback
    assert "candidates.push_back(format_hf_name(emb_spec->source, emb_layer_idx));" in fallback
    assert 'candidates.push_back("model.embed_tokens.weight");' in fallback
    assert "entry_ptr = &mReader.find_entry(candidate); break;" in fallback


@pytest.mark.parametrize(
    "config", [GEMMA3_270M_IT, EMBEDDINGGEMMA_300M], ids=["gemma-3-270m-it", "embeddinggemma-300m"]
)
def test_official_gemma3_configs_resolve_as_tied(config):
    """T10 is gated on PretrainedConfig.TiedWordEmbeddings. The official Gemma 3 configs omit the key, so the native
    trainer takes the value the DSL declaration carries (test below). Without tie_word_embeddings in
    surogate/dsl/models/gemma3.py they resolved as untied: lm_head, mapped to the embedding tensor, was trained as a
    second matrix, and the export wrote model.embed_tokens.weight twice and threw."""
    sys.path.insert(0, str(REPO))
    from surogate.dsl.ir_builder import build_dsl_ir_from_python, resolve_architecture

    assert "tie_word_embeddings" not in config and "tie_embeddings" not in config
    module = json.loads(build_dsl_ir_from_python(resolve_architecture(config), config))["modules"][0]
    assert module["config"].get("tie_word_embeddings") is True
    assert module["hf_mapping"]["lm_head"] == module["hf_mapping"]["embedding"]


def hub_cache() -> Path:
    """The Hugging Face hub cache, as huggingface_hub locates it."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"


@pytest.mark.parametrize("repo", ["gemma-3-270m-it", "embeddinggemma-300m"])
def test_the_published_gemma3_configs_still_omit_the_tie(repo):
    """The premise of the Gemma 3 declaration, checked against the published file when it is cached (the repos are
    gated). If Google starts writing the key, the native trainer reads it from config.json and the declaration only
    matters for older snapshots: update the errata and the ledger T10 notes, which say the official files omit it."""
    published = sorted((hub_cache() / f"models--google--{repo}" / "snapshots").glob("*/config.json"))
    if not published:
        pytest.skip(f"google/{repo} not cached (gated repo)")
    for path in published:
        config = json.loads(path.read_text(encoding="utf-8"))
        assert "tie_word_embeddings" not in config and "tie_embeddings" not in config, path


def test_a_missing_tie_resolves_to_the_declaration():
    """What the Gemma 3 declaration relies on: load_pretrained_config marks the tie as specified only when
    config.json has the key, the IR's module config is the declaration's, and DslModel takes the declared value for
    an unspecified tie before it builds the stores that read the flag."""
    loader = body(REPO / "csrc/src/config/pretrained_config.cpp", r"PretrainedConfig> load_pretrained_config\(")
    assert (
        'if (auto v = get_opt<bool>(config_json, "tie_word_embeddings")) { cfg->TiedWordEmbeddings = *v; '
        "cfg->TiedWordEmbeddingsSpecified = true; }" in loader
    )
    assert loader.count("TiedWordEmbeddingsSpecified") == 1
    ir = body(DSL / "ir.cpp", r"IRFile load_ir_from_json\(")
    assert 'mod.config = parse_attr_map(mod_json["config"]);' in ir
    declared = body(DSL / "dsl_model.cpp", r"void apply_declared_config_defaults\(")
    assert (
        "if (!cfg.TiedWordEmbeddingsSpecified) { if (const auto* value = internal::find_key(&module.config, "
        '"tie_word_embeddings")) { if (auto tied = internal::as_bool(*value)) { cfg.TiedWordEmbeddings = *tied;'
        in declared
    )
    model = body(DSL / "dsl_model.cpp", r"DslModel::DslModel\(")
    resolved = model.index("apply_declared_config_defaults(*mConfig, *mModule);")
    for built in (
        "std::make_unique<DslParamStore>(",
        "std::make_unique<DslGradStore>(",
        "std::make_unique<DslWeightManager>(",
    ):
        assert resolved < model.index(built), built
