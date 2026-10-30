"""CPU tests of ledger topic T10's dispatch-PP refusal (surogate/_fork/dispatch_pp_tied_head.py).

Dispatch-PP full training of a tied model keeps the whole tied gradient only when stage 0 and the loss stage run on
one GPU. The trainers refuse the other plans. These tests check the refusal against a replay of the native gradient
collection over the real stage planner, the tie resolution against the real DSL IR, the message's alternatives, and
pin the native code the refusal describes, so that an upstream change there fails here and prompts a review.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_tied_lm_head import EMBEDDINGGEMMA_300M, GEMMA3_270M_IT, REPO, body

sys.path.insert(0, str(REPO))

from surogate._fork.dispatch_pp_tied_head import (  # noqa: E402
    STAGE_BLOCKS_ENV,
    loses_tied_gradient,
    native_tie_word_embeddings,
    refuse_lost_tied_gradient,
)
from surogate.dsl.ir_builder import build_dsl_ir_from_python, resolve_architecture  # noqa: E402
from surogate.train.dispatch_pp import plan_stages  # noqa: E402

#: A Qwen3-0.6B-shaped config (28 layers), narrowed so the IR builds fast; the tie key is set per test.
QWEN3_28L = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 28,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 256,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
}


def ir_for(config: dict) -> str:
    return build_dsl_ir_from_python(resolve_architecture(config), config)


def plan(monkeypatch, layers: int, gpus: int, stage_blocks: int | None):
    if stage_blocks is None:
        monkeypatch.delenv(STAGE_BLOCKS_ENV, raising=False)
    else:
        monkeypatch.setenv(STAGE_BLOCKS_ENV, str(stage_blocks))
    return plan_stages(SimpleNamespace(num_hidden_layers=layers), gpus)


def replay_tied_update(los: list[int], gpus: int) -> set[str]:
    """What GPU 0's tied gradient buffer holds when the optimizer runs, per the native code pinned below.

    Each GPU's buffer (embedding and lm_head are one buffer) sums the parts computed there: the embedding lookup's
    in stage 0's backward, the head's in the loss stage's. dispatch_pp_read_block_grads reads `embedding` with the
    stage whose first layer is 0 and `lm_head` with the loss stage, each from its stage's GPU (s % gpus), and
    dispatch_pp_write_grads copies the reads into GPU 0's one buffer in stage order, so the last read wins.
    """
    stages = len(los)
    buffers = {gpu: set() for gpu in range(gpus)}
    for stage in range(stages):
        if los[stage] == 0:
            buffers[stage % gpus].add("embedding lookup")
        if stage == stages - 1:
            buffers[stage % gpus].add("lm_head")
    tied = set(buffers[0])
    for stage in range(stages):
        for wanted in (los[stage] == 0, stage == stages - 1):
            if wanted:
                tied = set(buffers[stage % gpus])
    return tied


@pytest.mark.parametrize(
    ("layers", "gpus", "stage_blocks", "loses"),
    [
        (28, 2, 14, True),  # Qwen3-0.6B/1.7B, 2 stages
        (28, 4, None, True),  # 7 stages, loss stage on GPU 2
        (28, 8, None, True),  # stage size shrinks to 3: 10 stages, loss stage on GPU 1
        (36, 3, None, True),  # Qwen3-4B
        (16, 2, None, True),  # Llama 3.2 1B
        (16, 4, None, True),
        (28, 1, None, False),
        (28, 2, None, False),
        (28, 3, None, False),
        (36, 2, None, False),
        (36, 4, None, False),
        (36, 8, None, False),
        (16, 1, None, False),
        (16, 3, None, False),
    ],
)
def test_the_documented_plans(monkeypatch, layers, gpus, stage_blocks, loses):
    """The configurations fork/docs/upstream-errata.md lists as affected and unaffected."""
    los, _, _, stages, _, _ = plan(monkeypatch, layers, gpus, stage_blocks)
    assert loses_tied_gradient(tied=True, lora=False, num_stages=stages, gpus=gpus) is loses
    assert (replay_tied_update(los, gpus) != {"embedding lookup", "lm_head"}) is loses


def test_the_refusal_matches_the_replay_for_every_small_plan(monkeypatch):
    for layers in range(1, 41):
        for gpus in range(1, 9):
            for stage_blocks in (None, *range(1, layers + 1)):
                los, _, _, stages, _, _ = plan(monkeypatch, layers, gpus, stage_blocks)
                lost = replay_tied_update(los, gpus) != {"embedding lookup", "lm_head"}
                assert loses_tied_gradient(tied=True, lora=False, num_stages=stages, gpus=gpus) is lost, (
                    layers,
                    gpus,
                    stage_blocks,
                )
                assert not loses_tied_gradient(tied=True, lora=True, num_stages=stages, gpus=gpus)
                assert not loses_tied_gradient(tied=False, lora=False, num_stages=stages, gpus=gpus)


@pytest.mark.parametrize(
    ("config", "tied"),
    [
        (dict(QWEN3_28L, tie_word_embeddings=True), True),
        (dict(QWEN3_28L, tie_word_embeddings=False), False),
        (QWEN3_28L, False),  # no key, nothing declared
        (GEMMA3_270M_IT, True),  # no key, declared by surogate/dsl/models/gemma3.py
        (EMBEDDINGGEMMA_300M, True),
        (dict(GEMMA3_270M_IT, tie_word_embeddings=False), False),  # an explicit key wins over the declaration
    ],
    ids=["qwen3-tied", "qwen3-untied", "qwen3-no-key", "gemma-3-270m-it", "embeddinggemma-300m", "gemma3-untied"],
)
def test_the_tie_resolves_as_the_native_trainer_resolves_it(config, tied):
    assert native_tie_word_embeddings(config, json.loads(ir_for(config))) is tied


def test_the_loader_rules_without_an_ir():
    """load_pretrained_config's order: LFM2 tied, tie_word_embeddings (specified), tie_embeddings (not specified)."""

    def model(**declared):
        return {"modules": [{"kind": "model", "config": declared}]}

    assert native_tie_word_embeddings({"model_type": "lfm2"}, model())
    assert native_tie_word_embeddings({"architectures": ["Lfm2ForCausalLM"]}, model())
    assert not native_tie_word_embeddings({"model_type": "lfm2", "tie_word_embeddings": "false"}, model())
    assert native_tie_word_embeddings({"tie_word_embeddings": 1}, model(tie_word_embeddings=False))
    assert native_tie_word_embeddings({"tie_embeddings": True}, model())
    # tie_embeddings does not mark the value specified, so a declaration overrides it.
    assert not native_tie_word_embeddings({"tie_embeddings": True}, model(tie_word_embeddings=False))
    assert not native_tie_word_embeddings({"tie_word_embeddings": None}, model(tie_word_embeddings="yes"))
    assert native_tie_word_embeddings({"tie_word_embeddings": None}, model(tie_word_embeddings=True))


def trainer_config(tmp_path: Path, model: dict, *, gpus: int, lora: bool = False, parallelism="dispatch_pp"):
    (tmp_path / "config.json").write_text(json.dumps(model), encoding="utf-8")
    return SimpleNamespace(
        parallelism=parallelism,
        lora=lora,
        gpus=gpus,
        model_dir=str(tmp_path),
        model_info=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=model["num_hidden_layers"])),
    )


def test_a_plan_that_loses_the_lookup_gradient_is_refused_with_alternatives_that_work(tmp_path, monkeypatch):
    model = dict(QWEN3_28L, tie_word_embeddings=True)
    ir = ir_for(model)
    monkeypatch.delenv(STAGE_BLOCKS_ENV, raising=False)
    config = trainer_config(tmp_path, model, gpus=4)
    with pytest.raises(ValueError) as refused:
        refuse_lost_tied_gradient(config, ir)
    message = str(refused.value)
    assert "7 stages of 4 layers on 4 GPUs" in message and "loss stage on GPU 2" in message
    assert message.endswith(
        f"Use one of: {STAGE_BLOCKS_ENV}=6 (5 stages); 3 GPUs with {STAGE_BLOCKS_ENV}=4; lora: true."
    )
    # Each alternative passes.
    monkeypatch.setenv(STAGE_BLOCKS_ENV, "6")
    refuse_lost_tied_gradient(config, ir)
    monkeypatch.setenv(STAGE_BLOCKS_ENV, "4")
    refuse_lost_tied_gradient(trainer_config(tmp_path, model, gpus=3), ir)
    monkeypatch.delenv(STAGE_BLOCKS_ENV)
    refuse_lost_tied_gradient(trainer_config(tmp_path, model, gpus=4, lora=True), ir)


@pytest.mark.parametrize(("layers", "gpus", "stage_blocks"), [(28, 2, 14), (28, 8, None), (36, 3, None), (16, 4, None)])
def test_each_named_alternative_keeps_stage_zero_and_the_loss_stage_together(
    tmp_path, monkeypatch, layers, gpus, stage_blocks
):
    model = dict(QWEN3_28L, num_hidden_layers=layers, tie_word_embeddings=True)
    ir = ir_for(model)
    plan(monkeypatch, layers, gpus, stage_blocks)
    with pytest.raises(ValueError) as refused:
        refuse_lost_tied_gradient(trainer_config(tmp_path, model, gpus=gpus), ir)
    options = str(refused.value).split("Use one of: ", 1)[1].rstrip(".").split("; ")
    assert options[-1] == "lora: true"
    for option in options[:-1]:
        if option.startswith(STAGE_BLOCKS_ENV):
            size, count = int(option.split("=")[1].split()[0]), gpus
        else:
            count, size = int(option.split()[0]), int(option.rsplit("=", 1)[1])
        assert count < gpus or option.startswith(STAGE_BLOCKS_ENV)
        los, _, _, _, _, _ = plan(monkeypatch, layers, count, size)
        assert replay_tied_update(los, count) == {"embedding lookup", "lm_head"}, option
        refuse_lost_tied_gradient(trainer_config(tmp_path, model, gpus=count), ir)


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (dict(QWEN3_28L, tie_word_embeddings=False), {}),  # untied
        (dict(QWEN3_28L, tie_word_embeddings=True), {"lora": True}),
        (dict(QWEN3_28L, tie_word_embeddings=True), {"parallelism": None}),
        (dict(QWEN3_28L, tie_word_embeddings=True), {"gpus": 2}),  # 7 stages on 2 GPUs: loss stage on GPU 0
    ],
    ids=["untied", "lora", "ddp", "loss-stage-on-gpu0"],
)
def test_other_runs_are_left_alone(tmp_path, monkeypatch, model, kwargs):
    monkeypatch.delenv(STAGE_BLOCKS_ENV, raising=False)
    refuse_lost_tied_gradient(trainer_config(tmp_path, model, **{"gpus": 4, **kwargs}), ir_for(model))


def test_a_tied_gemma3_declaration_is_refused_too(tmp_path, monkeypatch):
    """gemma-3-270m-it (18 layers, no tie key): 5 stages on 4 GPUs put the loss stage on GPU 0 and pass; on 3 GPUs
    it runs on GPU 1, and the plan is refused, because the native trainer takes the declared tie."""
    ir = ir_for(GEMMA3_270M_IT)
    monkeypatch.delenv(STAGE_BLOCKS_ENV, raising=False)
    refuse_lost_tied_gradient(trainer_config(tmp_path, GEMMA3_270M_IT, gpus=4), ir)
    with pytest.raises(
        ValueError, match="5 stages of 4 layers on 3 GPUs run stage 0 on GPU 0 and the loss stage on GPU 1"
    ):
        refuse_lost_tied_gradient(trainer_config(tmp_path, GEMMA3_270M_IT, gpus=3), ir)


def test_both_trainers_refuse_before_building_the_native_trainer():
    """The SFT and GRPO trainers run the refusal right after building the IR, before any native object exists."""
    for path in ("surogate/train/trainer.py", "surogate/grpo/trainer.py"):
        source = (REPO / path).read_text(encoding="utf-8")
        built = source.index("config.runtime_config.dsl_ir_json = ir_json")
        refused = source.index("refuse_lost_tied_gradient(config, ir_json)")
        assert built < refused < source.index("_surogate.SurogateTrainer("), path
        assert source.count("refuse_lost_tied_gradient(config, ir_json)") == 1, path


def test_the_native_collection_the_refusal_describes():
    """The refusal replays this code. When upstream makes dispatch-PP sum the tied gradient's copies (or otherwise
    changes how it reads and writes them), this pin fails: re-check the replay, and drop the refusal if the loss is
    gone (fork/docs/sync.md, T10)."""
    step = body(REPO / "csrc/src/binding/py_train.cpp", r"float MultiGPUPyTrainer::dispatch_pp_train_step_multigpu\(")
    collect = step[step.index("std::vector<std::pair<std::string, std::vector<std::byte>>> collected;") :]
    assert "for (int si = 0; si < num_stages; ++si) { const int gpu = si % ngpu;" in collect
    assert "const bool is_loss = (si == num_stages - 1);" in collect
    assert "/*include_head=*/is_loss, /*include_embed=*/lo == 0);" in collect
    execution = REPO / "csrc/src/runtime/dsl/dsl_model_execution.cpp"
    read = body(execution, r"DslModel::dispatch_pp_read_block_grads\(")
    assert 'else if (name.find("embed") != std::string::npos) { wanted = include_embed;' in read
    assert "} else { wanted = include_head;" in read
    write = body(execution, r"void DslModel::dispatch_pp_write_grads\(")
    full = write[write.index("if (!mGrads) return;") :]
    assert "Tensor* g = mGrads->get_param_grad(name, accumulate);" in full
    assert "CUDA_CHECK(cudaMemcpyAsync(g->Data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice, stream));" in full
    assert full.count("cudaMemcpyAsync") == 1 and "vector_add" not in full
    apply = body(REPO / "csrc/src/binding/py_train.cpp", r"void MultiGPUPyTrainer::dispatch_pp_apply_grads_\(")
    assert "model->dispatch_pp_write_grads(collected);" in apply
