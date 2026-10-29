"""The checkpoint and export contract (README: "checkpoint 与导出").

Native checkpoint directories are imitated from what csrc/src/runtime/training/checkpoint.cpp writes; the pins at
the end fail when upstream changes that layout or the file list its final export copies.
"""

import ast
import json
from pathlib import Path

import pytest
from surogate_tlc import checkpoint, modeldir

REPO = Path(__file__).resolve().parents[4]
# Keys save_pretrained_config (csrc/src/config/pretrained_config.cpp) writes for a model like A0.
NATIVE_KEYS = (
    "architectures",
    "model_type",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "hidden_size",
    "intermediate_size",
    "vocab_size",
    "num_attention_heads",
    "num_key_value_heads",
    "num_hidden_layers",
    "head_dim",
    "max_position_embeddings",
    "rope_theta",
    "rms_norm_eps",
    "tie_word_embeddings",
)
NATIVE_FILES = ("weights.shard_000_of_001.safetensors", "adam.m.shard_000_of_001.safetensors", "model.safetensors")


def _native_checkpoint(root: Path, step: int, source: Path, *, complete: bool = True) -> Path:
    path = root / f"step_{step:08d}"
    path.mkdir(parents=True)
    config = json.loads((source / "config.json").read_text())
    (path / "config.json").write_text(json.dumps({k: config[k] for k in NATIVE_KEYS} | {"torch_dtype": "bfloat16"}))
    for name in NATIVE_FILES:
        (path / name).write_bytes(f"{name} at step {step}".encode())
    if complete:
        (path / "checkpoint.json").write_text(json.dumps({"run": {"step": step}, "distributed": {"world": 1}}))
    return path


def _snapshot(path: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(path.iterdir())}


def test_raw_checkpoint_is_not_a_model_dir(model_dir, tmp_path):
    step = _native_checkpoint(tmp_path / "ckpt", 10, model_dir)
    with pytest.raises(modeldir.ModelDirError) as error:
        modeldir.check_model_dir(step)
    problems = "\n".join(error.value.problems)
    for missing in ("auto_map", "tlc_layer_pattern", "tokenizer files", "tlc_meta.json"):
        assert missing in problems


def test_finalize_checkpoints_copies_the_source_contract(model_dir, tmp_path):
    root = tmp_path / "ckpt"
    step = _native_checkpoint(root, 10, model_dir)
    partial = _native_checkpoint(root, 20, model_dir, complete=False)
    partial_before = _snapshot(partial)

    assert checkpoint.finalize_checkpoints(root, model_dir) == [step]

    assert _snapshot(partial) == partial_before  # still being written: left alone
    config = modeldir.check_model_dir(step)
    assert config == json.loads((model_dir / "config.json").read_text())
    for name in modeldir.companion_files(model_dir):
        if name != "tlc_meta.json":
            assert (step / name).read_bytes() == (model_dir / name).read_bytes(), name
    for name in NATIVE_FILES:
        assert (step / name).read_bytes() == f"{name} at step 10".encode()
    meta = json.loads((step / "tlc_meta.json").read_text())
    assert meta == json.loads((model_dir / "tlc_meta.json").read_text()) | {"step": 10}

    once = _snapshot(step)
    checkpoint.finalize_dir(step, model_dir, meta={"step": 10})
    assert _snapshot(step) == once
    assert not [p.name for p in step.iterdir() if p.name.endswith(".tmp")]


def test_finalize_refuses_a_different_model(model_dir, tmp_path, tokenizer_dir):
    from surogate_tlc import hf

    step = _native_checkpoint(tmp_path / "ckpt", 10, model_dir)
    before = _snapshot(step)
    other = modeldir.write_model_dir(tmp_path / "other", hf.default_config(hidden_size=512), tokenizer_dir)
    with pytest.raises(ValueError, match=r"disagrees .* on \['hidden_size'\]"):
        checkpoint.finalize_dir(step, other)
    assert _snapshot(step) == before


def test_export_writes_a_model_dir_from_one_checkpoint(model_dir, tmp_path):
    step = _native_checkpoint(tmp_path / "ckpt", 10, model_dir)
    out = checkpoint.export(step, model_dir, tmp_path / "export")

    assert sorted(p.name for p in out.iterdir()) == sorted([*modeldir.companion_files(model_dir), "model.safetensors"])
    assert (out / "model.safetensors").read_bytes() == (step / "model.safetensors").read_bytes()
    modeldir.check_model_dir(out)
    assert json.loads((out / "tlc_meta.json").read_text())["step"] == 10

    with pytest.raises(FileExistsError):
        checkpoint.export(step, model_dir, out)
    partial = _native_checkpoint(tmp_path / "ckpt", 20, model_dir, complete=False)
    with pytest.raises(ValueError, match="not a complete native checkpoint"):
        checkpoint.export(partial, model_dir, tmp_path / "export-20")


def test_upstream_final_export_needs_finalizing(model_dir, tmp_path):
    # SurogateTrainerWrapper.train() ends with export_model(output_dir), then copies this list from model_dir.
    out = tmp_path / "output"
    out.mkdir()
    (out / "model.safetensors").write_bytes(b"exported weights")
    for name in _upstream_export_copies():
        if (model_dir / name).is_file():
            (out / name).write_bytes((model_dir / name).read_bytes())
    with pytest.raises(modeldir.ModelDirError, match="configuration_tlc.py"):
        modeldir.check_model_dir(out)

    checkpoint.finalize_dir(out, model_dir)
    modeldir.check_model_dir(out)


FORK_LESS_LOAD = """
import json
from transformers import AutoConfig, AutoTokenizer

path = sys.argv[2]
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("surogate", "surogate_tlc"))
print(json.dumps({"pattern": config.tlc_layer_pattern, "eos": tokenizer.eos_token_id, "fork_modules": loaded}))
"""


def test_exported_dir_loads_without_the_fork(fresh_python, model_dir, tmp_path):
    step = _native_checkpoint(tmp_path / "ckpt", 10, model_dir)
    out = checkpoint.export(step, model_dir, tmp_path / "export")
    report = json.loads(fresh_python(FORK_LESS_LOAD, out).splitlines()[-1])
    config = json.loads((model_dir / "config.json").read_text())
    assert report == {"pattern": config["tlc_layer_pattern"], "eos": config["eos_token_id"], "fork_modules": []}


def _upstream_export_copies() -> list[str]:
    tree = ast.parse((REPO / "surogate/train/trainer.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_copy_tokenizer_files":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.List):
                    return [ast.literal_eval(element) for element in stmt.value.elts]
    raise AssertionError("SurogateTrainerWrapper._copy_tokenizer_files is gone: recheck the export path")


def test_pin_upstream_export_file_list():
    copies = _upstream_export_copies()
    assert {"config.json", "tokenizer.json", "tokenizer_config.json"} <= set(copies)
    assert not {"configuration_tlc.py", "tlc_meta.json"} & set(copies)  # why finalize_dir() exists


def test_pin_native_checkpoint_layout():
    source = (REPO / "csrc/src/runtime/training/checkpoint.cpp").read_text(encoding="utf-8")
    assert 'fmt::format("/step_{:08}", step)' in source
    assert checkpoint.STEP_DIR.fullmatch("step_00000010")
    for written in ('"/config.json"', '"/model.safetensors"', f'"/{checkpoint.NATIVE_DONE_FILE}"'):
        assert written in source, written
    # checkpoint.json is written after the barrier that follows every shard, so it marks a complete directory.
    assert source.index("comm.barrier();  // only write checkpoint.json") < source.index('"/checkpoint.json"')
