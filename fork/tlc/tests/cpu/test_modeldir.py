"""A model directory loads through the trainer's Python path in a fresh interpreter, both with
surogate_tlc.register() and with only its auto_map (the fork imported just for the DSL model); plus the
model-directory checks that compare config.json with the tokenizer.
"""

import json
from pathlib import Path

import pytest
from surogate_tlc import hf, modeldir

SRC = Path(__file__).resolve().parents[2] / "src"

# get_model_info_and_tokenizer is what the trainer calls first; build_dsl_ir_for_model builds the IR it compiles.
LOAD = """
import json
from surogate.core.model.registry import get_model_info_and_tokenizer
from surogate.dsl.ir_builder import build_dsl_ir_for_model
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

model_dir, src, mode = sys.argv[2:5]
sys.path.insert(0, src)
if mode == "register":
    import surogate_tlc

    surogate_tlc.register()
info, tokenizer = get_model_info_and_tokenizer(model_dir)
report = {"fork_imported_for_config": any(m.split(".")[0] == "surogate_tlc" for m in sys.modules)}
import surogate_tlc.dsl  # the DSL model; a no-op after register()

ir = json.loads(build_dsl_ir_for_model(model_dir))["modules"][0]
report |= {
    "tlc_registered": "tlc" in CONFIG_MAPPING,
    "config_class": type(info.config).__module__,
    "layer_pattern": info.config.tlc_layer_pattern,
    "max_model_len": info.max_model_len,
    "token_ids": [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id],
    "ir_model": ir["name"],
    "ir_pattern": ir["config"]["hybrid_pattern"],
    "ir_source_architectures": ir["hf_config"]["source_config"]["architectures"],
}
print(json.dumps(report))
"""


@pytest.mark.parametrize("mode", ["register", "auto_map"])
def test_model_dir_loads_in_a_fresh_interpreter(fresh_python, model_dir, mode):
    report = json.loads(fresh_python(LOAD, model_dir, SRC, mode).splitlines()[-1])
    config = json.loads((model_dir / "config.json").read_text())

    if mode == "register":
        assert report["config_class"] == "surogate_tlc.configuration_tlc"
        assert report["tlc_registered"]
    else:
        # transformers loaded the directory's own configuration_tlc.py; no fork code was involved.
        assert report["config_class"].startswith("transformers_modules")
        assert not report["fork_imported_for_config"]
        assert not report["tlc_registered"]
    assert report["layer_pattern"] == config["tlc_layer_pattern"]
    assert report["max_model_len"] == 512
    assert report["token_ids"] == [config["pad_token_id"], config["bos_token_id"], config["eos_token_id"]]
    assert report["ir_model"] == "TLCModel"
    assert report["ir_pattern"] == config["tlc_layer_pattern"]
    assert report["ir_source_architectures"] == [hf.ARCHITECTURE]


def test_written_dir_holds_the_contract_files(model_dir):
    names = {p.name for p in model_dir.iterdir()}
    assert names == {"config.json", "configuration_tlc.py", "tokenizer.json", "tokenizer_config.json", "tlc_meta.json"}
    assert (model_dir / "configuration_tlc.py").read_text() == hf.configuration_source()
    assert json.loads((model_dir / "tlc_meta.json").read_text()) == {
        "tlc_meta_version": 1,
        "init_seed": 0,
        "label_ontology_version": None,
        "shard_version": None,
        "sidecar_version": None,
    }
    assert modeldir.check_model_dir(model_dir, weights=False) == json.loads((model_dir / "config.json").read_text())
    with pytest.raises(modeldir.ModelDirError, match="missing model.safetensors"):
        modeldir.check_model_dir(model_dir)


def test_write_rejects_an_invalid_config_before_writing(tmp_path, tokenizer_dir):
    with pytest.raises(ValueError, match="tlc_layer_pattern must be a string"):
        modeldir.write_model_dir(tmp_path / "bad", hf.default_config(tlc_layer_pattern=["M"] * 12), tokenizer_dir)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "overrides, problem",
    [
        ({"bos_token_id": 3}, "bos_token '<|bos|>' has id 1 in the tokenizer, config.json says 3"),
        ({"vocab_size": 300}, "tokenizer ids reach {max_id}, but vocab_size is 300"),
    ],
)
def test_config_must_agree_with_the_tokenizer(tmp_path, tokenizer_dir, overrides, problem):
    from tokenizers import Tokenizer

    max_id = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json")).get_vocab_size(with_added_tokens=True) - 1
    assert max_id >= 300
    with pytest.raises(modeldir.ModelDirError) as error:
        modeldir.write_model_dir(tmp_path / "model", hf.default_config(**overrides), tokenizer_dir)
    assert problem.format(max_id=max_id) in error.value.problems


def test_tokenizer_needs_a_chat_template(model_dir):
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text())
    del tokenizer_config["chat_template"]
    (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    with pytest.raises(modeldir.ModelDirError, match="no chat template"):
        modeldir.check_model_dir(model_dir, weights=False)
