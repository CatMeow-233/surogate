"""surogate_tlc.register() is what makes a TLC model directory loadable, and it changes nothing else.

The test runs in a fresh interpreter because registration is process-global.
"""

import json
import shutil
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

PROBE = """
import json
from surogate.dsl import models  # upstream models, as the trainer imports them
from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.dsl.py_compiler import get_model_spec, list_registered_models
from transformers import AutoConfig

plain, src = sys.argv[2], sys.argv[3]


def outcome(load):
    try:
        load()
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return "ok"


def ir_model():
    return json.loads(build_dsl_ir_for_model(plain))["modules"][0]["name"]


report = {"before": {"ir": outcome(ir_model), "config": outcome(lambda: AutoConfig.from_pretrained(plain))}}
upstream = {name: get_model_spec(name) for name in list_registered_models()}
sys.path.insert(0, src)
import surogate_tlc

surogate_tlc.register()
surogate_tlc.register()
report["added"] = sorted(set(list_registered_models()) - set(upstream))
report["replaced"] = sorted(name for name, spec in upstream.items() if get_model_spec(name) is not spec)
report["ir_model"] = ir_model()
report["config_class"] = type(AutoConfig.from_pretrained(plain)).__module__
print(json.dumps(report))
"""


def test_register_is_needed_and_adds_only_tlc(fresh_python, model_dir, tmp_path):
    # A directory without auto_map: only the registered TLCConfig can read it.
    plain = tmp_path / "plain"
    shutil.copytree(model_dir, plain)
    (plain / "configuration_tlc.py").unlink()
    config = json.loads((plain / "config.json").read_text())
    del config["auto_map"]
    (plain / "config.json").write_text(json.dumps(config))

    report = json.loads(fresh_python(PROBE, plain, SRC).splitlines()[-1])

    assert report["before"]["ir"].startswith("RuntimeError") and "TLCForCausalLM" in report["before"]["ir"]
    assert report["before"]["config"].startswith("ValueError") and "tlc" in report["before"]["config"]
    assert report["added"] == ["TLCModel"]
    assert report["replaced"] == []
    assert report["ir_model"] == "TLCModel"
    assert report["config_class"] == "surogate_tlc.configuration_tlc"
