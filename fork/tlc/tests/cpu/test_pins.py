"""Pins on the upstream interfaces P0 builds on (README: "上游接口的 pin"). A failure here means upstream changed
one of them: check the TLC code that uses it, then update the pin.

Modules that need the native extension (the trainer, the shard writer, the CLI) are read as source.
"""

import ast
import inspect
import re
from pathlib import Path

from surogate_tlc import hf, ir
from surogate_tlc.data import tokenizer

REPO = Path(__file__).resolve().parents[4]


def _params(callable_) -> list[str]:
    return [name for name in inspect.signature(callable_).parameters if name != "self"]


def _tree(relative: str) -> ast.Module:
    return ast.parse((REPO / relative).read_text(encoding="utf-8"))


def _method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} is gone")


def _signature(fn: ast.FunctionDef) -> list[str]:
    """Positional parameters after self, with defaults: ["file_name", "masking=False", ...]."""
    names = [a.arg for a in fn.args.args][1:]
    defaults = [None] * (len(names) - len(fn.args.defaults)) + [ast.unparse(d) for d in fn.args.defaults]
    return [name if default is None else f"{name}={default}" for name, default in zip(names, defaults, strict=True)]


def test_dsl_constructors_tlc_calls():
    from surogate.dsl import nn
    from surogate.dsl.blocks.qwen3 import Qwen3Block
    from surogate.dsl.modules import GenericMLP, Mamba2Mixer

    mamba = "d_model mamba_num_heads mamba_head_dim ssm_state_size n_groups conv_kernel chunk_size eps dt_min dt_max"
    assert _params(Mamba2Mixer) == [*mamba.split(), "time_step_limit", "use_conv_bias", "use_bias"]
    assert _params(GenericMLP) == ["d_model", "d_ff", "config"]
    qwen3 = "d_model num_query_heads num_kv_heads head_size d_ff max_seq eps use_qkv_bias use_qk_norm"
    assert _params(Qwen3Block) == qwen3.split()
    stack = "block_configs block_types n_layers per_layer_input_name kv_sharing_map"
    assert _params(nn.HybridBlockStack) == stack.split()


def test_dsl_defaults_tlc_relies_on():
    from surogate.dsl.activations import Activation
    from surogate.dsl.blocks.qwen3 import Qwen3Block
    from surogate.dsl.modules import GenericMLP, Mamba2Mixer

    # No dt clamp unless time_step_limit is given: the numerical reference must not clamp either.
    assert Mamba2Mixer(64, mamba_num_heads=4, mamba_head_dim=16, ssm_state_size=16).time_step_limit == (0.0, 1e9)
    # TLC's SwiGLU: GenericMLP's default is a gated SiLU with up and gate fused into one weight, no bias.
    config = GenericMLP(64, 128).config
    assert (config.activation, config.gated, config.fuse_gate_up, config.bias) == (Activation.SILU, True, True, False)
    defaults = inspect.signature(Qwen3Block).parameters
    assert (defaults["use_qkv_bias"].default, defaults["use_qk_norm"].default) == (False, True)


def test_native_op_table_is_parsed():
    names = ir.native_op_names(REPO)
    assert len(names) > 100, "op_type_from_string's table not found: graph_compiler.cpp changed layout"
    assert {"mamba_ssm_scan", "flash_attention", "fused_lm_head_loss", "moe_softmax", "concat", "mul"} <= names


def test_cli_main_runs_the_command_module_in_process():
    """`python -m surogate_tlc` registers the TLC models and hands over to cli_main, so this must stay runpy."""
    tree = _tree("surogate/cli/main.py")
    mapping = next(
        node.value for node in tree.body if isinstance(node, ast.AnnAssign) and node.target.id == "COMMAND_MAPPING"
    )
    commands = ast.literal_eval(mapping)
    assert commands["sft"] == "surogate.cli.sft" and commands["pt"] == "surogate.cli.pt"
    cli_main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "cli_main")
    calls = {ast.unparse(node) for node in ast.walk(cli_main) if isinstance(node, ast.Call)}
    assert "runpy.run_module(module_name, run_name='__main__', alter_sys=True)" in calls
    assigns = {ast.unparse(node) for node in ast.walk(cli_main) if isinstance(node, ast.Assign)}
    assert "sys.argv = [module_name] + sys.argv[2:]" in assigns


def test_trainer_wrapper_construction():
    tree = _tree("surogate/train/trainer.py")
    assert _signature(_method(tree, "SurogateTrainerWrapper", "__init__")) == [
        "config",
        "train_files",
        "eval_files=None",
    ]
    assert _signature(_method(tree, "SurogateTrainerWrapper", "train")) == []


def test_tokenized_data_file_writer():
    tree = _tree("surogate/train/tokenize.py")
    cls = "TokenizedDataFileWriter"
    assert _signature(_method(tree, cls, "__init__")) == [
        "file_name",
        "vocab_size",
        "masking=False",
        "non_overlapping=False",
    ]
    assert _signature(_method(tree, cls, "add_document")) == ["tokens", "position_ids", "mask=None"]
    for name in ("__enter__", "__exit__"):
        _method(tree, cls, name)
    header = _method(tree, cls, "_write_header")
    source = ast.unparse(header)
    assert "header_str = 'BIN.TOK\\n'" in source and "version = 3" in source
    fields = "[version, bytes_per_token, self.n_tokens, self.vocab_size, self.has_masks, self.non_overlapping]"
    assert fields in source


CPP_STRING = r'"(?:[^"\\]|\\.)*"'


def _cpp_strings(code: str) -> list[str]:
    """The values of the C++ string literals in `code`, adjacent literals joined."""
    literals = (re.findall(CPP_STRING, joined) for joined in re.findall(rf"(?:{CPP_STRING}\s*)+", code))
    return ["".join(re.sub(r"\\(.)", r"\1", lit[1:-1]) for lit in group) for group in literals]


def test_native_tokenizer_reads_tlc_split_patterns():
    """A "tlc" model's split pattern comes from tokenizer.json (GPT-2's when there is none), and the native
    tokenizer has hand-written splitters for the "qwen2" pattern and GPT-2's; others run on std::regex."""
    tokenizer_cpp = (REPO / "csrc/src/tokenizer/tokenizer.cpp").read_text(encoding="utf-8")
    assert hf.MODEL_TYPE not in set(re.findall(r'architecture == "([^"]+)"', tokenizer_cpp))
    assert 'if (obj.at("type") == "Split" && obj.contains("pattern"))' in tokenizer_cpp
    gpt2_default = _cpp_strings(tokenizer_cpp[tokenizer_cpp.index("If still empty, use GPT-2 default") :])[0]
    unicode_cpp = (REPO / "csrc/src/tokenizer/unicode.cpp").read_text(encoding="utf-8")
    custom = unicode_cpp[unicode_cpp.index("unicode_regex_split_custom(const std::string& text") :]
    custom = custom[: custom.index("\n}\n")]
    patterns = {_cpp_strings(m)[0] for m in re.findall(rf"regex_expr == ((?:{CPP_STRING}\s*)+)", custom)}
    assert {tokenizer.PRETOKENIZERS["qwen2"], gpt2_default} <= patterns
    assert tokenizer.PRETOKENIZERS["python"] not in patterns


def test_native_data_loader_reads_padded_rows():
    """The reading rules surogate_tlc.data.shards.served_rows mirrors, for a non-overlapping shard with masks."""
    source = " ".join((REPO / "csrc/src/runtime/training/dataloader.cpp").read_text(encoding="utf-8").split())
    for snippet in (
        "int num_chunks = file_info->NonOverlapping ? (num_tokens / mSeqLen) : ((num_tokens - 1) / mSeqLen);",
        "const long target_offset = input_offset + element_size;",
        "const long pos_ids_start = element_size * file_info->NumTokens + header_offset;",
        "masks_start += element_size * file_info->NumTokens;",
        "bool mask_bit = (mMaskBuffer[byte_id] >> bit_id) & 1; if (!mask_bit) { target_tokens[i - start] = -100; }",
        "if (file_info->NonOverlapping) { targets.get<int>()[mSeqLen - 1] = -100; }",
    ):
        assert snippet in source, snippet


def _config_keys() -> set[str]:
    """Keys upstream's SFT config reads from the YAML: cfg["k"], cfg.get("k", ...) and "k" in cfg."""
    keys = set()
    for name in ("sft_config.py", "model_config.py", "train_dataset_config.py"):
        for node in ast.walk(_tree(f"surogate/core/config/{name}")):
            if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "cfg":
                keys.add(ast.literal_eval(node.slice))
            elif isinstance(node, ast.Call) and ast.unparse(node.func) == "cfg.get":
                keys.add(ast.literal_eval(node.args[0]))
            elif isinstance(node, ast.Compare) and ast.unparse(node.comparators[0]) == "cfg":
                keys.add(ast.literal_eval(node.left))
    return keys


def test_sft_keys_tlc_sets_are_read_upstream():
    """A misspelt or retired key would be ignored silently and the run would use upstream's default."""
    import yaml
    from surogate_tlc.train import config, driver

    used = {*config.FIXED, *driver.SCORING, "gpus", "gradient_accumulation_steps"}
    for example in sorted((REPO / "fork" / "tlc" / "examples").glob("*.yaml")):
        body = yaml.safe_load(example.read_text(encoding="utf-8"))
        used |= set(body) - {"tlc"}
        for stage in body["tlc"]["stages"].values():
            used |= set(stage) - config.STAGE_KEYS
    assert used - _config_keys() == set()


def test_sft_config_is_completed_by_an_explicit_post_init():
    """driver.native_wrapper calls SFTConfig.__post_init__() itself, as TokenizeDatasets does."""
    init = _method(_tree("surogate/train/tokenize.py"), "TokenizeDatasets", "__init__")
    assert "config.__post_init__()" in {ast.unparse(n) for n in ast.walk(init) if isinstance(n, ast.Call)}
    sft_init = _method(_tree("surogate/core/config/sft_config.py"), "SFTConfig", "__init__")
    assert "__post_init__" not in ast.unparse(sft_init)


def test_trainer_regression_artifact_holds_what_the_summary_reads():
    tree = _tree("surogate/train/trainer.py")
    init = ast.unparse(_method(tree, "SurogateTrainerWrapper", "__init__"))
    assert "os.environ.get('SUROGATE_REGRESSION_ARTIFACT')" in init
    record = ast.unparse(_method(tree, "SurogateTrainerWrapper", "_record_regression_step"))
    assert "'step': float(metrics.step + 1)" in record and "'loss': float(metrics.loss)" in record
    write = _method(tree, "SurogateTrainerWrapper", "_write_regression_artifact")
    payload = next(n for n in ast.walk(write) if isinstance(n, ast.Dict) and len(n.keys) > 5)
    keys = {ast.literal_eval(k) for k in payload.keys}
    assert {"convergence_curve", "step_time_ms", "cuda_peak_memory_bytes", "block_schema_summary"} <= keys
    percentiles = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_percentile_summary")
    assert "'p50': pct(0.5)" in ast.unparse(percentiles)
    train = ast.unparse(_method(tree, "SurogateTrainerWrapper", "train"))
    assert "self._write_regression_artifact()" in train


def test_native_training_log_is_a_json_array_with_eval_lines():
    source = (REPO / "csrc/src/runtime/training/logging.cpp").read_text(encoding="utf-8")
    assert 'mLogFile << "[\\n";' in source
    assert (
        '{{"log": "eval", "time": "{}", "step": {}, "epoch": {}, "eval_tokens": {}, "duration_ms": {}, "loss": {}}}'
        in (source)
    )


def test_native_out_of_memory_errors():
    """driver.is_oom recognises these: a RuntimeError naming "OOM" and the activation stack's std::bad_alloc."""
    allocator = (REPO / "csrc/src/utilities/allocator.cpp").read_text(encoding="utf-8")
    assert 'fmt::format("Cuda OOM when allocating tensor {} of shape {}' in allocator
    assert "throw std::runtime_error(message);" in allocator
    stack = (REPO / "csrc/src/utilities/stack.cpp").read_text(encoding="utf-8")
    assert "throw std::bad_alloc();" in stack


def test_tlc_never_calls_train_with_oom_recovery():
    """It halves the tokens per update once accumulation reaches 16 (support plan E22); the driver replaces it."""
    for path in (REPO / "fork" / "tlc" / "src").rglob("*.py"):
        nodes = ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        names = {
            n.attr if isinstance(n, ast.Attribute) else n.id for n in nodes if isinstance(n, ast.Attribute | ast.Name)
        }
        assert "train_with_oom_recovery" not in names, path


def test_compute_logprobs_binding():
    source = " ".join((REPO / "csrc/src/binding/binding.cpp").read_text(encoding="utf-8").split())
    start = source.index('"compute_logprobs",')
    args = re.findall(r'nb::arg\("(\w+)"\)', source[start : source.index(".def(", start)])
    assert args == ["input_ids", "targets", "use_lora", "position_ids", "temperatures"]
