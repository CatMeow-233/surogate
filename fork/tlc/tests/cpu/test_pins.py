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
