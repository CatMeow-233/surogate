"""TLCModel: embedding -> [model-level layer 1] -> one HybridBlockStack of TLC layers -> final RMSNorm -> LM head.

tlc_layer_pattern picks each layer: M = TLCMambaFFNBlock, A = upstream Qwen3Block. The P0 rows (hf.ROWS) are A0
(all M), A1 (all A), A1h (M with two A) and A0-split, which is A0 with layer 1 traced as model-level ops ahead
of a stack that holds layers 2..L. Importing this module registers the model with the DSL under the
architecture name TLCForCausalLM.
"""

from __future__ import annotations

import re

from surogate.dsl import nn
from surogate.dsl.blocks.common import STANDARD_MODEL_NAME_REMAP
from surogate.dsl.blocks.qwen3 import Qwen3Block
from surogate.dsl.modules import Embedding, LMHead, RMSNorm
from surogate.dsl.specs import ActivationScope

from ..hf import ARCHITECTURE, LAYER_KINDS, MODEL_TYPE, SPLIT_KEY
from .blocks import MAMBA_CONV_BIAS, MAMBA_PROJ_BIAS, TLC_MAMBA_FFN_BLOCK_REMAP, TLCMambaFFNBlock

# Name prefix of the model-level layer 1 of A0-split. Stack layers are blocks[i]; a model-level name must not look
# like one, or the native runtime would take it for a parameter of stack layer i.
FIRST_LAYER = "first_layer"
_BLOCK_NAME = re.compile(r"blocks\[(\d+)\]\.(.+)")


def split_param_name(name: str) -> str:
    """Name in A0-split of the parameter that A0 calls `name`: blocks[0].X -> first_layer_X, blocks[i] -> [i-1]."""
    match = _BLOCK_NAME.fullmatch(name)
    if not match:
        return name
    layer, field = int(match.group(1)), match.group(2)
    return f"{FIRST_LAYER}_{field}" if layer == 0 else f"blocks[{layer - 1}].{field}"


@nn.hf_config(
    architecture=ARCHITECTURE,
    model_type=MODEL_TYPE,
    vocab_size="vocab_size",
    d_model="hidden_size",
    num_layers="num_hidden_layers",
    d_ff="intermediate_size",
    max_seq="max_position_embeddings",
    eps="rms_norm_eps",
    layer_pattern="tlc_layer_pattern",
    split_first_layer=SPLIT_KEY,
    mamba_num_heads="mamba_num_heads",
    mamba_head_dim="mamba_head_dim",
    ssm_state_size="ssm_state_size",
    n_groups="n_groups",
    conv_kernel="conv_kernel",
    chunk_size="chunk_size",
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    head_size="head_dim",
)
class TLCModel(nn.Model):
    """P0 rows A0, A0-split, A1 and A1h; see hf.ROWS."""

    # The model-level layer 1 gets the block's names under the FIRST_LAYER prefix.
    _name_remap_ = {
        **STANDARD_MODEL_NAME_REMAP,
        **{f"{FIRST_LAYER}_{raw}": f"{FIRST_LAYER}_{name}" for raw, name in TLC_MAMBA_FFN_BLOCK_REMAP.items()},
    }
    # Block parameters load under their IR names. Model-level ones need explicit names: RMSNorm's module default
    # would leave the literal template "{prefix}.weight" for final_norm.
    _hf_block_mappings_ = {
        "embedding": "model.embed_tokens.weight",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
    }

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        d_ff: int,
        max_seq: int,
        eps: float,
        layer_pattern: str,
        split_first_layer: bool = False,
        mamba_num_heads: int = 0,
        mamba_head_dim: int = 0,
        ssm_state_size: int = 0,
        n_groups: int = 0,
        conv_kernel: int = 0,
        chunk_size: int = 0,
        num_query_heads: int = 0,
        num_kv_heads: int = 0,
        head_size: int = 0,
    ):
        super().__init__()
        if (
            not isinstance(layer_pattern, str)
            or len(layer_pattern) != num_layers
            or set(layer_pattern) - set(LAYER_KINDS)
        ):
            kinds = "".join(LAYER_KINDS)
            raise ValueError(f"tlc_layer_pattern {layer_pattern!r} is not {num_layers} characters from {kinds!r}")
        if split_first_layer and (num_layers < 2 or layer_pattern[0] != "M"):
            raise ValueError(f"{SPLIT_KEY} needs a Mamba first layer and at least one more, got {layer_pattern!r}")
        stack_pattern = layer_pattern[1:] if split_first_layer else layer_pattern
        self.vocab_size = vocab_size
        self.d_model = d_model
        # The native runtime's layer count is n_layers: the stack only, without a model-level layer 1.
        self.n_layers = len(stack_pattern)
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.eps = eps
        self.D = head_size
        self.split_first_layer = bool(split_first_layer)
        self.block_types = [LAYER_KINDS[c] for c in stack_pattern]
        self.n_mamba_blocks = stack_pattern.count("M")
        self.n_attn_blocks = stack_pattern.count("A")
        self.use_rope = self.n_attn_blocks > 0
        if "M" in layer_pattern:
            # Mamba2Mixer's register_param(when="use_conv_bias" / "use_bias") is evaluated against the model:
            # without these attributes conv_bias is dropped while the graph still reads it.
            self.use_conv_bias = MAMBA_CONV_BIAS
            self.use_bias = MAMBA_PROJ_BIAS
        if "M" in stack_pattern:
            # The native runtime types each layer from this string attribute (list attributes do not reach the
            # IR). Without it every layer is a dense block, as in upstream Qwen3; A1 relies on that.
            self.hybrid_pattern = stack_pattern
        if self.n_attn_blocks:
            # What upstream Qwen3 states for the same blocks.
            self.use_qk_norm = True
            self.use_qkv_bias = False
            self.rotary_dim = head_size

        mamba = dict(
            d_model=d_model,
            d_ff=d_ff,
            mamba_num_heads=mamba_num_heads,
            mamba_head_dim=mamba_head_dim,
            ssm_state_size=ssm_state_size,
            n_groups=n_groups,
            conv_kernel=conv_kernel,
            chunk_size=chunk_size,
            eps=eps,
        )
        attention = dict(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            d_ff=d_ff,
            max_seq=max_seq,
            eps=eps,
        )
        configs = []
        if self.n_mamba_blocks:
            configs.append(("mamba_blocks", TLCMambaFFNBlock, self.n_mamba_blocks, mamba))
        if self.n_attn_blocks:
            configs.append(("attn_blocks", Qwen3Block, self.n_attn_blocks, attention))

        self.embedding = Embedding(vocab_size, d_model)
        self.first_layer = None
        if split_first_layer:
            self.first_layer = TLCMambaFFNBlock(**mamba)
            # Resolve the model-level SwiGLU weight shapes to numbers, as the block does for stack layers.
            self.M = d_ff
            self.MUp = 2 * d_ff
        self.hybrid_blocks = nn.HybridBlockStack(
            block_configs=configs, block_types=self.block_types, n_layers=self.n_layers
        )
        self.final_norm = RMSNorm(d_model, eps=eps)
        self.lm_head = LMHead(vocab_size, d_model)

    def forward(self, token_ids, position_ids, targets):
        G = ActivationScope.GLOBAL
        self._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("position_ids", ("T",), dtype="int32", scope=G)
        self._register_activation("targets", ("B", "T"), dtype="int32", scope=G, aliases=["labels"])
        if self.use_rope:
            self._register_activation("freq_cis", ("max_seq", "D", 2), dtype="fp32", scope=G, aliases=["rope_freqs"])
        hidden = ("B", "T", "d_model")
        self._register_activation("residual0", hidden, scope=G)
        self._register_activation("x0", hidden, aliases=["encoded"], scope=G)
        self._register_activation("xN", hidden, scope=G)
        self._register_activation("residualN", hidden, scope=G)
        self._register_activation("residual_final", hidden, scope=G)
        self._register_activation("xF", hidden, aliases=["ln_final"], scope=G)
        self._register_activation("xF_flat", ("B * T", "d_model"), scope=G)
        self._register_activation("ln_final_rstd", ("B", "T"), dtype="fp32", save=True, scope=G)
        self._register_activation("loss", ("B * T",), dtype="fp32", aliases=["losses"], scope=G)

        x = self.embedding(token_ids)
        residual = self._zeros(["B", "T", "d_model"])
        if self.first_layer is not None:
            x, residual = self._trace_first_layer(x, residual, position_ids)
        x, residual = self.hybrid_blocks(x, residual, position_ids)
        residual, x = self.final_norm(residual, x)
        return self.lm_head(x, targets)

    def _trace_first_layer(self, x, residual, position_ids):
        """Trace layer 1's block forward into the model graph, under the FIRST_LAYER prefix."""
        tracer = nn._current_tracer.get()
        mapped = set(tracer.hf_mappings)
        tracer.push_prefix(FIRST_LAYER)
        try:
            out = self.first_layer.forward(x, residual, position_ids)
        finally:
            tracer.pop_prefix()
        # The modules registered "{prefix}..." templates for these weights; like stack-layer weights, they load
        # under their IR names instead.
        for name in set(tracer.hf_mappings) - mapped:
            del tracer.hf_mappings[name]
        return out
