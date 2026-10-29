"""TLCModel: embedding -> one HybridBlockStack of TLC layers -> final RMSNorm -> LM head.

tlc_layer_pattern picks each layer: M = TLCMambaFFNBlock (A0), A = upstream Qwen3Block (A1h). Importing this
module registers the model with the DSL under the architecture name TLCForCausalLM.
"""

from __future__ import annotations

from surogate.dsl import nn
from surogate.dsl.blocks.common import STANDARD_MODEL_NAME_REMAP
from surogate.dsl.blocks.qwen3 import Qwen3Block
from surogate.dsl.modules import Embedding, LMHead, RMSNorm
from surogate.dsl.specs import ActivationScope

from ..hf import ARCHITECTURE, LAYER_KINDS, MODEL_TYPE
from .blocks import MAMBA_CONV_BIAS, MAMBA_PROJ_BIAS, TLCMambaFFNBlock


@nn.hf_config(
    architecture=ARCHITECTURE,
    model_type=MODEL_TYPE,
    vocab_size="vocab_size",
    d_model="hidden_size",
    n_layers="num_hidden_layers",
    d_ff="intermediate_size",
    max_seq="max_position_embeddings",
    eps="rms_norm_eps",
    layer_pattern="tlc_layer_pattern",
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
    """P0 rows A0 (all M) and A1h (M with one or two A)."""

    _name_remap_ = STANDARD_MODEL_NAME_REMAP
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
        n_layers: int,
        d_ff: int,
        max_seq: int,
        eps: float,
        layer_pattern: str,
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
            or len(layer_pattern) != n_layers
            or set(layer_pattern) - set(LAYER_KINDS)
        ):
            kinds = "".join(LAYER_KINDS)
            raise ValueError(f"tlc_layer_pattern {layer_pattern!r} is not {n_layers} characters from {kinds!r}")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.eps = eps
        self.D = head_size
        # Mamba2Mixer's register_param(when="use_conv_bias" / "use_bias") is evaluated against the model's config:
        # without these attributes conv_bias is dropped while the graph still reads it.
        self.use_conv_bias = MAMBA_CONV_BIAS
        self.use_bias = MAMBA_PROJ_BIAS
        self.block_types = [LAYER_KINDS[c] for c in layer_pattern]
        # The native runtime types each layer from this string attribute; list attributes are not copied to the IR.
        self.hybrid_pattern = layer_pattern
        self.n_mamba_blocks = layer_pattern.count("M")
        self.n_attn_blocks = layer_pattern.count("A")
        self.use_rope = self.n_attn_blocks > 0

        configs = []
        if self.n_mamba_blocks:
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
            configs.append(("mamba_blocks", TLCMambaFFNBlock, self.n_mamba_blocks, mamba))
        if self.n_attn_blocks:
            attention = dict(
                d_model=d_model,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                d_ff=d_ff,
                max_seq=max_seq,
                eps=eps,
            )
            configs.append(("attn_blocks", Qwen3Block, self.n_attn_blocks, attention))

        self.embedding = Embedding(vocab_size, d_model)
        self.hybrid_blocks = nn.HybridBlockStack(block_configs=configs, block_types=self.block_types, n_layers=n_layers)
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
        x, residual = self.hybrid_blocks(x, residual, position_ids)
        residual, x = self.final_norm(residual, x)
        return self.lm_head(x, targets)
