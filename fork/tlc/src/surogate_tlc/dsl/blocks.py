"""TLC layer blocks, built from upstream DSL modules only.

Class names carry the TLC prefix: the DSL registry is keyed by class name, so a name shared with an upstream
class would replace the upstream definition without an error.
"""

from __future__ import annotations

from surogate.dsl import nn
from surogate.dsl.modules import GenericMLP, Mamba2Mixer, RMSNorm

# Mamba-2 as the reference implementation has it: conv1d bias on, in/out projection bias off. The model sets the
# same values as attributes because Mamba2Mixer's optional parameters are switched by the model's config.
MAMBA_CONV_BIAS = True
MAMBA_PROJ_BIAS = False

_MIXER_NAMES = (
    "in_proj_weight",
    "in_proj_bias",
    "conv_weight",
    "conv_bias",
    "A_log",
    "D_param",
    "dt_bias",
    "gated_norm_weight",
    "out_proj_weight",
    "out_proj_bias",
    "x_flat",
    "projected_flat",
    "projected",
    "gate",
    "conv_input",
    "dt",
    "conv_out",
    "hidden_states",
    "ssm_B",
    "ssm_C",
    "ssm_out",
    "ssm_out_flat",
    "ssm_state",
    "gated_out",
    "gated_flat",
    "out_flat",
    "out",
)

# Dense-block names for the two norms (ln1/res_ffn, ln2/res_att), Nemotron-H names for the mixer.
TLC_MAMBA_FFN_BLOCK_REMAP: dict[str, str] = {
    "attn_norm_weight": "ln1_weight",
    "attn_norm_res": "res_ffn",
    "attn_norm_y": "ln1",
    "attn_norm_rstd": "ln1_rstd",
    "mlp_norm_weight": "ln2_weight",
    "mlp_norm_res": "res_att",
    "mlp_norm_y": "ln2",
    "mlp_norm_rstd": "ln2_rstd",
    **{f"mixer_{name}": name for name in _MIXER_NAMES},
    "mixer_x_flat": "mamba_x_flat",
    "mixer_out": "mamba_out",
    "mixer_out_flat": "mamba_out_flat",
    "mlp_act": "swiglu",
    "mlp_act_flat": "swiglu_flat",
}


class TLCMambaFFNBlock(nn.Block):
    """A0 layer: RMSNorm -> Mamba-2 mixer -> RMSNorm -> SwiGLU, with the residual carried as in upstream hybrids."""

    _name_remap_ = TLC_MAMBA_FFN_BLOCK_REMAP

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        mamba_num_heads: int,
        mamba_head_dim: int,
        ssm_state_size: int,
        n_groups: int,
        conv_kernel: int,
        chunk_size: int,
        eps: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.C = d_model
        self.M = d_ff
        self.MUp = 2 * d_ff
        self.attn_norm = RMSNorm(d_model, eps=eps)
        self.mixer = Mamba2Mixer(
            d_model,
            mamba_num_heads=mamba_num_heads,
            mamba_head_dim=mamba_head_dim,
            ssm_state_size=ssm_state_size,
            n_groups=n_groups,
            conv_kernel=conv_kernel,
            chunk_size=chunk_size,
            eps=eps,
            use_conv_bias=MAMBA_CONV_BIAS,
            use_bias=MAMBA_PROJ_BIAS,
        )
        self.mlp_norm = RMSNorm(d_model, eps=eps)
        self.mlp = GenericMLP(d_model, d_ff)

    def forward(self, x, residual, position_ids):
        del position_ids
        residual, h = self.attn_norm(residual, x)
        h = self.mixer(h)
        residual, h = self.mlp_norm(residual, h)
        h = self.mlp(h)
        return h, residual
