"""The "W used twice" toy model that scouts the shared-weight depth loop of a later phase on the native trainer.

One weight W [d, d] feeds two matmuls in the same layer: h -> W h -> W (W h). Its gradient must be the sum of both
uses. Whether the native backward accumulates it, overwrites one use with the other, or fails, is what the GPU test
records (tests/gpu/test_w_twice.py); a crash, NaN or a wrong gradient is the expected P0 outcome and blocks nothing.
The layer is a dense block of RMSNorm and the twice-used linear map, so the reference gradient is plain PyTorch.

Importing this module registers TLCTwiceProbeModel with the DSL registry, so only a probe process imports it.
"""

from __future__ import annotations

from typing import Any

import torch

from surogate.dsl import nn
from surogate.dsl.blocks.common import STANDARD_MODEL_NAME_REMAP
from surogate.dsl.dim import B, T
from surogate.dsl.modules import Embedding, LMHead, RMSNorm
from surogate.dsl.specs import ActivationScope

ARCHITECTURE = "TLCTwiceProbeForCausalLM"
FIELD = "twice_weight"


class TLCTwiceLinear(nn.Module):
    """y = W (W x), one parameter in two matmuls."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.C = d_model

    def _trace(self, tracer: nn.Tracer, *args: nn.Proxy, **kwargs: Any) -> nn.Proxy:
        g = tracer.graph
        (x,) = args
        w = tracer.register_param("weight", ("C", "C"))
        out_slot = tracer.register_activation("out", ("B", "T", "C"), share_policy="when_recomputed")
        x_flat = g.view(x.ref, shape=[B * T, self.C], out_name=tracer.prefixed("x_flat"))
        once = g.matmul(x_flat, w, transpose="NT", out_name=tracer.prefixed("once"))
        twice = g.matmul(once, w, transpose="NT", out_name=tracer.prefixed("twice"))
        out = g.view(twice, shape=[B, T, self.C], out_name=out_slot)
        return nn.Proxy(out_slot, out)


class TLCTwiceProbeBlock(nn.Block):
    _name_remap_ = {
        "attn_norm_weight": "ln1_weight",
        "attn_norm_res": "res_ffn",
        "attn_norm_y": "ln1",
        "attn_norm_rstd": "ln1_rstd",
        "twice_out": "twice_out",
    }

    def __init__(self, d_model: int, eps: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.C = d_model
        self.attn_norm = RMSNorm(d_model, eps=eps)
        self.twice = TLCTwiceLinear(d_model)

    def forward(self, x, residual, position_ids):
        del position_ids
        residual, h = self.attn_norm(residual, x)
        return self.twice(h), residual


@nn.hf_config(
    architecture=ARCHITECTURE,
    model_type="tlc",
    vocab_size="vocab_size",
    d_model="hidden_size",
    num_layers="num_hidden_layers",
    max_seq="max_position_embeddings",
    eps="rms_norm_eps",
)
class TLCTwiceProbeModel(nn.Model):
    _name_remap_ = STANDARD_MODEL_NAME_REMAP
    _hf_block_mappings_ = {
        "embedding": "model.embed_tokens.weight",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
    }

    def __init__(self, vocab_size: int, d_model: int, num_layers: int, max_seq: int, eps: float) -> None:
        super().__init__()
        self.vocab_size, self.d_model, self.n_layers, self.max_seq, self.eps = (
            vocab_size,
            d_model,
            num_layers,
            max_seq,
            eps,
        )
        self.block_types = ["dense"] * num_layers
        self.n_dense_blocks = num_layers
        self.embedding = Embedding(vocab_size, d_model)
        self.hybrid_blocks = nn.HybridBlockStack(
            block_configs=[("dense_blocks", TLCTwiceProbeBlock, num_layers, {"d_model": d_model, "eps": eps})],
            block_types=self.block_types,
            n_layers=num_layers,
        )
        self.final_norm = RMSNorm(d_model, eps=eps)
        self.lm_head = LMHead(vocab_size, d_model)

    def forward(self, token_ids, position_ids, targets):
        G = ActivationScope.GLOBAL
        self._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("position_ids", ("T",), dtype="int32", scope=G)
        self._register_activation("targets", ("B", "T"), dtype="int32", scope=G, aliases=["labels"])
        hidden = ("B", "T", "d_model")
        for name, aliases in (("residual0", None), ("x0", ["encoded"]), ("xN", None), ("residualN", None)):
            self._register_activation(name, hidden, aliases=aliases, scope=G)
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


def probe_config(vocab_size: int, **tokens: int) -> dict[str, Any]:
    """config.json of a one-layer probe with width 64; `tokens` are the tokenizer's bos/eos/pad ids."""
    return {
        "architectures": [ARCHITECTURE],
        "model_type": "tlc",
        "vocab_size": vocab_size,
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "intermediate_size": 64,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
        "num_attention_heads": 1,
        "head_dim": 64,
        "torch_dtype": "bfloat16",
        **tokens,
    }


def reference_gradient(weights: dict[str, torch.Tensor], cfg: dict, ids: torch.Tensor) -> torch.Tensor:
    """dLoss/dW in fp32 for next-token targets of `ids` (mean over the targets), W used twice."""
    w = {name: t.float().clone().requires_grad_(name == f"blocks[0].{FIELD}") for name, t in weights.items()}
    eps = cfg["rms_norm_eps"]

    def norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    x = w["model.embed_tokens.weight"][ids[:, :-1]]
    h = norm(x, w["blocks[0].ln1_weight"])
    out = (h @ w[f"blocks[0].{FIELD}"].T) @ w[f"blocks[0].{FIELD}"].T
    logits = norm(x + out, w["model.norm.weight"]) @ w["model.embed_tokens.weight"].T
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten())
    loss.backward()
    return w[f"blocks[0].{FIELD}"].grad
