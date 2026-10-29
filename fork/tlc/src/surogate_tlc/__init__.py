"""The Little Coder (TLC) experiments on the Surogate native trainer.

A fork-owned package, kept out of the native wheel. Importing it must stay cheap and must not need the native
extension: the CPU tests import it in an environment without one, so modules that import `surogate` are only
imported inside functions. See fork/tlc/README.md for the layout and fork/tlc/docs/ for the contracts and the GPU
runbook.
"""

from __future__ import annotations


def register() -> None:
    """Make TLC models loadable in this process. Idempotent.

    Imports the TLC DSL models, whose @nn.hf_config adds them to the upstream DSL registry that the trainer
    resolves config.json's `architectures` against, and registers TLCConfig with transformers' AutoConfig for
    model_type "tlc", which then takes precedence over a model directory's `auto_map`.
    """
    from transformers import AutoConfig

    from . import dsl  # noqa: F401 - registers the DSL models on import
    from .hf import MODEL_TYPE, TLCConfig

    AutoConfig.register(MODEL_TYPE, TLCConfig, exist_ok=True)
