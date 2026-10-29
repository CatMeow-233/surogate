"""TLC DSL models. Importing this package registers them with the upstream DSL registry (via @nn.hf_config)."""

from .models import TLCModel

__all__ = ["TLCModel"]
