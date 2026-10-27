"""Experimental PyTorch training; importing this package starts no training."""

from .config import TrainConfig, load_config

__all__ = ["TrainConfig", "load_config"]
