"""Transformers configuration class for model_type "tlc" (The Little Coder).

surogate_tlc copies this file unchanged into every TLC model directory, and the directory's config.json names it
in `auto_map`, so a process that does not import the fork can still load the config. It must therefore import
nothing but transformers. The rules a TLC config.json must follow are checked by surogate_tlc.hf.check_config.
"""

from transformers import PretrainedConfig


class TLCConfig(PretrainedConfig):
    """Every config.json key becomes an attribute; there are no defaults, so nothing is filled in silently."""

    model_type = "tlc"
