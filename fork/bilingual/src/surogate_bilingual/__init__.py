"""Independently trained Chinese and English FFN experts on a frozen shared backbone: a pure PyTorch package.

A fork-owned package, kept out of every wheel the fork ships. Importing it loads neither torch nor any
`surogate` module; submodules import torch where they need it. Layout and environments: fork/bilingual/README.md;
decisions and data contracts: fork/bilingual/docs/spec.md.
"""

__version__ = "0.1.0.dev0"
