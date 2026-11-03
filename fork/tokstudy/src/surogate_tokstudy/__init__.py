"""The tokenizer study of TLC (decision O-10) and the bilingual experts (decision B-O11): pilots at equal training
FLOPs of subword, flat-byte and H-Net (learned dynamic chunking) arms on one PyTorch backend, compared by
bits-per-byte.

A fork-owned package, kept out of every wheel the fork ships. Importing it loads neither torch nor any `surogate*`
module; submodules import torch where they need it. It may read the projects' public pure-data APIs and their
`tokenizer.json` files; neither project imports it. Layout, environment and status: fork/tokstudy/README.md.
"""

__version__ = "0.1.0.dev0"
