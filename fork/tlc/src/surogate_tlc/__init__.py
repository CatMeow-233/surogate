"""The Little Coder (TLC) experiments on the Surogate native trainer.

A fork-owned package, kept out of the native wheel. Importing it must stay cheap and must not
need the native extension: the CPU tests import it in an environment without one. See
fork/tlc/README.md for the layout, the checkpoint contract and the GPU runbook.
"""
