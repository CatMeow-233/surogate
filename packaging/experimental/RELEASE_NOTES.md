This alpha release provides the independent **surogate-experimental** PyTorch
trainer for Python 3.12+. It includes the rank-scaled optimizer with optional BF16
stochastic rounding, LISA-style layer sampling, optimizer-in-backward, and PEFT
LoRA.

Download the wheel, `experimental-examples.zip`, and `SHA256SUMS` together. Verify
the artifacts against the manifest, install the wheel, and follow `README.md`
or the guide included in the examples archive. The bundle contains a completely
offline tiny CPU training example and editable RTX 5090 configurations.

The wheel contains only `surogate.experimental` Python code. It does not include
the native CUDA trainer, serving engines, PyTorch, model weights, or datasets.
CPU checks cover the experimental algorithms, package contents, and installed
CLI. RTX 5090 throughput, model quality, and 4–7B/10B model capacity have not been
validated by this release workflow. The algorithms may change convergence and
training stability.

This release is published only in **CatMeow-233/surogate**. The stable native
distribution and its release channel are separate.
