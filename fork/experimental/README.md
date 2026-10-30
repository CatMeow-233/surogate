# Surogate experimental training

`surogate-experimental` is an independent PyTorch trainer distributed by
[CatMeow-233/surogate](https://github.com/CatMeow-233/surogate). It installs the
`surogate.experimental` namespace, without the native Surogate training extension
or serving binaries. The original `surogate` distribution is not required.

The alpha trainer provides a rank-scaled optimizer, LISA-style layer sampling,
optional optimizer-in-backward, and PEFT LoRA. These are experimental training
algorithms: they may change convergence and stability. This release does not
claim measured RTX 5090 speed or capacity improvements.

## Install

Use Python 3.12 or newer. Download the wheel and `SHA256SUMS` from the same
`experimental-v0.1.0a3` release of **CatMeow-233/surogate**. Check the wheel's
SHA-256 against its entry in that manifest before installing it:

```bash
python -m pip install ./surogate_experimental-0.1.0a3-py3-none-any.whl
python -m surogate.experimental.train --help
```

Install a PyTorch build compatible with your NVIDIA driver before GPU training.
The wheel contains Python code; it does not bundle PyTorch, CUDA, model weights,
or datasets. Its `py3-none-any` tag describes this package's contents, not the
hardware requirements of its dependencies. It is not a replacement CUDA wheel
for the stable native trainer.

## Run the bundled offline example

Download and verify `experimental-examples.zip` from the same release, then
extract it in a working directory. Keep its directory structure:

```bash
python -m zipfile -e experimental-examples.zip ./experimental-examples
cd experimental-examples
python -m surogate.experimental.train examples/experimental/smoke.yaml --validate-only
python -m surogate.experimental.train examples/experimental/smoke.yaml
```

The smoke example trains a tiny randomly initialized model on local token IDs
using the CPU. It downloads no model and demonstrates the training workflow;
it is not a language-quality or GPU performance benchmark. The archive also
includes editable RTX 5090 configuration examples and the training guide at
`docs/experimental/training.md`.

Configuration paths resolve relative to the YAML or JSON file. Read the guide
before using real checkpoints, especially the restrictions on gradient
accumulation, optimizer-in-backward, layer sampling, and checkpoint compatibility.

## Build this distribution from a checkout

```bash
python fork/experimental/build_release.py
```

Between releases the checkout carries the next version with a `.dev0` suffix,
so its builds never share a file name with a published wheel. The guide in the
checkout, `fork/experimental/docs/training.md`, lists how it differs from the
latest release.

The builder stages only the experimental namespace and its package metadata in
a temporary directory, then invokes `uv --no-config build` there. It never
invokes the repository's CUDA/scikit-build backend. Artifacts, the examples
archive, source commit information, and `SHA256SUMS` are written to
`build/experimental-release/artifacts/` by default.
