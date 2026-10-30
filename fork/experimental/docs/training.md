# Experimental single-GPU training

This is a separate **PyTorch research backend**, invoked explicitly with
`python -m surogate.experimental.train`. It does not replace `surogate sft`,
modify the native optimizer registry, or enable experiments by importing the
regular package. Its checkpoints and configuration schema are separate too.

The initial target is one RTX 5090 with 32 GB, dense 4–7B models and roughly 10B
total-parameter MoE models. The techniques trade optimizer fidelity, numerical
precision, or which layers are trained for lower memory use. Neither a 32 GB fit
nor faster training than the native engine is established for these models.
The release validates real tiny-model CPU training; GPU throughput and task
quality need measurements on the target card.

## Install the experimental release

The prerelease channel belongs only to
[CatMeow-233/surogate](https://github.com/CatMeow-233/surogate/releases).
For version `0.1.0a3`, download these assets from tag `experimental-v0.1.0a3`:

- `surogate_experimental-0.1.0a3-py3-none-any.whl`
- `experimental-examples.zip`
- `SHA256SUMS`

Keep the appropriate CUDA-enabled PyTorch installation in your GPU environment.
Install the wheel there using Python 3.12 or later:

```bash
python -m pip install ./surogate_experimental-0.1.0a3-py3-none-any.whl
python -m zipfile -e experimental-examples.zip experimental-examples
python -m surogate.experimental.train --help
python -m surogate.experimental.train experimental-examples/examples/experimental/smoke.yaml
```

Verify downloaded assets against `SHA256SUMS`; a complete downloaded asset set can
be checked with `sha256sum --check SHA256SUMS`. The tiny smoke example uses a
randomly initialized local Llama configuration and local token data: it requires
no model download or GPU. Its output is a workflow check, not a useful language
model.

The wheel contains only the experimental namespace. It does not ship native CUDA
binaries or turn the earlier native performance changes into a new binary build.
The existing native training installation and its release artifacts are separate.
From a checkout use `fork/dev test experimental` and `fork/dev exp-train <config>`.
They run in an isolated virtual environment, `.venv-experimental` under `fork/`,
because an installed native `surogate` hides this namespace. Never use
`uv run`/`uv sync` at the repository root.

## Experiments

### Rank-scaled low-state AdamW

`optimizer: rank_scaled` is an independently implemented prototype inspired by
[APOLLO-Mini](https://arxiv.org/abs/2412.05270). It is **not the official APOLLO
implementation** and does not reproduce all its heuristics. No APOLLO source code
is vendored.

For each matrix gradient `G`, including each matrix in a batched MoE parameter:

1. Form a small random Gaussian sketch, `R = G P` or `P^T G`, with basis
   entries drawn from `N(0, 1/r)` for rank `r`.
2. Maintain FP32 Adam moments in the sketch coordinates and obtain a
   bias-corrected sketch update `U`.
3. Update using the **original full gradient** multiplied by
   `projection_scale * d * norm(U) / (norm(R) + eps)` per matrix.

`d` brings matrix updates to AdamW's per-element scale. The basis preserves
`norm(G)` in expectation, so the update's norm is about `learning_rate *
projection_scale * d * norm(U)`, while the sketch update spreads `norm(U)` over
its `nnz(U)` entries. The default `projection_scale_rule: rms` uses
`d = sqrt(S(G) / nnz(U))` per matrix. `S(G)`, the gradient's effective support,
estimates how many entries carry its norm: `min(nnz(G), pi/2 *
(sum(abs(G)) / norm(G))^2)`, a participation ratio scaled so that a dense
gradient with Gaussian-like entries counts all of them. Those entries then move about
`learning_rate * projection_scale * rms(U)`, the counterpart of AdamW's
`learning_rate * rms(Adam direction)`: one learning rate suits matrices and
vectors, and `projection_scale: 1` matches AdamW's step size. For a dense
Gaussian-like gradient `d = sqrt(n / r)`, where `n` is the projected dimension
of that matrix (the larger one).

`S(G)` also covers gradients whose norm lies in a few rows. An untied
embedding's gradient is zero outside the rows of the batch's tokens. An LM
head's softmax gradient, and therefore that of a tied embedding and LM head such
as Qwen3-4B's, is nonzero in every vocabulary row, but its norm lies in the rows
of the batch's target tokens. Counting nonzeros there would move the target
rows about `sqrt(vocabulary / target rows)` times further than AdamW.

One scalar per matrix cannot reproduce AdamW's coordinate-wise steps: the update
follows the gradient, so each entry moves in proportion to its gradient, while
AdamW moves every entry by about the learning rate. A CPU emulation with a
softmax LM-head gradient (151,936-token vocabulary, Zipf-distributed targets,
2,048 tokens and about 1,200 distinct targets per batch) measured these per-step
movements relative to AdamW's:

| Rows | `rms` | nonzero count instead of `S(G)` |
| --- | --- | --- |
| All target rows | 0.6–1.4 | 6–10 |
| Tokens seen once in the batch | 0.5–1.1 | 6–8 |
| Tokens seen ten or more times | 3–7 | 38–46 |
| Other rows (AdamW moves them a full step) | 0.0002–0.04 | 0.002–0.3 |

A heavy-tailed dense gradient, for example one with a few outlier input
channels, has a smaller `S(G)`: in emulations its whole-matrix update RMS was
0.4–0.9 times AdamW's, with the outlier channels moving 4–6 times AdamW's step
(0.7–1.3 and 4–9 times with the nonzero count). In a 20-step CPU run of a tiny
Llama with the real trainer, the attention and MLP matrices moved 8–23% less in
total than with the nonzero count.

`projection_scale_rule: constant` uses `d = 1`, the rule of releases
0.1.0a1–a3, for comparisons with those runs. Matrix updates are then about
`sqrt(r / n)` times AdamW's: at Qwen3-4B shapes and rank one, 50–100 times
smaller for the attention and MLP matrices and about 390 times smaller for the
151,936-row embedding, while vectors and norms, which use plain AdamW, take full
steps at the same learning rate. CPU tests compare the per-step update RMS with
AdamW's for several dense shapes and ranks, with consistent and with noisy
gradients, on the touched rows of a row-sparse gradient, and on the target rows
of a softmax LM-head gradient.

The projection side minimizes the projector plus two moment tensors. With rank
one, a 4096-by-4096 matrix stores 12,288 FP32 projector/moment elements instead
of 33,554,432 full Adam moment elements. This is an optimizer-state comparison,
not total training memory: weights, gradients, activations, working casts, and
attention workspaces remain.

Projection refreshes explicitly **reset the sketch moments and their bias
correction clock**, a prototype variation. The original-gradient direction and
per-matrix scale differ from coordinate-wise AdamW. Vectors/scalars and matrices
whose smaller dimension is no larger than the requested rank use ordinary AdamW
with FP32 moments. `projection_scale` applies to the projected path only.

The defaults are `projection_rank: 1`, `projection_interval: 200`,
`projection_scale: 1`, and `projection_scale_rule: rms`. There is no
persistent FP32 master copy. With BF16 parameters, `stochastic_rounding: true`
applies experimental stochastic rounding
to avoid systematically losing sub-ULP updates. Its noise is seeded per parameter
and per update from the parameter's step counter, which only increases, so no
update reuses another's noise. FP32 update temporaries are bounded in chunks;
disable the option for round-to-nearest comparisons. This is
BF16 weight updating, **not NVFP4 training**.
The chunk limit bounds individual conversion/update inputs, not total allocator
peak: rounding and moment temporaries can coexist. The initial stochastic path
requires contiguous BF16 parameters/gradients. Values beyond BF16's finite range
saturate, where unbiased rounding is impossible. Chunking also adds Python loops
and kernel launches; it is not a measured GPU throughput optimization.

### LISA-style layer sampling

`lisa_layers: 2` samples two decoder layers at a time. The original
[LISA work](https://arxiv.org/abs/2403.17919) and
[LMFlow implementation](https://github.com/OptimalScale/LMFlow/blob/main/src/lmflow/pipeline/utils/lisa_trainer.py)
motivate selective layer optimization. This implementation offers explicit state
policies rather than assuming that freezing parameters releases history:

- `lisa_state_policy: drop` releases inactive optimizer state. It saves state
  memory but resets momentum when that layer is selected again, making this a
  LISA-style variation. With `rank_scaled` the step counter and projection index
  of each parameter survive the drop: a re-activated layer starts new moments
  with a new bias-correction clock and draws a **new** basis and new rounding
  noise. Replaying the first activation's draws would round the same elements on
  every activation and bias stochastic rounding. With `adamw` the state entry is
  deleted.
- `lisa_state_policy: cpu` retains inactive state on the CPU. It requires the
  rank-scaled optimizer, which migrates state lazily when the parameter next has
  a gradient. GPU savings then require additional host RAM and transfers.

Selection changes only at completed optimizer-step boundaries, every
`lisa_interval` steps. Non-decoder parameters keep their original trainability.
The implementation changes parameter masks, not the forward graph: it does not
put frozen middle blocks under `no_grad`, which would cut gradients to earlier
active layers. Optimizer groups retain all initially eligible parameters so that
later-selected layers can actually update.

Layer discovery accepts common decoder `ModuleList` paths; `layer_path` specifies
one explicitly. Cross-block shared parameters or parameter storage are rejected. LoRA and LISA are
separate initial modes. Rotating layers is not equivalent to simultaneously
training every parameter, even if different subsets are visited over time.

### Optimizer in backward

`optimizer_in_backward: true` uses PyTorch's
[post-accumulate gradient hooks](https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html)
to update a parameter and release its gradient when accumulation completes. This
reduces simultaneous gradient residency, complementing low-state optimization.

The initial implementation requires:

- `optimizer: rank_scaled` and `gradient_accumulation_steps: 1`;
- `max_grad_norm: 0`, without a global clipping pass;
- no gradient checkpointing, compilation, or LISA;
- no distinct trainable parameters that share storage.

Tied weights, one parameter reachable under two names such as the embedding and
LM head of Qwen3-4B (`tie_word_embeddings`), are supported. PyTorch sums the
gradients of both uses before that parameter's hook runs, so it is updated once
per step with its full gradient, exactly as the regular optimizer step would.
The cost is residency: that gradient stays allocated from the LM head's backward
until the embedding's, which is the whole backward pass.

FP16 GradScaler is not implemented. Use FP32 or BF16. These constraints protect
the ordering of updates; accepting convergence risk does not make repeated or
premature updates a valid experiment. A backward failure can leave some
parameters updated, so recovery uses the preceding complete checkpoint.

## RTX 5090 starting configurations

The examples archive and `fork/experimental/examples/` provide:

| File | Purpose |
| --- | --- |
| `smoke.yaml` | Offline CPU end-to-end test with local tiny weights and data. |
| `5090-lisa.yaml` | Rank-scaled state, two rotating layers, nonreentrant activation checkpointing, gradient accumulation. |
| `5090-oib.yaml` | All-parameter updates with early gradient release, no checkpointing or accumulation; the tied embedding/LM head is updated once per step. |
| `5090-lora.yaml` | PEFT adapter control run with ordinary AdamW on this backend. |

Replace the model and `train_jsonl` path with your own compatible model and token
data. The defaults are editable experiments, not tuned hyperparameters or
validated capacity presets. Start with short sequences and inspect actual peak
memory and supervised tokens per second before increasing the token budget.

For 7B full fine-tuning, low-state optimization alone does not eliminate the
roughly 14 GB of BF16 weights and 14 GB of full BF16 gradients. OIB changes
gradient lifetime; LISA changes which weights receive gradients. MoE memory must
be budgeted against **total parameters**, including experts not selected for a
particular token. Fused three-dimensional expert parameters are eligible for
batched sketches, while PEFT `all-linear` does not guarantee those experts were
adapted. Inspect the manifest's actual trainable parameter report.

NVIDIA's [Transformer Engine training primer](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fp8_primer.ipynb)
is relevant to later FP8/NVFP4 experiments, but those require matching backward
kernels and numerical recipes. This module does not simulate an FP4 cast and
call it hardware-accelerated FP4 training.

## Data contract

Use local JSONL, one independent sequence per row. Valid forms include:

```json
{"input_ids":[1,7,8,9,2],"labels":[-100,-100,8,9,2]}
{"input_ids":[1,7,8,9,2],"loss_mask":[false,false,true,true,true]}
{"input_ids":[1,7,8,9,2]}
{"input_ids":[1,7,8,9,2],"attention_mask":[1,1,1,1,1],"id":"doc-7","metadata":{"source":"x"}}
```

A record may contain only these keys: `input_ids`, `labels` or `loss_mask`,
`attention_mask`, `id`, and `metadata`. `id` and `metadata` are free-form and
never read. `attention_mask`, which tokenizers often emit, must be all ones. Any
other key is an error that names the line and suggests the intended key for a
near miss (`label`, `loss-mask`), because a misspelled `labels` would otherwise
supervise every token.

IDs must come from the selected model's tokenizer. Without `labels` or
`loss_mask`, all causal targets are supervised. `loss_mask[t]` says whether
**token `input_ids[t]` is a target**, unlike the native shard's input-aligned
mask. Labels and loss masks are mutually exclusive. The dataset rejects invalid
IDs, incompatible lengths, and sequences with no supervised target after the
causal shift. Attention masks always come from real lengths and padding, not
from supervision masks: prompts remain visible to attention.

No sample packing is implemented. Packed/segment/position-reset metadata
(`segment_ids`, `position_ids`, `cu_seqlens`, `seq_lengths`, `document_ids`, and
any other key) is rejected rather than silently permitting cross-document
attention. Long examples fail by default; `truncation: right` enables an
explicit right cut. Check that the remaining sequence still contains a
supervised target.

Across gradient accumulation, loss is `sum(causal CE) / total supervised shifted
targets` for the whole update. Averaging per-microbatch means would change the
objective for unequal completion lengths. The current objective is causal CE
only: it does not add model-specific MoE router auxiliary or balancing losses.

Only line offsets are indexed; token lists are loaded on demand. A deterministic
private sampler records the consumed cursor, including partial final batches and
epoch transitions, without changing the global training RNG.

## Checkpoints and exports

```bash
python -m surogate.experimental.train experiment.yaml --validate-only
python -m surogate.experimental.train experiment.yaml
python -m surogate.experimental.train experiment.yaml --resume PATH_TO_CHECKPOINT.pt
```

Paths in YAML are relative to the YAML file. Checkpoints are written atomically
at complete optimizer steps. Their experimental format includes model weights,
optimizer/sketch state, selection state, dataset fingerprint and consumed cursor,
configuration contract, and CPU plus selected-device CUDA RNG state. CPU runs do
not require CUDA RNG state. A CPU-retained layer history is
not eagerly copied to GPU during restore. Ordinary native checkpoints are not
accepted.

You may increase `max_steps` or change `save_every` and `output_dir` on resume;
numerical/model configuration changes and different dataset contents are rejected.
The checkpoint format is version 3. The trainer refuses version-2 checkpoints,
written by 0.1.0a1–a3: their dropped LISA layers lost the step counters that
seed rounding noise and bases, and they predate `projection_scale_rule`. Finish
such a run with the release that started it, or start a new one.
Use a new output directory if its recorded metrics extend beyond the requested
checkpoint. The runner does not overwrite an existing unrelated training run.

`metrics.jsonl` reports supervised tokens, current trainable parameters, selected
layers, optimizer-state bytes by device, and peak allocated CUDA bytes. Its
`step_seconds` includes data acquisition, forward/backward and update, but excludes
checkpoint and final export IO; measure complete run wall time separately.

The `final/` directory contains a Hugging Face model or a PEFT adapter plus an
experimental export manifest. Adapters require the original base model/revision;
an adapter trained against randomly initialized smoke weights also requires that
exact base initialization, not an arbitrary pretrained model. Full experimental
checkpoints retain the base state needed for their own resume.

## Validation scope

CPU tests exercise actual tiny Transformers and PEFT training, sketch mathematics
and state budgets, stochastic rounding, selection state ownership, token-weighted
loss, early gradient release, and uninterrupted versus checkpoint/resume behavior.
Each `5090-*.yaml` example also trains for two steps with its own settings on a
tiny Qwen3 stand-in that ties its embeddings like Qwen3-4B.
The installed wheel also runs the offline smoke CLI outside the checkout.

These checks establish executable algorithms and packaging, not GPU speed,
large-model convergence, or a 32 GB capacity guarantee. Compare loss/task quality,
complete optimizer-step time, peak CUDA memory, host memory, and data transfer on
the actual 5090. Keep the native trainer as a separate control run.
