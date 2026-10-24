# Performance

Optimize the complete training iteration, including input acquisition and communication.
An isolated kernel improvement only helps when it shortens the critical path. Keep a
fixed model, recipe, sequence length, batch, hardware topology, and quality target when
comparing changes.

## Measure the limiting resource

For a phase with `F` operations, `H` bytes of HBM traffic, and `L` bytes crossing a
device link, an optimistic lower bound is the largest of `F / compute_rate`,
`H / HBM_bandwidth`, and `L / link_bandwidth`. Dependencies, launch overhead,
pipeline bubbles, and incomplete overlap increase the actual time. Use measured
sustained rates for the target GPU; specification peaks alone are insufficient.
The existing SOL estimate measures the compute side and does not establish that
memory or communication is saturated.

Capture a short, warmed-up training interval with Nsight Systems, then benchmark
again without the profiler. Inspect GPU idle gaps, CUDA API waits, memory copies,
collectives, and stage imbalance before choosing a kernel to optimize:

```bash
nsys profile -o train-trace <training-command>
nsys stats --report cuda_gpu_kern_sum train-trace.nsys-rep
nsys stats --report cuda_api_sum train-trace.nsys-rep
nsys stats --report cuda_gpu_mem_size_sum train-trace.nsys-rep
```

For the identified kernels, use Nsight Compute to inspect achieved memory
bandwidth, tensor-core utilization, occupancy, and register spills. Distinguish
useful model work from checkpoint recomputation. For GRPO, also compare the
[cycle timings](metrics.md#grpo-cycle-timing) with compute-step throughput and
track policy lag, KL, and reward against wall time. `turn_diagnostics` deliberately
uses a slower diagnostic path; compare runs with the same setting.

## Avoid redundant work and handoffs

The training runtime applies these optimizations without changing configuration:

- FP8 backward consumes a cached transposed weight without also quantizing an
  unused untransposed copy. Full fine-tuning still refreshes mutable weight
  caches through the existing weight manager.
- A frozen or unrequested weight gradient does not require a weight-sized
  temporary in recipe-driven matmul backward. This reduces stack/arena demand;
  it does not imply a CUDA allocator call was removed from every matmul.
- A ready GRPO batch is packed immediately. Empty receives retain polling
  backoff and cancellation checks.
- The GRPO trainer's single-worker, single-run packer passes packed micro-batches
  directly to its local loader. External orchestrator traffic still uses the
  configured filesystem or ZMQ transport. A bounded one-slot handoff prevents
  overwriting an unconsumed batch, and the existing NumPy conversion preserves
  masks, document positions, teacher/turn metadata, and routing replay data.
  Standalone or multi-worker packers retain the transport path.
- Dispatch-PP reads the two resident boundary tensors as a batch, allocating
  host destinations before enqueueing copies and waiting once for the batch.
  This removes a redundant explicit completion fence; transfers still use the
  existing host boundary protocol and are not P2P or zero-copy.

## RTX 5090 32 GB: dense and MoE fine-tuning

For a single RTX 5090, target SM120 kernels and the existing CUDA 13 toolchain.
An SM100 kernel or an NVLink-only communication backend is not automatically
applicable to this card.

Separate capacity from compute when planning 4–7B dense or approximately 10B
MoE training:

- BF16 frozen base weights cost about 2 bytes per parameter before adapters,
  activations, quantization/transpose caches, and workspaces. LoRA and QLoRA can
  keep optimizer state proportional to the adapters rather than the whole base.
- BF16 trainable weights and gradients plus two FP32 Adam moments already cost
  about 12 bytes per parameter. FP32 masters/gradients and separate work copies
  can increase this. Even BF16 weights/gradients plus two 8-bit moment buffers
  cost roughly 6 bytes per parameter before scale metadata and activations:
  about 42 GB for 7B parameters, exceeding this GPU's capacity.
- MoE active parameters determine much of the compute, but all trainable expert
  weights and optimizer states still need storage. Size offload against total
  parameters, not just activated parameters.

For adapter training, first keep base weights resident if they fit, compare BF16
and the supported FP8/NVFP4 recipe with the same effective batch and quality
target, and increase micro-batch/token occupancy within the activation budget.
Budget cached packed/transposed weights explicitly; a saved quantization kernel
is not free if its cache forces another offload.

For full fine-tuning that cannot fit, use the existing CPU-training/offload path
and measure CPU optimizer time, host RAM demand, and PCIe traffic alongside GPU
work. Start with [CPU offload](../../examples/training/cpu-offload.yaml) and the
[full-finetune example](../../examples/training/full-finetune.yaml); those examples
are configuration references, not validated 5090 performance presets. Gradient
accumulation amortizes optimizer publication/update costs, but does not eliminate
per-microbatch weight transfers or checkpoint recomputation. Avoid moving weights
off device when state/activation offload alone is sufficient.

Check the configuration's resolved behavior: `cpu_training: true` enables master
weight streaming and, for full fine-tuning, gradient offload with CPU optimizer
execution. `offload_optimizer: true` by itself does not move state off device
when `use_zero_copy` is false. Do not assume independent offload flags compose
into a validated configuration merely because each saves a different buffer.

The [NVIDIA TensorRT-LLM hardware table](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md)
distinguishes SM120 from SM100/103 and lists inference support for NVFP4, MXFP4,
per-tensor FP8, and FP8 KV on SM120. Inference support does not establish training
backward support or numerical equivalence: validate the actual Surogate recipe.
For training-specific low-precision design, NVIDIA's
[Transformer Engine primer](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fp8_primer.ipynb)
and [NVFP4 pretraining paper](https://arxiv.org/abs/2509.25149v1) discuss stochastic
rounding, bidirectional weight scaling, and outlier treatment. Those numerical
choices are part of the training recipe, not transparent inference substitutions.
Check SM120 support separately: a library exposing an MXFP8 recipe does not imply
its full training layout is implemented on RTX 5090.

## Research sources and compatibility

The following implementations inform the optimization direction. They are not
new mandatory dependencies or interchangeable backends:

| Source | Training-relevant mechanism | Constraint |
| --- | --- | --- |
| [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) / [TensorRT](https://github.com/NVIDIA/TensorRT) | Fuse consumers, select tactics for actual shapes, reuse valid layouts and bounded workspace, and reduce host launch overhead. | These are inference frameworks. Training must preserve saved activations, weight versions, and backward semantics; this runtime does not use them as training backends. |
| [DeepInfer, V4.1 source](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9) | Async scheduling and workload-aware execution; rollout efficiency matters to GRPO. | Its inference and speculative-decoding gains do not measure a main-model training update. |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/tree/5081f8513af57e61696de4f4f04b0e707ffb503e) | Grouped forward and weight-gradient GEMMs; account for layout/scale preparation as well as GEMM time. | This version targets SM90/SM100. It is not a drop-in SM120 backend. |
| [DeepEP](https://github.com/deepseek-ai/DeepEP/tree/f6aa38fe06cd4a0489e7730365d4e0114970a2f3) | Reuse routing handles, reduce host count synchronization, overlap expert movement with compute. | This version requires CUDA 13.1+, NCCL 2.32.3+, and NVLink; inter-node execution additionally needs suitable RDMA networking. |
| [DualPipe](https://github.com/deepseek-ai/DualPipe/tree/030ce4325f4ebeb437da4ebc6d00a70469dd58ae) | Schedule forward, dgrad, wgrad, and communication with explicit dependency overlap. | Requires a compatible multi-stage schedule and additional weight/activation capacity. Renaming a pipeline schedule does not create overlap. |

The runtime already has grouped MoE kernels, routing/recompute plan reuse,
control streams, graph capture, and prefetching. Extend their coverage rather
than assuming they are missing. A future native GRPO dispatch-PP loss path must
also preserve packed position resets, temperatures, and single application of
token normalization before it can replace the current logprob/gradient path.
Likewise, preserve existing offload/capture restrictions: Surogate disables graph
capture when `offload_master` streams weights across streams. NVIDIA's
[TensorRT runtime contract](https://github.com/NVIDIA/TensorRT/blob/16ed0636df41e4a27d46f82d9028fc50fadc7200/include/NvInferRuntime.h)
also places limits on weight streaming and capture. Adding auxiliary streams
consumes activation/workspace capacity and does not remove lifetime dependencies.

Native optimizations require compilation, numerical tests, graph replay checks,
and end-to-end measurements on the target GPU. CPU-only tests cannot establish
GPU throughput or proximity to a physical limit.

After configuring the native test build for the target GPU, run the focused
regressions before a model benchmark (replace `BUILD_DIR` with that build path):

```bash
cmake --build BUILD_DIR --target integration-tests dsl-run-state-tests --parallel 4
BUILD_DIR/integration-tests '[fp8][recipe]'
BUILD_DIR/integration-tests '[lora][fused-dx]'
BUILD_DIR/dsl-run-state-tests '[dispatch-pp][host-copy]'
```

Then compare the same warmed-up training workload with and without
`SUROGATE_FUSE_LORA_DX=1`, including gradients/loss, peak memory, and optimizer-step
time. A passing kernel test does not by itself establish a training speedup.

---

# High-Level Optimizations

This section describes high-level optimizations implemented in the Surogate framework to improve memory efficiency and training speed.

## Kernel Fusions

## LoRA Fusions

Dense shared-input LoRA backward has an experimental combined input-gradient
path:

```bash
SUROGATE_FUSE_LORA_DX=1 <training-command>
```

For two or three compatible projections, each adapter's dA/dB computation stays
unchanged. The runtime packs their scaled intermediate gradients and A matrices
into existing scratch, then computes the combined contribution to dX with one
GEMM instead of repeatedly reading and updating the full dX matrix. This trades
small packing copies for fewer GEMM launches and dX read-modify-writes; it is most
plausible when the token and input-feature dimensions are large compared with
the adapter rank.

The path is off by default pending numerical and performance validation on
target hardware. It requires compatible BF16 operands, supported small ranks,
non-overlapping output slices and storage, enough existing scratch, and no active
input dropout. Unsupported layouts, dtypes, missing dX, and the transposed-A
debug mode fall back to the existing per-slice implementation before launching
the fused work. Unequal Q/K/V output widths are supported when these conditions
hold. There are no new hot-path allocations or streams.

Combining the reduction rounds dX once rather than after every BF16 slice update,
so bitwise dX parity is not expected. Compare against an FP32 reference and the
unfused path with an appropriate error bound, check adapter gradients and loss,
then measure complete step time and scratch use. Keep the option disabled if
packing costs or the numerical tradeoff outweigh the gain.

## Fused RoPE

Rotary Position Embedding (RoPE) applies position-dependent rotations to query and key vectors. The standard implementation precomputes frequency tensors (`freq_cis`) and stores them in GPU memory. The fused implementation computes these frequencies on-the-fly within the kernel.

### Enable via config

```yaml
use_fused_rope: true
```

### How it works

**Standard RoPE:**
1. Precompute a `freq_cis` tensor of shape `(T, 2*head_dim)` containing cos/sin values for all positions
2. Store in GPU global memory
3. During forward/backward, load these values from global memory for each head

**Fused RoPE:**
1. No precomputed tensor allocation
2. Each CUDA block computes cos/sin values on-the-fly using `sincosf()`
3. Results cached in shared memory and reused across all heads in the block

### When to use

Fused RoPE is beneficial when:
- Training with long sequence lengths where `freq_cis` allocation is significant
- Memory-constrained scenarios where every MB matters
- The model uses many attention layers (savings compound)

## Multi-threaded multi-GPU

## Chunked Attention

## Chunked LM

---

## See also

- [Multi-GPU](multi-gpu.md)
- [Precision & recipes](precision-and-recipes.md)
- [Back to docs index](../index.mdx)
