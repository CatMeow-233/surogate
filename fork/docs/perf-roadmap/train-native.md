# train-native synthesis: efficiency roadmap for one RTX 5090 (SM120)

> **About this copy.** This is the train-native roadmap from the no-GPU efficiency study of October 2026, kept as written except for the changes listed here. The Chinese overview of all three groups (method, top items, zero-code experiments, defects, quick wins) is [README.md](README.md).
>
> - **Line numbers** in upstream files refer to upstream commit 282f9901, the fork's merge base. The study read the branch base 488e5f00, whose upstream files are byte-identical to 282f9901 except where the fork had already patched them; the references into those patched regions (`trainer.py`, `dsl_grad_store.cpp`, `matmul.cpp`, `dsl_weight_manager.cpp`, `lora_model_utils.h`) were re-checked and converted to 282f9901 numbering. Line numbers in fork-owned files (`fork/experimental`, `fork/native`) refer to 488e5f00. Later commits on this branch edit many of the cited files, so search for the named symbol at HEAD.
> - **Proposed new files** are described in words, because they do not exist.
> - **Status.** Items fixed or implemented on this branch are marked **Status:** with the commit and the ledger topic ([patches/README.md](../../patches/README.md)); the table below lists them. Every item not marked is still open. No result on a GPU exists for any item, including the landed ones: their RTX 5090 checks are in [upstream-errata.md](../upstream-errata.md) (defects) and [performance.md](../performance.md) (performance changes).

**Status on this branch** (every item not listed is open):

| Item | Status | Commits | Ledger | Details |
|---|---|---|---|---|
| C1 | fixed: checkpoints hold the CPU AdamW moments (new format v1) and a resume reads them | 563df25e | T21 | [errata](../upstream-errata.md#cpu_training-的全量训练不保存-cpu-adamw-的矩fork-已修复ledger-t21) |
| C2 | fixed, together with two defects found while verifying it (a boundary-free micro-step during capture was captured dense; the first step with documents after a document-free capture allocated under capture) | d3d36439 | T17 | [errata](../upstream-errata.md#打包-sft-的整步-cuda-graph-在没有行内边界的微步上跨行或跨文档注意fork-已修复ledger-t17) |
| C3 | fixed: the scale is passed through to cuDNN | db4a840a | T22 | [errata](../upstream-errata.md#cudnn-注意力忽略-softmax_scalegemma3-27b-的全局层按-1sqrt128-训练fork-已修复ledger-t22) |
| C4 | fixed; review found the same int index in the FP8 abs_max/quantize loops, also fixed | e067f546, a58ff65f | T23 | [errata](../upstream-errata.md#交叉熵-kernel-按-int-计算行偏移一个-lm-head-分块超过-231-个-logits-时越界读写fork-已修复ledger-t23), [errata](../upstream-errata.md#fp8-abs_max-与量化-kernel-的循环下标是-int一次处理约-231-个元素时越界读写fork-已修复ledger-t23) |
| C5 | fixed, plus the 4/6 encode/decode scale mismatch and SR noise frozen across graph replays | c32c9996, fecde777, 2f7b5d37 | T24 | [errata](../upstream-errata.md#nvfp4-随机舍入与-four-over-six-块比例的缺陷fork-已修复ledger-t24) |
| C6 | fixed: the cache refuses trainable expert weights | 000452cd | T26 | [errata](../upstream-errata.md#moe-的-fp8-权重缓存不拒绝可训练的专家权重fork-已修复ledger-t26) |
| C7 | fixed: the frozen-expert dgrad runs by default; `SUROGATE_GPT_OSS_PREQUANT_SKIP_EXPERT_DGRAD=1` restores upstream's skip | d4b3ab07 | T25 | [errata](../upstream-errata.md#gpt-oss-预量化-checkpoint-的-lora-训练跳过冻结专家的输入梯度fork-已修复ledger-t25) |
| C8 | rejected: the fallback cannot be scheduled (no autodiff rule for `matmul_swiglu`, the fusion is forward-only, no DSL module calls it) | none | none | |
| C9 | guarded: the pre-quant blocks compute their own amax (unreachable today, bit-identical) | eb34d5e0 | T27 | [errata](../upstream-errata.md#fp8-的生产者侧预量化使用没有计算过的-amax潜在缺陷fork-已加-guardledger-t27) |
| C10 | replay half fixed, and review found the chunked KV sweep counting twice, also fixed; the warmup half is refuted (micro-step 0's `begin_step` resets the stats) | 66647915, 67ec9ef7 | T19 | [errata](../upstream-errata.md#moe-路由统计在重计算回放和分块训练的-kv-sweep-中重复累加fork-已修复ledger-t19) |
| C11 | documented; upstream docs are not edited | 2737f5b8 | none | [errata](../upstream-errata.md#lora-配置文档中的默认值与代码不符) |
| #1 | implemented, on by default; `SUROGATE_FORK_FORCE_GRAPH_WARMUP=1` restores upstream | f73ad034 | T20 | [performance.md](../performance.md#不能捕获的整步只预热一次) |
| #7 | partly implemented: step-start memset dropped (always on); parallel SIMD host accumulate and its own D2H stream (opt-in); vectorized CPU AdamW (opt-in); per-layer GPU norm deferred | 796fdfa0, e2f56d21, be5b9191, 54954e9b, ce939c53 | T13, T14, T15 | [performance.md](../performance.md#cpu_training-主机端步) |
| #24 | open, now unblocked: C2's per-row synthesis has landed | none | none | |
| #33 | the replay stats skip landed with C10's fix; the counting sort is open | 66647915 | T19 | |

Scope: the native training path on one RTX 5090 32 GB. It covers the executor and DSL, attention and long context, kernels and loss, MoE, LoRA/QLoRA, low-precision recipes, and optimizer/offload. Workloads are the ones the owner targets: dense 0.5-8B LoRA/QLoRA/FFT, ~10B-total MoE, and GRPO.

How the inputs were filtered:
- Hotspots refuted by the code skeptic are dropped: attn-4 and KL7.
- Techniques refuted by the code skeptic are dropped. So are citations refuted by the literature skeptic.
- The skeptics' corrections are applied to everything that remains. Section 4 lists what was cut and why.

No GPU was available. Every gain below is arithmetic from code paths, byte counts and FLOP counts, and each item names the 5090 measurement that would confirm it.

---

## 0. Read before acting

**The baselines are stale.** All 5090 step times quoted here come from `docs/reference/benchmarks.md` (commit bb003825, 2026-02-18). That predates three relevant commits:
- the auto FP8 weight cache (9b6ded21, 2026-09-12)
- the vec8 small-rank LoRA kernel (17546473)
- fused-dx (bc6b1dbc, ledger T2)

Every percentage below is relative to those old step times. Re-baseline the published table on current code before committing to any item.

**L2 makes small models cheaper than DRAM-only models predict.** The 5090 has 96 MB of L2. 9b6ded21 measured 0.6B fp8-hybrid LoRA at 415.9 vs 416.8 ms/step with per-op FP8 weight quantization on vs off, where a DRAM model predicts ~5 ms. Treat 0.6B "bytes saved" estimates as upper bounds that may be ~10x too high. For 8B, count only tensors larger than ~48 MB.

**Measured 5090 rates replace the spec-sheet ones.**

| Path | Rate | Source |
|---|---|---|
| BF16, FP32 accumulate | 234-255 TF at boost (209.5 nominal) | QuACK |
| Plain FP8 `mma.sync`, f32 accumulate | ~507-510 TF (half rate on GeForce) | QuACK, FlashInfer #5963 |
| Block-scaled `kind::mxf8f6f4` with unit UE8M0 scales | ~1005-1014 TF, bit-identical to plain FP8 | CUTLASS #3732 on a 5080: 0 of 7e8 accumulator registers differ, 1.55-1.72x in full GEMMs; no gap on RTX PRO 6000 |
| NVFP4 cuBLAS | 1162-1241 TF (1676 nominal) | Quartet II |

Other hardware facts:
- `sol.cpp` FP8_32 = 419 is the half-rate figure.
- FP8 with f32 accumulate keeps ~21-22 mantissa bits, so Hopper-style promotion is unnecessary.
- `sm_120a` has no hardware FP4 stochastic rounding (`cvt.rs`) and no FMUL2.
- FA2 forward on the 5090 runs at 147-209 TOPS (1k-32k causal hdim128). FA2 backward on GeForce reaches 43-90% of peak.

**What the published FP8/FP4 step times imply.** The NVFP4 GEMMs probably achieve no more than ~0.54-0.65 PF. This is the skeptic's arithmetic.

### Zero-code experiments to run first

| # | Experiment (5090) | Settles |
|---|---|---|
| Z1 | Re-run the published table on current code, with an nsys category breakdown: GEMM, quant, LoRA, attention, CE, copies, idle | all percentages |
| Z2 | Non-QLoRA MoE LoRA SFT, defaults vs `SUROGATE_DSL_GRAPH_SKIP_WARMUP=1` vs `use_cuda_graphs:false`. #1 has since landed (T20), so on this branch compare the default with `SUROGATE_FORK_FORCE_GRAPH_WARMUP=1` | #1 |
| Z3 | `recompute: false` on Qwen3-0.6B/1.7B/4B LoRA, bf16/fp8/fp4 | #6 |
| Z4 | cpu_training Qwen3-8B FFT at B=8/GA=1 vs B=2/GA=4, peak VRAM | #2 |
| Z5 | `SUROGATE_FLASH_ATTN_VARLEN_BWD_NONDETERMINISTIC=1` on packed 24k+8k rows | #5 bound |
| Z6 | Qwen3-8B FP8 LoRA: grep stderr for `[fp8 weight cache] device 0: auto:`; A/B `cpu_training=true` + `SUROGATE_RESIDENT_LAYERS=36` (frozen masters quantized in place, BF16 freed) | #8, #16 |
| Z7 | Throughput and SASS of cuBLASLt FP8 at 8192³ vs a CUTLASS SM120 unit-scale block-scaled kernel | #13, #14 |
| Z8 | `deviceQuery` asyncEngineCount, `nvbandwidth` bidirectional host-device, x8/x8 lane check | #7, #12, #22 |
| Z9 | `SUROGATE_DEBUG_MATMUL_HEURISTICS=1`: split-K and tile chosen for the LM-head dX (m=1024, n=512, k=151936) | KL6 |
| Z10 | `JEV_SAVED_OFFLOAD_TRACE=1`: bytes moved by the saved-tensor cache per layer | optim H6 (deferred) |
| Z11 | Clip-trigger rate (`NormHost[0] > max_grad_norm`) from existing SFT logs | STV decision |
| Z12 | Fraction of SFT target tokens that are −100 on the benchmark data | #18 |

---

## 1. Correctness fixes (do these before any performance work)

| ID | Defect | Trigger | Effect | Fix | Effort |
|---|---|---|---|---|---|
| C1 | **Status: fixed in 563df25e (T21).** cpu_training never checkpoints the CPU AdamW moments (`CPUAdamWState`, adamw_8bit_optimizer.cpp:161-162, not exposed by the containers at 197-203). On resume, `prepare_for_checkpoint_load` → `init_state` allocates ON_DEVICE 8-bit state for every parameter (~16.4 GB for 8B). Missing names are skipped silently (safetensors.cpp:287-295). | Any cpu_training FFT resume | Possible OOM on resume. Otherwise moments restart at zero with the resumed step `t`, giving sign-like steps up to ~3x normal size. | Expose host state through the containers (pinned memory, or a host branch in the writer); allocate host state on resume; add a CPU-only round-trip test. #9 makes the format the existing `adamw8bit.state1/2`. | M |
| C2 | **Status: fixed in d3d36439 (T17).** A captured packed graph replaying a step that has no in-row boundaries stages `cu_seqlens={0,B·T}` (py_train.cpp:1600-1606, also 1622-1626 and 1739-1742). | B>1, sample packing, full-step graph | Row b attends causally to rows 0..b-1. mem_eff already caps max_seqlen at T, so Hs>256 layers may leave rows unwritten on such steps; verify. | Synthesize per-row `{0,T,…,B·T}`. Prerequisite for #24. | S |
| C3 | **Status: fixed in db4a840a (T22), by passing the scale through.** `CuDNNAttention::supports()` ignores `softmax_scale`; attention_cudnn.cpp:330/394 hard-codes `1/sqrt(HS)`. | Gemma3-27B global layers (query_pre_attn_scalar 168, head_dim 128) on unpacked rows | Wrong attention scale | Reject when the scale ≠ 1/sqrt(Hs), or pass it through | S |
| C4 | **Status: fixed in e067f546 and a58ff65f (T23).** int32 `row_idx * P + i` in the chunked CE forward and backward kernels (fused_classifier.cu) | chunk rows × V ≥ 2³¹: fp8-hybrid LoRA+recompute at BT ≥ 28,270, or LoRA with recompute=false at BT ≥ 14,135 | Out-of-bounds reads and writes | int64 offsets | S |
| C5 | **Status: fixed in c32c9996, fecde777 and 2f7b5d37 (T24).** The 4o6 stochastic-rounding kernel (quant_nvfp4_4o6_sm100.cu:325-341, 502-507) adds fixed ±0.25 noise before round-to-nearest, using the low byte of an LCG. | Full-FT NVFP4 wgrad dout^T | Unbiased only where E2M1 spacing is 0.5. Biased on [2,6]: x=2.4 rounds up with p=0.3 instead of 0.4. | Unbiased few-bit SR with the SRC correction (2504.20634), Philox once per 16 values | S-M |
| C6 | **Status: fixed in 000452cd (T26), by refusing trainable experts.** The env-gated MoE FP8 weight cache (`SUROGATE_FP8_MOE_WEIGHT_CACHE`) is invalidated only by QLoRA refresh. | Trainable experts with the env set | FP8 expert images go stale after the optimizer step | Invalidate on optimizer step, or refuse when experts are trainable | S |
| C7 | **Status: fixed in d4b3ab07 (T25); the skip was judged a defect.** GPT-OSS prequant LoRA-only zeroes `d_input` and skips the base dgrad (moe_grouped_gemm_gate_up.cpp:1043-1056, down.cpp:902-914). | GPT-OSS MXFP4 LoRA | The input gradient to earlier layers carries only the LoRA term | Review whether this is intended; if not, restore the base dgrad (affects #27 W-C costs) | review |
| C8 | **Status: rejected; the path cannot be scheduled.** Reported, not independently verified: matmul_swiglu.cpp:336-365 recomputes mlp_up from the base weight only when the saved tensor was freed, dropping the gate/up LoRA delta. | recompute + LoRA on gate/up | Wrong activations in replay | Verify with a replay-vs-forward tensor diff | S |
| C9 | **Status: guarded in eb34d5e0 (T27).** Latent: the FP8 co-located producer pre-quant and JIT hooks call `quantize_with_abs_max` on an abs_max that nothing computes (producers receive `abs_max_ptr=nullptr`). | Currently unreachable under fp8-hybrid | Would quantize with a stale amax | Fix before enabling #29's forward half | S |
| C10 | **Status: the replay half is fixed in 66647915 (T19), the chunked KV sweep in 67ec9ef7; the warmup half is refuted.** The MoE warmup (#1) and the replay both accumulate `moe_stats_device` (moe_topk.cpp:158-192). | Default MoE SFT | Routing statistics are double counted | #1 plus a replay guard (#33) | S |
| C11 | **Status: documented in the errata (2737f5b8).** The `lora_dropout` docstring says 0.05; the default is 0 (sft_config.py:280 vs :445). | Docs | Misleading | Fix the doc | S |

---

## 2. Ranked roadmap

**Score.** score ≈ expected gain (% of step for the named workload) × confidence (H=1, M=0.6, L=0.3) × applicability / effort (S=1, M=2, L=4, XL=8).
- Applicability is 1 for owner targets: dense LoRA/QLoRA/FFT 0.5-8B, ~10B MoE, GRPO. It is 0.25 for long-context-only or NorMuon-only items.
- For memory enablers, the score is a judgement.

**Workload names used in the table:**
- **D0.6 / D4 / D8**: Qwen3-0.6B/4B/8B LoRA, rank 16, all linear layers, seq 2048, B=2, GA=4, packing (the published config).
- **F8**: Qwen3-8B FFT via cpu_training.
- **M10**: ~10B-total MoE LoRA with a BF16 base (LFM2-8B-A1B-like W-B shape).
- **W-A**: a Qwen3-30B-A3B-like layer.
- **G-MoE**: `grpo-colocate` on a 7-16B BF16 MoE.
- **G0.6**: `examples/grpo/train.yaml`.
- **L32k**: packed long-context rows.

| # | Subsystem | Hotspot | Technique | Papers (verified) | Expected gain on 5090 (workload) | Conf | Effort | Score | First experiment |
|---|---|---|---|---|---|---|---|---|---|
| 1 | executor-dsl | H1: capture-unsafe MoE graph runs an extra eager fwd+bwd every step | **Status: implemented in f73ad034 (T20), on by default.** Check `has_capture_unsafe_ops()` before the warmup (valid from `allocate_run_state`); keep the one-time stack shrink | none needed (2603.07685 §4.3.6 background) | 1.9-2.0x tok/s (non-QLoRA, non-prequant M10 SFT, BF16/FP8/FP4) | H | S | 90 | Z2: GemmGrouped count per step |
| 2 | optim-offload | H1: serial host BF16 grad accumulation in `cudaLaunchHostFunc` on the prefetch stream (GA>1) | Run cpu_training FFT at GA=1 with the largest micro-batch that fits; warn on GA>1 | 2603.16428, 2604.05091 | ~1.6-1.9x step at equal tokens/step (F8), if B=8×2048 fits | M | S | 48 | Z4 |
| 3 | attention / moe | attn-1: MoE rollout parity runs CUDA-core paged attention plus Triton 16×64 GEMMs in every training forward | `moe_rollout_policy: r3_corrected`: parity fully off, R3 routing replay, TIS/CISPO, optional score centering | 2510.11370, 2605.14220, 2609.20807 | ≥1.5x learner step (G-MoE; unmeasured; attention is 25-50% of the parity step, plus the GEMM tax) | M | S | 30 | nsys of one step, parity vs off; `mismatch_kl` over 200 steps |
| 4 | executor / moe | H5: every routed MoE intermediate is D2D-copied into persistent per-layer buffers, although recompute regenerates them | Return early from `save_moe_layer_tensors` under replay (keep `moe_out` if the `mlp_down` view needs it) | 2603.07685 §4.1.4, 1910.02653 | Peak −~15 GB (M10, BT=4096); −31 GB of D2D per micro-step (~2-4%); may decide whether M10 fits | H | S | ~20 | `saved_buffers_total_bytes()` and allocator high-water, before/after |
| 5 | attention | attn-2: deterministic FA2 varlen backward hits the 512 MiB budget, leaving 1 dQ split and 16-32 CTAs per long document | Interim: `SUROGATE_FLASH_ATTN_VARLEN_BWD_NONDETERMINISTIC=1`. Then an integer-grid (fixed-point) dQ backend | 2609.33410 | Attention backward 4.8-9.6x faster on packed 24k+8k rows (0.6B: ~12-14 s → 1.4-1.8 s); −2-3% at 2k documents | M | M | 15 | Z5 plus ncu on `cu_seqlens={0,24576,32768}` |
| 6 | executor-dsl | H2: global, binary, default-on full-layer recompute | `recompute: false` for ≤4B now; then a `recompute: auto` fit planner | 2205.05198, 2609.33570 | 1.15-1.3x (D4), 1.05-1.2x (D0.6); compute-bound ceiling 1.37-1.45x | M | S | 12 | Z3 |
| 7 | optim-offload | H2/H1/H3: serial CPU step (memset, latency-bound norm, scalar AdamW); host accumulate coupled to the prefetch stream | **Status: partly implemented (T13, T14, T15; see item 7).** Drop the step-start memset; per-layer GPU norm at GA=1; `-fno-math-errno` SIMD inner kernel; OmpTeam host accumulate on its own D2H stream | 2101.06840, 2509.21271, 2604.05091, 2608.11919 | −0.8-1.3 s/step (F8); if GA>1 stays, the ~2.3-2.8x penalty drops to ~1.1-1.3x | H | S | 12 | `fork/native/bench/cpu-optimizer` plus nsys side_stream spans |
| 8 | executor / recipes | H4/H1: 8B FP8 Auto cache does not fit, so 17 B/param of weight re-quantization per micro-step | (A) cached frozen amax + fused `quantize_and_transpose`; (B) forward-only cache with derived W^T and a tiered Auto rule | 2603.07685 §5.3.1/§5.4.3 | (A) −37 ms (~5%); (B) −70 ms (~9%) of 759 ms (D8 FP8) | H | S | 7 | Z6 plus nsys sums of abs_max/quant/transpose |
| 9 | optim-offload | H2: CPU AdamW keeps FP32 moments although `adamw_8bit` is selected (22 B/param) | SIMD CPU port of flash_adamw_8bit, with state in the existing containers (C1 has since been fixed separately by T21; see item 9) | 2602.23349, 2101.06840 | −1.4-2.3 s/step; host RAM −~49 GB (8B fits on a 64 GB host) (F8) | M | M | 5 | Same benchmark: ns/elem and DRAM bytes on the target host |
| 10 | lora-qlora | qlora-weight-remat: dequant to BF16 runs before the cache lookup; per-op FP8 re-quant without a cache | Resolve-skip behind the caches; storage→E4M3 conversion with an import-time amax; FP4 W^T cache in auto mode | 2510.24061 (bg), 2608.23816 (bg) | −23-28 ms (3%) with the cache; −75-80 ms (9-10%) without it and with no 13.9 GB cache (D8 QoFP8/QoNF4) | H | S | 4.5 | nsys dequant/abs_max/quant sums, cache on vs off |
| 11 | executor-dsl | H3: replay executes the dead down GEMM, its input quant and LoRA-down; LN1 replay uses a zero tensor | Compile-time replay-live mask; LN1 replay via `rmsnorm_forward(stored_res)` | 1910.02653 | −8.1% FLOPs (D8), −5.2% (D0.6), bit-identical; all recompute runs including QLoRA | H | S-M | 4.3 | Base GEMMs per replayed layer 4→3; gradient bit equality |
| 12 | optim-offload | H5: lm_head gathered twice per micro-step; non-final non-block grad D2H | Track buffer contents and version; D2H only on the last micro-step; prefetch lm_head on the copy stream | 2604.05091 | −117-185 ms/micro-step (8-12%) at GA=4; <0.5% at GA=1 (F8) | H | S | 4 (if GA>1) | nsys MainStream gaps around embedding/lm_head |
| 13 | kernels-loss / recipes | fp8-hybrid dense GEMMs: unknown whether cuBLASLt uses plain (half-rate) or block-scaled FP8 MMA on GeForce | Probe; if plain, a unit-scale block-scaled CUTLASS SM120 GEMM for the fp8-hybrid matmuls | hardware evidence only (CUTLASS #3732, FlashInfer #5963); 2506.08027 bg | If plain: FP8 GEMMs 1.55-1.72x, ~15-20% of the D8 FP8 step | L (until the probe) | M | 3 (→~10 if the probe is positive) | Z7 |
| 14 | moe / recipes | moe-1/H4: routed experts run BF16 grouped GEMMs under every recipe | Unit-scale block-scaled FP8 (or MXFP8) grouped GEMM for frozen experts, BF16 dgrad first | 2506.08027, 2603.07685 | Expert fwd+replay 7.0→~4.1 ms/layer, ~−60 ms/micro-batch (~20%, M10 FP8); needs +8 GB or #16 | M | L | 3.3 | Grouped microbenchmark at R=8k/16k/32k |
| 15 | recipes | H2: NVFP4 256×128 data-parallel tiles give 75% wave efficiency on 512-tile shapes | StreamKScheduler variant plus a per-shape selector (128×128 fallback) | 2301.03598 | −3-10% (D8 FP4) | M | S | 3 | N=4096 vs N=3840 microbenchmark |
| 16 | recipes / executor | H8: BF16 frozen base stays resident beside the FP8/FP4 caches | Resident quantized masters (reuse the stream_fp8/fp4 blobs, read in place); reload for export | 2603.07685 §5.3.5, 2610.25048 | −12.4-13.9 GB peak (D8); completes #8 (−79 ms) or enables recompute off or a larger B | M | M | 3 | Z6 |
| 17 | kernels-loss | KL1: backward recomputes the LM-head logits GEMM | (a) reuse the still-resident last chunk (bit-identical); (b) forward_grad: U=G·W in the forward | 2410.10989, 2511.17599 | (a) 0.7-1.9%; (b) 6.4-7.7% (D0.6 BF16/NVFP4), 3-5.5% (D8) | H | S / M | 3 | nsys logits-GEMM count in backward |
| 18 | executor / kernels-loss | SFT runs the LM head on prompt rows; opt-in compaction disables graphs and syncs | Capture-safe compaction: host-side n_valid plus a bucketed or eager head segment | 2411.09009 (App B) | ~9% of D0.6 FLOPs at 50% masked tokens (~3% D8); scales with the masked fraction | M | M | 2.7 | Z12, then A/B `lmhead_drop_ignored_rows` |
| 19 | kernels-loss | KL3: GRPO pays three no-op temperature passes, unfused softcap, and an unused argmax pass | Null `inv_temperature` when all are 1; `compute_accuracy` off for GRPO/DPO; argmax inside the chunk max | 1805.02867, 2410.10989 | −2.5% learner step (G0.6), bit-identical part | H | S | 2.5 | nsys `scale_logits_rows`/argmax sums |
| 20 | moe | moe-4: replay runs the down GEMM, down LoRA and unpermute only to get d_w | d_w = ⟨dA′, SwiGLU(H)⟩ in swiglu backward; prune via #11 | 2512.14080, 2603.07685 §4.1.2 | −51 ms/micro-batch (~7%, W-A); −22 ms (M10); not Gemma-4 | M | M | 2.4 | `SUROGATE_OP_TRACE=1` replay list; GemmGrouped count 4→3 |
| 21 | moe | moe-6: a single 128×128×32 4-stage tile (1 CTA/SM, padding) | 128×64×32 or 3-stage 128×128×32, chosen by mean rows per expert | 2607.23099 (EP=1 caveat) | ≤10% of expert GEMM at ~256 rows/expert; ≤15-20% at ~64 (GRPO MoE) | M-L | S | 2 | Tile microbenchmark plus packing parity test |
| 22 | optim-offload | H2: the CPU update is serialized after backward | Version-gated, layer-pipelined update into the next forward (exact clipping kept) | 2609.34162, 2604.27085 | Hides ~1-2 s/step once #9 lands (F8, GA=1) | M | L | 1.5 | Per-layer update timestamps vs forward waits |
| 23 | kernels-loss | KL2: under NVFP4 all three LM-head GEMMs stay BF16 | Opt-in per-tensor FP8 head through the existing FP8 head path (SFT only) | 2506.08027, 2509.25149, 2610.00053 (risk) | ~−10% (D0.6 NVFP4), ~−7.5% (D8 NVFP4); numerics risk | L | M | 1.35 | cuBLASLt FP8 heuristic at m=151936; 300-step loss A/B |
| 24 | attention / executor | attn-6/H6: cap-and-pad pins FA2 max_seqlen to B·T | **Status: open; unblocked, since C2's fix has landed.** max_seqlen = T, together with C2's per-row synthesis | — | −1.5-3% (D0.6 packed, B=2); −2-4% at B=8 | M | S | 1.2 (free with C2) | ncu grid size and duration under capture |
| 25 | recipes / kernels-loss | H5/H6/KL5: on-device FFT re-quantizes per use, does BF16 wgrad transposes, and runs short-K dW | Version-stamped trainable cache; dual-orientation wgrad quant with β-accumulate; memory-budgeted head chunks | 2603.07685 §5.4.3, 2411.09009, 2410.10989 | ~4-5% + 2-5% + 0.5-2% (Qwen3-1.7B FFT, FP8/NVFP4) | M | M | 1.2 | Quant-kernel launches per optimizer step |
| 26 | attention | attn-3: att/lse are always recomputed | Keep (or offload) O+LSE above a length threshold; part of #6's plan | 2410.06511, 2407.12117 | −12-17% step at 8-32k (Qwen3-0.6B); ~0 at 2k | M | M | 1.1 | `flash_fwd` launches per layer 2→1 |
| 27 | moe / lora | moe-2: QLoRA MoE experts are re-dequantized on every visit | Fused-dequant W4A16 grouped GEMM; W4A8 for GPT-OSS MXFP4 | 2408.11743, 2607.15810, 2610.25048 | −60-80 ms/micro-batch (8-11%, W-A NVFP4 QLoRA); −~57 ms (GPT-OSS-20B) | L-M | L | 1.1 | Per-layer dequant vs GemmGrouped time |
| 28 | lora-qlora | shared-input slices: 7 kernels per slice in backward, X read 4 times | (a) fold scaling into the dB/G GEMM α; (b) grouped LoRA with saved S | 2510.00206 | (a) ~1% (D0.6); (b) 1-4% after the L2 discount | M | S / M-L | 0.6 | ncu L2 hit rate on the S/dA GEMMs |
| 29 | recipes | H3/H7: separate amax passes; dead `from_amax` hook | Wire the producer amax into `get_fp4_forward_buffers` (bit-identical); delayed E5M2 scale for FP8 dout | 2509.25149, 2601.22813, 2610.25048 | 1-2% (D0.6 FP4); ~1% (D8 FP8) | M | S-M | 0.6 | Launch counts for abs_max, memset and alpha |
| 30 | optim-offload | H4: NorMuon Newton-Schulz uses full GEMMs at batch 1 | Megabatching + FP16 + workspace fix; then Gram NS with a symmetric SM120 GEMM | 2608.11612, 2606.27153, 2610.26742 | 4B NS 1.17→0.54 s per optimizer step (~10-15% at GA=4) | M | S-M / L | 0.45 | Per-shape NS kernel time |
| 31 | attention | attn-5: redundant d_qkv zero-fill; scalar dK/dV reduces | Skip the fill when cu_seqlens covers all rows and EP is off; vectorize the reduces | — | ~0.5-1% (D0.6) | M | S | 0.4 | Per-layer kernel list |
| 32 | attention | long context: attention runs at the BF16 rate | INT8/FP8 attention backend (SageBwd-style plus the Delta-Matching delta) | 2505.21136, 2505.11594, 2603.02170, 2609.37852 | Attention fwd+bwd 1.2-1.5x; 10-35% end to end at 8-32k | L | L | 0.4 | Per-kernel TOPS plus gradient cosine |
| 33 | moe | moe-8: ~18 dependent routing ops per layer pass | **Status: the replay stats skip landed in 66647915 (T19); the counting sort is open.** Skip the stats kernel in replay; counting-sort scan-and-place | 2512.14080 App D, 2603.07685 §4.3.3 | ~1% at BT=4096; more for GRPO micro-batches | M | S / M | 0.3 | Launch count between the router GEMM and GemmGrouped |
| 34 | kernels / lora / executor | KL6/H7: per-call cuBLASLt descriptors and `getenv` | Plan cache consulted before the heuristic; static env reads | 2503.19779 (analogy only) | ~0 when graphed; a few ms for eager QLoRA 0.6B | L | S | 0.3 | nsys `cuda_api_sum` with CPU sampling |
| 35 | lora-qlora | lora-out-rmw: per-slice output read-modify-write | Row-tiled fused gate/up accumulate + SwiGLU kernel | 2510.00206, 2411.05007 | ≤2-4% (D8 BF16), ≤4-8% (D8 FP4); ~0 at 0.6B | L | L-XL | 0.15 | ncu DRAM bytes of the gate_up read-modify-write |
| 36 | executor-dsl | H7: GRPO/DPO run fully eager | Capturable native GRPO step, or split-segment graphs for eager modes | 2503.19779, 2610.23861 | Unknown; GRPO 0.6B is not host-bound on average | L | L | measure first | nsys GPU-idle % on `examples/grpo/train.yaml` |

---

## 3. Per-item detail

**General module rule.** Fork C++/CUDA goes in `fork/native/src` and `fork/native/include/surogate_fork`, compiled into `surogate-common` through `fork_native.cmake` (do not create new files in `csrc/src`). Tests go in `fork/native/tests`. Python glue goes in `surogate/_fork`. Every edit to an upstream file belongs to a new ledger topic (T30 and later: this branch has taken T11-T29), or extends the topic that already owns that file (at the base: T2: lora_slice_dispatch.h; T4: matmul.cpp and fp8_hybrid_recipe.cpp; T5: cpu_adamw.cpp; T10: dsl_grad_store.cpp; T1: py_train.cpp and graph_executor. Since then T13/T14 also own dsl_grad_store, T15 cpu_adamw.cpp, T17/T18 graph_executor, T17/T20 py_train.cpp and T21 adamw_8bit_optimizer.cpp; [patches/README.md](../../patches/README.md) lists every topic's files). Run `fork/dev map` and `fork/dev check`. Anything that changes numerics sits behind an explicit switch that defaults off. Record results in `fork/docs/performance.md` with hardware and workload.

**Measurement terms.** Throughput is tok/s over ≥50 post-warmup steps, attributed with `nsys stats` (`cuda_gpu_kern_sum`, `cuda_gpu_mem_size_sum`, `cuda_api_sum`). Peak is the allocator high-water (DebugMemoryBreakdown) plus `cudaMemGetInfo`. Lifetime is how long a buffer is held: whole run, step, forward-to-backward per layer, or per op. Allocation traffic is allocator churn per step: counts and bytes of `cudaMalloc`/`cudaMallocAsync`/`temp_alloc`/stack, from nsys `--cuda-memory-usage=true` and allocator counters. It is reported separately from DRAM bytes (ncu `dram__bytes_read/write.sum`, `lts__t_sector_hit_rate.pct`).

### 1. Skip the full-step warmup when capture is impossible
- **Status: implemented in f73ad034 (ledger T20).** On by default, with `SUROGATE_FORK_FORCE_GRAPH_WARMUP=1` to restore upstream. The decision is `run_warmup` in `fork/native/include/surogate_fork/full_step_warmup.h`. Differences from the plan below: a capture-unsafe model warms up once, on its first executed step, so the one-time stack shrink stays; `ep_size > 1` keeps upstream's per-step warmup, because the warmup advances the sticky EP plan's age; and the side effect on routing stats does not exist (see C10). Not run on a GPU; the 5090 check is in [performance.md](../performance.md#不能捕获的整步只预热一次).
- **Evidence.**
  - py_train.cpp:1506 sets `warmup_full_graph = !gs.captured && !env`.
  - The capture-unsafe branch (1554-1566) returns before capturing, so `gs.captured` is never set. Every step therefore runs a warmup fwd+bwd over all micro-steps, `zero_grads`, and a stream sync, followed by the real eager pass.
  - The graph is already compiled in `allocate_run_state` (dsl_model_execution.cpp:507-508), so the check is valid on step 0.
  - MoE*/Ep* ops are capture-unsafe (graph_executor.cpp:72-98). trainer.py:1276 forces full-step graphs, and no Python rule turns graphs off for MoE.
  - Affected: non-QLoRA, non-prequant MoE SFT. Unaffected: shipped QLoRA/prequant/EP MoE examples (their graphs are already off).
- **Side effects.**
  - The one-time stack shrink happens only inside the warmup (py_train.cpp:1531-1550), and the Python post-step shrink is gated off when `use_cuda_graphs` is true (trainer.py:1599-1605). Keep the warmup on step 0 only, or call `_maybe_shrink_stack_after_warmup` for capture-unsafe graphs.
  - Reported routing stats halve, because the double counting (C10) goes away. (Refuted while implementing: micro-step 0's `begin_step` resets the stats after the warmup, so T20 changes no reported stat.)
- **Boundary / switch.** One hunk in `train_step_graphed` (new ledger topic). On by default; `SUROGATE_FORK_FORCE_GRAPH_WARMUP=1` restores the old behaviour. Today's workaround is `SUROGATE_DSL_GRAPH_SKIP_WARMUP=1`.
- **Measure.** M10 LoRA in BF16 and FP8.
  - Throughput: GemmGrouped launches per optimizer step should go from 2·GA·L·passes to GA·L·passes; step time.
  - Peak: unchanged, since the two passes never overlap. Check the stack size is still shrunk.
  - Allocation traffic: unchanged per pass, halved per step.
  - Loss must be bit-identical at a fixed seed, because the warmup restores RNG state.
- **Success.** ≥1.8x tok/s, identical loss for 100 steps, peak within ±1%.

### 2. cpu_training FFT: GA=1 with the largest micro-batch that fits
- **Evidence.**
  - On micro-steps ≥1, a serial BF16 add (dsl_grad_store.cpp:1055-1072) runs in `cudaLaunchHostFunc` on `side_stream`, the same stream that carries the H2D prefetch (compiled_ops.cpp:1093-1106). Backward then runs at host-loop speed.
  - On a desktop core that is ~80-115 ms per W8 layer against 26 ms of compute, adding +7.5-11 s per GA=4 step to 6 s of GPU work.
  - GA=1 never enters the staged path. H2D drops from 111 to ~26 GB per step (the version skip reuses the boundary layers), D2H from 55.6 to 13.9 GB, and #12's costs drop 4x.
  - SlideFormer Table 1 (4090 + 9950X): the overlap factor η goes from 0.66 to 1.55 when batch goes from 16 to 32.
  - Loss scaling is preserved (adamw_8bit_optimizer.cpp:695-705).
- **Caveats.**
  - This does not remove the 2.6-4.3 s CPU-step tail; pair it with #7 and #9.
  - B=8×2048 at 8B needs the chunked LM head (BF16 logits alone are ~5 GB).
  - `examples/training/cpu-offload.yaml` is Qwen3-0.6B, which fits on-device anyway. The policy matters for 4-8B.
- **Boundary / switch.** Config only. Add a GA>1 warning in the cpu_training branch of sft_config.py (ledger topic) and a fork-owned example config.
- **Measure.** F8 and Qwen3-4B FFT at B=2/GA=4 vs B=8/GA=1, same tokens per step.
  - Throughput: step time; nsys `side_stream` host-function spans per layer.
  - DRAM/PCIe: H2D and D2H bytes per step.
  - Peak: VRAM via `cudaMemGetInfo`; host RSS.
  - Lifetime: grad slots go from per-layer reuse to one D2H per layer per step.
  - Loss over 200 steps within run-to-run noise (packing changes with B).
- **Success.** ≥1.6x throughput at equal tokens per step, no OOM.

### 3. MoE GRPO colocate: `r3_corrected` instead of bitwise rollout parity
- **Evidence.**
  - native_colocate.py:202-203 forces `moe_rollout_parity` for any `num_experts`.
  - Under parity, every non-decode forward goes through `consistent_forward(false)`: scoring, the training forward and its recompute replay. That is CUDA-core `page_attention`+`merge_pages`, about 20x slower than FA2 at 2k documents and 35x at 512.
  - `ScopedGlmMatmul` also sends every dense and grouped GEMM through Triton 16×64×32 fixed-reduction kernels and disables grouped MoE LoRA.
  - Attention is plausibly 25-50% of the parity step (skeptic correction).
  - The R3 plumbing already exists (`set_routing_replay`, `set_routing_capture`). Prefix-cache hits produce −1 routing rows, which are not replayed.
- **Literature.**
  - R3: k3-KL 1.535e-3 → 7.5e-4 and prevents collapse. TIS on top of R3 gave no clear gain.
  - VeXact: TIS plus sequence rejection tracks the zero-mismatch reference (dense 1.7B).
  - Score centering tolerates FP8/FP4-KV sampler mismatch, but it was run without routing replay.
  - Residual mismatch after switching: FA2 rounds P to BF16, cuBLAS algorithms differ between decode and training shapes, and gated-delta layers leave the parity branch. This is ordinary trainer-vs-sampler mismatch.
- **Boundary / switch.** `moe_rollout_policy: exact | r3_corrected` in `surogate/_fork`, plus one hunk in native_colocate.py (ledger). `r3_corrected` turns parity off entirely; switching only attention would keep the GEMM tax and still lose parity. Fail fast if unreplayed routing rows exceed a threshold.
- **Option for exact-mode users.** Size `np` per query tile in decode.cu. This is bit-identical (merge_pages skips l=0 partials) and cuts scratch and launches by up to 87% when T is much larger than the document length (S).
- **Measure.** G-MoE, one step, nsys.
  - Throughput: sum of `page_attention`+`merge_pages`, `glm_matmul*`, grouped GEMMs and `flash_fwd`; learner `train/duration_ms` (T6 metrics).
  - Peak: page-partial scratch (~136 MB per tile) goes away.
  - Quality over ≥200 steps: `mismatch_kl` (k3), fraction of tokens with ratio >2, reward.
- **Success.** Learner step ≥1.5x faster, `mismatch_kl` ≤2x the dense level (~6.4e-4), no reward collapse relative to exact mode.

### 4. Do not persist MoE intermediates that recompute regenerates
- **Evidence.**
  - compiled_ops_execute_forward.cpp:690-693 calls `save_moe_layer_tensors` unconditionally.
  - compiled_ops_save.cpp:140-221 copies every MoE-named tensor into per-layer persistent buffers (a 256 MiB arena, then `cudaMalloc`). Unlike `persist_forward_saved_layer_tensors`, it has no recompute check.
  - Replay regenerates the same tensors and copies them again into the replay-persist arena (compiled_ops_execute.cpp:485-530). Replay adds no host sync, because offsets are reused.
  - Example (C=2048, E=64, k=8, M_e=1024, 4096 tokens): 0.49 GB/layer, 15.6 GB over 32 layers, plus ~31 GB of D2D copies per micro-step. Beside ~20 GB of BF16 weights, this config may not fit on 32 GB today.
- **Hidden dependency.** `mlp_down` is a view of `moe_out` (qwen3_moe.py:150-163), and the save rebinds `moe_out` to the persistent copy. If the next layer's forward relies on that copy, keep persisting `moe_out` (16.8 MB/layer) and skip the rest. Also check whether replay itself calls `on_fwd_layer_end`.
- **Boundary / switch.** Guard using the same predicate as `prepare_saved_buffers_for_capture` (ledger topic); `SUROGATE_FORK_MOE_PERSIST_SAVES=1` restores the old path.
- **Measure.** M10, recompute on.
  - Peak: `saved_buffers_total_bytes()`, allocator high-water, `cudaMemGetInfo`; expect −~15 GB.
  - Lifetime: buffers held across steps today; afterwards only the replay arena for one layer.
  - Allocation traffic: `cudaMalloc` count beyond the arena should go to 0.
  - Throughput: nsys D2D bytes per forward; step time.
  - Gradients bit-identical.
- **Success.** Peak −≥12 GB, bit-identical gradients, no step-time regression, and BT=4096 trainable for M10 if it OOMs today.

### 5. Long-document deterministic dQ
- **Evidence.**
  - attention_backend.h:57-77 sets splits = clamp(ceil(170/(B·Hq)), 1, min(4, 512 MiB / stride)).
  - At T=32k that is 1 split, so 16 CTAs (Hq=16) or 32 (Hq=32) on 170 SMs. In a 24k+8k row the attention backward runs 4.8-9.6x longer than with a full grid.
  - Applies to: packed long rows, any batch with in-row boundaries, sequence_chunks/kvprefix, sliding-window and Hs>128 layers, and MoE parity.
  - Unpacked single-document rows use cuDNN and are not affected.
  - At 2k documents the split copies cost ~2-3%, which matches upstream's measured +3.2%.
- **Technique (FoldAttention 2609.33410).**
  - A per-(document, head) bound: scale·max‖K‖·(max‖dO‖·max‖V‖ + max|D|).
  - An int32 RED replaces the fp32 `atomicAdd` in the FA2 dQ epilogue (flash_bwd_kernel.h:676); `convert_dQ` rescales.
  - The result is order-free, so it is bit-identical across runs and packings while keeping the full nondeterministic grid.
  - Error ≤ B·2⁻²¹ at 32k (C≈512 key blocks).
- **Boundary / switch.** A patched FA2 copy as a new registry backend `flash_varlen_intgrid` (fork/native patch script and TU; FA2 is already patched via cmake/patch_flash_attn.sh). `attention_bwd_dq: split|atomic|intgrid`, first via env.
- **Measure.** Microbenchmark `cu_seqlens={0,24576,32768}`, Hq 16 and 32.
  - Throughput: ncu `gpu__time_duration` and `sm__cycles_active.avg` of `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` for split, atomic and intgrid; L32k step time.
  - Peak: dQ scratch drops from splits×stride to one int32 buffer.
  - Allocation traffic: memset and convert bytes.
  - Correctness: bitwise equality across 2 runs and across 2 packings; dQ rel-L2 against an fp64 reference.
- **Success.** intgrid within 10% of atomic at 32k, bitwise reproducible, error ≤ the atomic BF16 path, and −≥1.5% at 2k.

### 6. Memory-aware recompute
- **Evidence.**
  - `RecomputeLevel` is binary. With recompute on, the save set is empty and every op is replayed.
  - Replay is 0.46G of 1.68G MAC/token at 0.6B (27%) and 7.0G of 22.3G at 8B (31%).
  - At 32 GiB ≈ 34.4 GB, 4B full saves (17.7 GB plus 8 GB weights) fit; 8B (22.4 + 16.4 GB) does not.
  - 0.6B runs at ~101 TF (~48% of BF16 peak), so its compute-bound ceiling is loose.
  - HiLoRe's +3.5-14% includes lossy units. Korthikanti App C measured only +0.4-0.7% MFU, in a pipeline-parallel context.
- **Boundary / switch.**
  - Step 1 needs no code: `recompute: false`.
  - Step 2: `recompute: auto`. A CPU-testable planner header (a new file under `fork/native/include/surogate_fork`) estimates saved bytes from the compiled no-recompute plan, plus CUDA-graph pools, FP8/FP4 caches, LM-head chunk, workspace and a margin.
  - Per-layer counts are M-L effort, not M. `forward_replay_active`, `mRecomputeEnabled`, `compute_layout(fwd_per_layer_sections)`, `finalize_save_for_bwd` and the RecomputeBlock/SegmentDispatch triggers are all global today. The unused IR `recompute` list (ir.cpp:205-207) can carry the plan.
  - The default stays `true` until measured.
- **Measure.** D0.6/D1.7/D4 × bf16/fp8/fp4, same B/T/GA.
  - Throughput: tok/s; nsys share of replay kernels.
  - Peak: allocator high-water.
  - Lifetime: saved arena held forward→backward per layer.
  - Allocation traffic: replay temporaries removed.
  - Gradients bit-identical across auto/true/false (the `RecomputeLevel::None` contract).
- **Success.** auto picks `false` for ≤4B with no OOM over 1000 steps; ≥1.15x tok/s on D4 BF16.

### 7. cpu_training CPU-step and host-accumulate quick wins
- **Status: partly implemented.** [performance.md](../performance.md#cpu_training-主机端步) has the design, the CPU measurements and the 5090 checks.
  - The step-start memset is gone: 796fdfa0 and e2f56d21 (ledger T13), always on, host gradients byte-identical.
  - The parallel SIMD host accumulate (`SUROGATE_FORK_HOST_GRAD_ACCUM=1`) and the block-gradient D2H on its own stream (`SUROGATE_FORK_GRAD_D2H_STREAM=1`) landed in be5b9191 and 54954e9b (T14), both default off. The kernels are in `fork/native/src/cpu_step_kernels.cpp`. The residual put was not moved, so `compiled_ops.cpp` is untouched.
  - The vectorized CPU AdamW (`SUROGATE_FORK_CPU_ADAMW_SIMD=1`, default off, refused where the baseline ISA has FMA since ce939c53) landed in be5b9191 (T15).
  - Deferred: the per-layer GPU norm. It is valid only at GA=1 or breadth-first, and it cannot equal the CPU `ordered_block_sum` norm bit for bit without that summation order.
- **Evidence.**
  - `start_micro_step(0)` memsets every host grad (dsl_grad_store.cpp:296-299), and D2H overwrites all of them: 0.55-0.8 s/step at 8B.
  - The CPU norm is a separate, latency-bound pass: 0.25-0.5 s.
  - `std::sqrt` under math-errno blocks vectorization: 4.47 → 1.40 ns/elem with `-fno-math-errno`, measured with library flags. On a 16-core host the AdamW pass is already near DRAM-bound (2.6 s at 70 GB/s), so vectorizing matters mostly on 6-8-core hosts.
  - The host BF16 accumulate is serial and blocks the prefetch.
  - FP32 host accumulation has a 10 B/elem floor (1.93 GB per W8 layer, 28-46 ms ≥ 26 ms of backward). That is why GA=1 (#2) is the real fix and this is a mitigation.
- **Boundary / switch.**
  - SIMD inner kernels in fork/native/src, under T5 (OmpTeam, per-source `-fno-math-errno`). Put `target_clones` on a serial kernel each thread calls; a clone around `#pragma omp parallel` may not propagate the ISA. Use a branch-free bit-op RNE BF16 round.
  - Move grad D2H and the residual put to a dedicated D2H stream (upstream hunks in compiled_ops*.cpp, new topic), `SUROGATE_FORK_GRAD_D2H_STREAM=1`. A new stream alone is not enough: `prepare_layer_grads(L)` waits on `d2h_done(L+2)`, which queues behind `hostfn(L+3)`. Parallelize the add as well.
  - Per-layer GPU squared norm into the already allocated `mLayerNormAccum` (the TODO at dsl_grad_store.cpp:970-977), valid at GA=1 or breadth-first only.
- **Measure.**
  - The `fork/native/bench/cpu-optimizer` benchmark (runs CPU-only): ns/elem.
  - `perf stat` uncore IMC DRAM bytes.
  - nsys `side_stream` occupancy and host-function spans.
  - Host RSS (peak host memory) and pinned bytes.
  - Norm bitwise equal to `ordered_block_sum`.
- **Success.** Memset gone; norm bitwise identical; AdamW within 15% of the host's measured DRAM-bound time; F8 −≥1 s/step.

### 8. FP8 frozen-weight cache tiers
- **Evidence.**
  - The Auto check (graph_executor_weight_cache.cpp:157-191) needs forward plus W^T, i.e. 13.9 GB plus max(2 GiB, 10%), which fails at 8B.
  - Without the cache, forward and replay each pay abs_max+quantize (5 B/param) and dgrad pays quantize + `transpose()` + a D2D scale copy (7 B/param). That is 17 B/param, ~118 GB, ~79 ms per micro-step.
  - (A) Cache one amax float per frozen weight. `quantize_with_abs_max` already reads a device amax, and `quantize_and_transpose_with_abs_max` exists. That gives 9 B/param, ~−37 ms, with bit-identical FP8 images.
  - (B) Forward-only cache: the recipe already transposes a cached (N,K) FP8 weight per call (fp8_hybrid_recipe.cpp:441-461). Only the fallback from `mFP8CacheT` to `mFP8Cache` in matmul.cpp:874-880 is missing. That gives 2 B/param, ~−70 ms, and needs ~10.4 GB free, which is borderline. Use an MLP-first knapsack.
- **Boundary / switch.** Extend T4 (matmul.cpp, fp8_hybrid_recipe.cpp) and the weight-cache planner. `fp8_weight_cache: tiered` (env first).
- **Measure.** D8 FP8.
  - The `[fp8 weight cache]` line.
  - Throughput: nsys sums of `reduce_abs_max_kernel<bf16>`, `quantize_with_abs_max_kernel<e4m3>` and `transpose_kernel`; DRAM bytes via ncu; step time.
  - Peak: +6.95 GB for (B).
  - Allocation traffic: `temp_alloc`s per GEMM 4→1.
  - Calibrate against Z6.
- **Success.** (A) −≥25 ms/micro-step with bitwise-equal images and loss; (B) −≥50 ms with no OOM over 1000 steps.

### 9. SIMD CPU 8-bit AdamW
- **Evidence.**
  - `step_cpu_streaming` keeps FP32 `m` and `v` even when `adamw_8bit` is selected: 22 B/param of traffic (W8: 180 GB, 2.6-4.3 s at 70-42 GB/s) and 65.5 GB of host moments.
  - Porting the device flash_adamw_8bit math (int8 m/v, FP16 group scales, BF16 master) gives ~10 B/param (82 GB, 1.2-2.0 s). Host RAM becomes ~50 GB total for 8B, and the state matches the existing `adamw8bit.state1/2` checkpoint names (fixes C1). **C1 has since been fixed by T21 (563df25e) with its own format v1 (FP32 `cpu_adamw.m.*` and `cpu_adamw.v.*` plus a `cpu_adamw.format` version tensor). This item must therefore use new names or a new format version, convert or refuse v1 files, and keep a completeness check; the rules are in [performance.md](../performance.md#cpu_training-的-adamw-状态-checkpoint).**
  - LazyTrain saw a 0.3% gain from 8-bit CPU state, because the kernel must stay DRAM-bound. Benchmark it first.
  - FlashOptim's companded int8 is an alternative codec: GSM8k 75.09 vs 74.98 on 8B SFT. A 24-bit split master needs int16 correction for 1-D tensors at lr 2e-5.
- **Boundary / switch.** A new fork source in `fork/native/src` (T5 OmpTeam), called from `step_cpu_streaming` (new topic). Selected by `optimizer: adamw_8bit`; `SUROGATE_FORK_CPU_ADAM_FP32_STATE=1` keeps the old path.
- **Measure.**
  - Benchmark ns/elem and uncore DRAM bytes.
  - Peak: host RSS (expect −~49 GB).
  - Lifetime: state persists across the run either way.
  - Throughput: F8 step time.
  - Quality: 1-2k-step SFT loss vs FP32 moments.
  - Checkpoint round-trip test.
- **Success.** ≤12 B/param measured; host RSS −≥45 GB; loss within noise.

### 10. QLoRA: resolve-skip, direct storage→FP8, FP4 W^T auto
- **Evidence.**
  - `resolve_weight()` (dequant into the 4-buffer LRU pool) runs before the `mFP8Cache` lookup (matmul.cpp:403-427). The backward resolve is unconditional (matmul.cpp:621; matmul_swiglu.cpp:327-329).
  - With the cache primed, the waste is 2(s+2) B/param ≈ 35-42 GB, 23-28 ms at 8B, and ~0 at 0.6B because of L2.
  - Without a cache, the cost is ~22 B/param, 102-107 ms. Direct storage→E4M3 conversion costs ~3(s+1) B/param. Compute each frozen tensor's exact amax once at import.
  - Do not revive `fused_dequant_requant_per_block_to_tensor`: it does a D2H, a host sync, and pageable H2D per call.
  - NVFP4 recipe: `SUROGATE_ENABLE_FP4_WEIGHT_CACHE_T` exists but backward still resolves first. Auto-priming W^T (+3.6 GiB at 8B) plus resolve-skip saves ~33 ms/micro-step for prequant NVFP4 8B. The QoFP4 benchmark runs fp8-hybrid, so it does not exercise this.
- **Boundary / switch.** A `QuantizedOperandProvider` interface with the fork implementation in fork/native/src; the resolve moves behind the cache checks (extend T4). `SUROGATE_FORK_QLORA_DIRECT_FP8=1`.
- **Measure.** D8 and D4 QoFP8/QoNF4/QoFP4, `fp8_weight_cache` on and off.
  - Throughput: nsys sums of `dequantize_per_block*`, `kDequantizeBnBNF4*`, `dequantize_fp4_block_kernel`, abs_max, quant and transpose; DRAM bytes.
  - Peak: no-cache vs cache.
  - Lifetime: the BF16 pool buffer is no longer filled on cache hits.
  - Numerics: ≤1 E4M3 ulp vs the BF16 path.
- **Success.** −≥15 ms (cache primed) and −≥50 ms (no cache) per 8B micro-step, with loss parity.

### 11. Replay-live mask
- **Evidence.**
  - The replay loop skips only loss ops (compiled_ops_execute.cpp:336-339).
  - In a dense SwiGLU block, `mlp_down` is read only by the next layer's fused residual norm, whose backward reads `residual_out` and `rstd`.
  - Removing the down GEMM, its input quant and both LoRA-down GEMMs saves 3.15M MAC/token/layer at 0.6B and 50.3M at 8B, plus 3 nodes per layer and, under QLoRA, one W_down dequant.
  - The bundled LN1 fix: replay currently allocates and memsets a zero B·T·C tensor. Use `rmsnorm_forward(stored_res)`, which is bit-identical (same lane striding, `warpReduceSum`, `rsqrtf`), and rebind or copy `residual_out`. Worth ≤0.2%.
- **Hidden consumers.**
  - The FP8 ready flag: a skipped down matmul leaves `FP8Ready_SwiGLU` set, so skip the SwiGLU pre-quant or clear the flag.
  - `replay_saved_scope` and its exceptions.
  - `mlp_tile_groups`.
  - SSA aliases (`ssa_base_to_id`).
  - Derive the mask from actual backward reads. MoE and Gemma post-norm blocks then keep `mlp_down` automatically.
- **Boundary / switch.** A per-op replay bit in `CompiledGraph` computed by graph_compiler (new topic). `SUROGATE_FORK_REPLAY_DCE=0` disables it.
- **Measure.** D0.6 and D8 × bf16/fp8/fp4 plus QLoRA.
  - Throughput: base GEMMs per replayed layer 4→3; step time.
  - Peak: unchanged except one stack temp per layer.
  - Allocation traffic: −1 B·T·C replay temporary per layer.
  - Full-gradient dump comparison.
- **Success.** Bit-identical gradients; −≥4% (8B) and −≥2.5% (0.6B).

### 12. cpu_training non-block weights and grads
- **Evidence.**
  - The shared embedding/lm_head buffer plus the blanket `!cpu_training` version-skip disable (dsl_weight_manager.cpp:1081-1086) cause lm_head to be gathered twice per micro-step (forward fused loss, then backward) and the embedding once, all on MainStream.
  - Non-block grads are copied D2H every micro-step (dsl_grad_store.cpp:1027-1053), although only the last copy is used.
  - Untied 8B: ~117 ms per micro-step, ~185 ms with an FP32 master.
- **Boundary / switch.** Tag the buffer's tensor and version; gate the D2H on `mIsLastMicroStep`; prefetch lm_head on the copy stream after the embedding lookup. This touches dsl_grad_store (T10: coordinate) and dsl_weight_manager (new topic).
- **Measure.** F8 at GA=4 and GA=1.
  - nsys MainStream idle gaps; H2D/D2H bytes per step.
  - Peak unchanged (or +1.24 GB for the optional separate head buffer).
  - Step time.
- **Success.** −≥90 ms per micro-step at GA=4.

### 13. Dense FP8 GEMM instruction rate (cross-cutting, derived)
- **Evidence.** On GeForce, plain FP8 `mma.sync` with f32 accumulate is half rate (~510 TF). The block-scaled form with unit scales is ~1014 TF and bit-identical (CUTLASS #3732: 1.55-1.72x in full GEMMs at M≥4096 on a 5080). The published 8B FP8/BF16 ratio (5.4k/3.8k = 1.42x) is consistent with plain-rate FP8, but which instruction cuBLASLt issues for fp8-hybrid on sm_120 is unknown.
- **Boundary / switch.** A CUTLASS SM120 per-tensor FP8 GEMM using the block-scaled atom with constant UE8M0=0x7F scales, with the per-tensor scales folded into the epilogue. Put it in a fork/native/src TU called from the fp8_hybrid forward/backward matmul, with cuBLASLt as the fallback. `SUROGATE_FORK_FP8_GEMM=cublaslt|cutlass_bs`.
- **Measure.**
  - Z7: TFLOPS at 8192³ and at the Qwen3-8B shapes; ncu tensor-pipe utilization.
  - Then D8 FP8 step time.
  - Numerics: bitwise against a plain-FP8 reference kernel; within tolerance of cuBLASLt (different accumulation order).
- **Success.** Promote this item only if cuBLASLt runs below ~600 TF at 8192³. Target ≥1.4x FP8 GEMM throughput and −≥12% D8 FP8 step.

### 14. Low-precision routed-expert GEMMs
- **Evidence.**
  - The gate_up/down ops build `MoeMatmulContext` without `inp_quant`, and `weights_fp8`/`weights_fp4` are never written. Every recipe therefore runs the CUTLASS 2.x Sm80 BF16 grouped kernel: 141-153 TF at ~35k rows, 77-84 TF at 8k.
  - CUTLASS v4.6.0+ (the pinned version is v4.6.1) ships SM120 ptr-array FP8 grouped GEMM (tensor/token-scaled), 87c groupwise, and 79d NVFP4 grouped.
  - The unit-scale block-scaled route keeps fp8-hybrid's per-tensor numerics.
  - W-B arithmetic: FP8 forward+replay with BF16 dgrad goes from 7.0 to ~4.1 ms/layer/micro-batch, ~−60 ms per micro-batch. This assumes the grouped kernel reaches ~62% of dense FP8, i.e. ~380-410 TF.
- **Constraints.**
  - The FP8 expert copy is +8 GB. With a ~10B BF16 base it needs #16 or quantized storage.
  - The LoRA kernel reads BF16 permuted rows.
  - Dgrad needs a K-major copy unless QuACK-style MN-major fp8 loads with 2D-consistent scales are used.
  - The vendored TRT-LLM SM120 kernels are FP4×FP4 and FP8×FP4 only; there is no bf16×fp4 kernel on SM120.
- **Boundary / switch.** A new SM120 FP8 grouped-GEMM source in `fork/native/src`, recipe hooks for the forward plus a new dgrad hook (new topic). `moe_expert_precision: bf16|fp8`.
- **Measure.**
  - Raw instruction microbenchmark; grouped GEMM at R=8k/16k/32k with W-B histograms (ncu duration, tensor pipe).
  - nsys GemmGrouped time bf16 vs fp8-hybrid; identical today, which is itself the proof of the gap.
  - Peak with the FP8 expert cache.
  - DRAM bytes per layer once quantization is fused into permute/SwiGLU.
  - LoRA loss A/B.
- **Success.** Grouped FP8 ≥2.5x the in-tree BF16 grouped TFLOPS at R=16k; W-B FP8 step −≥15%.

### 15. Stream-K for the SM120 NVFP4 GEMM
- **Evidence.**
  - 256×128×128 tiles with `StageCountAutoCarveout` (one CTA per SM) and the `void` scheduler, which maps to data-parallel persistent scheduling.
  - At M=4096, 512 tiles take 4 waves, ~75% utilization. FLOP-weighted efficiency is 0.876/0.792/0.915 (fwd/dgrad/wgrad) at 8B and 0.813/0.753/0.59 at 0.6B.
  - CUTLASS v4.6.1 supports `StreamKScheduler` for SM120 NVFP4 (unit test `sm120_bs_gemm_nvf4_nvf4_f32_f32_stream_k.cu`).
  - The serve kernel `w4fp4_cutlass_gemm.cu` shows the pattern: epilogue tile `Shape<_64,_32>`, explicit `KernelTmaWarpSpecializedCooperative`, `hw_info.sm_count`.
  - Serve notes (PATCHES.md) found stream-K slower for small-token tiles, so select per shape (long K, poor last wave). The 128×128 tile measured 679-813 TF/s at 4B shapes.
  - The reduction is deterministic but not bit-equal to the data-parallel kernel.
- **Boundary / switch.** A second instantiation plus a host selector (new topic, or a fork TU). `SUROGATE_FORK_FP4_SCHED=dp|sk|auto`.
- **Measure.**
  - Microbenchmark at M=4096, K=24576 with N=4096 (512 tiles) vs N=3840 (480 tiles). Data-parallel scheduling predicts a ~1.33x time ratio for a 1.067x FLOP ratio.
  - Then the SK A/B: `gpu__time_duration`, `sm__throughput`.
  - Workspace partials ~22 MB, within the 128 MB workspace.
  - D8 and D0.6 FP4 step time; determinism across runs.
- **Success.** ≥10% faster on 512-tile shapes, no >2% regression on any Qwen3 shape, step −≥3%.

### 16. Resident quantized frozen base
- **Evidence.**
  - After priming, the BF16 copy of each quantized frozen block weight is dead in SFT/LoRA. Exceptions to handle: skip_quant layers, the BF16 fallback when cuBLASLt throws, `SUROGATE_DISABLE_FP4_WEIGHT_CACHE_T`, transposed priming at backward entry, export/merge/eval, and GRPO colocate (shared BF16 weights).
  - At 8B that is −13.9 GB, or −12.4 GB with `skip_quant_last_layers=4`.
  - The cpu_training `stream_fp8`/`stream_fp4` + `SUROGATE_RESIDENT_LAYERS` path already quantizes frozen masters in place and frees the BF16 copy, and matmul.cpp:458-466 fails loudly on misuse. What is missing is a no-offload policy that reads the device blob in place; today every traversal does a whole-blob D2D gather.
- **Boundary / switch.** `frozen_base_storage: bf16|quantized_resident`, reloading from safetensors for export (new topic). Excluded under GRPO colocate: serve folds per-row scales with alpha=1, which is a different contract.
- **Measure.**
  - Peak: allocator high-water and `cudaMemGetInfo` after priming.
  - Lifetime: the BF16 base is held only until priming.
  - Throughput: step time at the newly feasible setting (both FP8 caches, `recompute: false`, or larger B).
  - Count of assertions on access to released weights.
- **Success.** −≥12 GB resident; no released-weight access over a full run including checkpoint and export.

### 17. LM-head logits reuse, then forward_grad
- **Evidence.**
  - The backward reruns `lm_head_logits_matmul` per chunk (fused_lm_head_loss.cpp:642, 1173), even with `recompute=false`.
  - (a) Walking chunks in reverse reuses the resident chunk bit-identically: 0.7-0.9% on 0.6B bf16/nvfp4 (8 chunks), 1.9% on 0.6B fp8 (2 chunks), 1.1% on 8B fp8.
    - Allowed under temperature when softcap is absent or fused, since the buffer holds `scale(raw)`.
    - Forbidden for unfused softcap, the KD path, and FFT (dW accumulation order).
    - `chunks==1` cannot occur for LoRA+recompute (sft_config.py:752-758).
  - (b) Call the existing `chunked_cross_entropy_backward` with `dloss=nullptr` in the forward, run the dX GEMM into U (B·T×C), then row-scale U by dloss in the backward.
    - The pass costs 6 logits units (u), not 4u.
    - Savings: 0.6B BF16 ~8.7 ms (6.4%), 0.6B NVFP4 7.7%, 0.6B fp8 4.4%, 8B BF16 3.0%, 8B NVFP4 5.5%.
    - Needs a training-forward flag, since GRPO builds its forward with `make_eval_request`.
    - U costs 16.8 MB (0.6B FP32) or 67 MB (8B).
- **Boundary / switch.** A new LM-head strategy header in `fork/native/include/surogate_fork` plus a hook in fused_lm_head_loss.cpp (new topic). `lmhead_grad: recompute|reuse_last|forward_grad`.
- **Measure.**
  - Throughput: cuBLASLt kernel count inside the backward (chunks → chunks−1 for (a), → 0 logits GEMMs for (b)); step time.
  - Peak: +U.
  - Correctness: (a) bitwise; (b) tolerance across chunks {1,2,8} × temperature × softcap × compact.
- **Success.** (a) bitwise with one fewer GEMM; (b) −≥5% on D0.6 BF16/NVFP4.

### 18. Capture-safe LM-head row compaction for SFT
- **Evidence.**
  - `lmhead_drop_ignored_rows` defaults off in SFT (sft_config.py:401). When enabled it disables graphs (1186-1191) and reads `n_valid` back to the host with a sync (fused_lm_head_loss.cpp:860-868).
  - The host knows the targets before `cudaGraphLaunch`.
  - CCE App B: removing ignored tokens cuts loss+grad from 145 to 54 ms (CCE) and from 208 to 75 ms (baseline) on an A100.
  - The LM head is ~18-25% of D0.6 FLOPs.
- **Boundary / switch.** Stage host-side `n_valid` and row indices with the per-micro-step inputs; either capture per bucket of 256 rows (padded with −100) or run the head as an eager segment between captured segments. Keep the deterministic compaction order. `lmhead_compact: off|graph_safe`.
- **Measure.**
  - Z12 first (the masked-token fraction on the target data).
  - Throughput: step time.
  - Peak: graph memory per bucket.
  - Correctness: logits/loss parity against the uncompacted path.
- **Success.** Net gain ≥ 0.6 × masked fraction × LM-head share, with graphs still captured.

### 19. GRPO cross-entropy passes
- **Evidence.**
  - Rollouts default to T=1.0, yet `temperatures` are always uploaded, so three `scale_logits_rows` passes run as `x*1.0f` and softcap is un-fused.
  - Nulling `inv_temperature` when all values are 1 is a ~5-line change, bit-identical, and removes ~6u·n_valid/BT (≈2.9 ms at 0.6B with 70% valid tokens).
  - `compute_accuracy=True` is hard-coded (embedding.py:148), so the argmax pass runs although GRPO/DPO never use the result.
  - Carrying the argmax through the per-chunk max must reproduce `argmax_correct_kernel`'s tie order.
  - uint4 loads; no clusters needed.
- **Boundary / switch.** Python glue in `surogate/_fork` (coordinate with the T6-T8 GRPO patches), an `embedding.py` flag, and fused_classifier.cu (new topic).
- **Measure.** G0.6.
  - Throughput: nsys sums of `scale_logits_rows_kernel`, softcap kernels, `argmax_correct_kernel` and the CE kernels; ncu `dram__throughput.avg.pct_of_peak` on the CE kernels; `train/duration_ms`.
  - Correctness: bitwise logprob equality for the temperature change.
- **Success.** Bitwise logprobs, −≥2% learner step.

### 20. MoE routing gradient via ⟨dA′, A⟩
- **Evidence.**
  - Under recompute, the replay runs the down GEMM, the down-LoRA forward and unpermute only so that `moe_combine_backward` can compute d_w = ⟨dY, Y⟩. `moe_out` is unused by the backward.
  - SonicMoE eq. 10: ⟨dY_t, A·W_effᵀ⟩ = ⟨dA′, A⟩. dA′ comes from the down dgrad plus fused LoRA backward applied to the unscaled gathered dY, and A = SwiGLU(H) is recomputed in the SwiGLU backward.
  - W-A: 1.07 ms/layer. It also drops 134 MB of `expert_out` from the backward live set.
- **Exclusions and extras.**
  - Not for Gemma-4: `post_ff_layernorm_2` consumes `moe_out`, and its experts use a gelu split.
  - GPT-OSS needs the ⟨dY, b_e⟩ term and the clamped SwiGLU.
  - The LoRA backward needs a per-row w (dB = hᵀ(w·dY)). Do not divide by w.
- **Boundary / switch.** A fused `moe_swiglu_backward_with_routing_grad` in fork/native/src; replay pruning through #11's consumer-driven mask. `moe_routing_grad: combine|swiglu`.
- **Measure.** M10 and W-A.
  - `SUROGATE_OP_TRACE=1` replay op list; GemmGrouped launches per layer in backward 4→3; step time.
  - Peak: −134 MB per live layer.
  - Router and LoRA gradients within tolerance (rounding order changes).
- **Success.** −≥4% step; router-gradient rel error ≤1e-2; loss parity over 500 steps.

### 21. MoE grouped-GEMM tile variant
- **Evidence.**
  - Every expert pads to the 128-token tile, wasting ~E·64 rows, and 64 KB of shared memory allows 1 CTA/SM.
  - 128×64×32 at 4 stages (48 KB) or 128×128×32 at 3 stages (48 KB) both give 2 CTAs/SM with an unchanged K order, so packing invariance should hold.
  - The gain is capped by the weight-bandwidth floor. Gemma-4 gate_up at 8k rows reads 1.01 GB, a ~0.6 ms floor against 0.77 ms measured, so at most ~15-20% at ~64 rows/expert and ≤10% at W-A.
  - DA-MoE measured 1.00x at EP=1, so select by mean rows per expert only.
- **Boundary / switch.** A second `GroupedGemm` alias plus a selector in `run_grouped` (new topic). `SUROGATE_FORK_MOE_TILE=auto|…`.
- **Measure.**
  - Microbenchmark of the recipe.cpp shapes at R=8k/16k/32k: duration, achieved occupancy, DRAM bytes.
  - The existing row-packing bitwise test.
  - M10 and GRPO MoE step time.
- **Success.** ≥8% faster at ≤128 rows/expert, no regression at ≥512, packing invariance holds.

### 22. Layer-pipelined exact CPU update
- **Evidence.**
  - Exact clipping requires the norm before any update.
  - Exposed time = T_norm + max(0, Σ(T_upd(l) − T_fwd(l))) + T_upd(embedding).
  - F8 at GA=1/B=8: forward is 35 ms/layer.
    - At 12 B/param and 42 GB/s, an update takes 56 ms/layer, leaving ~0.76 s exposed.
    - At today's 22 B/param, ~3 s stays exposed, so #9 must come first.
  - `gather_block` already keys slots on (layer, `mVersion`).
  - SlideDP hides updates under backward; its clipping handling is not described.
- **Boundary / switch.** A `layer_pipelined_cpu_step` module in fork/native/src with a version gate in `gather_block`: a host wait or `cuStreamWaitValue32`. `cpu_optimizer_schedule: serial|layer_pipelined`.
- **Measure.**
  - Per-layer host timestamps.
  - nsys forward wait per layer.
  - Step time.
  - Weights bit-identical to serial after N steps.
- **Success.** Exposed optimizer tail ≤30% of today's; bitwise weights.

### 23. Opt-in FP8 LM head under NVFP4
- **Evidence.**
  - `fp8_lm_head_enabled` requires `is_fp8_hybrid`, so under NVFP4 the head's three GEMMs stay BF16: 21% of the 0.6B FP4 step and 16% at 8B.
  - Per-tensor FP8 is ~2x BF16 on the 5090. Under NVFP4, `has_fp8_forward()` is false and no FP8 lm_head cache is primed, so this needs cache and stats plumbing, not a gate flip.
  - cuBLASLt returned NOT_SUPPORTED for m=248320 on H200 (fe40773c); probe m=151936 on SM120.
- **Risk.** Every surveyed recipe keeps the head in high precision. An RL sampler produced NaN logits with a quantized head (2610.34915, cause not isolated). Graphcore's result is negative at 8B on GB200 (2610.00053 App B), which does not transfer directly to 0.6B. Exclude GRPO/DPO by default. The analytic per-tensor dX-scale idea gains nothing until the transposed-head cache is primed for chunks>1.
- **Boundary / switch.** `lm_head_precision: bf16|fp8`, orthogonal to the recipe (new topic).
- **Measure.**
  - cuBLASLt heuristic and time at m=151936, n=512, k=1024/4096.
  - nsys head GEMM time.
  - Peak: +148/593 MiB of caches.
  - Quality: 300-step loss and grad norm; logit max-abs error.
- **Success.** Head time −≥40%, loss delta within noise.

### 24. FA2 max_seqlen = T under capture
- **Status: open.** C2's per-row synthesis landed in d3d36439 (T17), so the precondition below is met: a boundary-free step now stages `{0,T,…,B·T}`. The CPU replay test `fork/patches/tests/native/test_doc_mask_staging.py` already compares replayed micro-steps with eager under FA2 and mem_eff semantics, and is the place for the replay check this item needs.
- **Evidence.**
  - In pad mode, `effective_max_seqlen = B·T` and `num_docs` = capture cap (graph_executor.cpp:2382-2392).
  - The forward grid becomes ceil(B·T/128)·cap·Hq, mostly early-exit CTAs. `dq_accum` also pads 128·cap rows per split, and those are memset.
  - ro_gsm8k (~250-400-token documents) gives a cap of 16-32. Expected cost 1.5-3% at B=2; the per-empty-CTA cost is unmeasured.
  - Must ship with C2: otherwise a `{0,B·T}` document leaves rows T..B·T unwritten.
  - Prefer `cudaGraphExecKernelNodeSetParams` over graph bucketing.
- **Boundary / switch.** graph_executor.cpp, backend_flash_varlen.cpp and py_train.cpp (new topic, coordinate with T1). Env switch.
- **Measure.**
  - ncu `launch__grid_size` and duration of `flash_fwd_kernel`, `flash_bwd_dot_do_o_kernel` and `flash_bwd_convert_dq_kernel`: captured vs eager vs capped.
  - Allocation traffic: `dq_accum` memset bytes.
  - D0.6 step time at B=2 and B=8.
  - Replay test: a boundary-free step must match eager per-row attention.
- **Success.** −≥1% at B=2 and −≥2% at B=8; C2 test passes.

### 25. On-device FFT recipe preparation
- **Evidence.**
  - (a) Trainable weights are re-quantized 12 times per optimizer step (GA=4 × 3 passes). Refreshing once per step: 1.7B FP8 96→11 GB (−56 ms), NVFP4 77→13 GB (−43 ms). The cache costs +2 B/param (FP8) or +1.125 B/param (FP4).
  - (b) The NVFP4 wgrad materializes two BF16 transposes and repeats the amax. One read can emit the row-wise dgrad copy (standard SR, 2688) and the column-wise wgrad copy (4/6 SR, 1536) from a shared dout amax. Also accumulate dW through β instead of `dw_temp` + `vector_add_sr` (4 B/param per accumulating micro-step and a 201 MB temp).
  - (c) FFT auto-chunks the head into 256-row pieces, so dW is read-modify-written 16 times. Excess ~1-2 ms (BF16 grads) to ~4 ms (FP32). Size chunks from free memory instead, and check against the lost L2 residency of small chunks.
- **Boundary / switch.** Version-stamped cache entries bumped by the optimizer step (single owner), a fused dual-orientation quant kernel in fork/native/src, and the chunk rule in sft_config.py (topics).
- **Measure.**
  - Launch counts of `quantize_nvfp4_weight_cutlass_transpose_auto_kernel` and `quantize_with_abs_max_kernel<e4m3>` per optimizer step: from layers×4×3×GA to layers×4.
  - Time of `transpose_kernel<8,bf16>` + `reduce_abs_max` in backward.
  - Stack high-water.
  - A staleness test (cache version vs optimizer step).
  - Step time.
- **Success.** (a) step −≥3%; (b) −≥3% on Qwen3-1.7B NVFP4 FFT.

### 26. Keep attention output and LSE for long context
- **Evidence.**
  - `att` and `lse` are `always_recompute`, so the replay reruns `flash_attention`.
  - At 8-32k on 0.6B the replay is ~20% of attention time, ~12-17% of the step.
  - Cost per token per layer: 4.1 KB (0.6B) or 8.3 KB (8B). That is 3.55 GiB for 0.6B at 32k and 9.1 GiB for 8B at 32k (offload: ~0.2 s each way over PCIe 5).
  - Correction to the hotspot: chunked mode with recompute costs 5.5F vs 4.5F (+22%), not +57%.
  - Required changes: per-layer storage in the buffer plan; the replay pre-bind must resolve kept tensors; under FP8/FP4, regenerate the o_proj input quant from the kept `att`.
- **Boundary / switch.** Part of #6's `RecomputePlan`. `recompute_keep_attention: auto|never|always|offload`.
- **Measure.** 0.6B at 8k/32k and 8B at 8k.
  - Throughput: `flash_fwd_kernel` launches per layer 2→1; step time.
  - Peak: +O/LSE.
  - Lifetime: forward→backward per layer.
  - D2H/H2D overlap for the offload variant.
- **Success.** −≥10% step at 0.6B 32k within the memory budget.

### 27. QLoRA MoE expert dequant
- **Evidence.**
  - The pool LRU holds 4 buffers, so every layer is re-dequantized on forward and replay.
  - W-A NVFP4: 1.55 GB per visit, ~87 ms per micro-batch (~12%).
  - GPT-OSS prequant MXFP4: there is no transpose on the standard path (correction), so 2.0 GB per visit and ~57 ms per micro-batch.
  - The serve Marlin MoE is forward-only, tuned for small M, and covers only uint4/NVFP4_A16. FLUTE-style LUT kernels degrade at training batch sizes.
  - The W4A8 GPT-OSS route uses the 24 vendored SM120 FP8×FP4 instantiations (serve API, no LoRA epilogue). Its upper bound of −~9.7 ms/layer relies on the skipped base dgrad (C7). C7 is fixed (d4b3ab07, T25): the base dgrad runs by default, so this bound now holds only with `SUROGATE_GPT_OSS_PREQUANT_SKIP_EXPERT_DGRAD=1`.
- **Boundary / switch.** A qlora-aware grouped kernel family in fork/native that takes the storage handle; the dequant LRU stays as fallback.
- **Measure.**
  - Per-layer dequant vs GemmGrouped time (nsys); DRAM bytes (ncu).
  - Peak: 1.2-2.7 GB of resident dequant scratch removed.
  - Gate: the fused kernel must reach ≥85% of BF16 grouped TFLOPS at W-A histograms.
- **Success.** −≥50 ms per micro-batch at W-A with gradients within tolerance.

### 28. LoRA backward scale folding, then grouped LoRA
- **Evidence.**
  - Backward per slice = 5 GEMMs + 2 `vector_add` scale kernels (lora_model_utils.h:981-988, :1028), 77 LoRA kernels per layer. X is read 4 times per slice and dY twice.
  - (a) Folding the scale into the dB/G GEMM α removes 14 launches per layer (~400 per micro-step at 0.6B).
  - (b) Grouping by shared input with a saved S_cat (0.9 MiB/layer at 8B), a fused dY pass producing dB and G, and one dA_cat/dX per group. Its DRAM savings are an upper bound (27 ms at 8B, 6 ms at 0.6B) because the redundant X reads mostly hit L2. The launch reduction (77 → ~30 per layer) is the robust part.
  - Work copies can be views into one [k·r, in] buffer written in `get_block`.
- **Boundary / switch.** Extend T2 (lora_slice_dispatch.h, lora_model_utils.h). `lora.grouped`.
- **Measure.**
  - nsys count and sum of GEMMs with n or k = 16 and of `vector_add`.
  - ncu `lts__t_sector_hit_rate` and DRAM bytes on the S-recompute and dA GEMMs (this decides whether (b) is worth it).
  - Check whether the chosen algorithms use split-K.
  - Parity of dA/dB/dX.
- **Success.** (a) −14 kernels per layer with gradients within 1 BF16 ulp. Do (b) only if ≥30% of the redundant reads hit DRAM.

### 29. Producer-fused amax / delayed E5M2
- **Evidence.**
  - NVFP4: RMSNorm, fused_residual_rmsnorm and SwiGLU already accept `abs_max_ptr` but are called with nullptr. `DslRunState` does not override `get_fp4_forward_buffers`, and `allocate_fp4_forward_buffers` has no caller.
  - Wiring them is bit-identical and covers ln1, ln2 and swiglu, but not `att`. Each producer needs a memset or batch zeroing.
  - Gain: ~1-2%; 0.6B is launch-bound (1344 small kernels), 8B saves ~5-9 ms after L2.
  - The dY side needs a new hook; the stochastic `from_amax` launcher exists.
  - FP8: the `quantize_with_delayed_scale` E5M2 overload and its history slots exist; only the quantizer index needs plumbing for dout. ~5 ms at 8B.
  - Fix C9 and C5 alongside.
- **Boundary / switch.** Hooks in the recipe and run state (new topic); env switch.
- **Measure.**
  - nsys counts of `reduce_abs_max_kernel`, `cudaMemsetAsync` and `compute_fp4_alpha*`.
  - ncu `lts__t_sector_hit_rate` on `reduce_abs_max` (L2 hits?).
  - Bitwise FP4 bytes and scales vs the two-pass path.
- **Success.** Bitwise identical, −≥1% on D0.6 FP4.

### 30. NorMuon Newton-Schulz
- **Evidence.**
  - Each matrix costs 20M²N + 10M³ FLOP, at batch 1 (normuon.cu `normuon_update_2d`). Qwen3-4B: 211 TFLOP per optimizer step, 1.17 s at 180 TF.
  - `polar_express_workspace_size` over-allocates (4·max_dim² vs 2M²+2MN), wasting ~0.6-0.9 GB.
  - FP16 costs nothing extra on `mma.sync` and gains 3 mantissa bits.
  - Batching fixes 0.6B occupancy: 1024² outputs give 64 tiles on 170 SMs.
  - Gram NS + symmetric kernels: 97 TFLOP at 4B (0.54 s). QuACK's `GemmSymmetricSm120` exists, but it is CuTeDSL Python and needs an AOT build or C++ port. It also needs a restart and FP16, and the official code falls back to PE for square matrices.
- **Boundary / switch.** A new Gram Newton-Schulz source in fork/native/src behind an `ns_algorithm` enum; polar_express stays the default.
- **Measure.**
  - Per-shape ncu kernel time; optimizer step time (nsys).
  - Peak: workspace bytes.
  - Orthogonality error and spectral-norm bound against an FP32 SVD.
- **Success.** −≥50% optimizer step at 0.6B (batching); −≥40% at 4B (Gram), with error no worse than BF16 PE.

### 31. GQA backward glue
- **Evidence.** FA2 writes every dK/dV row when `cu_seqlens` covers all B·T rows, which `compute_doc_masking` always produces, so the `d_qkv` zero-fill (backend_flash_varlen.cpp:115-152) is redundant. `reduce_scatter_d{k,v}` runs one bf16 element per thread. The in-CTA GQA loop is deferred to #5, because under the split grid it makes long-document starvation G times worse.
- **Boundary / switch.** Skip the fill unless EP is active or coverage is partial; vectorize the reduces (new topic).
- **Measure.** Per-layer kernel list (nsys); ncu DRAM bytes; gradients bit-identical.
- **Success.** −≥30 µs per layer, bitwise gradients.

### 32. 8-bit attention for long context
- **Evidence.**
  - SageAttention2++ inference forward: 609 vs 209 TOPS at 32k on the 5090.
  - SageBwd (INT8, Triton, not varlen): fwd+bwd 1.17-1.48x on a 4090, Llama-1B end to end 1.10-1.15x.
  - Delta-Matching recovers FP8 attention gradients (CE 1.4162 vs 1.4178 for BF16).
  - Prefer INT8 for QKᵀ and dS (more accurate dQ than FP8 per SA3 Tables 6-7) and FP8 with FP16 accumulate only for PV.
  - LoRA is untested.
- **Boundary / switch.** A separate registry backend in fork/native, recipe flag `attention_precision: int8`, default off.
- **Measure.**
  - Per-kernel TOPS vs FA2 on packed varlen batches.
  - dQ/dK/dV cosine and rel-L2 vs BF16 on captured Qwen3 activations.
  - Short LoRA SFT loss A/B.
- **Success.** ≥1.3x attention fwd+bwd at 16k, dQ cosine ≥0.999, loss within noise.

### 33. MoE routing metadata
- **Evidence.**
  - Each layer pass runs 3 fill_zero, a memset, a counts kernel, a 1-thread offsets kernel, then encode → `cudaMallocAsync` → CUB sort → `cudaFreeAsync` → decode → scatter, and in forward a D2H poll that drains the queue. The single-CTA stats kernel also runs during replay, with no `mInReplay` guard.
  - Skipping stats in replay is trivial and fixes C10. **Done in 66647915 (T19), with the chunked KV sweep in 67ec9ef7.**
  - A stable counting sort reproduces the radix sort's (expert, t·k+j) order exactly.
  - Gain ~60-120 µs per layer pass, ~1% at BT=4096; larger for GRPO micro-batches.
- **Boundary / switch.** Fused routing kernels behind a flag; the radix path stays as the reference.
- **Measure.**
  - nsys launch count and time between the router GEMM and the first GemmGrouped.
  - `cudaStreamQuery` time.
  - Allocation traffic: per-call `cudaMallocAsync` removed.
  - Bitwise index equality on random routings and the routing-replay fixtures.
- **Success.** ≤4 launches per pass, bitwise indices.

### 34. Host-side cuBLASLt plan cache and env reads
- **Evidence.**
  - Every call creates and destroys 6 objects and queries 32 heuristics before checking the autotune cache (kernels/matmul.cpp:809-907); `try_get_cached_matmul_algo` has no callers.
  - `resolve_tensor` calls `getenv` on every invocation (compiled_ops_save.cpp:1405-1408), as do the LoRA and replay loops.
  - cuBLASLt keeps its own heuristic cache, so expect ~5-12 µs per call.
  - Graph-captured runs see no gain. Eager QLoRA/GRPO/MoE at 0.6B issue ~1.4-2.5k calls per micro-step.
  - Copy the serve `bf16_cublaslt.cpp` `PlanKey`/`plan_for` pattern; key on the alignment class; keep the fallbacks.
- **Measure.** nsys `--sample=cpu` `cuda_api_sum` on QLoRA 0.6B and G0.6; GPU-idle fraction; perf top (getenv, `parse_block_param`, `std::hash`).
- **Success.** Host CUDA API time −≥50%; worth keeping only if GPU idle time drops measurably.

### 35. Consumer-side LoRA delta fusion
- **Evidence.**
  - Elementwise consumers would re-read B per token row, which is too much L2 traffic (gate+up at 8B: ~3.2 GB per pass). Only a row-tiled kernel works: 64+ rows per CTA with B chunks in shared memory, i.e. the shape of `lora_accum_b_vec8`.
  - The realistic first step is fused gate/up accumulate + SwiGLU, saving one 192 MiB read+write per pass at 8B, while the delta-included `mlp_up` is still written for the SwiGLU backward.
  - RMSNorm consumers need two-pass designs.
  - Ceiling: 24-45 ms at 8B; ~0 at 0.6B.
  - A template flag must keep the non-LoRA instantiation byte-identical.
- **Measure.** ncu `lts__t_bytes` and DRAM bytes of the prototype vs the three kernels it replaces.
- **Success.** Proceed only if the gate_up read-modify-write is DRAM-bound for ≥15 ms per 8B micro-step.

### 36. GRPO/DPO graph capture
- **Evidence.**
  - grpo/config.py:190-194 and dpo/config.py:126-130 force eager mode independently of the compact head (that is why KL7 was refuted).
  - Split-segment graphs require `UseCudaGraphs`, and QLoRA FP4 also forces it off inside DslModel.
  - Estimated host time (25-50 ms) is below GPU time (70-90 ms) for G0.6.
  - The `TransferDone` wait protects reuse of the pinned input buffers. Double-buffer the inputs before removing it.
- **Measure.** nsys with CPU sampling on `examples/grpo/train.yaml`: GPU idle % and CUDA API time per micro-step.
- **Success.** Start only if GPU idle ≥10%.

---

## 4. Refuted or downgraded (one line each)

**Refuted hotspots and techniques**
- attn-4 (FA2 sm80 tile heuristic as a 15-25% forward win): refuted. Stock FA2 already reaches 192-209 TOPS at ≥4k on the 5090; only 1-2k documents have headroom (<1% end to end at the published config). The `is_sm8x` patch for cc 12.x remains an optional S item.
- KL7 (compact LM head disables graphs for GRPO/DPO): refuted. The GRPO/DPO configs disable graphs on their own; host-side `n_valid` only removes a ~20-50 µs sync.
- KL7 bucketed graphs and device-side SWITCH nodes for GRPO/DPO: refuted, since there is no graph to restore. The SFT variant survives as #18.
- KL2 MXFP8 LM head: refuted premise. The tree has no SM120 MX-FP8 GEMM; `quantize_mx_fp8_*` are declared in kernels.h and defined nowhere. Replaced by #23 on the existing per-tensor path.
- KL5 TMA reduce-add dW: refuted. A 311 MB dW does not fit in the 96 MB L2, so DRAM read/write is unchanged, and the needed BF16 SM120 epilogue GEMM does not exist.
- KL8 deferred RMSNorm (FlashNorm/CODA): refuted. Gain <1%; it breaks per-tensor FP8/global NVFP4 scaling and needs per-row epilogue GEMMs that do not exist on SM120.
- moe-2 transpose-on-write dequant: refuted for the standard GPT-OSS prequant path, which never transposes (only `import_external_weights` does). The W-C cost drops from 145 to ~57 ms per micro-batch.
- moe-5 rank 1-2 expert LoRA "with no new kernels": refuted. `rank_supported` accepts only 8/16/32, and other ranks fall back to per-expert `cublasGemmEx`.
- Reusing `moe_lora_grouped.cu` for dense projections: refuted. There is no column-slice stride, and the FP32 partials (~256 MiB) exceed the X reads saved (~32 MiB).
- PDL for LoRA kernels: refuted. Only ~500 eligible kernels, and they sit between cuBLASLt kernels that do not call `launch_dependents`.
- 2D 16×16 NVFP4 weight scaling for W^T: refuted. After a once-per-step refresh it saves ~1-2 ms per step, while costing 4/6 on forward weights; Quartet II and Mosaic prefer 1D plus re-quantization.
- FA3 Sm80 port / DASH semaphore backward: refuted as sketched. `enable_sm80_to_sm89` compiles empty kernels on sm_120, and determinism is off at hdim 256. Benchmark `flash_attn/cute` SM120 classes first instead.
- Spectrum-adaptive NS as a FLOP saver: refuted. Adaptive K=4 (2.791) does worse than fixed K=5 (2.761) in the paper's own Table 4; at best it is an accuracy tweak at equal K.

**Refuted citations**
- 2411.09009 "43% of baseline backward is logits recompute": that 43% is CCE's own on-chip recompute.
- 2601.22813 "quantization ~21% of an RTX 5090 NVFP4 step": it mixes two denominators; the absolute figure is ~12-13%.
- 2610.09657 cited as a stale-norm or speculation scheme: it is a direct-host-access hybrid executor. Use 2509.21271 (STV) or 2101.06840 (DPU) instead.
- PolyGraph 2610.23861 as evidence for H6's empty-CTA cost: it measures payload padding, not guarded empty tiles.
- 2205.05198's "7% selective recompute" applied to a FlashAttention stack: that 7% comes from recomputing the s×s attention core, which FA never stores.
- Liger "+42.8%" as FLCE alone: it is all Liger kernels against eager HF.

**Downgraded (kept in backlog, not in roadmap)**
- attn-1 LLM-42 verify/rollback: FA2 per-row results depend on the tile origin, so a windowed verifier is not bit-exact. It is an add-on to the page-fold kernel.
- attn-1 batch-invariant page-fold kernel (L): `merge_pages` is a global-max combine, not a left fold. Revisit only if #3's mismatch is unacceptable.
- attn-2 LazyFlash: gains come from last-turn-only loss on full-FT H200 runs; measure the skip fraction on surogate batches first.
- attn-3 chunk-recurrent schedule: corrected to −18% attention forward, at L effort; it reintroduces full-sequence checkpoint memory.
- attn-6 graph bucketing and persistent tile list: small after #24; the tile list needs #5 for order-free bits.
- cuDNN FROST SM120: Python CuTe-DSL engines, a C++ integration above M, no BF16 training benefit, and the backward has no THD.
- KL1 CCE fused GEMM with gradient filtering (XL): gated on tile-density measurements; CCE is not faster than torch.compile at Qwen V/D.
- KL2 FP16-accumulate head: hardware premise unverified, and dominated by block-scaled FP8.
- KL3 collapse-free target gradient: lost mass <6e-8 per token at the FP32 cliff; ~0 benefit.
- KL4 per-row outer NVFP4 scales and delayed/post-hoc amax: L, numerics change, mostly overlapping #29.
- KL5 vocab-slab backward: same gain as #25c at higher complexity.
- KL6 split-K plan file: published throughput rules out a 19%-occupancy head dX.
- moe-1 NVFP4 W4A4 experts (XL), the W4A8 frozen cache (MXFP4 only, since NVFP4 cannot pair with FP8 activations), and the per-expert FP8 loop stopgap (Nemotron-H only).
- moe-3 offload slot pool and graph-compatible slots: not the owner's target. QoFP8 = QoFP4 = QoNF4 = 0.5k tok/s suggests offload is not the limiter; test qlora_fp4 without offload first.
- moe-4 keep-H policy: +4.8 GB at W-A is unrealistic beside the experts.
- moe-6 BF16 pingpong grouped kernel: realistic 1.15-1.3x, measure QuACK first. Token rounding: conflicts with LoRA/GRPO routing.
- moe-7 gather-on-load: requires the new grouped kernel plus LoRA fusion first.
- moe-8 sync-free MoE capture: ~1 blocking sync per layer per micro-step (not ≥3), so ~1-2% at BT=4096. Conditional-node tile switch: DA-MoE measured 1.00x at EP=1.
- LoRA K-extension: XL and BF16-only.
- Merged-weight execution: BF16 ulp swallows ΔW early in training; experimental with SR only.
- NVFP4 fused LoRA GEMM: XL; it must match the CUTLASS base GEMM.
- LoRA-FA and frozen-A: an algorithm change.
- SparseLoRA: transfer unproven.
- Dequant ALU-bound claims: the FP8/NF4 kernels are likely bandwidth-bound; the Fast-NF4 2x does not transfer.
- blockfp8-direct QLoRA: its value is numerics more than speed.
- QLoRA graph capture: re-test the 7c45e046 FP4 hang first.
- lora_dropout fused masks: only matters when dropout >0; the default is 0.
- recipes: SM120-shaped quant kernels (measure ncu first); MXFP4 block-only scales (quality cost); post-hoc range alignment (overlaps #29); fine-grained blockwise FP8 (TorchAO blockwise was slower than tensorwise); optimizer-fused quantization (~0.5% after #25a); IO-aware MoE dataflow (XL); serve sharing of quantized bases (scale contracts differ).
- executor: selective op-level recompute (no evidence with FlashAttention, and HiLoRe SAC was slower than gradient checkpointing); FP8/NVFP4 saved activations (use COAT's 1.65x, not 2x); PCIe activation swapping (simulation-only, breaks full-step capture); per-layer partial MoE graphs (env-gated today, low single digits); eager bookkeeping (GRPO is not host-bound on average).
- optim: breadth-first GA (XL, redundant with #2); K-slab gradient pool (a mitigation, keep for GA>1); STV (likely high SFT clip-trigger rate, so measure Z11); GPU-side streamed update (conflicts with exact clipping; copy-engine count unknown); ECO master-free FP8 (experimental); BF16-over-PCIe (only with an explicit FP32 master); trainable-weight residency (small at GA=1); swap runtime H6 (the residual is not in the saved-tensor cache, measure with Z10); async checkpoint (~1%); every multi-GPU item (off the single-5090 path); quantized collectives (pretraining-only evidence); Dion3 (a different optimizer); Polar Switch (depends on #30b).

**Handoffs to other groups (not in this group's scope)**
- GRPO zero-coefficient row skipping (HiLoRe: 54.8-73.6% of padded slots contribute nothing).
- DualKV shared-prompt deduplication for policy updates.
- QUADS trainer-side NVFP4 weight QDQ matched to the serve kernels.
- FP8 prefill attention for serving.

---

## 5. Already well optimised in this group (do not redo)

- **Executor**
  - Whole-step CUDA graph including the optimizer.
  - Cap-and-pad varlen capture.
  - Zero-copy saves when recompute is off.
  - Phase-tree arena colouring.
  - DSL stack shrink before capture.
  - FP8 transposed-weight cache.
  - Single-kernel gradient zeroing.
  - Fork T1 (dispatch-PP boundary reads), T3, T4 (FP8 and matmul temporaries, including no redundant untransposed FP8 quant when W^T is cached), and T6-T8.
- **Attention**
  - In-place fused QK-norm+RoPE forward.
  - Vectorized RoPE with on-chip sin/cos.
  - FA2 strided interleaved QKV with unpadded LSE.
  - Is_local window block skipping.
  - Per-shape cuDNN graph cache.
  - Pinned cu_seqlens in chunked mode.
  - dQ split count chosen from dense geometry (batch-invariant; upstream measured +3.2% vs atomic on a 5090).
  - Capture-stable cu_seqlens slices.
- **Kernels and loss**
  - x128 fused residual+RMSNorm forward and backward.
  - QKV+qk-norm+RoPE fusion.
  - Persistent cp.async SwiGLU with fused amax in backward.
  - Softcap fused into CE (no temperature).
  - Saved LSE.
  - Deterministic masked-row compaction (13-21% on GRPO).
  - FP8/FP4 forward and W^T weight caches.
  - Bit-identical-only cuBLASLt autotune.
  - 128 MB SM120 workspace.
  - Split-attention graphs.
  - Multi-tensor global norm.
  - Sort-based deterministic embedding backward.
- **MoE**
  - One persistent CUTLASS grouped launch per projection, deterministic and packing-invariant.
  - Radix-sort index build (replaced a 5 ms per layer serial kernel).
  - Per-layer host-offset cache reused by recompute.
  - Fused grouped expert LoRA.
  - dW zero-skip for full FT (T3).
  - Immutable offload eviction.
  - GRPO routing replay.
- **LoRA/QLoRA**
  - vec8 small-rank B accumulate.
  - Scaling folded into the forward α.
  - Work copy synced once per optimizer step.
  - β-accumulated LoRA gradients.
  - Graph-captured non-QLoRA LoRA.
  - Fused MoE LoRA.
  - NVFP4 prequant forward aliased into the CUTLASS cache.
  - FP8/FP4 caches primed when they fit.
  - Opt-in fused dX (T2).
- **Recipes**
  - Single-pass FP8 delayed-scaling forward quant with device-side scales and a tiny amax-history update.
  - Quantize-once FP4 caches for frozen weights.
  - Streamed pre-quantized NVFP4 W/W^T blobs for offloaded frozen weights.
  - NVFP4 alpha fused into the CUTLASS epilogue.
  - Weight W^T quant fused with the transpose.
  - RHT kept off the default CUTLASS path.
  - Vectorized mxfp4 dequant.
- **Optimizer, offload and distributed**
  - Multi-tensor flash 8-bit AdamW for LoRA with device hyperparameters (<1 ms/step at 8B rank 16).
  - Per-tensor flash 8-bit AdamW and deterministic norm for on-device FFT.
  - Double-buffered weight prefetch.
  - FP8/NVFP4 streaming and partial residency for frozen masters.
  - Shared master store.
  - EP-sliced streaming.
  - `resident_nonblock` for LoRA.
  - Fork T5 OmpTeam thread sizing with the ordered deterministic norm.
  - DDP per-layer all-reduce overlapped with the last backward.
  - ZeRO-1/2/3 with double-buffered reduce-scatter.
  - Accumulator-arena rebind and bulk grad zeroing.