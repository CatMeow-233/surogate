# python-rl group: efficiency roadmap for one RTX 5090

> **About this copy.** This is the python-rl roadmap from the no-GPU efficiency study of October 2026, kept as written except for the changes listed here. The Chinese overview of all three groups (method, top items, zero-code experiments, defects, quick wins) is [README.md](README.md).
>
> - **Line numbers** in upstream files refer to upstream commit 282f9901, the fork's merge base. The study read the branch base 488e5f00, whose upstream files are byte-identical to 282f9901 except where the fork had already patched them; the references into those patched regions (`dsl_grad_store.cpp`, `packer.py`) were re-checked and converted to 282f9901 numbering. Line numbers in fork-owned files (`fork/experimental`, `fork/native`) refer to 488e5f00. Later commits on this branch edit many of the cited files, so search for the named symbol at HEAD.
> - **Proposed new files** are described in words, because they do not exist.
> - **Status.** Items fixed or implemented on this branch are marked **Status:** with the commit and the ledger topic ([patches/README.md](../../patches/README.md)); the table below lists them. Every item not marked is still open. No result on a GPU exists for any item, including the landed ones: their RTX 5090 checks are in [upstream-errata.md](../upstream-errata.md) (defects) and [performance.md](../performance.md) (performance changes).

**Status on this branch** (every item not listed is open):

| Item | Status | Commits | Ledger | Details |
|---|---|---|---|---|
| §0 #1 | fixed (= train-native C2) | d3d36439 | T17 | [errata](../upstream-errata.md#打包-sft-的整步-cuda-graph-在没有行内边界的微步上跨行或跨文档注意fork-已修复ledger-t17) |
| §0 #2 | guarded: an eager call with more documents than the captured cap runs exact geometry through fork-owned buffers; pad mode still stays on after a capture with documents (the graph reads its slices), and T17 turns it off at a capture without documents | d3d36439 | T18 | [errata](../upstream-errata.md#整步-graph-的-pad-模式不会关闭超过上限的-eager-调用丢掉尾部文档fork-已加保护ledger-t18) |
| §0 #3 | fixed (= R22): FIFO admission and abort on client disconnect | 6bcf5231, ae28bc19 | T28 | [errata](../upstream-errata.md#训练路径的-sharedmodelserver-在满载时拒绝-rolloutfork-已修复ledger-t28) |
| §0 #4 | open: T14 and T15 keep upstream's numerics bit for bit; R19 is not started | none | none | |
| §0 #5 | open: `fork/experimental/examples/5090-oib.yaml` still uses the tied Qwen3-4B | none | none | |
| §0 #6 | fixed in `fork/experimental`: a versioned resume contract that classifies every `TrainConfig` field, and `gradient_checkpointing` is compared when `compile_model` is on | 57f75813, 4253b4de | none (fork-owned) | [training.md](../../experimental/docs/training.md#checkpoints-and-exports) |
| §0 #7 | documented | 2737f5b8 | none | [errata](../upstream-errata.md#asyncstaleadamw-的注释声称已接入-dispatch-pp) |
| R2 step 1 | implemented, opt-in: `SUROGATE_FORK_HOST_GRAD_ACCUM=1` and `SUROGATE_FORK_GRAD_D2H_STREAM=1`; step 2 deferred | be5b9191, 54954e9b | T14 | [performance.md](../performance.md#cpu_training-主机端步) |
| R4 | implemented differently: text built and parsed once, on the step that ends the completion, with no incremental detokenisation; a request with stop strings still decodes the whole completion on each step to check them (`SUROGATE_GRPO_DETOK=final`, default); windowed stop check (`incremental`, opt-in); live-session round dispatch (`SUROGATE_GRPO_LIVE_DISPATCH=1`, opt-in) | afe88d26, 2586d3ac, e7ee206a | T11, T12 | [performance.md](../performance.md#grpo-训练路径-rollout-服务的主机开销) |
| R8 | implemented, on by default; `SUROGATE_SHARED_SLEEP_KEEP_KV=1` restores upstream's order | deec1701 | T29 | [performance.md](../performance.md#原生-grpo-sharedserver-睡眠前释放保留的-kvledger-t29) |
| R22 | implemented as the §0 #3 fix | 6bcf5231, ae28bc19 | T28 | [performance.md](../performance.md#grpo-训练路径服务的排队与中止ledger-t28) |
| R12 | open; its hazard, defect 0.2, is guarded | none | none | |
| R32 | open, now unblocked: defect 0.1 is fixed | none | none | |

**Scope.** This report covers four subsystems:
- grpo-loop: colocated GRPO, i.e. orchestrator, packer, rollout servers and trainer.
- train-data-python: the native SFT data and step path.
- experimental-backend: the `fork/experimental` PyTorch trainer.
- cross-cutting-sync-alloc: host syncs and allocations across the stack.

Items the code or literature skeptics refuted are removed. Their corrections are applied to everything that remains.

**How solid the numbers are.** No GPU was used. Every magnitude below comes from code reading, arithmetic, CPU micro-benchmarks on a loaded 4-vCPU VM, or CPU emulation. Each one stays a hypothesis until its 5090 measurement has been run.

**Evidence scripts (not kept).** The study's CPU benchmarks, simulations and emulations lived in its session scratch directory, which was deleted; this roadmap quotes their results where it uses them. They were: for grpo-loop, hf_bench, tok_bench, sched_sim, tail_sim, surplus_sim, pack_sim, stream_bench, lpt_sim, prefix_pack_sim and a real-thread emulation (thread_emul); for train-data-python, gsm8k_stats and capgrid_model; for the experimental backend, count_optim, step_model, count_optim_isfinite4 and estimates; for sync-alloc, decode_bench, gil_bench, accum_bench, accum_omp, accum_cache and ds_check. The work on this branch rebuilt the measurements it relied on as repository files: `fork/patches/bench/grpo_rollout_host.py` (R4, T11/T12), `fork/patches/bench/grpo_shared_admission.py` (R22, T28) and `fork/native/bench/cpu-optimizer` (R2, T13-T15).

**Hardware assumptions (5090):** 170 SMs; 96 MB L2; about 209 TFLOPS dense BF16 with FP32 accumulate (120-150 TFLOPS achieved); 1.79 TB/s DRAM (about 1.6 TB/s achieved); about 100 KB of shared memory per block; PCIe 5 x16 with 40-50 GB/s pinned (to be measured); a desktop host with dual-channel DDR5 at 60-75 GB/s (to be measured).

**Score.** Score = expected gain × confidence ÷ effort.
- Gain is the midpoint % of wall-clock saved on the named workload.
- Confidence: H=1.0, M-H=0.8, M=0.6, L-M=0.45, L=0.3.
- Effort: S=1, S-M=1.5, M=2, L=4, XL=8.
- A follow-up item is scored on its gain after its prerequisite has landed.

**Named workloads**

| Label | Definition |
|---|---|
| W-MoE-GRPO | ~10B-total qwen3_moe-class LoRA GRPO via `grpo-colocate`; 16 prompts × 8 rollouts; max_tokens 1536; Python SharedModelServer; `moe_rollout_parity` on |
| W-0.6B-GRPO / W-8B-GRPO | Qwen3-0.6B / Qwen3-8B LoRA colocate on the native C++ SharedServer, same batch |
| W-0.6B-SFT | Qwen3-0.6B LoRA BF16; seq 2048; B=2; GA=4; packing with doc masking; full-step graph (published 30.1k tok/s ≈ 544 ms/step) |
| W-8B-FFT-cpu | Qwen3-8B full fine-tune with `cpu_training`; GA=4; 4k tokens per micro-step |
| W-EXP-LISA / W-EXP-LoRA | `fork/experimental/examples/5090-lisa.yaml` / `5090-lora.yaml` (Qwen3-4B, seq 2048, B=1, GA=8) |
| W-EXP-OIB-8B | Experimental optimizer-in-backward on untied Qwen3-8B; seq 1024; GA=1 |

**Top 5**
1. **R1, experimental backend.** The RankScaledAdamW stochastic-rounding update issues about 164k tiny kernels per step at 8B. Fusing it gives roughly 2.7-3.7x on 8B OIB.
2. **R2, cpu_training FFT.** Host gradient accumulation runs as a scalar loop inside a host function and blocks the gradient-copy/prefetch stream. Parallelising it with OpenMP is an S change and could cut an 8B FFT step by 30-50%. **Status: step 1 landed as T14 (opt-in).**
3. **R3, experimental backend.** Turning gradient checkpointing off in the 5090 LISA/LoRA configs gives 1.25-1.35x. The corrected memory estimate shows it fits.
4. **R4, GRPO training-path server.** This server handles every MoE, GLM and non-Qwen model. Incremental detokenisation plus round dispatch based on live sessions gives about 1.2x rollout for MoE at 1k-token completions, and the gain grows with length. **Status: landed differently. T11's default mode (`SUROGATE_GRPO_DETOK=final`) does no incremental detokenisation: it builds and parses the text once, on the step that ends the completion (a request with stop strings still decodes the whole completion on each step to check them). The windowed stop-string check is `SUROGATE_GRPO_DETOK=incremental`, opt-in. Live-session dispatch is T12 (`SUROGATE_GRPO_LIVE_DISPATCH=1`), opt-in.**
5. **R8, native GRPO server.** One line, `shrink_kv()` before `sleep()`, removes 2-3 s per step of pinned KV backup and restore. **Status: landed as T29 (on by default; `SUROGATE_SHARED_SLEEP_KEEP_KV=1` restores upstream's order); the skeptic's corrected saving is about 0.8-3 s per step at C=16.**

---

## 0. Defects found during verification (fix or guard before benchmarking)

1. **Graphed packed SFT can attend across rows.** In `py_train.cpp:1601-1610`, once a graph has been captured with documents, a later step with no in-row boundaries gets `cu_seqlens = {0, B*T}`. That is one document spanning all rows, so with B≥2 row b attends to rows below it. Eager mode instead clears doc masking. Fix: use per-row `{0, T, …, B*T}`. GPU check: B=2, GA=2, capture with short documents, then feed documents longer than T, and compare step loss graphed vs `use_cuda_graphs: false`. **Status: fixed in d3d36439 (T17).**
2. **Doc-mask pad mode is never cleared.** `GraphExecutor::enable_doc_masking_pad_to_max` (graph_executor.cpp:2240-2321) stays on for the rest of the run after the first capture with documents. A later `validate()` call with more documents per micro-step than the captured cap silently drops the extras from attention, because staging copies `min(count, cap+1)`. This blocks R12. Add an assert now. **Status: guarded in d3d36439 (T18): an eager call over the cap runs exact geometry instead of truncating.**
3. **SharedModelServer drops original rollouts under load.** It answers 429 and the OpenAI SDK backs off for 0.5-8 s. After about 41-55 s of retries the original rollout is lost, which counts toward `MAX_ROLLOUT_ATTEMPTS_PER_GROUP=16` and can drop whole groups. Fresh surplus requests overtake backed-off originals. R22 fixes this. **Status: fixed in 6bcf5231 and ae28bc19 (T28).**
4. **cpu_training numerics.**
   - The BF16 master is updated with round-to-nearest (`cpu_adamw.cpp:93-103`).
   - Host accumulation rounds to BF16 on every micro-step (`dsl_grad_store.cpp:1055-1061`).
   - Choosing `adamw_8bit` still leaves FP32 m/v on the CPU path: 16 B/param, about 131 GB for 8B.

   R19 addresses these. Record convergence before and after. **Status: open.**
5. **The shipped `5090-oib.yaml` cannot start.** It uses tied Qwen3-4B, which `_check_in_backward_aliases` rejects. Switch it to an untied model (Qwen3-8B) before any OIB benchmark. **Status: open.**
6. **Experimental resume contract breaks on any new switch.** `config.py:162-165` hashes every `TrainConfig` field except a fixed ignored set, so adding any field (R1, R3, R5, R7, R9-R11, R25) breaks resume of every existing checkpoint. Version the fingerprint first, for example by hashing only non-default values plus a contract version. Switches that only affect performance go in the ignored set. **Status: fixed in 57f75813 and 4253b4de** with a versioned contract: every field is classified (run control, performance, baseline, or with a legacy value), old checkpoints keep verifying under the old rule, and a refusal names each differing key.
7. **Mitigation recorded as existing, but absent.** `AsyncStaleAdamW` has no C++ caller; only `binding.cpp` exposes it. Do not count it as an existing overlap for the CPU optimizer. **Status: documented in the errata (2737f5b8).**

---

## 1. Ranked roadmap

| # | Score | Subsystem | Hotspot | Technique | Papers (verified ids) | Expected gain on 5090 (workload) | Conf | Effort | First experiment |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 34 | experimental-backend | H1: SR optimizer launch-bound (85.8k launches/step at 4B, 164k at 8B) | Fused Triton SR update with counter-based Philox; fused sketch GEMV + rank-r Adam; fused vector-param AdamW+SR | 2502.20566, 2412.05270, 2602.23349, 2610.31859 | Step 2.7-3.7x faster (-63..-73%) on W-EXP-OIB-8B; -9..-14% on 4B full-param GA=8; ~-2% on LISA | H | M | CUDA-event timing of `RankScaledAdamW.step()` plus kernel count on synthetic Qwen3-8B shapes |
| R2 | 24 | sync-alloc | H1: host gradient accumulation blocks side_stream (also carries weight prefetch) | **Status: step 1 implemented in be5b9191 and 54954e9b (T14), opt-in.** OpenMP/AVX2 `accumulate_staged_layer`; gradient D2H on its own stream (step 2: K-slab worker thread) | 2604.05091, 2101.06840 | Step -30..-50% on W-8B-FFT-cpu (removes most of 4-18 s/step) | M | S | nsys: backward duration of micro-steps 1-3 vs micro-step 0, with OpenMP build and OMP_NUM_THREADS sweep |
| R3 | 23 | experimental-backend | H2: GC all-or-nothing | `gradient_checkpointing: false` (fits); expose `every_n_layers` | 2205.05198, 2404.02827 | 1.25-1.35x tok/s (-20..-26%) on W-EXP-LISA/LoRA | H | S | 20-step runs with GC off; record peak_cuda_bytes |
| R4 | 12 | grpo-loop + sync-alloc | grpo H1 / sync H3: O(n²) detokenisation under GIL; 1 ms gather in every tail round | **Status: implemented differently in afe88d26, 2586d3ac (T11: by default the text is built once, on the ending step, with no DecodeStream; the windowed stop check is the opt-in `incremental` mode) and e7ee206a (T12, opt-in).** Proposed: DecodeStream + suffix stop scan + parse once at end; dispatch at `min(max_batch, live)` | 2603.22774 (context only) | MoE rollout ≈1.2x at ~1k-token completions (emulated), more at 2k+ → -10..-15% cycle on W-MoE-GRPO; ~2x rollout for non-Qwen dense 1B | H | S | Golden-equivalence test, then rollout tok/s and decode_rounds/tokens A/B |
| R5 | 10.8 | experimental-backend | H6: LISA back-propagates all 36 layers | `lisa_outside_policy=freeze` + `disable_input_require_grads()` (BAdam-style truncation) | 2404.02827, 2403.17919, 2511.00056 | -22% FLOPs per microbatch with GC (≈-15..-20% step) on W-EXP-LISA; -42% FLOPs with R3 | M | S | Backward time with embedding/head trainable vs frozen, same seed |
| R6 | 9.6 | grpo-loop | H2: default concurrency 16; surplus groups refilled into the tail | `max_num_seqs: auto` from free VRAM after the trainer is built; stop surplus issue when buffer filters are off | 2610.14448, 2509.21009 (opposite trade) | -13..-20% cycle on W-0.6B-GRPO (C 16→64, analytic) | M | S | Manual max_num_seqs 16/32/64 on W-0.6B-GRPO after R8; per-round batch and rollout wall time |
| R7 | 9 | experimental-backend | H5: one document per microbatch; per-microbatch host syncs | GA-window FFD packing, reset position_ids, FA2 varlen attention interface; pinned non-blocking H2D | 2407.09105, 2412.05496, 2610.25048 | 1.3-2x tok/s on W-EXP-LoRA with 300-600-token-mean data; ~0 for long documents | M | M | Dataset token-length histogram, then packed vs unpacked tok/s |
| R8 | 8 | grpo-loop | H5: retained prefix KV backed up at sleep and restored at wake, then discarded | **Status: implemented in deec1701 (T29).** `engine_->shrink_kv()` before `engine_->sleep()` | 2610.14448 | -8..-14% cycle on W-0.6B-GRPO (C=16, grows with C); -2..-3% on W-8B-GRPO | M-H | S | Time `begin_training()` + `publish()` with SUROGATE_SLEEP_DEBUG=1, before/after |
| R9 | 6.9 | experimental-backend | H3: unfused HF elementwise ops | Liger RMSNorm (incl. q/k-norm), RoPE, SwiGLU | 2410.10989 | -8..-15% step on W-EXP-LISA/LoRA | M | S | torch.profiler split of GEMM/flash vs elementwise, Liger on/off |
| R10 | 6.6 | experimental-backend | H3: PEFT fp32 input casts per adapted module | `peft.helpers.disable_input_dtype_casting` (fp32 adapter weights kept) | 2510.00206 | 40-70 ms per microbatch (-8..-14%) on W-EXP-LoRA | M | S | Per-linear fwd/bwd timing and step time, casting on/off |
| R11 | 6 | experimental-backend | H7: synchronous full-state checkpoints | Async pinned snapshot + writer thread; trainable/touched-only contents + base fingerprint | 2406.10707, 2602.22158 | -17..-29% wall-clock on W-EXP-OIB-8B after R1 (-6..-14% before); LoRA save 4-8 s → ≤0.3 s | M | M | Time `save_checkpoint()` alone and bytes per save |
| R12 | 5.7 | train-data-python | H1: eval on 85-90%-padded rows | Pack eval windows with RowPacker, documents per micro-step ≤ cap, token-weighted eval loss | 2107.02027, 2407.09105 | -8% wall-clock at default eval_steps=100 (W-0.6B-SFT); -9.5% at 8B; ~-20% on shipped benchmark configs (eval_steps 25) | H | S-M | log_eval duration_ms, padded vs packed prototype |
| R13 | 5.4 | grpo-loop | H3: parity GEMM cost (algorithmic arm) | `moe_mismatch_policy=r3_is`: parity off, routing replay, TIS/IPO; log F(τ=2) | 2510.11370, 2605.14220, 2610.32951 | -20..-35% cycle on W-MoE-GRPO if grouped cuBLAS/CUTLASS reaches the expected rate (unmeasured); stability risk | L | S-M | 300-step A/B: mismatch_kl, share of ratio>2, reward, cycle split |
| R14 | 5.4 | grpo-loop (extra) | Training share of the colocate cycle | PODS max-variance down-sampling (train on m of n rollouts) | 2504.13818 | -8..-15% cycle on W-0.6B-GRPO with m=n/2 | L-M | S | Reward vs wall-clock over 300 steps |
| R15 | 4.5 | grpo-loop | H3: one 64x16x32 tile for every training GEMM | K-order-preserving 128x128 retile for large M; cuBLAS/CUTLASS for dgrad/wgrad only (parity kept in forward and recompute) | 2511.17826, 2609.25624, 2608.14376 | MoE training GEMMs ~2.5x if s≈3 → -15..-25% cycle on W-MoE-GRPO; bitwise on-policy kept | L-M | M | Bitwise row-slice test across M, then nsys of one packed micro-step |
| R16 | 4.5 | sync-alloc | H1 follow-up | GPU-resident gradient accumulation; one D2H on the last micro-step; GPU global norm | 2512.15306 | A further -10..-20% step on W-8B-FFT-cpu after R2 | M | M | Peak GPU memory at 8B with resident gradients (22-26 GB estimated) |
| R17 | 4.5 | grpo-loop | H6: KV capacity on 8B | Pass `kv_cache_dtype` (fp8_e4m3) through to the Qwen3-dense SharedServer | 2610.14448 | ~2x KV tokens → up to -15..-39% rollout on W-8B-GRPO when KV-bound; adds mismatch | L | S | Largest feasible max_num_seqs at 2k context, plus mismatch_kl A/B |
| R18 | 3.5 | grpo-loop | H1: per-token thread-handoff floor | Batch-level token bookkeeping in DecodeScheduler (one completion future per request) | 2603.22774 (context) | MoE from 77-83% to >90% of GPU-bound ideal after R4 → ~-10% cycle on W-MoE-GRPO | M | M | decode_rounds/tokens and nsys gaps between decode graphs after R4 |
| R19 | 3.4 | sync-alloc | H2: serialised CPU optimizer | Port the existing `flash_adamw_8bit` state codec to the CPU; stochastic rounding for the BF16 master | 2602.23349, 2010.06192, 2608.11919 | ~1.8x less CPU-update DRAM traffic → -10..-20% step on W-8B-FFT-cpu; host RAM -40%+ | L-M | M | OMP sweep timing of `step_cpu_streaming` with DRAM counters |
| R20 | 3.4 | sync-alloc | H4: per-MoE-layer D2H drain | `moe_sync_free` for parity decode (device offsets only), then whole-round graphs | 2603.07685, 2609.12978 | -6..-25% per decode round at B=16 (more at B≤4) → -5..-12% cycle on W-MoE-GRPO | M | S-M | nsys of one MoE decode round: cudaStreamQuery spins and per-layer gaps |
| R21 | 3.2 | train-data-python | H2: LM head over prompt rows | Graph-safe row compaction with host-planned indices; static max-M graph | 2501.01005, 2411.09009 | -7..-9% step on W-0.6B-SFT (34.6% ignored rows); -5..-6% on 8B nvfp4 | M | S-M | Eager A/B of lmhead_drop_ignored_rows; nsys sum of lm_head GEMM + CE |
| R22 | 3 | grpo-loop | H4: cancelled rollouts run to completion; 429 + backoff | **Status: implemented in 6bcf5231 and ae28bc19 (T28).** In-process abort; bounded FIFO admission in SharedModelServer | 2509.21009, 2606.26997 | -5..-10% cycle on W-MoE-GRPO; no lost originals or group drops | M | S-M | Per step: 429s, rollout errors, time in `begin_training()` |
| R23 | 3 | grpo-loop | H7: 1 s / 0.2 s polled handoffs; file round trip | Event-driven STABLE signal and in-memory TrainingBatch | 2606.26997 | ~0.6 s/step: -3% cycle on W-0.6B-GRPO | H | S | T6 timing: wait_for_ckpt minus train time |
| R24 | 3 | train-data-python | H3: deterministic dq_accum memsets (S=4 copies) | A/B the existing `SUROGATE_FLASH_ATTN_VARLEN_BWD_NONDETERMINISTIC=1` | — (code-derived) | ~-5% step on W-0.6B-SFT (≈30 of ~40 ms/step) | M | S | nsys memset bytes and flash_bwd time; 500-step loss curves |
| R25 | 3 | experimental-backend | H4: full-vocabulary logits plus fp32 CE | `logits_to_keep` = supervised indices (CCE optional, for memory) | 2411.09009 | -2..-4% step and -1.7 GiB backward peak on W-EXP-LISA (prompt fraction 0.35-0.5) | H | S | peak_cuda_bytes and memory snapshot around backward |
| R26 | 2.8 | grpo-loop | H6: trainer arenas stay mapped during rollouts | VMM-backed trainer run-state arenas released while rolling out | 2610.14448, 2610.05748, 2405.04437 | +28-42k KV tokens (14-20 lanes at 2k) → rollout -20..-39% on W-8B-GRPO (analytic) | L-M | L | cudaMemGetInfo after GRPOTrainer and SharedServer; largest feasible C |
| R27 | 2.25 | sync-alloc | H2: serialised CPU optimizer | Layer-wise CPU update overlapped with the last backward, plus speculation-then-validation (or stale clip) | 2403.06504, 2603.16428, 2509.21271 | -10..-20% step on W-8B-FFT-cpu | M | L | nsys gap from last backward kernel to next forward kernel |
| R28 | 1.8 | train-data-python | H4: per-document GDN pipelines force eager mode | Route GDN+packing through split-segment graphs (S); then varlen GDN (L) | 2406.06484, 2412.06464 | Unknown; est. -3..-10% (split) / -12..-25% (varlen, low confidence) on Qwen3.5-0.8B packed SFT | L | S, then L | nsys gdr_* kernel count and time, packed vs padded |
| R29 | 1.8 | train-data-python | H5: synchronous checkpoint, duplicate HF export, inline plot | Pinned snapshot + writer thread; HF export at final save only; plot off the critical path | 2406.10707, 2406.13768 | -4..-8% wall-clock on Qwen3-0.6B FFT; ~-1% on LoRA | M | M (plot/export part S) | Time `save_checkpoint`, split into D2H / serialise / writeback |
| R30 | 1.5 | sync-alloc | H5: decode-variant LRU thrash | Update `last_use` on hit (S); power-of-two B/T buckets with a shared workspace (M) | 2610.23861, 2603.07685 | Unknown; est. -1..-10% rollout on W-MoE-GRPO | L | S, then M | `get_decode_batch_stats()` evictions and compiles per step |
| R31 | 1.35 | grpo-loop | H2: FIFO admission order | Longest-predicted-first, whole-group admission at the orchestrator (native path) | 2511.14617 | -2..-8% rollout on W-8B-GRPO at C=32 | L-M | S | Per-round batch size vs time, FIFO vs LPT |
| R32 | 1.2 | train-data-python | H3: cap-and-pad geometry | **Status: open; defect 0.1 is now fixed.** `max_seqlen = T` (after defect 0.1 is fixed) | 2501.01005 | -1..-3% step on W-0.6B-SFT | M | S | Flash kernel durations, graphed vs eager on the same batch |
| R33 | 0.6 | grpo-loop | H8: padded tail bins | 1-2 shorter T buckets for partial bins | 2608.14376 | -1..-4% cycle | L-M | M | Σ T_actual / (n_mb · seq_len) per step |

Ordering notes:
- R13 and R15 are close. If bitwise on-policy behaviour is a hard requirement, R15 is the path; VeXact (2605.14220) argues for zero-mismatch execution. R13 is a cheap A/B arm.
- R8 should land before or with R6, because retained KV grows with C (13 GB at C=64 for 0.6B).
- R16 is scored after R2.
- R11 is scored after R1.

---

## Standard measurement protocol (applies to every item)

- **Throughput.** Run unprofiled: at least 50 steps after warm-up, 3 repeats. Report median step time or GRPO cycle time, tok/s and the spread. Keep profiled runs separate. The native `step_seconds` and the experimental `metrics.jsonl` `step_seconds` both exclude checkpoint and eval, so also report end-to-end wall-clock.
- **Allocation traffic.** Use `nsys profile --cuda-memory-usage=true`.
  - `nsys stats -r cuda_api_sum`: per-step counts and time of cudaMalloc/cudaFree, cudaMallocAsync/cudaFreeAsync, cudaMallocHost/cudaHostAlloc/cudaFreeHost, cuMemMap/cuMemUnmap/cuMemSetAccess, cudaStreamSynchronize and cudaDeviceSynchronize.
  - `-r cuda_gpu_mem_size_sum`: memcpy bytes per direction and memset bytes.
  - PyTorch: deltas of `torch.cuda.memory_stats()` `allocation.all.allocated` and `allocated_bytes.all.allocated`.
  - Host: pinned bytes allocated per step.
- **Lifetime.** How long a buffer stays resident across phases, read from the nsys memory-usage timeline or `torch.cuda.memory._record_memory_history` snapshots. Examples: KV kept through training; pinned backups kept until wake; trainer arenas kept through rollout; activations kept from forward to backward.
- **Peak memory.** Native: minimum `cudaMemGetInfo` free per phase. Experimental: `max_memory_allocated` / `max_memory_reserved` and `num_alloc_retries`. Host: RSS and total pinned memory.
- **Record the hardware:** driver, CUDA 13.x, clocks, PCIe generation and width (`nvidia-smi -q`), `asyncEngineCount`, pinned `bandwidthTest` or nvbandwidth in both directions, CPU model, DRAM channels, and host DRAM GB/s from AMD uProf or Intel PCM. Several items are bound by host DRAM.
- **Correctness gates.** Exact (bitwise) where the item claims it. Otherwise a tolerance plus a 200-500-step loss/reward curve compared with the baseline. Convergence differences are recorded as results, not used as blockers (AGENTS.md risk policy).
- **Isolation.** Fork-owned code lives under `fork/` (or an `[owned]` ledger path such as `surogate/_fork/`). Each edit to an upstream file is a narrow hook under a new ledger topic. Every switch defaults to current behaviour. Run `fork/dev map` and `fork/dev check` before commit.

---

## 2. Per-item detail

### R1. Fused stochastic-rounding optimizer (experimental backend)
- **Evidence.**
  - In `optim.py:32-96, 277-332`, each 1 Mi-element chunk issues 17 launches. `isfinite` is composite in torch 2.11 (4 kernels), and the vector/norm path issues 25 launches.
  - Each parameter creates a `torch.Generator` on every step.
  - Corrected launch counts: Qwen3-4B 85.8k/step (0.43-0.69 s of CPU at 5-8 µs/op); 8B 164k (0.82-1.31 s); ~10.7B MoE 210k; LISA 11.8k.
  - In OIB the hooks (`engine.py:341-347`) run on the autograd thread with GA=1. 8B forward+backward at seq 1024 is about 50 TFLOP (≈0.34 s), so the optimizer is 65-75% of the step.
  - Fused floor is about 8 B/elem: 20 ms for 4B, 41 ms for 8B.
  - SR by dithering costs almost nothing in throughput (2502.20566, AISTATS 2025). SR has to stay on: without it, bf16 updates are erased (2610.31859).
  - The APOLLO-Mini reference step (0.142 s on 7B, A100) is about 4-7x faster per parameter. That comparison crosses hardware and is only an order of magnitude.
- **Boundary and switch.**
  - A new kernels module under `fork/experimental/src/surogate/experimental/train` with:
    - a sketch GEMV in left and right variants (split-K for the 151,936-row embedding);
    - a fused rank-r Adam with the two norms;
    - `apply_sr_update`: p ← SR((1−lr·wd)·p − lr·scale·factor·g), with factor passed as a device pointer;
    - a fused AdamW+SR kernel for the 145 vector parameters.
  - Philox keyed by (param seed, step, element index), so draws do not depend on chunk size.
  - `optim.py` dispatches on `TrainConfig.optimizer_kernels: reference|triton` (default `reference`). The field enters the resume contract because the random stream changes; apply defect 0.6 first (done: a new field must now be classified in the contract, which also records its legacy value).
  - Skip `mul_(1−lr·wd)` when wd=0 on both paths.
- **Measurement.**
  - Microbench: synthetic Qwen3-4B/8B parameters and gradients; CUDA-event timing of `step()`; `torch.profiler` kernel count.
  - End to end: W-EXP-OIB-8B; nsys timeline of GPU idle gaps during backward.
  - Allocation traffic: per-step allocated bytes should lose the FP32 chunk temporaries.
  - Lifetime: no new persistent buffers.
  - Peak: `max_memory_allocated` at or below the reference.
  - Throughput: step_seconds and valid_tokens_per_second.
  - Correctness, statistical: E[SR(x)]=x within a confidence interval; P(round up) equals the discarded fraction; independence across chunk, parameter and step; nonfinite classes preserved; saturation at the BF16 max. GEMV checked against a tolerance.
- **Success.** `step()` ≤ 60 ms on 8B shapes and ≤ 5k launches per step; 8B OIB step ≥ 2.3x faster; 200-step loss within run-to-run noise.

### R2. cpu_training: parallel host accumulation off the prefetch stream
- **Status: step 1 implemented in be5b9191 and 54954e9b (ledger T14), default off.** The switches are `SUROGATE_FORK_HOST_GRAD_ACCUM=1` (vectorized add in `fork/native/src/cpu_step_kernels.cpp`, one `OmpTeam` region per tensor; BF16 results bit-identical for all 2^32 operand pairs) and `SUROGATE_FORK_GRAD_D2H_STREAM=1` (block-gradient D2H and host function on the grad store's own stream). The proposed `SUROGATE_CPU_GRAD_ACCUM_THREADS` does not exist. Step 2 is deferred. CPU measurements and the 5090 checks are in [performance.md](../performance.md#cpu_training-主机端步).
- **Evidence.**
  - `dsl_grad_store.cpp:1007-1022` queues `cudaLaunchHostFunc(accumulate_staged_layer)` on `side_stream`. The loop at 1055-1072 is scalar.
  - `kNumGradSlots=2`, so compute waits on `d2h_done` two layers later.
  - The weight prefetch (`compiled_ops.cpp:1084-1111`) uses the same `side_stream`. The per-layer period is therefore D2H + host accumulation + gather H2D.
  - 8B (193M params/layer): the scalar host loop takes 77-193 ms per layer against 25-38 ms of GPU work. That is 4-18 s of extra time per step at GA=4.
  - An OpenMP version is bitwise identical to the serial loop (tested on the VM). `cpu_vector_add_*` exists unused in `cpu_adamw.cpp:160-170`.
  - The Release build has no `-march`, so AVX2 needs intrinsics or function multiversioning.
  - DRAM floor at 6 B/elem: about 17-30 ms per layer at 40-70 GB/s.
  - The K-slab pattern comes from MegaTrain (2604.05091). Its ablation was on GH200 and does not size this case.
- **Boundary and switch.**
  - Step 1 (S): an OpenMP/AVX2 loop inside the existing callback, plus a dedicated gradient-D2H stream separate from the gather stream. Ledger topic `cpu-train-host-accum`; env `SUROGATE_CPU_GRAD_ACCUM_THREADS` (0 = old scalar path).
  - Step 2 (M): a fork-owned `HostGradAccumulator` worker with K pinned slabs, selected by `cpu_training.grad_accum: hostfn|worker`.
- **Measurement.**
  - Qwen3-4B and Qwen3-8B FFT using `examples/training/cpu-offload.yaml` with GA=4.
  - nsys: per-layer backward duration in each micro-step, `side_stream` gaps, host-function duration, H2D/D2H bytes.
  - Host: DRAM GB/s (PCM/uProf); OMP sweep 1/4/8/16.
  - Allocation traffic: pinned staging bytes (unchanged in step 1; K×layer in step 2).
  - Lifetime: slabs persist for the run.
  - Peak: host RSS and pinned bytes.
  - Throughput: unprofiled step time.
- **Success.** Micro-steps 1-3 backward ≤ 1.3x micro-step 0; 8B step ≥ 1.5x faster; CPU gradients bitwise equal to the scalar path.

### R3. Gradient checkpointing off / every_n_layers (experimental)
- **Evidence.**
  - `engine.py:180-183` passes only `use_reentrant=False`. `config.py:52` defaults to True, and both 5090 configs set it.
  - FLOPs per microbatch: LISA 55.8 → 39.7 TFLOP (1.41x); LoRA 53.4 → 37.3.
  - Corrected memory: autograd does not save the inputs of frozen linears. No-GC therefore needs about 20-21 GiB for LISA and about 24 GiB for LoRA at B=1, T=2048, which fits.
  - transformers 5.20.2 already supports `every_n_layers` (`modeling_utils.py:3109-3200`).
- **Boundary and switch.**
  - First a new example yaml with GC off.
  - Then `gradient_checkpointing: bool|"auto"` and `checkpoint_every_n_layers` forwarded in `_build_model`.
  - Both keys go in the resume-contract ignored set, since the math is unchanged. (Since 57f75813 the contract already treats `gradient_checkpointing` as a performance key, except that 4253b4de compares it when `compile_model` is on, where it changes the bits; `checkpoint_every_n_layers` would be a new field to classify.)
- **Measurement.**
  - 20-step runs on W-EXP-LISA and W-EXP-LoRA with GC on, GC off, every_n 2 and every_n 3. Include a worst-case 2048-token document.
  - Allocation traffic: `allocated_bytes` per step.
  - Lifetime: saved activations live from forward to backward and should grow.
  - Peak: `max_memory_reserved` and `num_alloc_retries`.
  - Throughput: valid_tokens_per_second.
- **Success.** ≥ 1.2x tok/s; peak ≤ 29 GiB; zero allocator retries; loss identical within bf16 noise.

### R4. Incremental detokenisation and live-count round dispatch (training-path server)
- **Status: implemented in afe88d26 and 2586d3ac (ledger T11) and e7ee206a (T12).** The text side was built differently from the sketch below: `SUROGATE_GRPO_DETOK=final` (the default) decides on each step whether it ends the completion and runs upstream's text block only on that step; `incremental` (opt-in) adds a ByteLevel window for the stop-string check; `per_token` is upstream's loop. A `DecodeStream(ids=prompt_ids)` would not match upstream, which decodes the completion alone. Live-session dispatch is `SUROGATE_GRPO_LIVE_DISPATCH=1`, off by default because it changes the rule that starts a round. For a given token stream each response equals upstream's; above concurrency 1 the token stream itself moves within upstream's own timing nondeterminism. CPU measurements and the 5090 checks are in [performance.md](../performance.md#grpo-训练路径-rollout-服务的主机开销).
- **Evidence.**
  - `shared_model.py:411-439` runs `tokenizer.decode` over every id so far, a full-text stop scan, and `protocol.parse` on every token.
  - `shared_execution` (shared_model.py:47-55) sends every model except dense Qwen3/Qwen3.5 down this path.
  - `decode_scheduler.py:313-321` waits up to 1 ms until `len(pending) ≥ max_batch`. In tail rounds fewer sessions are live, so every such round pays the full millisecond.
  - VM cost per token with a transformers fast tokenizer: 135 / 257 / 511 µs at L = 1k / 2k / 4k. DecodeStream costs 1-8 µs with identical text.
  - Real-thread emulation at L=1024 (thread_emul, not kept):
    - Dense, T(16)=2.8 ms: 27-30% of ideal today, 61% with DecodeStream.
    - MoE, T(16)=12.3 ms: 63% today, 75-78% with DecodeStream.
    - The handoff floor of 52-60% (dense) and 77-83% (MoE) is addressed by R18.
  - 2603.22774 supports only the general point that host work sits on the critical path. It does not measure detokenisation.
- **Equivalence traps.**
  - `clean_up_tokenization_spaces`: apply cleanup once to the final text.
  - Terminal-EOS `keep_terminal` rule.
  - Stop strings become eligible only after `minimum`: do one full scan at the first eligible step, tail scans after that.
  - Harmony already returns early from parse.
- **Boundary and switch.**
  - `surogate/_fork/grpo_detok.py`: one `DecodeStream` per request, a text buffer, a suffix window as long as the longest stop string, and a final parse when there is no callback.
  - A ledgered hook in `_generate`.
  - `_run` tracks live sessions (admit +1, release −1) and dispatches on `len(pending) ≥ min(max_batch, live)`.
  - Infer setting `incremental_detok: true`, with an env fallback to the old path.
- **Measurement.**
  - Golden test on recorded generations (multibyte text, special tokens, stop strings that straddle tokens, tool calls).
  - W-MoE-GRPO at max_tokens 1536 and 3584: rollout tok/s; decode_rounds vs generated tokens from `shared_weights.jsonl` (effective batch); nsys GPU-busy % and gaps between decode graphs; `py-spy dump --native` on handler threads.
  - Allocation: `tracemalloc` per token should be O(1).
  - Peak: host RSS.
  - Throughput: rollout and cycle time.
- **Success.** Identical outputs; MoE rollout tok/s ≥ 1.15x at 1536; ≥ 1.5x on a dense non-Qwen ~1B; effective batch ≥ 0.9·min(16, live).

### R5. LISA with frozen embedding/head and truncated backward (experimental)
- **Evidence.**
  - `layers.py:146-156` toggles only decoder blocks, so the tied 389M embedding/head stays trainable.
  - HF `gradient_checkpointing_enable` calls `enable_input_require_grads()` (`modeling_utils.py:3167-3175`).
  - With 2 of 36 layers sampled, E[lowest index] = 11.33, so about 31% of layers need no backward.
  - FLOPs: 55.8 → 43.5 TFLOP (-22%); 32.4 TFLOP with R3 (-42%).
  - Also removes the 0.72 GiB embedding gradient and about 5k optimizer launches.
  - BAdam backward is 42.9 ms vs LISA 84.1 ms per step (MISA Table 8, LLaMA2-7B).
  - This changes the LISA algorithm, which keeps E+H trainable by design (2403.17919).
- **Boundary and switch.**
  - `LayerSampler(outside_policy='train'|'freeze'|'rotate')`, recorded in `_signature` and in the resume contract.
  - The engine calls `model.disable_input_require_grads()` after `gradient_checkpointing_enable` when the policy is not `train`.
  - Config `lisa_outside_policy` (default `train`).
- **Measurement.**
  - W-EXP-LISA at a fixed seed: CUDA events around `loss.backward()`; step_seconds; peak with the lowest layer forced to 0 (worst case).
  - Allocation: gradient bytes per step.
  - Lifetime: activations are now saved only above the lowest active layer.
  - Quality: eval loss after N steps vs the `train` policy.
- **Success.** ≥ 15% step reduction with GC on; worst-case peak fits; quality delta recorded.

### R6. Auto-size colocate concurrency and stop surplus issue
- **Evidence.**
  - `native_colocate.py:171-186` sets concurrency to `max_num_seqs` or `min(16, batch_size)`.
  - `scheduler.py:380-396` refills up to `max_inflight=128` and cancels the surplus at the end of the batch (750-756).
  - On the native FIFO path lanes stay full until the batch ends; the loss is wasted surplus decoding, not a drain. `tail_sim` (drain) and `surplus_sim` (refill) model alternative regimes and must not be added together.
  - Analytic: 0.6B with C 16 → 64 cuts the rollout phase 11.5 → 7.8 s. The surplus costs 7.7-11.9% (0.6B) or 3-4% (8B) and discards 16k-37k tokens per step.
  - Surplus is not pure waste when `flat_group_filtering` or `online_difficulty_filtering` is on. In that case issue one replacement per dropped group.
  - The training-path server already sizes decode memory; only `max_concurrency` needs to rise there.
- **Boundary and switch.**
  - A new `surogate/_fork` module sizes `max_num_seqs` and `kv_capacity` from `cudaMemGetInfo` after `GRPOTrainer()`.
  - Ledgered hooks in `native_colocate.py` and `scheduler.py`.
  - `infer.max_num_seqs: auto` and `orch.colocate_stop_surplus: true`, gated on async_level 0 and the filters being off.
- **Measurement.**
  - W-0.6B-GRPO and W-8B-GRPO at max_tokens 1536/3584: per-round batch size vs time (engine stats / NVTX); rollout wall time; `cancelled_rollouts_count`; discarded tokens.
  - Peak: `cudaMemGetInfo` after trainer and server construction.
  - Allocation: elastic KV granule map/unmap counts.
  - Throughput: cycle time.
- **Success.** Rollout -20% on 0.6B; no OOM in 50 steps; about zero discarded tokens with the filters off.

### R7. GA-window packing (experimental)
- **Evidence.**
  - `data.py:66-67` rejects packing, so each microbatch is one document with about 7.5k eager launches.
  - Each microbatch also does three blocking pageable `.to(device)` copies (`engine.py:399` → `cudaStreamSynchronize`) plus a host read inside transformers' `fast_all` mask check. The GPU queue drains at every microbatch boundary, contrary to the "one sync per step" claim.
  - Microbatches become CPU-bound below about 250-350 tokens.
  - Minibatch packing preserves the loss and step count (2407.09105).
  - FA2 2.8.3 + torch 2.11 runs on SM120 (2610.25048).
  - In torch 2.11 `varlen_attn` has no `enable_gqa` argument but calls FA2, which handles GQA natively. This must be tested. The transformers SDPA packed path builds a 4D mask, so a custom attention interface is needed.
- **Boundary and switch.**
  - A new packing module in the experimental `train` package: FFD within the GA window, document-start labels set to −100, reset `position_ids`, `cu_seqlens`.
  - A new varlen-attention module there registers `AttentionInterface "surogate_varlen"` plus a mask interface that returns None.
  - Config `packing: none|window` (in the contract).
  - Separately (S): pinned buffers with `non_blocking=True`, and `attention_mask=None` for unpadded batches.
- **Measurement.**
  - Length histogram first.
  - W-EXP-LoRA tok/s, unpacked vs packed.
  - nsys: SM idle between microbatches and `cudaStreamSynchronize` count per microbatch.
  - Leakage test: per-document loss equals the unpacked loss within bf16 tolerance.
  - Peak: per-row `peak_cuda_bytes`, with bin capacity = max_sequence_length.
- **Success.** ≥ 1.3x on data with mean length ≤ 600 tokens; per-document loss parity; peak no higher than one 2048-token row.

### R8. Discard retained prefix KV before sleep (native SharedServer)
- **Status: implemented in deec1701 (ledger T29), on by default.** `SUROGATE_SHARED_SLEEP_KEEP_KV=1` restores upstream's order for A/B runs, as proposed. The code skeptic corrected the expected saving to about 0.8-3 s per step at C=16, growing with context and C. The `SUROGATE_SLEEP_DEBUG` timings named below do not cover the elastic KV hooks; the `engine: awake, restored ... GiB` line does. The 5090 check is in [performance.md](../performance.md#原生-grpo-sharedserver-睡眠前释放保留的-kvledger-t29).
- **Evidence.**
  - `generation_service.cpp:2189-2211` sleeps without `shrink_kv`.
  - `elastic_kv_region.cu:648-720` does `cudaMallocHost` plus a synchronous `cudaMemcpy` for each used (granule, plane) of 2 MiB. That is about 2000 such pairs for 4 GB, plus `cudaFreeHost` at wake.
  - `publish_shared_adapter` (2216-2241) wakes the engine and then immediately calls `shrink_kv()`, so every restored byte is dead. This is correct for GRPO: the adapter changes, so KV must be re-prefilled (2610.14448, App. A.1).
  - Retained KV: Qwen3-8B with 16 lanes × 1.8k tokens × 147,456 B ≈ 4.2 GB; 0.6B 3.3 GB at C=16 and 13 GB at C=64.
  - Qwen3.5 GDN hybrids retain about 8x less (e4m3 KV, full-attention layers only).
  - `sleep()` already waits for queued KV jobs, so the reorder is safe.
- **Boundary and switch.** One line in `begin_shared_training` under ledger topic `grpo-colocate-discard-kv`. Env `SUROGATE_SHARED_SLEEP_KEEP_KV=1` restores the old order for A/B runs.
- **Measurement.**
  - T6 timing of `begin_training()` and `publish()`.
  - `SUROGATE_SLEEP_DEBUG=1` "released/restored" lines.
  - Allocation traffic: `cudaMallocHost`/`cudaFreeHost` count and bytes per step.
  - Lifetime: pinned backups should disappear entirely.
  - Peak: host pinned memory.
  - D2H/H2D bytes per step and cycle time.
- **Success.** Elastic-KV pinned bytes per step ≈ 0; `begin_training` + `publish` ≤ 0.2 s; ≥ 1.5 s saved per cycle at 0.6B, C=16.

### R9. Liger fused kernels (experimental)
- **Evidence.**
  - Qwen3 RMSNorm is 8 eager ops and is applied 4 times per layer; RoPE uses `cat`; SwiGLU is unfused.
  - The DRAM upper bound is 636 KB/token/layer, ≤ 117 ms per 2048-token microbatch. With L2 reuse, 8-15% is realistic.
  - The hub-kernel decorators are inert without the `kernels` package, so Liger's monkeypatch takes effect on 5.20.2 (pin and test).
  - FLCE will not engage, because the engine calls the model without labels.
- **Boundary and switch.** `train/kernels/fusion.py: apply_fused_kernels(model_type, names)`, called before model construction. Config `fused_kernels: []` (in the contract).
- **Measurement.** Profiler CUDA-time split; kernels per microbatch (target about 7.5k → about 4k); peak; 200-step loss parity.
- **Success.** ≥ 8% step reduction; loss within noise.

### R10. Disable PEFT input dtype casting (experimental LoRA)
- **Evidence.**
  - peft 0.21.2 (the installed package, not this repository) `tuners/lora/layer.py:1100` casts x to the fp32 adapter dtype in each of the 7 adapted modules.
  - Estimated 40-70 ms per 2048-token microbatch, and +53 KB/token/layer of saved bf16 copies. That is about 3.9 GB at T=2048 with GC off; small with GC on.
  - LoRAFusion (2510.00206) quantifies unfused-LoRA overhead on H100. Its fused kernels have no licence and no 5090 config, so use only the cast removal.
  - Dropped citation: 2610.31859 does not show bf16 adapters are safe; keep the fp32 adapter weights.
- **Boundary and switch.** Engine LoRA build; config `lora_input_cast: peft_default|disabled`.
- **Measurement.** Per-linear forward/backward timing; step time; peak with GC off (expect several GiB less); loss parity.
- **Success.** ≥ 6% step reduction; ≥ 2 GiB lower peak with GC off.

### R11. Async and trainable-only checkpoints (experimental)
- **Evidence.**
  - `engine.py:48-62, 354-375`: synchronous `torch.save` with pageable D2H and `fsync`.
  - `PeftModel.state_dict()` includes the frozen 7.5 GiB base.
  - An 8B OIB save is about 16.4 GB (8-16 s). Today that is 6-14% of 100 steps; after R1 it becomes 17-29%.
  - Deselected LISA layers are bitwise unchanged (the optimizer returns early on `grad is None`), so delta checkpoints are exact.
  - LISA `on_step` deletes or moves optimizer state at the start of the next step: snapshot references before it runs.
  - In OIB, weights mutate inside backward, so expect about 0.5 s of stall per save at 8B.
- **Boundary and switch.**
  - A new checkpoint-I/O module in the experimental `train` package: `AsyncCheckpointer` (persistent pinned arena, side-stream D2H, writer thread with `fsync` + `os.replace`).
  - v2 layout {base fingerprint, trainable or touched tensors, optimizer, meta}; `CHECKPOINT_VERSION` bump.
  - `checkpoint_mode: sync|async`, `checkpoint_contents: full|trainable`, `keep_last`.
- **Measurement.**
  - Training-thread stall per save; bytes per save.
  - Allocation traffic: one pinned allocation per run.
  - Lifetime: the pinned arena lives for the run.
  - Peak: host RSS.
  - Kill during a write, then resume.
- **Success.** Stall ≤ 0.6 s per 8B OIB save (≤ 0.1 s for LoRA); LoRA checkpoint ≤ 0.5 GB; bit-exact resume.

### R12. Packed eval (native SFT)
- **Evidence.**
  - `tokenize.py:709-712` always writes eval as padded single-example rows; GSM8K rows are 85-90% padding.
  - Defaults are `eval_steps=100` and `validation_split_ratio=0.1`, and each eval is 100 batches.
  - 0.6B: about 5.3 s per eval against about 54 s of training per 100 steps.
  - The shipped benchmark configs (max_steps 100, eval_steps 25) evaluate 3 times: about 16 s against 54 s.
  - Hazard: defect 0.2. (Guarded in d3d36439, T18: an eager call with more documents than the captured cap now runs exact geometry.)
  - GDN hybrids run packed eval through the eager per-document GDN pipeline, which shrinks the gain.
  - Packed/padded equivalence rests on masking (2107.02027) and the repo's own bf16 bit-identity measurements. 2407.09105 is evidence about training loss only.
- **Boundary and switch.**
  - A new `surogate/_fork` module with a ledgered hook in `trainer.run_evaluation`.
  - `eval_packing: false|true`.
  - Documents per eval micro-step ≤ the captured cap, asserted.
  - Optional per-call `SkipIgnoredLMHeadRows` for eval (`validate()` flag; ledger topic). It is worth only tens of ms per eval once eval is packed.
- **Measurement.**
  - `log_eval` duration_ms, padded vs packed, on W-0.6B-SFT and 8B LoRA.
  - Log both the token-weighted loss and the mean of batch means.
  - Per-row loss with document counts above and below the cap.
  - CPU check of padding fraction from the eval `.bin` mask bits.
  - Allocation/lifetime: none new (eager eval reuses arenas). Peak unchanged. Report end-to-end wall-clock.
- **Success.** Eval time ≤ 20% of the padded run; per-example losses equal within bf16 tolerance; no truncation.

### R13. R3 + importance-sampling arm instead of bitwise parity (MoE GRPO)
- **Evidence.**
  - The switch (`native_colocate.py:202-203`) is small. The training-path server already returns routed experts. `routing_replay`, ratio clip e^2, the IPO mask and CISPO already exist.
  - Turning parity off also moves rollout decode GEMMs and attention off the parity kernels, so rollout speed changes too (unmeasured).
  - R3 (2510.11370, ICML 2026): mismatch KL drops from 1.5e-3 to 7.5e-4 against 6.4e-4 for dense 8B. Adding TIS on top gave no gain.
  - VeXact (2605.14220): mismatch alone collapsed MoE RL, and TIS plus sequence rejection only approximately tracks the zero-mismatch reference.
  - Rounding-location alignment (2610.32951) prevented BF16 collapse at +0.9-2.0% latency.
- **Boundary and switch.** `grpo.moe_mismatch_policy: parity|r3_is` (default `parity`) requires `routing_replay=True`. Add a per-step F(τ=2) metric next to `mismatch_kl`. Experimental only.
- **Measurement.** 300-step A/B on W-MoE-GRPO: `mismatch_kl`, share of ratio > 2, reward curve, cycle split rollout/training, nsys GEMM time.
- **Success.** Training phase ≥ 1.8x faster with `mismatch_kl` ≤ 1e-3 and no reward divergence. Otherwise record it as a negative result.

### R14. PODS down-sampling (GRPO, experimental)
- **Evidence.**
  - PODS (2504.13818, TMLR): reached peak accuracy 1.8-3.0x faster on a single L40S with LoRA on 3B. Re-normalising advantages on the subset matters.
  - In colocate the rollout phase dominates, so the gain is capped by the 20-40% training share.
- **Boundary and switch.** An orchestrator-side selector in `surogate/_fork/` with `orch.downsample: none|pods(m)`. It composes with the flat-group drop.
- **Measurement.** Cycle split; trained tokens per step; reward vs wall-clock over 300 steps against the baseline.
- **Success.** Reaches the reward target ≥ 1.2x sooner; quality recorded.

### R15. K-order-preserving retile; parity only where logprobs are produced (MoE GRPO)
- **Evidence.**
  - One config, TM=64/TN=16/TK=32 (`glm_matmul.py:91`), with grid `(n+15)/16 × (m+63)/64`, intercepts every BF16/FP32 matmul (`matmul.cpp:758-788`) in both forward and backward. Fused MoE-LoRA is disabled.
  - L2→SM traffic for a 4096³ GEMM is 10.7 GB, against about 2.1 GB at 128×128. That is 6-7x a cuBLAS-style tiling, not "100x". The slowdown s≈3 is unmeasured.
  - Forward-only parity is refuted (see section 3). The safe variant keeps parity in forward and recompute and uses cuBLAS/CUTLASS for dgrad/wgrad: 9 → 7 units (~1.3x) at s=3, about 2.5x with the retile (s≈1.3). Fused MoE-LoRA also returns outside the scope, which removes a 128 MiB/layer/pass intermediate.
  - On SM120, BF16 `tl.dot` lowers to `mma.sync` m16n8k16 with ascending k16 accumulation, so TM/TN/warps/stages should not change the result if TK=32 and there is no split-K. This must be tested: TBIK (2511.17826) never varied M/N tiles, and the only explicit "tile-inert" evidence (2609.25624) is for CUDA-core FMA. CoRun (2608.14376) shows the small-M batch-invariant cost to avoid.
  - The parity tile also leaves decode GEMMs with tiny grids (8 CTAs for k/v at B≤16). The retile does not fix that.
- **Boundary and switch.**
  - Second manifest family in `glm_matmul.py` (TM=128/TN=128/TK=32, warps 8, stages 3-4; TN=64 for grouped), with M-based dispatch and grid math in `glm_matmul_kernels.cpp`.
  - `rollout_parity_scope: all|fwd_recompute` (default `all`). The grouped-GEMM ops and `flash_attention` test parity directly, so the hook touches those too.
  - Ledger topic. Bitwise test in the fork GPU lane: `torch.equal` on row slices for M ∈ {1..16, 64, 4096}, every dtype triple including tf32x3.
- **Measurement.**
  - nsys of one packed seq-4096 micro-step on W-MoE-GRPO: glm_matmul / glm_grouped_matmul time vs cutlass/cublas; `step_grpo_native` wall time.
  - ncu: `lts__t_bytes`, `dram__bytes`, tensor-pipe utilisation.
  - Peak and workspace bytes with fused MoE-LoRA back on in backward.
  - `mismatch_kl` must stay 0.
- **Success.** Bitwise equality across M for all dtypes; training-phase GEMM time -40%; `mismatch_kl` 0.

### R16. GPU-resident gradients for cpu_training (follow-up to R2)
- **Evidence.**
  - The cpu_training constructor deliberately skips block gradients on the GPU (`dsl_grad_store.cpp:785-787`).
  - The micro-step-0 direct-D2H branch can be reused.
  - The GPU accumulates in the GEMM epilogue in FP32, so test against the old path with a tolerance, not bitwise. The GPU path is numerically better.
  - A GPU norm removes one CPU gradient pass.
  - 8B estimate: 22-26 GB, unmeasured.
  - LLMQ (2512.15306) keeps BF16 gradients on a 24 GB 4090 for 7B.
  - Zero-code variant for ≤ 4-5B: legacy `offload_master + offload_optimizer + use_zero_copy` without `cpu_training` keeps both work weights and gradients on the GPU. It is untested for FFT and cannot fit 8B (16.4 + 16.4 GB).
- **Boundary and switch.** `cpu_training.grad_residency: stream|gpu_accumulate` (ledger). Fall back to streaming if free memory after the step-0 probe is too small.
- **Measurement.** As R2, plus peak GPU (minimum `cudaMemGetInfo` free), D2H bytes per step (expect 1/GA), and the CPU norm time removed.
- **Success.** A further ≥ 10% step gain at 8B with ≥ 4k tokens per micro-step.

### R17. FP8 KV for the Qwen3-dense SharedServer
- **Evidence.**
  - `py_serve_shared.cpp` passes no KV dtype, so Auto applies.
  - Auto already gives e4m3 for Qwen3.5 GDN hybrids. Qwen3 dense stays BF16 on purpose: e4m3 costs +1.5-2.6% perplexity on 0.6B (`types.h:46-52`).
  - The gain is capacity: 8B has 7k-62k free KV tokens.
  - Per-round speedups need hundreds of active requests and custom kernels. FP8 KV through FlashInfer gave only 0.97-1.09x at 8B (2610.14448).
- **Boundary and switch.** `settings['kv_cache_dtype']` passed through (ledger); `grpo infer.kv_cache_dtype: auto|fp8_e4m3`.
- **Measurement.** Largest feasible `max_num_seqs` at 2k context; rollout time vs C; `mismatch_kl`; share of ratio > 2; reward curve.
- **Success.** ≥ 1.6x feasible C and -15% rollout on 8B, with the mismatch increase recorded.

### R18. Batch-level token bookkeeping (training-path server, after R4)
- **Evidence.**
  - `decode_scheduler.py:65-77` creates a Future, appends to a deque and notifies once per token, and results are set one at a time.
  - The emulated handoff floor is 52-60% (dense) and 77-83% (MoE) of ideal. The VM was oversubscribed, so treat these as upper bounds on the loss.
  - Per-request state has to move into the scheduler: per-request numpy RNG draws (`shared_model.py:385, 395`), min_tokens blocked sets, EOS and stop sets.
- **Boundary and switch.** Fork-owned scheduler mixin (a new `surogate/_fork` module) behind `batched_bookkeeping`; ledger hook in `decode_scheduler`.
- **Measurement.** As R4, plus host time between rounds (NVTX) and nsys gaps between consecutive decode graphs.
- **Success.** MoE rollout ≥ 0.9 of the GPU-only T_gpu(B) model at max_tokens 1536.

### R19. Compressed CPU optimizer state and stochastic rounding (cpu_training)
- **Evidence.**
  - `adamw_8bit_optimizer.cpp:672-793`: stream sync, a norm pass, then FP32 AdamW. The default BF16 master moves 24 B/param per step (28 with an FP32 master).
  - 8B: 2.8-3.3 s at 60-70 GB/s. Calibrated to SlideFormer's measured ~0.7 ns/param (9950X), it could be about 5.7 s.
  - The GPU path already uses the FlashOptim codec (`flash_adamw_8bit.cu`: int8 softsign m, uint8 sqrt v, group-32 fp16 scales). Porting it gives about 13-14 B/param (~1.8x).
  - RNE BF16 master updates cancel small updates. In 2010.06192 ResNet-50 dropped 75.70 → 67.10, and SR recovered it.
  - 8-bit state adds CPU compute (2608.11919). Without AVX2/AVX-512 the update may become compute-bound.
- **Boundary and switch.**
  - `cpu_optimizer_state: fp32|flash8` (fork-owned codec with runtime ISA dispatch).
  - `cpu_master_rounding: rne|stochastic` (S, separate).
  - A checkpoint round-trip test that preserves dtypes.
- **Measurement.** CPU update time with an OMP sweep; DRAM GB/s; host RSS; nsys gap between the last backward kernel and the next forward kernel; 0.6B FFT loss vs FP32 state.
- **Success.** CPU update ≤ 0.6x of today; host RSS -40%; loss within noise.

### R20. Sync-free MoE routing for parity decode
- **Evidence.**
  - `moe_permute.cpp:93-123` does a pinned D2H plus a `cudaStreamQuery` spin in every MoE layer. The MoE op types are capture-unsafe, so MoE layers run eagerly (`graph_executor.cpp:72-98, 1680-1684`).
  - Under parity (always on for MoE colocate) the base and LoRA grouped GEMMs already take device offsets (`moe_grouped_gemm_gate_up.cpp:290-304, 450-463`). In decode the host copy only feeds validation and compact checks.
  - Corrected cost: 0.7-2 ms per round over 24-48 layers, 6-25% of a 6-12 ms round at B=16, more at B≤4.
  - SeqMoE's 1.6-1.9x (2609.12978) is a batch-1 upper bound.
  - Pool trim and `cudaMallocAsync` do not block capture.
  - Training remains L: the backward grouped GEMMs, FP8/FP4, QLoRA and EP all read host offsets.
- **Boundary and switch.** `moe_sync_free` applies only to decode with rollout parity, EP==1 and non-compact routing. It skips the D2H and spin and removes those op types from the capture-unsafe list in that mode. Ledger topic.
- **Measurement.** nsys of one decode round: spin count and time, per-layer GPU gaps, round time, then graph replay counts. Token-identical rollouts under a fixed seed. Peak graph-exec memory.
- **Success.** Round time -10% at B=16; identical tokens.

### R21. Graph-safe LM-head row compaction (native SFT)
- **Evidence.**
  - Training runs three [rows×C]×[C×V] GEMMs per row: forward, the backward logits recompute, and dX. That is 6CV ≈ 0.933 GF/token at 0.6B, about 25% of about 3.77 GF/token.
  - At 34.6% ignored rows this wastes about 5.3 TF per step: about 40-50 ms of 544 ms, plus about 4 ms of CE traffic. 8B: 2.8% (BF16), 5-6% (nvfp4).
  - Compaction disables graphs (`sft_config.py:1186-1191`) because n_valid is read on the device mid-forward (`fused_lm_head_loss.cpp:858-869`). The host already holds the targets in pinned `gs.targets` before launch.
  - First cut: train shards are fixed B*T chunks, so the dataset-wide maximum n_valid per chunk can be computed at load time. That gives one graph with a static M.
  - Follow-up lever (unmeasured): the forward logits GEMM is redundant because backward recomputes the logits. Computing loss and LSE in backward saves about 8% of 0.6B LoRA FLOPs.
- **Boundary and switch.** `lmhead_drop_ignored_rows` plus `graph_safe_compaction: off|static_max`. Indices are staged through the existing cu_seqlens H2D pattern; rows ≥ n_valid gather a zero row with target −100. Ledger topic.
- **Measurement.** nsys sum of lm_head GEMM + CE per step; eager A/B with `use_cuda_graphs: false` ± compaction to isolate the row saving; loss equality; peak (smaller logits chunks).
- **Success.** lm_head+CE time -30% on GSM8K-like data with graphs still enabled; loss within bf16 tolerance.

### R22. Abort and FIFO admission on SharedModelServer
- **Status: implemented in 6bcf5231 and ae28bc19 (ledger T28), on by default as a defect fix.** `SUROGATE_SHARED_SERVER_QUEUE=fifo` (default) or `reject` (upstream), and `SUROGATE_SHARED_SERVER_ABORT=1` (default) or `0`. The logic is in `surogate/_fork/shared_admission.py`. Two deviations from the design below: disconnects are detected per connection, as the native server does, instead of a server-wide abort generation, which would also have aborted validation rollouts and RULER calls on the same server; and the queue holds one extra slot per lane to cover the handoff race measured on CPU. CPU emulation and the 5090 check are in [performance.md](../performance.md#grpo-训练路径服务的排队与中止ledger-t28).
- **Evidence.**
  - `_generate` never checks for cancellation, and `begin_training` waits for active == 0.
  - Corrected MoE cost of cancelled surplus: 3-8 s worst case per step, because low-lane rounds read only the routed experts.
  - `admit` raises Busy → 429 → SDK backoff with `max_retries=10`. The HTTP/1.0 handler opens a new connection and thread per retry.
  - `capacity.py` already computes `max_pending_requests` and `pending_timeout_ms`, but this server ignores them.
  - Aborted rollouts never reach the WAL.
- **Boundary and switch.**
  - `surogate/_fork/shared_admission.py`: a per-request cancelled flag or server-wide abort generation set by a colocate hook; a condition-variable FIFO bounded by `max_pending_requests`; `begin_training()` and `close()` wake waiters.
  - Ledgered hook in `shared_model.py`.
  - `shared_server_queue: fifo|reject`.
- **Measurement.** Per step: 429s, retries, rollout errors, dropped groups, `cancelled_rollouts_count`, tokens decoded after cancel, time blocked in `begin_training()`; cycle time.
- **Success.** Zero 429s and lost originals; `begin_training` wait ≤ one round after cancel; cycle -5% on W-MoE-GRPO.

### R23. Event-driven phase handoff (GRPO)
- **Evidence.**
  - `scheduler.py:419-436` polls STABLE every 1 s and `packer.py:80-88` sleeps 0.2 s, with an in-process msgspec file round trip.
  - Expected delay is about 0.6 s per step (worst case 1.2 s), plus 30-100 ms of encode/IO, more with R3 routing payloads.
  - `validate_configs` forces filesystem transport, so this needs a colocate-only bypass.
- **Boundary and switch.** `surogate/_fork/grpo_handoff.py` `PhaseSignal`, set in `SharedPolicy.broadcast`. A pluggable waiter goes through the `SharedInferencePool` injection point. An in-memory `TrainingBatch` queue keeps an async `rollouts.bin` write for the WAL. Switch `colocate_handoff: fs|memory`.
- **Measurement.** T6 `wait_for_ckpt` minus train time; `pack_duration_ms` vs `rollouts.bin` mtime; GPU idle at phase switches.
- **Success.** Handoff latency ≤ 50 ms per step.

### R24. Nondeterministic varlen backward A/B (native SFT)
- **Evidence.**
  - S = clamp(⌈170/(B·Hq)⌉, 1, 4) = 4 at 0.6B, B=2 (`attention_backend.h:56-81`).
  - Per layer-microstep, `dq_accum` plus memsets zero about 335 MB, and convert_dq reads 268 MB: ≈ 0.4 ms × 112 ≈ 40 ms per step. Cap padding is only about 50 MB of that.
  - `SUROGATE_FLASH_ATTN_VARLEN_BWD_NONDETERMINISTIC=1` (`flash_attention.cpp:712`) forces S=1. It is undocumented. Atomics contention may claw some of the gain back.
- **Boundary and switch.** Existing env. Propose documenting it in `fork/docs/performance.md` and adding a config key `attention_bwd_deterministic` (ledger).
- **Measurement.** nsys memset bytes, flash_bwd kernel time, peak (smaller `dq_accum`), step time, 500-step loss vs deterministic.
- **Success.** Step -3% or better; loss curve within run-to-run noise.

### R25. Supervised-row LM head (experimental)
- **Evidence.**
  - At T=2048 the loss path peaks at 2.90 GiB (forward) and 3.48 GiB (backward).
  - Head FLOPs are 6·T·h·V for trainable tied LISA and 4·T·h·V for frozen LoRA.
  - transformers 5.20.2 accepts a tensor `logits_to_keep`.
  - CCE (2411.09009) is time-neutral at Qwen's vocabulary/hidden ratio and helps memory only.
  - Freeing memory gives no extra FLOP saving here, because no-GC already fits (R3).
- **Boundary and switch.**
  - A new loss module in the experimental `train` package, `LossHead`: dense | dense_supervised | cce, selected by `loss_impl`.
  - Indices are computed from the CPU labels (keep j where `labels[j+1] != -100`), flattened for B>1.
- **Measurement.** `peak_cuda_bytes` and a memory snapshot around backward; step time; loss equality.
- **Success.** Backward peak -1.5 GiB at p≈0.35; identical loss.

### R26. Release trainer arenas during rollouts (GRPO, 8B)
- **Evidence.**
  - `GRPOTrainer` is built first and kept for the whole run.
  - Run-state arenas come from `TensorAllocator` with plain `cudaMalloc` (`allocator.cpp:79-80`), so they must move to VMM first.
  - Serve-side sleep already unmaps its own regions during training.
  - With recompute (the default) the gain is +28-42k tokens (14-20 lanes at 2k context), not 85k.
  - Analytic: 8B C 16 → 32 cuts rollout 83.0 → 50.8 s.
- **Boundary and switch.** A fork-owned VMM arena for activations, recompute, workspace, logits/loss and GRPO scratch, with `trainer.release_run_state()` / `restore_run_state()` called by `SharedPolicy`. `kv_capacity` becomes an elastic upper bound, and the engine must tolerate granule-map OOM. Switch `colocate_release_trainer_arenas`.
- **Measurement.** `cudaMemGetInfo` after the trainer, after release, and after the server; largest feasible C; cuMemMap/SetAccess/Unmap calls and time per switch; arena lifetime per phase; rollout time vs C.
- **Success.** +12 or more lanes at 2k context; rollout -20%; remap ≤ 0.3 s per switch.

### R27. Overlapped layer-wise CPU update with speculation (cpu_training)
- **Evidence.**
  - A layer's master is not read again until the next forward gather, so updating it in place after its last backward is safe.
  - Horizontal GA=4 can hide only 0.25-0.4 of the update.
  - Speculation-then-validation reverts and redoes on clipping (2509.21271, GH200: +45%, 0.12% rollbacks). In-place inversion is approximate, so a one-step-stale clip is the cheap comparison.
  - LoHan (2403.06504) got 1.33x at batch 64 on an NVMe server. SlideFormer (2603.16428) shows η=0.66 at batch 16 on a 9950X.
- **Boundary and switch.** Fork-owned `OverlappedCpuAdam` subscribing to per-layer "final" events. `cpu_optimizer_overlap: off|speculative|stale_clip` and `speculation_warmup_steps`.
- **Measurement.** nsys gap from last backward to next forward; rollback count; DRAM contention; loss vs baseline.
- **Success.** -10% step at 8B; < 1% rollbacks after warm-up.

### R28. GDN with packing (native SFT, Qwen3.5 hybrids)
- **Evidence.**
  - Each packed document in each GDN layer launches 6 forward kernels and about 16 backward kernels (`gated_delta_rule.cpp:354-430, 811-960`).
  - `trainer.py:318-332` forces eager mode for the whole run, so every non-GDN op is also eager.
  - Estimate: 18 GDN layers × 20 documents × 22 ≈ 7.9k launches per micro-step. The per-launch cost is assumed, hence low confidence.
  - Missed mitigation: the split-segment graph path (`py_train.cpp:1550-1560`) already treats ChunkGatedDeltaRule/MambaConv1d as graph-breaking.
  - Varlen precedent: vendored `fla_kda/state.py` (USE_G, IS_VARLEN) and the device-side `kda_metadata`. Moving to it turns per-row bit-identity into a tolerance.
- **Boundary and switch.** `gdn_packing_graphs: eager|split_segments|varlen`.
- **Measurement.** nsys gdr_* kernel count and time, launch-gap histogram, step time on Qwen3.5-0.8B/2B packed vs padded with the same tokens; per-row loss parity.
- **Success.** split_segments ≥ 5% step reduction; varlen ≥ 15% at 0.8B within tolerance.

### R29. Native async checkpoint (SFT FFT)
- **Evidence.**
  - `trainer.py:1458-1476` saves, plots and cleans inline every 50 steps.
  - `checkpoint.cpp` writes weights, Adam m/v and a duplicate HF export.
  - `safetensors.cpp:564-566` uses a value-initialised vector, a pageable `cudaMemcpy`, then a copy into a MAP_SHARED mmap. That is 3.6 GB at 0.6B FFT, about 2-3 s.
  - The plot costs 0.19-0.31 s per call; matplotlib is imported at startup.
  - The simplest correct variant: after the existing end-of-step `cudaDeviceSynchronize`, D2H into a persistent pinned arena (~0.1 s), then a writer thread with `pwrite`. `checkpoint.json` is already written last and serves as the completion marker.
- **Boundary and switch.** A new C++ async-checkpoint module (ledger) plus a new `surogate/_fork` module. `async_checkpoint`, `hf_export: every_save|final`, plot in the background or at the end. Cleanup and resume skip directories without the marker.
- **Measurement.** nsys GPU idle at step % 50 == 0; save wall time split into D2H / serialise / writeback (Dirty in `/proc/meminfo`); a single pinned allocation; host peak +3.6 GB.
- **Success.** Stall ≤ 0.2 s per save at 0.6B FFT; resume from an async checkpoint works.

### R30. Decode-variant cache fix and buckets
- **Evidence.**
  - `max_variants=8` (`graph_executor.cpp:444-575`).
  - `prepare_decode_variant` never updates `last_use`; only the execute path at :1985 does. Shapes that admission compiles keep `last_use=0`, are always the LRU victim, and get recompiled on the next admission (ping-pong).
  - Eviction costs a stream sync, graph destruction and `cudaFree`.
  - With decodes active, prefill chunks are a power of two from `prefill_chunk/4`, plus a remainder of `remaining−1`.
  - The cost is unmeasured.
- **Boundary and switch.**
  - S: touch `last_use` on hit (ledger).
  - M: `decode_shape_buckets: pow2`: B rounded to {1, 2, 4, 8, 16}; prefill split into power-of-two pieces; admission does a capacity check without compiling; one shared workspace sized for the largest bucket, charged to the VRAM budget; padded rows use a dummy page and are masked from sampling.
- **Measurement.** `get_decode_batch_stats()` (compiled shapes, evictions, replays) per step; nsys compile/capture ranges; `cudaMalloc`/`cudaFree` per step; decode workspace peak; rollout tok/s.
- **Success.** Zero evictions after warm-up; compiles only in the first step.

### R31. LPT whole-group admission (native path)
- **Evidence.**
  - The native pending queue is FIFO (`capacity.py:89-94`), so ordering at the orchestrator is enough. It needs a batch-level sampler, since today one example is sampled per call.
  - It is ineffective on the training path until R22 lands.
  - Seer's scheduling-only increment was 1.04-1.10x (2511.14617). Group-level prediction is close to the σ=1.0 simulation, i.e. 1.8-8.1%.
- **Boundary and switch.** Fork-owned per-example EMA of completion length; `orch.admission: fifo|lpt`.
- **Measurement.** Per-round batch vs time; rollout time; peak KV, since long sequences are front-loaded.
- **Success.** Rollout -3% at 8B, C=32.

### R32. Tighten the doc-mask bound to max_seqlen = T (after defect 0.1)
- **Status: open, now unblocked.** Defect 0.1 is fixed in d3d36439 (T17); this is the same item as train-native #24.
- **Evidence.**
  - Today `b = cap` and `max_seqlen = B*T` (`graph_executor.cpp:2383-2391`). `cu_seqlens` are per row, so documents are ≤ T.
  - `unpadded_lse=true`, so there is no `seqlen_q_rounded` stride hazard.
  - Halves the fwd, dot_do_o and convert_dq grids. Empty-CTA cost is assumed at 0.2-1 µs.
  - A cap-bucket cache would be M (global pad state).
- **Boundary and switch.** Ledger topic; `SUROGATE_DOCMASK_ROW_BOUND=1` until validated.
- **Measurement.** Graphed vs eager flash kernel durations on the same batch; Catch2 microbench with padded `cu_seqlens` for b ∈ {16, 32, 64, 128}.
- **Success.** Graphed flash time ≤ 1.2x of eager exact geometry.

### R33. Length buckets for partial GRPO bins
- **Evidence.**
  - Padded compute is 3.7-7.0% at batch 128 and 8.6-10.8% at batch 32 (`pack_sim`).
  - MoE colocate forces `sequence_chunks` 1.
  - Whether the run state supports a smaller T is unverified.
- **Boundary and switch.** `grpo.pack_buckets: [seq/4, seq/2, seq]`.
- **Measurement.** Σ T_actual / (n_mb · T_bucket); time of a full bin vs a half bin.
- **Success.** Training phase -3%.

---

## 3. Refuted or downgraded (one line each)

**Refuted, dropped**
- grpo H3 "parity only in the logprob forward": recompute without parity can flip near-tie top-k choices (`routing_replay` is off by default), so backward runs through different experts. Replaced by forward+recompute parity (R15).
- train-data H2 "CCE gradient-filtered LM-head backward": against a cuBLAS fused baseline CCE is time-neutral (V/D 111) or slower, and a frozen head does not cut FLOPs. Memory only.
- sync-alloc H2 "one-step-delayed update through the existing AsyncStaleAdamW": it has no C++ caller and takes FP32 `std::vector` copies. Weights stream from the same pinned master, so reads would tear. It would need a second host weight buffer (M+).
- sync-alloc H4 "persistent MoE scratch and pool threshold to unlock capture": `cudaMallocAsync` is capturable and trims are already skipped under capture. Direct saving < 1%.
- sync-alloc H6 "graph-compatible expert-offload decode runtime": shared-model GRPO requires unquantized BF16, and a ~10B MoE fits.
- sync-alloc extra "≤3B on-GPU FFT needs a FlashOptim SM120 Triton port": the CUDA `flash_adamw_8bit` codec is already the default GPU optimizer. Only a split master or SR is missing.
- Citation 2610.31859 as evidence that bf16 LoRA adapters are safe at 2-5e-4: it measured only LR-sized steps. Keep fp32 adapters or SR.
- Mitigation "AsyncStaleAdamW is wired into dispatch-PP": false.

**Downgraded or corrected**
- grpo H1: DecodeStream alone gives ~1.2x for MoE (emulated), not 2x. The host ceiling becomes the handoff floor (R18), not >200k tok/s.
- grpo H2: drain and refill simulations are alternative regimes, not additive. The 16-lane cap is only a default; 8B is KV-bound.
- grpo H2 LPT: RolloutPipe's frontier dispatch is not LPT. Seer's scheduling increment is only 1.04-1.10x. Realistic gain 1.8-8.1%.
- grpo H2 sibling-suffix speculative tail: n-gram drafting was 0.5-0.7x in NVIDIA synchronous GRPO; BubbleSpec's gains came from 48-64k-token tails. L+, parked.
- grpo H2 same-GPU tail overlap: every positive result used separate GPUs; BubbleSpec's intra-GPU run was 39% slower. Parked until interference is measured.
- grpo H3: L2 traffic is 6-7x a cuBLAS-style tiling, not 100x. The 3x GEMM slowdown is unmeasured. Parity also leaves decode GEMMs with tiny grids.
- grpo H1 native MoE serving + R3: L-XL. `shared_weights` only binds dense qwen3/qwen3_5, and it would publish about 9k adapter modules per step. Parked.
- grpo H4(a): 3-8 s per step, not 18 s. (b): the cost is admission inversion and lost originals, not about 2 s of idle lanes.
- grpo H4 page-table/page pool: M; 0.1-1 s per step, unmeasured. Measure `cuda_api_sum` before building.
- grpo H5 persistent pinned backup pool: moot after R8.
- grpo H6: the serve workspace already unmaps during training. With recompute, freed KV is +28-42k tokens, not 85k.
- grpo H6 NVFP4 shared base: XL. QeRL's gains come from Marlin W4A16 (not native FP4) on H100; venue ICLR 2026. Parked.
- grpo H8 prefix-aware packing: DualKV gave only 1.04x at P=1K, and its code is CC-BY-NC. Parked for short-prompt math RL.
- train-data H1: the claim "eval is eager, so the cap doesn't matter" is wrong; pad mode persists (defect 0.2).
- train-data H2 device-side M (CUTLASS kDeviceOnly / SWITCH nodes): the host already has the targets, and the repo has no conditional-node code. Parked.
- train-data H3 flattened FA2 scheduler (L), device-updatable kernel nodes (same-launch grid-update semantics unconfirmed; host-side `cudaGraphExecKernelNodeSetParams` is the better variant), and FlashMask port (L, no SM120 data): all parked behind R24/R32.
- train-data H4 decay-mask reset: exact in real arithmetic, but kernels that factor out exp(±g) can overflow to NaN, and short documents stack offsets. Needs an fp64 test. Prototype only.
- train-data H6 row-packing graph cache: only non-KD, dense, non-GDN runs benefit. M, not S. IF-conditional graphs are M-L. The token-budget packer changes the documented contract.
- experimental H1: launch counts were 16% low (isfinite is 4 kernels). The APOLLO comparison is 4-7x, not 5-8x.
- experimental H1 bucketed side-stream OIB update: gain unproven (DRAM contention); per-bucket gradient peak.
- experimental H1 ULP split master: a convergence ablation with no throughput gain; +8.2 GB rules out 8B OIB.
- experimental H2 selective checkpointing and activation offload: only useful beyond B=1/T=2048, since no-GC already fits. Offload evidence is simulation-only (2610.13003).
- experimental H3 regional compile: `dynamic=False` recompiles for every sequence length. It needs R7 first.
- experimental H3 FusedLoRA kernels: no licence, configs only for H100/A100/3090, and the 5090's lower FLOP/byte ratio means smaller gains.
- experimental H4: freed memory no longer buys ~10% FLOPs (no-GC fits). Direct gain is 2-3%.
- experimental H5: 1.3-2x only for data with a mean around 300 tokens. cuBLAS small-M tiling makes the "47% of SMs" figure pessimistic.
- experimental H6 depth-aware windows: -23%, not -27%; quality unknown. Parked.
- experimental H7: the OIB arithmetic mixed the tied 4B and 8B. Corrected: 8B 6-14% today, 17-29% after R1.
- Polar Express rectangular algorithm: saves 14-27% of Newton-Schulz FLOPs, not ~50%, and is bf16-fragile.
- Quartet MXFP4: a from-scratch pretraining method (eff_N 0.65); fine-tuning quality is unproven. QuTLASS sm_120 backward is a prototype.
- FlashOptim full-AdamW control arm: 5 B/param needs GA=1 and no clipping. At GA>1 it is 7 B/param, 28 GB for 4B, which does not fit.
- MISA: "equal or better than LISA" holds only at δ=3%.
- PACT NVFP4 frozen projections: 1.7-1.9x on a 5080 with 1-3B classification LoRA. Extrapolating to 4-8B is untested; reuse the native FP4 recipe instead. Parked.
- sync-alloc H1: the VM ns/element figures were bound by memory bandwidth. Extra cost is 4-18 s, not 6-18 s.
- sync-alloc H2: 24 B/param by default, and 8B GPU time is 6-7 s. The added cost is +40-55%, not 60-90%, unless the CPU update is ~5.7 s, as SlideFormer calibration suggests.
- sync-alloc vertical (layer-major) accumulation: XL; R2+R16+R27 capture most of it more cheaply. Parked.
- ZenFlow: about 2x amortised CPU traffic, not S×. Top-1% channels hold only 45-68% of the gradient norm. Parked.
- APRIL partial rollouts: GRPO +24-50% and DAPO +8.5-13.5% on 8-GPU nodes; on one GPU every resume needs a re-prefill. Parked.
- sync-alloc H4 cost was 2x too high at B=16: 6-25% per round. Event Tensor has only B200/TVM evidence. Plan-buffer replay is a synthetic prototype that excludes cuBLAS. All parked.
- sync-alloc H6: on the target model sizes the groups fit. The per-op sync guards an unexplained EP hang (commit 7cb7f6d1), so drop it only for EP==1 behind a flag. The fixed-slot pool is uncertain.
- sync-alloc H7 step pipelining: 0.1-1.5%, gap unmeasured. Parked until nsys shows the inter-graph gap.
- Literature context:
  - 2603.22774 does not measure detokenisation.
  - The MegaTrain and SuperOffload ablations are on GH200.
  - LoHan's testbed is a dual-Xeon NVMe server.
  - 2407.09105 is evidence about training, not eval.
  - 2609.23536 measures Python/TE eager under overhead-dominated settings and does not bound a C++ executor.
  - CCE's 3.4x is filter vs no filter.
  - FlexAttention backward is 0.86-1.05x of FA2.
  - TBIK never varied M/N tiles; 2609.25624 covers CUDA-core FMA only.
  - PolyGraph is a single-author prototype (v1 Aug 2026, id 2610).
  - Venues: FlashOptim ICML 2026 Spotlight; FlexAttention MLSys 2025; LISA NeurIPS 2024; MISA NeurIPS 2025; NanoFlow OSDI 2025; QeRL ICLR 2026; LOMO ACL 2024; FlexLLM NSDI 2026; RLHFuse NSDI 2025; LoHan as "Ratel" ICDE 2025.
  - CUDA IF/WHILE conditional nodes date from 12.4 per NVIDIA's blog (one skeptic recalled 12.3); SWITCH and IF-ELSE from 12.8.

---

## 4. Already well optimised in this group (do not redo)

**GRPO colocate**
- Base weights shared zero-copy between trainer and server (`borrow_weights`, `base_upload_bytes=0`).
- Policy publish stays on the GPU: about an 87 MB adapter snapshot for 8B r16 on the native path, a live adapter on the training path.
- Trainer logprob forward, GRPO dloss kernel and backward fused in `step_grpo_native`. Behaviour and KL logprobs come from the rollout engine, so there is no old-policy or reference forward. Metrics are read once per step. Device-side loss uses a pinned staging ring.
- Native serving: continuous batching, CUDA graphs, GPU sampling, prefix reuse for group siblings, a FIFO pending queue sized by `capacity.py`, client-disconnect detection, chained decode rounds with one sync per round.
- Training-path decode: per-(B,T) compiled variants, segment graph capture, GPU sampling (`decode_batch_sample`), GIL released inside native calls.
- Elastic VMM KV sleep/wake keeps captured graphs valid. Serve sleep unmaps every registered region, including workspace, during training.
- Fork T6/T7: immediate packing, NumPy boundary detection, in-process micro-batch handoff, cycle-timing metrics.

**Native SFT step and data path**
- Full-step CUDA graph with cap-and-pad `cu_seqlens`, so no recapture per packing. The power-of-two cap gives 0-1 recaptures per run on GSM8K.
- Native multi-threaded tokenizer with the GIL released, cached by config hash. The train split is packed as a stream, so training padding is about 0.
- DSL IR build: 0.25 s import plus 10-19 ms compile. Per-step Python monitors: 350 µs p50. RowPacker host cost: 0.17-2.1 ms per step.
- The inter-step bubble is about 1 ms (< 0.3%), so double-buffering the data loader is not worth doing.
- No steady-state per-step allocation: DeviceMemoryStack plus a 256 MiB replay-persist arena, a recycling saved-tensor cache, per-shape matmul autotune cache, LoRA refresh only on generation change, and MoE CUTLASS grouped-GEMM arguments through a 256-slot pinned ring with kDeviceOnly scheduling.
- CPU AdamW and norm are OpenMP-parallel (T5). Micro-step 0 gradients go D2H straight into the CPU gradients. QLoRA auto-tune maximises resident groups after step 0, and frozen QLoRA weights skip re-dequantisation.

**Experimental backend**
- bf16 logits and their FP32 cast are freed before backward (0.1.0a3).
- Per-expert factors stay device scalars, with no `.item()` per parameter. FP32 temporaries are chunk-bounded. The rank-1 sketch state is tiny (12,288 floats for a 4096² matrix).
- OIB frees each gradient right after its update. LISA has drop/CPU state policies.
- With B=1 and no padding, SDPA takes flash with `is_causal` and `enable_gqa` in eager mode. Note that `fast_all` still costs one host sync per forward (see R7).
- MoE experts use `grouped_mm` on torch ≥ 2.9. Data rows are indexed by offset.
- Correction: the step is not "one sync per step", because each microbatch does three blocking H2D copies (R7).