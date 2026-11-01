# Serve group: ranked optimisation roadmap for one RTX 5090 (SM120)

> **About this copy.** This is the serve roadmap from the no-GPU efficiency study of October 2026, kept as written except for the changes listed here. The Chinese overview of all three groups (method, top items, zero-code experiments, defects, quick wins) is [README.md](README.md).
>
> - **Line numbers** in upstream files refer to upstream commit 282f9901, the fork's merge base. The study read the branch base 488e5f00, whose upstream files are byte-identical to 282f9901 except where the fork had already patched them; no reference in this roadmap falls in such a region (the serve sources were unpatched at 488e5f00). Line numbers in fork-owned files (`fork/experimental`, `fork/native`) refer to 488e5f00. Later commits on this branch edit many of the cited files, so search for the named symbol at HEAD.
> - **Proposed new files** are described in words, because they do not exist.
> - **Status.** Items fixed or implemented on this branch are marked **Status:** with the commit and the ledger topic ([patches/README.md](../../patches/README.md)); the table below lists them. Every item not marked is still open. No result on a GPU exists for any item, including the landed ones: their RTX 5090 checks are in [upstream-errata.md](../upstream-errata.md) (defects) and [performance.md](../performance.md) (performance changes).

**Status on this branch** (every item not listed is open):

| Item | Status | Commits | Ledger | Details |
|---|---|---|---|---|
| F0 | partly implemented: `_surogate_fork_attach_serve()` in `fork/native/cmake/fork_native.cmake` attaches the sources in `fork/native/serve` and the fork include root to `sinfer_ops` (and the GQA test targets); `sinfer_nvfp4_cutlass`, `sinfer_fp8_block_sm120` and `sinfer_text` are not attached yet | 7a45b4f7 | T16 | [performance.md](../performance.md#服务端-gqa-形状注册) |
| R1 | implemented, opt-in: `SUROGATE_SERVE_GQA_FORK_SHAPES=1` | 7a45b4f7, 2e5f766b | T16 | [performance.md](../performance.md#服务端-gqa-形状注册) |

## 0. Scope, assumptions and scoring

- **Inputs.** This report combines four subsystems: serve-attention-kv, serve-gemm-moe, serve-scheduling-spec and linear-attn-ssm. For each one I had the hotspot list, the research notes, a code-skeptic verdict and a literature-skeptic verdict.
  - Hotspots, techniques and citations that a skeptic refuted are dropped.
  - Every number below already includes the skeptics' corrections.
- **No GPU was used.** Every gain is modeled from code, counts and arithmetic unless it is marked as measured. Each item names the 5090 measurement that would confirm it.
- **5090 constants used:**
  - 170 SMs, 1.79 TB/s, 96 MB L2, about 99 KB of shared memory per block.
  - Dense rates: BF16 with FP32 accumulate 209.5 TFLOPS; FP8 419 TFLOPS with FP32 accumulate and 838 with FP16 accumulate; INT8 838 TOPS; FP4 with FP32 accumulate 1676 TFLOPS; FP64 about 1.64 TFLOPS (1/64 rate).
  - The INT8 and FP16-accumulate rates are not verified on a real card, and FA2 has been measured at 214 TFLOPS, so clocks run above the spec boost. R5 settles both.
- **Score** = mid gain (% of the named workload's metric) × confidence × relevance / effort.
  - Confidence: H 0.9, M 0.6, L 0.3.
  - Relevance: 1.0 for the owner's core targets (Qwen3/Qwen3.5 0.5-8B training, GRPO rollouts, ~10B MoE); 0.8 for adjacent serving; 0.5 for secondary serving (Llama/Gemma defaults, Qwen3.5 NVFP4, structured output); 0.3 for long-context Gemma or latency-only items.
  - Effort: S 1, S-M 1.5, M 2, M-L 3, L 4.
  - "gate" marks an enabler whose rank comes from the decisions it unblocks.
- **Memory axes required by AGENTS.md.** Each item reports these separately:
  - **alloc**: allocation calls or bytes per round or step.
  - **life**: how long buffers stay live.
  - **peak**: the high-water mark.
  - **tput**: end-to-end tok/s or step time.
  - DRAM and L2 bytes are kernel traffic metrics. They are a different thing from allocation traffic.
- **Measurement hygiene for every A/B:**
  - Same build with only the switch flipped.
  - Lock SM clocks or record them; BENCHMARKS notes a ±8% spread between cards.
  - Use CUPTI or nsys GPU time rather than host timers (2607.16831 shows how a harness can flip a result).
  - At least 3 repeats, stated cold or warm cache.
- **Module-boundary conventions** (AGENTS.md and fork/native/README.md):
  - **F0, serve-side fork hook (one-time prerequisite).** Today `fork/native/cmake/fork_native.cmake` attaches only to `surogate-common` and the test targets. Add a deferred `_surogate_fork_attach_serve()` that runs through `cmake_language(DEFER)` and is guarded by `if(TARGET ...)`. It adds `fork/native/serve/*.cu` and the PRIVATE include `fork/native/include` to `sinfer_ops`, `sinfer_nvfp4_cutlass`, `sinfer_fp8_block_sm120` and `sinfer_text`. Because `fork_native.cmake` is fork-owned, this needs no edit to `csrc/CMakeLists.txt`. **Status: partly implemented in 7a45b4f7 (T16): the hook exists and attaches `sinfer_ops` (and the GQA test targets); the item that first needs `sinfer_nvfp4_cutlass`, `sinfer_fp8_block_sm120` or `sinfer_text` adds it.**
  - **Training C++** goes to `fork/native/src`, which is compiled into `surogate-common`.
  - **Python and Triton** go to `surogate/_fork/`, which is an owned prefix.
  - Each upstream call-site edit is its own ledger `[[topic]]` with a `switch` field. Run `fork/dev map` and `fork/dev check`.
  - Switch names marked "(new)" are proposals.

## 1. Ranked roadmap

| # | Subsystem | Hotspot | Technique | Papers (verified ids) | Expected 5090 gain (named workload) | Conf | Effort | Score | First experiment |
|---|---|---|---|---|---|---|---|---|---|
| R1 | attention-kv | S1: unregistered GQA 32q/8kv at head dim 128 (Qwen3-4B/8B, Llama-3-8B) runs on the CUDA-core generic kernel | **Status: implemented in 7a45b4f7 and 2e5f766b (T16), opt-in with `SUROGATE_SERVE_GQA_FORK_SHAPES=1`.** Register Gqa128_32q8, Gqa128_40q8 and MHA Gqa128_32q32; keep int8 KV on the generic route | 2501.01005 (App. A), 2609.33135 (§5.3, design only) | Qwen3-8B GRPO rollout decode, 16 lanes × 4K, BF16 KV: attention 7-14 → ~5.5-6 ms/step, decode 1.1-1.6×. A 40-token turn on a 30K history: 30-180 → 3-5 ms | H | S | 22.5 | `sinfer_causal_softmax_attention_bench --entry append --geometry d128-h32-kv8 --kv-dtype bf16 --tokens 1,40 --batch 1,16,32 --context 4096,32768 --execution graph --cache cold`, before and after a scratch registration |
| R2 | gemm-moe | H1: generic NVFP4 at T=1 through linear_add, attn_input and gdn_input takes the 256-row W4A4 tile | Wire the existing generic-K GEMV with residual-add and segment epilogues | 2509.23202 (App. G), 2609.04098 | Qwen3.5-9B NVFP4, 1 user, MTP off: linears ~6.3 → ~2.5 ms/token, ~1.5-2× tok/s (modeled) | M | S | 18 | `sinfer_nvfp4_route_bench`, T=1, GEMV vs tile 1 on 4096², 12288×4096, 4096×12288 |
| R3 | linear-attn | S1: GDN decode stores state in BF16 every step | Accuracy audit of FP32 vs BF16 state; gate for R26 | 2609.38166, 2609.38169, 2610.13778, 2609.04098 | Qwen3.5-4B/9B serving and GRPO rollouts. The paper measures −15.8 AIME on Qwen3.5-9B; switching to FP32 would give back #57's +10-13% | M | S | gate | Replay 64K tokens with FP32 and BF16 state, compare per-layer state MSE, then a long-generation eval |
| R4 | gemm-moe | H1: generic NVFP4 at 2-63 tokens, rows ≥ 4096, takes tile 1 | 5090 size rule built from existing tiles (swapped 4/5, cuBLASLt); no new kernels | 2311.01282, 2408.11743, RTX Blackwell whitepaper | Qwen3.5-9B / Gemma 3 12B NVFP4 at 2-16 users and MTP verify: +20-40% decode (GB10 analog measured +32%) | M | S | 9 | Engine A/B `SUROGATE_SERVE_NVFP4_CUTLASS=off\|5\|rows` at 1/4/8/16 users |
| R5 | cross-cutting | Ceiling arithmetic (209.5/419/838, INT8, FP22, FP64) is unverified on SM120 | MMA, accumulator and FP64 microbenchmarks, clock logging, and FlashInfer, FlashQLA and SA2++ reference rows | 2505.11594 (Tab. 17-18), 2505.21136, 2411.10958 (§3.4), 2507.10789 | No direct speedup. Decides R6, R7, R15, R24 and the S2 FP16-accumulate option | H | S | gate | Per-instruction rate table with locked and unlocked clocks |
| R6 | scheduling | H2: FP64 exp/div per vocabulary token (nucleus route, wide speculative accept) at 1/64 rate | FP32 compensated normalisers, parallel boundary search, column-parallel verify; phase 2 draws the residual only at the first rejection | 2607.20475 (fused-verify structure only) | Llama-3-8B FP8 with defaults (T 0.6, top_p 0.9): −0.67-0.8 ms/round, ≈ +15-18% tok/s. Llama-3.2-1B/3B: +30-100% | H | S-M | 7.5 | ncu `sm__inst_executed_pipe_fp64.sum` and duration of `sampling_wide_kernel` |
| R7 | gemm-moe | H8: block-FP8 at 5-16 tokens runs a 32-CTA narrow tile | Enable `mma_kernel` on SM120 through a per-arch token table | 2604.26039 (its rule does not apply), 2301.03598 | Qwen3-8B-FP8 serving at 5-16 users and verify rounds: o/down 1.5-1.8× faster, round +10-20% | M | S | 5.8 | `SUROGATE_SERVE_FP8_BLOCK_MMA=1` A/B |
| R8 | linear-attn | T1: per-document GDN pipeline, and CUDA graphs off for packed SFT | Varlen chunked GDN derived from the in-repo kernels, device metadata as in KDA, graphs back on for SFT | 2406.06484, 2412.06464, 2609.12379 | Qwen3.5-0.8B packed LoRA SFT example: +7-13% tok/s. GRPO packed: +2-5% | H | M | 4.5 | nsys of 20 steps: gdr_* launches per step and GPU idle gaps |
| R9 | attention-kv | S3: 2,048-key decode partitions past 16K on SM120 | Hopper's fine partition table on SM120, plus a 2-slot ring for head dim ≤ 256 | 2405.10480 (indirect), 2511.22333 | Gemma 4 12B, BF16 KV, 1 user at 32K: global layers ~185 → ~40 µs, ~1.25-1.5× tok/s. Qwen3 with 8 KV heads: ≈0 | M | S | 4.5 | Bench `--entry cached --geometry d512-h16-kv2 --kv-dtype bf16,fp8 --tokens 1 --batch 1 --context 16384,32768,65536` with both tables |
| R10 | scheduling | H6: serial grammar masks and pageable copies on the worker thread | Thread-pool mask fill plus one pinned H2D; phase 2 splits the forward graph from the sampler graph | 2411.15100, 2601.04426 | Qwen3-8B JSON-schema and tool-call serving at 16-64 users: hide 0.4-16 ms of host time per round (+5-30%) | M | S | 4.5 | NVTX range around `update_constraint`; constrained vs unconstrained TPOT |
| R11 | attention-kv + scheduling | S5: the TTFT 0.49× gap is the round budget | Budget sweep, then a static backlog `RoundBudgetPolicy` | 2403.02310, 2609.07883, 2609.38386, 2606.09061 | 5090 closed loop, 32 users × 2048 prompt / 16 output: TTFT p50 toward 0.8-1.0× vLLM at a proportional TPOT cost; throughput unchanged | H | S-M | 3.6 | Sweep `--max-num-batched-tokens 2048/3072/4096`, reporting TTFT and TPOT together |
| R12 | scheduling | H1: a GRPO group prefills its shared prompt G times | In-flight leader/follower hold plus a configurable shared-prefix page budget | 2610.18143, 2312.07104 (code) | Qwen3-8B GRPO, Q=32, G=8: at P=1000 the step loses 11-22% (prefill); at P=300, 3-6%. Prompt KV divided by G | M | M | 3.0 | Scratch build without the capacity/8 cap, plus `SUROGATE_SERVE_SHARED_PREFIX_MIN_TOKENS=64` and `SUROGATE_SERVE_SHARED_PREFIX_SLOTS=64` |
| R13 | linear-attn | T3: transpose sandwich around the conv, and a naive packed conv backward | Vendored Dao channel-last conv forward and backward with seq_idx, for packed and padded rows | n/a (code: causal-conv1d, FLA causal_conv1d) | Qwen3.5-0.8B packed SFT: ~3.1 → ~1 GB per GDN layer-step, +4-8% tok/s | M | S-M | 2.4 | ncu DRAM bytes per GDN layer; nsys time of transpose and `packed_convolution` |
| R14 | gemm-moe | H2: W8 shared expert runs on the BF16 family kernel, serialized | Keep the shared expert in block-FP8 or NVFP4 and route it through the `native_shared` precompute | 2408.12757, 2605.21603, 2609.04244, 2603.07685 | ~10B MoE NVFP4 prefill (35B-A3B geometry as proxy): shared expert ≥123 → ~30-60 µs per layer per 4K tokens, prefill +3-8% | M | S | 2.4 | nsys per-layer time of shared vs routed kernels at 2048/4096 tokens |
| R15 | gemm-moe | H6: W8 decode uses the medium-T kernel on SM120 | Measure first: pipelined W8, SIMT SwiGLU at T=1, borrowed SIMT r8_c4 | 2610.34907, 2408.11743, 2507.10789 | Owner's fine-tuned Qwen3-8B W8 at 1-8 users: +5-15% decode if the H100 result transfers. Only k/v/down/head (~36% of bytes) are affected | L | S | 2.4 | `sinfer_w8_pipelined_test` plus `SUROGATE_SERVE_W8_PIPELINED=1` A/B |
| R16 | gemm-moe | H1: 2-63 tokens beyond what the existing tiles cover | Swapped NVFP4 tiles of 128×{16,32,64} with fixed split-K (not Stream-K) | 2301.03598, 2311.01282, 2408.11743 | Same workloads as R4: o/down reach ~1.2-1.5× the streaming floor, +10-20% on top of R4 | M | M | 2.25 | Route bench T=2..63 after R4 |
| R17 | scheduling | H4: full-vocabulary sort plus 7-8 merges for top_k > 20 or top_p only | Two-stage tile top-128 with a mass certificate, sort kept as fallback | 2607.20475, 2603.15854 (App. D.6), 2602.01518 | Gemma 3/4 12B defaults at B=32: −0.6-1.0 ms/round (+10-15%). Frees 311-537 MB of sort workspace | M | M | 1.5 | nsys sort plus merge time, Gemma defaults vs top_k=20, at B=1/8/32 |
| R18 | attention-kv | S1: GRPO rollouts default to BF16 KV | FP8 KV arm in GRPO, after R1, with a logprob-KL readout | 2606.20474 | Qwen3-8B GRPO: ~2× KV lanes; attention DRAM floor 5.4 → 2.7 ms/step. Risk: +1.5-2.6% PPL | L | S | 1.5 | Three-arm GRPO run: tok/s, lanes, rollout-vs-trainer KL |
| R19 | linear-attn | S3: staged GDN prefill leaves SMs idle on 170 SMs | A/B fused vs staged now, then varlen segments per round | 2406.06484 | Qwen3.5-2B/0.8B prefill and Qwen3.8-27B 100-user 2048/16: GDN core is 3-8% of the round; recover up to half | M | S then M | 1.4 | `SUROGATE_SERVE_GDN_PREFILL=fused` vs staged in `sinfer_gated_delta_net_bench --chunked-only` |
| R20 | scheduling | H8: per-row getenv, O(B²) scans, state copies | Static flags, a lane bitmask, borrowed preview | n/a | Qwen3-0.6B GRPO rollouts at B=128: −25-35 µs/round (+0.5-2%) | H | S | 1.35 | ROUND_TIMING boundary, preview and append segments at B=128 |
| R21 | attention-kv | S2: causal prompt CTAs dispatch shortest-first | Longest-first (LPT) order: reverse q-blocks, swizzle heads within a KV group | 2603.05451 (§3.3) | Qwen3-0.6B/8B cold 2K prompt: prompt attention −0-20% at 2 CTAs/SM, prefill −1-4%; outputs bit-identical | M | S | 1.2 | cuobjdump register count → CTAs/SM; bench `--tokens 2048,8192 --context 0` |
| R22 | scheduling | H3: single-CTA Gumbel scan, 4 more full-row passes, 13 route kernels | Split-row multi-CTA sampler with fused online LSE (phase 1 bit-identical) | 2603.15854, 2607.20475 | Qwen3-0.6B/1.7B GRPO rollouts (T=1, top_k=-1, logprobs): +2-3%, or +4-10% at contexts ≤ 300 tokens | H | M | 1.1 | nsys per-round sampler kernels under GRPO args vs greedy |
| R23 | scheduling | H5: one serial finish unit per prompt (head, sample, 3 syncs) | Batched zero-suffix finish, then finalize inside the round | 2403.02310 | 32-64 users × 200-512-token prompts: burst-phase prefill +6-25%, TTFT minus one unit. GRPO step <0.4% | H | M | 1.1 | ROUND_TIMING "finish ms/count"; TTFT p50/p99 |
| R24 | attention-kv | S2: BF16-cache prompt body is capped at 209.5 TFLOPS | 8-bit Q·Kᵀ plus FP8 P·V (SA2++ recipe) as an isolated kernel, gated by R5 | 2505.21136, 2411.10958, 2505.11594 | Qwen3-0.6B 2K prefill −10-20%, only if SA2++-like rates are reached | L | L | 1.1 | ncu tensor-pipe utilisation and barrier stalls of the current body first |
| R25 | attention-kv | S4: per-segment prompt, append and scalar launches, all eager | Varlen persistent prompt kernel over a longest-first tile list, plus a batched append | 2501.01005, 2603.05451 | Qwen3-0.6B GRPO step start (8×256-token prompts per round): attention ~3.8 → ~0.7 ms/round; a few % of the rollout phase | H | M | 0.9 | nsys with 64 concurrent 256-token prompts: segment time and launches per round |
| R26 | linear-attn | S1: GDN state is read and written every step | Windowed deferred fold (W=8-16) on the existing record/fold path | 2610.13778, 2608.15533, 2605.19049, 2609.38166 | Qwen3.8-27B NVFP4 at 100 lanes: up to 8-11% decode (upper bound); 4B ~4%. Restores accuracy if R3 fails | M | M-L | 0.8 | ncu DRAM bytes of the decode kernels at B=64/100 |
| R27 | gemm-moe | H1/H4: standalone activation quantize per projection | Fuse FP8/NVFP4 quantize into the producer norms; quantize once per Gemma projection group | 2610.25048, 2609.12379 | Qwen3-8B-FP8 / Gemma 3 12B NVFP4 at 5-64 tokens: round +1-3% | M | S-M | 0.8 | nsys count and total time of quantize kernels per round |
| R28 | attention-kv | S1: coverage of future GQA ratios | Small-T geometry with the group size given at run time (a separate type) | 2501.01005 (code) | Qwen2.5-7B (28q/4kv) and Mistral leave the generic kernel; within 5% of a compile-time shape | M | M-L | 0.5 | Bench d128-h28-kv4 with the runtime kernel vs generic |
| R29 | gemm-moe | H3: serial one-CTA scan; router re-reads its input once per 16-row tile | Expert-parallel deterministic scan; all-rows router tile later | 2206.01784, 2603.07685, 2609.36301 | MoE prefill on Q4/GGML/W8 routes: −40-60 µs per layer per 4K tokens (~1-2%); router −20-30 µs on the 5090 | M | S | 0.45 | nsys scan and router time per layer; property test on offsets |
| R30 | scheduling | H7: a frontier-band switch re-instantiates the decode graph | LRU of K executables per (B, band) | 2604.06664 | GRPO tail (Qwen3-1.7B): removes up to 127 rebuilds per step (<1%) and the 1-10 ms ITL p99 spikes | H | S | 0.45 | nsys count of `cudaGraphInstantiate`/`cudaGraphUpload` per step |
| R31 | linear-attn | S2: single-user GDN chain without PDL | PDL with base-state prefetch; fold the norm into the out-proj prologue | 2609.12379, 2610.35336 | Qwen3.8-27B, 1 user, MTP on and off: +0.3-1.5% | L | S/M | 0.24 | `nsys --cuda-graph-trace=node` gaps |
| R32 | linear-attn | T4: with recompute on, the GDN forward chain runs 3× | Stash the replay's intermediates for the immediately following backward | 2406.06484 | Qwen3.5-0.8B SFT: +1-2%, peak +0.2 GB during one layer's window | L | M | 0.2 | nsys gdr_* backward time plus peak memory |

**Suggested execution order.**

1. **Zero-code A/Bs first.** None needs a build change, so they fit in one 5090 session:
   - R4: `SUROGATE_SERVE_NVFP4_CUTLASS`
   - R7: `SUROGATE_SERVE_FP8_BLOCK_MMA`
   - R11: `--max-num-batched-tokens`
   - R15: `SUROGATE_SERVE_W8_PIPELINED`
   - R19: `SUROGATE_SERVE_GDN_PREFILL`
   - R18 and R24: existing `--kv-cache-dtype fp8`
   - R12: the proxy run
2. **Gates.** Run R5 and R3 in parallel with step 1.
3. **S-effort code.** F0, then R1, R2, R6, R10, R20, R21 and R9. (F0 for `sinfer_ops` and R1 have landed as T16. R1 is opt-in, so its 5090 A/B comes first.)
4. **M-effort code.** R8, R12, R13, R14, R16, R17, R22, R23 and R25.

Dependencies:
- R1 before R18 and R25 (Qwen3-8B can only pack once it is registered). Since T16 this means running with `SUROGATE_SERVE_GQA_FORK_SHAPES=1`.
- R4 before R16.
- R6 before any speculative GRPO work.
- R3 decides whether R26 is a quality fix.

## 2. Item details

### R1. Register 128-wide 32q/8kv, 40q/8kv and MHA GQA shapes

- **Status: implemented in 7a45b4f7, with the review fix 2e5f766b (ledger T16).** Differences from the plan below:
  - The switch works the other way round. `SUROGATE_SERVE_GQA_FORK_SHAPES=1` turns the registration on, and it is off by default. With it off, 32q8 and 40q8 take their pre-registration routes, and 128-wide MHA is refused as before (its KV pair is withheld). The proposed `SUROGATE_SERVE_GQA_FORCE_GENERIC` does not exist.
  - The decode units are fork files, `fork/native/serve/gqa_attention_decode_gqa128_32q8.cu` and its 40q8 and 32q32 siblings, attached through F0. Routing and acceptance read `fork/native/include/surogate_fork/gqa_fork_shapes.h`.
  - int8 KV stays on the generic route with the switch on, as the skeptic required.
  - The bench refuses B>1 with more than 16 columns, so the single command in the table cannot run. [performance.md](../performance.md#服务端-gqa-形状注册) has the split commands and the 5090 checks. Nothing was run on a GPU.
- **Evidence.** On SM120 the call takes the CUDA-core `gqa_generic_split_kernel`:
  - Why: FA3's `available()` requires cc 9.0, `optimized_decode_shape(128,32,8)` is false, so `generic_takes` wins.
  - Inefficiencies: one CTA per (column, query head, split), so G=4 means each KV byte is read 4× from L2. Math is FP32 FMAs. Each iteration of 8 keys is a dependent chain. Splitting stops once q_heads × columns ≥ 340.
  - ptxas sm_120a, run on clang-generated PTX, reports 48 registers, an un-unrolled loop, and the V load issued after the shuffles.
  - H100 anchor: 157 µs against FA3's 65 µs per layer.
  - The GRPO server sets no KV dtype, so the cache is BF16.
  - Corrected model (16 lanes × 4K): attention is 7-14 ms per step against ~5.5-6 ms with the tensor-core path. A 40-token turn on a 30K history takes 30-180 ms against 3-5 ms.
  - Skeptic caveats:
    - Add an int8 exception, because `kGqaI8DecodeRegistered` is head dim 256 only. Without it, int8-KV Qwen3-8B would go from working to throwing.
    - Every X() line instantiates every prefill variant, so build time grows.
    - Qwen3-14B (G=5) supports widths 1-6 and 8 only; width-16 verify takes the chunked route.
    - Output bits change: the generic kernel keeps P in FP32, the tensor-core kernel rounds P to BF16. Batch invariance holds because the partition table is unchanged.
- **Boundary and switch.**
  - One ledger topic covering:
    - the X() lines in `SINFER_GQA_FOR_EACH_GEOMETRY` (`gqa_attention_geometry.cuh`);
    - the int8 exception in `optimized_decode_shape` (`wrapper/gqa_attention.cpp`);
    - CMake source lines, or launcher TUs placed in `fork/native/serve/` through F0, following `gqa_attention_decode_gqa128_16q2.cu`.
  - Switch: `SUROGATE_SERVE_GQA_FORCE_GENERIC=1` (new), which routes back to the generic kernel for A/B.
- **Measure.**
  - The bench line in the table, before and after.
  - ncu: `lts__t_sectors_srcunit_tex.sum` against unique KV bytes (expect ~4× → ~1×), `dram__bytes_read.sum`, stall_long_scoreboard sample share, `sm__pipe_tensor_op_hmma_cycles_active`.
  - End to end: Qwen3-8B with 16 concurrent 4K rollouts. Report decode tok/s and the nsys share of the attention kernels.
  - Correctness: `sinfer_gqa_attention_test` precision cases, plus the mean |Δ logprob| of rollout vs trainer.
- **Memory axes.**
  - alloc: none new; the split workspace is sized by the same partition table.
  - life: unchanged.
  - peak: unchanged at runtime; report the build-time increase.
  - tput: decode tok/s.
- **Success.**
  - ≥1.15× decode tok/s at 16 × 4K with BF16 KV.
  - L2-to-SM sectors ≤ 1.3× unique KV.
  - Attention for the 30K-history turn ≤ 10 ms.
  - int8 KV still served.

### R2. Generic-K NVFP4 GEMV at T=1 in linear_add, attn_input_proj and gdn_input_proj

- **Evidence.**
  - The current T=1 path:
    - `nvfp4_linear_add_plan.cpp` and `nvfp4_gdn_input_plan.cpp` give the T=1 GEMV only to the hidden-2560 family.
    - Generic shapes quantize, then `nvfp4_cublaslt_gemm` picks tile 1 (256×128×128, not swapped), which yields 32 CTAs for 4096 rows.
    - Per-shape cost: o 4096² takes 27 µs against a 5.3 µs floor; down 4096×12288 takes 82 µs against ~16 µs.
  - Qwen3.5-9B (24 GDN layers and 8 attention layers): ~6.3 ms of linears per token against ~2.2 ms.
  - Precedent: `linear` already serves generic shapes at T=1 on BF16 activations, so A16 at T=1 is an accepted numeric. The GEMV is templated on its epilogue, and `Nvfp4AddResidualEpilogue` exists.
  - Literature: Minima 2609.04098 finds activation-quantize overhead is only a few % in vLLM, so the gain comes from dropping the padded tile. MR-GPTQ 2509.23202 (App. G, B200) finds W4A4 brings no benefit at small batch.
  - Scope:
    - Affected: Qwen3.5 dense generic shapes, i.e. 2B and 9B, and 0.8B shapes with ≥ 4096 rows. The 4B shapes and Gemma at T=1 are already covered.
    - T=1 only. MTP verify at T=2-4 needs R4 and R16.
- **Boundary and switch.**
  - A new fork TU in `fork/native/serve` (via F0) with generic-K launchers for a residual-add epilogue and a segment-pointer epilogue.
  - One ledger topic for the T==1 branches in the two plans and in `nvfp4_attn_input_w4a4.cu`.
  - Switch: the existing `generic_gemv_enabled()` plus `SUROGATE_SERVE_NVFP4_GENERIC_GEMV_FUSED=0` (new) to disable.
- **Measure.**
  - Route bench on the four 9B shapes at T=1: µs and `dram__throughput.avg.pct_of_peak_sustained_elapsed`.
  - End to end: Qwen3.5-9B NVFP4, 1 user, 512/512 tokens, MTP off. nsys `cuda_gpu_kern_sum` per token; tok/s.
  - Accuracy: WikiText PPL window against the W4A4 path.
- **Memory axes.**
  - alloc: no activation-quantize launches at T=1; count them in nsys.
  - life and peak: unchanged.
  - tput: tok/s.
- **Success.**
  - o-proj ≤ 8 µs at T=1, and the GEMV at ≥ 70% of DRAM peak.
  - Single-user tok/s ≥ 1.4×.
  - PPL within noise.

### R3. Accuracy audit of BF16 per-step GDN state

- **Evidence.**
  - #57 moved GDN state to BF16 storage and validated it only with temperature-0 coherence checks and throughput.
  - LeapQuant 2609.38166 (Qwen3.5-9B): per-step BF16 storage gives AIME 87.9 → 72.1 and LCB 64.1 → 49.6. Re-quantizing once per 16-token window gives 87.8. Fig. 3 shows state MSE growing over 64K tokens.
  - ReplaySSM 2610.13778 and STEPQuant 2609.38169 report FP32 as the default state dtype in vLLM and SGLang for these models. That contradicts #57's premise that vLLM stores state at model dtype, so check which dtype the vLLM comparison rows actually used.
  - Prefill rounds the state to BF16 once per 64-token chunk; decode rounds it every token.
  - Swamping hypothesis: in BF16, s·α rounds back to s when 1−α < ~2^-9. Minima's decay horizons imply 1−α ≈ 1.6e-5.
  - Relevance: dense Qwen3.5 GRPO rollouts run on the serve engine (BF16 state) while the trainer uses chunked FP32, so the two diverge.
- **Boundary and switch.** No new kernels. `GdnStateStorage` is a single alias in `recurrent.cuh`. Add a build option `SUROGATE_SERVE_GDN_STATE_FP32` (new, CMake). Put the eval scripts under `fork/tools` or `fork/experimental`.
- **Measure.**
  1. A fixed 64K-token decode replayed with FP32 and with BF16 state, logging relative state MSE per layer per 1K tokens.
  2. A long-generation eval with ≥ 16K generated tokens (AIME-style or long-CoT GSM8K), Qwen3.5-4B and 9B, several seeds.
  3. On a fixed GRPO batch, rollout-vs-trainer per-token logprob KL and mean |Δ|.
- **Memory axes.**
  - peak: FP32 state doubles the footprint (27B: 72 → 144 MiB per lane); report the runtime reservation at 64 and 128 lanes.
  - DRAM bytes per decode step: doubled.
  - tput: −10-13% per #57.
- **Decision.**
  - If BF16 loses more than 2 points (or more than 1σ across seeds), or state MSE keeps growing: make FP32 the quality default and promote R26.
  - Otherwise: record a null result.

### R4. A 5090 size rule for generic NVFP4 from existing tiles

- **Evidence.**
  - `nvfp4_cutlass_tile_for` returns `rows >= 4096 ? 1 : -1` off GB10. `device_is_gb10` checks 12.1, so the 5090 (12.0) keeps the rows rule, which #134 measured on prompt rounds only.
  - On GB10, below 64 tokens, tile 1 took 1.9× cuBLASLt's time (3.5× the best tile). Gemma 4 12B at 8 users went from 131.2 to 172.8 tok/s.
  - Swapped tiles 4 and 5 exist but are never picked off GB10. Tile 4 gives a 4096-row weight only 16 CTAs, so do not copy `gb10_tile` as-is.
  - Gemma is affected at T=2..63, where `linear_projections` quantizes the same activation once per projection.
  - The whitepaper lists dense FP4 with FP32 accumulate at 1676 TFLOPS, not half rate.
- **Boundary and switch.**
  - The table data lives in a new header under `fork/native/include/surogate_fork`. A ledger topic changes `nvfp4_cutlass_tile_for` in `nvfp4_cublaslt.cpp` to read it.
  - Switch: `SUROGATE_SERVE_NVFP4_SM120_RULE=table|rows` (new; default `rows` until measured). The existing `SUROGATE_SERVE_NVFP4_CUTLASS=<id>|off` still overrides.
- **Measure.**
  - `sinfer_nvfp4_route_bench` over cuBLASLt, the six tiles and the GEMV, on the 9B shapes (4096², 12288×4096, 4096×12288, 24576×4096) and the Gemma 3 12B shapes, at T=1..64. Record µs and DRAM throughput.
  - Engine A/B at 1/4/8/16 users and with MTP verify; nsys per round.
- **Memory axes.**
  - alloc: Stream-K tiles 2/3 refuse to allocate their workspace during capture, so it must be prewarmed. Report the workspace bytes as peak, and any memset nodes per graph.
  - Other tiles: none.
- **Success.**
  - ≥ 1.2× decode tok/s at 2-16 users on one of the two models.
  - No regression on prompt rounds of ≥ 2048 tokens.

### R5. SM120 microbenchmark suite (gate)

- **Evidence.**
  - The repo's ceiling arithmetic uses 209.5 and 419 TFLOPS, while PATCHES #19 implies an INT8 ceiling of ~420 TOPS. The whitepaper and SA3 Table 17 imply 838 for INT8 and for FP8 with FP16 accumulate.
  - FA2 has been measured at 214 TFLOPS, above spec, so record the clock.
  - FP22 truncation of FP8 MMA with FP32 accumulate (2411.10958 §3.4) has been shown only on Ada and Hopper.
  - FP64 at 1/64 rate underlies R6.
  - FlashQLA's `benchmark_results_5090.txt` and FlashInfer's SM120 GDN and attention kernels are the only public 5090 references.
- **Boundary.** A new directory under `fork/native/bench` (owned): a standalone nvcc `-arch=sm_120a` build with a `run.sh` modeled on the cpu-optimizer bench. No upstream edits.
- **Measure.**
  - Per-SM throughput with an ILP sweep, and dependent-chain latency, for: mma.sync bf16/f32, f16/f16, e4m3/f32, e4m3/f16, s8/s32, kind::mxf8f6f4 block_scale, kind::mxf4nvf4 block_scale m16n8k64, and DFMA.
  - The FP22 probe (A=B=0, vary the mantissa bits of D).
  - `nvidia-smi --query-gpu=clocks.sm`, both locked and unlocked; 3 runs.
  - Add FlashInfer and FlashQLA reference rows to `sinfer_gated_delta_net_bench`, and SA2++ rows to the attention bench.
- **Memory axes.** Not applicable.
- **Success.** A rate table checked into `fork/docs/performance.md` that every ceiling claim cites. Decision rules:
  - If s8 and FP8 with FP16 accumulate both reach ≥ 1.8× the BF16 rate, R24 may proceed.
  - If FP8 with FP32 accumulate truncates the mantissa, add two-level accumulation to the existing PvFp8 loop.

### R6. FP32 normalisers for nucleus and wide sampling and for speculative accept

- **Evidence.**
  - `sampling_untruncated_exact` computes `exp((double(x)-max)/T)` per token. The nucleus mass in `sampling_sorted` is FP64 over the full support, and the thread holding the boundary rescans its segment serially.
  - SASS (nvcc 13.1, sm_120a): ~26 FP64-pipe instructions per token, with ~64-cycle stalls between dependent DFMAs.
  - At 2 DFMA per clock per SM: ~0.8 ms per 151,936-token column, ~0.67 ms per 128,256-token row, plus 10-130 µs for the serial rescan. On H100 the same work is ~30 µs, which is why it never showed upstream.
  - Exposure:
    - Exposed: top_p < 1 with top_k unset (the preset default is top_k=0), i.e. Llama 3.x defaults, on every decode round and every DFlash verify.
    - Not exposed by default: Qwen, whose top_k=20 avoids the route; MTP on untruncated rows needs a client to override top_k.
    - GRPO's untruncated rows use the Gumbel route. They become exposed once speculative rollouts exist.
  - Precision: the accept test compares against a 24-bit uniform, and compensated FP32 is ~1e-7. Keep FP64 only for the 256-way `exact_sum`.
  - SonicSampler 2607.20475 supports the fused multi-column verify structure only. Its date is uncertain and its code is not released.
- **Boundary and switch.**
  - A new header in `fork/native/serve` holds:
    - `sampling_untruncated_fp32`;
    - the FP32 nucleus mass with a parallel boundary search;
    - `speculative_column_normalizers`, grid (chunks, k+1, rows), writing per-column max, sum and draft logit into `SamplingWorkspace`.
  - One ledger topic for the hooks in `ops/launcher/sampling.cu` and `ops/launcher/speculative_round.cu`.
  - Switch: `SUROGATE_SERVE_SAMPLER_FP64=1` (new) keeps the old path as a reference.
  - Phase 2: draw the residual only at the first rejection. That uses a new RNG purpose, so version the stream.
- **Measure.**
  - ncu `gpu__time_duration.sum` and `sm__inst_executed_pipe_fp64.sum` for `sampling_wide_kernel` and `speculative_accept_wide_kernel`, on Llama-3.2-1B and Llama-3-8B, T=0.6, top_p=0.9, top_k unset, B=1/8/32.
  - Single-stream and 8-user tok/s.
  - `sinfer_qwen3_5_mtp_round_bench` with T=1 and top_k=-1.
  - Statistical check: chi-square on 10^6 draws from fixed logits, FP32 vs FP64, plus the acceptance rate within its CI.
- **Memory axes.**
  - alloc: none per round.
  - peak: workspace of (k+1)·B·12 B, negligible.
  - tput: tok/s.
- **Success.**
  - Sampler ≤ 50 µs per round at B ≤ 32.
  - Llama-3-8B ≥ 1.12× tok/s; Llama-3.2-1B ≥ 1.3×.
  - Chi-square p > 0.01.

### R7. `mma_kernel` on SM120 for 5-16 tokens

- **Evidence.**
  - Current routing: `mma_enabled()` is `device_cc()==90` unless `SUROGATE_SERVE_FP8_BLOCK_MMA` is set. Off Hopper the GEMV covers up to 4 tokens, and 5-32 tokens take CUTLASS Narrow-32 with grid n/128, i.e. 32 CTAs on 170 SMs.
  - H100 measurement (#107), at 4 tokens: o 17.7 vs 8.6 µs, down 42.7 vs 21.1 µs.
  - Expected on the 5090 (skeptic): ~1.5-1.8× the floor. `mma_kernel` covers only ≤ 16 tokens (`kMmaMaxTokens`). Its shared memory is a few KB, so the 99 KB limit does not bind, and the 8-vs-4-warp choice already adapts to occupancy.
  - RaMP's split-K regime rule does not apply, because `mma_kernel` splits K inside one CTA.
- **Boundary and switch.**
  - Zero-code A/B first.
  - If it wins: a ledger topic in `fp8_block.cu` replaces `device_cc()==90` with a per-arch token range.
  - For 17-32 tokens: a fork TU with a swapped 128×16 blockwise CUTLASS tile using fixed split-K, attached to `sinfer_fp8_block_sm120` through F0. Vendored CUTLASS 4.6.1 accepts small N.
- **Measure.**
  - `sinfer_linear_fp8_block_test` with `SUROGATE_SERVE_FP8_BLOCK_MMA=1` vs `SUROGATE_SERVE_FP8_BLOCK_SM120_TILE=p32` on 4096², 4096×12288 and 12288×4096 at T=5..32. ncu DRAM throughput.
  - End to end: Qwen3-8B-FP8 at 4/8/16 users; nsys per round.
  - Report logprob drift: activations are exact BF16, so outputs differ from the FP8-quantized tile.
- **Memory axes.** Static shared-memory partials only; alloc, life and peak are unchanged.
- **Success.**
  - o/down ≤ 1.3× the streaming floor at 5-16 tokens.
  - Decode +8% or more at 8-16 users.

### R8. Varlen chunked GDN for packed training

- **Evidence.**
  - `gated_delta_rule.cpp` loops over documents on the host:
    - forward: 6 launches per document;
    - backward: 12 per document, including `cumsum_rev`;
    - replay: recompute defaults to True, so the GDN op runs again.
  - That is ~24·D + 10 launches per GDN layer, ~14.5k per micro-step on the 0.8B example (D ≈ 33). Each per-document kernel has ~64 CTAs.
  - `row_packing.cuda_graphs_for_gated_delta_rule` turns graphs off for packed SFT. GRPO keeps graphs off regardless, and QLoRA is eager anyway.
  - Estimate: ~45-60 ms per micro-step today against ~34 ms varlen. The varlen figure comes from FlashQLA's 5090 FLA rows (0.406 ms forward, 1.089 ms backward per layer at 8k tokens), which are graph-replay numbers and a proxy shape.
  - In-repo pieces already exist:
    - `fla_kda/state.py` has IS_VARLEN `fwd_h` and `bwd_dhu`, built for KDA only.
    - KDA's `prepare_metadata` and `prepare_indices` use −1 sentinels.
    - `set_doc_masking(pad_to_max)` already pads `cu_seqlens` with zero-length documents.
  - Bitwise caveat: deriving the varlen kernels from the in-repo GDN kernels (natural `tl.exp`, same tiles, BF16 h/w/u) keeps bitwise equality with the per-document path. Re-vendoring from FLA v0.5.2 (exp2, FP32 intermediates) needs a tolerance test instead.
- **Boundary and switch.**
  - Kernel source in a new module under `surogate/_fork`, registered in the AOT manifest.
  - A `GatedDeltaRuleKernels` dispatch class in `fork/native/src`, mirroring `KimiDeltaRuleKernels`.
  - One ledger topic for the hook in `gated_delta_rule.cpp` that replaces both per-document loops, and for the flag flip in `row_packing`.
  - Switch: `SUROGATE_GDN_VARLEN=1` (new; default off until the A/B).
- **Measure.**
  - nsys over 20 steps of `examples/sft/qwen35/qwen35-text-lora-bf16.yaml`: gdr_* launches per step, `cuda_api_sum` launch time, GPU idle gaps.
  - Four arms: per-document; varlen; varlen plus graphs; `sample_packing: false`.
  - Correctness: bitwise (or tolerance) against per-document; the document-isolation tests; N_max overflow must fail loudly; capture once and replay with varying document counts.
- **Memory axes.**
  - alloc: workspace sized for NT_max = ceil(B·T/64) + N_max, allocated once and graph-stable.
  - life: per op.
  - peak: report the trainer high-water mark (expect tens of MB more).
  - tput: tok/s.
- **Success.**
  - ≥ +7% tok/s on the 0.8B example.
  - ≤ 1.5k launches per micro-step.
  - Peak memory +3% or less.

### R9. Fine decode partitions on SM120 (and a 2-slot ring for head dim ≤ 256)

- **Evidence.**
  - `gqa_fine_key_partitions` is gated on `__CUDA_ARCH__==900` and host cc 9.0. Past 16K, SM120 keeps 2,048-key partitions.
  - BF16 decode tiles are staged synchronously (`kPrefetch = QResident && kFp8Cache`).
  - Little's law: saturating the bus needs 1.25 MB in flight. 32 tail CTAs × 16 KB is only 0.52 MB.
  - Corrected tile times: ~1.5-1.7 µs at D=128, ~2.8-3 µs for the D=512 resident tile.
  - Gemma 4 12B global layers (16q/1kv/512; BF16 is the default because the model is attention-only), at 32K: ~185 → ~40 µs per layer.
  - Repo anecdote: on H100 at 17K, ~80 µs per layer sat behind one 2,048-key partition.
  - The ring cannot be used for the D=512 resident tile: 144 KB > 99 KB.
  - Little effect for Qwen3 shapes with 8 KV heads (near the floor already); Qwen3-30B-A3B is off-target.
  - Literature: LeanAttention 2405.10480 supports this only indirectly. FlashDecoding++ was mis-cited for the ring and is dropped; PAT 2511.22333 remains.
- **Boundary and switch.**
  - One ledger topic changing the gate in both the device and host branches of `gqa_attention_decode.cuh`.
  - Switch: `SUROGATE_SERVE_GQA_FINE_PARTITIONS=1` (new). It must be read once at engine start so it stays a per-device constant and batch invariance holds.
  - The ring is a template flag chosen when the grid is under 2 waves.
- **Measure.**
  - The bench in the table for d512-h16-kv2 and d128-h32-kv4. ncu `dram__throughput.avg.pct_of_peak_sustained_elapsed` and `sm__warps_active`.
  - End to end: Gemma 4 12B, 1 user, at 32K.
  - Batch-invariance test: the same row must give the same output at B=1 and B=8.
- **Memory axes.**
  - alloc: once, at plan time.
  - peak: the split partial workspace grows ~3×, because it scales with partitions × batch × width. Report the plan high-water mark at max context × max batch.
  - Graphs must be re-captured.
- **Success.**
  - 32K decode attention ≤ 1.3× the DRAM floor.
  - Gemma 4 12B tok/s ≥ 1.2× at 32K.
  - Workspace growth ≤ 2% of the card at the configured max batch.

### R10. Grammar masks: thread-pool fill, pinned batched H2D, then overlap

- **Evidence.**
  - `staged_sampling → update_constraint` runs serially per row on the worker thread, with a 19 KB pageable `cudaMemcpyAsync` per row, between a synchronous consume and the next launch. Constrained rounds force burst=1.
  - The llguidance v1.8.0 header says mask computation takes "up to a millisecond for a 100k tokenizer ... call on a background thread".
  - Published per-token costs: XGrammar under 40 µs for JSON (its own engine); XGrammar-2 measured llguidance v1.2.0 at 250 µs-1 ms. XGrammar-2 also reports end-to-end overhead ≤ 6% on a 5090.
  - Matchers are independent per request. Not used by GRPO.
- **Boundary and switch.**
  - A new constraint-pool module in `fork/native/serve`: a persistent thread pool runs accept plus fill per row on its own thread, and CUDA calls stay on the worker. A pinned arena of [max_lanes, V/32] (2.4 MB at 128 lanes).
  - One ledger topic for the `staged_sampling` loop in `program_impl.h`.
  - Switch: `SUROGATE_SERVE_CONSTRAINT_THREADS=N` (new; 0 means serial).
  - Phase 2 (M): constrained rounds use two graphs, forward-to-logits and sampler tail. A graph's event wait cannot wait for a host copy enqueued after the launch, so one graph is not enough.
- **Measure.**
  - NVTX range around `update_constraint`; nsys worker-thread timeline against GPU idle time.
  - Structured-output bench (JSON schema plus tool calls) at 16/32/64 users on Qwen3-8B: TPOT p50 and p99, constrained vs unconstrained.
- **Memory axes.**
  - alloc: one fewer pageable copy per row per round (count the `cudaMemcpyAsync` calls).
  - peak: +2.4 MB of pinned host memory for the engine's lifetime.
  - tput: TPOT.
- **Success.**
  - Phase 1: host mask time per round ≤ 25% of the serial time.
  - Phase 2: constrained TPOT within 10% of unconstrained.

### R11. Round-budget policy (TTFT/TPOT)

- **Evidence.**
  - The closed-loop identity holds: 32 × 2048 / 52.5k = 1.248 s, τ = 39 ms, and TTFT = L − 15τ = 0.663 s against 661 ms measured. vLLM implies τ ≈ 57 ms (~3.2k tokens per step).
  - Only the 0.95× prefill throughput is lost work; the rest of the TTFT gap is budget policy.
  - Constraint: activation workspace and mixed-graph buckets are planned from the configured chunk. `try_mixed_graph_chunk` refuses `chunk_bucket + batch_bucket > prefill_chunk_`. An adaptive policy therefore needs a larger configured `--max-num-batched-tokens`, which costs memory.
  - Literature:
    - Sarathi 2403.02310 picks the budget by profiling and notes that chunked KV re-reads are cheap, so the GRPO gain is ≈ 0.
    - SLOWeave 2609.07883 is mostly simulation.
    - DLFP 2609.38386 is a negative result: a completion-aware controller also failed at 32B.
    - LPRS 2606.09061 is the only SM120-class evidence: ~4% at P99.
  - Conclusion: use a static backlog rule.
- **Boundary and switch.** A `RoundBudgetPolicy` interface factored out of `run_mixed_round` in `concurrent_executor.h` (ledger topic), implemented in a new header in `fork/native/serve`. Switch: `--round-budget-policy=fixed|backlog` (new; default fixed).
- **Measure.**
  - First a zero-code sweep of 2048/3072/4096 on the 5090 closed loop (32 users × 2048 / 16) and on 100 users × 512 / 128.
  - Report TTFT p50/p99, TPOT p50/p99, prompt tok/s and output tok/s together, and check TTFT ≈ L − 15C/X.
- **Memory axes.**
  - peak: activation workspace grows with the configured chunk; report the runtime reservation and the KV pages it costs.
  - alloc: none per round.
  - tput: should move ±2% at most.
- **Success.**
  - TTFT p50 ≥ 0.8× vLLM on the same machine, at a TPOT increase no larger than the budget ratio.
  - Throughput within ±2%.
  - The board row reports TPOT alongside TTFT.

### R12. GRPO in-flight prefix dedup

- **Evidence.**
  - `plan_shared_prefix_capture` needs an earlier request's page hashes and at least 1024 tokens. There are 4 slots, and `shared_prefix_page_budget` is hardcoded to capacity/8, which caps the env-only proxy.
  - On the initial fill and after a checkpoint barrier, deferred-first-chunk admission stages all G members at once. In steady state, members trickle in one per freed slot, so redundancy is ~2/G when P ≥ 1024. Nothing is shared when P < 1024.
  - Cost:
    - Q=32, G=8, P=1000: 224k redundant prefill tokens, 15-28 s at 8-15k tok/s, i.e. 11-22% of a ~128 s step.
    - P=300: 3-6%.
    - KV: the prompt share of a live lane is P/(P + R/2) ≈ 57%.
  - Disabled under DFlash and `--batch-invariant`. On GDN hybrids each slot holds a recurrent-state snapshot of tens of MB.
  - Literature:
    - RadixFlow 2610.18143 supports a one-round hold on hybrids; its workload is prefill-heavy candidate scoring, not GRPO.
    - SGLang's in-batch deprioritisation (thresholds 32/32) is evidence from code.
    - FlashInfer's 13-17% does not apply here (see section 3).
  - Design per the skeptics:
    - Use the hold only, with no same-round borrowing.
    - Followers stay unplanned in `pending_`.
    - The leader plans the capture whenever matching followers are pending.
- **Boundary and switch.**
  - A new in-flight prefix-dedup module in `fork/native/serve`, called from `top_up_prefill_lanes`.
  - The same ledger topic makes the page budget configurable.
  - The orchestrator's group tag is sent through a `surogate/_fork` patch.
  - Switches: `SUROGATE_SERVE_INFLIGHT_PREFIX_DEDUP=1` and `SUROGATE_SERVE_SHARED_PREFIX_PAGE_FRACTION` (both new), plus the existing `..._MIN_TOKENS` and `..._SLOTS`.
- **Measure.**
  - Proxy first: a scratch build without the cap.
  - Qwen3-8B and Qwen3-1.7B GRPO steps with Q=32, G=8, P = 300/1000/2048, R ≈ 1500. Measure the burst phase and the steady-state phase separately.
  - Metrics: `computed_prefill_tokens` vs Q·P; `prefix_cache_hit_tokens_total{tier="shared_prefix"}`; peak KV pages in use; preemptions per step; step time; follower TTFT delay.
  - Correctness: greedy outputs equal token for token with and without dedup.
- **Memory axes.**
  - alloc: page allocations per step fall as followers borrow.
  - life: the leader's prompt pages stay live until the last follower finishes (refcount); measure page residency.
  - peak: KV pages in use.
  - tput: step time.
- **Success.**
  - `computed_prefill_tokens` ≤ 1.05·Q·P + Q·G.
  - Peak prompt-KV pages fall by about (G−1)/G of the prompt share.
  - Step time −5% or better at P=1000, or concurrency +20% when KV-bound.
  - Zero equality failures.

### R13. Channel-last GDN conv for training

- **Evidence.**
  - The DSL graph is transpose → `mamba_conv1d` → transpose → split (three `cudaMemcpy2DAsync` calls), with transposes launched once per row.
  - The `packed_convolution` backward recomputes the conv and the expf 5 times, runs 5 binary searches and 5 block reductions.
  - With the default recompute, traffic is ~3.1 GB per GDN layer-step at 0.8B, ~55 GB per micro-step, ≈ 30 ms at 1.8 TB/s before compute.
  - The cheapest fix: the vendored Dao `causal_conv1d_channellast_{fwd,bwd}` kernels. They take `seq_idx` (a [B,T] int32 built on the device from `cu_seqlens`), are instantiated for BF16, are already compiled into the trainer, and have a `kDeterministic` option for dweight.
  - Packed and padded rows must switch together to keep the row-packing bit-identity contract.
- **Boundary and switch.** A native op wrapper in a new `fork/native/src` source. One ledger topic for the rewrite in `surogate/dsl/modules/gated_delta_rule.py`. Switch: `SUROGATE_GDN_CONV_CHANNEL_LAST=1` (new).
- **Measure.**
  - nsys time of transpose, `packed_convolution` and memcpy2D against the channel-last kernels.
  - ncu `dram__bytes_read.sum` and `dram__bytes_write.sum` per GDN layer (expect ~3.1 → ~1 GB).
  - tok/s on the 0.8B example, and on 4B and 9B unpacked at seq 4096.
  - Gradients within tolerance; document isolation; padded vs packed identity.
- **Memory axes.**
  - alloc: the two [B, ConvDim, T] transposed arena slots disappear; seq_idx adds B·T·4 B.
  - peak: should not increase.
  - tput: tok/s.
- **Success.** +4% tok/s or better on the 0.8B example, +1% on 4B, peak not higher.

### R14. Shared expert through the dense precompute

- **Evidence.**
  - `sparse_moe_prefill_w8_gate_up_kernel` runs 64×64 tiles at `__launch_bounds__(256,1)`, each K step does cp_wait<0>, a shared-memory int8→BF16 dequant, two barriers and 16 MMAs per warp. It is capped at the 209.5 BF16 rate: ≥ 123 µs per layer per 4096 tokens on the 35B geometry.
  - The routed experts run NVFP4 on TRT-LLM at the FP4 rate, so the shared expert costs 30-100% of the routed time. It is also serialized on the main stream.
  - `native_shared()` already precomputes non-W8 shared experts. Two things block it:
    - the validation in `sparse_moe.cpp` admits only GGML;
    - the converter requantises the shared expert to W8.
  - Sending a W8 expert to the precompute lands on A16 W8 kernels (`A16Only`), so keep block-FP8 or NVFP4 instead.
  - An aux stream helps only below ~256 tokens (vLLM's threshold).
- **Boundary and switch.** One ledger topic covering a converter flag `--shared-expert-format=keep` (new) in `surogate/serve/convert/qwen3_5_moe` and the relaxed validation. Switch for the aux stream: `SUROGATE_SERVE_SHARED_EXPERT_STREAM_TOKENS` (new; default 0).
- **Measure.**
  - nsys per-layer time of the shared gate/up and down kernels against the routed kernels at 2048 and 4096 tokens on the owner's ~10B MoE (35B-A3B layer geometry as proxy).
  - The same after the change; prefill tok/s.
  - Output diff against the W8 path.
- **Memory axes.**
  - alloc: a [T,H] BF16 shared_output plane (16.8 MB at 4096 × 2048).
  - life: one MoE layer, reused.
  - peak: +17 MB.
  - tput: prefill tok/s.
- **Success.** Shared expert ≤ 50 µs per 4096 tokens; prefill +3% or better.

### R15. SM120 W8 decode sweep (measure first)

- **Evidence.**
  - `pipelined_route` is major==9 unless forced.
  - Coverage, corrected by the code skeptic, for Qwen3-8B:
    - q and o (4096²) use the measured SIMT r8_c4 route at T ≤ 8;
    - gate/up (53% of bytes) takes `w8_swiglu_generic_kernel` (one warp per output) at T=1;
    - only k, v, down and the head (~36% of bytes) reach `launch_w8_consistent`.
  - H100: the 8B head went 330 → 216 µs.
  - Little's law: 2610.34907 needs ~32 KiB in flight per SM on H100, which scales to ~12 KB on the 5090 — more than medium-T's 8 KiB per CTA. A latency-bound kernel is plausible.
  - The pipelined layouts take 50-73.5 KB, which fits 99 KB.
- **Boundary.** Measure first. If it wins, a per-arch shape table in `pipelined_route` (ledger topic), not a blanket major ≥ 9.
- **Measure.**
  - `sinfer_w8_pipelined_test` on (1024,4096), (4096,12288), (151936,4096) and 4096² at T=1..64, with ncu DRAM throughput and the long-scoreboard stall share.
  - nsys of the gate/up `w8_swiglu_generic_kernel` at T=1.
  - End to end: Qwen3-8B W8 at 1 and 8 users, A/B with `SUROGATE_SERVE_W8_PIPELINED=1`.
- **Memory axes.** Shared memory only; alloc, life and peak are unchanged.
- **Success.**
  - Enable each shape that is ≥ 10% faster.
  - End to end +5% or better.
  - If the SIMT SwiGLU kernel runs below 80% of DRAM peak, open a new item.

### R16. Narrow swapped NVFP4 tiles with fixed split-K

- **Evidence.**
  - The vendored CUTLASS 4.6.1 SM120 blockscaled builder accepts N ≥ 8: Pingpong needs N ≥ 16, Cooperative uses an 8×1 warp layout below 16. CUTLASS 4.5.0 added 128×32 and 128×64 tiles.
  - A swapped 128×32 tile per K block moves 9.2 KB (0.88 µs at the per-SM bandwidth share) for 0.11 µs of MMA. It is weight-streaming bound and needs ≥ 160 CTAs.
  - Stream-K split points depend on the token count, which breaks batch invariance, and 2301.03598 §6 warns against Stream-K on memory-bound shapes. Use a fixed split-K per (rows, k).
- **Boundary and switch.** A new fork TU in `fork/native/serve` attached to `sinfer_nvfp4_cutlass` through F0. New tile ids go into the R4 table. Switch: `SUROGATE_SERVE_NVFP4_CUTLASS=<id>`.
- **Measure.** Route bench at T=2..63 on the 9B shapes: µs against the 5.3 and 15.8 µs floors, DRAM throughput, and the split-K workspace.
- **Memory axes.**
  - alloc: split-K partials preallocated before capture.
  - peak: report the partials size.
  - tput: tok/s.
- **Success.**
  - o ≤ 8 µs and down ≤ 20 µs at T ≤ 32.
  - Row outputs at T=2 identical to T=32.

### R17. Bounded top-128 sampler with a mass certificate

- **Evidence.**
  - Wide-filtered rows sort the whole vocabulary as 64-bit keys, then run 7-8 merge passes every round and every verify column.
  - Gemma at B=32 has 128 MB of keys, more than L2: 0.6-1.0 ms plus 0.1-0.2 ms of tile sorts. Llama at B=32 stays L2-resident: 0.15-0.3 ms.
  - Raising `kSamplerFastCandidates` does not work: the shared merge arrays at K=128 need 256 KB, more than 99 KB. `multiblock_ok` fails, and the per-thread arrays spill.
  - The sort workspace (311 MB for Qwen, 537 MB for Gemma) is reserved whatever route runs.
  - Rows with top_k ≤ 128 need no certificate. The certificate is needed only for top_p < 1 with top_k unset.
  - SonicSampler's bounded k=128 is "effectively lossless", not exact; the certificate is our addition. Qrita (2602.01518) is the later fallback at 0.1-0.25 ms.
- **Boundary and switch.** A new header in `fork/native/serve`: stage 1 builds a per-tile top-128 plus (m, s) in shared memory; stage 2 merges, checks the certificate and draws. Hooks share R6's launcher files under a separate ledger topic. Switch: `SUROGATE_SERVE_SAMPLER_BOUNDED=1` (new). The existing sort serves certificate misses.
- **Measure.**
  - nsys sort plus merge time per round with Gemma 3 12B defaults vs top_k=20, at B=1/8/32.
  - Certificate fallback rate on real chat traffic.
  - tok/s; chi-square against the sort route.
- **Memory axes.**
  - peak: if the sort workspace is resized for fallback rows only, the reservation drops by 311-537 MB (report the runtime reservation).
  - alloc: none per round.
  - tput: tok/s.
- **Success.**
  - Sampler ≤ 60 µs per round at B=32 on Gemma.
  - tok/s ≥ 1.1×.
  - Record the fallback rate; target ≤ 5%.

### R18. FP8 KV arm for GRPO (after R1)

- **Evidence.**
  - `surogate_engine.py` forwards `kv_cache_dtype` only when set, and the GRPO config default is None, so the cache is BF16.
  - Repo measurement (`api/types.h`): e4m3 KV costs +1.5% (Q4_K_M) and +2.6% (IQ4_XS) perplexity on pure-attention Qwen3-0.6B.
  - The codec is a raw saturating cast with no scale plane; check per-layer K/V amax. Qwen3 has QK-norm, Llama does not.
  - UltraQuant 2606.20474 is MI355X data, and its SWE-bench result (n=100) is weak.
  - FP8 KV buys speed only after R1, because the generic kernel is latency-bound.
- **Boundary and switch.** A recipe under `fork/experimental/examples` plus a `surogate/_fork` config default. Switch: `kv_cache_dtype: fp8` in the GRPO config.
- **Measure.** Qwen3-1.7B and 8B GRPO with fixed seeds, BF16 vs FP8 arms: rollout tok/s, maximum concurrent lanes, rollout-vs-trainer per-token KL and mean |Δ|, and the reward curve.
- **Memory axes.**
  - peak: KV bytes per lane halve, so more lanes fit.
  - alloc: unchanged.
  - tput: rollout tok/s.
- **Success.**
  - Rollout throughput ≥ 1.3×, or resident lanes ≥ 1.5×.
  - KL < 2× the BF16 arm, reward within the seed band.
  - Otherwise record a null result.

### R19. GDN serve prefill on 170 SMs

- **Evidence.**
  - Each segment makes its own `linear_recurrence` call: 2 l2norms, up to 2 memsets and 3 kernels.
  - `state_passing` runs H_v × D_STRIPS CTAs: 384, 128 or 64 at H_v 48, 32 or 16. That is 1.13 waves, 75% of SMs and 38% of SMs respectively.
  - The next chunk's loads are issued only after the MM2 barrier.
  - The fused one-CTA-per-head path is never selected on the 5090; it was measured only on DGX Spark (3.9-5.4×).
  - Baseline correction: BENCHMARKS' ~0.6 µs/token is a gap against vLLM, not the total.
    - FLOP floor: ≥ 1.9 µs/token for the 27B.
    - Public 5090 numbers: FlashInfer 2.4 µs/token, FLA 6.6 µs/token.
    - So the GDN core is ~3-8% of a 27B prompt round, and more on 0.8B/2B.
- **Boundary and switch.**
  - Zero-code first: `SUROGATE_SERVE_GDN_PREFILL=fused|staged`.
  - Then a varlen entry in `chunked/launch.cu` with `cu_seqlens` and per-sequence state slots, as a fork TU through F0, with a hook in `text_context_impl.h` (ledger topic).
  - New switch value: `SUROGATE_SERVE_GDN_PREFILL=varlen`.
- **Measure.**
  - `sinfer_gated_delta_net_bench --chunked-only` at H_v 16/32/48 and T 2048/4096, staged vs fused, with FlashInfer and FlashQLA reference rows.
  - SUROGATE_SERVE_ROUND_TIMING on the 27B at 100 users 2048/16, and on 0.8B/2B prefill.
- **Memory axes.**
  - peak: the staged workspace (W/U/v_new ~75 MB plus h_chunk ~50 MB per 2,048-token segment) disappears with the fused path.
  - alloc: varlen sizes the workspace for NT_max.
  - tput: prompt tok/s.
- **Success.**
  - GDN core within 1.3× of FlashInfer's 5090 time.
  - Prompt round −2% or more on 27B, −3% or more on 2B.

### R20. Host overheads per row

- **Evidence.**
  - `getenv("SUROGATE_SERVE_LORA_DEBUG")` runs inside the per-row loop (138-217 ns per call, measured on the analysis host).
  - Duplicate checks with `std::find` are O(B²).
  - `preview_state` copies per row per round; `PREFILL_TIMING` getenv runs per finish unit.
  - Total ~18-28 µs plus 3-8 µs per round at B=128.
- **Boundary.** A behaviour-preserving ledgered edit to `program_impl.h` and `frontend.cpp`: static const flags and a reusable lane bitmask. No switch.
- **Measure.** SUROGATE_SERVE_ROUND_TIMING boundary, preview and append segments at B=128 on Qwen3-0.6B rollouts; nsys CPU sampling of the worker thread.
- **Memory axes.** alloc: host heap allocations per round from DecoderState copies go to 0 (count with malloc counters).
- **Success.** Host gap from sync to the next launch −20 µs or more at B=128; rollout tok/s +0.5% or more.

### R21. Longest-first causal tile order for the prompt kernel

- **Evidence.**
  - `q_block = blockIdx.x` ascending and head-major, so the longest tiles start last.
  - Tail simulation, Qwen3-0.6B with a 2K prompt: 96 vs 66 units at 2 CTAs/SM, 70 vs 66 at 3 CTAs/SM.
  - ptxas (clang PTX) gives 170-172 registers, likely 2 CTAs/SM; verify on the nvcc build.
  - FA4 2603.05451 §3.3 measured 4-8% (MHA) and 7-14% (MQA) via FA3 on H200.
  - Bit-identical.
- **Boundary and switch.** A 1-D grid remap plus a prologue mapping function in `gqa_attention_prompt_cached_launch` and `gqa_attention_prefill_bf16.cuh` (ledger topic). Switch: `SUROGATE_SERVE_GQA_PROMPT_ORDER=lpt|asc` (new).
- **Measure.**
  - `cuobjdump --dump-resource-usage` for the register count.
  - Bench d128-h32-kv8 and d128-h16-kv8 with `--tokens 2048,8192`, at `--context 0` and at 2048 (continuation chunks).
  - ncu `sm__warps_active` and `lts__t_sector_hit_rate`.
- **Memory axes.** None.
- **Success.** Cold 2K kernel time −5% or better at 2 CTAs/SM; outputs bit-identical; no regression on continuation chunks.

### R22. Split-row multi-CTA sampler with fused LSE

- **Evidence.**
  - `sampling_full_vocab_kernel` runs one CTA per row with 256 threads. Each iteration does one LDG.E.U16 and is not unrolled; ~110 dynamic instructions per token (64-bit splitmix and log1pf) give ~70-150 µs.
  - On top of that: `sampled_logprob` ~16 µs, `score_kernel` 20-30 µs (logprobs=true maps to top_logprobs=0), and 13 route kernels always launched for ~35-50 µs.
  - Realistic GRPO rounds are KV-bound, so the share is ~2-3%, or 4-10% at contexts ≤ 300 tokens.
  - FlashSampling's group Gumbel-max is exact in distribution.
- **Boundary and switch.** A new header in `fork/native/serve`. One ledger topic for the hooks in `sampling.cu` and `decode_impl.h`, where `sampled_logprob` and `score` read its outputs. Switch: `SUROGATE_SERVE_SAMPLER_SPLIT_ROW=1` (new). Phases:
  1. Keep the existing per-token key, so tokens are bit-identical. Compute two LSEs (raw logits and temperature-scaled) when T ≠ 1; the rank needs a second pass from L2.
  2. Group Gumbel-max, with a versioned RNG stream.
  3. Fold the route kernels in.
- **Measure.** nsys `cuda_gpu_kern_sum` per round under GRPO args (T=1, top_k=-1, logprobs=true) vs greedy, at B=64 and 128, with contexts of 300 and 1500 tokens; rollout tok/s.
- **Memory axes.** rows × chunks × 16 B of partials (negligible); alloc none.
- **Success.**
  - Phase 1: bit-identical tokens and logprobs within 1 FP32 ulp.
  - Sampler tail ≤ 40 µs per round at B=128; rollouts +2% or more.

### R23. Batched finish, then finalize inside the round

- **Evidence.**
  - `launch_mixed_round` sets `.finalize=false`, so each prompt gets its own unit:
    - an eager `sample_from_hidden` (~15 launches);
    - for logprob rows, an observer sync, then `score_logprobs_batch` (`cudaMallocAsync`, pageable H2D, kernel, D2H, stream sync, `cudaFreeAsync`), then a device sync.
  - That is 3 syncs per prompt, and only one unit runs per worker iteration.
  - In a burst, ~10 × 200-token prompts per 2048-token window means 3.5-9 ms of serial units. Steady state ~1%; GRPO step < 0.4%.
  - The existing multi-segment finaliser runs a separate head GEMM with a single sampling config.
- **Boundary and switch.**
  - Phase 1 (M): a new batched-finish module in `fork/native/serve`, hooked into the eager-finish path of `concurrent_executor.h`. Gather all ready tails, run one head GEMM, one sampling launch with per-row configs, one batched score, one sync.
  - Phase 2 (M-L): finalize inside the round.
  - Switch: `SUROGATE_SERVE_BATCHED_FINISH=1` (new).
- **Measure.** SUROGATE_SERVE_ROUND_TIMING "finish ms/count"; TTFT p50/p99, ITL and prompt tok/s at 32-64 users × 200-512-token prompts with 32-token outputs; a 256-prompt GRPO step-start burst.
- **Memory axes.**
  - alloc: the per-prompt `cudaMallocAsync`/`cudaFreeAsync` pair disappears (count via nsys `cuda_api_sum`).
  - life: the gathered tail buffer lives for one unit.
  - peak: + N × hidden in BF16, negligible.
- **Success.** ≤ 0.15 ms per finished prompt amortized; TTFT p50 −10% or better in the burst workload; ≤ 1/N syncs per prompt.

### R24. 8-bit prompt attention for BF16-cache models (gated by R5)

- **Evidence.**
  - The BF16 body is capped at 209.5 TFLOPS, and it is bound by non-MMA work (GB10 profiling note; PATCHES #20 found an MMA-type swap "alone gains ZERO"). Reaching SA2++ rates needs its pipeline structure too.
  - SA2++ 2505.21136 on a 5090 (hd128, causal): FA2 147-209 TOPS vs Sage2++ 388-609. That is 2.64× and 2.54× at 1K and 2K, 2.7-2.9× at 4K-32K. Quantization time is excluded. The only LLM result is Llama3.1-8B, PPL +0.007.
  - FP16 accumulate requires P_r × V_r ≤ 2047, so V needs per-channel scaling inside the kernel; with a paged cache that adds per-block dequant (~13% of MMA time).
  - The existing opt-in FP8-cache path measured 1.54× on GB10 but costs +1.5-2.6% PPL.
- **Boundary and switch.** A new 8-bit prompt-kernel header in `fork/native/serve`, selected in `gqa_attention_prefill.cu` (ledger topic). Switch: `SUROGATE_SERVE_PROMPT_ATTENTION=q8` (new; default off).
- **Measure.**
  - ncu of the current body: `sm__pipe_tensor_op_hmma_cycles_active`, barrier stalls, issue-slot utilisation.
  - Bench d128-h32-kv8 bf16 and fp8 with `--tokens 2048,8192 --context 0`: TFLOP/s against 209.5, 419 and 838, with the clock recorded.
  - WikiText PPL and NIAH on Qwen3-0.6B and 8B.
- **Memory axes.** None persistent; quantization happens in staging.
- **Success.** ≥ 1.8× TFLOP/s over the BF16 body; PPL +0.2% or less; 0.6B 2K prefill −10% or better.

### R25. Varlen persistent prompt kernel for multi-prompt rounds

- **Evidence.**
  - Each segment and layer pays a prompt kernel, an append and a `set_i32_scalar`, and the packed path refuses capture. For Qwen3-0.6B with 8 × 256 tokens that is 672 launches per round, with grids of 64 CTAs; ~3.8 → ~0.6-0.7 ms per round.
  - The host mutex cost is negligible.
  - Existing pieces: `gqa_kv_append_batch_launch` already takes `table_rows`, and `GqaPrefillBatchMetadata` reads `table_rows[0]`.
  - FlashInfer's plan/run split needs one host plan step per round.
- **Boundary and switch.** A work-list mode in the prompt kernel (template flag) and a plan builder in the round metadata, as a fork TU with a hook in the packed path of `wrapper/gqa_attention.cpp` (ledger topic). Switch: `SUROGATE_SERVE_GQA_VARLEN_PROMPTS=1` (new).
- **Measure.** nsys of surogate-engine on Qwen3-0.6B with 64 concurrent 256-token prompts and SUROGATE_SERVE_ROUND_TIMING=1: attention segment time per packed round, `sm__warps_active` of the prefill grids, launches per round, launch gaps.
- **Memory axes.**
  - peak: a fixed-capacity tile list in the round metadata (small); if the round becomes capturable, report graph count and memory.
  - alloc: none per round.
- **Success.** ≤ 1 ms per 8×256 round on 0.6B; launches −80% or more; per-tile outputs bit-identical.

### R26. Windowed deferred GDN fold for ordinary decode

- **Evidence.**
  - Record mode already exists today and is used only across MTP verify:
    - `recurrent_bf16_body` reads the base state, applies pending records in registers (`fold_pending`) and appends records without writing the state;
    - `launch_replay_fold_pending` writes it.
  - Traffic falls from 2C to (1 + 1/W)C. On the 27B at W=8: ~84 vs 144 MiB per lane-step, i.e. up to 8-11% of a 100-lane step.
  - LeapQuant measured per-window BF16 only ~4% faster than per-step on B200; the 5090 is more bandwidth-bound, so the gain should be larger there.
  - ReplaySSM is slower at B=1 (0.41-0.88× kernel), so keep the per-step kernel at small B. DeltaLog found M=8 best for GDN and keeps its update log in FP32.
- **Boundary and switch.** Extend the pending slots in `GdnReplayRecordSpec` to all lanes and add `recurrent_window_kernel<W>` as a fork TU. Preemption, snapshot and prefix-checkpoint paths must fold first. Switch: `SUROGATE_SERVE_GDN_WINDOW=W` (new; 0 means off).
- **Measure.**
  - ncu `dram__bytes_read.sum` and `dram__bytes_write.sum` of the decode kernels at B=64 and 100 (Qwen3.8-27B NVFP4, Qwen3.5-4B).
  - Decode-heavy 128/512 at 100 users.
  - State MSE against FP32 over 64K tokens; the R3 long-generation eval.
- **Memory axes.**
  - peak: record planes add ~16.8 KB per token-layer on the 27B, ~6 MB per lane at W=8; report the reservation at 128 lanes.
  - life: records live until flush (≤ W steps).
  - alloc: preallocated.
- **Success.** Decode +5% or better at 100 lanes on the 27B; no regression at B ≤ 8; state MSE ≤ 1/10 of per-step BF16.

### R27. Activation quantize fused into producer norms

- **Evidence.**
  - Block-FP8 `quantize_blocks_kernel` runs standalone per projection. A GDN layer has 3 fusable quantizes (input RMSNorm, gated norm, post-mixer `rmsnorm_add`); an attention layer has 2.
  - The norm must still write its BF16 output for other consumers, so the saving is one launch plus one read.
  - The fused kernel must match `quantize_warp_block` bit for bit (amax/448, SATFINITE) and produce the k-block-major scale layout.
  - Gemma NVFP4's `linear_projections` quantizes once per projection; quantizing once per group is the first win.
  - On SM120, block-FP8 at T ≤ 4 uses the BF16 GEMV and never quantizes.
  - #109 saw end-to-end insensitivity to small kernel savings.
- **Boundary and switch.** A new fork TU in `fork/native/serve` plus one call site per runner (ledger topic). Switch: `SUROGATE_SERVE_FUSED_NORM_QUANT=1` (new).
- **Measure.** nsys count and total time of the quantize kernels per round at 5-64 tokens on Qwen3-8B-FP8 and Gemma 3 12B NVFP4; bitwise output comparison.
- **Memory axes.** None new; the fused kernel writes into the GEMM's activation workspace.
- **Success.** Aux kernels per round −40% or more; tok/s +2% or more; outputs bitwise equal.

### R28. Run-time group size for the small-T kernel

- **Evidence.**
  - GroupSize drives compile-time choices beyond the row division: `kWidthOneWarps`, Br/RowWarps, the width guards, the resident row warps, the int8 schedules, and `gqa_partial_acc_index`. Effort is M-L.
  - The registry's invariant is that head counts are compile-time, so this needs a separate geometry type. FlashInfer's code uses `uint_fastdiv` for the group size.
- **Boundary and switch.** A `GqaRuntimeGroup` geometry in a fork TU, with a wrapper branch placed before `generic_takes`. Switch: `SUROGATE_SERVE_GQA_RUNTIME_GROUP=1` (new).
- **Measure.** Bench d128-h28-kv4 and d128-h32-kv8 with the runtime kernel against the compile-time kernel and the generic kernel.
- **Success.** Within 5% of the compile-time kernel.

### R29. Parallel per-expert scan (router tile later)

- **Evidence.**
  - `scan<<<1, kExperts>>>` walks 512 tiles twice at 4096 tokens, ~30-40 µs per pass. It runs on the FP8, W8, Q4 and GGML routes, not on TRT-LLM NVFP4.
  - The router re-reads its input once per 16-row tile. Counting the weight, an all-rows tile cuts L2 traffic 2-4×, not 17×. The 5090 BF16 compute floor is 20.6 µs.
  - A fused top-k must keep `select_count`'s outputs and tie-breaking, because GRPO routing replay reads `ids`.
- **Boundary and switch.** Widen the scan to 1024 threads (4 lanes per expert), keeping the `route_job_*` outputs and the documented barrier, in a fork TU. Switch: `SUROGATE_SERVE_MOE_PARALLEL_SCAN=1` (new).
- **Measure.** nsys per-layer time of router, select, scan and gather; a property test that offsets are identical to the serial kernel on random routings.
- **Success.** Scan ≤ 5 µs; Q4/GGML MoE prefill +1% or more.

### R30. LRU of decode-graph executables per (B, band)

- **Evidence.**
  - `install_graph_profile` instantiates and uploads on every profile switch, and `graph_profiles_through` never sets `topology_class`.
  - In the GRPO tail that is up to one rebuild per B per step: 0.1-0.4 s per step and 1-10 ms ITL spikes.
  - In-place executable update was rejected for correctness after the 2026-08-29 battery (23% bad answers).
  - Foundry measured 31-69 ms to construct a vLLM graph, so measure surogate's real cost.
- **Boundary and switch.** `topology_class` encodes (band, B) lazily with a K-per-B cap, in `program_impl.h` (ledger topic). Switch: `SUROGATE_SERVE_GRAPH_BANDS_PER_B=K` (new).
- **Measure.** nsys CUDA API count and time of `cudaGraphInstantiate`/`cudaGraphUpload` on the worker during a 256-rollout Qwen3-1.7B step; ITL p99; SUROGATE_SERVE_PREFILL_GRAPH_LOG.
- **Memory axes.**
  - peak: executable memory × K × B; measure the `cudaMemGetInfo` delta per instantiation.
  - alloc: instantiations per step.
- **Success.** ≤ 5 instantiations per step after warm-up; ITL p99 −50% in the tail; graph memory ≤ 200 MB.

### R31. GDN single-user chain: PDL and norm fold

- **Evidence.**
  - The recurrence is a plain launch and gated RMSNorm is a separate kernel.
  - The 0.98× row is with MTP off. With MTP, surogate runs at 107.3 tok/s, ~1.5× vLLM, on `recurrent_record_kernel`; its pending-fold reads depend on earlier kernels, so only base-state tiles may be prefetched.
  - The predecessor kernel does not trigger dependents.
- **Boundary and switch.** `pdl::launch_dependent` in `recurrent.cu`, with the state loads issued before the wait. Bisect with the existing `SUROGATE_SERVE_NO_PDL`.
- **Measure.** `nsys --cuda-graph-trace=node` gaps between proj, recurrence, norm and out-proj for 1-user decode, MTP on and off.
- **Success.** +0.5% tok/s or better, outputs bitwise identical.

### R32. Stash GDN replay intermediates for the backward

- **Evidence.**
  - With recompute on, the GDN forward chain runs 3 times.
  - The replay's eager stack restore (`compiled_ops_execute.cpp`) frees g_cum, Ai, w, u, h, v_new and the normalized q/k just before the backward recomputes them.
  - The stash is ~0.2 GB (0.8B) or ~0.45 GB (9B) per layer at 8k tokens, live for one layer.
  - Gain ~1-2%, more under FP8/FP4.
- **Boundary and switch.** Turn the replay's stack restore into a planned arena slot (upstream `compiled_ops_execute.cpp`, ledger topic); the GDN op uses the stash if present. Switch: `SUROGATE_GDN_STASH_REPLAY=1` (new).
- **Measure.** nsys gdr_* backward kernels; tok/s.
- **Memory axes.** This is explicitly a lifetime-vs-peak trade: + stash bytes live through one layer's replay-to-backward window, as a planned slot with no per-layer malloc.
- **Success.** +1% tok/s or better with peak +0.5 GB or less on 0.8B; gradients bitwise identical.

## 3. Refuted or downgraded items

**Refuted and dropped:**
- S6 lazy (τ-slack) online-softmax rescaling: the prompt kernels already skip the rescale exactly with an `__any_sync` vote; τ-slack adds under 2% and conflicts with e4m3-P saturation.
- Per-step INT8/E4M3 GDN state (S1's proposal): LeapQuant measured AIME 87.9 → 14.6 (FP8) and 7.1 (INT8); STEPQuant's rowwise INT8 gave 80.60 → 71.86 on Qwen3.8-27B.
- "Low-bit KV doubles resident rollouts vs FP8": NVFP4-K + FP8-V gives 1.28×; full MXFP4 K+V gives ~1.6-1.7×.
- 2506.05433 (Prefix Grouper) cited as "no faster than FA2 varlen": DualKV's Table 12 shows it is faster in most configurations; its real drawback is 3-5× memory.
- 2311.01282 cited for the decode cp.async ring: its §4 double buffering is for flat GEMM, so it is dropped from R9; PAT remains.
- FlashInfer's 13-17% cited for n>1 prefill dedup: that number measures composable-format attention and peaks at n=4.
- FlashInfer Fig. 8 cited as proof of head-fusion gains: the GQA bars tie, and the paper credits its scheduler and tile size.
- H1 "if FP4 with FP32 accumulate is half rate, the overhead doubles": the whitepaper lists 1676 TFLOPS dense.

**Downgraded, attention-kv:**
- S1 decode slowdown: 1.2-1.75× (not 1.3-1.9×). The 30K-turn floor is ≥ 30 ms because instruction issue is the bound.
- S2 "existing e4m3 kernels halve prompt attention": the cut is ~35% (1.54× on GB10), and the FP8 cache costs +1.5-2.6% PPL on pure-attention 0.6B.
- S2 FP16-accumulate P·V in the FP8-cache path: raw e4m3 V reaches 448 against an overflow bound of |v| ≤ 4.6, and the kernel is issue-bound; keep only the FP22 probe (R5).
- S2 TMA warp-specialized double-buffered prompt kernel: an L refactor with unmeasured gain; the lazy-rescale part already exists.
- S3 shared-QKᵀ ("seesaw") resident decode tile: the MMA-latency-bound premise is unmeasured, and after R9 the tile is near DRAM-bound; ncu first.
- S3 persistent stream-K-style decode: absolute partitions remove stream-K's mechanism, and FlashInfer's scheduler is not batch-invariant.
- S3 NVFP4-K / FP8-V KV: on SM120 an NVFP4 K forces an FP4 Q (mxf4nvf4); the papers show 1.06× decode over FP8 FlashInfer at best.
- S4 POD fused prefill+decode: the two kernels' shared memory and block shapes differ, and rounds are linear-dominated at ≤ 8K; L effort.
- S6 seesaw D=512 prompt kernel: needs an FP8 cache, while Gemma defaults to BF16 KV (which does not fit 99 KB); only long-context Gemma benefits.
- Cascade (Hydragen) shared-prefix decode attention: FlashInfer measures +9% at n=8 and ~0 at n=64, and 8 groups' prompt pages (~32 MB) fit in 96 MB of L2; measure DRAM bytes per group vs per lane first.
- 8-bit training attention (SageBwd): the numbers are misattributed (Table 8 vs Table 3) and come from a Triton kernel on a 4090; Graphcore's FP8 P/V shows +0.090 validation loss and is bundled with NVFP4 projections.
- MpFA 2.08×: B200 tcgen05; against FP8 FlashInfer it is only 1.06×.

**Downgraded, gemm-moe:**
- H2 folding the shared expert into the routed grouped GEMM: M-L across the converter, runner, FP8 path and decode, with numerics changes; R14 gets most of the gain.
- H3 dropping the FP8 gather by emitting `column_token` from the scan: the scan never sees per-assignment data; use one thread per assignment instead (S, ~10 µs per layer, H100 only).
- H4 DeepGEMM PDL with a weight prologue: H100 only, ≤ 0.1-0.3 ms per round, and the producer restructure is M.
- H4 per-segment epilogue stores: DeepGEMM's BLOCK_N of 80-192 does not divide the segment boundaries; ≤ 0.78 ms per mixed round; H100 only.
- H5 SwiGLU + FP8 requant in the grouped epilogue: L, not M (N=256 tile, accumulator registers); H100 only.
- H5 folding the top-k reduce into the combine: valid S, but H100 FP8 route only (~1%), not the 5090 TRT-LLM route.
- H5 contiguous M-grouped DeepGEMM: padding adds ~50% rows and the descriptor cost is unmeasured; H100 only.
- H6 "every linear, 8.1 → 5.4 ms": only ~36% of 8B bytes reach that path, so ≤ 10-15% (R15).
- H7 compiling N/K into the DeepGEMM AOT tiles: H100 only; the decode-width gain is under 6%; ~50-90 extra instantiations.
- H7 FP8 head and FlashSampling epilogue: NVFP4_ALL already emits an FP8-row head; head precision flips greedy outputs (2609.26621: 7-30% exact agreement); needs one epilogue per backend.
- MonoMoE megakernel: WGMMA plus atomicAdd (non-deterministic); an SM120 port is L/XL.
- Native MXFP8 on SM120: the fix landed in CUTLASS 4.5.1 on the CuTe DSL side (C++ since 3.9), and the UE8M0 accuracy cost for FP8 is untested.
- RaMP histogram tactic selection: its `.item()` host sync breaks graph capture.
- Whole-step decode megakernels: XL, and gains recede with batch size (2610.35336).

**Downgraded, scheduling:**
- n>1 fork-after-prefill: the API and orchestrator cost is real, and the R12 hold gives the same prefill and KV saving.
- FlashSampling LM-head epilogue: ~20 µs per round once R22 lands, and needs a custom head GEMM per backend.
- Removing empty route waves with conditional nodes: folded into R22; SWITCH nodes exist from CUDA 12.8 but their overhead is unpublished.
- Qrita pivot fallback: needed only on certificate misses, at 0.1-0.25 ms; keep the sort as the fallback until the miss rate is known.
- Plan-buffer frontier band: a low-band round would launch the max-band grid in every layer, and PolyGraph is a synthetic prototype; L for a throughput-neutral gain.
- Conditional SWITCH over bands: unmeasured overhead; R30 is simpler.
- Speculative GRPO rollouts with logprobs: dense Qwen3 has no MTP, EAGLE-3 needs an in-domain drafter, n-gram drafting lost in NeMo-RL, and R6 must come first.
- DualKV trainer-side prompt dedup: realised 1.04× at P=1K/R=256, and micro-batch co-location is limited on 32 GB.
- FP32 logprob path for RL: MiniMax-M1 says the mismatch did not appear in small dense softmax models; measure the correlation first.
- KV4 rollouts: the KV-bound regime came from ~5.5K-token responses on H200, and unprotected INT4 loses 5.24 points.
- Seer context-aware scheduling: its 89% includes divided rollout and KV migration across instances; the single-engine gain is unmeasured.
- SLOWeave and DLFP adaptive controllers: one is simulation, the other a negative result; R11 uses a static rule.

**Downgraded, linear-attn:**
- T2 MoE rollout-parity GDN (token-sequential FP32 forward, local-memory adjoint): not reachable on one 32 GB 5090. MoE GDN hybrids start at 35B-A3B, and native colocate rejects QLoRA and non-BF16. Its chunked-backward split remains a valid S fix for multi-GPU; mirror KDA's tf32x3.
- T2 register-resident parity forward and the T2 drop-parity + R3 + CIS arm: off-target, for the same reason.
- FlashQLA backend (T1/T4): its backward is 0.61-0.99× FLA on packed 8-sequence shapes, it needs TileLang JIT, and its WARMUP_TOL is an approximation.
- T3 folding the q/k L2 norm into the chunk kernels: FLA does not fuse it, only the backward recompute goes away; better as a conv epilogue.
- T4 saving A_inv (FLA's save/recompute split): 0.5-1.5%; R32 is the better variant.
- T4 multiplication-only Neumann inverse (2606.06034): NPU-specific; solve_tril is ~1% of GDN FLOPs.
- LeapQuant 8-bit window state: XL, no public code, and the kernel gain over per-step BF16 is ~1.5× (4.25× is against FP32); do after R26.
- S3 intra-card context parallelism: +50-100% FLOPs on split parts, unmeasured.
- S3 split-issue pipelining of `state_passing`: small, and double buffering loses the second CTA per SM.
- TreeWY draft trees: not yet a throughput win; ReplaySSM beat it on chains.
- DASC INT8 GDN checkpoints: the 8.11× is KDA; head-wise GDN gets 1.1-1.6×, and INT8 alone is ~1.9× vs BF16 checkpoints.
- SketchSSM: approximate and calibrated; GDN is 5.02× on B300 (not 5.22×); after R26.
- ReplaySSM dual precision: never under MoeRolloutParity.

## 4. Already well optimised in this group (do not redo)

- **Attention and KV:**
  - FA3 one-warpgroup decode tiles and FA3 varlen prompts on Hopper; per-round attention metadata (#110).
  - Window block skip; QSA gather.
  - Resident-query tile with one-tile-ahead prefetch for 256/512-wide FP8 decode.
  - FP8-cache e4m3 Q·Kᵀ and pipelined e4m3 P·V (110 → 169 TFLOP/s on GB10).
  - Exact O-rescale skip by warp vote in the prompt kernels.
  - Absolute-position decode partitions started widest-first, so results are batch-invariant.
  - BF16, e4m3 and int8 codecs fused into append and staging, with no standalone quantize kernels.
  - Packing of < 64-column lanes across prompts (#14).
  - Host KV tier and prefix cache: contiguous cudaMemcpy2DAsync on its own stream, event-fenced, no host sync.
- **GEMM and MoE:**
  - DeepGEMM AOT tiles with the cost model and 2-CTA stand-ins (0.96× the CUTLASS tiles).
  - Padded k-block-major scales; staged chained GEMMs (#108); fused SwiGLU-quantize (#106); residual seeded into the accumulators.
  - Rewritten decode GEMV; the Hopper narrow tensor-core kernel; I2F-free W8 dequant; the W8 wgmma head.
  - Registered 27B NVFP4 ladders (GEMV, small-T, TMA W4A4, and the whole-program 256×128 prompt tile); the generic-K NVFP4 GEMV in `linear`; the GB10 size-based tile rule (#134).
  - MoE: device-side expert offsets; GEMV and swapped tiles for narrow rounds; Marlin for Q4; the int8 tile for GGML experts; TRT-LLM fused MoE with a per-bucket tactic cache and deterministic finalize; graph-capturable offload slot cache.
  - The converter's `_same_divisor` guard against fused NVFP4 parts with different scales.
- **Scheduling and sampling:**
  - Pure decode rounds fully graph-captured with device-side chaining (bursts of up to 8).
  - Multi-prompt packed and mixed rounds; on-demand KV with a watermark.
  - Prefix reuse: retained lanes, radix archive, host tier, shared-prefix forks.
  - Sampler: greedy route with 16-byte loads (12 µs at 64 columns); resident-grid route skipping; vectorised `sampled_logprob`; Gumbel full-vocabulary draw for untruncated rows; FP32 multi-block candidate route for top_k ≤ 20.
  - Detokenization and stop matching at ~1.4% of wall time at 100 users; HTTP thread pool, 1024 backlog, Nagle off; tokenization off the worker thread.
  - Measured and rejected upstream, so not to be retried: launch/consume overlap (#56) and in-place graph-executable update (correctness).
- **Linear attention:**
  - Serve GDN decode and prefill kernels: swizzled shared memory, cp.async, 72-82 registers, 0 spills.
  - Projection, conv, SiLU and split fused per quantization route; in-place state update with no ping-pong slot; opt-in rewrite checkpoints (#73); record/stash/fold for MTP.
  - T_inv kept in shared memory; W stored pre-interleaved for ldmatrix.
  - BF16 state storage (+10-13%, #57) is fast, but its accuracy is under audit (R3).
  - Training: document isolation is correct; the conv already takes a single `cu_seqlens` launch; KDA's graph-safe device varlen metadata is the design to copy for GDN; kernels are autotuned per (H, K, V).