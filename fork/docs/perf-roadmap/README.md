# 性能路线图：单张 RTX 5090（SM120）

本目录保存 2026 年 10 月对本仓库做的一次效率研究的结论：三个组各一份按收益排序的路线图（英文），以及这份中文总览。研究时没有 GPU，下文所有 GPU 上的收益和行为都是代码阅读、计数和算术推出的估计，没有在 5090 上测过；每一项都写明了确认它要跑的 5090 测量。

| 文件 | 内容 |
|---|---|
| [train-native.md](train-native.md) | 原生训练路径：executor 与 DSL、注意力与长上下文、kernel 与 loss、MoE、LoRA/QLoRA、低精度 recipe、优化器与 offload |
| [serve.md](serve.md) | 服务引擎：注意力与 KV、GEMM 与 MoE、调度与采样（含投机解码）、线性注意力与 SSM |
| [python-rl.md](python-rl.md) | Python 与 RL：GRPO colocate 循环、原生 SFT 的数据与步路径、实验训练器 `fork/experimental`、跨栈的主机同步与分配 |

三份英文路线图保持研究结束时的原文，只做了这些改动：行号统一指上游提交 282f9901（fork 的 merge base）；研究在分支基点 488e5f00 上读代码，488e5f00 中 fork 已改过的上游文件里有几处行号按 488e5f00 计，已逐一核对并换算为 282f9901；fork 自有文件（`fork/experimental`、`fork/native`）的行号指 488e5f00；本分支之后的提交又改了其中许多文件，所以在 HEAD 上请按文中写的符号名查找。提议新增、仓库中并不存在的文件改成文字描述；研究会话临时目录中的证据脚本已不存在，文中注明“未保留”；本分支已修复或实现的项标为 **Status:**，写明提交和 ledger topic，每份文件开头有一张状态表。

## 研究了什么、怎么做的

三个组共 15 个子系统：

- train-native（7 个）：executor-dsl、attention-longctx、kernels-loss、moe-ep、lora-qlora、recipes-lowprec、optim-offload-dist。
- serve（4 个）：serve-attention-kv、serve-gemm-moe、serve-scheduling-spec、linear-attn-ssm。
- python-rl（4 个）：grpo-loop、train-data-python、experimental-backend、cross-cutting-sync-alloc。

目标负载是所有者关心的那些：单张 5090 32 GB 上 0.5–8B 稠密模型的 LoRA/QLoRA/全参微调、约 10B 总参数的 MoE、GRPO（colocate），以及与之相邻的服务场景。

每个子系统经过四步，后两步是对抗式的核查：

1. 找热点：沿默认配置实际走到的路径读代码，每个热点写明位置、机制、为什么在热路径上（哪些配置会走到）以及代价估计（字节数、FLOP、launch 数）。
2. 找技术：针对每个热点查找可用的技术与文献（论文在 alphaXiv 上查阅原文），并另列热点之外的机会。
3. 代码怀疑者：逐条重读引用的代码，给出 confirmed / uncertain / refuted 的判定，并修正机制、触发条件和数量级。
4. 文献怀疑者：逐条核对引用（论文是否存在、是否真的这样说、数字是否属于同一实验），同样给出判定与修正。

被代码怀疑者否定的热点和技术、被文献怀疑者否定的引用都删掉，其余的按两位怀疑者的修正改写，再按“收益 × 置信度 × 适用性 ÷ 工作量”排序（各组的打分细则写在各自文件开头）。被删掉或降级的条目各组都有一节逐条列出理由。原始数据量：热点 110 个（三组 53 / 29 / 28），技术 283 项（143 / 70 / 70），代码判定 408 条（confirmed 243、uncertain 149、refuted 16），文献判定 869 条（confirmed 740、uncertain 116、refuted 13）。这些原始数据（每个子系统的热点、研究和两位怀疑者的判定）没有放进仓库。

没有 GPU 时用到的证据：代码阅读与计数；在 4 vCPU 的 KVM 虚拟机（Intel Xeon family 6 model 207，AVX-512）上的 CPU 微基准；对 kernel 算术与调度的 CPU 模拟；把上游 kernel 截取出来编译成主机 C++ 的探针；以及按 sm_120a 的编译命令对改动过的翻译单元做的编译检查。凡是只能在 GPU 上确认的说法都标为未验证，并给出 5090 上的检查方法。

## 每组最值得做的几项

“置信度”和“工作量”沿用各组文件中的记法（H/M/L；S/M/L/XL）；预期收益都没有在 GPU 上测过。

### train-native

负载记号：D0.6 / D4 / D8 = Qwen3-0.6B / 4B / 8B LoRA（rank 16，全部线性层，seq 2048，B=2，GA=4，打包，即上游基准表的配置）；F8 = Qwen3-8B 经 `cpu_training` 的全参微调；M10 = 约 10B 总参数、BF16 base 的 MoE LoRA（LFM2-8B-A1B 一类）；G-MoE = 7–16B BF16 MoE 的 `grpo-colocate`。注意：这里的百分比都相对上游基准表（2026-02-18）中的旧步时，先做下面的零代码实验 Z1 重新定基线。

| 项 | 内容 | 预期收益（5090，负载） | 置信度 | 工作量 | 本分支 |
|---|---|---|---|---|---|
| #1 | 不能捕获的 MoE 整步不再每步多跑一遍 eager 前向+反向 | 1.9–2.0 倍 tok/s（非 QLoRA、非预量化的 M10 SFT） | H | S | 已实现，默认开启（T20） |
| #2 | `cpu_training` 全参微调用 GA=1 和放得下的最大 micro-batch | 同样每步 token 数时步时约快 1.6–1.9 倍（F8，前提是 B=8×2048 放得下） | M | S | 未做（只改配置，见 Z4） |
| #3 | MoE GRPO colocate 用 `r3_corrected`（关闭 parity，R3 路由回放，TIS/CISPO）替代逐位 rollout parity | learner 步至少快 1.5 倍（G-MoE） | M | S | 未做 |
| #4 | 重计算时不再把 MoE 中间张量复制进持久缓冲 | 峰值约 −15 GB（M10，BT=4096），每 micro-step 少约 31 GB 的 D2D（约 2–4%），可能决定 M10 放不放得下 | H | S | 未做 |
| #5 | 长文档的确定性 dQ（先用现有的非确定性开关，再做整数网格的 dQ 后端） | 打包 24k+8k 行上注意力反向快 4.8–9.6 倍；2k 文档时 −2–3% | M | M | 未做（先跑 Z5） |
| #6 | ≤4B 用 `recompute: false`，之后做按显存的 `recompute: auto` | D4 1.15–1.3 倍，D0.6 1.05–1.2 倍 | M | S | 未做（先跑 Z3） |
| #7 | `cpu_training` 的 CPU 步与主机梯度加法 | F8 每步 −0.8–1.3 s；若保留 GA>1，主机加法的 2.3–2.8 倍惩罚降到 1.1–1.3 倍 | H | S | 部分实现（T13、T14、T15） |
| #8 | FP8 冻结权重缓存分层（缓存 amax + 融合转置；只缓存前向） | D8 FP8 每 micro-step (A) −37 ms（约 5%），(B) −70 ms（约 9%） | H | S | 未做（先跑 Z6） |
| #9 | CPU 上的 SIMD 8-bit AdamW | F8 每步 −1.4–2.3 s，主机内存约 −49 GB | M | M | 未做；它原本顺带修复的 C1 已由 T21 另行修复 |
| #10 | QLoRA：缓存命中时跳过反量化、直接从存储格式转 FP8 | D8 QoFP8/QoNF4：有缓存 −23–28 ms（3%），无缓存 −75–80 ms（9–10%） | H | S | 未做 |
| #11 | 回放时只执行反向会读到的 op | D8 FLOPs −8.1%、D0.6 −5.2%，结果逐位相同 | H | S-M | 未做 |

### serve

| 项 | 内容 | 预期收益（5090，负载） | 置信度 | 工作量 | 本分支 |
|---|---|---|---|---|---|
| R1 | 注册 128 宽的 GQA 32q/8kv、40q/8kv 与 MHA 32q/32kv | Qwen3-8B GRPO rollout 解码（16 lane × 4K，BF16 KV）：注意力每步 7–14 ms 降到约 5.5–6 ms，解码 1.1–1.6 倍；30K 历史上 40 token 的一轮 30–180 ms 降到 3–5 ms | H | S | 已实现，默认关闭（T16） |
| R2 | T=1 的通用 K NVFP4 GEMV 接入 linear_add、attn_input、gdn_input | Qwen3.5-9B NVFP4，1 用户，MTP 关：线性层每 token 约 6.3 ms 降到约 2.5 ms，约 1.5–2 倍 tok/s | M | S | 未做 |
| R3 | GDN 解码状态按 BF16 存储的精度审计（门槛） | 不直接提速；论文在 Qwen3.5-9B 上测得 AIME −15.8，若确认需改回 FP32（让出 #57 的 +10–13%） | M | S | 未做 |
| R4 | 用现有 tile 给 5090 定 NVFP4 的尺寸规则 | Qwen3.5-9B / Gemma 3 12B NVFP4，2–16 用户与 MTP 验证：解码 +20–40%（GB10 上的类比实测 +32%） | M | S | 未做（先做零代码 A/B） |
| R5 | SM120 的 MMA、累加器与 FP64 微基准（门槛） | 不直接提速；决定 R6、R7、R15、R24 | H | S | 未做 |
| R6 | nucleus/宽采样和投机接受改用补偿的 FP32 归一化 | Llama-3-8B FP8 默认采样（T 0.6、top_p 0.9）：每轮 −0.67–0.8 ms，约 +15–18% tok/s；Llama-3.2-1B/3B +30–100% | H | S-M | 未做 |
| R7 | SM120 上 5–16 token 的 block-FP8 用 `mma_kernel` | Qwen3-8B-FP8，5–16 用户与验证轮：o/down 快 1.5–1.8 倍，每轮 +10–20% | M | S | 未做（先做零代码 A/B） |
| R8 | 打包训练的 varlen 分块 GDN，SFT 重新打开 graph | Qwen3.5-0.8B 打包 LoRA SFT 示例 +7–13% tok/s；GRPO 打包 +2–5% | H | M | 未做 |
| R9 | SM120 用细粒度的解码分区（D≤256 加两槽 ring） | Gemma 4 12B，BF16 KV，1 用户，32K：全局层约 185 µs 降到约 40 µs，约 1.25–1.5 倍 tok/s | M | S | 未做 |
| R10 | 语法掩码改由线程池填充，一次锁页 H2D | Qwen3-8B 的 JSON schema 与工具调用，16–64 用户：每轮遮住 0.4–16 ms 主机时间（+5–30%） | M | S | 未做 |
| R11 | 轮次预算策略（TTFT/TPOT） | 32 用户 × 2048 prompt / 16 输出：TTFT p50 接近 vLLM 的 0.8–1.0 倍，TPOT 按比例变慢，吞吐不变 | H | S-M | 未做（先做零代码 sweep） |

### python-rl

负载记号：W-MoE-GRPO = 约 10B 总参数 qwen3_moe 一类的 LoRA GRPO（`grpo-colocate`，16 题 × 8 个 rollout，`max_tokens` 1536，Python 的 `SharedModelServer`）；W-0.6B-GRPO / W-8B-GRPO = Qwen3-0.6B / Qwen3-8B LoRA colocate，原生 C++ SharedServer；W-0.6B-SFT = Qwen3-0.6B LoRA BF16，seq 2048，B=2，GA=4，打包，整步 graph；W-8B-FFT-cpu = Qwen3-8B `cpu_training` 全参微调，GA=4；W-EXP-* = 实验训练器的 5090 示例配置（Qwen3-4B，seq 2048，B=1，GA=8），W-EXP-OIB-8B 是不 tie 的 Qwen3-8B optimizer-in-backward。

| 项 | 内容 | 预期收益（5090，负载） | 置信度 | 工作量 | 本分支 |
|---|---|---|---|---|---|
| R1 | 实验训练器：融合的随机舍入优化器（每步 8B 约 16.4 万个小 kernel） | W-EXP-OIB-8B 步快 2.7–3.7 倍；4B 全参 GA=8 −9–14%；LISA 约 −2% | H | M | 未做 |
| R2 | `cpu_training`：主机梯度加法并行化并移出预取 stream | W-8B-FFT-cpu 步 −30–50% | M | S | 第一步已实现，默认关闭（T14） |
| R3 | 实验训练器的 5090 LISA/LoRA 配置关闭 gradient checkpointing | 1.25–1.35 倍 tok/s（修正后的显存估计放得下） | H | S | 未做（零代码） |
| R4 | 训练路径服务：增量去 token 化，按存活 session 分轮 | W-MoE-GRPO：约 1k token 的 completion 下 rollout 约 1.2 倍（模拟），周期 −10–15%；非 Qwen 的 1B 稠密模型 rollout 约 2 倍 | H | S | 已实现（T11 默认开启，T12 默认关闭） |
| R5 | LISA 冻结 embedding/head 并截断反向 | W-EXP-LISA 每 microbatch FLOPs −22%（步约 −15–20%） | M | S | 未做 |
| R6 | colocate 并发按空闲显存自动设定，停止补发多余请求 | W-0.6B-GRPO 周期 −13–20%（C 16→64，解析估计） | M | S | 未做 |
| R7 | 实验训练器：GA 窗口内打包 + FA2 varlen | W-EXP-LoRA 在平均 300–600 token 的数据上 1.3–2 倍 tok/s | M | M | 未做 |
| R8 | 原生 SharedServer 睡眠前丢弃保留的前缀 KV | W-0.6B-GRPO 周期 −8–14%（C=16）；W-8B-GRPO −2–3%；怀疑者修正后每步约省 0.8–3 s | M-H | S | 已实现，始终开启（T29） |
| R9 | 实验训练器：Liger 融合 RMSNorm、RoPE、SwiGLU | W-EXP-LISA/LoRA 步 −8–15% | M | S | 未做 |
| R10 | 实验训练器：关闭 PEFT 的输入 dtype 转换 | W-EXP-LoRA 每 microbatch 40–70 ms（−8–14%） | M | S | 未做 |
| R12 | 原生 SFT 的 eval 打包 | W-0.6B-SFT 默认 `eval_steps=100` 时墙钟 −8%，基准配置（`eval_steps` 25）约 −20% | H | S-M | 未做；它依赖的缺陷 0.2 已加保护（T18） |

各组之间有重叠：train-native C2 即 python-rl §0 #1；train-native #24 即 python-rl R32；train-native #2、#7、#9、#22 与 python-rl R2、R16、R19、R27 都针对 `cpu_training` 的主机端步；train-native #3 与 python-rl R13 是同一个方向；train-native #18 与 python-rl R21 都是 LM head 跳过被忽略的行；serve R8 与 python-rl R28 都是 GDN 的打包训练。

## 先在 5090 上跑的零代码实验

这些实验不需要改代码或重新构建（个别需要临时构建的已注明），可以在一次 5090 会话中跑完，结论直接决定上面多项的取舍。

| 实验 | 决定 |
|---|---|
| Z1：在当前代码上重跑上游基准表，用 nsys 按 GEMM、量化、LoRA、注意力、CE、拷贝、空闲分类 | 所有百分比的基线 |
| Z2：非 QLoRA 的 MoE LoRA SFT，默认 vs `SUROGATE_FORK_FORCE_GRAPH_WARMUP=1` vs `use_cuda_graphs: false` | 验证 train-native #1（T20） |
| Z3：Qwen3-0.6B/1.7B/4B LoRA 在 bf16/fp8/fp4 下设 `recompute: false` | train-native #6 |
| Z4：`cpu_training` 的 Qwen3-8B 全参微调，B=8/GA=1 vs B=2/GA=4，记录峰值显存 | train-native #2 |
| Z5：打包的 24k+8k 行上设 `SUROGATE_FLASH_ATTN_VARLEN_BWD_NONDETERMINISTIC=1`；在 W-0.6B-SFT 上同一开关预计约 −5%，比较 500 步 loss 曲线 | train-native #5 的上限，python-rl R24 |
| Z6：Qwen3-8B FP8 LoRA，在 stderr 中查 `[fp8 weight cache] device 0: auto:`；A/B `cpu_training: true` 加 `SUROGATE_RESIDENT_LAYERS=36` | train-native #8、#16 |
| Z7：cuBLASLt FP8 在 8192³ 上的吞吐和 SASS，对比 CUTLASS SM120 单位 scale 的 block-scaled kernel | train-native #13、#14 |
| Z8：`deviceQuery` 的 `asyncEngineCount`、`nvbandwidth` 双向主机-设备带宽、PCIe 通道检查 | train-native #7、#12、#22，以及 T14 的 D2H stream |
| Z9：`SUROGATE_DEBUG_MATMUL_HEURISTICS=1`，看 LM-head dX（m=1024, n=512, k=151936）选的 split-K 和 tile | train-native KL6 |
| Z10：`JEV_SAVED_OFFLOAD_TRACE=1`，每层 saved-tensor 缓存搬运的字节数 | train-native optim H6 |
| Z11：从现有 SFT 日志统计梯度裁剪触发率（`NormHost[0] > max_grad_norm`） | 是否做 STV |
| Z12：基准数据中 SFT 目标为 −100 的 token 比例 | train-native #18、python-rl R21 |
| `SUROGATE_SERVE_NVFP4_CUTLASS=off`、`5`、`rows` 在 1/4/8/16 用户下的引擎 A/B | serve R4 |
| `SUROGATE_SERVE_FP8_BLOCK_MMA=1` 的 A/B | serve R7 |
| `--max-num-batched-tokens` 2048/3072/4096 的 sweep，TTFT 与 TPOT 一起报告 | serve R11 |
| `SUROGATE_SERVE_W8_PIPELINED=1` 的 A/B | serve R15 |
| `SUROGATE_SERVE_GDN_PREFILL=fused` vs staged | serve R19 |
| 现有的 `--kv-cache-dtype fp8` | serve R18、R24 |
| 去掉 capacity/8 上限的临时构建，加 `SUROGATE_SERVE_SHARED_PREFIX_MIN_TOKENS=64` 与 `SUROGATE_SERVE_SHARED_PREFIX_SLOTS=64` 的代理运行 | serve R12（需要临时构建） |
| 实验训练器 `fork/experimental/examples/5090-lisa.yaml` 与 `5090-lora.yaml` 设 `gradient_checkpointing: false` 跑 20 步，记录 `peak_cuda_bytes` | python-rl R3 |
| W-0.6B-GRPO 手动设 `max_num_seqs` 16/32/64（R8 已经落地，先决条件已满足），记录每轮 batch 和 rollout 墙钟 | python-rl R6 |
| `use_cuda_graphs: false` 下 `lmhead_drop_ignored_rows` 开与关的 eager A/B | python-rl R21 |

serve 组另有两个应同时启动的门槛实验，它们需要少量新代码：R5（SM120 指令速率表）和 R3（GDN 状态 FP32 与 BF16 的长生成评测）。

## 发现的正确性缺陷及其在本分支的状态

热点核查与实现过程中一共发现下列缺陷。“已修复”“已加保护”的都有 CPU 测试在改动前失败、改动后通过，GPU 上的行为都未验证；上游缺陷的 5090 检查写在 [upstream-errata.md](../upstream-errata.md) 对应的一节（链接见“说明”列）；实验训练器一项是 fork 自有代码，它的检查写在该行。

| 缺陷 | 来源 | 本分支状态 | 提交 | ledger | 说明 |
|---|---|---|---|---|---|
| 打包 SFT 的整步 graph 在没有行内边界的步上把 `cu_seqlens` 设成 `{0,B·T}`，B≥2 时第 b 行注意到前面各行 | train-native C2 = python-rl §0 #1 | 已修复 | d3d36439 | T17 | [errata](../upstream-errata.md#打包-sft-的整步-cuda-graph-在没有行内边界的微步上跨行或跨文档注意fork-已修复ledger-t17) |
| 捕获时没有边界的微步被捕获成稠密注意力，之后同一微步中的文档被忽略 | 核查 C2 时发现 | 已修复 | d3d36439 | T17 | 同上 |
| 无文档的捕获之后，第一个带文档的步在没有 warmup 的捕获中 `cudaMalloc` | CPU 重放测试发现 | 已修复 | d3d36439 | T17 | 同上 |
| pad 模式从不关闭，`validate()` 一个微步的文档多于捕获上限时静默丢掉多出的文档 | python-rl §0 #2 | 已加保护：上限以内照旧，超过上限的 eager 调用改用精确几何 | d3d36439 | T18 | [errata](../upstream-errata.md#整步-graph-的-pad-模式不会关闭超过上限的-eager-调用丢掉尾部文档fork-已加保护ledger-t18) |
| MoE 路由统计在重计算回放中再累加一次（`aux_loss`、`z_loss` 报成 2 倍） | train-native C10 | 已修复；C10 中“warmup 也重复累加”一半被否定（micro-step 0 的 `begin_step` 清零） | 66647915 | T19 | [errata](../upstream-errata.md#moe-路由统计在重计算回放和分块训练的-kv-sweep-中重复累加fork-已修复ledger-t19) |
| 分块训练（`sequence_chunks > 1`）的 KV sweep 中每个块再计一次路由统计 | 评审 T19 时发现 | 已修复（`train_step_graphed`）；`step_chunked` 未覆盖，记在 errata | 67ec9ef7 | T19 | 同上 |
| `cpu_training` 全参微调不把 CPU AdamW 的矩写进 checkpoint，恢复时在设备上分配 8-bit 状态（8B 约 16 GB）且矩从零开始 | train-native C1 | 已修复：新的 checkpoint 格式 v1；`SUROGATE_FORK_CPU_ADAMW_SAVE_STATE=0`、`SUROGATE_FORK_CPU_ADAMW_ALLOW_MISSING_STATE=1` 可回到上游行为 | 563df25e | T21 | [errata](../upstream-errata.md#cpu_training-的全量训练不保存-cpu-adamw-的矩fork-已修复ledger-t21) |
| cuDNN SDPA 忽略 op 的 `softmax_scale`，Gemma3-27B 的全局层按 1/sqrt(128) 训练 | train-native C3 | 已修复：把 scale 传给 cuDNN；不声明 scale 的 op 逐位不变 | db4a840a | T22 | [errata](../upstream-errata.md#cudnn-注意力忽略-softmax_scalegemma3-27b-的全局层按-1sqrt128-训练fork-已修复ledger-t22) |
| 交叉熵 kernel 按 int 计算 `row * P + i`，一个分块超过 2^31 个 logits 时越界读写 | train-native C4 | 已修复 | e067f546 | T23 | [errata](../upstream-errata.md#交叉熵-kernel-按-int-计算行偏移一个-lm-head-分块超过-231-个-logits-时越界读写fork-已修复ledger-t23) |
| FP8 abs_max 与量化 kernel 的 grid-stride 下标是 int（fp8_hybrid 的 LM-head dX 路径） | 评审 C4 时发现 | 已修复 | a58ff65f | T23 | [errata](../upstream-errata.md#fp8-abs_max-与量化-kernel-的循环下标是-int一次处理约-231-个元素时越界读写fork-已修复ledger-t23) |
| NVFP4 Four Over Six 的随机舍入有偏（只在 E2M1 间距为 0.5 的区间无偏） | train-native C5 | 已修复；`SUROGATE_NVFP4_4O6_SR_LEGACY=1` 运行上游 kernel | c32c9996 | T24 | [errata](../upstream-errata.md#nvfp4-随机舍入与-four-over-six-块比例的缺陷fork-已修复ledger-t24) |
| 4/6 用未舍入的比例编码、用存储的 E4M3 比例解码，1/128 下限让小块解码放大 | 修复 C5 时发现 | 已修复；`SUROGATE_NVFP4_4O6_RN_LEGACY=1` 恢复上游 RN kernel | c32c9996 | T24 | 同上 |
| SR 的种子在 CUDA graph 捕获时固定，每次重放用同一份噪声 | 修复 C5 时发现 | 已修复；`SUROGATE_NVFP4_SR_GRAPH_LEGACY=1` 恢复；2f7b5d37 修正了它在 Debug 构建中的 ODR 问题。`fp4_backend: cudnn` 的 SR kernel 与 RHT 的种子同样固定，记在 errata，未改 | fecde777、2f7b5d37 | T24 | 同上 |
| MoE FP8 权重缓存（`SUROGATE_FP8_MOE_WEIGHT_CACHE`）不拒绝可训练的专家，优化器步之后缓存过期 | train-native C6 | 已修复：可训练专家不用缓存 | 000452cd | T26 | [errata](../upstream-errata.md#moe-的-fp8-权重缓存不拒绝可训练的专家权重fork-已修复ledger-t26) |
| GPT-OSS 预量化 LoRA 把冻结专家的输入梯度置零，前面各层只得到 LoRA 项 | train-native C7 | 已修复：默认计算 dgrad；`SUROGATE_GPT_OSS_PREQUANT_SKIP_EXPERT_DGRAD=1` 恢复上游的跳过 | d4b3ab07 | T25 | [errata](../upstream-errata.md#gpt-oss-预量化-checkpoint-的-lora-训练跳过冻结专家的输入梯度fork-已修复ledger-t25) |
| FP8 生产者侧预量化使用没有算过的 amax（当前所有配置都走不到） | train-native C9 | 已加保护：三处预量化先算 amax，当前运行逐位不变 | eb34d5e0 | T27 | [errata](../upstream-errata.md#fp8-的生产者侧预量化使用没有计算过的-amax潜在缺陷fork-已加-guardledger-t27) |
| 训练路径的 `SharedModelServer` 满载时回 429，原始 rollout 在 SDK 退避中丢失，可能丢掉整组 | python-rl §0 #3（R22） | 已修复：默认 FIFO 排队并在客户端离开时停止；`SUROGATE_SHARED_SERVER_QUEUE=reject` 恢复上游 | 6bcf5231、ae28bc19 | T28 | [errata](../upstream-errata.md#训练路径的-sharedmodelserver-在满载时拒绝-rolloutfork-已修复ledger-t28) |
| 实验训练器的 resume 指纹对 `TrainConfig` 的每个字段取哈希，新增任何字段都使旧 checkpoint 无法恢复 | python-rl §0 #6 | 已修复（fork 自有代码，没有 ledger topic）：带版本的 resume 契约，逐键比较并报出不同的键；`compile_model` 打开时比较 `gradient_checkpointing`。5090 检查（未做）：`fork/experimental/examples/5090-lora.yaml` 以同一 seed、GC 开不间断跑两次 4 步作噪声基线；再跑 2 步（`save_every: 1`）后关掉 GC、`--resume` 第 2 步的 checkpoint 跑到第 4 步；每步 loss 和第 4 步 checkpoint 中每个张量与不间断运行的差异，不应超过两次不间断运行之间的差异；再用 `5090-lisa.yaml` 重复 | 57f75813、4253b4de | 无 | [experimental/docs/training.md](../../experimental/docs/training.md#checkpoints-and-exports) |
| `lora_dropout` 文档写默认 0.05，代码是 0（同处还有 `lora_rank` 等） | train-native C11 | 已记录（fork 不改上游文档） | 2737f5b8 | 无 | [errata](../upstream-errata.md#lora-配置文档中的默认值与代码不符) |
| `AsyncStaleAdamW` 没有 C++ 调用者，注释却说已接入 dispatch-PP | python-rl §0 #7 | 已记录 | 2737f5b8 | 无 | [errata](../upstream-errata.md#asyncstaleadamw-的注释声称已接入-dispatch-pp) |
| 通用 `moe_grouped_gemm` 不产生权重梯度，Nemotron-H 全参微调从不训练 `experts_up` | 修复 C6 时发现 | 已记录，未修复 | 无 | 无 | [errata](../upstream-errata.md#moe-的-fp8-权重缓存不拒绝可训练的专家权重fork-已修复ledger-t26) |
| 上游精度文档中的 `no_fp4_stochastic_rounding` 一行：没有任何代码读取这个选项，无法关掉 SR | 修复 C5 时发现 | 已记录（fork 不改上游文档） | c32c9996 | 无 | [errata](../upstream-errata.md#nvfp4-随机舍入与-four-over-six-块比例的缺陷fork-已修复ledger-t24) |
| 标准 NVFP4 SR kernel 的 curand 子序列不含 `blockIdx.y`，相隔 128 列的元素用同一个随机数（相关但不偏） | 修复 C5 时发现 | 已记录，未修复（修复会改变 eager 结果） | 无 | 无 | [errata](../upstream-errata.md#nvfp4-随机舍入与-four-over-six-块比例的缺陷fork-已修复ledger-t24) |
| `matmul_swiglu` 的回放在已释放保存张量时只用 base 权重重算 mlp_up，丢掉 LoRA delta | train-native C8 | 否定：这条路径无法被调度（`matmul_swiglu` 没有 autodiff 规则，融合只在前向，没有 DSL 模块调用它），未改 | 无 | 无 | [train-native.md](train-native.md) |
| `cpu_training` 的数值：BF16 master 就近舍入、主机加法每个 micro-step 舍入到 BF16、选 `adamw_8bit` 时 CPU 上仍是 FP32 矩 | python-rl §0 #4 | 未处理（T14、T15 与上游逐位相同；python-rl R19 未开始） | 无 | 无 | [python-rl.md](python-rl.md) |
| `fork/experimental/examples/5090-oib.yaml` 用 tie 的 Qwen3-4B，optimizer-in-backward 拒绝它，配置无法启动 | python-rl §0 #5 | 未处理 | 无 | 无 | [python-rl.md](python-rl.md) |

## 本分支实现的快速改进

只去掉可证明是冗余或重复、结果不变的工作的改动默认开启（并留有恢复上游的开关）；其余默认关闭。全部没有在 GPU 上运行过，下表的 5090 检查都未做；完整的设计、CPU 测量和检查步骤见 [performance.md](../performance.md) 中链接的一节。

| 改动 | 来源 | ledger | 开关 | 提交 | 5090 检查 |
|---|---|---|---|---|---|
| [不能捕获的整步只在第一步跑 warmup](../performance.md#不能捕获的整步只预热一次) | train-native #1 | T20 | 默认开启；`SUROGATE_FORK_FORCE_GRAPH_WARMUP=1` 恢复上游；`ep_size > 1` 仍每步 warmup | f73ad034 | LFM2-8B-A1B 或 OLMoE-1B-7B，BF16 LoRA，B=2，GA=4，至少 50 步，默认 vs `SUROGATE_FORK_FORCE_GRAPH_WARMUP=1`：tok/s 至少 1.8 倍，每个优化器步的 GemmGrouped 数减半，100 步 loss 与 MoE 统计逐位相同，allocator 峰值在 ±1% 内 |
| [`cpu_training` 步开始不再 memset 主机梯度](../performance.md#步开始的清零t13) | train-native #7 | T13 | 默认开启（主机梯度逐字节相同）；`SUROGATE_FORK_HOST_GRAD_MEMSET=1` 恢复上游的 memset | 796fdfa0、e2f56d21、4a983cc6 | `examples/training/cpu-offload.yaml` 改为 Qwen3-8B、20 步（C8），在同一构建上设与不设 `SUROGATE_FORK_HOST_GRAD_MEMSET=1` 各跑一次（设的一次先连跑两次确认可复现）：loss 与梯度范数逐位相同，不设时 `step_seconds` 至多少 memset 的时间 |
| [主机梯度加法向量化并在 OmpTeam 上并行](../performance.md#主机梯度加法t14) | train-native #7、python-rl R2 第一步 | T14 | `SUROGATE_FORK_HOST_GRAD_ACCUM=1`，默认关闭 | be5b9191、54954e9b | C8 设与不设：逐位相同；nsys 中 micro-step 1–3 每层反向不超过 micro-step 0 的 1.3 倍；记录回调线程与 team 线程的 `Cpus_allowed_list` |
| [块梯度的 D2H 与主机加法放到自己的 stream](../performance.md#主机梯度加法t14) | train-native #7、python-rl R2 第一步 | T14 | `SUROGATE_FORK_GRAD_D2H_STREAM=1`，默认关闭，与上一项一起打开 | be5b9191 | 先记 `asyncEngineCount`；两个开关都开：nsys 中块梯度 D2H 与权重预取的 H2D 在不同 stream 上，结果与只开上一项时逐位相同 |
| [CPU AdamW 向量化](../performance.md#cpu-adamwt15) | train-native #7 | T15 | `SUROGATE_FORK_CPU_ADAMW_SIMD=1`，默认关闭；基线 ISA 有 FMA 的构建忽略它 | be5b9191、ce939c53 | C8 改为 5 步、`save_steps: 5`，设与不设：loss 逐位相同，权重分片 sha256 相同，日志中没有 “is ignored”，最后一个反向 kernel 到下一步前向之间的空隙缩短 |
| [训练路径服务的非流式请求只在结束的一步构造文本](../performance.md#非流式请求只在结束的那一步构造文本t11) | python-rl R4 | T11 | `SUROGATE_GRPO_DETOK=final`，默认；`incremental`（另加 ByteLevel 窗口检查停止字符串）默认不用；`per_token` 恢复上游 | afe88d26、2586d3ac | 走训练路径的约 10B qwen3_moe LoRA `grpo-colocate` 至少 50 步，`per_token` vs 默认 vs 默认加 T12：比较 `train/pack_duration_ms`、`train/cycle_duration_ms` 中位数与有效 batch，MoE rollout 预期至少 1.15 倍；`max_concurrency=1` 时与 `per_token` 逐位相同 |
| [decode 轮次按存活 session 决定大小](../performance.md#按存活-session-决定-decode-轮次的大小t12) | python-rl R4 | T12 | `SUROGATE_GRPO_LIVE_DISPATCH=1`，默认关闭 | e7ee206a | 同上一项的第三组：尾部轮次的有效 batch 约为 min(16, 存活数)，尾部的 pack 时间降低 |
| [原生 GRPO SharedServer 睡眠前释放保留的 KV](../performance.md#原生-grpo-sharedserver-睡眠前释放保留的-kvledger-t29) | python-rl R8 | T29 | 始终开启（只去掉没人读的备份与恢复）；`SUROGATE_SHARED_SLEEP_KEEP_KV=1` 恢复上游顺序 | deec1701 | W-0.6B-GRPO 与 W-8B-GRPO 各 20 步，带与不带该开关：0.6B、C=16 时每步至少省 1 s；每步的 `cudaMallocHost`/`cudaFreeHost` 降到约 0；发布后第一轮的前缀命中与 rollout token 不变 |
| [服务端注册 128 宽的 GQA 32q8、40q8 与 MHA 32q32](../performance.md#服务端-gqa-形状注册) | serve R1 | T16 | `SUROGATE_SERVE_GQA_FORK_SHAPES=1`，默认关闭（关闭时 MHA 与注册前一样被拒绝）；int8 KV 始终走通用 kernel | 7a45b4f7、2e5f766b | `make serve-test-build` 后在服务构建目录运行 `ctest -R '^sinfer_gqa_(attention|fork_shapes)'`，四个条目都通过；同一构建只切换开关跑 `sinfer_causal_softmax_attention_bench`；成功标准：16×4K BF16 解码至少 1.15 倍，30K 历史上 40 token 一轮不超过 10 ms，int8 KV 仍可服务 |

另外两项性能相关的改动属于缺陷修复，已列在上一节：T28 的 FIFO 排队（python-rl R22，CPU 模拟中一批从 155.6 s 降到 74.1 s），以及 T19 在回放中少跑的统计 kernel。T13–T15 的 CPU 测量来自 `fork/native/bench/cpu-optimizer/step.sh`：它用构建目录中 `compile_commands.json` 记录的命令编译真实的 `cpu_adamw.cpp` 与 `cpu_step_kernels.cpp`，再按 ISA 做逐位检查，复现方法见 [performance.md](../performance.md#复现)。部分做了但未完成的：train-native #7 中的逐层 GPU 范数（只在 GA=1 时成立）推迟；python-rl R2 第二步（K 个暂存槽加专用累加线程）推迟；serve 的 F0 挂接点已有（`fork/native/cmake/fork_native.cmake` 的 `_surogate_fork_attach_serve()`），但目前只挂到 `sinfer_ops` 和 GQA 测试目标。

## 已经优化得很好的部分（不要重做）

各组文件末尾一节有逐条清单，这里只列要点：

- 训练 executor：含优化器的整步 CUDA graph；cap-and-pad 的 varlen 捕获（打包变化不需要重新捕获）；不重计算时零拷贝保存；arena 着色；捕获前缩小 DSL stack；FP8 转置权重缓存；单 kernel 的梯度清零；fork 的 T1、T3、T4、T6–T8。
- 注意力：原地融合的 QK-norm+RoPE；FA2 的交错 QKV 与不补齐的 LSE；局部窗口跳块；按形状缓存的 cuDNN graph；按稠密几何选 dQ 切分数（batch 不变，上游在 5090 上测得只比 atomic 慢 3.2%）。
- kernel 与 loss：x128 的融合残差+RMSNorm；softcap 融进 CE；保存 LSE；确定性的被屏蔽行压缩（GRPO 上 13–21%）；FP8/FP4 的前向与 W^T 权重缓存；只选逐位相同算法的 cuBLASLt autotune；128 MB 的 SM120 workspace。
- MoE：每个投影一次持久的 CUTLASS grouped launch（确定、与打包无关）；基数排序建索引；重计算复用每层的主机 offset；融合的 grouped 专家 LoRA；GRPO 路由回放。
- LoRA/QLoRA：vec8 小 rank 的 B 累加；scaling 折进前向 α；每个优化器步同步一次工作副本；β 累加的 LoRA 梯度；放得下时预先填好 FP8/FP4 缓存。
- 低精度 recipe：单遍的 FP8 delayed scaling 前向量化；冻结权重只量化一次的 FP4 缓存；NVFP4 的 α 融进 CUTLASS epilogue；RHT 不在默认 CUTLASS 路径上。
- 优化器与 offload：LoRA 的多张量 flash 8-bit AdamW（8B rank 16 每步不到 1 ms）；双缓冲的权重预取；冻结 master 的 FP8/NVFP4 流式与部分常驻；fork T5 的 OmpTeam 线程数与有序确定的范数。
- 服务：Hopper 上的 FA3 解码与 varlen prompt；按绝对位置分区、最宽者先开始的解码（batch 不变）；BF16/e4m3/int8 编码融进追加与暂存；纯解码轮完全 graph 化并在设备上串联（最多 8 轮连发）；多 prompt 打包与混合轮；前缀复用（保留的 lane、radix 归档、主机层、共享前缀 fork）；采样器的贪心路线与 Gumbel 全词表抽样；去 token 化与停止匹配在 100 用户时只占墙钟约 1.4%。上游测过并否定、不要再试的：launch 与 consume 重叠（#56）、原地更新 graph executable（正确性问题）。
- 线性注意力：服务端 GDN 解码与预填充 kernel（swizzle 的共享内存、cp.async、72–82 个寄存器、无溢出）；按量化路线融合的投影、卷积、SiLU 与切分；原地更新状态；MTP 的记录/暂存/折叠。BF16 状态存储（+10–13%）很快，但精度正在审计（serve R3）。
- GRPO colocate：trainer 与服务零拷贝共享 base 权重；策略发布留在 GPU 上；logprob 前向、GRPO dloss 与反向融合在 `step_grpo_native` 中，没有 old-policy 或 reference 前向；原生服务的连续批处理、CUDA graph、GPU 采样与组内前缀复用；elastic VMM 的 KV 睡眠/唤醒保持已捕获的 graph 有效；fork 的 T6/T7。
- 原生 SFT 步：带 cap-and-pad 的整步 graph（GSM8K 上每次运行 0–1 次重新捕获）；释放 GIL 的多线程 tokenizer，训练集按流打包、几乎没有补齐；步间空隙约 1 ms（<0.3%），不值得给数据加载做双缓冲；稳态下每步没有分配。
- 实验训练器：反向之前释放 bf16 logits 及其 FP32 副本；每个专家的因子保留为设备标量，不逐参数 `.item()`；OIB 在更新后立即释放梯度；B=1 且无补齐时 SDPA 走 flash。
