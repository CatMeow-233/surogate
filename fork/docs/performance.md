# Fork 性能说明

本文记录 fork 的性能改动、测量方法和开关。这些内容原先写在上游用户文档 `docs/guides/performance.md` 和 `docs/guides/metrics.md` 中；为让这两个文件与上游保持一致、同步时不产生冲突，现集中到这里。各节与 [ledger](../patches/ledger.toml) topic 的对应关系：

| 节 | ledger topic |
|---|---|
| [避免冗余工作与交接](#避免冗余工作与交接) | T1、T3、T4、T6、T7、T8、T19 |
| [不能捕获的整步只预热一次](#不能捕获的整步只预热一次) | T20 |
| [CPU 优化器并行](#cpu-优化器并行) | T5 |
| [cpu_training 主机端步](#cpu_training-主机端步) | T13、T14、T15 |
| [LoRA 融合输入梯度](#lora-融合输入梯度) | T2 |
| [GRPO 周期计时](#grpo-周期计时) | T6 |
| [cpu_training 的 AdamW 状态 checkpoint](#cpu_training-的-adamw-状态-checkpoint) | T21 |
| [GPT-OSS 预量化 LoRA 的专家输入梯度](#gpt-oss-预量化-lora-的专家输入梯度) | T25 |
| [GRPO 训练路径 rollout 服务的主机开销](#grpo-训练路径-rollout-服务的主机开销) | T11、T12 |
| [服务端 GQA 形状注册](#服务端-gqa-形状注册) | T16 |

尚未实现的改进项、要先在 5090 上跑的零代码实验和研究中发现的缺陷见[性能路线图](perf-roadmap/README.md)（2026 年 10 月无 GPU 的效率研究，按收益排序，标出各项在本分支的状态）。

优化应针对完整的训练迭代，包括输入获取和通信。孤立的 kernel 改进只有在缩短关键路径时才有帮助。比较改动时，保持模型、recipe、序列长度、batch、硬件拓扑和质量目标不变。

## 找出受限资源

对于一个包含 `F` 次运算、`H` 字节 HBM 流量、`L` 字节跨设备链路传输的阶段，乐观下界是 `F / compute_rate`、`H / HBM_bandwidth` 与 `L / link_bandwidth` 三者中的最大值。依赖关系、启动开销、流水线气泡和不完全的重叠都会使实际时间变长。请使用目标 GPU 上实测的持续速率，仅凭规格峰值不够。现有的 SOL 估计衡量的是计算侧，并不能说明显存或通信已经饱和。

用 Nsight Systems 采集一段预热后的短训练区间，然后在不挂 profiler 的情况下重新做基准测试。在选择要优化的 kernel 之前，先检查 GPU 空闲间隙、CUDA API 等待、内存拷贝、集合通信和各 stage 之间的不均衡：

```bash
nsys profile -o train-trace <training-command>
nsys stats --report cuda_gpu_kern_sum train-trace.nsys-rep
nsys stats --report cuda_api_sum train-trace.nsys-rep
nsys stats --report cuda_gpu_mem_size_sum train-trace.nsys-rep
```

对已确定的 kernel，用 Nsight Compute 检查实际达到的显存带宽、tensor core 利用率、occupancy 和寄存器溢出。区分有效的模型计算与 checkpoint 重计算。对于 GRPO，还要把[周期计时](#grpo-周期计时)与计算步吞吐量对比，并按墙钟时间跟踪 policy lag、KL 和 reward。`turn_diagnostics` 有意使用较慢的诊断路径；比较的各次运行应使用相同设置。

## 避免冗余工作与交接

训练运行时无需修改配置即应用以下优化：

- FP8 反向使用缓存的转置权重，不再额外量化一份用不到的未转置副本。全参微调仍通过现有的 weight manager 刷新可变权重缓存。
- 在 recipe 驱动的 matmul 反向中，冻结或未请求的权重梯度不再需要一个权重大小的临时缓冲区。这降低了 stack/arena 需求；并不意味着每个 matmul 都少了一次 CUDA 分配调用。
- 已就绪的 GRPO batch 会被立即打包。接收为空时仍保留轮询退避和取消检查。
- 打包文档的边界改用 NumPy 运算检测，不再用 Python 逐 token 遍历。只读源数组使用 `ravel` 视图；填充后的原生输入保留独立的可写存储。按 rank 分配 batch 时保持顺序，不再反复从列表头部删除元素。
- GRPO trainer 中单 worker、单 run 的 packer 把打包好的 micro-batch 直接交给本地 loader。外部 orchestrator 的流量仍使用配置的文件系统或 ZMQ 传输。只有一个槽位的有界交接可防止覆盖尚未消费的 batch，现有的 NumPy 转换保留 mask、文档位置、teacher/turn 元数据和 routing replay 数据。独立或多 worker 的 packer 仍走传输路径。
- Dispatch-PP 把两个常驻的边界张量作为一批读取：先分配主机端目标缓冲区再排队拷贝，整批只等待一次。这去掉了一个多余的显式完成栅栏；传输仍使用现有的主机边界协议，不是 P2P 或零拷贝。
- 通用 FP8 MoE 反向不再分配只有前向 recipe 才使用的输入量化缓冲区。这节省了与 `routed_tokens * input_features` 成正比的对齐 scratch 及其统计量。它降低的是 stack 需求；已预留的 arena 不一定立即释放显存。
- 全参数 MoE 权重反向在每个 expert 都有 token、且每个目标元素都会被 beta 为零的 GEMM 覆盖时，跳过对整个 dW 的清零。覆盖步中，空 expert 的梯度仍保持为零；累加时从不清除已有梯度。冻结的 LoRA base 不走这条 base 权重路径。
- 重计算回放 MoE 层时，以及分块训练（`sequence_chunks > 1`）中 `train_step_graphed` 的 KV sweep 中，不再计算路由统计（每个回放或扫过的 MoE 层少一次单 CTA 的 kernel）。这同时修复了 `aux_loss` 与 `z_loss` 偏大的问题（重计算时为 2 倍；分块时第一个微步之后的每个微步再多计一次），见 [upstream-errata.md](upstream-errata.md#moe-路由统计在重计算回放和分块训练的-kv-sweep-中重复累加fork-已修复ledger-t19)。

## 不能捕获的整步只预热一次

`train_step_graphed` 在捕获整步 graph 之前，先 eager 跑一遍所有微步的前向和反向，再丢弃结果（`zero_grads`，恢复 RNG 状态），并按这一遍测得的高水位把 DSL stack 缩小一次。含捕获不安全 op 的模型（MoE 路由与 grouped GEMM、EP 的 dispatch/combine，见 `graph_executor.cpp` 的 `is_capture_unsafe_op_type`）从不捕获：每一步都走 eager 分支，`gs.captured` 一直为 false，所以上游每一步都多跑一遍被丢弃的前向+反向和一次流同步。受影响的是非 QLoRA、非预量化的 MoE SFT：`use_cuda_graphs` 默认开启，Python 侧没有为 MoE 关掉 graph 的规则。QLoRA、预量化、`offload_master`、`sequence_chunks > 1` 等已经关掉 graph 的配置不受影响（eager 分支不跑 warmup），`ep_size > 1` 保留 warmup（见下文）。发布的示例中受影响的是 GLM 的 BF16 dummy MoE 配置：`examples/sft/glm/dummy-full.yaml`、`dummy-lora.yaml` 与 `dummy-long-context.yaml`（模型由 `examples/sft/glm/create_dummy.py` 生成，4 个路由专家、每个 token 选 2 个；`recipe: bf16`、`sample_packing: true`、`ep_size` 为默认的 1，graph 保持开启）。其余 MoE 示例要么是 QLoRA 或预量化的 checkpoint，要么用 `sequence_chunks > 1` 或 `ep_size > 1`。

ledger T20：`has_capture_unsafe_ops()` 在第一次 warmup 之前就已知（graph 在 `allocate_run_state` 中编译），所以这类模型只在第一次执行的步跑 warmup，stack 照旧在那里缩小一次，之后的步跳过它。判断在 `fork/native/include/surogate_fork/full_step_warmup.h` 的 `run_warmup` 中。warmup 不留下 eager 分支会用到的状态：梯度在微步 0 清零，RNG（包括 executor 自己的状态）被恢复，loss 缓冲在微步 0 重新初始化，FP8 记录的 amax 在 `begin_step` 中清零、历史只由优化器滚动，MoE 统计在微步 0 清零。因此结果应逐位相同，这只是读代码的结论。例外是专家并行：warmup 会推进 sticky EP plan 的年龄（`ep_strategy.cpp`，`ep_plan_refresh_interval` 默认 16），跳过后 plan 按不同的节奏刷新，结果仍正确但不再逐位相同，所以 `ep_size > 1` 时每步仍跑 warmup。EP 也想跳过时，可以用上游的 `SUROGATE_DSL_GRAPH_SKIP_WARMUP=1`，它对所有模型都跳过 warmup，也不再缩小 stack。

开关：默认开启，因为它只去掉重复且被丢弃的工作。`SUROGATE_FORK_FORCE_GRAPH_WARMUP=1` 恢复上游行为，即每步都跑 warmup。

预期：设一遍前向+反向为 P、优化器为 U，一步从 2P+U（加一次同步）变为 P+U，加速 (2P+U)/(P+U)。roadmap 对 LoRA 估计为 1.9–2.0 倍 tok/s，MoE 全量微调的优化器占比更大，加速更少。峰值显存不变，因为两遍从不重叠；每遍的分配流量不变，每步减半。

尚未在 GPU 上验证。5090 检查：LFM2-8B-A1B 或 OLMoE-1B-7B 一类模型，BF16 LoRA，B=2，GA=4，至少 50 步，比较默认与 `SUROGATE_FORK_FORCE_GRAPH_WARMUP=1`：

- tok/s：期望至少 1.8 倍。
- `nsys stats --report cuda_gpu_kern_sum`：每个优化器步的 GemmGrouped kernel 数减半。
- 100 步的 loss 逐位相同，报告的 MoE 统计相同。
- allocator 峰值（`debug_memory_breakdown`）在 ±1% 内，stderr 中仍然有一次 “DSL stack shrunk before full-step capture”。

便宜的冒烟检查：用 `examples/sft/glm/create_dummy.py` 生成 dummy 模型（见 `examples/sft/glm/README.md`），以 `examples/sft/glm/dummy-lora.yaml` 跑默认与 `SUROGATE_FORK_FORCE_GRAPH_WARMUP=1` 各一次：10 步的 loss 与报告的 MoE 统计逐位相同。这只检查结果不变；dummy 模型太小，不用来测速。

零代码对照：上游的 `SUROGATE_DSL_GRAPH_SKIP_WARMUP=1` 跳过包括第 0 步在内的全部 warmup 和 stack 缩小，stack 因此会大一些。

## CPU 优化器并行

训练库直接链接 `OpenMP::OpenMP_CXX`，因此其现有的 CPU AdamW、梯度范数和向量加法 pragma 的启用与服务端 encoder 无关。该 imported target 只对 C++ 源文件施加主机编译选项，不会把 `-fopenmp` 直接传给 nvcc。安装匹配的编译器运行库，例如 GCC 的 `libgomp` 或 Clang 的 `libomp`。

### 每个并行区的线程数

`cpu_training` 的全参微调在每个 rank 线程上执行 CPU 优化器步（`AdamW8BitOptimizer::step_cpu_streaming`）：先对所有主机梯度求范数，再对每个参数调用一次 AdamW，每个张量各开一个并行区，每步两轮。rank 线程启动时用 `nvmlDeviceSetCpuAffinity` 把自己绑定到 GPU 所在 NUMA 节点的 CPU 上（`set_cpu_affinity()`，只作用于调用线程），team 线程继承这个掩码；libgomp 的默认线程数却在库加载时按整个进程的掩码算出。于是在多路或多 NUMA 主机上，默认 team 的线程比它能用的 CPU 多，并在每个 barrier 上空转。上游在 `cudaSetDevice` 之前调用 `set_cpu_affinity()`（`comm.cpp` 的 rank 线程先绑定、再构造通信器），新线程的当前设备是 0，所以每个 rank 都绑定到 GPU 0 的 CPU 上，多个 rank 共用同一组 CPU；fork 没有改这一点。

因此 `cpu_adamw.cpp` 的每个并行区都由 `fork/native/include/surogate_fork/omp_team.h` 的 `OmpTeam` 决定线程数（ledger T5）：

- 元素数少于 `kOmpMinParallelElems` = 65536 时用 1 个线程：并行区的 `if` 子句为假，循环在调用线程上执行。
- 否则用调用线程当前掩码中的 CPU 数（每次进入这样的并行区都重新读取，一次 `sched_getaffinity` 系统调用），除以掩码与它重叠、且正在并行区中或离开并行区还不超过自己共享窗口的调用线程数（含它自己），至少 1，至多 `omp_get_max_threads()`。一个调用线程的共享窗口是它两次并行区之间最长空闲间隔的 2 倍，至少 1 秒（`kOmpSharerGapFactor`、`kOmpSharerWindowNs`）。同一个 serial group 中的调用线程（rank 线程与它排队的主机函数，ledger T14，见[主机梯度加法](#主机梯度加法t14)）合起来只算一个。

team 因而不超过调用线程掩码中的 CPU 数；同一组 CPU 上的几个 rank 平分这些 CPU，绑在不相交 CPU 上的 rank 互不影响。rank 线程两步之间的空闲是一次前向/反向加梯度卸载，它的共享窗口随之变长，所以从第 3 步起，无论前向/反向多长，同一组 CPU 上的 rank 在步与步之间仍互相计入。前两步例外：第 1 步时各 rank 还没有登记，第 2 步时它们还没有步间间隔可参考，窗口只有 1 秒；前向/反向超过 1 秒时，这一步最先进入的 rank 在其他 rank 的第一个并行区之前用满整组 CPU。之后某次步间空闲超过此前最长间隔的 2 倍时（例如中途保存 checkpoint），那一步也是如此。用满整组 CPU 时，libgomp 为该 rank 创建的 worker 在并行区结束后先空转一段时间（`GOMP_SPINCOUNT`）再睡眠；每个 rank 只分到 1 个 CPU 时，它与其他 rank 争用 CPU（见下文 R=2 共用 2 个 CPU 的几行）。线程退出时注销；只连续调用过几次优化器就停下的线程（例如 Python 线程通过 `AsyncStaleAdamW` 调试接口以 `overlap=False` 运行）空闲 1 秒后不再计入，不会长期占用份额。

环境变量：

- `OMP_NUM_THREADS` 是上限，只能把线程数调低。在设置了 CPU 配额（cgroup `cpu.max`）的容器中设置它：libgomp 和 `OmpTeam` 都不读取配额。
- `OMP_WAIT_POLICY=PASSIVE` 让 team 线程在两个并行区之间睡眠而不是空转，把 CPU 让给同一组 CPU 上的 CUDA 与 NCCL 线程，代价是每个并行区多一次唤醒。阈值按 worker 已睡眠的情况选取（见下文阈值测量中“冷”的两列），在本机上 PASSIVE 与默认等待策略的步耗时相同（23.2–24.5 ms 对 24.2–28.1 ms，8M + 64×4096，R=1，绑定 2 个 CPU）。
- 不要设置 `OMP_PROC_BIND` 或 `OMP_PLACES`：libgomp 按进程的掩码计算 place，会把 team 线程绑到 rank 的掩码之外。

### 数值

逐元素的 AdamW 更新与串行构建逐位相同，与线程数无关：每个元素由一个线程按同样的算术计算。梯度范数不再用 `reduction(+)`：`ordered_block_sum` 把每个张量分成 `kOrderedSumBlock` = 16384 个元素的块，块内按下标顺序求和，块和再按块的顺序相加。结果与线程数、绑定、阈值以及是否启用 OpenMP 都无关。不超过一块的张量与上游的串行循环逐位相同；更大的张量与上游从左到右的单个累加器在 double 的末几位不同。在下表的 8M + 64×4096 负载上，相对于精确和（`math.fsum`），上游顺序的误差是 −7.9e-12，分块顺序是 −1.2e-14；裁剪用的 float 范数在上游顺序、分块顺序和修复前的 OpenMP 结果之间都相同，但在舍入边界上不能保证。

修复前的 `reduction(+)` 结果随 team 大小变化，线程多于 2 个时还取决于各线程完成的先后：在下表的 toy 负载上，同一进程中两个 rank 对相同的梯度求得的范数在 12 次 R=2 运行中有 9 次相差 1 ulp（Llama 形状的 6 次运行中没有出现），单 rank 的重复运行也在两个值之间变化。修复后每个 team 大小、每次运行、每个 rank 的结果都相同。请在目标主机上验证梯度裁剪与数值容差，以及实际的优化器步耗时。

### 测量

主机：4 vCPU 的 KVM 虚拟机（Intel Xeon，family 6 model 207，2.1 GHz，1 个 socket，无 SMT，16 GB 内存），没有 GPU，测量时与其他会话共用。“两个 socket”用两组不相交的 2-CPU 掩码模拟：rank 线程像 NVML 那样只绑定自己。代码：`cpu_adamw.cpp` 用训练库的编译命令（GCC 13.3，`-O3 -fopenmp`，libgomp）编译，链接进 `fork/native/bench/cpu-optimizer/pinbench.cpp`；该程序在每个 rank 线程上执行 `step_cpu_streaming` 的 CPU 部分（范数，然后每个参数一次 `cpu_adamw_step_bf16`，FP32 master、BF16 梯度）。负载：toy 是一个 8M 元素的张量加 64 个 4096 元素的张量；Llama 形状是 L=8、h=896、im=4864、V=32000（embedding、`lm_head` 与各层矩阵，共 75 个张量、1.78 亿元素，近似 Qwen 0.5B 但词表较小）。每格是 3 轮的中位数，每轮取最慢 rank 的步耗时中位数（toy 6 步，Llama 4 步，均不含预热步）；只在机器其余进程占用低于 0.25 个 CPU 时计时，否则重测。“每步前空闲 1.2 s”的两行在每步之前让所有 rank 先睡 1.2 秒（`pinbench` 的 `GAP_MS`），代替 GPU 上超过 1 秒的前向/反向。R=2 共用 2 个 CPU 的 4 行是共享窗口改为随步间间隔放大之后重测的（同一台虚拟机，toy 9 轮，Llama 形状 3 轮）；其余各行只有一个调用线程或掩码不相交，不受共享窗口影响，沿用第一次测量。

| 配置 | 负载 | 串行（T5 之前） | OpenMP 默认（修复前） | 修复后 |
|---|---|---:|---:|---:|
| R=1，绑定 2/4 个 CPU | toy | 43.6 ms | 1056 ms | 27.3 ms |
| R=2，两个“socket”各 2 个 CPU | toy | 46.9 ms | 32.0 ms | 25.7 ms |
| R=2，共用 2 个 CPU（上游的 GPU 0 绑定） | toy | 47.1 ms | 56.2 ms | 50.1 ms |
| R=2，共用 2 个 CPU，每步前空闲 1.2 s | toy | 49.1 ms | 56.1 ms | 51.8 ms |
| R=1，不绑定（对照） | toy | 44.5 ms | 13.1 ms | 13.3 ms |
| R=1，绑定 2/4 个 CPU | Llama 形状 | 945 ms | 1540 ms | 479 ms |
| R=2，两个“socket”各 2 个 CPU | Llama 形状 | 949 ms | 554 ms | 504 ms |
| R=2，共用 2 个 CPU | Llama 形状 | 997 ms | 1010 ms | 1007 ms |
| R=2，共用 2 个 CPU，每步前空闲 1.2 s | Llama 形状 | 1008 ms | 1057 ms | 1022 ms |
| R=1，不绑定（对照） | Llama 形状 | 925 ms | 279 ms | 255 ms |

- 修复前，绑定 2 个 CPU 的单 rank 开 4 个线程（`/proc/<pid>/task` 中 4 个线程的掩码是 0-1），两个 rank 共用 2 个 CPU 时开 8 个；修复后分别是 2 个和 3 个（第 3 个是一步开始时短暂用满 2 个 CPU 时创建的 worker，之后在 libgomp 的线程池中睡眠）。修复前在该配置下设 `OMP_NUM_THREADS=2` 得 21.1–24.3 ms，`OMP_WAIT_POLICY=PASSIVE` 得 28.0–30.9 ms；修复后设 `OMP_NUM_THREADS=1` 得 44.8–48.6 ms，与串行相同。
- 两个 rank 共用 2 个 CPU 时没有可并行的 CPU，修复前的 8 个线程比串行慢。修复后从第 3 步起每个 rank 的每个并行区都只有 1 个线程（在测量程序中逐个并行区记录线程数核对过），但仍比串行慢约 3 ms（toy 约 6%，9 轮分别是 47.8–58.2 对 46.5–48.7 ms；Llama 形状约 1%）。这个差距不来自线程数，具体来源没有单独测量。
- 每步前空闲 1.2 s 时，第一版修复的固定 1 秒共享窗口每步都会过期：每步最先进入的 rank 都用满 2 个 CPU，并留下一个空转的 worker，toy 为 57.9 ms（9 轮 55.5–60.1 ms），Llama 形状为 1060 ms，与修复前相当。共享窗口随步间间隔放大后，这种情况只出现在前两步（计时的只有第 2 步），toy 为 51.8 ms，Llama 形状为 1022 ms。
- 不绑定时 toy 修复后略慢（13.3 对 13.1 ms，3 轮分别是 11.9–14.4 对 11.8–15.3 ms）：64 个 4096 元素的张量在阈值以下串行执行。
- 修复前 toy 负载的两个 R=2 格是 5 次有效运行的中位数：第一版测量程序在 rank 的结果不同时中止，第 2 轮的这两次因此没有计时，之后又补测了 3 轮。
- 所有配置中，更新后的 master 与两份矩逐位相同（FNV 哈希一致）。

阈值的测量：K 个 N 元素的张量（共约 4M 元素），串行构建对比总是开 team 的构建（`-DSUROGATE_FORK_OMP_MIN_PARALLEL_ELEMS=0`，team 按上面的规则确定）；“热”是 libgomp 默认的等待（worker 在并行区之间空转），“冷”是 `GOMP_SPINCOUNT=0`（worker 每次都要唤醒）。每格是一次运行（6 步的中位数），串行 → 并行，单位 ms。先是 AdamW 更新的耗时：

| N | 2 个 CPU，热 | 2 个 CPU，冷 | 4 个 CPU，热 | 4 个 CPU，冷 |
|---:|---:|---:|---:|---:|
| 4096 | 17.5 → 12.9 | 16.8 → 37.0 | 16.5 → 7.7 | 18.2 → 71.0 |
| 16384 | 17.0 → 9.7 | 16.5 → 15.6 | 17.4 → 11.8 | 18.1 → 23.9 |
| 32768 | 18.5 → 9.4 | 17.5 → 12.2 | 16.3 → 6.3 | 17.5 → 14.1 |
| 65536 | 17.8 → 11.3 | 17.2 → 10.4 | 16.9 → 9.3 | 17.5 → 8.2 |
| 131072 | 17.7 → 11.4 | 17.6 → 12.1 | 17.6 → 6.0 | 18.7 → 7.6 |

再是整步（范数 + 更新）的耗时：

| N | 2 个 CPU，热 | 2 个 CPU，冷 | 4 个 CPU，热 | 4 个 CPU，冷 |
|---:|---:|---:|---:|---:|
| 4096 | 21.7 → 18.8 | 20.9 → 41.8 | 20.6 → 12.6 | 23.0 → 78.2 |
| 16384 | 21.0 → 14.0 | 20.4 → 19.7 | 22.0 → 17.4 | 22.5 → 28.5 |
| 32768 | 23.1 → 11.8 | 21.4 → 16.9 | 20.2 → 11.0 | 22.1 → 26.6 |
| 65536 | 21.9 → 14.6 | 21.4 → 13.9 | 20.9 → 11.6 | 21.6 → 12.9 |
| 131072 | 22.0 → 14.8 | 21.7 → 16.6 | 21.6 → 8.0 | 24.8 → 11.7 |

阈值按整步选取：65536 是 4 种条件下整步都快于串行的最小 2 的幂。只看更新，32768 在 4 种条件下已经都更快；但在 32768 元素、4 个 CPU、冷的情况下，范数从 4.00 ms 变成 10.97 ms，整步反而更慢（22.1 → 26.6 ms）。每格只测了一次，32768 与 65536 的差别在这台共享虚拟机的噪声范围内，选 65536 偏保守。更小的张量在 worker 已睡眠时并行反而更慢（4096 元素时更新慢 2–4 倍）。范数每个元素的计算量更小：65536 元素时，热的情况下并行更快（2 个 CPU 4.2 → 3.3 ms，4 个 CPU 4.0 → 2.3 ms），冷的情况下相对串行在 −9%（2 个 CPU，4.09 → 3.73 ms）到 +13.5%（4 个 CPU，4.08 → 4.63 ms）之间。

复现：`run.sh` 读取构建目录中的 `compile_commands.json`，`make configure` 不生成它，因此先运行 `make configure`，再运行 `cmake -S csrc -B csrc/build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`。`fork/native/bench/cpu-optimizer/run.sh csrc/build` 用其中 `cpu_adamw.cpp` 的编译命令，把同一份源码编译成带 OpenMP（`openmp`）和去掉 `-fopenmp`（`serial`）的两份对象，并运行上表的配置。用检出的源码时，`openmp` 是“修复后”，`serial` 是这份源码的串行构建（范数仍按分块顺序求和）；加上 `CPU_ADAMW_SRC=<(git show 9a6018f2:csrc/src/runtime/optimizers/cpu_adamw.cpp)` 则编译修复前的源码（T5 只链接了 OpenMP，没有改这个文件），这时 `serial` 是“串行（T5 之前）”，`openmp` 是“OpenMP 默认（修复前）”。`CONFIGS` 换成其他配置（例如阈值测量用的 `flat:N:K` 形状）。尚未在真实的多路主机上测量；GPU 主机上的端到端优化器步耗时也未测量。

### OpenMP 测试

独立的回归测试用 `fork/dev test openmp` 运行，需要 C++ 编译器和 OpenMP，但不需要 GPU 或 CUDA 头文件。`training-openmp-contract` 检查训练用的 OpenMP 辅助函数能让一个真实的双 worker 主机库生效，并得到精确的整数输出；运行时的线程数限制可能导致显式跳过。`training-openmp-team` 用同一个 OpenMP 辅助函数（`surogate_enable_training_openmp()`）编译 `omp_team.h`（该测试不设构建类型，优化级别与训练库的 `-O3` 不同），检查阈值、绑定线程的线程数及其 team 线程的掩码、共用与不相交掩码的分配规则和共享窗口，以及 `ordered_block_sum` 在 1–4 个线程下与串行分块顺序逐位相同。这两个测试都不编译 CUDA 训练库；`cpu_adamw.cpp` 中的并行区由源码 pin `fork/patches/tests/native/test_cpu_adamw_openmp.py`（`fork/dev test patches`）检查。

## cpu_training 主机端步

`cpu_training` 全参微调的主机在每个优化器步上做三件事：micro-step 0 开始时准备主机梯度；梯度累积步数 GA>1 时，从 micro-step 1 起把每层暂存到主机的 BF16 梯度加到主机梯度上；最后一个 micro-step 之后在 rank 线程上求范数并执行 CPU AdamW（线程数与范数的求和顺序见上节）。

| 改动 | ledger | 开关 | 数值 |
|---|---|---|---|
| 删去 micro-step 0 对每个主机梯度的 memset | T13 | 默认开启（去掉的是冗余工作）；`SUROGATE_FORK_HOST_GRAD_MEMSET=1` 恢复上游的 memset | 主机梯度逐字节相同 |
| 主机梯度加法向量化，并在 `OmpTeam` 的 team 上并行 | T14 | `SUROGATE_FORK_HOST_GRAD_ACCUM=1` | BF16 逐位相同；FP32 除 NaN 的 payload 外逐位相同 |
| 块梯度的 D2H 与主机加法移到 grad store 自己的 stream | T14 | `SUROGATE_FORK_GRAD_D2H_STREAM=1` | 不改变数值，只改变排队的 stream |
| CPU AdamW 向量化 | T15 | `SUROGATE_FORK_CPU_ADAMW_SIMD=1` | 基线 ISA 没有 FMA 的构建（上游的 x86-64 wheel）中除 NaN 的 payload 外逐位相同；基线 ISA 有 FMA 时开关被忽略（见“数值”） |

T13 默认开启，`SUROGATE_FORK_HOST_GRAD_MEMSET=1` 恢复上游的做法；T14、T15 的开关默认关闭，关闭时行为与上游相同。四个开关都在进程中第一次用到时读取一次环境变量，只有值恰好为 `1` 时生效。

### 步开始的清零（T13）

上游在 `DslGradStore::start_micro_step(0)` 中、micro-step 0 的反向排队之前，在 rank 线程上逐个 `std::memset` 全部主机梯度：Qwen3-8B 是 16.4 GB，桌面 CPU 的单核约 0.55–0.8 s；本机单线程 memset 约 8.5 GB/s（见下文测量），16.4 GB 约 1.9 s。这段时间里 rank 线程不排队反向，只有已经排队的前向能遮住它。而 micro-step 0 的 D2H（块梯度的 `offload_layer_grads`、其余梯度的 `offload_non_block_grads`）在任何读取之前就整块覆盖了这些缓冲。

grad store 在排队每次写入主机梯度的拷贝之前，调用 `fork/native/include/surogate_fork/host_grad_overwrites.h` 中 `HostGradOverwrites` 的 `before_copy()`，传入与拷贝相同的目标地址和字节数；`zero_rest()` 每步至多一次，只清零这一步没有任何拷贝写过的缓冲，调用点在第一个读者之前：micro-step 1 起的 `start_micro_step`，以及只有一个 micro-step 时的 `wait_all_offloads`。全参微调中没有这样的缓冲；没有设备存储的非块梯度、或某层在 micro-step 0 没有卸载时才会有。如果某个缓冲这一步的第一次拷贝只覆盖它的一部分（比缓冲短，或从缓冲中间开始），`before_copy()` 在这次拷贝排队之前就把整个缓冲清零，与上游先 memset 再拷贝的顺序相同。现在的三处拷贝都按分配主机缓冲时所用张量的 `bytes()` 整块拷贝，不会走到这一步；这条规则保证以后新增的部分拷贝结果也正确。主机梯度因此与 memset 路径逐字节相同：整块覆盖的缓冲，两种做法都是拷贝来的字节；部分覆盖的缓冲，两种做法都是在零上叠加各次拷贝；其余缓冲，两种做法在第一次被读之前都是零。清零不会与 GPU 竞争：`zero_rest()` 只清零这一步没有拷贝写过的缓冲；`before_copy()` 只在这一步还没有拷贝写过该缓冲时清零，并且在它对应的拷贝排队之前完成；上一步的拷贝已在上一步的 `wait_all_offloads` 中完成；读这些缓冲的主机函数和优化器都在清零之后才排队或运行。主机梯度唯一的读者是 `step_cpu_streaming`，它在 `wait_all_offloads` 之后读；源码 pin `fork/patches/tests/native/test_cpu_step_hooks.py` 检查这一点，并检查 grad store 中写入主机梯度的三处拷贝都先调用 `before_copy()`。`fork/dev test openmp` 中的 `cpu-step-contract` 在随机的 GA（1–4）、覆盖缺口、比缓冲短或从中间开始的拷贝、共用的 tied 缓冲和没有设备拷贝的缓冲上重放这个协议，与 memset 路径逐字节比较。

开关 `SUROGATE_FORK_HOST_GRAD_MEMSET=1`（`host_grad_memset_enabled()`）恢复上游的 memset，用于在同一个构建上做 A/B，也是怀疑 T13 时的退路。`allocate_streaming_buffers` 以 `upstream_memset` 构建 `HostGradOverwrites`：`add()` 只按 `mCpuGrads` 的遍历顺序记下每一项的 `Data` 与 `bytes()`（不合并共用的缓冲，也不检查重叠，与上游一样不会因此抛异常）；`begin_step()` 仍在 `start_micro_step(0)` 中、`zero_all` 之后，即上游循环原来的位置，逐项 memset，与上游的循环相同；`before_copy()` 与 `zero_rest()` 不清零任何字节。上游文件中只有构建记录器的那一行多了这个参数，调用点不变。`cpu-step-contract` 在开关打开时同样重放上面的协议，与 memset 路径逐字节比较，并确认记录器没有清零任何字节；它还检查只有恰好为 `1` 的值生效、进程中只读取一次（读过之后再改环境变量不起作用）。源码 pin 检查 grad store 构建记录器时传入开关、登记的是 `mCpuGrads` 的每一项，以及头文件中开关打开时的提前返回。

### 主机梯度加法（T14）

GA>1 时，从 micro-step 1 起，每层梯度先 D2H 到所有层共用的暂存缓冲，再由 `cudaLaunchHostFunc` 排队的主机函数 `accumulate_staged_layer` 加到主机梯度上。上游是单线程的标量循环，在执行器传入的 copy stream（`side_stream`）上占住这条 stream 直到返回；同一条 stream 还承载 `gather_block` 的权重预取和残差的取回。GPU 梯度槽只有 2 个，所以反向大约每层要等一次主机加法加一次 H2D。

- `SUROGATE_FORK_HOST_GRAD_ACCUM=1`：主机函数改为调用 `fork/native/src/cpu_step_kernels.cpp` 的 `host_grad_add_bf16`/`host_grad_add_f32`：表达式与上游相同，BF16 舍入改为不含分支的位运算（就近舍入到偶数，NaN 一律为 `0x7fff`，与 `cuda_bf16.hpp` 的主机实现相同），循环向量化（`target_clones`：x86-64-v4、AVX2、基线），每个张量一个 `OmpTeam` 并行区（少于 65536 个元素时串行）。
- 线程数：主机函数在 CUDA 的回调线程上运行，与 rank 线程是两个调用线程。按上节的规则，它们在共享窗口内互相计入，各自只得到一半 CPU，而它们的并行区从不同时运行（rank 线程在 `wait_all_offloads` 等完所有主机函数之后才执行优化器）。`omp_team.h` 因此增加了 serial group：开关打开时，`start_micro_step` 用 `set_omp_serial_group(this)` 把 rank 线程放进以 grad store 地址为键的组，每个 `HostAccumTask` 带着同一个键，主机函数用 `OmpTeam(n, OmpHostFunction{key})`。同组的调用线程互不计入；组外的调用线程（例如另一个 rank）把整个组算作一个。但主机函数运行时 rank 线程并不空闲，只是不在并行区里：它在排队反向的 kernel，等 GPU 时 CUDA 默认的调度方式（`cudaDeviceScheduleAuto`；`csrc/src` 中训练部分没有调用 `cudaSetDeviceFlags`）在进程的 CUDA 上下文数不超过 CPU 数时自旋。所以分组的主机函数给 rank 线程留一个 CPU：team 是组分到的 CPU 数减 1，至少为 1（rank 线程自己的并行区不减）。team 线程继承回调线程的 CPU 掩码，team 大小按回调线程自己的掩码计算；回调线程的掩码由 CUDA 驱动决定，没有验证。
- `SUROGATE_FORK_GRAD_D2H_STREAM=1`：`create_streaming_events` 另建一条 stream（与 `side_stream` 一样用 `cudaStreamCreate`），`offload_layer_grads` 把块梯度的 D2H、`d2h_done`、主机函数和 `accum_done` 都排在这条 stream 上，不再排在调用方传入的 copy stream 上，权重预取和残差取回不再排在主机函数之后。顺序仍由事件保证（`compute_done`、`reduce_done`、`d2h_done`，以及 `wait_all_offloads` 等待的事件）；一步中所有拷贝与加法仍在同一条 stream 上，`start_micro_step` 每个 micro-step 清空待等事件列表依赖这一点。只换 stream 不够：`prepare_layer_grads(L)` 等第 L+2 层的 `d2h_done`，它排在第 L+3 层的主机函数之后，所以两个开关应一起打开。这条 stream 与槽的事件一样不会销毁（`DslGradStore` 没有析构函数）。
- 没有做：把前向的残差写回（`compiled_ops.cpp`）也移到这条 stream（先要确认 5090 的 `asyncEngineCount`）；K 个暂存槽加专用累加线程（python-rl R2 的第二步）；GPU 上逐层的梯度范数（只在 GA=1 时成立）。主机加法每个元素至少 6 字节的内存流量（读两个 BF16、写一个），GA>1 时这一项不会消失；能用 GA=1 和更大 micro-batch 的配置（train-native #2）完全不走这条路径。

### CPU AdamW（T15）

`cpu_adamw.cpp` 的三个逐元素函数（`step_cpu_streaming` 对每个参数调用其中一个；Dispatch-PP 的异步优化器调试接口 `AsyncStaleAdamW` 在它的工作线程上也调用 `cpu_adamw_step`，开关同样作用于它，结果与上游逐位相同）都不向量化：`std::sqrt` 保留 errno；BF16 master 的循环中主机 `__float2bfloat16` 的舍入分支依赖被丢掉的低 16 位，更新后的权重让它频繁预测失败，本机单核上这个循环比 FP32 master 的慢约 4 倍（见下文）。`examples/training/cpu-offload.yaml` 没有设 `master_dtype`，用的正是 BF16 master。`SUROGATE_FORK_CPU_ADAMW_SIMD=1` 时，每个函数照旧算出 `bc1`、`bc2`、`wd_factor`，再调用 `cpu_step_kernels.cpp` 中对应的 `adamw_step_*`：同样的表达式与运算顺序，向量化，在调用线程的 `OmpTeam` team 上执行。源码 pin 比较 fork 循环与上游循环的表达式，上游改动 AdamW 的算术时 pin 失败，而不会悄悄分叉。用 8-bit CPU 状态替换这条路径（python-rl R19、train-native #9）是另一项工作，那时这条路径作为 FP32 状态的后备保留。

### 数值

- 每个元素由一个线程按上游的表达式计算，team 大小和分块不影响结果。`cpu_step_kernels.cpp` 单独使用 `-fno-math-errno`（让 `std::sqrt` 向量化；IEEE 的 sqrt 在各种宽度下都正确舍入）和 `-ffp-contract=off`（`fork/native/cmake/cpu_step_kernels.cmake`）。上游的 x86-64 构建按 x86-64 基线 ISA 编译，没有 FMA；不加后者时 GCC 13 在 x86-64-v4 克隆中融合 57 处乘加，`cpu-step-contract` 的 AdamW 检查在本机全部失败。
- 上面的论证只在构建的基线 ISA 没有 FMA 时成立。aarch64（上游的 DGX Spark wheel）的基线 ISA 总有 FMA；x86-64 的 CXXFLAGS 选了 FMA（`-mfma`、`-march=x86-64-v3` 及以上、`-march=native`）时也有。这时 `surogate-common` 按编译器默认的收缩规则编译 `cpu_adamw.cpp`（GCC 是 `-ffp-contract=fast`，Clang 是 `on`），上游的循环融合乘加，fork 的循环不融合，结果不同：本机以 `-march=x86-64-v3` 模拟，2^20 个元素中 p 有 559 个、m 有 267946 个、v 有 239611 个不同。所以目标 ISA 有 FMA 时（`__FP_FAST_FMAF`、`__FMA__`、`__FMA4__` 或 `__ARM_FEATURE_FMA`，与 `-ffp-contract` 无关，两个文件看到的相同），`cpu_adamw_simd_supported()` 返回 false，`SUROGATE_FORK_CPU_ADAMW_SIMD=1` 被忽略，并在 stderr 上提示一次。`cpu-step-contract` 的参考循环与生产中的 `cpu_adamw.cpp` 一样按 `-O3`、默认收缩编译，在有 FMA 的基线上与上游一样融合：本机以 `-march=x86-64-v3` 构建时，强制打开开关（`-DSUROGATE_FORK_UPSTREAM_FMA=0`）后 AdamW 检查失败；不强制时它确认开关被拒绝，并跳过 AdamW 的逐位比较。之前按 `-O0` 编译的参考循环看不到这种融合，在同样的构建上照样通过。没有在 aarch64 上构建过（这里没有交叉工具链）。主机梯度加法不做乘法，在所有 ISA 上都逐位相同。
- BF16 结果与上游逐位相同：主机梯度加法在全部 2^32 个 BF16 操作数对上与上游的循环比较过（`step.sh`，带克隆的构建以及基线、AVX2、x86-64-v4 各一份不带克隆的构建）。FP32 结果除 NaN 的 payload 外逐位相同：两个 NaN 参与一次运算时返回哪一个没有规定（IEEE 754 不规定，GCC 也会交换 `+` 和 `*` 的操作数），向量化的循环可能保留另一个 NaN 的 payload。BF16 输出把所有 NaN 映射为 `0x7fff`，不受影响。
- 浮点环境（例如 FTZ/DAZ）由各线程从创建它的线程继承，与上游的 OpenMP 区域相同。

### 测量

主机：与上节相同的 4 vCPU KVM 虚拟机（Intel Xeon，family 6 model 207，2.1 GHz，1 个 socket，无 SMT，L3 260 MB，16 GB 内存，有 AVX-512，运行时选中 x86-64-v4 克隆），没有 GPU。机器与其他会话共用：每行只在其余进程平均占用不超过 0.25 个 CPU 时计时，否则重测。作为参照（一次单独的测量，不在 `step.sh` 中），本机单线程 memset 约 8.5 GB/s，4 线程对 1 GiB 数组做 64 位整数加法（读两个数组、写一个）约 56 GB/s。代码：`step.sh` 从构建目录的 `compile_commands.json` 取出 `cpu_adamw.cpp` 与 `cpu_step_kernels.cpp` 的编译命令（GCC 13.3，`-O3 -fopenmp`，后者另加上述两个选项），链接进 `stepbench.cpp`；“上游”是 `cpu_adamw.cpp` 中开关关闭时的循环，以及 `stepbench.cpp` 中逐字复制的 `accumulate_staged_layer` 的 BF16 循环（源码 pin 检查两者一致）。负载：小模型与 `run.sh` 的 Llama 形状相同（L=8、h=896、im=4864、V=32000，1.78 亿个参数，其中块梯度 1.21 亿个元素）；8B 层是两个 Qwen3-8B 块（h=4096、im=12288，k/v 为 h×h/4，每层 1.93 亿个元素）。PIN 是 rank 线程（以及模拟回调线程的线程）掩码中的 CPU 数，即 team 的上限。每格是 3 轮的中位数，括号中是每个元素的耗时；所有行中 T14、T15 的结果与上游逐位相同（FNV 哈希一致）。

步开始的清零（T13）：

| 负载 | 主机梯度 | 上游 memset | T13 |
|---|---:|---:|---:|
| 小模型 | 356 MB | 43.0 ms（8.3 GB/s） | 2.7 µs |
| 8B 层 | 772 MB | 84.7 ms（9.1 GB/s） | 1.1 µs |

T13 在两种负载上都清零 0 字节（micro-step 0 的拷贝覆盖了所有缓冲），只剩登记的开销（几微秒）。按本机的速度，Qwen3-8B 的 16.4 GB 要约 1.9 s；在桌面 CPU 上约 0.55–0.8 s。

micro-step 1 起的主机梯度加法，每层耗时（T14，`SUROGATE_FORK_HOST_GRAD_ACCUM=1`；上游的循环是单线程的，不随 PIN 变化）：

| 负载 | PIN | 上游 | T14 | 加速 | T14 的内存流量 |
|---|---:|---:|---:|---:|---:|
| 小模型 | 1 | 18.1 ms（1.20 ns） | 7.6 ms（0.50 ns） | 2.4× | 11.9 GB/s |
| 小模型 | 2 | 18.9 ms（1.26 ns） | 4.2 ms（0.28 ns） | 4.6× | 21.8 GB/s |
| 小模型 | 4 | 19.1 ms（1.26 ns） | 2.7 ms（0.18 ns） | 7.1× | 33.8 GB/s |
| 8B 层 | 1 | 250.1 ms（1.30 ns） | 95.5 ms（0.49 ns） | 2.6× | 12.1 GB/s |
| 8B 层 | 2 | 252.2 ms（1.31 ns） | 53.7 ms（0.28 ns） | 4.7× | 21.6 GB/s |
| 8B 层 | 4 | 242.2 ms（1.25 ns） | 29.1 ms（0.15 ns） | 8.3× | 39.8 GB/s |

CPU AdamW 一整步（T15，`SUROGATE_FORK_CPU_ADAMW_SIMD=1`），小模型的全部 1.78 亿个参数，FP32 矩：

| master | PIN | 上游 | T15 | 加速 | T15 的内存流量 |
|---|---:|---:|---:|---:|---:|
| BF16 | 1 | 3512 ms（19.73 ns） | 228 ms（1.28 ns） | 15.4× | 17.2 GB/s |
| BF16 | 2 | 1806 ms（10.14 ns） | 115 ms（0.65 ns） | 15.7× | 34.1 GB/s |
| BF16 | 4 | 972 ms（5.46 ns） | 64 ms（0.36 ns） | 15.1× | 61.0 GB/s |
| FP32 | 1 | 851 ms（4.78 ns） | 242 ms（1.36 ns） | 3.5× | 20.6 GB/s |
| FP32 | 2 | 450 ms（2.53 ns） | 129 ms（0.72 ns） | 3.5× | 38.7 GB/s |
| FP32 | 4 | 243 ms（1.37 ns） | 75 ms（0.42 ns） | 3.2× | 66.2 GB/s |

内存流量按每个元素的字节数计：加法 6 B（读两个 BF16、写一个），AdamW 的 BF16 master 22 B、FP32 master 28 B（读写 master、m、v，读梯度）。

线程数与 serial group（`stepbench group`，PIN=0，即进程的全部 4 个 CPU）：一个常驻线程模拟 CUDA 回调线程，每步先做 3 个 micro-step 的主机加法；这期间 rank 线程或者阻塞等待，或者自旋（模拟反向期间的 rank 线程：它在排队 kernel，等 GPU 时按 CUDA 默认的调度方式自旋）；然后 rank 线程对全部参数运行一次 AdamW（BF16 master，T15 的 `adamw_step_bf16_param`，每 2^24 个元素一个并行区）。两者都按 `OmpTeam` 确定线程数。“之前”是这一项修正之前的规则（`ce939c53`：分组的主机函数得到整个份额），用同一个 `stepbench.cpp` 与那一版的 `omp_team.h`、`cpu_step_kernels.cpp` 构建；“现在”给 rank 线程留一个 CPU。这里的 8B 层是一个 Qwen3-8B 块（`llama:1:4096:12288:0`，1.93 亿个元素），不是上面两张表的两个块。每格是 3 轮的中位数；AdamW 一列取自“现在”的构建，两种构建的 rank 线程 team 相同。

| 负载 | serial group | rank 线程 | 回调线程的 team（之前 → 现在） | 3 个 micro-step 的加法：之前 | 现在 | rank 线程的 team | AdamW |
|---|---|---|---:|---:|---:|---:|---:|
| 小模型 | 无 | 阻塞 | 2 → 2 | 94.5 ms | 95.7 ms | 2 | 115.8 ms |
| 小模型 | 无 | 自旋 | 2 → 2 | 99.1 ms | 98.3 ms | 2 | 113.9 ms |
| 小模型 | 有 | 阻塞 | 4 → 3 | 67.0 ms | 72.0 ms | 4 | 64.2 ms |
| 小模型 | 有 | 自旋 | 4 → 3 | 104.3 ms | 78.6 ms | 4 | 63.0 ms |
| 8B 层 | 无 | 阻塞 | 2 → 2 | 150.9 ms | 154.9 ms | 2 | 123.7 ms |
| 8B 层 | 无 | 自旋 | 2 → 2 | 153.3 ms | 152.1 ms | 2 | 125.8 ms |
| 8B 层 | 有 | 阻塞 | 4 → 3 | 83.8 ms | 112.0 ms | 4 | 72.1 ms |
| 8B 层 | 有 | 自旋 | 4 → 3 | 143.8 ms | 117.5 ms | 4 | 75.7 ms |

没有 serial group 时，按上节的共享规则，两个并行区从不同时运行的调用线程互相计入，各自只得到 2 个线程；有了它，rank 线程的 AdamW 得到 4 个。rank 线程自旋时，主机函数用整个份额（4 个线程）反而比留一个 CPU（3 个）慢 1.2–1.3 倍，8B 层上与不分组的 2 个线程差不多（144 对 153 ms）：team 按 static 分块，与 rank 线程共用一个 CPU 的那个线程拖住整个并行区。rank 线程阻塞时，留出的 CPU 闲着，加法慢 1.07–1.34 倍（8B 层 84 对 112 ms）。反向期间 rank 线程在排队 kernel，等 GPU 时按 `cudaDeviceScheduleAuto` 自旋，所以按自旋的情形取掩码减 1。CPU 更多的主机上，留一个 CPU 的代价约为掩码大小分之一（16 个 CPU 时约 6%）。5090 主机上主机函数运行期间 rank 线程是否一直占着 CPU，没有验证（下文第 2 项检查）。

逐位检查（`step.sh` 的最后一段）：带克隆的构建以及基线（`-march=x86-64`）、AVX2（`-mavx2`）、x86-64-v4（`-march=x86-64-v4`）三份不带克隆的构建（只有 `cpu_step_kernels.cpp` 按该 ISA 编译，`cpu_adamw.cpp` 仍是基线构建，所以这三份定义 `SUROGATE_FORK_UPSTREAM_FMA=0`，不按自身 ISA 的 FMA 拒绝开关），全部 4294967296 个 BF16 操作数对的加法结果都与上游的循环相同；AdamW 在小模型上开与不开 `SUROGATE_FORK_CPU_ADAMW_SIMD` 的哈希在四份构建中都相同。

解读与限制：

- 上游的两个循环受计算限制：加法单线程约 1.2 ns/元素，不随 PIN 变化（各格之间的差异是这台共用虚拟机的波动）；BF16 master 的 AdamW 单线程约 20 ns/元素，是 FP32 master 的约 4 倍，原因是舍入分支的预测失败。T15 在 4 个线程时达到约 61–66 GB/s，不低于上面的 4 线程流式参照（56 GB/s），即已受内存带宽限制；T14 的加法在 4 个线程时约 34–40 GB/s。桌面主机的核更快、内存带宽更高（双通道 DDR5 约 60–75 GB/s），上游循环在那里会快一些，但加法仍是单线程，BF16 master 的 AdamW 仍受分支预测失败拖累。
- 换算到 Qwen3-8B（每层 1.93 亿个元素，GA=4）：本机上游的主机加法每层约 240–250 ms，T14 用 4 个线程约 29 ms；在 16 核、约 60 GB/s 的桌面主机上，6 B/元素的下限约 19 ms，低于每层约 26 ms 的反向，但只有在加法不再挡住权重预取（`SUROGATE_FORK_GRAD_D2H_STREAM=1`）时才能藏在反向之下。这些换算没有在 GPU 主机上测过。
- AdamW 的换算：按上表 4 个线程的每元素耗时，BF16 master 的 8.19B 个参数约 3.0 s（上游约 45 s；本机内存放不下 8B 的状态，这只是换算）；桌面主机上 T15 受内存带宽限制（22 B/参数，180 GB，60–70 GB/s 时约 2.6–3 s）。

### 尚未在 GPU 上验证的部分

没有 GPU：GPU 上的行为由代码阅读确定；`dsl_grad_store.cpp`、`graph_executor.cpp`、`py_train.cpp`、`adamw_8bit_optimizer.cpp`、`cpu_adamw.cpp` 与 `cpu_step_kernels.cpp` 只按 sm_120a 构建的编译命令编译检查过。在 RTX 5090 主机上依次检查下列各项，配置都是 `examples/training/cpu-offload.yaml` 改为 `model: Qwen/Qwen3-8B`、`max_steps: 20`（下称 C8），用 `surogate sft C8` 运行：

1. T13：在同一个构建（当前 HEAD）上运行 C8，一次设 `SUROGATE_FORK_HOST_GRAD_MEMSET=1`（上游的 memset），一次不设（T13）；两次只有 T13 不同。先设开关连跑两次，确认这个配置本身可复现。每步记录的 loss 与梯度范数应逐位相同；不设开关的一次 `step_seconds` 应更短，至多短 memset 的时间（16.4 GB 除以主机单核的 memset 带宽），实际减少量取决于已排队的前向遮住了多少。不要比较 T13 的提交与它的父提交：本分支落地时把各 topic 依次放到一条线上，796fdfa0 的父提交是 1bba063b，而 488e5f00 与 796fdfa0 之间还有 T17–T29 等提交，比较这两个提交分不出 T13；796fdfa0 与 1bba063b 只差 T13 的第一版，没有 e2f56d21 的 `before_copy()`。
2. T14 的加法：C8 分别不设与设置 `SUROGATE_FORK_HOST_GRAD_ACCUM=1`。两者每步的梯度范数与 loss 应逐位相同（主机梯度逐位相同）。训练时先用 `grep -H Cpus_allowed_list /proc/<pid>/task/*/status` 记下 rank 线程、回调线程和 team 线程的掩码（主机函数的 team 应是回调线程掩码中的 CPU 数减 1，`OMP_NUM_THREADS` 更小时取它）。再用 `nsys profile -t cuda,osrt -s cpu -o t14 surogate sft C8` 比较 micro-step 1–3 与 micro-step 0 的每层反向耗时（目标：不超过 1.3 倍）和回调线程上主机函数的时长，并看主机函数运行期间 rank 线程是否占着一个 CPU（CPU 采样，或训练时 `top -H -p <pid>`）。然后按 team 大小比较 micro-step 1–3 的每层反向耗时：默认（掩码减 1）、`OMP_NUM_THREADS` 设为掩码的一半，以及本项修正之前的 `ce939c53`（54954e9b 的父提交，主机函数得到整个掩码；C8 走到的代码中它与 HEAD 只差这条规则，以及不设时不起作用的 `SUROGATE_FORK_HOST_GRAD_MEMSET`，其余差别在服务端与 GRPO 的 Python 代码中）。这一项没有开关，所以要换构建。默认应不慢于另外两种；如果整个掩码更快（rank 线程这时并不占 CPU），把三组数字记在 ledger T14 的 `conflict_history` 中，再决定是否去掉这一个 CPU 的预留。
3. T14 的 stream：再加 `SUROGATE_FORK_GRAD_D2H_STREAM=1`。nsys 中块梯度的 D2H 应出现在一条与权重预取的 H2D 不同的 stream 上，权重预取不再排在主机函数之后；梯度范数与 loss 与第 2 项逐位相同。先用 `deviceQuery` 记下 `asyncEngineCount`（为 2 时 D2H 与 H2D 可以同时进行）。有两张卡时加 `world_size: 2` 再跑一次，检查 NCCL 归约（`reduce_done`）之后的 D2H。
4. T15：C8 改为 `max_steps: 5`、`save_steps: 5`，分别不设与设置 `SUROGATE_FORK_CPU_ADAMW_SIMD=1`。5090 主机是 x86-64：用 fork 的 x86-64 wheel，或不在 CXXFLAGS 中选 FMA 的本地构建；第二次的日志中不应出现 `SUROGATE_FORK_CPU_ADAMW_SIMD=1 is ignored`。两次的 loss 应逐位相同，checkpoint 中权重分片的 sha256 应相同；nsys 中最后一个反向 kernel 到下一步第一个前向 kernel 之间的空隙（CPU 优化器步）应缩短。
5. `fork/dev test gpu` 照常通过（fork 的 C++ 用例不经过这些路径）。

第 2–4 项的逐位比较同样以第 1 项确认的可复现性为前提：同一提交连跑两次结果就不同时，改为比较各步梯度范数的相对差（主机梯度相同，差异只来自 GPU 的不确定性）。结果记在 ledger 中 T13–T15 各自的 `conflict_history` 末尾。

### 复现

`step.sh` 与 `run.sh` 一样读取构建目录中的 `compile_commands.json`：先运行 `make configure`，再运行 `cmake -S csrc -B csrc/build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`，然后 `fork/native/bench/cpu-optimizer/step.sh csrc/build`。它用 `cpu_adamw.cpp` 与 `cpu_step_kernels.cpp` 各自的编译命令编译这两个文件，链接进 `stepbench.cpp`，按 `ROUNDS` 轮运行上表各行，最后按 ISA 做逐位检查（`ISA=0` 跳过）。`SMALL`、`LARGE` 替换两种形状。各行在机器上其他进程活跃时会偏慢；上表各行的计时都满足上文的空闲条件。

## RTX 5090 32 GB：稠密与 MoE 微调

单张 RTX 5090 应使用 SM120 kernel 和现有的 CUDA 13 工具链。SM100 kernel 或只支持 NVLink 的通信后端并不自动适用于这张卡。

规划 4–7B 稠密或约 10B MoE 训练时，把容量与计算分开考虑：

- BF16 冻结 base 权重每个参数约 2 字节，这还不包括 adapter、激活、量化/转置缓存和工作区。LoRA 和 QLoRA 可以让优化器状态只与 adapter 成正比，而不是与整个 base 成正比。
- BF16 可训练权重和梯度加上两份 FP32 Adam 矩，每个参数已约 12 字节。FP32 master/梯度和单独的工作副本还会增加开销。即使是 BF16 权重/梯度加两份 8-bit 矩缓冲，在计入 scale 元数据和激活之前，每个参数也约需 6 字节：7B 参数约 42 GB，超出这张 GPU 的容量。
- MoE 的激活参数决定了大部分计算量，但所有可训练 expert 的权重和优化器状态仍需存储。按总参数量而不只是激活参数量来规划 offload。

训练 adapter 时，base 权重放得下就先让它常驻显存；在相同的有效 batch 和质量目标下比较 BF16 与受支持的 FP8/NVFP4 recipe，并在激活预算内提高 micro-batch/token 占用率。把缓存的打包/转置权重明确计入预算；如果某个缓存迫使额外的 offload，省下的量化 kernel 并不是免费的。

全参微调放不下时，使用现有的 CPU 训练/offload 路径，并在 GPU 计算之外同时测量 CPU 优化器耗时、主机内存需求和 PCIe 流量。可以从 [CPU offload](../../examples/training/cpu-offload.yaml) 和[全参微调示例](../../examples/training/full-finetune.yaml)入手；这些示例是配置参考，不是经过验证的 5090 性能预设。梯度累积能摊薄优化器发布/更新的开销，但不能消除每个 micro-batch 的权重传输或 checkpoint 重计算。如果只 offload 状态/激活就足够，就不要把权重移出设备。

检查配置最终解析出的行为：`cpu_training: true` 会启用 master 权重流式传输，对全参微调还会启用梯度 offload 并在 CPU 上执行优化器。`use_zero_copy` 为 false 时，单独设置 `offload_optimizer: true` 并不会把状态移出设备。不要因为各个 offload 开关分别能节省不同的缓冲区，就假定它们组合起来是经过验证的配置。

[NVIDIA TensorRT-LLM 硬件表](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md)区分了 SM120 与 SM100/103，并列出 SM120 上对 NVFP4、MXFP4、per-tensor FP8 和 FP8 KV 的推理支持。推理支持并不代表训练反向支持或数值等价：请验证实际使用的 Surogate recipe。关于训练专用的低精度设计，NVIDIA 的 [Transformer Engine primer](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fp8_primer.ipynb) 和 [NVFP4 预训练论文](https://arxiv.org/abs/2509.25149v1)讨论了随机舍入、双向权重缩放和离群值处理。这些数值选择属于训练 recipe 的一部分，不能当作透明的推理替换。SM120 支持需要单独核实：某个库提供 MXFP8 recipe，并不意味着它在 RTX 5090 上实现了完整的训练布局。

## 研究来源与兼容性

以下实现为优化方向提供参考。它们不是新的必需依赖，也不是可互换的后端：

| 来源 | 与训练相关的机制 | 约束 |
| --- | --- | --- |
| [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) / [TensorRT](https://github.com/NVIDIA/TensorRT) | 融合下游算子，按实际形状选择 tactic，复用有效的布局和有界工作区，降低主机端启动开销。 | 这些是推理框架。训练必须保留保存的激活、权重版本和反向语义；本运行时不把它们用作训练后端。 |
| [DeepInfer, V4.1 source](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9) | 异步调度与按负载执行；rollout 效率对 GRPO 很重要。 | 其推理和投机解码收益并不衡量主模型的训练更新。 |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/tree/5081f8513af57e61696de4f4f04b0e707ffb503e) | 分组的前向和权重梯度 GEMM；除 GEMM 时间外，还要计入布局/scale 准备。 | 该版本面向 SM90/SM100，不能直接用作 SM120 后端。 |
| [DeepEP](https://github.com/deepseek-ai/DeepEP/tree/f6aa38fe06cd4a0489e7730365d4e0114970a2f3) | 复用 routing handle，减少主机端计数同步，让 expert 数据移动与计算重叠。 | 该版本需要 CUDA 13.1+、NCCL 2.32.3+ 和 NVLink；跨节点执行还需要合适的 RDMA 网络。 |
| [DualPipe](https://github.com/deepseek-ai/DualPipe/tree/030ce4325f4ebeb437da4ebc6d00a70469dd58ae) | 以显式依赖调度 forward、dgrad、wgrad 和通信，使其相互重叠。 | 需要兼容的多 stage 调度以及额外的权重/激活容量。给流水线调度改名并不会产生重叠。 |

运行时已经有分组 MoE kernel、routing/重计算计划复用、控制流、graph capture 和预取。应扩大这些机制的覆盖范围，而不是假定它们不存在。未来的原生 GRPO dispatch-PP loss 路径在替换当前的 logprob/梯度路径之前，还必须保留打包位置重置和温度，并保证 token 归一化只应用一次。同样，要保留现有的 offload/capture 限制：当 `offload_master` 跨 stream 流式传输权重时，Surogate 会禁用 graph capture。NVIDIA 的 [TensorRT runtime contract](https://github.com/NVIDIA/TensorRT/blob/16ed0636df41e4a27d46f82d9028fc50fadc7200/include/NvInferRuntime.h) 也对权重流式传输和 capture 做了限制。增加辅助 stream 会占用激活/工作区容量，并不会消除生命周期依赖。

原生优化需要编译、数值测试、graph replay 检查，以及在目标 GPU 上的端到端测量。仅靠 CPU 测试无法确定 GPU 吞吐量，也无法判断离物理极限有多近。

在目标 GPU 主机上先运行 `fork/dev test native`（构建目录 `csrc/build`），再做模型基准。只想重跑与上述优化相关的用例时，按 Catch2 标签过滤：

```bash
csrc/build/integration-tests '[fp8][recipe]'
csrc/build/integration-tests '[lora][fused-dx]'
csrc/build/integration-tests '[wgrad-zero]'
csrc/build/dsl-run-state-tests '[dispatch-pp][host-copy]'
```

然后用同一个预热后的训练负载，对比启用与不启用 `SUROGATE_FUSE_LORA_DX=1` 的梯度/loss、峰值显存和优化器步耗时。kernel 测试通过本身并不能说明训练变快了。

## LoRA 融合输入梯度

稠密的共享输入 LoRA 反向有一条实验性的合并输入梯度路径：

```bash
SUROGATE_FUSE_LORA_DX=1 <training-command>
```

对两个或三个兼容的投影，每个 adapter 的 dA/dB 计算保持不变。运行时把它们缩放后的中间梯度和 A 矩阵打包到现有 scratch 中，然后用一次 GEMM 算出对 dX 的合并贡献，而不是反复读取并更新整个 dX 矩阵。这是用少量打包拷贝换取更少的 GEMM 启动和 dX 读-改-写；当 token 维度和输入特征维度相对 adapter rank 较大时，最有可能获益。

在目标硬件上完成数值和性能验证之前，该路径默认关闭。它要求兼容的 BF16 操作数、受支持的小 rank、互不重叠的输出切片与存储、足够的现有 scratch，并且没有启用输入 dropout。遇到不支持的布局或 dtype、缺少 dX，或处于转置 A 的调试模式时，会在启动融合计算之前回退到现有的逐切片实现。满足这些条件时，Q/K/V 输出宽度可以不相等。热路径上没有新增分配或 stream。

合并后的归约只对 dX 舍入一次，而不是在每次 BF16 切片更新后都舍入，因此不应期望 dX 按位一致。请以合适的误差界与 FP32 参考和非融合路径比较，检查 adapter 梯度和 loss，再测量完整的步耗时和 scratch 用量。如果打包开销或数值上的取舍超过收益，就保持该选项关闭。

## GRPO 周期计时

GRPO 在现有训练指标之外，还记录获取和发布的耗时。trainer 每隔 `logging_steps` 打印并写出这些指标：写出的指标与该步其他训练指标在同一条记录中；打印的是单独一行 `step=<n> cycle=<秒>s publish=<秒>s pack=<秒>s batch_wait=<秒>s receive=<秒>s`，位于该步常规日志行之前。日志行由 `surogate/_fork/grpo_cycle_timing.py` 生成。

| 指标 | 说明 |
| --- | --- |
| `train/publish_duration_ms` | 发布并清理上一次策略更新；初始步没有发布。 |
| `train/pack_duration_ms` | 等待 rollout 并打包下一个 batch。 |
| `train/batch_wait_duration_ms` | 等待打包好的 micro-batch 传输就绪。 |
| `train/batch_receive_duration_ms` | 接收并解码打包好的 micro-batch。 |
| `train/cycle_duration_ms` | 从开始发布策略到优化器更新完成，包括 batch 的获取和准备。 |

周期计时使用单调时钟，不包括步日志、checkpoint 和最后一次权重发布。它衡量的是一次 trainer 迭代，而不是整个运行的墙钟时间。空 batch 重试不会产生完整的周期指标。`train/duration_ms` 和 `train/tokens_per_second` 保持原有的训练步范围，因此获取延迟不会改变历史吞吐图表的含义。诊断输入饥饿或策略发布开销时，对比周期耗时与训练耗时。原生、诊断和 dispatch-PP 路径使用相同的计时边界。

## cpu_training 的 AdamW 状态 checkpoint

`cpu_training` 的全量训练（`lora: false`，`optimizer: adamw_8bit`）在主机上用 FP32 的一阶矩和二阶矩做 AdamW（`step_cpu_streaming`）。上游不把它们写进 checkpoint，恢复时矩从零开始，同时在 GPU 上分配一份不用的 8-bit 状态；现象、证据与 5090 上的检查见 [upstream-errata.md](upstream-errata.md#cpu_training-的全量训练不保存-cpu-adamw-的矩fork-已修复ledger-t21)。T21 修复这个缺陷，默认开启：checkpoint 带上矩，恢复时读回，不再分配设备状态。代码在 `fork/native/src/cpu_adamw_checkpoint.cpp`（布局、容器、加载检查）与 `fork/native/include/surogate_fork/pageable_host.h`（写入器的主机分支），上游文件中只有调用点。GPU 训练、LoRA 与新训练的 CPU 步都不变。`cpu_training` 配 `adamw` 或 `normuon` 的恢复变了：它们的 checkpoint 没有优化器状态，恢复时被拒绝，除非设 `SUROGATE_FORK_CPU_ADAMW_ALLOW_MISSING_STATE=1`，并且恢复时先分配主机上的矩（8 字节/参数），第一步换成配置的优化器时释放（见 errata 的“未修复”段）。

格式 v1（`adam.m.shard_<rank>_of_<n>.safetensors` 与 `adam.v.shard_...`）：

| 张量 | dtype 与形状 | 内容 |
|---|---|---|
| `cpu_adamw.m.<参数名>` | FP32，形状同 master | 一阶矩 |
| `cpu_adamw.v.<参数名>` | FP32，形状同 master | 二阶矩 |
| `cpu_adamw.format` | INT32 `[1]`，值为 1 | 格式版本，两个文件各有一份 |

`<参数名>` 是 CPU 步更新的每个 master 在 `DslGradStore::param_names()` 中的名字，按这个顺序；两个名字共用的 master（tied 的 `lm_head` 就是 embedding 的 master，T10）只出现一次，用先出现的名字。第一次 CPU 步与恢复用同一个布局函数（`build_layout`），所以读回的矩落在 CPU 步读取的位置。偏差校正的步数不在 checkpoint 中：trainer 按恢复的步数传入 `step + 1`。

开关（环境变量，在保存或加载时读取）：

| 变量 | 默认 | 作用 |
|---|---|---|
| `SUROGATE_FORK_CPU_ADAMW_SAVE_STATE` | 保存 | 设为 `0` 时 checkpoint 不写矩（与上游相同，`adam.m`/`adam.v` 只有文件头），省下下面的磁盘开销；用这样的 checkpoint 恢复需要下一个开关 |
| `SUROGATE_FORK_CPU_ADAMW_ALLOW_MISSING_STATE` | 拒绝 | 设为非 `0` 的值时，从缺少矩的 checkpoint 恢复只打印警告（`[surogate fork T21] WARNING`），缺少的矩从零开始（与上游相同）；格式版本不同时仍然拒绝 |

开销：

- 磁盘：每个 checkpoint、每个本地 rank 8 字节/参数。Qwen3-8B（8.19B 参数）为 65.5 GB，默认 `save_total_limit: 5` 时共 328 GB，另加原有的权重 shard 与导出（约 32.8 GB/checkpoint）。磁盘不够时设 `SUROGATE_FORK_CPU_ADAMW_SAVE_STATE=0`。
- 保存与恢复的时间：多写、多读 65.5 GB（8B）。写入器对 pageable 内存直接 memcpy 进 mmap，没有上游写入器那份零初始化的中转缓冲；没有测过耗时。
- 主机内存：不变。CPU 步本来就持有这两份矩，恢复时只是在加载之前分配（同样大小，零初始化）。
- 显存：恢复时不再分配 2.125 字节/参数的 8-bit 状态（8B 约 17.4 GB，16.2 GiB）。

兼容规则：

1. 名字与 GPU 路径的 `adamw8bit.state1`/`scales1`/`state2`/`scales2`（BYTE 与 FP16）不相交。只有 `cpu_training` 的恢复检查文件：`cpu_training` 从 GPU 训练的 checkpoint 恢复时，文件里没有任何应有的名字，被拒绝，不会悄悄归零。反方向不检查：GPU 训练从 `cpu_training` 的 checkpoint 恢复时，读取器跳过 `cpu_adamw.*`，8-bit 状态从零开始，没有警告，与上游相同。
2. 每个 master 一个张量：参数集变化（换了模型，或可训练参数不同）表现为缺少名字（拒绝，报错写明缺少的名字）或形状不符；dtype 或形状不符由读取器报错（`allow_cast` 为 false）。
3. 上游与不带 T21 的 fork 写出的 `cpu_training` checkpoint 没有矩（没有张量，或者是此后每次保存写下的无用 8-bit 状态），`adamw` 与 `normuon` 的 checkpoint 也没有优化器状态。以 `cpu_training` 恢复它们被拒绝，除非设 `SUROGATE_FORK_CPU_ADAMW_ALLOW_MISSING_STATE=1`。
4. `cpu_adamw.format` 不是 1 时总是拒绝。
5. 数据并行的各 rank 持有相同的矩，每个本地 rank 写、读自己的一份（N 个本地 rank 占 N 倍磁盘），与权重 shard 相同；shard 数由 `checkpoint.json` 的 `world` 检查。只有第一个节点写文件。
6. 以后的格式（例如路线图 #9 的 8-bit CPU 状态）用新的名字或新的版本号，并转换（量化）或拒绝 v1 文件。如果它改用 `adamw8bit.*` 名字，必须与 `init_state` 的槽位布局一致（每个槽位按 32 对齐的 `state_offset`，int8 softsign 的 m，uint8 sqrt 的 v，每 32 个元素一个 FP16 scale），并保留同样的完整性检查：`require_complete` 只检查 T21 的容器。

## GPT-OSS 预量化 LoRA 的专家输入梯度

ledger T25，开关 `SUROGATE_GPT_OSS_PREQUANT_SKIP_EXPERT_DGRAD=1`，默认关闭。

这是缺陷修复，不是优化：上游在预量化的 GPT-OSS checkpoint 上做 LoRA 训练时省掉 MoE 专家 gate_up 与 down 的输入梯度（dgrad）GEMM，梯度因此是错的（见 [upstream-errata.md](upstream-errata.md#gpt-oss-预量化-checkpoint-的-lora-训练跳过冻结专家的输入梯度fork-已修复ledger-t25)）。fork 默认计算它；开关打开时恢复上游的做法，每个 MoE 层少两个 BF16 分组 GEMM，但专家 gate_up LoRA 和更早的层得到的梯度是错的，只适合复现上游的吞吐或对照实验。开关每个进程只读一次，打开时 stderr 有一行说明。

代价没有在 GPU 上测量。按路线图的估计（GPT-OSS-20B 前向加重放的专家时间 10.9 ms/层），多出的 dgrad 约 +5 ms/层；`docs/reference/benchmarks.md` 中 gpt-oss-20B 的吞吐是带着跳过测的，与 fork 的默认设置不可直接比较。测量方法：在 5090 上用 `examples/sft/gpt-oss/gptoss-lora-mxfp4.yaml`（改为单卡，`ep_size: 1`）分别不设和设置开关，各跑 20 步，比较稳定后的步时间，并用 nsys 统计每层分组 GEMM 的个数与耗时（开关打开时每个 MoE 层少两个）。

## GRPO 训练路径服务的排队与中止（ledger T28）

训练路径的 `SharedModelServer`（MoE、GLM、非 Qwen 模型，以及 LoRA 目标或工具模板不标准的 Qwen3/3.5）原先在 lane 全忙时回 429，rollout 在 OpenAI SDK 的退避中丢失，补发的请求抢先拿到 lane；这是缺陷，记在 [upstream-errata.md](upstream-errata.md#训练路径的-sharedmodelserver-在满载时拒绝-rolloutfork-已修复ledger-t28)。修复在 `surogate/_fork/shared_admission.py`，默认打开：

| 开关 | 取值 | 作用 |
| --- | --- | --- |
| `SUROGATE_SHARED_SERVER_QUEUE` | `fifo`（默认）、`reject` | `fifo`：lane 全忙时按到达顺序排队，上限 `max_pending_requests` 加每个 lane 一个，等待超过 `pending_timeout_ms` 回 429；`reject`：上游的接纳方式，不做任何连接检查 |
| `SUROGATE_SHARED_SERVER_ABORT` | `1`（默认）、`0` | `1`：已接纳的请求在客户端关闭连接后的下一次 token 检查（最多每 50 ms 一次）停止；`0`：照样解码到结束（上游行为）。排队中的请求在客户端离开后总是被跳过，否则批末取消的请求会在打包间隙中拿到 lane。`reject` 下不起作用 |

两个变量在服务构建时读取并检查，任一取值不合法时构建报错（`reject` 下 `SUROGATE_SHARED_SERVER_ABORT` 也检查）。`max_pending_requests` 与 `pending_timeout_ms` 由 `surogate/grpo/utils/capacity.py` 按本次运行的在途数算出（至少 16 个等待者，`native_colocate.py` 传入），也可以在 `infer` 配置中显式设置为正数；配置不接受 0，运行中关掉排队只能用 `SUROGATE_SHARED_SERVER_QUEUE=reject`。每个阶段写入 `shared_weights.jsonl` 的服务摘要多了累计计数：`admission_refused_capacity`（因 lane 与队列都满回的 429，`reject` 模式下就是全部满载 429）、`admission_refused_timeout`、`admission_refused_training`（训练期间与关闭后的 429）、`admission_queued`、`admission_max_queued`、`admission_disconnected_waiting`（客户端离开、没有被接纳的等待者）和 `admission_aborted_running`（客户端离开后停止的请求）。

CPU 测量（`fork/patches/bench/grpo_shared_admission.py`，复现参数见 errata；前三行各 2 个种子，SDK 的退避常数与每 token 的时间缩为一半，后两行不缩放、种子 0）：

| 模式 | 一批的时间 | 满载 429 | 丢失的原始 rollout | 丢弃的组 | batch 中的原始组 | `begin_training()` 等待 | 取消后解码的 token |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `reject` | 82-106 s | 4383-5601 | 256-295 | 0-1 | 1/16 | 5.2-8.5 s | 5.8k-8.0k |
| `fifo` | 36.5-38.6 s | 0 | 0 | 0 | 15-16/16 | 0.01-0.02 s | 3-13 |
| `fifo` + `ABORT=0` | 38.2-38.6 s | 0 | 0 | 0 | 16/16 | 7.3-8.7 s | 7.0k-9.3k |
| `reject`，不缩放 | 155.6 s | 4269 | 252 | 4 | 0/16 | 16.4 s | 6.7k |
| `fifo`，不缩放 | 74.1 s | 0 | 0 | 0 | 16/16 | 0.02 s | 10 |

队列最多有 113 个等待者，比 `max_pending_requests`（112）多一个：没有每 lane 一个的余量时，每批有 4-9 个请求在交接的瞬间看到队列已满而得到 429（重试后成功）。CPU 上的数字只说明调度行为：解码是 sleep，没有 GIL 下的真实解码开销，也没有 MoE 低并发轮次读专家的成本。没有改动、留作后续的：`ThreadingHTTPServer` 的 listen backlog 是 5，一批开始时上百个连接同时到达，可能有连接等 SYN 重传（约 1 s），未测量。

5090 检查（未验证）：W-MoE-GRPO（约 10B 的 qwen3_moe LoRA，16 题 × 8，`max_tokens` 1536，`batch_size` 128，`max_num_seqs` 16）各跑 20 步 `fifo` 与 `SUROGATE_SHARED_SERVER_QUEUE=reject`（再加一组 `fifo` + `SUROGATE_SHARED_SERVER_ABORT=0`）。每步记录：`shared_weights.jsonl` 中 `admission_*` 计数的增量；orchestrator 日志中 “Rollout error … RateLimitError” 与 “Dropping group” 的条数；`scheduler/cancelled_rollouts`；`last_batch_generation_time`；`begin_training()` 的耗时，即 `train/cycle_duration_ms` 减去 `train/publish_duration_ms`、`train/pack_duration_ms`、`train/batch_wait_duration_ms`、`train/batch_receive_duration_ms` 与 `train/duration_ms`（余下的就是 `begin_training()` 和几行簿记）。期望 `fifo` 下没有满载 429、没有 RateLimitError 与丢组，`begin_training()` 在一轮解码之内返回。

## 原生 GRPO SharedServer 睡眠前释放保留的 KV（ledger T29）

dense Qwen3/Qwen3.5 的 `grpo-colocate` 用原生 SharedServer。每步进入训练时，`GenerationService::begin_shared_training` 等 rollout 排空后让引擎睡眠；上游睡眠时已完成 lane 的前缀 KV 仍被保留，`ElasticKvRegion::sleep` 为每个还有页的 granule、每个 plane 分配一块新的锁页内存（`cudaMallocHost`，Qwen3 一块 2 MiB）并同步拷到主机；发布新 adapter 时 `publish_shared_adapter` 唤醒引擎，逐块映射、拷回、`cudaFreeHost`，然后立即 `shrink_kv()` 把它们全部驱逐。两者之间不接纳任何请求（`acquire_request_lifetime` 与 `wake_up` 在训练阶段拒绝），所以备份和恢复的字节没有人读。T29 在 `sleep()` 之前调用 `shrink_kv()`，驱逐的正是发布时会驱逐的那些前缀，结果不变，默认打开；`SUROGATE_SHARED_SLEEP_KEEP_KV=1`（每个进程读一次）恢复上游顺序，用于 A/B。

预期（按代码与 CPU 上的锁页代理估算，未在 GPU 上测）：备份量是保留的上下文按 1024 token 一个 granule 取整，Qwen3-0.6B 每 token 112 KiB、Qwen3-8B 144 KiB，上限是 KV 池（上下文 × `max_num_seqs`，2048 × 16 时 0.6B 3.76 GB、8B 4.83 GB）；约 1k token 的上下文、16 个 lane 时约 1.8-2.3 GB。省下的是首次锁页（约 2-3.5 GB/s，0.6-2.4 s）、两次拷贝（0.1-0.4 s）和 `cudaFreeHost`（0.2-0.5 s），每步约 0.8-3 s，随上下文和 `max_num_seqs` 增长。`SUROGATE_SLEEP_DEBUG` 的锁页与拷贝计时只覆盖 dense Offload 区域，不含 elastic KV 的钩子，看不到这部分；`engine: asleep, released …`/`engine: awake, restored … GiB` 两行总会打印，按映射的字节计。

验证（没有 GPU）：源码 pin `fork/patches/tests/native/test_shared_sleep_kv.py`（`fork/dev test patches`）在改动前失败，并检查上面论证所依据的上游代码；改动在 upstream/main `82d42c72` 上同样适用，pin 也通过；`generation_service.cpp`（`sinfer_serve` 的主机翻译单元，按 sm_120a 配置）编译通过，`-Wall -Wextra` 下改动的几行没有警告。

5090 检查（未验证）：W-0.6B-GRPO（Qwen3-0.6B LoRA colocate，`max_num_seqs` 16，`max_tokens` 1536）与 W-8B-GRPO 各跑 20 步，带与不带 `SUROGATE_SHARED_SLEEP_KEEP_KV=1`。每步记录：发布耗时 `train/publish_duration_ms`；`begin_training()` 耗时（同上一节的差值）；`engine: awake, restored X GiB` 一行，T29 下应少掉保留 KV 的那部分；`nsys profile --cuda-memory-usage=true` 后 `nsys stats -r cuda_api_sum` 中每步的 `cudaMallocHost`/`cudaFreeHost` 次数，应从约（有页的 granule 数 × 层数 × 2）降到约 0；主机锁页内存的峰值；发布后第一轮的前缀命中与 rollout token 应不变。成功标准：0.6B、C=16 时每步至少省 1 s。

## GRPO 训练路径 rollout 服务的主机开销

`shared_execution()`（`surogate/grpo/shared_model.py`）返回 `"training"` 的模型由 Python 的 `SharedModelServer` 生成 rollout：全部 MoE 与 GLM 模型、Qwen3/Qwen3.5 以外的稠密模型，以及 LoRA 目标或工具协议不标准的 Qwen。每个请求在一个 HTTP 处理线程中逐 token 运行 `_generate`，与 decode 调度线程和同进程的 orchestrator 共用一个 GIL；一个请求处理完自己这一步的 token 之后，才能把下一步交给调度器，所以每个 token 的主机工作直接落在 decode 轮次之间。

### 非流式请求只在结束的那一步构造文本（T11）

上游 `_generate` 每生成一个 token，都把整个 completion 重新解码一次、在全文中查找停止字符串、再对全文运行 `protocol.parse`：每个 token O(n)，每个请求 O(n²)。非流式请求（GRPO orchestrator 的请求都是非流式的）只用得到最后一步的结果：`message` 每步重建，只返回最后一个；`delta` 和 `emitted_fields` 只供流式回调使用；不结束的步上 `parse(final=False)` 只做字符串查找和切片，工具定义已由 `request_tools` 校验，不会抛异常。

`surogate/_fork/grpo_detok.py` 的 `CompletionEnd.ends()` 判断一步是否结束 completion：这一步的 token 是停止 token，或这是允许的最后一步，或（与上游相同，从第 `minimum - 1` 步起）`tokenizer.decode(ids)` 中出现了停止字符串。不结束的步跳过文本块；结束的那一步照常运行上游的文本块（一次完整解码、停止截断、一次 `parse(final=True)`、构造 message），返回的就是上游自己的计算结果。

| `SUROGATE_GRPO_DETOK` | 行为 |
|---|---|
| `final`（默认） | 如上。请求有停止字符串时，从第 `minimum - 1` 步起每步仍对全文解码一次来检查（解码开销与上游相同，但不再 parse）；没有停止字符串时，一步不做任何文本工作 |
| `incremental` | 同 `final`；另外，tokenizer 是未改动的 transformers fast tokenizer 且 decoder 是 ByteLevel（Qwen、Llama 3、GLM、DeepSeek、GPT-OSS 等）时，停止字符串检查只解码最近一个完整字符之后的 token，只扫描一个与最长停止字符串等长的窗口。其他 tokenizer 用 `final` 的检查 |
| `per_token` | 上游的逐 token 循环 |

每个请求读取一次该变量，非法值使请求以 400 失败。流式请求（有回调）始终走上游循环。

`final` 默认开启，因为它只去掉结果被丢弃的主机工作：对同一个 token 序列，上游能完成的每个请求得到的文本、message、logprobs 和 finish reason 都与上游相同。唯一的差别出现在失败的情况：如果某个 tokenizer 的 `decode` 对 completion 的某个前缀抛异常、对完整的 completion 不抛，上游会在那一步失败，这里不会。HF fast tokenizer 不会出现这种情况：未知 id 被丢弃，非法 UTF-8 被替换。

这个等价是对单个请求、给定的 token 序列而言的，并不是说 GPU 收到的调用不变。trainer 没有 `decode_batch_logits` 时没有调度器，请求在 `compute_lock` 下逐个执行，GPU 收到的调用与上游相同。有调度器时（连续批处理；`binding.cpp` 总是为 `SurogateTrainer` 绑定这个方法，所以用当前原生扩展的训练路径都走这里），每一步经 `DecodeScheduler.step` 与其他 session 的步合成一轮，交给 `decode_batch_sample` 或 `decode_batch_logits`，哪些 session 同处一轮取决于各 session 交出下一步的时刻。T11 缩短了一个 session 拿到这一步的结果到交出下一步之间的主机时间（下文每个 token 143–626 µs 降到约 7 µs），因此与任何主机加速（更快的 CPU、更低的负载）一样，改变了轮次的数目、大小和组成：下文 rollout 模拟中 2.8 ms、等长的一行，`per_token` 为 1176 轮、平均每轮 13.93 个 session，`final` 为 1074 轮、15.26 个（`grpo_rollout_host.py rollout` 输出的 `rounds` 与 `mean_batch`）。批量 decode kernel 对不同 batch 不保证逐位相同的结果（上游 `tests/grpo/test_batched_decode_gpu.py` 比较批量与逐个 decode 时，GLM 以外的模型只要求 `atol=0.1`），所以并发高于 1 时，采样出的 token 只在上游自身的时序不确定性范围内与上游一致，不能逐位比较。上游的轮次组成本来就取决于主机时序，同样的配置跑两次也可能不同；T11 没有引入新的不确定来源，这是它默认开启的依据。

`incremental` 的结果等于上游的依据（tokenizers 0.23.3 的 Rust 源码与 transformers 5.21，推导写在 `_ByteLevelTail` 的 docstring 中）：`decode` 把每个 id 映射成 token 字符串后交给 decoder；ByteLevel decoder 把每个 token 各自转成字节，对整串字节做一次 `from_utf8_lossy`；一段字节的 lossy 解码若不以 U+FFFD 结尾，就停在完整字符的边界上，之后的字节与它无关。因此已提交部分的文本是之后每一步文本的前缀，而上一步文本中没有停止字符串，新出现的匹配只能从已提交文本的最后 `len(stop) - 1` 个字符或更晚处开始。窗口文本以 U+FFFD 结尾时（字符还没写完）继续累积 token；超过 16 个 token（一个不完整字符最多 4 个字节 token，只有连续的非法字节会到这里）时，这个请求改用全文检查。是否使用窗口由 `byte_level_backend()` 判断：类上与实例上都没有覆盖 `decode`/`_decode`，decoder 的类型恰好是 `tokenizers.decoders.ByteLevel`，transformers 不会对全文做 `clean_up_tokenization`，并用一段包含全部清理模式的探针文本确认 `tokenizer.decode` 与后端 `decode` 一致。这个论证依赖已安装的 tokenizers 与 transformers 的内部实现，而检查只看类和探针、看不到 Rust 代码，所以 `incremental` 默认关闭。

### 按存活 session 决定 decode 轮次的大小（T12）

`DecodeScheduler._run` 最多等 1 ms，直到有 `max_batch` 个待执行的步。存活的 session 少于 `max_batch` 时（每个 rollout 阶段的尾部，或 `max_concurrency` 高于实际并发时）这个数永远凑不够，每一轮都等满 1 ms。session 从 `new_session` 到 `release` 之间是存活的，每个 session 同时最多有一个待执行的步。`SUROGATE_GRPO_LIVE_DISPATCH=1` 时，所有存活 session 都交了一步（`min(max_batch, 存活数)`，至少 1）就开始一轮；截止时间以及遇到 release 或关闭时提前结束等待的规则不变，所以一轮等待的时间不会比上游长。等待期间新建的 session 交的步进入下一轮。

该开关默认关闭，调度器在构造时读取一次。它对结果的影响与 T11 在连续批处理路径上的影响属于同一类：轮次形成的时刻不同，各轮的 batch 组成也就不同，而批量 decode kernel 对不同 batch 不保证逐位相同的结果；上游的轮次组成本来就取决于时序，同样的配置跑两次也可能不同。两者的区别在改动的性质：T11 只去掉结果被丢弃的工作，轮次时刻的变化是任何主机加速都有的副作用；T12 改的是调度器决定一轮何时开始的规则本身，这样的调度策略改动在 5090 上测出收益之前不默认开启。

### 测量

主机与上文 CPU 优化器的测量相同（4 vCPU KVM 虚拟机，Intel Xeon family 6 model 207，2.1 GHz，没有 GPU，与其他会话共用，测量期间 load average 0.3–2.5）。Python 3.12、tokenizers 0.23.3、transformers 5.21.0。tokenizer 由 `fork/patches/bench/grpo_rollout_host.py tokenizer <目录>` 生成：在检出目录的 markdown 和 `surogate/` 源码上训练的 32k ByteLevel BPE，使用 Qwen2 的切分正则和聊天、工具特殊 token；completion 是同一批文件（中英文 markdown 与 Python 代码）编码后的前 L 个 token。

每个 token 的主机开销：`grpo_rollout_host.py per-token <目录>`，直接调用 `_generate`，采样器按脚本返回 token（没有调度器、没有 GPU），5 次的中位数，`logprobs=True`。“有停止字符串”的请求带 `["</answer>", "\n\n\n\n"]`，两者都不出现，所以每一步都要检查。三种模式在同一 L 下的响应摘要相同。

| L（token） | 改动前的代码 | `per_token` | `final`，无停止字符串 | `final`，有停止字符串 | `incremental`，有停止字符串 |
|---:|---:|---:|---:|---:|---:|
| 1024 | 143 µs | 145 µs | 7.3 µs | 142 µs | 9.0 µs |
| 2048 | 336 µs | 306 µs | 11.4 µs | 361 µs | 8.6 µs |
| 4096 | 626 µs | 587 µs | 7.0 µs | 653 µs | 8.7 µs |

“改动前的代码”是在 T11 所基于的 `488e5f00` 上用同样的输入测得的（T11 当时在单独的分支上）。本分支落地后 T11 的父提交是 `6450868b`，其中的 `_generate` 还有 T28 每步一次的 `check_client()`。复现这一列可以在当前代码上设 `SUROGATE_GRPO_DETOK=per_token` 运行 `grpo_rollout_host.py per-token <目录>`（即表中 `per_token` 一列）：这个模式跑上游的逐 token 循环，每步另有 T28 的 `check_client()` 和 T11 的一次 `view is not None` 判断；脚本不经构造函数建服务端，给它的 admission 对象没有记录连接，`check_client()` 立即返回。也可以把当前的脚本放到 `488e5f00` 检出目录的同一路径下运行：那里还没有这个脚本，代码也不读 `SUROGATE_GRPO_DETOK`，输出中的 `mode=` 没有意义；completion 取自所在检出目录中的文件，所以两处的 token 序列和响应摘要不同。另两次重复测量中 `final` 无停止字符串为 6.6–10.6 µs，`incremental` 有停止字符串为 8.4–14.3 µs（load average 到 3.5）；同一列不同 L 之间的差别是虚拟机的噪声，剩下的开销与 L 无关。`incremental` 第一次用到窗口时检查 tokenizer 一次（约 0.5 ms），结果按 tokenizer 缓存。剩余的约 7 µs 中，每个 token 的 `tokenizer.decode([token])`（logprob 条目里的 token 文本）约 2.9 µs，其余是采样参数字典、score 字典和脚本采样器本身。

整个 rollout 路径：`grpo_rollout_host.py rollout <目录> <毫秒>`，16 个请求在真实线程上经 `generate()` 和真实的 `DecodeScheduler` 运行，一轮 GPU 计算用释放 GIL 的 `time.sleep` 代替（原生 decode 调用同样释放 GIL），每轮耗时固定：2.8 ms 是对 1B 稠密模型 B=16 一轮的估计，12.3 ms 是对每轮读取全部专家的约 10B 总参数 MoE 的估计（20 GB 权重 / 1.53 TB/s，加上 KV）。“等长”是 16 个请求都生成 1024 个 token；“不等长”是 256–1024 之间的固定随机长度，模拟 rollout 阶段的尾部。理想值假设每轮都包含全部存活请求。3 次的中位数：

| 一轮 | 长度 | `per_token` | `final` | `final` + `SUROGATE_GRPO_LIVE_DISPATCH=1` |
|---|---|---:|---:|---:|
| 2.8 ms | 等长（理想 5714 tok/s） | 1704（29.8%） | 2938（51.4%） | 2860（50.0%） |
| 2.8 ms | 不等长（理想 3669 tok/s） | 1393（38.0%） | 2005（54.6%） | 2158（58.8%） |
| 12.3 ms | 等长（理想 1301 tok/s） | 759（58.3%） | 1042（80.1%） | 1056（81.2%） |
| 12.3 ms | 不等长（理想 835 tok/s） | 576（68.9%） | 692（82.8%） | 706（84.5%） |

文本改动使稠密模型的 rollout 快 1.4–1.7 倍，MoE 快 1.2–1.4 倍（不等长 / 等长）；剩下的差距主要是每个 token 一次的线程交接：每一步一个 `Future`、一次 `Condition` 通知和一次处理线程的唤醒（`DecodeScheduler.step`），这里没有改动它。存活 session 调度只在长度不等时起作用（等长一行的差别在噪声范围内），这一组数据中不等长时稠密 +7.6%、MoE +2.0%。同一模拟的另一组测量（同一台虚拟机，早些时候）得到稠密等长 1659 → 3022、不等长 1444 → 2070 → 2267，MoE 等长 781 → 1031、不等长 598 → 694 → 719 tok/s。

### 在 RTX 5090 上尚未验证

以上吞吐量都来自 CPU 模拟。对给定的 token 序列，每个请求的响应与上游相同，这一点由 CPU 测试覆盖（`fork/patches/tests/grpo/test_shared_model_detok.py` 用随机生成的 completion 逐字段比较三种模式，包括拆开的多字节字符、非法 UTF-8、U+FFFD、特殊 token、跨 token 的停止字符串、`min_tokens`、停止 token、工具调用与 Harmony，并覆盖 SentencePiece 风格、WordPiece 和强制清理的 tokenizer）。token 序列本身在连续批处理路径上随轮次组成变化（见 T11 一节），CPU 测试不覆盖这部分：并发为 1 时每轮只有一个 session，轮次组成与主机耗时无关，GPU 上应与上游逐位相同；并发高于 1 时只能比较分布。下面各项都没有在 5090 上运行过：

1. 在一个走训练路径的模型上（例如约 10B 总参数的 qwen3_moe LoRA，`max_tokens` 1536，16 个 prompt × 8 个 rollout），用同一份配置运行 `surogate grpo-colocate --train <train.yaml> --infer <infer.yaml> --orch <orch.yaml>` 至少 50 步三次：一次设 `SUROGATE_GRPO_DETOK=per_token`，一次不设，一次不设并设 `SUROGATE_GRPO_LIVE_DISPATCH=1`。比较 `train/pack_duration_ms`（colocate 下主要是等 rollout）和 `train/cycle_duration_ms` 的中位数，以及有效 batch：每步生成的 completion token 数除以 `<output_dir>/shared_weights.jsonl` 中相邻两条 `rollouts` 记录的 `decode_rounds` 之差（含 prefill 轮次）。预期 MoE 的 rollout 至少快 1.15 倍、尾部的有效 batch 接近 `min(16, 存活数)`；1B 左右的非 Qwen 稠密模型至少快 1.5 倍。
2. rollout 期间对服务进程运行 `py-spy dump --native --pid <pid>`：默认模式下处理线程的栈中不应出现 `tokenizer.decode` 解码整个 completion 或 `parse_output`。
3. 第 1 项的三次运行中 orchestrator 的请求不固定 seed，token 不可逐位比较，比较分布：每步 reward 均值和 completion 长度在三次之间应无系统差异。
4. 并发为 1 时与上游逐位相同：用一个走训练路径的真实 checkpoint（例如 qwen3_moe）和它的 tokenizer，按 `tests/grpo/test_shared_model_gpu.py` 中 `_check_policy` 的方式建 trainer 和 `SharedModelServer`，但 `max_concurrency=1`；对同一组 prompt（至少 8 个，`max_tokens` 不少于 256，其中一半带 `stop` 和 `tools`）用固定的 `seed`、`temperature=1`、`logprobs=True`、`top_logprobs=5` 逐个请求 `/v1/chat/completions`，一次设 `SUROGATE_GRPO_DETOK=per_token`、一次不设（模式按请求读取，可以在同一个进程中两次请求之间切换），要求 `token_ids`、`logprobs`、`message` 和 `finish_reason` 完全相同。
5. 默认模式（不设 `SUROGATE_GRPO_DETOK`）下运行经过 `SharedModelServer` 的上游 GPU 测试：`SUROGATE_SHARED_CASES=all .venv/bin/python -m pytest -q --slow tests/grpo/test_shared_model_gpu.py tests/grpo/test_tool_protocol_gpu.py tests/grpo/test_routing_capture_gpu.py`，再设 `SUROGATE_GRPO_LIVE_DISPATCH=1` 运行一次 `tests/grpo/test_shared_model_gpu.py`。其中 `_check_policy` 要求两个并发的生成与单独一次生成的 `token_ids` 相同、`max_decode_batch >= 2`，并且 rollout 的 logprobs 与打包打分一致。这些测试标记为 `slow`，`fork/dev test gpu` 中的 `make test-py` 不带 `--slow`，不会运行它们。

## 服务端 GQA 形状注册

ledger T16，开关 `SUROGATE_SERVE_GQA_FORK_SHAPES=1`，默认关闭。

上游的 GQA 形状注册表（`csrc/src/serve/ops/kernel/gqa_attention_geometry.cuh` 的 `SINFER_GQA_FOR_EACH_GEOMETRY`）没有收录 128 宽的 32q/8kv 与 40q/8kv。在 Hopper 以外的 GPU（RTX 5090、GB10）上，这两种形状的解码、验证（MTP/DFlash 最多 16 列）和 7–63 列的 prompt 段都走通用 CUDA-core kernel（`gqa_attention_generic.cu` 的 `gqa_generic_split_kernel`）：每个 (列, query head, split) 一个 CTA，同一个 KV head 的 K/V 按 query head 各读一遍（G=4 时 L2 到 SM 的流量是 KV 本身的 4 倍），用 FP32 FMA 计算，且 `q_heads × 列数` 达到 2 × SM 数（5090 上 340）后不再切分历史。64 列及以上的 prompt 已经走 tensor-core prompt kernel，不受影响。128 宽的 MHA 32q/32kv（Llama-2-7B 一类）此前没有任何 128 宽的注册形状带 32 个 KV head，直接报“unregistered head geometry”。

fork 按上游的注册方式加入三个形状：在 `Gqa128_32q4` 之后加 `using` 与 `X()` 行，每个形状一个 small-T 解码翻译单元。翻译单元放在 `fork/native/serve/`，由 `fork/native/cmake/fork_native.cmake` 的 `_surogate_fork_attach_serve()` 加入 `sinfer_ops`（`csrc/CMakeLists.txt` 不改）。路由由 `fork/native/include/surogate_fork/gqa_fork_shapes.h` 决定，上游按“是否注册”分路由的三处都读它：`wrapper/gqa_attention.cpp` 的 `optimized_decode_shape` 与 `generic_takes`，以及 `launcher/gqa_attention_prefill.cu` 的 `dispatch_prompt_geometry`。

接受也由它决定。`X(Gqa128_32q32)` 让 (128 宽, 32 个 KV head) 成为注册过的 KV 对，上游的 KV 接受检查（`gqa_kv_shape_is_registered`）因此会对落在这个 KV 对上的任何 query group 放行。开关关闭时，`gqa_fork_kv_pair_withheld` 扣下只有 fork 的行才带的 KV 对；它由注册表推出，上游以后注册同一 KV 对的形状时不会被扣下。wrapper 的四处 KV 检查都同时看它：`supported_attention_shape`（它也决定 workspace 规划和 `gqa_attention_packs_prompt`）、`require_kv_heads`、`validate_cache` 与 `validate_batch_cache`。所以开关关闭时，128 宽 MHA 模型与注册前一样在 workspace 规划处就被拒绝。上游 `test_gqa_attention.cpp` 的 `verify_geometry_registration_contract` 要求每个注册形状都被接受，fork 让它跳过被扣下的 KV 对（一个条件，加上对 fork 头文件的 include）。

| 形状 | 模型 | 开关关闭（默认） | 开关打开，BF16 或 e4m3 KV | 开关打开，int8 KV |
|---|---|---|---|---|
| `Gqa128_32q8` | Qwen3-4B、Qwen3-8B、Llama-3-8B、Mistral-7B | 与注册前相同：通用 kernel；64 列起用 KV 几何 `Gqa128_16q8` 的 prompt kernel | small-T tensor-core kernel（GQA 组装进 MMA 的行，KV 每个 head 只读一次；固定的 key 分区与 reducer），7–63 列走 split-KV tile，多 prompt 的轮次可以打包；64 列起仍是 prompt kernel | 与关闭相同 |
| `Gqa128_40q8` | Qwen3-14B | 同上 | 同上；组大小 5，small-T 直接服务宽度 1–6 和 8，更宽的验证轮按 6 列分块或拆成 prompt tile | 与关闭相同 |
| `Gqa128_32q32` | Llama-2-7B 一类的 128 宽 MHA | 与注册前相同：拒绝（workspace 规划、KV 追加与注意力调用都抛 `std::invalid_argument`，报 head geometry 不受支持） | 同第一行 | 通用 kernel；64 列起是 KV 几何的 prompt kernel，即 `Gqa128_32q32` 自己的实例 |

- int8 KV 无论开关都留在通用 kernel：int8 的 small-T 解码 kernel 只服务 256 宽、组大小 4–8 的形状（`gqa_attention_decode_launch.cuh` 的 `kGqaI8DecodeRegistered`），否则会抛异常；通用 kernel 在 128 宽上服务 int8。按代码阅读，上游自己的 128 宽注册形状（Qwen3-0.6B 的 16q8、Qwen3-30B-A3B 的 32q4 等）在 int8 KV 的第一步解码就会抛这个异常；fork 没有改动这一点。
- 数值：开关打开后输出的低位会变化。通用 kernel 的 P 保持 FP32，tensor-core kernel 在 P·V 之前把 P 舍入为 BF16。key 分区按绝对位置固定（与已注册形状相同），所以 batch 不变性保持；`--batch-invariant` 下通用 kernel 不切分历史，注册路线照常按这些固定分区切分。
- Hopper：BF16 KV 的解码与验证行仍走 FA3（`fa3_takes_rows` 不看注册表），FA3 能接的 prompt 也仍走 FA3；开关改变的是 e4m3 KV 的行、FA3 不接的窄 prompt 段和带 mask 的段，以及 `--batch-invariant` 下的所有宽度。
- 不在范围内：其他未注册的组（例如 Qwen2.5-7B 的 28q/4kv、Qwen3-32B 的 64q/8kv）照旧走通用 kernel。开关打开时，落在 (128, 32) 这个 KV 对上的其他组（例如 64q/32kv）也被接受，按未注册的组走通用 kernel；开关关闭时与注册前一样被拒绝。

预期收益来自代码阅读和计数（路线图 serve R1，未在 GPU 上测量）：Qwen3-8B、BF16 KV、16 条 4K rollout 时每步注意力约 7–14 ms 降到约 5.5–6 ms，解码约 1.1–1.6 倍；在 30K 历史上追加 40 个 token 的一轮约 30–180 ms 降到 3–5 ms。

内存的四个方面：

- 分配：运行时没有新的分配调用。开关打开后，注意力 workspace 按已注册形状的公式预留，与其他已注册形状相同，而不是通用 kernel 的公式。引擎用 `gqa_attention_history_workspace_capacity_bytes` 规划（`family/impl/runtime/attention_workspace.h` 与 `layouts_impl.h`），开关打开后它对这些形状不再提前返回。下面的数字都是代码算术，未在 GPU 上核对；S 是 envelope 最大可见 key 处的固定分区数，默认分区表（Hopper 以外）下 4K 为 32、32K 为 64、128K 为 112。
  - 解码与验证（small-T 分支）：FP32 partial 约为 4·D·H_q·W·S·B 字节。通用 kernel 只在 `q_heads × 列数` 小于 2 × SM 数时才切分，B·W 较大时不需要 partial。Qwen3-8B（D=128，H_q=32）在 32K 上下文、16 条 lane 的解码约 17 MB，验证宽度 4 时约 67 MB。
  - 单序列的 prompt 规划（预填充、prompt 分块，batch 为 1、宽度跨过 7–63 列）多出两项，取二者中较大的一个，不相加：7–63 列 prompt 段的 tile partial（`exact_prompt_capacity`，tile 宽度与 lane 数由 `prompt_query_capacity` 按 32 MiB 封顶选出），Qwen3-8B 与 MHA 在 32K 上是 16 列一个 tile、约 17 MB，Qwen3-14B 是 8 列 × 3 个 lane、约 32 MB；以及 `gqa_attention_history_workspace_capacity_bytes` 为较短历史预留的区间载荷，min(32 MiB, (D+2)·H_q·S·4 × min(tiled_max, 128)) + 2 KiB，其中规划的最大宽度由 prompt kernel 接时 tiled_max 是 63，三个形状在 32K 上都到达 32 MiB 的上限，即 32 MiB + 2 KiB。开关关闭时这些形状的单序列规划只有通用 kernel 切分时的 partial（Hopper 上另有 FA3 的 scratch，两种设置相同）。
  - 以引擎启动时报告的内存摘要为准：在 GPU 上对解码规划和预填充规划各记录开关 0/1 两个值（见下文的 5090 检查）。
- 存活时间、峰值：运行时不变（workspace 预留在启动时确定）。
- 构建：每个 `X()` 行除了解码单元，还会让 `gqa_attention_prefill.cu` 与 `gqa_attention_decode.cu` 为该形状实例化一套 kernel；本机的编译时间与目标文件大小见下文“本机编译检查”。
- 吞吐：解码 tok/s，见下文的 5090 检查。

### 验证

无 GPU 时：`fork/dev test patches` 中的 `fork/patches/tests/native/test_gqa_fork_shapes.py` 从上游源码中截取 `supported_attention_shape`、`require_kv_heads`、`optimized_decode_shape`、`generic_takes`、`gqa_generic_attention_serves` 和 `dispatch_prompt_geometry` 的原文，与真实的注册表头文件和 fork 头文件一起用主机编译器编译成探针，在开关未设置、为 `0` 及其他值、为 `1` 时运行：注册表含这三个形状且 `(128, 8)` 的第一个匹配仍是 `Gqa128_16q8`；开关关闭时 32q8 与 40q8 的路线与未注册的组相同，MHA 与落在 (128, 32) 上的其他组被拒绝；打开时三个形状都被接受，BF16/e4m3 走注册路线、int8 不走；其他形状两种设置下接受与路线都相同。它还检查 `validate_cache` 与 `validate_batch_cache` 的 KV 检查和上游注册契约的豁免（按源码文本）、`csrc/src` 中注册表谓词与分发函数的全部调用点没有增减（上游新增调用点时失败，见 [sync.md](sync.md) 的 T16）、每个 `X()` 行在某个 CMake 源列表中恰好有一个解码单元、下面三个 ctest 条目的接线，以及本节写有开关与 GPU 检查命令。这些 kernel 与受影响的翻译单元已为 sm_120a 编译（见下文），没有在 GPU 上运行。

GPU 测试。上游的 `sinfer_gqa_attention_test` 只覆盖 32q/8kv（qwen3_8b 的几组用例，e4m3 缓存的参考对照是 `verify_fp8_wide_prompts` 中的三组：200 列的 prompt 和宽度 1、3 的行），不覆盖 40q/8kv 和 128 宽 MHA。`fork/native/tests/test-fork-gqa-shapes.cpp` 把上游的测试源文件原样编译进来（只把它的 `main` 改名），对三个形状跑同类用例：解码与验证宽度（追加 A1、读缓存 A3）、7–63 列与 64 列以上的 prompt、跨页和跨分区的历史、滑动窗口、带 mask 列与重排 table row 的 batch，以及 int8 的 batch 行（两种设置下都走通用 kernel），对照该文件的双精度参考（BF16 与 int8 缓存）。e4m3 缓存另有一组参考对照（`verify_fork_shape_fp8_against_reference`，仿照上游的 `verify_fp8_wide_prompts`）：用 `gqa_kv_append` 写入分散页上的 e4m3 缓存，按缓存中实际存放的 code 以双精度计算参考，宽度 1、3、6、17、66，历史约 2000 个 key，有无 300 的窗口，默认与 `--batch-invariant` 各一次（66 列再加 prompt kernel 的 FP8 query-key 积，参考同样舍入 query），界限与上游相同；另外 e4m3 的追加与读缓存逐位相同。仿照上游的 `verify_packed_prompt_invariance` 检查多 prompt 打包：开关打开时打包的输出和缓存与各段单独调用逐位相同，关闭时不打包。开关关闭时 MHA 不跑这些用例，而是检查它被拒绝：两种 workspace 规划、`gqa_kv_append` 和 `gqa_attention_cached` 都抛带“geometry”字样的 `std::invalid_argument`，也不打包。`fork_native.cmake` 把它构建为 `sinfer_gqa_fork_shapes_test`（与上游测试同目录，属于 `serve-tests`），并登记三个 ctest 条目：`sinfer_gqa_fork_shapes_test`（开关为 0）、`sinfer_gqa_fork_shapes_on_test` 与 `sinfer_gqa_attention_fork_shapes_on_test`（上游测试，开关为 1）。

### 本机编译检查

2026-10-31，4 vCPU 虚拟机，无 GPU，nvcc 13.1.115，GCC 13.3，`SUROGATE_SERVE_CUDA_ARCHS=120a`（只编 sm_120a；默认的 `89;90a;120a` 大约是三倍），与其他会话共用机器。在仓库外配置一个临时构建目录（`cmake -S csrc`，打开 `SUROGATE_BUILD_SERVE`、`SUROGATE_SERVE_TESTS` 与 `SUROGATE_SERVE_BENCH`），只编译受影响的翻译单元，之后删除。

| 翻译单元 | 注册前 | 注册后 |
|---|---|---|
| `gqa_attention_prefill.cu` | user 15 分 39 秒，17.32 MB | user 17 分 16 秒（+10%），18.92 MB（+9%） |
| `gqa_attention_decode.cu` | user 5.2 秒，136 KB | 约 7 秒，150 KB |
| `fork/native/serve/` 三个解码单元 | — | 各约 200 秒（3 个并行），4.35–4.59 MB |
| `wrapper/gqa_attention.cpp`、`gqa_attention_generic.cu`、`test_gqa_attention.cpp`、`test-fork-gqa-shapes.cpp` | | 编译通过 |

ptxas（`--resource-usage`）给新解码单元中 small-T kernel 的寄存器是 244–255，spill store 最多 80 B（40q8；32q8 为 76 B，32q32 为 40 B），上游已注册的 `Gqa128_32q4` 单元是 244–255 与 76 B。`fork_native.cmake` 的接线在该构建目录中确认：三个解码单元在 `sinfer_ops` 的编译命令中，`sinfer_gqa_fork_shapes_test` 属于 `serve-tests`，三个 ctest 条目带各自的环境变量。没有链接，也没有在 GPU 上运行。

同日第二轮（开关关闭时扣下 MHA 的 KV 对、加入 e4m3 参考用例之后；fork 头文件现在包含注册表头文件），同样的配置在一个新的临时构建目录中重新编译了 `wrapper/gqa_attention.cpp`、`gqa_attention_prefill.cu`（user 约 16 分钟，与另外两个单元并行）、三个 fork 解码单元、`test_gqa_attention.cpp` 与 `test-fork-gqa-shapes.cpp`，全部通过；`sinfer_gqa_attention_test` 的编译命令带 fork 头文件目录。同样没有链接，也没有在 GPU 上运行。

RTX 5090 上（`make serve-test-build` 的构建目录 `csrc/build-serve`，`SERVE_TEST_CUDA_ARCH=120a`）：

```bash
make serve-test-build
# 正确性：四个条目都应通过（上游测试开关关闭的一次就是它原有的条目）。
cd csrc/build-serve && ctest -R '^sinfer_gqa_(attention|fork_shapes)' --output-on-failure && cd -
# 等价的手动运行，每次都应打印 PASS：
csrc/build-serve/serve_tests/sinfer_gqa_attention_test
SUROGATE_SERVE_GQA_FORK_SHAPES=1 csrc/build-serve/serve_tests/sinfer_gqa_attention_test
SUROGATE_SERVE_GQA_FORK_SHAPES=0 csrc/build-serve/serve_tests/sinfer_gqa_fork_shapes_test
SUROGATE_SERVE_GQA_FORK_SHAPES=1 csrc/build-serve/serve_tests/sinfer_gqa_fork_shapes_test

# 速度：同一构建，只切换开关；锁定或记录 SM 时钟，每组至少 3 次，冷缓存。
# B>1 时 bench 只接受 W<=16，所以解码行与 40 token 的一轮分两条命令。
cmake -S csrc -B csrc/build-serve -DSUROGATE_SERVE_BENCH=ON
cmake --build csrc/build-serve --target sinfer_causal_softmax_attention_bench
B=csrc/build-serve/serve_bench/sinfer_causal_softmax_attention_bench
for s in 0 1; do
  SUROGATE_SERVE_GQA_FORK_SHAPES=$s $B --entry append --geometry d128-h32-kv8 --kv-dtype bf16 \
    --tokens 1 --batch 1,16,32 --context 4096,32768 --execution graph --cache cold \
    --csv-out gqa_fork_shapes_decode_$s.csv
  SUROGATE_SERVE_GQA_FORK_SHAPES=$s $B --entry append --geometry d128-h32-kv8 --kv-dtype bf16 \
    --tokens 40 --batch 1 --context 32768 --execution graph --cache cold \
    --csv-out gqa_fork_shapes_turn_$s.csv
  # int8 留在通用 kernel：两种设置的时间应在噪声内相同。
  SUROGATE_SERVE_GQA_FORK_SHAPES=$s $B --entry append --geometry d128-h32-kv8 --kv-dtype int8 \
    --tokens 1 --batch 16 --context 4096 --execution graph --cache cold \
    --csv-out gqa_fork_shapes_int8_$s.csv
done
```

bench 只有 32q/8kv 这一个 128 宽的组（`d128-h32-kv8`）；40q/8kv 与 MHA 的速度在端到端运行中测量。在同一张卡上还应记录：

- ncu 对 `gqa_generic_split_kernel` 与 small-T kernel 的 `lts__t_sectors_srcunit_tex.sum`（相对于 KV 本身的字节数，预期由约 4 倍降到约 1 倍）、`dram__bytes_read.sum`、`smsp__pcsamp_warps_issue_stalled_long_scoreboard` 的采样占比和 `sm__pipe_tensor_op_hmma_cycles_active`。
- 端到端：Qwen3-8B，16 条并发 4K rollout，开关 0/1 各一次，记录解码 tok/s、nsys 中注意力 kernel 的时间占比，以及 rollout 与 trainer 的平均 |Δ logprob|；Qwen3-14B 做一次开关 0/1 的解码 tok/s，一个 128 宽 MHA 模型在开关 1 下做一次（开关 0 时它应在启动时被拒绝）。
- 内存：Qwen3-8B 与 Qwen3-14B 在 32K 上下文下，开关 0/1 各启动一次，记录引擎启动时报告的内存摘要中解码规划（16 条 lane，以及验证宽度 4）和预填充规划的注意力 workspace，与上文“分配”中的估计对照。
- 成功标准（路线图 R1）：16 × 4K、BF16 KV 下解码 tok/s ≥ 1.15 倍；L2 到 SM 的 sector ≤ KV 本身的 1.3 倍；30K 历史上 40 token 一轮的注意力 ≤ 10 ms；int8 KV 照常服务。
