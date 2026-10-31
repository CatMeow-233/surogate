# Fork 性能说明

本文记录 fork 的性能改动、测量方法和开关。这些内容原先写在上游用户文档 `docs/guides/performance.md` 和 `docs/guides/metrics.md` 中；为让这两个文件与上游保持一致、同步时不产生冲突，现集中到这里。各节与 [ledger](../patches/ledger.toml) topic 的对应关系：

| 节 | ledger topic |
|---|---|
| [避免冗余工作与交接](#避免冗余工作与交接) | T1、T3、T4、T6、T7、T8、T19 |
| [不能捕获的整步只预热一次](#不能捕获的整步只预热一次) | T20 |
| [CPU 优化器并行](#cpu-优化器并行) | T5 |
| [LoRA 融合输入梯度](#lora-融合输入梯度) | T2 |
| [GRPO 周期计时](#grpo-周期计时) | T6 |
| [cpu_training 的 AdamW 状态 checkpoint](#cpu_training-的-adamw-状态-checkpoint) | T21 |
| [GPT-OSS 预量化 LoRA 的专家输入梯度](#gpt-oss-预量化-lora-的专家输入梯度) | T25 |

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
- 否则用调用线程当前掩码中的 CPU 数（每次进入这样的并行区都重新读取，一次 `sched_getaffinity` 系统调用），除以掩码与它重叠、且正在并行区中或离开并行区还不超过自己共享窗口的调用线程数（含它自己），至少 1，至多 `omp_get_max_threads()`。一个调用线程的共享窗口是它两次并行区之间最长空闲间隔的 2 倍，至少 1 秒（`kOmpSharerGapFactor`、`kOmpSharerWindowNs`）。

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

