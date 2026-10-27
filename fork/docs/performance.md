# Fork 性能说明

本文记录 fork 的性能改动、测量方法和开关。这些内容原先写在上游用户文档 `docs/guides/performance.md` 和 `docs/guides/metrics.md` 中；为让这两个文件与上游保持一致、同步时不产生冲突，现集中到这里。各节与 [ledger](../patches/ledger.toml) topic 的对应关系：

| 节 | ledger topic |
|---|---|
| [避免冗余工作与交接](#避免冗余工作与交接) | T1、T3、T4、T6、T7、T8 |
| [CPU 优化器并行](#cpu-优化器并行) | T5 |
| [LoRA 融合输入梯度](#lora-融合输入梯度) | T2 |
| [GRPO 周期计时](#grpo-周期计时) | T6 |

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

## CPU 优化器并行

训练库直接链接 `OpenMP::OpenMP_CXX`，因此其现有的 CPU AdamW、梯度范数和向量加法 pragma 的启用与服务端 encoder 无关。该 imported target 只对 C++ 源文件施加主机编译选项，不会把 `-fopenmp` 直接传给 nvcc。

安装匹配的编译器运行库，例如 GCC 的 `libgomp` 或 Clang 的 `libomp`。根据主机可用核数、内存带宽和 NUMA 布局选择 `OMP_NUM_THREADS`，不要假定线程越多越快。逐元素的 AdamW 更新保持原有算术；并行的浮点范数归约可能改变最终的舍入位。请在目标主机上验证梯度裁剪与数值容差，以及实际的优化器步耗时。

独立的接线回归测试需要 C++ 编译器和 OpenMP，但不需要 GPU 或 CUDA 头文件：

```bash
source .venv/bin/activate
cmake -S fork/native/openmp-contract -B build/fork-openmp -G Ninja
cmake --build build/fork-openmp --parallel 2
ctest --test-dir build/fork-openmp --output-on-failure
```

它检查训练用的 OpenMP 辅助函数能让一个真实的双 worker 主机库生效，并得到精确的整数输出。运行时的线程数限制可能导致显式跳过。该测试不编译 CUDA 训练库，也不验证 AdamW 数值。

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

为目标 GPU 配置好原生测试构建后，先运行这些聚焦的回归测试，再做模型基准（把 `BUILD_DIR` 换成该构建目录）：

```bash
cmake --build BUILD_DIR --target integration-tests dsl-run-state-tests --parallel 4
BUILD_DIR/integration-tests '[fp8][recipe]'
BUILD_DIR/integration-tests '[lora][fused-dx]'
BUILD_DIR/integration-tests '[moe][gemm][wgrad-zero]'
BUILD_DIR/dsl-run-state-tests '[dispatch-pp][host-copy]'
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
