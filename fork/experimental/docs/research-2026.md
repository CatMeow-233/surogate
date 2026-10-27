# 2026 架构与训练方法研究

更新：2026-10-24。目标是单张 RTX 5090 32GB、4–7B 稠密模型及约 10B **总参数** MoE 的微调与全参实验。

模块化与可维护性为第一准则。用户接受实验算法的收敛、数值和训练稳定性风险。下面按实验价值筛选，不将稳定性保证作为门槛；每个方向仍需有明确接口、状态契约与可复现验证。

本文是研究与接入设计，表中方法尚未因此成为框架已支持的功能。当前云环境没有 GPU，未测这些方法在 5090 上的吞吐或峰值显存。

## 年份与证据口径

- 区分首次公开年份、会议年份和软件发布日期。2025 年预印本在 2026 年发表，不记作 2026 首次提出。
- 优先读取作者代码、README、引文及仓库内论文。已读取 Engram、Attention Residuals 和 LLaDA2.2 的官方 PDF；部分 arXiv/OpenReview 正文受当前环境网络限制，未声称已通读全部正文。
- 方法比较、硬件数值和代码支持范围分别核实。作者的 H100/H800/多卡结果不视为本框架的 5090 测量。

## 架构：2026 首发候选

### Gated DeltaNet-2：分离擦除与写入

- 论文：[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)，2026-05；[NVlabs 官方代码](https://github.com/NVlabs/GatedDeltaNet-2)，README 标注 2026-05-21 发布。
- 将共享标量门扩展为 key 侧逐通道 erase、value 侧逐通道 write，并保留逐通道 decay，提高递归状态更新的表达能力。
- 重要接入机会：GDN/KDA 是其退化情形。若按原递推令新门退化为原标量广播，可研究已有线性注意力 checkpoint 的函数保持扩展，再继续训练。这是方程层面的转换设计，尚未验证 Qwen3.5 或其他实际模型的转换收益。
- 它不能无损替换任意 Llama 的 softmax attention。固定大小递归状态对长上下文有意义，但训练仍有激活、状态和反向成本。
- 原论文代码有 1.3B/100B-token 实验和 H100 测量。原实现为非商业许可证；[FLA 层模块](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gdn2.py)及[梯度测试](https://github.com/fla-org/flash-linear-attention/blob/main/tests/ops/test_gdn2.py)可用于核对接口与数学，代码复用另核适用许可证。
- **模块边界**：递推算子、层模块、checkpoint 转换器分别实现；先验证退化参数下逐 token 状态、输出和梯度，再比较 chunk/recurrent 实现。

### Attention Residuals：沿网络深度选择信息

- 论文：[Attention Residuals](https://arxiv.org/abs/2603.15031)，2026-03；[作者代码与 PDF](https://github.com/MoonshotAI/Attention-Residuals)。
- 让层通过深度方向的 softmax attention 选择过去的表示，替代固定权重残差累加。这里是跨层的信息选择，不是 token attention。
- Full 版本需要保留很多层输出；Block AttnRes 只保留块级表示，更适合 32GB 的独立模型实验。
- 零初始化 pseudo-query 会得到均匀平均，不等于原残差求和。不能将它描述为现有预训练模型的函数保持替换。论文主要证据是 48B 总参数/3B 激活模型的 1.4T-token 预训练。
- **模块边界**：单独的 residual mixer 与实验模型类，明确 block 输出的生命周期；不能通过给原训练循环追加层输出列表来绕开模型接口。验证 checkpointing 重算与 block 缓冲的关系。

### Engram：将知识存储与动态计算分开

- 论文：[Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372)，2026-01；[作者代码与 PDF](https://github.com/deepseek-ai/Engram)。
- 对 token N-gram 做确定性哈希查表，取得静态记忆，再用上下文门控与隐藏状态融合。条件记忆与 MoE 的条件计算是互补维度。
- 值得先试小型 memory table 与独立残差分支，再评估容量扩展和 CPU 表存储。新增容量需要学习，不能视为现成权重的透明加速开关。
- 论文训练使用跨 GPU 分表和 All-to-All；低开销 CPU offload 例子是 H800 的推理实验。单卡训练仍需预算表参数、梯度和优化器状态。
- 官方演示会 mock Attention/MoE/mHC，默认 `nn.Embedding` 仍产生 dense gradient；演示不等于完整训练实现。
- **模块边界**：token hash、表存储、lookup、门控融合及表优化状态分离。CPU/GPU 存放、稀疏梯度和缓存策略是显式配置，不混入 tokenizer 或原 native trainer。

### Mamba-3：改进状态空间更新

- 论文：[Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569)，2026-03；[官方仓库](https://github.com/state-spaces/mamba)，代码于 2026-03-17 发布。
- 从状态空间离散化、旋转状态更新和 MIMO 结构改进递归序列建模，提高每次状态读写的表达能力。
- 适合独立 recurrent/hybrid 模型与预训练实验；替换现有 Transformer 需要转换、蒸馏或继续训练。固定推理状态不意味着训练没有激活或优化器内存。
- [Mamba3 模块](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)已有实现，但 SISO、MIMO、decode 分别涉及 Triton、TileLang、CuTe 等路径；SM120 需要实际编译与数值验证。
- **模块边界**：模型适配器与算子 backend 分开，保留小型 reference recurrence 作数值对照，单列训练态和推理 cache 契约。

### LLaDA2.2：可编辑的扩散训练与块级专家池

- 官方：[LLaDA2.X](https://github.com/inclusionAI/LLaDA2.0)，README 标注 2026-07 发布 LLaDA2.2；[技术报告 PDF](https://github.com/inclusionAI/LLaDA2.0/blob/main/LLaDA2_2_tech_report.pdf)，题名 *LLaDA2.2: Enabling Agentic Diffusion Language Models via Levenshtein Editing*。
- 在扩散生成中学习 KEEP、SUBSTITUTE、DELETE、INSERT，用草稿与目标的 LCS 对齐构造编辑监督；L-EBPO 将编辑动作纳入环境奖励下的策略优化。
- 不能简单称其实现为任意可变长度 block：报告在 block 内应用增删，再 pad/truncate 恢复固定 block 长度。
- 对 MoE 特别值得借鉴的是 **block expert admission**：先用 block 内每个 expert 的最高 token score 选一个容量为 C 的专家池，再在池内执行 token-wise top-k。它限制一个 block 实际触及的专家数，但不直接减少模型的总参数存储。
- 这属于训练目标、采样器和路由的共同改变。原报告包含大模型、长上下文继续预训练和分布式 agentic RL；单卡应研究小模型中的编辑目标或专家池机制，不能直接外推其 TPS。
- **模块边界**：编辑标签构造、扩散 objective、编辑 sampler、专家准入策略各自独立。与现有 causal CE 的标签对齐和跨样本 attention 契约分别定义。

## 训练：2026 新方法

### Dion3：缩小正交化矩阵，保留误差反馈

- 论文：[Dion3: Full-Stack Orthogonal Updates](https://arxiv.org/abs/2608.11612)，2026-08 预印本；[Microsoft 官方实现](https://github.com/microsoft/dion)，同时提供 Dion2/Dion3。
- Dion2 选择动量矩阵的子矩阵再正交化，并通过 error feedback 管理压缩误差；Dion3 结合逐神经元归一化。
- 支持普通单设备 Tensor，单卡不必初始化 process group；支持将三维及更高维参数视为独立矩阵 batch，对专家张量有研究价值。仍使用反向梯度。
- 它保留完整形状的 momentum，Dion3 还有逐神经元统计。不能当作只存小 sketch 的优化器；stack、子矩阵选择和 Newton-Schulz 的临时内存也需测量。
- embedding、LM head、归一化等需正确分组，不能仅按 `ndim == 2` 分类。默认实现为分布式 checkpoint 会提前建立候选状态，现有 LISA drop/CPU 与 OIB 不能假定直接兼容。
- **模块边界**：参数分组器、正交更新、状态迁移与调度适配器分开。先做普通 optimizer-step 路径，再独立验证 layer sampling、offload 和逐参数 hook。官方实现为 MIT。

### MuonQ：低比特优化器状态

- 论文：[MuonQ: Enhancing Low-Bit Muon Quantization via Directional Fidelity Optimization](https://arxiv.org/abs/2605.11396)，2026-05；[作者代码](https://github.com/YupengSu/MuonQ)，作者标注 CoLM 2026，会议录状态本轮未独立核实。
- 用低比特表示压缩 Muon 的优化器状态，重点维持更新方向；这是状态压缩，不是 NVFP4 GEMM 或完整 FP4 训练。
- 对 32GB 有研究价值，但需同时预算权重、梯度、投影/正交化临时量和其他状态。作者代码明确微调尚未验证；这作为实验范围记录，不作为拒绝试验的理由。
- 接入发现：标准 optimizer state loader 可能将 `uint8` packed state 转成参数 dtype，破坏压缩契约。独立原型应先保证 packed dtype、scale、shape 等状态完整恢复。
- **模块边界**：quantizer、packed-state codec、解码/更新过程和 checkpoint loader 分离；验证恢复后的状态字节数、dtype 及后续更新，避免只测能否保存文件。官方实现为 Apache-2.0。

### AIM-ZO：激活驱动的前向搜索子空间

- [官方代码](https://github.com/EkkoXy/AIM-ZO)的 `CITATION.cff` 标注软件发布于 **2026-09-26**，作者为 Anonymous Authors；本轮没有一手确认论文编号及发表状态，因此只列为 2026 开源研究原型。
- 用前向激活的 Oja 更新维护较宽子空间，再结合高分共享方向与采样尾部方向；在活动子空间中使用 rank-one probes 和 centered population estimator，免反向传播。
- 默认历史 basis 宽度 128，活动宽度 64（共享 48、尾部 16），15 个 probes 加中心评估，约 16 次 forward/update；不能按 MeZO 的两次前向预算。
- **模块边界**：activation collector、subspace maintainer、probe sampler、参数扰动/恢复与估计器分离，训练调度只消费统一的更新结果。恢复需覆盖 basis、方向采样 RNG 和统计状态。

## 实验优先级

| 顺序 | 实验 | 最小可维护原型 | 主要评估量 |
| --- | --- | --- | --- |
| 1 | Dion3 / MuonQ 优化器对照 | 独立参数分组与状态适配；先普通 step，不同时叠加 LISA/OIB | 相同负载下的 optimizer 时间、状态及临时字节、loss/质量对训练时间曲线 |
| 2 | AIM-ZO 子空间估计 | 先小型 FP32 adapter 或小模型，保持参数扰动/恢复为独立组件 | 配对 loss、恢复误差、实际 forward 次数、更新质量对总前向预算 |
| 3 | GDN-2 结构扩展 | 独立层和旧递推退化映射，再接已有 GDN 类模型 | 转换前后状态/输出/梯度、长序列内存、chunk/recurrent 吞吐 |
| 4 | 小型 Engram 分支 | 小表、显式存储位置和独立 gate | 总表状态、lookup/transfer 时间、容量对任务质量的影响 |
| 5 | Block AttnRes / 扩散编辑 | 独立模型类或 objective/sampler，使用小规模训练对照 | block 缓冲生命周期、有效监督、实际训练计算与生成效率 |

这里按可接入性排序，不以算法稳定性筛掉激进方向。架构实验和优化器实验分别做消融，随后再研究组合，保持结果可归因。

## 相关但不改标为 2026 首发

- **mHC**：arXiv `2512.24880` 属于 2025-12；2026 年讨论或引用不改变首预印本年份。
- **Kimi Linear**：`2510.26692`；原 Gated DeltaNet 为 `2412.06464`。
- **FZOO**：作者标注 ICLR 2026，但预印本 `2506.09034` 是 2025；仍可作为前向训练基线。
- **EGGROLL / Evolution Strategies at the Hyperscale**：`2511.16652` 是 2025 预印本；2026 的 MoE/vLLM 更新值得单独研究，不改写论文首发年份。
- **Pretraining Large Language Models with NVFP4**：`2509.25149` 是 2025。2026 软件支持更新与论文年份分别记录，SM100 与 RTX 5090 的 SM120 分别验证。
- **ZO-Muon**：[作者 abstract 与实现](https://github.com/OPTML-Group/ZO-Muon)可核对子空间估计和谱正交化；README 旧注释中的 `2510.00761` 是无关论文。本轮不将未获一手确认的 2026 编号作为确定书目。

Qwen3.5 是直接使用 2026 模型的另一个选择：官方 2026-02-16 发布系列、03-02 发布 0.8/2/4/9B 小模型。它的模型发布时间与 GDN 混合机制的首次提出年份分别记录。
