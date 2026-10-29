# TLC：The Little Coder

本目录是 TLC 实验在 Surogate 原生训练器上的 fork 包 `surogate_tlc`。P0 从零训练三条基线：A0（Mamba-2 + SwiGLU，无 attention）、A1（参数量与 A0 相差在 ±2% 以内的 Transformer）和 A1h（Mamba-2 加 2 层 attention）。后续阶段在 A0 上逐项加入角色监督、双 FFN 和代码记忆，每次只改一个变量；A0 没有通过 G1 或 G2 时停止，不进入后续阶段。

P0 不修改任何上游文件：TLC 模型在 import 时注册进上游 DSL，权重由 fork 的初始化工具写成 safetensors，训练直接调用上游训练器。

## 目录

| 路径 | 内容 |
|---|---|
| [pyproject.toml](pyproject.toml) | 包 `surogate-tlc`（src 布局），不发布。运行时需要本仓库构建的原生 `surogate`（训练扩展 `_surogate`）；它故意不写进依赖，免得解析器从索引装上另一个同名发行包 |
| [src/surogate_tlc/](src/surogate_tlc/) | 包代码，导入它不需要原生扩展。原生训练器只在 `surogate_tlc.native` 中构建 |
| [examples/p0.yaml](examples/p0.yaml) | 唯一的运行配置：各行只差路径（环境变量 `TLC_ROW`），含 smoke、overfit、pilot、throughput 等阶段 |
| [docs/](docs/) | 契约、数据、训练与评测、运行手册和实测记录，见下文“文档” |
| [tests/cpu/](tests/cpu/) | CPU 测试，用 `fork/dev test tlc` 运行 |
| [tests/gpu/](tests/gpu/) | GPU 测试，在 GPU 主机上按 [运行手册](docs/runbook.md) 运行；没有 CUDA 时全部跳过 |
| [prototypes/](prototypes/) | 纯 PyTorch 的 Mamba-2 参照与 G1、G2 参照套件，不打包 |

TLC 代码不进任何其他 wheel：根目录 `pyproject.toml` 的原生 wheel 只打包 `surogate/`，实验 wheel 只打包 `surogate.experimental*`。`tests/cpu` 中的测试检查这两条边界，所以不要把 TLC 代码放进 `surogate/_fork`。

## 环境

- CPU 测试：`fork/dev test tlc` 用 `FORK_PY` 指定的解释器，默认是仓库根目录的 `.venv/bin/python`，那里不需要装本包也不需要原生扩展，准备方法见 [fork/README.md 的“首次准备环境”](../README.md#首次准备环境)。测试把仓库根目录和 `fork/tlc/src` 放在 `sys.path` 最前面，只测检出目录中的代码，不在仓库中写入文件。
- GPU 主机（RTX 5090）：专用环境 `fork/.venv-tlc` 的依赖与根目录 `.venv` 相同，本包以 editable 方式装入；`surogate` 经一个 `.pth` 文件从本检出导入，原生扩展就是 `fork/dev build train` 复制进 `surogate/` 的那一份。创建步骤见 [运行手册的 G0](docs/runbook.md#g0-环境)。
- 原型：在 `fork/.venv-experimental` 或任一装有 CPU torch 的环境中运行，见 [prototypes/README.md](prototypes/README.md)。

## 命令行

在 GPU 主机上用 `fork/.venv-tlc/bin/python -m surogate_tlc 命令 ...` 运行；`--help` 列出参数。命令都作用于模型目录，由 `surogate_tlc.__main__.COMMANDS` 一张表定义：

| 命令 | 作用 |
|---|---|
| `init ROW OUT --tokenizer DIR --seed N` | 写出 P0 一行的模型目录及初始权重。`OUT` 必须不存在或为空 |
| `train CONFIG [--stage S] [--microbatch B]` | 训练一个运行配置。退出码：0 完成；1 训练失败，失败报告在运行记录中；2 配置有问题，没有开始训练 |
| `eval MODEL DATA [--split val] [--max-rows N] [--out FILE]` | teacher-forced 的 loss、按来源和按种类（代码、英文）的 loss 与 unigram 基线 |
| `probe MODEL DATA SUITE [--max-items N] [--out FILE]` | 在绑定探针套件上做候选打分；`--out` 另写每项的结果 |
| `sample MODEL DATA [--max-new 64] [--out FILE]` | 固定 prompt 上的 greedy 样例 |
| `export CHECKPOINT SOURCE OUT` | 从一个 step 目录导出干净的模型目录 |

数据工具是各自模块的入口：`python -m surogate_tlc.data.tokenizer`、`.data.shards`、`.data.synth`、`.data.decontam`、`.data.natural`，报告是 `python -m surogate_tlc.train.report`。上游 `surogate` 命令行的透传和 rollout 服务属于 P5，P0 不提供。

## P0 的行

`surogate_tlc.hf.ROWS` 定义各行相对 `default_config()` 的差异，`row_config(row)` 给出完整的 config.json。四行都是同一个 DSL 模型 `TLCModel`（架构名 `TLCForCausalLM`），词表 16384，宽度 384，12 层，上下文 512，tied embedding。参数量由 IR 的参数形状算出（`surogate_tlc.ir.param_count`），不含冻结的 RoPE 表；tied 时不计 LM head，非 embedding 参数不计 embedding 与 LM head。前向 FLOPs/token 是 `surogate_tlc.ir.flops_per_token` 的估计：matmul 乘加数的两倍加序列混合项，训练约为它的 3 倍。

| 行 | 与 A0 的差异 | 参数量（tied / untied） | 非 embedding | 相对 A0 | 前向 MFLOP/token（T=512 / 2048） |
|---|---|---|---|---|---|
| A0 | 12 × `M` | 31,782,192 / 38,073,648 | 25,490,736 | — | 68.42 / 68.42 |
| A0-split | `tlc_split_first_layer: true`：第 1 层为模型级 ops | 与 A0 相同 | 与 A0 相同 | 0 | 与 A0 相同 |
| A1 | 12 × `A`，`intermediate_size` 1504 | 31,812,480 / 38,103,936 | 25,521,024 | +0.10% / +0.08% | 68.32 / 82.48 |
| A1h | `MMMMMAMMMMMA`：第 6、12 层为 `A` | 30,681,320 / 36,972,776 | 24,389,864 | −3.46% / −2.89% | 66.19 / 68.55 |

- Mamba-2 几何固定为 `surogate_tlc.hf.MAMBA_GEOMETRY`：H=12 个头、每头 P=64（内宽 768）、状态 N=64、G=1 组，conv 核 4，chunk 128。N 取 64 而不取 128，因为 scan kernel 在状态维上的循环是串行的。
- A1 的 FFN 宽度 1504 使它与 A0 的参数量相差在 0.1% 以内；attention 几何与 A1h 相同（6 个头、2 个 KV 头、head_dim 64）。A1h 只作上界参照，不做参数匹配。
- A0-split 为 P2/P3 去风险（角色特征和槽写入要在模型级从第 1 层的输出计算），G1(d) 检查它与 A0 等价。各行的细节见 [docs/model.md](docs/model.md#dsl-模型与-ir-检查)。
- 参数匹配的混合行 A2-pm（attention 层的 SwiGLU 宽约 1502，与 A0 相差约 0.001%）推迟到 Phase 1：它要按层设置 FFN 宽度，而编译出的 IR 中所有层共用一组按名字合并的激活槽（例如 `mlp_up` 是 [B, T, 2048]），更宽的 attention 层 MLP 会写出这个缓冲区；要给 attention 层的 MLP 激活另起槽名，并在 GPU 上确认原生侧接受。P0 用 A1h 代替。

## 文档

- [docs/model.md](docs/model.md)：注册，config.json 的不变量、模型目录与 checkpoint 契约，DSL 模型与 IR 检查，初始化，数值参照与 G1、G2 参照套件，上游接口的 pin。
- [docs/data.md](docs/data.md)：语料 spec、去污染、合成绑定任务、tokenizer 与数据 shard。
- [docs/training.md](docs/training.md)：训练驱动（运行配置、恢复、失败报告、运行记录与收尾）与评测（loss、绑定探针、生成样例）。
- [docs/runbook.md](docs/runbook.md)：RTX 5090 上的 G0–G3 运行手册与 GPU 预算。
- [docs/measurements.md](docs/measurements.md)：注明主机、提交和日期的实测记录。

## P0 之后

实验笔记为后续步骤计划、P0 有意不做的部分：

- 矩阵 config（`matrix.py`：各行的 config 继承与 diff、声明变量的校验）、结果 schema（`results/schema.py`：`result/1` 与 `comparison/1` 两类记录及校验器，pilot 行单独标记）和统计工具（`eval/stats.py`）。P0 的运行记录 `tlc_run.json`、`eval`/`probe` 的输出已带上述记录需要的哈希与 `pad_fraction`，等 schema 落地后转成结果行。
- 按角色分层的 loss（要 P1 的角色标签）、自然代码的标识符 NLL 与 α-重命名不变性。
- 生成与 rollout 服务、执行沙箱、上游命令行透传（P5）；原生 SSM 解码（P5-d），所以 `sample` 每生成一个 token 就重算整行。
- 参数匹配的 A2-pm（见上文）、正式训练的 2048 上下文（owner 待定；改 `surogate_tlc.hf.CONTEXT_LENGTH` 并用 `--seq-len 2048` 重建 shard）。
- 运行时长与磁盘的上限：每个运行只由 `max_steps` 限定长度。
