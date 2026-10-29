# TLC：The Little Coder

本目录是 TLC 实验在 Surogate 原生训练器上的 fork 包 `surogate_tlc`。P0 从零训练三条基线：A0（Mamba-2 + SwiGLU，无 attention）、A1（参数量与 A0 相差在 ±2% 以内的 Transformer）和 A1h（Mamba-2 加 1–2 层 attention）。后续阶段在 A0 上逐项加入角色监督、双 FFN 和代码记忆，每次只改一个变量；A0 没有通过 G1 或 G2 时停止，不进入后续阶段。

P0 不修改任何上游文件：TLC 模型在 import 时注册进上游 DSL，权重由 fork 的初始化工具写成 safetensors，训练直接调用上游训练器。

## 目录

| 路径 | 内容 |
|---|---|
| [pyproject.toml](pyproject.toml) | 包 `surogate-tlc`（src 布局），不发布。运行时需要本仓库构建的原生 `surogate`（训练扩展 `_surogate`）；它故意不写进依赖，免得解析器从索引装上另一个同名发行包 |
| [src/surogate_tlc/](src/surogate_tlc/) | 包代码。导入它不需要原生扩展 |
| [tests/cpu/](tests/cpu/) | CPU 测试，用 `fork/dev test tlc` 运行 |
| [tests/gpu/](tests/gpu/) | GPU 测试，在 GPU 主机上按下文“运行手册”运行；没有 CUDA 时全部跳过 |
| [prototypes/](prototypes/) | 纯 PyTorch 原型与数值参照，不打包 |

TLC 代码不进任何其他 wheel：根目录 `pyproject.toml` 的原生 wheel 只打包 `surogate/`，实验 wheel 只打包 `surogate.experimental*`。`tests/cpu` 中的测试检查这两条边界，所以不要把 TLC 代码放进 `surogate/_fork`。

## 环境

- CPU 测试：`fork/dev test tlc` 用 `FORK_PY` 指定的解释器（默认 `.venv/bin/python`），需要 CPU torch 和 `fork/tools/requirements-cpu-lane.txt`，准备方法见 [fork/README.md 的“首次准备环境”](../README.md#首次准备环境)。测试把仓库根目录和 `fork/tlc/src` 放在 `sys.path` 最前面，只测检出目录中的代码，不在仓库中写入文件。
- GPU 主机（RTX 5090）：专用环境 `fork/.venv-tlc` 装有原生构建，本包以 editable 方式装入。创建步骤写在下文 G0 中。
- 原型：在 `fork/.venv-experimental` 或任一装有 CPU torch 的环境中运行，见 [prototypes/README.md](prototypes/README.md)。

## 模型目录与 checkpoint 契约

### config.json 的不变量

由 CPU 测试断言：

- `architectures=["TLCForCausalLM"]`，`model_type="tlc"`。`model_type` 不能是 `qwen*`（模型会被交给服务引擎执行），也不能含 `vl` 或 `vision`（会打开视觉输入）。DSL 类名一律带 `TLC` 前缀：注册表按类名做键，同名类会悄悄覆盖上游的定义。
- 通用键齐全：`hidden_size`、`num_hidden_layers`、`vocab_size`、`max_position_embeddings=512`、`bos_token_id`、`eos_token_id`、`pad_token_id`、`tie_word_embeddings`。`vocab_size` 至少是最大 token id 加 1，特殊 token 也在词表之内（原生侧不检查 id 范围）。
- 没有 attention 的 A0 也给出 `num_attention_heads>0` 或 `head_dim>0`：原生侧在 `head_dim` 为 0 时用 attention 头数做除数。
- `hybrid_pattern` 是字符串（A0 为 L 个 `M`）。list 类型的属性在 DSL 编译时被丢弃。
- Mamba 维度（头数 H、头维 P、状态维 N、组数 G）显式写入，不依赖默认值。P0 暂按 H=12、P=64、N=64、G=1；N 取 64 而不取 128，因为状态维上的循环是串行的。
- 含 GDN 的模型显式给出全部 `linear_*` 键：缺少时 kernel 会退回按 attention 的几何编译，而且不报错。

### 目录中的文件

可用作 `model_dir` 的目录包含：

- `config.json`，带 `auto_map`，指向同目录的 `configuration_tlc.py`，因此不 import fork 的进程也能用 transformers 读取它；
- `configuration_tlc.py`；
- tokenizer 文件：`tokenizer.json` 与带 `chat_template` 的 `tokenizer_config.json`；
- `model.safetensors`：从零训练时由 fork 的初始化工具按 IR 的参数名写出全部权重。Mamba-2 的 `A_log`、`dt_bias`、`D` 用标准初始化（原生侧按参数名初始化的结果对 Mamba 不对：`dt_bias` 为 0，`A_log` 与 `D` 服从正态分布）；tied 时不写 `lm_head`。训练经 `from_pretrained(model_dir)` 加载它；
- `tlc_meta.json`：标签本体版本、shard 与 sidecar 版本、初始化种子。

### checkpoint 与导出

- 原生 checkpoint 只写通用 config 键，没有 `auto_map`、TLC 自有的键和 tokenizer；导出只复制一份固定的文件清单。所以每次保存和导出之后，由收尾 hook（计划为 `surogate_tlc.checkpoint.finalize_dir()`）把完整的源 config.json、上面的附带文件和 fork 的 sidecar 状态（sampler 状态、标签本体版本、shard 版本）补进目标目录。
- 未经收尾的原始 checkpoint 目录不得直接用作 `model_dir`。
- 收尾后的目录能在不 import fork 的新解释器中加载（CPU 测试）；save → resume 后 logits 和下一步 loss 能否复现，在 G2 中验证。

## 数据打包

- A0、A1、A1h 一律每行一个文档：长文件切成不超过 512 token 的窗口，末尾 padding，不跨文档打包。原因是 `mamba_ssm_scan` 在打包的文档之间不复位状态；A1 也这样做，保证各行的数据语义相同。
- 记录 padding 浪费（非 pad token 按 ValidTokenCount 计）。超过约 25% 时，再考虑实现文档复位后改用打包 shard，所有行同时切换。

## OOM 与每次更新的 token 数

- 每次更新固定 16,384 个名义 token（microbatch B × 梯度累积 GA × 512），另行记录非 pad 的有效 token 数。
- 不用 `train_with_oom_recovery`（`surogate/train/sft.py`）：GA ≥ 16 时它只把 B 减半、GA 不变，每次更新的 token 数悄悄减半，只留一条 warning，最多重试 10 次。驱动直接调用 `SurogateTrainerWrapper(config, train_files, eval_files).train()`（`surogate/train/trainer.py`）。
- 遇到 OOM，按配置二选一：
  - 默认立即失败，写 failure_report（峰值显存和实际配置）；
  - 显式开启降档时，B 减半、GA 加倍（B × GA 不变），写入 provenance；降档次数有上限，到达上限后失败。
- 降档只改 B 和 GA，不改架构、序列长度或每次更新的名义 token 数。

## GPU 预算

G3 测出 tokens/s 之后填写。

GPU 小时 = Σ 种子数 × 名义 token 数 ÷ 实测 tokens/s × (1 + 评测与保存开销) ÷ 3600。

- 关键比较至少 3 个种子，pilot 可以只用 1 个。
- G3 在每次更新 16,384 token 不变的前提下测 microbatch 2、8、32（GA 相应为 16、4、1）；每行注明所用档位。

| 行 | 阶段 | 种子数 | 名义 token（每个种子） | B × GA | 实测 tokens/s | 评测与保存开销 | GPU 小时 |
|---|---|---|---|---|---|---|---|
| A0 | 300 步 pilot | 1 | 4,915,200 | 待 G3 | 待 G3 | 待 G3 | 待 G3 |
| A1 | 300 步 pilot | 1 | 4,915,200 | 待 G3 | 待 G3 | 待 G3 | 待 G3 |
| A1h | 300 步 pilot | 1 | 4,915,200 | 待 G3 | 待 G3 | 待 G3 | 待 G3 |
| A0、A1、A1h | 正式训练 | 3 | 待定 | 待 G3 | 待 G3 | 待 G3 | 待 G3 |

## 运行手册（RTX 5090，G0–G3）

本节由后续步骤填写：每个门槛给出准确的命令、预期输出和未通过时的动作。门槛未通过就停止，不进入下一个门槛。GPU 测试单独运行，不与 `tests/cpu` 在同一次 pytest 中收集：CPU 测试会把仓库根目录放到 `sys.path` 最前面，而 GPU 测试要用环境中装好的原生扩展。

### G0 环境

待填写。范围：创建 `fork/.venv-tlc`，`fork/dev build train`（sm_120a），环境指纹，BF16 GEMM，selective scan 的前向与反向，A1 的 flash-attention（head_dim 64），至少 10k 文档上原生与 HF tokenizer 的一致性。

### G1 因果性

待填写。范围：修改位置 t 之后的 token 不改变位置 ≤ t 的 logprobs（前缀长度 1、2、7、64、511）；原生全序列 logprobs 与参照实现的逐 token 递归一致；A1 打包与分开运行一致；A0-split 与 A0 等价。

### G2 数学与梯度

待填写。范围：step-0 loss ≈ ln 16384 ≈ 9.70；梯度全部有限；单层梯度与参照一致；固定 batch 100 步过拟合；save → resume 后 logits 和下一步 loss 复现。

### G3 基线

待填写。范围：每行依次 10 步冒烟、100 步过拟合、300 步 pilot；记录 train/val loss、microbatch 2/8/32 的 tokens/s、峰值显存、ValidTokenCount 和样例，然后填写上文的 GPU 预算。
