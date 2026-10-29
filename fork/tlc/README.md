# TLC：The Little Coder

本目录是 TLC 实验在 Surogate 原生训练器上的 fork 包 `surogate_tlc`。P0 从零训练三条基线：A0（Mamba-2 + SwiGLU，无 attention）、A1（参数量与 A0 相差在 ±2% 以内的 Transformer）和 A1h（Mamba-2 加 1–2 层 attention）。后续阶段在 A0 上逐项加入角色监督、双 FFN 和代码记忆，每次只改一个变量；A0 没有通过 G1 或 G2 时停止，不进入后续阶段。

P0 不修改任何上游文件：TLC 模型在 import 时注册进上游 DSL，权重由 fork 的初始化工具写成 safetensors，训练直接调用上游训练器。

## 目录

| 路径 | 内容 |
|---|---|
| [pyproject.toml](pyproject.toml) | 包 `surogate-tlc`（src 布局），不发布。运行时需要本仓库构建的原生 `surogate`（训练扩展 `_surogate`）；它故意不写进依赖，免得解析器从索引装上另一个同名发行包 |
| [src/surogate_tlc/](src/surogate_tlc/) | 包代码。导入它不需要原生扩展 |
| [examples/](examples/) | 运行配置：A0、A1、A1h 各一个，含 smoke、overfit、pilot 三个阶段 |
| [tests/cpu/](tests/cpu/) | CPU 测试，用 `fork/dev test tlc` 运行 |
| [tests/gpu/](tests/gpu/) | GPU 测试，在 GPU 主机上按下文“运行手册”运行；没有 CUDA 时全部跳过 |
| [prototypes/](prototypes/) | 纯 PyTorch 原型与数值参照，不打包 |

TLC 代码不进任何其他 wheel：根目录 `pyproject.toml` 的原生 wheel 只打包 `surogate/`，实验 wheel 只打包 `surogate.experimental*`。`tests/cpu` 中的测试检查这两条边界，所以不要把 TLC 代码放进 `surogate/_fork`。

## 环境

- CPU 测试：`fork/dev test tlc` 用 `FORK_PY` 指定的解释器（默认 `.venv/bin/python`），需要 CPU torch 和 `fork/tools/requirements-cpu-lane.txt`，准备方法见 [fork/README.md 的“首次准备环境”](../README.md#首次准备环境)。测试把仓库根目录和 `fork/tlc/src` 放在 `sys.path` 最前面，只测检出目录中的代码，不在仓库中写入文件。
- GPU 主机（RTX 5090）：专用环境 `fork/.venv-tlc` 的依赖与根目录 `.venv` 相同，本包以 editable 方式装入；`surogate` 经一个 `.pth` 文件从本检出导入，原生扩展就是 `fork/dev build train` 复制进 `surogate/` 的那一份。创建步骤见下文 G0。
- 原型：在 `fork/.venv-experimental` 或任一装有 CPU torch 的环境中运行，见 [prototypes/README.md](prototypes/README.md)。

## 注册

训练器按 config.json 的 `architectures` 在上游 DSL 注册表中查找模型，并用 transformers 的 `AutoConfig` 读取 config。`surogate_tlc.register()` 做两件事，可重复调用：

- import `surogate_tlc.dsl`，其中 `@nn.hf_config` 把 `TLCModel`（架构名 `TLCForCausalLM`）加进 DSL 注册表。构建 TLC 的 IR 总是需要这一步。
- 把 `TLCConfig` 注册为 `model_type="tlc"` 的 `AutoConfig`。注册后它优先于目录中的 `auto_map`；不注册时，带 `auto_map` 的目录照样能被读取。

DSL 类名一律带 `TLC` 前缀：注册表按类名做键，同名类会悄悄覆盖上游的定义。CPU 测试检查 `register()` 只新增 `TLCModel`，不替换任何上游条目。

## 命令行

在 GPU 主机上用 `fork/.venv-tlc/bin/python -m surogate_tlc 子命令 ...` 运行（`surogate_tlc.__main__`），`--help` 列出各子命令的参数。

| 子命令 | 作用 |
|---|---|
| `init ROW OUT --tokenizer DIR --seed N` | 写出 P0 一行（`A0`、`A0-split`、`A1`、`A1h`）的模型目录及初始权重，即 `write_model_dir()` 加 `init_model_dir()`。`OUT` 必须不存在或为空 |
| `train-tokenizer ...` | 即 `python -m surogate_tlc.data.tokenizer train ...`，参数相同 |
| `build-data ...` | 即 `python -m surogate_tlc.data.shards ...`，参数相同 |
| `train CONFIG [--stage S] [--microbatch B]` | 训练一个运行配置，见下文“训练驱动”。退出码：0 完成；1 训练失败，失败报告在运行记录中；2 配置有问题，没有开始训练 |
| `eval MODEL DATA [--split val] [--batch 32] [--max-rows N] [--out FILE]` | teacher-forced 的 loss 与 unigram 基线，见下文“评测” |
| `export CHECKPOINT SOURCE OUT` | 从一个 step 目录导出干净的模型目录（`surogate_tlc.checkpoint.export`） |
| 其他命令 | 先 `register()`，再交给上游的 `surogate` 命令行，例如 `python -m surogate_tlc sft CONFIG`。上游在同一进程中运行子命令模块，所以它看得到 TLC 模型。上游的 `sft` 会自己 tokenize，并经过 `train_with_oom_recovery`，所以 TLC 的训练用 `train` |

## P0 的行

`surogate_tlc.hf.ROWS` 定义各行相对 `default_config()` 的差异，`row_config(row)` 给出完整的 config.json。四行都是同一个 DSL 模型 `TLCModel`（`TLCForCausalLM`），词表 16384，宽度 384，12 层，上下文 512，tied embedding。参数量由 IR 的参数形状算出（`surogate_tlc.ir.param_count`），不含冻结的 RoPE 表；tied 时不计 LM head。

| 行 | 与 A0 的差异 | 层 | 参数量（tied / untied） | 相对 A0 |
|---|---|---|---|---|
| A0 | — | 12 × `M` | 31,782,192 / 38,073,648 | — |
| A0-split | `tlc_split_first_layer: true` | 第 1 层为模型级 ops，其后 11 × `M` 在 stack 中 | 与 A0 相同 | 0 |
| A1 | `tlc_layer_pattern` 为 12 个 `A`，`intermediate_size` 1504 | 12 × `A` | 31,812,480 / 38,103,936 | +0.10% / +0.08% |
| A1h | `tlc_layer_pattern` 为 `MMMMMAMMMMMA` | 10 × `M`，第 6、12 层为 `A` | 30,681,320 / 36,972,776 | −3.46% / −2.89% |

- **Mamba-2 几何**（`surogate_tlc.hf.MAMBA_GEOMETRY`）：H=12 个头、每头 P=64（内宽 768，即 2 × 宽度）、状态 N=64、G=1 组，conv 核 4，chunk 128。支持方案 §4.3 与 v0.7 notebook（§0.8、§13）都固定为这组几何：N 取 64 而不取 128，因为 scan kernel 在状态维上的循环是串行的。A0 的参数量由这组几何唯一确定，CPU 测试钉住它。Mamba-2 的 conv 带 bias、输入输出投影不带 bias；dt 不做 clamp（上游 `Mamba2Mixer` 不给 `time_step_limit` 时的行为），数值参照必须一致。
- **A1**：FFN 宽度 1504 使 A1 与 A0 的参数量相差在 0.1% 以内（目标 ±2%）；attention 几何与 A1h 相同（6 个头、2 个 KV 头、head_dim 64）。A1 的 IR 与同尺寸的上游 `Qwen3ForCausalLM` 逐 op、逐参数相同（CPU 测试），只是 checkpoint 中的权重名不同。A1 不设 `hybrid_pattern`，所以原生侧把每层当作 dense 块，与上游 Qwen3 的路径一致；它的 config.json 不带 Mamba 键。
- **A1h**：只作上界参照，不做参数匹配（attention 层比 Mamba 层少约 55 万参数）。原生侧按 `hybrid_pattern` 把 `A` 层当作 attention 块，而 `Qwen3Block` 带 MLP：这在 GPU 上验证（G1）。
- **A0-split**：为 P2/P3 去风险（角色特征和槽写入要在模型级从第 1 层的输出计算）。第 1 层的 ops 由同一个 `TLCMambaFFNBlock` 在模型级追踪，名字加 `first_layer_` 前缀（例如 `first_layer_in_proj_weight`），其后的 stack 是第 2..L 层，层号从 `blocks[0]` 重新开始。CPU 测试检查 A0-split 与 A0 的 op、属性和顺序完全相同，张量一一对应，A0 的参数按 `surogate_tlc.dsl.models.split_param_name()` 改名后形状不变；所以同一份权重改名后即可做 G1(d) 的等价测试。IR 的 `n_layers` 为 11、`hybrid_pattern` 为 11 个 `M`（原生侧的层数只算 stack）。模型级第 1 层的权重与 stack 中的权重一样按 IR 名加载；它的激活槽沿用各 module 的默认作用域，与上游模型级 module 的做法相同。原生侧能否正确执行、重算这些模型级 ops，由 G1(d) 判定；若不等价，改走 fork IR 后处理 pass。

## 模型目录与 checkpoint 契约

相关模块：`surogate_tlc.hf`（config 的 schema 与检查）、`surogate_tlc.modeldir`（写出和检查模型目录）、`surogate_tlc.checkpoint`（收尾与导出）。

### config.json 的不变量

`surogate_tlc.hf.check_config()` 逐条检查并列出全部问题；fork 的入口在构建或加载模型之前调用它。原生训练器自己读 config.json，下面几种错误它都不报错。所有 P0 行（含 A1 这类非 TLC 架构）都要满足：

- 通用键齐全：`vocab_size`、`hidden_size`、`num_hidden_layers`、`intermediate_size` 为正整数，`max_position_embeddings=512`，`tie_word_embeddings` 显式给出，`bos_token_id`、`eos_token_id`、`pad_token_id` 都在词表之内（原生侧不检查 id 范围）。
- `num_attention_heads>0` 或 `head_dim>0`，没有 attention 的 A0 也一样：原生侧在 `head_dim` 为 0 时用 attention 头数做除数。
- 出现任一 GDN 几何键（`linear_num_key_heads`、`linear_num_value_heads`、`linear_key_head_dim`、`linear_value_head_dim`）时四个都要给出：缺少时 kernel 退回按 attention 的几何编译，而且不报错。

TLC 行另外要满足：

- `architectures=["TLCForCausalLM"]`，`model_type="tlc"`，`auto_map={"AutoConfig": "configuration_tlc.TLCConfig"}`。`model_type` 不能是 `qwen*`（模型会被交给服务引擎执行），也不能含 `vl` 或 `vision`（会打开视觉输入）。
- `tlc_layer_pattern` 是字符串，长度等于 `num_hidden_layers`，每层一个字符：`M` 为 Mamba-2 + SwiGLU，`A` 为 attention 层（复用上游 `Qwen3Block`）。各行的取值见下文“P0 的行”。含 `M` 时 DSL 把它（A0-split 为去掉第 1 层后的部分）写进 IR 的字符串属性 `hybrid_pattern`；list 类型的属性在 DSL 编译时会被丢弃。
- `tlc_split_first_layer` 可省略（即 false）；给出时必须是布尔值。为 true 时第 1 层必须是 `M`，且至少还有一层。
- 含 `M` 时 Mamba 的几何显式写入，不依赖默认值：`mamba_num_heads`（H）、`mamba_head_dim`（P）、`ssm_state_size`（N）、`n_groups`（G，整除 H）、`conv_kernel`、`chunk_size`。P0 固定为 `surogate_tlc.hf.MAMBA_GEOMETRY`，见下文“P0 的行”。
- 含 `A` 时 `num_attention_heads`、`num_key_value_heads`、`head_dim` 为正，且头数是 KV 头数的整数倍。

`surogate_tlc.hf.default_config()` 给出 TLC v0.6 的 A0 候选（12 × 384，SwiGLU 1024，词表 16384），`surogate_tlc.hf.row_config(row)` 给出各行的 config，两者都可按键覆盖。

### 目录中的文件

可用作 `model_dir` 的目录包含：

- `config.json`，满足上面的不变量；
- `configuration_tlc.py`：`auto_map` 指向它，内容就是包里的 `surogate_tlc.configuration_tlc`，只依赖 transformers，因此不 import fork 的进程也能读取 config；
- tokenizer 文件：`tokenizer.json` 与带 `chat_template` 的 `tokenizer_config.json`（也可以是单独的 `chat_template.jinja`）。`bos_token`、`eos_token`、`pad_token` 在 tokenizer 中的 id 必须等于 config.json 中的值，最大 token id 小于 `vocab_size`；
- `tlc_meta.json`：`tlc_meta_version`（当前为 1）、`init_seed`、`label_ontology_version`、`shard_version`、`sidecar_version`，后几项在用到之前为 null；checkpoint 与导出目录另有 `step`；
- `model.safetensors`：从零训练时由 `surogate_tlc.init.init_model_dir()` 写出，见下文“初始化”。训练经 `from_pretrained(model_dir)` 加载它。

`surogate_tlc.modeldir.write_model_dir()` 写出除权重以外的全部文件；`check_model_dir()` 检查一个目录并列出全部问题，`weights=False` 时不要求 `model.safetensors`。

### checkpoint 与导出

原生训练器把 checkpoint 存到 `<checkpoint_dir>/step_<8 位步数>/`，最后写 `checkpoint.json`。其中的 config.json 只有通用键，没有 `auto_map`、TLC 自有的键和 tokenizer；训练结束时的导出只从源目录复制一份固定的文件清单，不含 `configuration_tlc.py` 和 `tlc_meta.json`。所以每次保存和导出之后都要收尾，`surogate_tlc.checkpoint` 提供：

- `finalize_dir(target, source, meta=None)`：把源模型目录的 config.json、`configuration_tlc.py`、tokenizer 文件和 `tlc_meta.json`（并入 `meta`）复制进 `target`，原生写的 config.json 被源目录的完整版本替换。`target` 中原生写的结构键（架构、模型类型、词表、宽度、层数、FFN 宽度）与源目录不一致时拒绝执行，以免配错源目录；A0-split 的层数允许少 1，因为原生 config 的层数取自 DSL，而 DSL 的层数不含模型级的第 1 层。可重复执行；`tlc_meta.json` 最后写入，中途失败的目录通不过 `check_model_dir()`。
- `finalize_checkpoints(checkpoint_dir, source, meta=None)`：对每个已写完 `checkpoint.json` 的 step 目录执行 `finalize_dir`，`tlc_meta.json` 并入 `meta` 并记下 `step`；还在写的目录不动。
- `export(checkpoint, source, out)`：从一个 step 目录导出干净的模型目录，只带 `model.safetensors` 和收尾文件，不带优化器状态和分片；step 目录已收尾时，保留其 `tlc_meta.json` 中的字段（训练驱动写入的数据与 tokenizer 的 sha256）。`out` 必须不存在或为空。

规则：

- 未经收尾的原始 checkpoint 目录不得直接用作 `model_dir`。
- 收尾后的目录能在不 import fork 的新解释器中用 transformers 读取 config 与 tokenizer（CPU 测试）；save → resume 后 logits 和下一步 loss 能否复现，在 G2 中验证。
- CPU 测试钉住原生的 step 目录命名、`checkpoint.json` 在全部分片之后写入，以及上游导出复制的文件清单；上游改动这些时测试失败。

## DSL 模型与 IR 检查

- `surogate_tlc.dsl.blocks`：`TLCMambaFFNBlock`，即 RMSNorm → 上游 `Mamba2Mixer` → RMSNorm → 上游 `GenericMLP`（SwiGLU，up 与 gate 融合为一个权重）。它没有 `BlockSchema`：原生侧的 buffer plan 把 schema 当作可选信息（只有设置 `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1` 时才要求每层都有），能否接受由 G0 的附带检查确认。
- `surogate_tlc.dsl.models`：`TLCModel`。整个模型只有一个 `HybridBlockStack`；Mamba 的可选参数由模型级别名 `use_conv_bias`、`use_bias` 打开；embedding、final_norm、lm_head 的权重名显式给出，其余权重按 IR 名加载。
- `surogate_tlc.ir`：
  - `compile_ir(cfg)`：先 `register()`，再按 config 构建训练器会用的 IR。
  - `param_shapes()`、`param_count(module, tied=...)`：符号维度按原生参数存储的定义求值（`C`、`M`、`MUp`、`AttnDim`、`QKV`）。
  - `native_op_names(repo)`：从 `csrc/src/runtime/dsl/graph_compiler.cpp` 的 `op_type_from_string` 表读出原生接受的 op 名。
  - `ir_problems(module, native_ops)`：列出前向图的问题，空列表表示通过。检查项：op 名在原生表中；每个张量只写一次，不覆盖参数或图输入；每个输入都是参数、图输入或之前的 op 写出的张量；图输出都被写出；`moe_softmax` 的输入是 2 维（原生 kernel 直接把前两维当作 token 数和专家数）；`concat`、`mul` 的输入在有 dtype 标注时 dtype 相同（原生侧没有 cast op）。
- 这些检查对应的 DSL 陷阱都有回归测试：两个 layer stack 都把层命名为 `blocks[0..]`，后一个覆盖前一个；同一个 module 调用两次时第二次覆盖第一次的激活（D1 的朴素共享权重循环就是这样）；模型缺少 `use_conv_bias` 别名时 `conv_bias` 从参数中消失，而图仍读取它。
- 所有行的 IR 都通过上述检查（CPU 测试）。后续阶段新增的图结构（`tlc_aux_ce` 是 loss 的祖先、GDN 几何与 `linear_*` 键一致）在对应阶段加入 `ir_problems`。

## 初始化

`surogate_tlc.init.init_model_dir(path)` 往 `write_model_dir(..., init_seed=...)` 写出的目录中写入 `model.safetensors`（命令 `init` 一次做完这两步），种子取自 `tlc_meta.json` 的 `init_seed`；已有权重文件时拒绝执行，`init_seed` 为 null 时报错。`init_state_dict(cfg, seed=...)` 只生成张量，不写文件。

不用原生侧的初始化：它只按参数名决定初值（bias 为 0、norm 为 1、其余服从正态分布），Mamba-2 因此从 `dt_bias=0`（每个头 dt = softplus(0) = 0.69）、`A_log≈0`（每个头 A = −1）、`D≈0` 开始，所有头衰减相同，跳连关闭。fork 改用 Mamba-2 参照实现的初始化，四行用同一套规则，A0、A1、A1h 的起点可比：

| 参数（层内字段名） | 初值 | 来源 |
|---|---|---|
| `dt_bias` | softplus⁻¹(dt)，dt = exp(U(ln 10⁻³, ln 10⁻¹))，下限 10⁻⁴ | `Mamba2.__init__` |
| `A_log` | ln U(1, 16) | `Mamba2.__init__` |
| `D_param` | 1 | `Mamba2.__init__` |
| `conv_weight`、`conv_bias` | U(±1/√K)，K 为 `conv_kernel` | PyTorch `Conv1d` 默认值（depthwise，fan_in = K） |
| `in_proj_weight`、`qkv_weight`、`mlp_up_weight`，untied 的 LM head | U(±1/√fan_in) | PyTorch `Linear` 默认值（`kaiming_uniform_`，a = √5） |
| `out_proj_weight`、`out_weight`、`mlp_down_weight` | U(±1/√fan_in) ÷ √(2L)，L 为 `num_hidden_layers` | `_init_weights`，`n_residuals_per_layer=2` |
| embedding | N(0, 0.02) | `_init_weights` |
| 各 norm 权重 | 1 | — |

- 来源：state-spaces/mamba（main，提交 e9594ce1）中 mamba_ssm 的 `modules/mamba2.py`（`Mamba2.__init__`）和 `models/mixer_seq_simple.py`（`_init_weights`）。attention 层按 mamba_ssm 混合模型中 MHA 的做法处理：qkv 是普通 Linear，输出投影与 Mamba 的 `out_proj` 一样缩放。
- 残差缩放取 √(2L)：每层有一次 mixer 写入和一次 MLP 写入，`MixerModel` 在层带 MLP 时传 `n_residuals_per_layer=2`。`MambaLMHeadModel` 随后用默认值 1 再执行一次 `_init_weights`，实际效果是 √L；TLC 按支持方案（M4）取 √(2L)，常量为 `surogate_tlc.init.RESIDUAL_WRITES_PER_LAYER`。
- 新增的参数字段必须在 `surogate_tlc.init.RULES` 中有规则，否则报错。
- 张量名是 IR 的 `hf_mapping` 名，没有映射时就是 IR 参数名，与原生 loader 查找的名字相同；冻结的 RoPE 表由运行时计算，不写入；tied 时不写 `lm_head.weight`，原生 loader 退回读取 embedding。dtype 与 IR 参数一致：`A_log`、`D_param`、`dt_bias` 为 fp32，其余为 bf16。safetensors 元数据记录 `tlc_init_version`（方案改动时加一）和 `tlc_init_seed`。
- 每个张量用自己的随机数发生器，种子由 (`init_seed`, 层号, 字段名) 派生。同一 seed 下 A0-split 得到的就是 A0 的权重，只是名字按 `split_param_name()` 改过，所以 G1(d) 直接比较两个用同一 seed 初始化的目录。
- CPU 测试：张量名由单层模型的层内参数名和模型级映射重新推导，不复用初始化代码的命名逻辑；形状、dtype、各类参数的分布统计；同一 seed 可复现、不同 seed 不同；A0-split 与 A0 一致。初始化后的 A0 在下文的数值参照上，随机 token 的初始 loss 与 ln 16384 ≈ 9.70 相差不到 0.25（tied 的 N(0, 0.02) embedding 使期望值高出约 0.08），这是 G2 中 step-0 loss 的 CPU 预测。

## 数值参照（G1(b)）

[prototypes/ref_mamba2.py](prototypes/ref_mamba2.py) 是纯 PyTorch 的 Mamba-2 参照实现：读取模型目录的 config.json 与 `model.safetensors`，逐 op 复现原生图对 A0、A0-split 的计算（只支持 `M` 层，A1、A1h 不在范围内）。与原生对齐的细节：

- in_proj 的输出顺序为 [z | x B C | dt]，conv 带 bias 后接 SiLU；
- dt = clamp(softplus(dt_raw + dt_bias), time_step_limit)，softplus 在输入大于 20 时取线性；TLC 不设 `time_step_limit`，原生的 clamp 区间为 (0, 10⁹)；
- gated norm 是 RMSNorm(y · silu(z))（先乘门再归一化），按 `n_groups` 分组求 RMS；
- 残差与 RMSNorm 融合：norm 用 fp32 的和计算，存下的残差舍入到激活 dtype；
- 求值顺序：goldens 用逐 token 递归，与原生 selective scan 的顺序一致；分块 SSD 与逐 token 解码（带 conv 和 SSM cache）作交叉检查，三者在 fp32 下一致（CPU 测试，误差小于 10⁻⁴）。
- bf16 模式在原生以 bf16 存储张量的每个位置舍入，用来估计 bf16 与 fp32 的差距。

CPU 测试另外检查参照实现的假设与编译出的 IR 一致（每层的 op 序列、scan 的 dt clamp、gated norm 的分组与顺序、所有 norm 的 eps），并检查参照读取的张量名与初始化写出的完全相同（A0 与 A0-split、tied 与 untied）。

生成 G1(b) 的 goldens（CPU 即可，A0、batch 2、512 token 在单线程上约 6 秒）：

```bash
python -I -B fork/tlc/prototypes/ref_mamba2.py MODEL_DIR OUT.npz --batch 2 --seq-len 512 --seed 0
```

`.npz` 中的数组：`input_ids`、`targets`（int32，[batch, seq_len]，正是 `compute_logprobs` 的两个参数；`targets` 是下一个 token，最后一列为 −100）；`logprobs`（fp32 参照，逐 token 递归）；`logprobs_bf16`（bf16 模式）；`tolerance` = 2 × max |`logprobs_bf16` − `logprobs`|；`meta`（JSON：`model.safetensors` 的 sha256、seed、dt clamp 区间、求值顺序）。G1(b) 的判据：原生 `compute_logprobs(input_ids, targets, use_lora=False)` 与 `logprobs` 的最大绝对差不超过 `tolerance`，且 `meta` 中的 sha256 与所测目录的权重文件一致。本环境中 `init_seed=1` 的 A0 得到的 `tolerance` 约为 0.045。

## 上游接口的 pin

P0 直接依赖下列上游接口；CPU 测试钉住它们，上游改动时测试失败，需检查 TLC 中的用法后再更新 pin。需要原生扩展的模块按源码解析。

- `Mamba2Mixer`、`GenericMLP`、`Qwen3Block`、`HybridBlockStack` 的构造参数，以及 TLC 依赖的默认值：`Mamba2Mixer` 不 clamp dt；`GenericMLP` 默认是融合的 SwiGLU；`Qwen3Block` 默认开 QK-norm、不带 QKV bias。
- 原生 op 名表能被解析，且多于 100 个名字。
- `surogate.cli.main.cli_main` 用 `runpy.run_module(..., run_name="__main__", alter_sys=True)` 在同一进程中运行子命令模块，所以先 `register()` 再交给它，注册对子命令有效。
- `SurogateTrainerWrapper(config, train_files, eval_files=None)` 与无参数的 `train()`；`SFTConfig` 要由调用方执行 `__post_init__()`（上游 `TokenizeDatasets` 也是这样做的）；训练驱动和示例配置设置的每个上游键都确实被 `SFTConfig` 读取（拼错或已废弃的键会被忽略，悄悄用上默认值）。
- 训练器在环境变量 `SUROGATE_REGRESSION_ARTIFACT` 给出路径时写出 regression artifact（loss 曲线、每步时间的分位数、峰值显存、block schema 统计）；原生训练日志是 JSON 数组，验证行带 `step` 与 `loss`。运行记录的摘要读取这些字段。
- 原生 OOM 的形式：分配器抛出消息为 "Cuda OOM when allocating tensor ..." 的 `std::runtime_error`，activation 栈抛出 `std::bad_alloc`。`compute_logprobs(input_ids, targets, use_lora, position_ids, temperatures)` 的参数顺序。TLC 的代码不调用 `train_with_oom_recovery`。
- `TokenizedDataFileWriter(file_name, vocab_size, masking=False, non_overlapping=False)`、`add_document(tokens, position_ids, mask=None)` 与 BIN.TOK v3 的文件头。
- 原生 tokenizer：`tlc` 不在按架构写死正则的名单中，因此从 `tokenizer.json` 读取 split 正则（没有时用 GPT-2 的）；手写 splitter 覆盖 `qwen2` 与 GPT-2 的正则，不覆盖 `python`。
- 原生 DataLoader 读取 non-overlapping 文件的规则：块数为 token 数 ÷ `seq_len`，目标是下一个 token，position 块紧跟 token 块，mask 位为 0 的目标与每行最后一个目标置为 −100。

## 语料、tokenizer 与数据 shard

相关模块：`surogate_tlc.data.corpus`（语料 spec、文档、train/val 划分与 provenance）、`surogate_tlc.data.tokenizer`（训练 tokenizer，以及与原生 tokenizer 的一致性检查）、`surogate_tlc.data.shards`（BIN.TOK v3 shard 与 manifest）。它们都不需要原生扩展，只有 tokenizer 的 `check` 命令会加载原生 tokenizer。

### 语料

语料由一个 TOML spec 描述：顶层是划分参数，每个 `[[source]]` 是一个来源及其 provenance，对应 v0.7 §9 的来源表。

```toml
seed = 0               # 划分用的种子
val_fraction = 0.01    # 进入 val 的文档比例

[[source]]
name = "stack-python"  # 各来源不同名
kind = "code"          # code 或 text
files = ["python/*.jsonl"]   # glob，相对 spec 所在目录，每个至少匹配一个文件
url = "https://huggingface.co/datasets/..."
revision = "..."       # 可选
license = "..."
downloaded = 2026-10-20
stage = "M1"           # 可选
filters = ["..."]      # 下载之后、入库之前做过的过滤；可以为空列表
```

- 缺少必需键、出现未知键、glob 匹配不到文件时报错。
- `.jsonl` 文件每行一个文档，正文在 `text` 字段；可选的 `group`（例如仓库名）让同组的文档落在同一划分。其他文件整个是一个文档。
- 入库规则：换行按 Python 的通用换行统一为 `\n`（`\r\n` 和单独的 `\r`），之后为空的文档和不是合法 UTF-8 的文档被丢弃并计数。后续的标签按 UTF-8 字节偏移对齐时，看到的就是模型训练的文本。
- 划分：sha256(seed, group) 按 2⁶⁴ 的比例小于 `val_fraction` 的文档进 val；没有 `group` 的文档自成一组。tokenizer 只用 train 划分训练，shard 按同一划分写出。
- 配比（v0.7：70% Python、30% 英文）在准备输入文件时决定；shard manifest 按来源记录实际的 token 数和 train 中的占比，供核对。

### Tokenizer

```bash
python -m surogate_tlc.data.tokenizer train CORPUS.toml OUT_DIR --pretokenizer qwen2 --vocab-size 16384
```

- 16384 个 id 的字节级 BPE，没有 normalizer，decode(encode(x)) == x。特殊 token 的 id 固定为 0–4：`<|pad|>`、`<|bos|>`、`<|eos|>`、`<|code_start|>`、`<|code_end|>`，其中前三个与 `hf.default_config()` 的 pad、bos、eos 一致。语料不够学满词表时报错。
- 预分词有三个候选（v0.7 §3.5），由 `--pretokenizer` 选择，默认 `qwen2`：

  | 名称 | 规则 | 原生 tokenizer 的实现 |
  |---|---|---|
  | `qwen2`（默认） | Qwen2 的 split 正则：GPT-2 一族，字母和数字用 Unicode 类 `\p{L}`、`\p{N}`，每个数字单独成片 | 手写 splitter |
  | `gpt2` | GPT-2 正则，连续数字成一片 | 手写 splitter |
  | `python` | v0.7 的 pyaware2 改用 `\p{L}`、`\p{N}`：标识符（含 `_`）、常见的多字符运算符、换行连同其后的缩进、整个数字（含小数和指数）各成一片 | std::regex 回退 |

  - 默认取 `qwen2`，因为工作单要求数字逐位切分；v0.7 只给出候选和选择规则（Python 加英文的总 token 数、跨词素率、多角色率），还没有选定。比较总 token 数时，用各候选分别建 shard，对比 manifest 中每个来源的 `characters` 与 `real_tokens`；跨词素率和多角色率要用 P1 的对齐与角色标注来算。
  - v0.7 的第三个候选"严格按 Python 词法切分"不是正则预分词，原生 tokenizer 推理时做不到，所以没有提供。
  - `model_type` 为 `tlc` 时，原生 tokenizer 从 `tokenizer.json` 读取 split 正则，没有时用 GPT-2 的。`qwen2` 与 GPT-2 的正则各有手写实现；其他正则（包括 `python`）走 std::regex 回退，用之前必须通过下面的一致性检查。CPU 的 pin 测试钉住这几点。本环境中用原生 tokenizer 源码编译的 CPU 程序，在本机的 stdlib 与仓库文档上训练三个候选，各比较了 295 段文本（640 万字符，含边界用例和 CRLF 版本），全部一致；这不代替 G0。
- 编码：文档正文按普通文本编码，其中的特殊 token 字符串不会变成特殊 id（v0.7：原始文本永远不产生特殊 token）；prompt 按带特殊 token 的方式编码。
- chat template：`<|bos|>` 加各条消息的内容，不要求生成提示时最后加 `<|eos|>`。prompt 就是文档的开头，与 shard 的框定一致。
- 输出目录：`tokenizer.json`、`tokenizer_config.json` 和 `tokenizer_manifest.json`（配方、`tokenizer_sha256`、语料 spec 的 sha256、各来源的 provenance 和参与训练的文档数）。tokenizer 的身份是 `tokenizer.json` 的 sha256（`surogate_tlc.data.tokenizer.tokenizer_sha256()`），shard 的 manifest 也记录它。同样的语料与参数训练出逐字节相同的文件。
- 一致性检查：`compare()` 比较本 tokenizer 与原生 tokenizer 的特殊 id 和词表大小，并逐段比较普通编码、带特殊 token 的编码，以及两侧对普通编码 id 的解码。文本是固定的边界用例（`EDGE_CASES`：CRLF、单独的 CR、tab、CJK 标识符、emoji、f-string、字符串中的 U+2028、特殊 token 字符串等）加语料样本。CPU 测试用替身检查 harness 能发现各类差异；与真实原生 tokenizer 的比较在 GPU 主机上进行，见 G0。

### 数据 shard

```bash
python -m surogate_tlc.data.shards CORPUS.toml TOKENIZER_DIR OUT_DIR --seq-len 512 --train-files 1
```

- 输出 `train-00000.bin` 等、`val-00000.bin`，最后写 `manifest.json`。`OUT_DIR` 必须不存在或为空；同样的输入得到逐字节相同的输出。
- 格式：BIN.TOK v3（1024 字节的头、int32 token、int32 position、按位存储的 mask），头中标为 non-overlapping。写出的文件与上游 `TokenizedDataFileWriter(masking=True, non_overlapping=True)` 的逐字节相同（CPU 测试）；上游模块 import 时就需要原生扩展，所以 fork 自己写。
- 每行一个文档，不跨文档打包：`mamba_ssm_scan` 在打包的文档之间不复位状态。A1、A1h 也这样做，保证各行的数据语义相同。
  - 框定：代码文档为 `<|bos|> <|code_start|> 正文 <|code_end|> <|eos|>`；英文文档为 `<|bos|> 正文 <|eos|>`，其中的 markdown 代码块按行转换（v0.7 §3.5）：去掉缩进后以 ``` 开头的行打开代码块，换成 `<|code_start|>`；代码块内恰为 ``` 的行关闭它，换成 `<|code_end|>`；文档结束时仍未关闭的代码块在末尾关闭。
  - 窗口：超过 `seq_len` 的文档切成 `seq_len` 个 token 的窗口，相邻窗口共享一个 token，因此除文档的首个 token 外，每个 token 恰好作一次目标。每个窗口是一行，末尾用 `<|pad|>` 补齐；每行的 position 为 0..seq_len−1。
  - mask：输入位置 i 的 mask 位为 1，当且仅当它的目标（第 i+1 个 token）是同一行中的真实 token。原生 DataLoader 对 non-overlapping 文件恰好给出这些行，并把每行最后一个位置的目标置为 −100。
- ValidTokenCount：有 n 个真实 token 的行有 n−1 个有效目标。manifest 按 shard 和按划分记录 `rows`、`tokens`、`real_tokens`、`valid_targets`（一个 epoch 中 ValidTokenCount 之和）和 `pad_fraction`；按来源记录文档数、字符数、行数、真实 token 数、train 中的 token 占比、丢弃计数和 provenance（URL、revision、license、下载日期、过滤规则，以及每个输入文件的大小与 sha256）；`rules` 列出入库与框定规则。
- 身份：数据的身份是 `manifest.json` 的 sha256（`surogate_tlc.data.shards.manifest_sha256()`），与 `tokenizer_sha256` 一起写进每个结果行（v0.7 IMPL-08）。训练驱动在开始前核对 shard 的 `seq_len` 等于训练配置的 `sequence_len`，manifest 中的 tokenizer sha256 等于模型目录中 `tokenizer.json` 的 sha256。
- 原生 DataLoader 依次读取各文件，只在文件内打乱行，所以 train 文档按 hash 分到 `--train-files` 个文件，每个文件都是各来源的随机样本。单个文件少于 2³¹ 个 token。
- `surogate_tlc.data.shards.served_rows()` 按原生 DataLoader 的读取规则给出每行的输入、目标与 position。CPU 测试用它核对上面的窗口与 ValidTokenCount；GPU 主机上的测试核对它与原生 DataLoader 一致。本环境中用原生 `dataloader.cpp` 编译的 CPU 程序读取本机语料建出的 shard，逐行一致。
- padding 浪费（`pad_fraction`）超过约 25% 时，再考虑实现文档复位并改用打包 shard，所有行同时切换。本机 stdlib 与仓库文档按整文件切窗时约为 4.6%。
- 上下文：P0 用 512。v0.7 默认正式训练用 2048（O-1，待 owner 确认）：shard 用 `--seq-len 2048` 生成，模型 config 的 `max_position_embeddings` 也要改，而 `check_config` 目前要求 512。

## 训练驱动

`surogate_tlc.train.driver.train()`（命令 `train`）直接调用 `SurogateTrainerWrapper(config, train_files, eval_files).train()`（`surogate/train/trainer.py`），不经过上游的 `sft` 命令。上游 `sft` 先 tokenize 数据集，再用 `train_with_oom_recovery`（`surogate/train/sft.py`）包住训练：OOM 时它把 microbatch 减半，GA 达到 16 以后不再加倍，于是每次更新的 token 数悄悄减半，只留一条 warning。TLC 的数据是 fork 写的 shard，各行比较时每次更新的名义 token 数固定。

### 运行配置

运行配置是上游的 SFT 配置文件（`SFTConfig` 读取的键）加一张 `tlc` 表，由 `surogate_tlc.train.config` 读取。示例 [examples/a0.yaml](examples/a0.yaml)、[examples/a1.yaml](examples/a1.yaml)、[examples/hybrid.yaml](examples/hybrid.yaml) 分别对应 A0、A1、A1h，路径都放在环境变量 `TLC_RUNS` 之下（见运行手册）；字符串中的 `${VAR}` 从环境替换，变量未设置时报错。

```yaml
tlc:
  data: ${TLC_RUNS}/data      # build-data 的输出目录
  tokens_per_update: 16384    # 每次更新的名义 token = microbatch × GA × sequence_len × gpus
  on_oom: fail                # fail 或 downshift
  max_downshifts: 2           # downshift 时 microbatch 最多减半几次
  stages:                     # 用 --stage 选择，覆盖 tlc 表之外的键
    smoke: {max_steps: 10, save_steps: 5}
    overfit: {max_steps: 100, fixed_batch: true}
```

- `gradient_accumulation_steps` 由 `tokens_per_update` 和 microbatch 推出，配置里不写；写了而且不一致就报错。示例的 microbatch 是 32 行 × 512 token、GA 为 1，不开 recompute（梯度检查点），这是 v0.7 §13 的默认值（按 v0.7 的估计，v0.6 的 2 × 16 受 kernel launch 开销限制）。`--microbatch B` 改 microbatch，GA 随之调整。
- 运行目录是 `<output_dir>/<阶段>`，给出 `--microbatch B` 时是 `<output_dir>/<阶段>-mb<B>`，所以 microbatch 扫描中的各次运行不会接着别人的 checkpoint 训练。checkpoint、上游日志和运行记录 `tlc_run.json` 都在运行目录中。
- 驱动替配置设定下列值，配置写了别的值就拒绝：`lora: false`、`sample_packing: false`、`row_packing: false`、`from_scratch: false`。上游默认开 LoRA 和 tokenize 期打包；`from_scratch` 会用原生初始化覆盖 fork 写的权重。
- 不能出现的键：`datasets`、`validation_datasets`（数据只来自 `tlc.data`），`distillation`（P0 没有蒸馏），`checkpoint_dir`（上游不读它，checkpoint 总在 `output_dir` 中，写了只会误导）。`max_steps` 必须是正整数：每次运行的长度固定。
- 训练前检查（`surogate_tlc.train.config.problems()`，有任何问题就不开始，退出码 2）：模型目录满足契约且带权重；`sequence_len` 等于 shard 的行长，且不超过 `max_position_embeddings`；manifest 中 tokenizer 的 sha256 等于模型目录中 `tokenizer.json` 的 sha256；数据目录中有 `train-*.bin`。
- 训练文件是 `tlc.data` 中全部的 `train-*.bin`，评测文件是全部的 `val-*.bin`。阶段设 `fixed_batch: true` 时，驱动把第一个 train 文件的前 `tokens_per_update / sequence_len` 行（示例中 32 行）原样写成运行目录中的 `fixed-batch.bin`，每一步都训练这一批行，用于 100 步过拟合。
- 每次更新中非 pad 的有效目标数（ValidTokenCount 的期望值）写进运行记录：固定 batch 时是这批行的精确值，否则按 manifest 中 train 划分的 `valid_targets / tokens` 折算。上游训练器不按步记录 ValidTokenCount。

### OOM

- 默认（`on_oom: fail`）：OOM 时立即失败，运行记录中写 `failure_report`：错误类型、消息与 traceback、当时的 microbatch 与 GA、显存（`torch.cuda.mem_get_info`）、最后一个 checkpoint，以及不降档的原因。
- `on_oom: downshift`：microbatch 减半、GA 加倍，每次更新的名义 token 数不变，最多 `max_downshifts` 次；microbatch 为奇数时不降。只在运行还没有 checkpoint 时降档，因为 checkpoint 中数据加载器的位置以 microbatch 为单位，换了 microbatch 就接不上。每次降档都记进运行记录；降档不改架构、序列长度和每次更新的 token 数。
- 识别 OOM：消息含 "out of memory" 或单词 "OOM" 的 `RuntimeError`（原生分配器的 "Cuda OOM when allocating tensor ..."，CUDA 的 out of memory），以及 `MemoryError`（原生 activation 栈的 `std::bad_alloc`）。其他错误一律直接失败。
- 在同一进程中重试之前，驱动清除 traceback 对失败 trainer 的引用并回收内存。原生侧 OOM 之后能否在同一进程中重建 trainer，要在 GPU 上确认 [I]；上游的 `train_with_oom_recovery` 也是在同一进程中重试。

### 运行记录与收尾

运行记录 `tlc_run.json`：

- `identity`：初始权重、config.json、tokenizer 和数据 manifest 的 sha256。运行目录中已有记录而身份不同时拒绝运行。
- `attempts`：每次尝试的 microbatch、GA、名义 token 数、配置文件的 sha256、从哪一步恢复、环境指纹、上游日志的路径、状态，以及完成时的摘要。
  - 环境指纹（`surogate_tlc.train.report.environment()`）：提交、工作区是否有未提交的改动、Python 与各包的版本、原生扩展的路径与 sha256、GPU 名称、计算能力、显存和驱动版本。
  - 摘要：loss 曲线、首末 loss、验证 loss、按每步时间中位数折算的 tokens/s（名义 token，含 pad）、峰值显存、block schema 统计。驱动为每次尝试设置环境变量 `SUROGATE_REGRESSION_ARTIFACT`，上游训练器据此写出 `regression-<尝试序号>.json`；摘要取自它和上游日志。
- `downshifts`、`status`（running、completed、failed）、`failure_report`、`data`（文件、固定 batch 的统计、每次更新的有效目标数）、`sft`（最后一次尝试实际使用的上游设置）。

收尾：训练完成后，驱动对运行目录中每个完整的 step 目录执行 `finalize_checkpoints()`，再对运行目录本身（上游的最终导出）执行 `finalize_dir()`；训练失败时只收尾已有的 checkpoint。`tlc_meta.json` 另记 `tokenizer_sha256`、`data_manifest_sha256` 和 `shard_version`（shard 格式版本，即 `surogate_tlc.data.shards.SHARD_VERSION`，现为 1）。因此运行目录和其中的 step 目录都可以直接用作 `model_dir`；`export` 保留 step 目录 `tlc_meta.json` 中的这些字段。

恢复：再次运行同一条命令时，上游从运行目录中最新的 checkpoint 继续（`resume_from_checkpoint: true`）。运行已有 checkpoint 时，驱动沿用最后一次尝试的 microbatch 与 GA（降档后也是如此）；配置的 microbatch 与第一次尝试不同则拒绝。

`python -m surogate_tlc.train.report DIR` 把 DIR 下每个运行记录的最后一次尝试打印成一行 JSON（阶段、状态、microbatch、GA、有效目标数、降档次数、首末 loss、验证 loss、tokens/s、峰值显存），用于 G3 的记录和下文的 GPU 预算。

## 评测

`python -m surogate_tlc eval MODEL DATA`（`surogate_tlc.eval.loss.evaluate()`）按原生 DataLoader 的读法（`served_rows()`）取 `DATA` 的 val 划分，用原生 `compute_logprobs(use_lora=False)` 打分；loss 是有效目标上的平均负对数概率。基线是 train 划分中目标 token 的 add-one 平滑 unigram 分布下的 loss，v0.7 的 G3 要求验证 loss 低于它（输出中的 `below_unigram`）。模型目录的 tokenizer 必须与 shard 的一致。输出另有行数、有效目标数、困惑度、tokenizer 与数据 manifest 的 sha256。

绑定探针、按角色分层的 loss 等 teacher-forced 指标由后续步骤加入 `surogate_tlc.eval`。P0 没有生成样例：原生 `next_token_logits` 要求 trainer 带 LoRA，TLC 的生成要等 P5-a 的 rollout server（给全量训练的 checkpoint 挂一个零初始化的 rank-1 LoRA）。

## GPU 预算

按 G3 的实测填写。tokens/s 取自 `python -m surogate_tlc.train.report` 的输出（名义 token，含 pad；按 `valid_targets_per_update / tokens_per_update` 可折算成有效 token）。

GPU 小时 = Σ 种子数 × 名义 token 数 ÷ 实测 tokens/s × (1 + 评测与保存开销) ÷ 3600。

- 关键比较至少 3 个种子，pilot 可以只用 1 个。
- G3 在每次更新 16,384 token 不变的前提下测 microbatch 2、8、32（GA 相应为 16、4、1）；tokens/s 只在相同 microbatch 之间比较，每行注明所用档位（v0.7 §13）。
- v0.7 的 G0.5 还要测 T = 2048；那要等 owner 确认 O-1，并放开 `check_config` 中上下文为 512 的要求。

| 行 | 阶段 | 种子数 | 名义 token（每个种子） | B × GA | 实测 tokens/s | 评测与保存开销 | GPU 小时 |
|---|---|---|---|---|---|---|---|
| A0 | 300 步 pilot | 1 | 4,915,200 | 32 × 1 | 待 G3 | 待 G3 | 待 G3 |
| A1 | 300 步 pilot | 1 | 4,915,200 | 32 × 1 | 待 G3 | 待 G3 | 待 G3 |
| A1h | 300 步 pilot | 1 | 4,915,200 | 32 × 1 | 待 G3 | 待 G3 | 待 G3 |
| A0、A1、A1h | 正式训练 | 3 | 待定 | 待 G3 | 待 G3 | 待 G3 | 待 G3 |

## 运行手册（RTX 5090，G0–G3）

命令都在 GPU 主机的仓库根目录运行。门槛按顺序进行：某一步的命令以非零退出码结束、或判据不成立，就停止，把输出和运行记录中的 `failure_report` 存进 `$TLC_RUNS/env/`，不进入下一个门槛（v0.7 §14；A0 没有通过 G1 或 G2 时不进入后续阶段）。标 [P] 的阈值是暂定的：不成立时记录实测值并停下，不要放宽阈值。GPU 测试单独运行，不与 `tests/cpu` 在同一次 pytest 中收集：CPU 测试会把仓库根目录放到 `sys.path` 最前面，GPU 测试则用 `fork/.venv-tlc` 的导入路径。

```bash
set -o pipefail                  # `pytest ... | tee` 保留 pytest 的退出码
export TLC_RUNS=/data/tlc        # 所有产物的根目录，放在仓库之外
PY=fork/.venv-tlc/bin/python
mkdir -p "$TLC_RUNS/env" "$TLC_RUNS/corpus" "$TLC_RUNS/goldens"
```

语料 spec `$TLC_RUNS/corpus/corpus.toml` 由 owner 按上文“语料”准备（v0.7 §9 的来源表、70/30 配比、合成数据占比）。

### G0 环境

1. 原生扩展。根目录 `.venv` 按 [fork/README.md 的“首次准备环境”](../README.md#首次准备环境) 创建（只需一次；首次完整构建需要数小时），然后用本检出重建训练扩展。CMake 按本机 GPU 自动选择 `120a`；之后每次同步上游或修改训练侧 C++，都重新运行 `fork/dev build train`。

   ```bash
   uv venv --python 3.12
   uv pip install -e .
   fork/dev build train
   cuobjdump --list-elf surogate/libsurogate-common.so | grep -o 'sm_[0-9]*a*' | sort -u
   ```

   预期最后一条只输出 `sm_120a`。

2. `fork/.venv-tlc`。依赖与根目录 `.venv` 相同，并锁定成文件（G0 的“依赖锁定”）；`surogate` 经 `.pth` 文件从本检出导入，所以 `fork/dev build train` 复制进 `surogate/` 的扩展就是 `.venv-tlc` 加载的扩展，不需要第二次原生构建，也不会留下过期的副本。

   ```bash
   uv venv --python 3.12 fork/.venv-tlc
   uv pip freeze -p .venv --exclude-editable > "$TLC_RUNS/env/requirements.lock"
   uv pip install -p fork/.venv-tlc -r "$TLC_RUNS/env/requirements.lock" pytest
   uv pip install -p fork/.venv-tlc --no-deps -e fork/tlc
   echo "$PWD" > fork/.venv-tlc/lib/python3.12/site-packages/surogate-checkout.pth
   ```

   根目录 `.venv` 的 torch 若来自 download.pytorch.org（锁文件中的版本带 `+cu…`），第三条命令要加上对应的 `--index-url`。

3. 环境指纹与 GPU。

   ```bash
   $PY -c 'import json; from surogate_tlc.train import report; print(json.dumps(report.environment(), indent=2))' > "$TLC_RUNS/env/environment.json"
   $PY -m pytest -c fork/tlc/pyproject.toml fork/tlc/tests/gpu/test_gpu_environment.py
   ```

   预期 2 个测试通过：`surogate._surogate` 与 `surogate_tlc` 都来自本检出，GPU 的计算能力为 (12, 0)，指纹中提交、原生扩展的 sha256、驱动版本齐全。每个运行记录都带同样的指纹。

4. tokenizer、模型目录与 shard。

   ```bash
   $PY -m surogate_tlc train-tokenizer "$TLC_RUNS/corpus/corpus.toml" "$TLC_RUNS/tokenizer" --pretokenizer qwen2
   for row in A0 A0-split A1 A1h; do
     $PY -m surogate_tlc init "$row" "$TLC_RUNS/models/$row" --tokenizer "$TLC_RUNS/tokenizer" --seed 1
   done
   $PY -m surogate_tlc.data.tokenizer check "$TLC_RUNS/models/A0" "$TLC_RUNS/corpus/corpus.toml" --limit 10000
   $PY -m pytest -c fork/tlc/pyproject.toml fork/tlc/tests/gpu/test_data_native.py
   $PY -m surogate_tlc build-data "$TLC_RUNS/corpus/corpus.toml" "$TLC_RUNS/tokenizer" "$TLC_RUNS/data" --seq-len 512
   ```

   - 预分词候选还没有选定（v0.7 §3.5；两份研究备忘另提出 Qwen 子集词表和“标识符保留”预分词两个候选）。owner 在 A0 之前定下，四行共用同一个 tokenizer 和同一个 `--seed`；A0 与 A0-split 用同一个种子，得到同一份权重（G1(d)）。
   - `check`：预期输出 `consistent on N texts`，退出码 0；有差异时停止，不建 shard。`test_data_native.py`：预期 4 个测试通过。
   - `build-data` 输出各划分的 `pad_fraction`；超过约 25% 时记下来（P0b 的触发条件），pilot 照常进行。

5. kernel 与图。

   ```bash
   $PY -m pytest -c fork/tlc/pyproject.toml fork/tlc/tests/gpu/test_gates.py -k g0
   ```

   预期 3 个测试通过（A0、A1、A1h）：在 2 × 512 token 上做一次前向与反向，每个可训练参数的梯度都存在、有限且不全为零。它在 TLC 的图中跑过 BF16 GEMM、selective scan 的前向与反向（A0、A1h）和 head_dim 64 的 flash attention（A1、A1h），也完成两项附带检查：没有 `BlockSchema` 的 `TLCMambaFFNBlock` 被 buffer plan 接受，A1h 中带 MLP 的 `A` 层能运行。A0-split 的模型级第 1 层由 G1(d) 检查。执行沙箱的检查属于 P5-a，不在 P0 的 G0 中。

### G1 因果性

```bash
$PY -I -B fork/tlc/prototypes/ref_mamba2.py "$TLC_RUNS/models/A0" "$TLC_RUNS/goldens/A0.npz" --batch 2 --seq-len 512 --seed 0
$PY -m pytest -c fork/tlc/pyproject.toml fork/tlc/tests/gpu/test_gates.py -k g1 -s | tee "$TLC_RUNS/env/g1.txt"
```

第一条在 CPU 上生成 G1(b) 的 goldens（见上文“数值参照”）。`-s` 让测试打印实测的误差（JSON 行）。

| 选择 | 行 | 判据 |
|---|---|---|
| `-k g1a` | A0、A1、A1h | 把位置 p 起的 token 全部改掉，位置 p 之前的 logprobs 逐位不变，p ∈ {1, 2, 7, 64, 511}；p 之后的 logprobs 确有变化 |
| `-k g1b` | A0 | goldens 记录的权重 sha256 与模型目录一致；原生 logprobs 与参照的最大绝对差不超过 goldens 中的 `tolerance` |
| `-k g1c` | A1；A0 为负对照 | 两个文档打包在一行（第二个文档的 position 从 0 重新开始），与各自单独打分比较：A1 最大绝对差 ≤ 2e-2 [P]；A0 的第一个文档 ≤ 2e-2，第二个文档的差异 > 1e-3，因为 `mamba_ssm_scan` 不在文档之间复位，这正是 Mamba 行每行只放一个文档的原因 |
| `-k g1d` | A0、A0-split | 两者的 `init_seed` 相同；logprobs 最大绝对差不超过 goldens 的 `tolerance`；梯度按 `split_param_name()` 改名后一一对应，最小余弦 ≥ 0.999 [P] |

G1(d) 不成立时，模型级第 1 层的方案不能承载 P2/P3 的 c_t 与槽写入，按支持方案改走 fork IR 后处理 pass。

### G2 数学与梯度

```bash
$PY -m pytest -c fork/tlc/pyproject.toml fork/tlc/tests/gpu/test_gates.py -k g2 -s | tee "$TLC_RUNS/env/g2.txt"
```

| 测试 | 判据 |
|---|---|
| `test_g2_initial_loss_is_near_ln_vocab`（A0、A1、A1h） | 初始权重下随机 token 的交叉熵与 ln 16384 ≈ 9.70 相差不到 0.25（A0 的 CPU 预测为 9.78） |
| `test_g2_gradients_point_where_the_reference_does` | A0 第 1 层与第 12 层的每个参数：原生 bf16 梯度与参照实现（`prototypes/ref_mamba2.py` 的分块形式，fp32 autograd，2 × 128 token）的梯度余弦 ≥ 0.99 [P] |
| `test_g2_resume_reproduces_the_next_steps` | A0 的 smoke 阶段跑两次：一次不中断，一次从第 5 步的 checkpoint 恢复；第 6–10 步的 loss 相差 ≤ 1e-3（v0.7 G2.6 的容差：原生反向用浮点 atomicAdd，不要求逐位相等）。同时检验收尾过的 checkpoint 仍能恢复 |

梯度全部有限由 G0 的第 5 步检查；固定 batch 的过拟合是 G3 的 overfit 阶段。

### G3 基线

```bash
for config in a0 a1 hybrid; do
  for stage in smoke overfit pilot; do
    $PY -m surogate_tlc train "fork/tlc/examples/$config.yaml" --stage "$stage" || break 2
  done
done
for config in a0 a1 hybrid; do
  for microbatch in 2 8; do
    $PY -m surogate_tlc train "fork/tlc/examples/$config.yaml" --stage pilot --microbatch "$microbatch"
  done
done
for row in A0 A1 A1h; do
  $PY -m surogate_tlc eval "$TLC_RUNS/train/$row/pilot" "$TLC_RUNS/data" --out "$TLC_RUNS/train/$row/pilot/eval-val.json"
done
$PY -m surogate_tlc.train.report "$TLC_RUNS/train" | tee "$TLC_RUNS/train/report.jsonl"
```

第一组循环依次跑每行的 10 步冒烟、100 步过拟合和 300 步 pilot（microbatch 32）；第二组在每次更新 16,384 token 不变的前提下，用 microbatch 2 和 8 重跑 pilot（运行目录 `pilot-mb2`、`pilot-mb8`）；`eval` 给出验证 loss 与 unigram 基线；`report` 汇总全部运行。判据：

- 每条 `train` 退出码为 0，`status` 为 completed，loss 曲线中没有 NaN 或 Inf，没有降档（示例配置 `on_oom: fail`）。
- smoke：第 1 步的 loss 接近 ln 16384（同 G2）。
- overfit：在固定 batch 上，第 100 步的 loss 低于第 1 步的一半 [P]。
- pilot：`eval` 输出 `below_unigram: true`（v0.7 G3）。
- 记录每行在 microbatch 2、8、32 下的 tokens/s 与峰值显存，以及每次更新的有效目标数，然后填写上文的 GPU 预算。
- 这些结果是 pilot：G3 只认证流水线，不认证模型质量；生成样例在 P0 中暂缺（见“评测”）。未通过 G3 时不进入 Phase 1。
