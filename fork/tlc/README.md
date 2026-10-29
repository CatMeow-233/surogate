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

## 注册

训练器按 config.json 的 `architectures` 在上游 DSL 注册表中查找模型，并用 transformers 的 `AutoConfig` 读取 config。`surogate_tlc.register()` 做两件事，可重复调用：

- import `surogate_tlc.dsl`，其中 `@nn.hf_config` 把 `TLCModel`（架构名 `TLCForCausalLM`）加进 DSL 注册表。构建 TLC 的 IR 总是需要这一步。
- 把 `TLCConfig` 注册为 `model_type="tlc"` 的 `AutoConfig`。注册后它优先于目录中的 `auto_map`；不注册时，带 `auto_map` 的目录照样能被读取。

DSL 类名一律带 `TLC` 前缀：注册表按类名做键，同名类会悄悄覆盖上游的定义。CPU 测试检查 `register()` 只新增 `TLCModel`，不替换任何上游条目。

## P0 的行

`surogate_tlc.hf.ROWS` 定义各行相对 `default_config()` 的差异，`row_config(row)` 给出完整的 config.json。四行都是同一个 DSL 模型 `TLCModel`（`TLCForCausalLM`），词表 16384，宽度 384，12 层，上下文 512，tied embedding。参数量由 IR 的参数形状算出（`surogate_tlc.ir.param_count`），不含冻结的 RoPE 表；tied 时不计 LM head。

| 行 | 与 A0 的差异 | 层 | 参数量（tied / untied） | 相对 A0 |
|---|---|---|---|---|
| A0 | — | 12 × `M` | 31,782,192 / 38,073,648 | — |
| A0-split | `tlc_split_first_layer: true` | 第 1 层为模型级 ops，其后 11 × `M` 在 stack 中 | 与 A0 相同 | 0 |
| A1 | `tlc_layer_pattern` 为 12 个 `A`，`intermediate_size` 1504 | 12 × `A` | 31,812,480 / 38,103,936 | +0.10% / +0.08% |
| A1h | `tlc_layer_pattern` 为 `MMMMMAMMMMMA` | 10 × `M`，第 6、12 层为 `A` | 30,681,320 / 36,972,776 | −3.46% / −2.89% |

- **Mamba-2 几何**（`surogate_tlc.hf.MAMBA_GEOMETRY`）：H=12 个头、每头 P=64（内宽 768，即 2 × 宽度）、状态 N=64、G=1 组，conv 核 4，chunk 128。v0.7 notebook 不存在，按支持方案 §4.3 定稿：N 取 64 而不取 128，因为 scan kernel 在状态维上的循环是串行的。A0 的参数量由这组几何唯一确定，CPU 测试钉住它。Mamba-2 的 conv 带 bias、输入输出投影不带 bias；dt 不做 clamp（上游 `Mamba2Mixer` 不给 `time_step_limit` 时的行为），数值参照必须一致。
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
- `model.safetensors`：从零训练时由 fork 的初始化工具按 IR 的参数名写出全部权重。Mamba-2 的 `A_log`、`dt_bias`、`D` 用标准初始化（原生侧按参数名初始化的结果对 Mamba 不对：`dt_bias` 为 0，`A_log` 与 `D` 服从正态分布）；tied 时不写 `lm_head`。训练经 `from_pretrained(model_dir)` 加载它。

`surogate_tlc.modeldir.write_model_dir()` 写出除权重以外的全部文件；`check_model_dir()` 检查一个目录并列出全部问题，`weights=False` 时不要求 `model.safetensors`。

### checkpoint 与导出

原生训练器把 checkpoint 存到 `<checkpoint_dir>/step_<8 位步数>/`，最后写 `checkpoint.json`。其中的 config.json 只有通用键，没有 `auto_map`、TLC 自有的键和 tokenizer；训练结束时的导出只从源目录复制一份固定的文件清单，不含 `configuration_tlc.py` 和 `tlc_meta.json`。所以每次保存和导出之后都要收尾，`surogate_tlc.checkpoint` 提供：

- `finalize_dir(target, source, meta=None)`：把源模型目录的 config.json、`configuration_tlc.py`、tokenizer 文件和 `tlc_meta.json`（并入 `meta`）复制进 `target`，原生写的 config.json 被源目录的完整版本替换。`target` 中原生写的结构键（架构、模型类型、词表、宽度、层数、FFN 宽度）与源目录不一致时拒绝执行，以免配错源目录；A0-split 的层数允许少 1，因为原生 config 的层数取自 DSL，而 DSL 的层数不含模型级的第 1 层。可重复执行；`tlc_meta.json` 最后写入，中途失败的目录通不过 `check_model_dir()`。
- `finalize_checkpoints(checkpoint_dir, source)`：对每个已写完 `checkpoint.json` 的 step 目录执行 `finalize_dir`，`tlc_meta.json` 记下 `step`；还在写的目录不动。
- `export(checkpoint, source, out)`：从一个 step 目录导出干净的模型目录，只带 `model.safetensors` 和收尾文件，不带优化器状态和分片。`out` 必须不存在或为空。

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

## 上游接口的 pin

P0 直接依赖下列上游接口；CPU 测试钉住它们，上游改动时测试失败，需检查 TLC 中的用法后再更新 pin。需要原生扩展的模块按源码解析。

- `Mamba2Mixer`、`GenericMLP`、`Qwen3Block`、`HybridBlockStack` 的构造参数，以及 TLC 依赖的默认值：`Mamba2Mixer` 不 clamp dt；`GenericMLP` 默认是融合的 SwiGLU；`Qwen3Block` 默认开 QK-norm、不带 QKV bias。
- 原生 op 名表能被解析，且多于 100 个名字。
- `surogate.cli.main.cli_main` 用 `runpy.run_module(..., run_name="__main__", alter_sys=True)` 在同一进程中运行子命令模块，所以先 `register()` 再交给它，注册对子命令有效。
- `SurogateTrainerWrapper(config, train_files, eval_files=None)` 与无参数的 `train()`。
- `TokenizedDataFileWriter(file_name, vocab_size, masking=False, non_overlapping=False)`、`add_document(tokens, position_ids, mask=None)` 与 BIN.TOK v3 的文件头。

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

待填写。范围：创建 `fork/.venv-tlc`，`fork/dev build train`（sm_120a），环境指纹，BF16 GEMM，selective scan 的前向与反向，A1 的 flash-attention（head_dim 64），至少 10k 文档上原生与 HF tokenizer 的一致性；附带检查原生侧接受四行的图：没有 `BlockSchema` 的 `TLCMambaFFNBlock`、A1h 中带 MLP 的 `A` 层、A0-split 的模型级第 1 层。

### G1 因果性

待填写。范围：修改位置 t 之后的 token 不改变位置 ≤ t 的 logprobs（前缀长度 1、2、7、64、511）；原生全序列 logprobs 与参照实现的逐 token 递归一致；A1 打包与分开运行一致；A0-split 与 A0 等价（同一份权重按 `split_param_name()` 改名，logits 与梯度在容差内一致）。

### G2 数学与梯度

待填写。范围：step-0 loss ≈ ln 16384 ≈ 9.70；梯度全部有限；单层梯度与参照一致；固定 batch 100 步过拟合；save → resume 后 logits 和下一步 loss 复现。

### G3 基线

待填写。范围：每行依次 10 步冒烟、100 步过拟合、300 步 pilot；记录 train/val loss、microbatch 2/8/32 的 tokens/s、峰值显存、ValidTokenCount 和样例，然后填写上文的 GPU 预算。
