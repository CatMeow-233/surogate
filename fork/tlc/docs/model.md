# 模型、checkpoint 与数值参照

相关模块：`surogate_tlc.hf`（config 的 schema 与检查）、`surogate_tlc.modeldir`（写出和检查模型目录）、`surogate_tlc.checkpoint`（收尾与导出）、`surogate_tlc.dsl`（DSL 模型）、`surogate_tlc.ir`（IR 检查、参数量与 FLOPs）、`surogate_tlc.init`（初始权重），以及 [prototypes/](../prototypes/) 中的数值参照。

## 注册

训练器按 config.json 的 `architectures` 在上游 DSL 注册表中查找模型，并用 transformers 的 `AutoConfig` 读取 config。`surogate_tlc.register()` 做两件事，可重复调用：

- import `surogate_tlc.dsl`，其中 `@nn.hf_config` 把 `TLCModel`（架构名 `TLCForCausalLM`）加进 DSL 注册表。构建 TLC 的 IR 总是需要这一步。
- 把 `TLCConfig` 注册为 `model_type="tlc"` 的 `AutoConfig`。注册后它优先于目录中的 `auto_map`；不注册时，带 `auto_map` 的目录照样能被读取。

DSL 类名一律带 `TLC` 前缀：注册表按类名做键，同名类会悄悄覆盖上游的定义。CPU 测试检查 `register()` 只新增 `TLCModel`，不替换任何上游条目。DSL 类叫 `TLCModel`，架构名是 `TLCForCausalLM`；`configuration_tlc.py` 是包里的一个静态文件，原样复制进每个模型目录。

## config.json 的不变量

`surogate_tlc.hf.check_config()` 逐条检查并列出全部问题；fork 的入口在构建或加载模型之前调用它。原生训练器自己读 config.json，下面多数错误它不报错，或者要到构建模型时才失败：

- `architectures=["TLCForCausalLM"]`，`model_type="tlc"`，`auto_map={"AutoConfig": "configuration_tlc.TLCConfig"}`。`model_type` 不能是 `qwen*`（模型会被交给服务引擎执行），也不能含 `vl` 或 `vision`（会打开视觉输入）。
- `vocab_size`、`hidden_size`、`num_hidden_layers`、`intermediate_size` 为正整数；`max_position_embeddings` 是整数且等于 `surogate_tlc.hf.CONTEXT_LENGTH`（512；`512.0` 和 `true` 都不行）；`rms_norm_eps` 是有限的正数（缺少时 DSL 要到编译才失败）；`tie_word_embeddings` 显式给出；`bos_token_id`、`eos_token_id`、`pad_token_id` 都在词表之内（原生侧不检查 id 范围）。
- `num_attention_heads>0` 或 `head_dim>0`，没有 attention 的 A0 也一样：原生侧在 `head_dim` 为 0 时用 attention 头数做除数。
- `tlc_layer_pattern` 是字符串，长度等于 `num_hidden_layers`，每层一个字符：`M` 为 Mamba-2 + SwiGLU，`A` 为 attention 层（复用上游 `Qwen3Block`）。含 `M` 时 DSL 把它（A0-split 为去掉第 1 层后的部分）写进 IR 的字符串属性 `hybrid_pattern`；list 类型的属性在 DSL 编译时会被丢弃。
- `tlc_split_first_layer` 可省略（即 false）；给出时必须是布尔值。为 true 时第 1 层必须是 `M`，且至少还有一层。
- 含 `M` 时 Mamba 的几何显式写入，不依赖默认值：`mamba_num_heads`（H）、`mamba_head_dim`（P）、`ssm_state_size`（N）、`n_groups`（G，整除 H）、`conv_kernel`、`chunk_size`。
- 含 `A` 时 `num_attention_heads`、`num_key_value_heads`、`head_dim` 为正，且头数是 KV 头数的整数倍。
- 以后加入 GDN 层时，四个 `linear_*` 几何键都要求给出：缺少时 kernel 退回按 attention 的几何编译，而且不报错。

`surogate_tlc.hf.default_config()` 给出 A0，`row_config(row)` 给出各行的 config，两者都可按键覆盖。

## 模型目录

可用作 `model_dir` 的目录包含：

- `config.json`，满足上面的不变量；
- `configuration_tlc.py`：`auto_map` 指向它，内容就是包里的 `surogate_tlc.configuration_tlc`，只依赖 transformers，因此不 import fork 的进程也能读取 config；
- tokenizer 文件：`tokenizer.json` 与带 `chat_template` 的 `tokenizer_config.json`（也可以是单独的 `chat_template.jinja`）。`bos_token`、`eos_token`、`pad_token` 在 tokenizer 中的 id 必须等于 config.json 中的值，最大 token id 小于 `vocab_size`；
- `tlc_meta.json`：`tlc_meta_version`（当前为 1）、`init_seed`、`label_ontology_version`、`shard_version`、`sidecar_version`，后几项在用到之前为 null。checkpoint 与导出目录另有 `step`、`kind`（`checkpoint`、`run` 或 `export`），训练驱动收尾时还写入 `tokenizer_sha256`、`data_manifest_sha256` 和 `run`（训练设置与运行配置文件的 sha256、提交、工作区是否有改动）；
- `model.safetensors`：从零训练时由 `surogate_tlc.init.init_model_dir()` 写出，见下文“初始化”。训练经 `from_pretrained(model_dir)` 加载它。

`surogate_tlc.modeldir.write_model_dir()` 写出除权重以外的全部文件；`check_model_dir()` 检查一个目录并列出全部问题，`weights=False` 时不要求 `model.safetensors`。

## checkpoint 与导出

原生训练器把 checkpoint 存到 `<checkpoint_dir>/step_<8 位步数>/`，最后写 `checkpoint.json`。其中的 config.json 只有通用键，没有 `auto_map`、TLC 自有的键和 tokenizer；训练结束时的导出只从源目录复制一份固定的文件清单，不含 `configuration_tlc.py` 和 `tlc_meta.json`。所以每次保存和导出之后都要收尾，`surogate_tlc.checkpoint` 提供：

- `finalize_dir(target, source, meta=None)`：把源模型目录的 config.json、`configuration_tlc.py`、tokenizer 文件和 `tlc_meta.json`（并入 `meta`）复制进 `target`，原生写的 config.json 被源目录的完整版本替换。`target` 中原生写的结构键（架构、模型类型、词表、宽度、层数、FFN 宽度）与源目录不一致时拒绝执行，以免配错源目录；A0-split 的层数允许少 1，因为原生 config 的层数取自 DSL，而 DSL 的层数不含模型级的第 1 层。step 目录归原生训练器所有，收尾只能往里加文件：每个文件原子替换，`tlc_meta.json` 最后写入，第一次收尾中途失败的目录通不过 `check_model_dir()`。可重复执行。
- `finalize_checkpoints(checkpoint_dir, source, meta=None)`：对每个已写完 `checkpoint.json` 的 step 目录执行 `finalize_dir`，`tlc_meta.json` 并入 `meta` 并记下 `step`；还在写的目录不动。
- `export(checkpoint, source, out)`：从一个 step 目录导出干净的模型目录，只带 `model.safetensors` 和收尾文件，不带优化器状态和分片；step 目录已收尾时，保留其 `tlc_meta.json` 中的字段。导出先在 `out` 旁边的临时目录中完成并通过检查，再整体改名为 `out`，所以 `out` 要么是完整的模型目录，要么不变。`out` 必须不存在或为空。

规则：

- 未经收尾的原始 checkpoint 目录不得直接用作 `model_dir`。
- 收尾后的目录能在不 import fork 的新解释器中用 transformers 读取 config 与 tokenizer（CPU 测试）；save → resume 后 logits 和下一步 loss 能否复现、导出到另一个 seed 的源目录后权重是否逐位不变，在 G2 中验证。
- CPU 测试钉住原生的 step 目录命名、`checkpoint.json` 在全部分片和导出之后写入，以及上游导出复制的文件清单；上游改动这些时测试失败。

## DSL 模型与 IR 检查

- `surogate_tlc.dsl.blocks`：`TLCMambaFFNBlock`，即 RMSNorm → 上游 `Mamba2Mixer` → RMSNorm → 上游 `GenericMLP`（SwiGLU，up 与 gate 融合为一个权重）。名字映射取自上游：两个 norm 与 SwiGLU 用 dense 块的名字，mixer 用 Nemotron-H 的名字，因此上游给 `Mamba2Mixer` 新增的激活也会被改名。它没有 `BlockSchema`：原生侧的 buffer plan 把 schema 当作可选信息（只有设置 `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1` 时才要求每层都有），能否接受由 G0 的附带检查确认。
- `surogate_tlc.dsl.models`：`TLCModel`。整个模型只有一个 `HybridBlockStack`；Mamba 的可选参数由模型级别名 `use_conv_bias`、`use_bias` 打开；embedding、final_norm、lm_head 的权重名显式给出，其余权重按 IR 名加载。
  - A1 的 IR 与同尺寸的上游 `Qwen3ForCausalLM` 逐 op、逐参数相同（CPU 测试），只是 checkpoint 中的权重名不同。A1 不设 `hybrid_pattern`，所以原生侧把每层当作 dense 块，与上游 Qwen3 的路径一致；它的 config.json 不带 Mamba 键。
  - A1h：原生侧按 `hybrid_pattern` 把 `A` 层当作 attention 块，而 `Qwen3Block` 带 MLP：这在 GPU 上验证（G0）。
  - A0-split：第 1 层的 ops 由同一个 `TLCMambaFFNBlock` 在模型级追踪，名字加 `first_layer_` 前缀（例如 `first_layer_in_proj_weight`），其后的 stack 是第 2..L 层，层号从 `blocks[0]` 重新开始。CPU 测试检查 A0-split 与 A0 的 op、属性和顺序完全相同，张量一一对应，A0 的参数按 `surogate_tlc.dsl.models.split_param_name()` 改名后形状不变；所以同一份权重改名后即可做 G1(d) 的等价测试。IR 的 `n_layers` 为 11、`hybrid_pattern` 为 11 个 `M`（原生侧的层数只算 stack）。原生侧能否正确执行、重算这些模型级 ops，由 G1(d) 判定；若不等价，改走 fork IR 后处理 pass。模型级追踪用到上游 tracer 的私有成员，由 pin 测试钉住。
- `surogate_tlc.ir`：
  - `compile_ir(cfg)`：先 `register()`，再按 config 构建训练器会用的 IR。
  - `param_shapes()`、`param_count(module, tied=..., embedding=...)`：符号维度按原生参数存储的定义求值（`C`、`M`、`MUp`、`AttnDim`、`QKV`）。
  - `flops_per_token(cfg, seq_len)`：前向 FLOPs/token 的估计，用于预算和 MFU；不含 norm、激活和 loss。
  - `native_op_names(repo)`：从 `csrc/src/runtime/dsl/graph_compiler.cpp` 的 `op_type_from_string` 表读出原生接受的 op 名。
  - `ir_problems(module, native_ops)`：列出前向图的问题，空列表表示通过。检查项：op 名在原生表中；每个张量只写一次，不覆盖参数或图输入；每个输入都是参数、图输入或之前的 op 写出的张量；图输出都被写出；`moe_softmax` 的输入是 2 维（原生 kernel 直接把前两维当作 token 数和专家数）；`concat`、`mul` 的输入在有 dtype 标注时 dtype 相同（原生侧没有 cast op）。
- 这些检查对应的 DSL 陷阱都有回归测试：两个 layer stack 都把层命名为 `blocks[0..]`，后一个覆盖前一个；同一个 module 调用两次时第二次覆盖第一次的激活（共享权重循环的朴素写法就是这样）；模型缺少 `use_conv_bias` 别名时 `conv_bias` 从参数中消失，而图仍读取它。所有行的 IR 都通过上述检查（CPU 测试）。后续阶段新增的图结构（`tlc_aux_ce` 是 loss 的祖先、GDN 几何与 `linear_*` 键一致）在对应阶段加入 `ir_problems`。
- 编译出的 IR 中，所有层共用一组按名字合并的激活槽（`mlp_up` 等只有一个形状），所以各层不能有不同的 FFN 宽度；按层设宽度的行（A2-pm）要先给激活另起槽名。

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
- 残差缩放取 √(2L)：每层有一次 mixer 写入和一次 MLP 写入，`MixerModel` 在层带 MLP 时传 `n_residuals_per_layer=2`。`MambaLMHeadModel` 随后用默认值 1 再执行一次 `_init_weights`，实际效果是 √L；TLC 按每层两次写入取 √(2L)，常量为 `surogate_tlc.init.RESIDUAL_WRITES_PER_LAYER`。
- weight decay：参照实现不对 `A_log`、`D`、`dt_bias` 做 weight decay，原生 AdamW 却按名字对所有参数一视同仁（norm 和 embedding 也一样），decay 会把它们拉向上面那种原生默认状态。所以 P0 的运行配置设 `weight_decay: 0.0`；按参数排除需要原生改动，留作后续的 ledger topic。
- 新增的参数字段必须在 `surogate_tlc.init.RULES` 中有规则，否则报错。
- 张量名是 IR 的 `hf_mapping` 名，没有映射时就是 IR 参数名，与原生 loader 查找的名字相同；冻结的 RoPE 表由运行时计算，不写入；tied 时不写 `lm_head.weight`，原生 loader 退回读取 embedding。dtype 与 IR 参数一致：`A_log`、`D_param`、`dt_bias` 为 fp32，其余为 bf16。safetensors 元数据记录 `tlc_init_version`（方案改动时加一）和 `tlc_init_seed`。
- 每个张量用自己的随机数发生器，种子由 (`init_seed`, 层号, 字段名) 派生。同一 seed 下 A0-split 得到的就是 A0 的权重，只是名字按 `split_param_name()` 改过，所以 G1(d) 直接比较两个用同一 seed 初始化的目录。
- CPU 测试：张量名由单层模型的层内参数名和模型级映射重新推导，不复用初始化代码的命名逻辑；形状、dtype、各类参数的分布统计；同一 seed 可复现、不同 seed 不同；A0-split 与 A0 一致。初始化后的 A0 在数值参照上，随机 token 的初始 loss 与 ln 16384 ≈ 9.70 相差不到 0.25（tied 的 N(0, 0.02) embedding 使期望值高出约 0.08）。

## 数值参照

[prototypes/ref_mamba2.py](../prototypes/ref_mamba2.py) 是纯 PyTorch 的 Mamba-2 参照实现：读取模型目录的 config.json 与 `model.safetensors`（或直接接收权重），逐 op 复现原生图对 A0、A0-split 的计算（只支持 `M` 层，A1、A1h 不在范围内）。与原生对齐的细节：

- in_proj 的输出顺序为 [z | x B C | dt]，conv 带 bias 后接 SiLU；
- dt = clamp(softplus(dt_raw + dt_bias), time_step_limit)，softplus 在输入大于 20 时取线性；TLC 不设 `time_step_limit`，原生的 clamp 区间为 (0, 10⁹)；
- gated norm 是 RMSNorm(y · silu(z))（先乘门再归一化），按 `n_groups` 分组求 RMS；
- 残差与 RMSNorm 融合：norm 用和计算，存下的残差舍入到激活 dtype；
- 三种求值顺序：逐 token 递归（`scan`，与原生 selective scan 的顺序一致，goldens 用它）、分块 SSD（`chunked`）和 prefill 加逐 token `step`（带 conv 和 SSM cache 的解码形式）。
- 给出 position（像 `compute_logprobs` 那样，在第一个 token 之后出现的 0 开始一个新文档）时，三条路径都在文档开头复位 SSM 状态和 conv 窗口。原生 `mamba_ssm_scan` 不复位，所以 shard 每行只放一个文档；文档感知的形式是将来复位 kernel 的参照。
- `dtype=torch.float64` 全程用 fp64；bf16 模式在原生以 bf16 存储张量的每个位置舍入，用来估计 bf16 与 fp32 的差距。

CPU 测试另外检查参照实现的假设与编译出的 IR 一致（每层的 op 序列、scan 的 dt clamp、gated norm 的分组与顺序、所有 norm 的 eps），并检查参照读取的张量名与初始化写出的完全相同（A0 与 A0-split、tied 与 untied）。

### G1 与 G2 的参照套件

原生的 G1、G2 拿原生训练器与这个参照比，所以参照本身先要被证明是因果的、各条路径一致的。[prototypes/g1_suite.py](../prototypes/g1_suite.py) 与 [prototypes/g2_suite.py](../prototypes/g2_suite.py) 在一个两层的小模型上检查（CPU 测试 `test_reference_gates.py`，全套单线程约 7 CPU 秒）：

| 检查 | 判据 |
|---|---|
| G1.1 路径 | T ∈ {1, 2, 7, c−1, c, c+1, 2c+1, 512}（c 为 chunk 长度，这里 16），batch 2：`chunked`、`scan`、prefill+`step` 三者的 logits 之差 fp64 不超过 10⁻¹⁰·max(1, \|logits\|∞)，fp32 不超过 10⁻⁵·max(1, \|logits\|∞) |
| G1.2 前缀 | forward(x[:j]) 与 forward(x)[:j] 在 G1.1 的界内一致（更短的输入可能走 conv1d 的另一种算法，所以不要求逐位） |
| G1.3 交接 | 先 prefill k 个 token 再逐个 step，等于完整前向，k ∈ {1, c−1, c, c+1, T−1}，两种 prefill 顺序 |
| G1.4 未来前向 | 改动 p 及其后的 token，logits[:, :p] 逐位不变 |
| G1.5 未来梯度 | 对 s > t，∂logits_t/∂embedding_s 恰为 0；在 chunk 64、每 token 对数衰减约 −160 的输入下梯度也有限 |
| G1.6 文档 | 两个文档打包在一行（position 重新从 0 开始）与分开运行一致；不给 position（不复位，与原生相同）时，第二个文档的差异大于 10⁻³ |
| G1.7 padding | 右 padding 不改变 logits；prefill 拒绝左 padding |
| G1.8 变异 | 五种植入的错误各让至少一项检查失败：对称的 conv padding、分块 SSD 的输出错位一格、conv 交接差一、漏掉文档复位、先 exp 后掩码的 segsum（前向几乎不变，梯度变成 NaN） |
| G2.1 梯度 | 一次前向与反向，每个参数都有梯度且有限 |
| G2.2 初始 loss | 随机 token 的交叉熵在 ln V + var(logits)/2 ± 0.1 之内，logits 的标准差小于 1 |
| G2.3 过拟合 | 一个 16 个 token id 的固定 batch，其 H(next \| current) ≥ 1.5：完整模型 200 步内 loss < 0.05，mixer 输出置零的消融模型 loss ≥ 0.9 × 下限。在 16384 的词表上随机 batch 中几乎每个 token 的后继都唯一，消融模型也能拟合，过拟合测试就证明不了序列混合有效 |

G2.4（保存后加载进另一个 seed 构建的模型，logits 逐位相等）与 G2.5（resume 与不中断的运行逐位相等）测的是原生 checkpoint，不在 CPU 参照上做：P0 的训练循环、优化器状态、调度器与数据游标都在原生训练器里。它们在 GPU 上的对应检查见 [运行手册的 G2](runbook.md#g2-数学与梯度)；数据游标的恢复由 `fork/tlc/tests/gpu/test_data_native.py` 逐位检查。

### G1(b) 的 goldens

生成 G1(b) 的 goldens（CPU 即可，A0、batch 2、512 token 在单线程上约 6 秒）：

```bash
python -I -B fork/tlc/prototypes/ref_mamba2.py MODEL_DIR OUT.npz --batch 2 --seq-len 512 --seed 0
```

`.npz` 中的数组：`input_ids`、`targets`（int32，[batch, seq_len]，正是 `compute_logprobs` 的两个参数；`targets` 是下一个 token，最后一列为 −100）；`logprobs`（fp32 参照，逐 token 递归）；`logprobs_bf16`（bf16 模式）；`tolerance` = 2 × max \|`logprobs_bf16` − `logprobs`\|；`meta`（JSON：`model.safetensors` 的 sha256、seed、dt clamp 区间、求值顺序）。G1(b) 的判据：原生 `compute_logprobs(input_ids, targets, use_lora=False)` 与 `logprobs` 的最大绝对差不超过 `tolerance`，且 `meta` 中的 sha256 与所测目录的权重文件一致。bf16 对拍的 mean-KL 判据要完整的 logits，而 `compute_logprobs` 只返回目标的 log-probability，所以 P0 不做。

## 上游接口的 pin

P0 直接依赖下列上游接口；CPU 测试（`test_pins.py`、`test_checkpoint.py`）钉住它们，上游改动时测试失败，需检查 TLC 中的用法后再更新 pin。pin 只检查 TLC 依赖的部分，锚定在代码上而不是注释上；需要原生扩展的模块按源码解析。

- `Mamba2Mixer`、`GenericMLP`、`Qwen3Block`、`HybridBlockStack` 接受 TLC 传入的构造参数，以及 TLC 依赖的默认值：`Mamba2Mixer` 不 clamp dt；`GenericMLP` 默认是融合的 SwiGLU；`Qwen3Block` 默认开 QK-norm、不带 QKV bias。A0-split 用到的 tracer 成员（`nn._current_tracer`、`Tracer.push_prefix`/`pop_prefix`/`hf_mappings`）。块的名字映射直接 import 上游的 `DENSE_BLOCK_NAME_REMAP` 与 `NEMOTRON_MAMBA_BLOCK_REMAP`，shard 的文件头用上游的 `read_token_shard_header` 读：它们改名时 import 就失败。
- 原生 op 名表能被解析，且多于 100 个名字。
- `SurogateTrainerWrapper(config, train_files, eval_files=None)` 与无参数的 `train()`；`SFTConfig` 要由调用方执行 `__post_init__()`（上游 `TokenizeDatasets` 也是这样做的）；TLC 设置的每个上游键（运行配置、`surogate_tlc.native` 中的固定值与采样用的 LoRA 设置）都确实被 `SFTConfig` 读取（拼错或已废弃的键会被忽略，悄悄用上默认值）。
- 训练器在环境变量 `SUROGATE_REGRESSION_ARTIFACT` 给出路径时写出 regression artifact（loss 曲线、每步时间的分位数、峰值显存、block schema 统计）；原生训练日志是 JSON 数组，验证行带 `step` 与 `loss`。运行记录的摘要读取这些字段。
- 原生 OOM 的形式：分配器抛出消息为 "Cuda OOM when allocating tensor ..." 的 `std::runtime_error`，activation 栈抛出 `std::bad_alloc`。TLC 的代码不调用 `train_with_oom_recovery`。
- `compute_logprobs(input_ids, targets, use_lora, position_ids, temperatures)` 的参数顺序；`next_token_logits(input_ids, last_positions)` 只在带常驻 LoRA 权重的 trainer 上运行，且原生 LoRA 的 B 初始化为零。
- BIN.TOK v3 的文件布局与上游 `TokenizedDataFileWriter` 逐字节相同（`test_shards.py` 执行上游类的源码比较）。
- 原生 tokenizer：`tlc` 不在按架构写死正则的名单中，因此从 `tokenizer.json` 读取 split 正则（没有时用 GPT-2 的）；手写 splitter 覆盖 `qwen2` 与 GPT-2 的正则，不覆盖 `python`。
- 原生 DataLoader 读取 non-overlapping 文件的规则：块数为 token 数 ÷ `seq_len`，目标是下一个 token，position 块紧跟 token 块，mask 位为 0 的目标与每行最后一个目标置为 −100。
- 原生 checkpoint：step 目录命名、`checkpoint.json` 在全部分片和导出之后写入、训练结束时导出复制的文件清单。

后续阶段才用到、P0 还没有钉住的私有耦合，在引入它们的阶段加 pin：dense LoRA 的字段名、`decode_cache_specs` 中没有 `MambaSsmScan`、`SharedModelServer._generate` 及其构造与模块全局名、`DecodeScheduler` 的创建条件（P5 的生成与 rollout）；`DataLoader.enable_kd`/`load_batch` 的签名、`_load_training_microbatch` 的 sidecar 透传（P1 的标签 sidecar）。
