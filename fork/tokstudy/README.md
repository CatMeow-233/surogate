# fork/tokstudy：分词器研究

本包做 TLC 的分词器决定 O-10 与双语可组合专家的分词器决定 B-O11 的“效果对比”：在同一个 PyTorch 后端上，以相同的训练 FLOPs 训练子词、平坦字节与 H-Net（学习式动态分块，即“另類神經網絡分詞”）各臂，按语言、按领域比较 bits-per-byte，并按预注册的阈值给出判定。owner 的决定（原文）：「分詞器得對比效果要不然就搞另類神經網絡分詞 語料用minicpm的ultradata那些的」。

分工：子词与子词之间的 TLC 比较仍在原生训练器上做（TLC 自己的 TOK-AB）；本包的 TLC 表只放 N0（原生领先臂的 PyTorch 孪生）、N1（H-Net）与 N2（平坦字节），永不与原生的行同表。双语的全部臂 B0–B5 都在本包中经真实的双语 trunk 训练。生产用的 tokenizer 构建器留在各自的项目里；本包只按路径与 sha256 读取它们产出的 `tokenizer.json`。语料由 [fork/corpora](../corpora/README.md) 备好，经各项目自己的去污染与划分后才进入本包。

## 状态

本机没有 GPU：凡 RTX 5090 的数字都是 [估计] 或 TBD，以 G0.5-ts 实测为准。

| 步骤 | 内容 | 命令 | 状态 |
|---|---|---|---|
| S0 | 包骨架、`fork/dev test tokstudy`、CI、导入边界测试 | — | 已落地 |
| S6、S7（构建器部分） | 项目内的生产 tokenizer 构建器：TLC 的 freqtrim 与 `python-digit`（[fork/tlc/docs/data.md 的“freqtrim”](../tlc/docs/data.md#freqtrim)），双语的 `tokenizer.pretokenizer`（[fork/bilingual/docs/spec.md](../bilingual/docs/spec.md)） | 各项目的 `tokenizer` 命令 | 已落地；双语的 `RoutedLM.forward_hidden`/`logits` 随 S11–S12 落地；TLC 的按字节 BPB 与双语的 `length_unit` 未落地 |
| S8 | 臂登记与编解码器 | `arms` | 已落地 |
| S9 | 文档、分帧、训练流 | — | 只落地文档类型 `EvalDoc` 与样本文件的读取；按项目分帧的 `tlc_documents`、`bilingual_documents` 与训练流未落地 |
| S10 | 内在指标与教师对齐（只作 CPU 初筛，不作判定依据） | `intrinsic`、`align` | 已落地 |
| S11–S12 | SSD/EMA/卷积 kernel、Mamba-2、A0 孪生（TW-1）、H-Net、LM 包装、双语 trunk 适配器、由臂构建模型 | — | CPU 上已落地（见下文“模型与内核”）；`triton` 路线与 G0-ts 未在 GPU 上运行 |
| S13 | FLOPs 与 H-Net 配置匹配 | `flops` | 未落地 |
| S14 | 训练循环（按 FLOP 进度的 WSD 与衰减分支） | `train`、`branch` | 未落地 |
| S15 | 评测（公共字节前缀上的 BPB）与探针 | `eval`、`probe` | 未落地 |
| S16 | 统计、预注册与判定 | `verdict` | 未落地 |
| S17 | GPU：G0-ts、G0.5-ts、kernel 一致性、孪生门 | `bench` | 未运行 |

未落地的命令以退出码 2 报出所属步骤（`python -m surogate_tokstudy` 列出全部命令）。

工作流的 S2（“tokenizer 构建器与内在指标”）对应实现规格的 S6、S7 中的 tokenizer 构建器部分，以及 S8、S10；S9 只落地了 S10 需要的文档类型。工作流的 S3（“字节级动态分块模型与子词 LM，同一后端，CPU 验证”）对应实现规格的 S11、S12，以及 S7 中双语的 `forward_hidden`/`logits`；臂到模型的构建（`models.build`）是规格未单列、训练步骤（S14）要用的入口。

## 测量契约

- **字节完全相同。** 一张表的所有臂对同一段原始 UTF-8 字节打分：编解码器给出每个 token 的字节，`b"".join(token_bytes(i) for i in ids) == text.encode()`；任何臂都不带 normalizer（带 normalizer 的 `tokenizer.json` 被拒绝，参照或教师 tokenizer 以 `--drop-normalizer` 去掉它并记录）。孤立的代理码元不是 Unicode 标量值，没有 UTF-8 编码，被拒绝。
- **按段编码。** 文档是若干段，每段是一个特殊 token 或一段文本；各段分别编码，token 永不跨过特殊 token 合并。
- **特殊 id。** TLC 与双语各自的 id 0–4（`codecs.PROJECT_SPECIALS`，测试与两个项目的表比对）；字节编解码器为 5 个特殊 id 加 256 个字节 id（id = 5 + 字节），词表 261，`pad_to` 可补齐。
- **内在指标不作判定依据**：它们只用于淘汰明显被支配的臂和解释结果；判定靠等 FLOPs 的训练运行（S14–S16）。

## 模型与内核

所有臂在同一个 PyTorch 后端上：子词臂、平坦字节臂与 H-Net 臂共用 Mamba-2 层和两个主网络，`models.build.build_model(arm, vocab_size=…, init_seed=…)` 是从臂到模型的唯一入口。

| 模块 | 内容 |
|---|---|
| `kernels` | `ssd`：Mamba-2 的分块 SSD（Dao & Gu 2024 的 ssd_minimal_discrete，带分组与初始状态），`ssd_reference` 是逐位置的递推；`ema_scan`：H-Net 平滑模块的 EMA，分块形式不取对数（P = 1 时衰减为 0，对数没有梯度），段内与段间的衰减乘积都由带掩码的 cumprod 得到，`ema_scan_reference` 是循环；`causal_conv`：深度可分离因果卷积。低精度输入一律在 fp32 中计算（fp64 输入用 fp64），autocast 关闭，返回输入的 dtype |
| `models.mamba2` | Mamba-2 mixer（与 `fork/tlc/prototypes/ref_mamba2.py` 的运算一致）、SwiGLU、pre-norm 层、以 RMSNorm 结尾的层栈；初始化沿用 TLC 的方案（分布相同，抽样顺序不同） |
| `models.twin_a0` | `A0Twin`：TLC 的 A0（12 层，宽 384，SwiGLU 1024，tied），几何用 TLC config.json 的键；`from_tlc_model_dir` 读 TLC 模型目录（含 A0-split 的命名）；`vocab_size=None` 时没有嵌入，作 H-Net 的主网络 |
| `models.trunk_bilingual` | `BilingualTrunk`：双语种子模型（一个名为 `seed` 的专家，plan 为 None），经公开的 `RoutedLM.forward_hidden`/`logits` 使用；子词臂的模型就是生产模型本身 |
| `models.hnet` | 一级 H-Net：字节嵌入 → Mamba-2 编码器 → 路由（相邻余弦，p₁ = 1，p ≥ 0.5 为边界，强制边界处 p = 1）→ 下采样 → 加宽 → 主网络 → 取前 d₀ 维 → EMA 平滑、因果重复 × STE(c) + 残差投影 → Mamba-2 解码器 → tied 字节头；ratio loss（α = 0.03）；按阶段返回参数以便各阶段的 LR 倍率 |
| `models.lm` | `SubwordLM`、`ByteLM`：主网络自己的嵌入、层与 tied head |
| `models.build` | 臂的 `model` 表：`profile`、`overrides`（主网络的几何）与 `hnet`（只用于 H-Net 臂，`HNetConfig` 的字段）；H-Net 外层的初始化种子由 `init_seed` 派生，与主网络不共用随机流 |

H-Net 的公式按论文（2507.07955 §2.1–2.3，式 (1)–(10)）；论文没写的细节按作者代码（goombalab/hnet 的 `modules/dc.py`、`models/hnet.py`，本次会话读过）：W_q、W_k 初始化为单位阵；平滑前把 P 截到 [1e-4, 1 − 1e-4]；残差投影初始化为 0、在 fp32 中计算；加宽向量初始化为 0、属于主网络阶段。与作者代码不同的三处：残差投影不带 bias（代码中有，按默认分布初始化；论文写的是“接近 0”的 Linear）；边界取 p ≥ 0.5（论文的写法；代码的 argmax 使 p = 0.5 不成为边界）；STE 的前向值恰为 1（与代码相同用 autograd 函数；`c + (1 − c).detach()` 可能舍入到 1 − ulp）。行按右侧补齐，`valid` 标出真实位置：补齐位置永不成为边界，不计入 F、G。

CPU 上的验证（`tests/cpu/test_kernels.py`、`tests/cpu/test_models.py`，fp64，单线程，确定性算法）：

- 分块 SSD（chunk 4 与 128，长度 1–256，分组，初始状态）对递推 ≤ 1e-10；分块 EMA（含 P = 0、P = 1）对循环 ≤ 1e-12；`ssd`、`ema_scan`、`causal_conv` 通过 `gradcheck`；把 EMA 映射到 SSD（`triton` 路线用的映射）经 torch 后端验证 ≤ 1e-10。
- **TW-1**：由 TLC 自己的 `write_model_dir` 与 `init`（在独立解释器中运行）写出的两层 A0 与 A0-split 目录，norm 与 D 再乘随机因子，`A0Twin.from_tlc_model_dir` 的 logits 与 `ref_mamba2.Reference` 的相对差 ≤ 1e-9（batch 2 × 17；本机实测约 1e-15）；孪生读写的权重名与 TLC 的完全相同。A0 几何下非嵌入参数恰为 25,490,736（12 × 2,124,196 + 384）。
- 对两张表的子词、字节、H-Net 六种模型：改动位置 t 之后的 id 不改变 ≤ t 的 logits（≤ 1e-12）与 H-Net 的边界（逐位，强制边界开与关）；一行单独计算与和其他长度、其他 chunk 数的行同批计算结果相同；bf16 autocast 下前向与反向的 loss、梯度有限，路由与 ratio loss 保持 fp32。
- 双语 trunk 适配器的子词模型与同 profile、同种子的 `RoutedLM.forward` 逐位相同；`RoutedLM` 的 `logits(forward_hidden(emb(t))) == forward(t)` 在种子模型与带 plan 的双专家模型上逐位成立（`fork/bilingual/tests/cpu/test_model.py`）。
- 一个很小的 H-Net 在合成词序列上训练 50 步（CPU，约 1 秒）：字节 loss 下降超过 1 nat，边界比例 F 比起点更接近 1/N。这只证明能学，不是任何收敛结论。
- 对实现做过的变异（上采样错位、路由读到未来、去掉 b₁、补齐成为边界、c 取反、SSD 掩码、EMA 段间状态、卷积偏移、去掉 D、权重名错配、p₁ ≠ 1）都被测试查出。

**`triton` 路线 [已核实源码，未在 GPU 上运行]**：`kernels.ssd(backend="triton")` 调用 `mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined`（签名读自 mamba-ssm 2.3.2.post1 的 sdist），导入失败或张量不在 CUDA 上时报错，从不悄悄回落。但该 sdist 的 `mamba_ssm/__init__.py` 无条件导入 `selective_scan_interface`，后者无条件 `import selective_scan_cuda`：用 `MAMBA_SKIP_CUDA_BUILD=TRUE` 装出的包导入这条路径时会失败。实现规格 §10 第 3 步的假设因此不成立；G0-ts 的选择是完整编译 mamba-ssm 的 CUDA 扩展（需要 sm_120 的 nvcc）或用 `torch` 后端并记录。`torch` 后端在 5090 上的吞吐 TBD。

## 命令行

在 `fork/.venv-tokstudy` 中运行。每个 tokenizer 按 `名字=路径@sha256` 给出，sha256 不符即拒绝；所有拒绝都以退出码 1 结束并给出原因。

```bash
python -m surogate_tokstudy arms arms/tlc.toml
python -m surogate_tokstudy intrinsic --arms arms/tlc.toml --docs py=PY.jsonl --docs en=EN.jsonl --out intrinsic.json
python -m surogate_tokstudy intrinsic --tokenizer qwen=DIR/tokenizer.json@SHA256 --drop-normalizer --bytes bytes \
    --docs en=rows_en.json --docs zh=rows_zh.json --docs py=rows_py.json --out intrinsic.json
python -m surogate_tokstudy align --teacher QWEN/tokenizer.json@SHA256 --arms arms/tlc.toml --docs py=PY.jsonl --out align.json
```

- **`arms`**：读取臂登记 `tokstudy-arms/1`（格式见 `surogate_tokstudy.arms` 的模块说明）：一个文件只放一张表（`tlc-N` 或 `bilingual-B`），表决定项目的特殊 id 与主网络（`twin_a0`、`bilingual_trunk`）；逐个核对 tokenizer 的 sha256 与特殊 id，拒绝重复的 id、第二张表、未知的键与种类、字节臂或 H-Net 臂带 tokenizer。`arm_sha256` 取臂的规范 JSON，其中 tokenizer 只记 sha256，所以移动文件不改变臂，改变文件内容才改变。真实的 `arms/tlc.toml` 与 `arms/bilingual.toml` 要等各臂的 tokenizer 在 owner 主机上建好、有了 sha256 之后再写，现在还没有。
- **`--docs 领域=路径`**：`.jsonl`（每行一个 JSON 对象，正文在 `text`；fork/corpora 的导出分片与 TLC 的语料 JSONL 都是这种）或 datasets-server 的 rows 响应 `.json`（正文在 `content`）；`--field` 改字段名，`--limit` 限制每个文件的文档数。正文原样使用，不做任何规范化或换行转换。按项目 spec 与划分读取 train 划分（S9）之前，初筛用这两种样本文件。
- **`intrinsic`**（`tokstudy-intrinsic/1`）：每个臂、每个领域的字节、字符、token 数与比值；汉字数（Unicode Script=Han）与每 token 汉字数；词数（`str.split()`，只在 `--word-domain`，默认 `en`）与每 token 词数；Rényi 效率 H₂.₅(p)/ln|V|；出现过的 token 数与词表利用率；单独看不是合法 UTF-8 的 token 数；数字串（Unicode 十进制数字的极大串）、其字符数与覆盖它们的 token 数；至少有 512（2048）个 token 的文档中前 512（2048）个 token 覆盖的字节数的上中位数；在 `--python-domain`（默认 `py`）上，能被 `ast.parse` 与 stdlib `tokenize` 接受、且词素位置自检无误的段的跨词素率（与两个及以上词素重叠的 token 占比；词素为 NAME、OP、NUMBER、STRING、COMMENT 与 f-string 各部分，空白、换行与缩进不算，同实验笔记的词素口径）与每个标识符的片数。记录所用的 Python 版本，因为 3.12 起 f-string 的切分方式不同。所有计数都是精确整数，比值由它们算出。H-Net 臂没有 token 级的内在指标，列在 `skipped_hnet_arms` 中。
- **`align`**（`tokstudy-align/1`）：学生 tokenizer 对教师 `tokenizer.json` 的规范切分（教师的 normalizer 去掉，并统计它本会改变的文档数），按实验笔记的对齐原型逐个学生 token 分类：`one_to_one`（与教师 token 重合）、`head`（从教师 token 的开头开始、在其内部结束，即 BPM 的 1:N 头）、`interior`（从教师 token 内部开始，即 1:N 内部）、`cross_boundary`（首字节与末字节落在不同的教师 token 中，BPM 屏蔽这些位置；一个学生 token 盖住几个完整的教师 token 也属于这一类）。另计只由空格与 tab 组成、又不是 `one_to_one` 的学生 token（非恒等全空白行）：研究建议在冻结的 tokenizer 上它占代码领域学生 token 的 0.5% 以上时默认开启全空白行屏蔽 [建议]。这些是初筛读数；权威的 BPM 检查仍是 TLC 的 G-BPM。

## 实测记录（本地，不进 CI）

2026-11-03，本机 CPU，`FORK_PY`（Python 3.12.3、tokenizers 0.23.3）：

- `intrinsic` 在三个 20 行 datasets-server 样本（Ultra-FineWeb en、zh，UltraData-Code-L2 py）上，用 Qwen/Qwen2.5-Coder-1.5B-Instruct 的 `tokenizer.json`（revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`，sha256 `c0382117…`，带 NFC，以 `--drop-normalizer` 去掉）与 openai-community/gpt2 的 `tokenizer.json`（revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`，sha256 `8414cab9…`），`--check` 打开：字节/token 为 4.147 / 4.030 / 3.780（Qwen）与 4.306 / 1.415 / 2.084（GPT-2）；字节、字符、token 数、非法 UTF-8 占比、512 与 2048 token 覆盖字节的中位数共 42 项与此前的探针 `compress_probe.json` 逐项相同 [CPU实测]。约 2 秒。
- 同一批样本上的 `align`（教师同上；NFC 在 60 篇文档中改变 0 篇）与用这 40 篇 en+py 文档、4,096 id 建的 freqtrim（`qwen` 与 `fine-a` 两种正则，用真实 Qwen 文件，`freqtrim-check` 通过）只用来确认真实文件能被读入和检查；子集在评测文档上建成、词表只有 4K，其比率不代表任何候选 [CPU实测，只作软件检查]。
- 样本行与 tokenizer 文件都只在本地使用，用后删除，没有进入仓库。

## 环境

所有 `python -m surogate_tokstudy …` 命令都在 `fork/.venv-tokstudy` 中运行（AGENTS.md 第 4 条）：先装 torch（GPU 主机装 sm_120 的 CUDA 版并在 G0-ts 记录 wheel；只有 CPU 的机器装 CPU 版），再装依赖与三个 fork 包：

```bash
python3.12 -m venv fork/.venv-tokstudy
fork/.venv-tokstudy/bin/pip install -r fork/tokstudy/requirements-gpu.txt
fork/.venv-tokstudy/bin/pip install -e fork/tokstudy -e fork/tlc -e fork/bilingual --no-deps
```

CPU 测试 `fork/dev test tokstudy` 用 `FORK_PY`（默认根目录 `.venv`），只从检出目录导入 `fork/tokstudy/src`、`fork/tlc/src`、`fork/bilingual/src`，不安装任何包；`conftest.py` 在任何测试导入 torch 之前隐藏 CUDA，所以在 GPU 主机上也只用 CPU。

## 导入边界

| 模块 | 可以导入 | 不能导入 |
|---|---|---|
| 全部模块 | 标准库、numpy、torch、tokenizers、safetensors | `surogate`、`surogate._fork`、`surogate_corpora` |
| `surogate_tokstudy.kernels` | 另加 `mamba_ssm`，只能在 `try/except ImportError` 中 | — |
| `surogate_tokstudy.docs` | 另加 `surogate_tlc.data.corpus`、`surogate_bilingual.data.docs` | 项目的其他模块 |
| `surogate_tokstudy.models.trunk_bilingual` | 另加 `surogate_bilingual.model.lm`、`surogate_bilingual.config` | 项目的其他模块 |
| 测试 | 另可按路径加载 `fork/tlc/prototypes/ref_mamba2.py`，并导入 `surogate_tlc.ir`、两个项目的 tokenizer 与 shard 模块；TLC 模型目录由 TLC 自己的代码在独立的解释器中写出（它的初始化要用仓库里的 Python `surogate` 包编译 IR，本进程从不导入该包） | — |

`tests/cpu/test_scaffold.py` 按 AST 检查这张表，并用埋入的违规样例确认检查本身有效；两个项目的 `test_scaffold.py` 各有一条检查确认它们从不导入本包或 `surogate_corpora`。
