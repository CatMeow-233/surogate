# fork/tokstudy：分词器研究

本包做 TLC 的分词器决定 O-10 与双语可组合专家的分词器决定 B-O11 的“效果对比”：在同一个 PyTorch 后端上，以相同的训练 FLOPs 训练子词、平坦字节与 H-Net（学习式动态分块，即“另類神經網絡分詞”）各臂，按语言、按领域比较 bits-per-byte，并按预注册的阈值给出判定。owner 的决定（原文）：「分詞器得對比效果要不然就搞另類神經網絡分詞 語料用minicpm的ultradata那些的」。

分工：子词与子词之间的 TLC 比较仍在原生训练器上做（TLC 自己的 TOK-AB）；本包的 TLC 表只放 N0（原生领先臂的 PyTorch 孪生）、N1（H-Net）与 N2（平坦字节），永不与原生的行同表。双语的全部臂 B0–B5 都在本包中经真实的双语 trunk 训练。生产用的 tokenizer 构建器留在各自的项目里；本包只按路径与 sha256 读取它们产出的 `tokenizer.json`。语料由 [fork/corpora](../corpora/README.md) 备好，经各项目自己的去污染与划分后才进入本包。

## 状态

本机没有 GPU：凡 RTX 5090 的数字都是 [估计] 或 TBD，以 G0.5-ts 实测为准。

| 步骤 | 内容 | 命令 | 状态 |
|---|---|---|---|
| S0 | 包骨架、`fork/dev test tokstudy`、CI、导入边界测试 | — | 已落地 |
| S6、S7 | 项目内的改动：TLC 的 freqtrim 与 `python-digit`（[fork/tlc/docs/data.md 的“freqtrim”](../tlc/docs/data.md#freqtrim)）、原生的按字节 BPB 与 `eval --dump-nll`（[fork/tlc/docs/training.md 的“评测”](../tlc/docs/training.md#评测)）；双语的 `tokenizer.pretokenizer`、`RoutedLM.forward_hidden`/`logits` 与 `eval.length_unit`（[fork/bilingual/docs/spec.md](../bilingual/docs/spec.md)） | 各项目的 `tokenizer`、`eval`、`eval-tasks` 命令 | 已落地 |
| S8 | 臂登记与编解码器 | `arms` | 已落地；提交进仓库的两张表的登记 [arms/tlc.toml](arms/tlc.toml) 与 [arms/bilingual.toml](arms/bilingual.toml) 等 tokenizer 建好后 `--pin`（见下文“tokenizer 交接”） |
| S9 | 文档、分帧、训练流 | `stream` | 已落地（见下文“文档与训练流”）；TLC 的 `force_lexeme_starts` 消融未落地 |
| S10 | 内在指标与教师对齐（只作 CPU 初筛，不作判定依据） | `intrinsic`、`align` | 已落地 |
| S11–S12 | SSD/EMA/卷积 kernel、Mamba-2、A0 孪生（TW-1）、H-Net、LM 包装、双语 trunk 适配器、由臂构建模型 | — | CPU 上已落地（见下文“模型与内核”）；`triton` 路线与 G0-ts 未在 GPU 上运行 |
| S13 | FLOPs 与 H-Net 配置匹配 | `flops` | 已落地（见下文“FLOPs”） |
| S14 | 训练循环（按 FLOP 进度的 WSD 与衰减分支） | `train`、`branch` | CPU 上已落地（见下文“训练”）；GPU 上未运行 |
| S15 | 评测（公共字节前缀上的 BPB）与探针 | `eval`、`probe` | BPB 部分已落地（见下文“评测”）；探针 `probe` 与分段带状态打分未落地 |
| S16 | 统计、预注册与判定 | `verdict` | 运行登记、预注册文件的读取与锁定已落地（见下文“运行登记与预注册”）；统计与判定 `verdict` 未落地 |
| S17 | GPU：G0-ts、G0.5-ts、kernel 一致性、孪生门 | `bench` | 未运行；步骤见 [运行手册](docs/runbook.md) |
| 接线（工作流 S5） | 导出 → 项目语料 spec（[fork/corpora](../corpora/README.md) 的 `compose`）、tokenizer 交接（`arms --pin`）、运行手册与实测记录、各项目文档中的 owner 决定 | `arms --pin`、`arms --tlc/--bilingual` | 已落地；备料与 GPU 步骤未运行 |

未落地的命令以退出码 2 报出所属步骤（`python -m surogate_tokstudy` 列出全部命令）。

工作流的 S2（“tokenizer 构建器与内在指标”）对应实现规格的 S6、S7 中的 tokenizer 构建器部分，以及 S8、S10；S9 当时只落地了 S10 需要的文档类型。工作流的 S3（“字节级动态分块模型与子词 LM，同一后端，CPU 验证”）对应实现规格的 S11、S12，以及 S7 中双语的 `forward_hidden`/`logits`；臂到模型的构建（`models.build`）是规格未单列、训练步骤（S14）要用的入口。工作流的 S4（“pilot 训练器、BPB 评测、运行登记”）对应实现规格 S9 的其余部分（按项目分帧、截断、训练流）、S13、S14、S15 中的 BPB 部分，以及 S16 中的运行登记与预注册的读取、锁定；S15 的探针、S16 的统计与判定、S6 中 TLC 原生的按字节 BPB、S7 中双语的 `eval.length_unit` 不在其中。

工作流的 S5（“按项目接线：导出到两个项目的语料格式与 tokenizer 交接；文档、运行手册与各项目文档中的 owner 决定”）对应实现规格 S6 中 TLC 原生的按字节 BPB（`surogate_tlc.eval.loss`）、S7 中双语的 `eval.length_unit` 与研究期间的去污染规则（文档），§1.3 的 tokenizer 交接（`handoff.py`、两张表的登记），§10 的运行手册（[docs/runbook.md](docs/runbook.md)）与 [docs/measurements.md](docs/measurements.md)；另加规格没有的 `surogate_corpora compose`。S15 的探针、S16 的统计与判定仍未落地。

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
python -m surogate_tokstudy arms arms/tlc.toml --pin --tlc CORPUS/spec.toml --record handoff.json
python -m surogate_tokstudy arms arms/bilingual.toml --bilingual SPLITS/manifest.json
python -m surogate_tokstudy intrinsic --arms arms/tlc.toml --docs py=PY.jsonl --docs en=EN.jsonl --out intrinsic.json
python -m surogate_tokstudy intrinsic --tokenizer qwen=DIR/tokenizer.json@SHA256 --drop-normalizer --bytes bytes \
    --docs en=rows_en.json --docs zh=rows_zh.json --docs py=rows_py.json --out intrinsic.json
python -m surogate_tokstudy align --teacher QWEN/tokenizer.json@SHA256 --arms arms/tlc.toml --docs py=PY.jsonl --out align.json
python -m surogate_tokstudy stream --prereg prereg/tlc.toml --tlc CORPUS/spec.toml --out STREAM
python -m surogate_tokstudy stream --prereg prereg/bilingual.toml --bilingual SPLITS/manifest.json \
    --measure arms/bilingual.toml:B0 --out STREAM
python -m surogate_tokstudy flops arms/tlc.toml --k-train 2048 --bytes-per-token N0=3.8 --match N1 --reference N0
python -m surogate_tokstudy train --arms arms/tlc.toml --arm N1 --seed 0 --stream STREAM --prereg prereg/tlc.toml \
    --runs RUNS --lr 1e-3 --lambda-outer 2
python -m surogate_tokstudy branch RUNS/tlc-N/N1-s0-<run id> --at 0.5
python -m surogate_tokstudy eval prefix --arms arms/tlc.toml --prereg prereg/tlc.toml --tlc CORPUS/spec.toml --out EVALSET
python -m surogate_tokstudy eval score RUNS/tlc-N/N1-s0-<run id> --point 0.5 --evalset EVALSET --out N1-s0-E0.5.nll
```

- **`arms`**：读取臂登记 `tokstudy-arms/1`（格式见 `surogate_tokstudy.arms` 的模块说明）：一个文件只放一张表（`tlc-N` 或 `bilingual-B`），表决定项目的特殊 id 与主网络（`twin_a0`、`bilingual_trunk`）；逐个核对 tokenizer 的 sha256 与特殊 id，拒绝重复的 id、第二张表、未知的键与种类、字节臂或 H-Net 臂带 tokenizer。`arm_sha256` 取臂的规范 JSON，其中 tokenizer 只记 sha256，所以移动文件不改变臂，改变文件内容才改变。两张表的登记已提交（[arms/tlc.toml](arms/tlc.toml)、[arms/bilingual.toml](arms/bilingual.toml)），子词臂的 sha256 等 tokenizer 在 owner 主机上建好后由 `arms --pin` 写入（见下文“tokenizer 交接”）。
- **`--docs 领域=路径`**：`.jsonl`（每行一个 JSON 对象，正文在 `text`；fork/corpora 的导出分片与 TLC 的语料 JSONL 都是这种）或 datasets-server 的 rows 响应 `.json`（正文在 `content`）；`--field` 改字段名，`--limit` 限制每个文件的文档数。正文原样使用，不做任何规范化或换行转换。按项目 spec 与划分读取 train 划分（S9）之前，初筛用这两种样本文件。
- **`intrinsic`**（`tokstudy-intrinsic/1`）：每个臂、每个领域的字节、字符、token 数与比值；汉字数（Unicode Script=Han）与每 token 汉字数；词数（`str.split()`，只在 `--word-domain`，默认 `en`）与每 token 词数；Rényi 效率 H₂.₅(p)/ln|V|；出现过的 token 数与词表利用率；单独看不是合法 UTF-8 的 token 数；数字串（Unicode 十进制数字的极大串）、其字符数与覆盖它们的 token 数；至少有 512（2048）个 token 的文档中前 512（2048）个 token 覆盖的字节数的上中位数；在 `--python-domain`（默认 `py`）上，能被 `ast.parse` 与 stdlib `tokenize` 接受、且词素位置自检无误的段的跨词素率（与两个及以上词素重叠的 token 占比；词素为 NAME、OP、NUMBER、STRING、COMMENT 与 f-string 各部分，空白、换行与缩进不算，同实验笔记的词素口径）与每个标识符的片数。记录所用的 Python 版本，因为 3.12 起 f-string 的切分方式不同。所有计数都是精确整数，比值由它们算出。H-Net 臂没有 token 级的内在指标，列在 `skipped_hnet_arms` 中。
- **`align`**（`tokstudy-align/1`）：学生 tokenizer 对教师 `tokenizer.json` 的规范切分（教师的 normalizer 去掉，并统计它本会改变的文档数），按实验笔记的对齐原型逐个学生 token 分类：`one_to_one`（与教师 token 重合）、`head`（从教师 token 的开头开始、在其内部结束，即 BPM 的 1:N 头）、`interior`（从教师 token 内部开始，即 1:N 内部）、`cross_boundary`（首字节与末字节落在不同的教师 token 中，BPM 屏蔽这些位置；一个学生 token 盖住几个完整的教师 token 也属于这一类）。另计只由空格与 tab 组成、又不是 `one_to_one` 的学生 token（非恒等全空白行）：研究建议在冻结的 tokenizer 上它占代码领域学生 token 的 0.5% 以上时默认开启全空白行屏蔽 [建议]。这些是初筛读数；权威的 BPM 检查仍是 TLC 的 G-BPM。

## tokenizer 交接

生产用的 tokenizer 都由项目自己建、自己拥有（规格 §1.3）；本包只按路径与 sha256 读取 `tokenizer.json`，并核对它训练于哪份语料（`surogate_tokstudy.handoff`）。

- **登记。** 两张表的臂登记提交在 [arms/](arms/) 中：[arms/tlc.toml](arms/tlc.toml)（N0、N1、N2）与 [arms/bilingual.toml](arms/bilingual.toml)（B0–B5），臂、种子、阶段与模型配置和 prereg 一致（测试检查）。子词臂的 tokenizer 路径相对登记文件，指向检出目录中被 git 忽略的 `data/tokstudy/<项目>/tok/<臂>/tokenizer.json`；sha256 先写成占位符 `unpinned`，`load_arms` 遇到它就拒绝并指出 `--pin`，所以 tokenizer 建好之前任何运行都开始不了。H-Net 臂的 `[arm.model.hnet]` 是按假设的字节/token 匹配的 [估计]，第一次运行之前用实测值重新匹配（[运行手册第 4 步](docs/runbook.md#4-交接项目的-tokenizer-进入臂登记)）。
- **核对与写入**（`arms REGISTRY --tlc SPEC | --bilingual MANIFEST [--pin] [--record FILE]`）。对每个子词臂读项目自己写在 `tokenizer.json` 旁边的记录：TLC 的 `tokenizer_manifest.json`（`train` 与 `freqtrim` 都写）必须记着该文件的 sha256、给出的 corpus spec 的 sha256 与去污染记录、训练于 train 划分，且没有 normalizer；双语的 `tokenizer.meta.json`（`bilingual-tokenizer/1`）必须记着该文件的 sha256、给出的派生 manifest 的 `manifest_sha256`、训练于 `seed_train`、normalization 为 `none`。表的项目决定要给哪一个参数，给错的、缺记录的、记录对不上的都拒绝。不带 `--pin` 时登记中的 sha256 必须已经等于文件的；带 `--pin` 时只改写每个子词臂那一行 `tokenizer = {path = "...", sha256 = "..."}` 中的 sha256（注释与排版不变；不是这种一行写法就拒绝，文件不动），再按 `load_arms` 重新核对。`--record` 写 `tokstudy-handoff/1`：登记的 sha256、语料身份，以及每个子词臂的 tokenizer sha256 与项目的配方（种类、预分词或正则、词表大小，freqtrim 另记教师的仓库、revision 与文件 sha256）。
- **反方向（冻结）。** 本包从不写进项目。判定选出的臂的 tokenizer 本来就是项目建的文件，冻结是项目自己的步骤；`handoff.json` 记下冻结的文件必须有的 sha256（[运行手册第 7 步](docs/runbook.md#7-判定与冻结)）。
- **原生的表。** TLC 子词之间的比较在原生训练器上做；`python -m surogate_tlc eval … --dump-nll` 写同一种 `tokstudy-nll/1`（后端 `native`、表 `tlc-T`），字段与本包的 `DocNLL` 相同（测试按文本读 TLC 的源码比对，两个包互不导入），所以以后的统计（S16）可以读它，但永不与 `tlc-N` 同表。

## 文档与训练流

- **按项目分帧**（`docs.tlc_documents`、`docs.bilingual_documents`）：TLC 由 `surogate_tlc.data.corpus` 读 corpus spec（含去污染报告）并按它的划分规则取 `train`/`val`；代码文档为 `<|bos|> <|code_start|> 正文 <|code_end|> <|eos|>`，文本文档为 `<|bos|> 正文 <|eos|>`，其中的 fence 行换成 `<|code_start|>`/`<|code_end|>`（与 `surogate_tlc.data.shards` 相同，正文是 TLC 规范化换行后的文本），领域 `py`/`en`。双语读 `split materialize` 写出的派生 manifest，逐个文件核对 sha256，用 `surogate_bilingual.data.docs.iter_file_docs` 读出，分帧为一篇 shard 文档：`<eos>`（上一篇的结束）+ 各段（标签段为特殊 token，文本段带语言）+ `<eos>`；领域为文档唯一的语言，混语文档为 `mixed`；`loss = false` 的段被拒绝。CPU 测试在独立解释器中用 TLC 自己的代码训练 tokenizer、写 shard，确认本包的编码与 `served_rows()` 的每一行逐 id 相同、val 的键与 TLC 的划分相同；双语用项目自己的 manifest → split → materialize → tokenizer，确认与 shard 写入方 `encode_doc` 的 token 相同。
- **截断**（`docs.truncate`）：取前 k 个文本字节，在字符边界切开；文本不超过 k 字节的文档原样返回，否则保留文本偏移小于 k 的特殊 token，之后的全部去掉。
- **训练流**（`stream`，`tokstudy-stream/1`）：每篇文档的秩键为 sha256("种子\0键")，领域内按秩升序（与输入顺序无关的确定性洗牌）；给出 `[mixture]` 时按“已取单位 / 份额”最小的领域优先交错，直到各领域达到份额 × 总量（最后一篇最多超出一篇）。单位是 UTF-8 文本字节，或参照臂的 token（prereg 的 `[stream] measure`，STUDY M9：双语按 B0 的 token 定中英平衡）。不给 `--max-units` 时由最稀缺的领域决定总量；不在混合中的领域的文档计数后丢弃。文档一次读入，按秩前缀溢写到桶文件，再按选定的顺序写成 `docs-NNNNN.jsonl`（每行一篇规范 JSON 的文档）与清单 `stream.json`（来源的文件名与 sha256、划分、种子、声明与实际的混合、有序键列表的 sha256、每个分片的 sha256）。流的身份是清单规范 JSON 的 sha256，进入每个运行 id。测试：同样的文档与种子给出逐字节相同的分片，打乱输入顺序不变，换种子改变顺序；实际混合与声明相差不超过 0.5%，且前缀上也接近。
- **行**（`stream.rows_for_arm`）：一篇文档切成至多 K_train 个位置的窗口（文本字节与特殊 token 各占一个位置，即字节编解码器的位置数），在字符边界切开；所有臂编码同一批窗口，所以各臂的行按同样的顺序覆盖同样的字节区间（测试比较子词臂与字节臂），一行一篇文档、不打包（与 TLC shard 相同）。子词行平均约 t_arm = round8(K_train / c_arm) 个 token（c_arm 为该臂在 train 流前 2,000 篇上实测的字节/token，记入运行记录）。续接窗口的第一个 token 只作输入。H-Net 行的强制边界是所有特殊 token 与双语混语文档中每个语言切换点。游标为（文档序号，窗口序号），续训从游标处精确接上。

## FLOPs

`flops.py` 只用纯 Python，模型由规格字典描述（`models.build.flop_spec`）。两种口径（每位置前向 FLOPs，训练 = 3 × 前向）：

| 口径 | 计法 | 用途 |
|---|---|---|
| `matmul`（默认，进度与 F_target 用它） | 每个 matmul 的 2 × 乘加，加序列混合项：Mamba 的 SSD 与卷积项、注意力的 2·ctx·q（ctx 为该行的实际长度）；H-Net 另加路由 W_q、W_k、残差投影与字节头；norm、激活、EMA 不计（两个项目的公式也不计） | 与 `surogate_tlc.ir.flops_per_token` 逐项相同（测试导入它）；双语种子 trunk 等于 6 × `surogate_bilingual.audit.flops.macs_per_token`（T = 1024 时每 token 652,738,560） |
| `params` | 6 × 一个位置读到的参数数（含 norm，tied 头算一次 V·d）加 3 × 序列混合项（注意力始终计入，Mamba 的 SSD 与卷积只在 `scan=True` 时） | 复现研究的预算探针：双语种子每 token 652,889,088（T = 1024），平坦字节 A0 每字节 153,545,760（`scan=False`），打开 scan 恰好多出 SSD 与卷积项 [已核实算术] |

H-Net 一行的 FLOPs = 外层（编码器、路由、残差投影、解码器、字节头）× 字节数 + 主网络 × 该行实际的 chunk 数（注意力上下文也用 chunk 数）；补齐位置不计。`match_hnet_config` 在网格上（编码器层数 = 解码器层数，从深到浅、每个深度从宽到窄，d₀ 不超过主网络宽度）找出与参照臂每字节 FLOPs 相差 ±3% 以内的全部配置，返回第一个并列出全部，找不到时拒绝并给出最接近的一个；结果是建议，H-Net 的 `model.hnet` 由臂登记固定。按 `matmul` 口径、K_train 行长 [CPU算术，c 为假设值，需用冻结候选 tokenizer 实测]：

| 表，参照臂的 c（字节/token） | 参照每字节 MFLOP | ±3% 以内的（层数, d₀, 误差） | STUDY §10 的默认 |
|---|---|---|---|
| TLC，c = 3.8 | 54.0 | (4, 128, −0.6%)、(2, 192, −0.8%)、(1, 256, −2.8%) | 2+2、d₀ 256：+9.9%，不在 ±3% 内 |
| TLC，c = 3.0 | 68.4 | (3, 192, +1.7%)、(1, 320, −0.3%) | 2+2、d₀ 256：+3.9% |
| 双语，c = 3.5 | 189.6 | (4, 256, +0.1%)、(3, 320, +1.9%)、(2, 384, +1.1%)、(1, 512, −0.1%) | 3+3、d₀ 256：−3.5% |

## 训练

`train`（`train.train`）在一张表的流上训练一个臂的一个种子，到表的 F_target；`branch` 取衰减分支。

- **进度**：π = 累计实际训练 FLOPs / F_target（prereg 的口径）。学习率（`schedule.WSD`，按每步之前的 F 取值）、稳定检查点与停止都是 F 的函数，所以一张表的所有臂都在 F_target 处停下（误差不超过一步），与字节/token 或 chunk 比无关；便宜的臂在同一个流中读得更远。开始时按该臂的名义每字节 FLOPs 检查流够不够长，不够即拒绝；运行中流用尽时报错停止，检查点保留。
- **WSD**：边界取整数 FLOPs（W = round(warmup × F_target)，D = round(decay_start × F_target)），F < W 时线性预热（第一步的学习率为 0，只填充 Adam 的矩），W ≤ F < D 时为峰值，之后线性（或余弦）衰减到 floor × 峰值。预算点 e 的衰减分支保留 W，终点 E_e = round(e × F_target)，衰减起点 D_e = round(decay_start × E_e)，从稳定检查点（到达 D_e 的那一步之后的状态）出发。测试确认：分支与一个 F_target = E_e、绝对预热相同（warmup 分数乘以 1/e）的运行逐位相同。
- **余弦调度**（prereg 的 `[schedule] kind = "cosine"`，`schedule.Cosine`）：预热到峰值后按 FLOP 进度余弦衰减到 floor × 峰值，即 TLC 原生训练器与双语训练循环的生产调度（它们按步数，这里按 F）；只给孪生门 TW-2、TW-3 用。它没有稳定阶段，所以 prereg 的预算点只能是 [1.0]，没有稳定检查点，`branch` 被拒绝。
- **数据与损失**：每步 `batch_rows` 行，按 `micro_rows` 分成微批，微批内右侧补齐；目标是每行第一个之后的所有 id（含特殊 token），补齐位置的目标不进损失。损失 = 本步目标 NLL 之和 / 本步目标数（微批划分不改变它）；H-Net 另加 α × ratio loss，按微批位置数加权。测试确认：一步的交叉熵在 1 行与整批两种微批下都等于逐行、不补齐算出的逐目标交叉熵，H-Net 的 ratio 项等于各微批 ratio loss 按位置份额的加权和。
- **优化器与精度**：AdamW，β = (0.9, 0.95)，梯度裁剪 1.0；权重衰减只作用于矩阵（嵌入即 tied 头除外），norm、向量与嵌入不衰减；H-Net 外层的学习率乘以 `lambda_outer`。`bf16` 为 fp32 主权重上的 bf16 autocast（路由与残差投影在模型内保持 fp32），`fp32`、`fp64` 不用 autocast。损失或梯度范数非有限时停止。
- **检查点**（`tokstudy-ckpt/1`，原子写入）：每 `checkpoint_every` 步一个 `ckpt/step-<n>.pt`（保留最近两个），每个预算点 e 在到达 D_e 的那一步之后写 `stable/E<e>.pt`，到达 F_target 时写 `final.pt`；内容为可训练权重、优化器、RNG 状态（torch CPU 与 CUDA、numpy、python）、流游标、步数、F 与至此训练过的文本字节（判定的等字节视角用它）。从最近的检查点续训；测试确认在 0、1、7 步后中断（以及从早于中断点的检查点续训，被重算的那一步的指标行被替换）与不中断的运行逐位相同（单线程 CPU）。
- **运行的 prereg**：一个已存在的运行（续训、衰减分支、评测）只接受它开始时的 prereg，或表的 prereg 锁在它之后接受的 prereg；别的 prereg（包括原地改过的文件）要 `--amend 原因`（`train` 与 `branch` 都有），追加进锁。改变计算的字段（F_target、调度、FLOP 口径、预算点）都在运行 id 里，所以改了它们的 prereg 即使给了 `--amend` 也不能继续旧运行，只能开新运行。prereg 没有签字时训练照常进行，但在 stderr 警告：这样的运行只能进 `verdict --draft`。
- **阶段**：prereg 的 `[stages.<名>] stop_after = e` 让主运行停在预算点 e 的稳定检查点（双语初筛：F_target = F_dec，取 0.25、0.5 两个分支，花费 0.525 F_dec）；以没有停止点的阶段再次运行同一臂、同一种子，就是同一个运行（同一个运行 id）继续到 F_target。臂只能参加它在登记中列出的阶段，种子必须是登记中的种子。
- **记录**：`metrics.jsonl`（`tokstudy-metrics/1`：每步的 F、π、学习率、行、位置、目标、字节、交叉熵、损失、梯度范数、字节/秒与 FLOP/秒、峰值显存、游标；H-Net 另有 ratio loss、F、G、chunk 数、各领域的位置/chunk）与 `events.jsonl`（开始、续训、停止、完成及当时的环境）。

## 评测

`eval prefix` 建一张表的评测集（`tokstudy-evalset/1`）：留出文档（TLC 的 val、双语的 dev，或按项目分帧的样本文件）各自截到公共字节前缀，即不超过 K_eval、在字符边界上、表中每个臂都能一次放下的最大前缀（双语 trunk 的 `max_seq` 约束其 token、字节与 H-Net 臂，H-Net 按每字节一个 chunk 的最坏情况计；Mamba 孪生没有限制）；token 数不一定随前缀单调，所以从上往下逐个字符边界检查，第一个可行的就是最大值。前缀与文档一起存入 `docs.jsonl`，清单记下所用的臂及其 `arm_sha256`；`eval score` 拒绝两种情形：评测集未列出该运行的臂（或其 sha256 不同），以及评测集的臂集合（id 与 `arm_sha256`）与该运行当前的登记不同（臂增删之后要用 `eval prefix` 重建评测集、重新打分）；同一张表里来自不同评测集的 NLL 文件由 `verdict` 拒绝。双语混语文档不进评测集；生成的来源（TLC 的 `generated = true`、双语 kind 为 `synthetic` 的 Track S）默认不进评测集，`--include-generated` 才保留，清单的 `source.generated_excluded` 记下被排除的来源名；`--max-docs` 在每个领域按秩键取前若干篇。

`eval score` 给一个运行的一个预算点打分（1 为 `final.pt`，其余为对应分支的 `final.pt`），写 `tokstudy-nll/1`：首行为表头（表、后端 `tokstudy-torch`、臂与其 sha256、运行 id、种子、预算点、检查点及其 sha256、步数、F、π、K_eval、评测集 sha256、精度），之后每篇文档一行：文本字节、来自文本段的目标的 NLL 之和（nats）与个数，另列 `<eos>` 目标（敏感性变体）与其他特殊 token 目标的 NLL。每篇文档一次前向（按长度分批、右侧补齐；因果，补齐不影响真实位置）；要求第一个位置是特殊 token，并核对文本 token 的字节之和等于文档的文本字节。BPB(领域) = Σ nll_nats /(ln 2 × Σ 字节)，按字节加权；双语只在同一语言内比较。评测不要求与训练时相同的代码 commit：运行 id 用记录中的 commit 重算（其余输入照样核对），表头另记评测时的 commit；续训与衰减分支则要求同一 commit。测试：均匀 logits 的模型在两张表的六种模型上都给出 n_text_targets · ln V /(ln 2 · 字节)；零合并 BPE 上的 `SubwordLM` 与权重相同（行按 BPE 的 id 重排）的 `ByteLM` 给出相同的逐文档 NLL；分批与单篇结果相同；公共前缀等于暴力枚举的最大可行边界。

## 运行登记与预注册

- **预注册**（`prereg.py`，`tokstudy-prereg/1`）：[prereg/tlc.toml](prereg/tlc.toml) 与 [prereg/bilingual.toml](prereg/bilingual.toml) 写入了预算口径、F_target、预算点 {0.25, 0.5, 1.0}、K_train/K_eval、WSD 参数、阶段、混合与其单位、每臂种子数，以及判定阈值（STUDY §8.4(e)、§9.2；由 S16 的判定读取）。全部是 [建议]，`[signature]` 为空，等 owner 签字。F_target：TLC 为 A0 在 0.8B token 处（3 × `ir.flops_per_token` × 8e8 = 1.642e17），双语为 B0（种子几何，T = 1024）在 0.8B token 处（6 × MACs × 8e8 = 5.222e17）；测试由两个项目的公式重算。文件字节的 sha256 是它的身份，记入每个运行记录。孪生门另有 [prereg/tlc-twin.toml](prereg/tlc-twin.toml)（TW-2）与 [prereg/bilingual-twin.toml](prereg/bilingual-twin.toml)（TW-3）：余弦调度、只有预算点 1.0，用自己的运行根目录（运行手册第 5 步）。
- **运行登记**（`registry.py`）：运行目录为 `<runs>/<表>/<臂>-s<种子>-<运行 id>/`；运行 id 取 {arm_sha256, stream_sha256, config_hash, code_commit, seed, f_target, schedule, budget} 规范 JSON 的 sha256 前 16 位（budget 为 prereg 的 FLOP 口径与预算点：前者决定每一步的 F，后者决定稳定检查点的位置；阈值、种子数、阶段与签字不进 id），任何一项改变都是另一个运行（测试逐项确认），指向另一个运行 id 的目录被拒绝；代码的 dirty 标志记在 `run.json`（`tokstudy-run/1`），不进 id。`<runs>/<表>/prereg-lock.json` 记下该表第一次运行所用的 prereg sha256；之后换了 prereg 而不给 `--amend 原因` 即拒绝，给了就把新 sha256 与原因追加进历史；新运行可以用接受过的任何 prereg，已存在的运行只接受它自己的或在它之后接受的（锁按目录布局找：运行目录旁边的 `prereg-lock.json`；不在该布局中的运行只接受它自己的 prereg）。`list_runs` 列出一张表的全部运行记录并重算它们的 id。

## 与实现规格的出入（S4）

- **行的长度**：规格写“子词行携带 t_arm 个 token”，同时要求两臂的行覆盖同样的字节区间；两者只能择一。这里取后者：行是所有臂共用的 K_train 位置字节窗口，t_arm 作为名义平均长度记录（显存规划用它的上界 K_train）。
- **FLOPs 口径**：规格的 652,889,088（双语种子每 token）是 `params` 口径；默认的 `matmul` 口径（与 TLC 原生的 `ir.flops_per_token` 一致、F_target 用它）给出 652,738,560，差的是 norm 参数。prereg 记录所用口径。
- **`stop_after`** 停在该预算点的稳定检查点（0.9·e），即规格所说初筛花费 0.525 F_dec 的读法；分支与“F_target = E 的运行”逐位相同的前提是绝对预热相同。
- **未落地**：S15 的探针与“分段带状态打分与一次打分相同”（每篇评测文档都在 K_eval 内一次打分，现在没有需要它的路径）；S16 的统计与判定；TLC 的 `force_lexeme_starts` 消融。S6 中 TLC 原生的按字节 BPB 与 `--dump-nll`、S7 中双语的 `eval.length_unit` 随工作流的 S5 落地。
- **双语的 D_seed**：双语项目的 D_seed 混合（Track S、带标签文档等，`surogate_bilingual.data.mixture`）只在写 shard 时生成，不输出文档清单；本包的双语流目前读 `seed_train` 的文档并按中英平衡混合，要与 D_seed 完全一致，需要项目先导出该混合的文档清单 [待定]。
- **新增**：`stream` 命令（规格的文件树里有 `stream.py` 但命令表中没有）、评测集格式 `tokstudy-evalset/1`、prereg 锁 `tokstudy-prereg-lock/1`、prereg 的 `[stream] measure`。

## 与实现规格的出入（S5）

- **原生 BPB 的键。** 规格只说“每篇文档的 UTF-8 字节”。原生 shard 不记文档键，所以 `--dump-nll` 的键默认是 `<shard 文件>:<序号>`（同一份语料 spec 与去污染报告下各臂的文档顺序相同，同一个键在任何 tokenizer 下字节数都相同），给出 `--corpus` 时改用语料的文档键并逐篇核对字节数。被 `--max-rows` 截断的文档不进 BPB。原生的评测按 512 token 的窗口给整篇文档打分，本包按公共字节前缀一次打分：两张表本来就不混，但 TW-2 两边的上下文因此不同 [待定，运行手册第 5 步]。
- **`eval.length_unit = byte` 的长度**取候选的 token 所代表的 UTF-8 字节（解码后的文本），在没有 normalizer 时就是“分隔符 + 候选”的字节；每条预测另记 `candidate_bytes`，两种单位的读数都可以事后重算。
- **交接**：规格只写“按路径与 sha256 读取”；本包另核对项目自己的训练记录（语料身份、划分、无 normalizer），并提供 `--pin` 与 `tokstudy-handoff/1`。两张表的登记现在就提交，sha256 用占位符 `unpinned`。
- **登记中的取值**：双语各臂的 trunk 设 `max_seq = 4096`（每行 K_train = 4096 个字节位置，H-Net 每字节至多一个 chunk，所以每个臂都要放得下 4096；参数不变，只有位置表变大）；子词臂与 H-Net 臂的种子列出可能用到的全部（M6：|Δ| < 3σ 时补第 3 个），prereg 的 `[seeds]` 记计划的个数。N1 取 2+2 层、d₀ 192（在假设的 3.8 字节/token 下 −0.8%），B4 取 4+4 层、d₀ 256（3.5 字节/token 下 +0.1%）：研究的默认（2+2、d₀ 256；3+3、d₀ 256）在这些假设下超出 ±3%。
- **新增**：`surogate_corpora compose`（规格只说“把 spec 片段贴进项目 spec”）；`tokstudy-handoff/1`。

## 实测记录（本地，不进 CI）

2026-11-03，本机 CPU，`FORK_PY`（Python 3.12.3、tokenizers 0.23.3）：

- `intrinsic` 在三个 20 行 datasets-server 样本（Ultra-FineWeb en、zh，UltraData-Code-L2 py）上，用 Qwen/Qwen2.5-Coder-1.5B-Instruct 的 `tokenizer.json`（revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`，sha256 `c0382117…`，带 NFC，以 `--drop-normalizer` 去掉）与 openai-community/gpt2 的 `tokenizer.json`（revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`，sha256 `8414cab9…`），`--check` 打开：字节/token 为 4.147 / 4.030 / 3.780（Qwen）与 4.306 / 1.415 / 2.084（GPT-2）；字节、字符、token 数、非法 UTF-8 占比、512 与 2048 token 覆盖字节的中位数共 42 项与此前的探针 `compress_probe.json` 逐项相同 [CPU实测]。约 2 秒。
- 同一批样本上的 `align`（教师同上；NFC 在 60 篇文档中改变 0 篇）与用这 40 篇 en+py 文档、4,096 id 建的 freqtrim（`qwen` 与 `fine-a` 两种正则，用真实 Qwen 文件，`freqtrim-check` 通过）只用来确认真实文件能被读入和检查；子集在评测文档上建成、词表只有 4K，其比率不代表任何候选 [CPU实测，只作软件检查]。
- 训练与评测的软件检查（S4）：在 py（UltraData-Code-L2）与 en（Ultra-FineWeb）两个 20 行样本上按 TLC 分帧建流（K_train 512，py/en = 0.6/0.4；文本字节合计 254,810 = 153,269 + 101,541，与 §9 的样本字节数相同，没有任何改动）与评测集（K_eval 256），用 tiny 几何的 N2（字节）与 N1（H-Net）各训练到一个很小的 F_target（4e9 FLOPs，10 步与 27 步）并取 0.5 的衰减分支，`eval score` 在两个预算点上写出 NLL 文件。评测文档与训练文档相同、模型极小，BPB 数字没有任何意义，只确认真实文本能走通整条路径 [CPU实测，只作软件检查]。
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
| `surogate_tokstudy.docs` | 另加 `surogate_tlc.data.corpus`、`surogate_bilingual.data.docs`、`surogate_bilingual.data.manifest`（派生 manifest 用项目自己的 `load_manifest` 读，按内容重算 `manifest_sha256`） | 项目的其他模块 |
| `surogate_tokstudy.models.trunk_bilingual` | 另加 `surogate_bilingual.model.lm`、`surogate_bilingual.config` | 项目的其他模块 |
| 测试 | 另可按路径加载 `fork/tlc/prototypes/ref_mamba2.py`，并导入 `surogate_tlc.ir`、两个项目的 tokenizer 与 shard 模块；TLC 模型目录由 TLC 自己的代码在独立的解释器中写出（它的初始化要用仓库里的 Python `surogate` 包编译 IR，本进程从不导入该包） | — |

`tests/cpu/test_scaffold.py` 按 AST 检查这张表，并用埋入的违规样例确认检查本身有效；两个项目的 `test_scaffold.py` 各有一条检查确认它们从不导入本包或 `surogate_corpora`。
