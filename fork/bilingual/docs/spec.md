# 决定记录与数据契约（P0）

- **状态：** 规格。P0 的步骤 I1–I6 已全部落地，并通过 CPU 测试：脚手架（I1），模型核心、路由、组装、冻结、哈希与参数审计（I2），融合算子、合并与 router 门控（I3），数据管线与 Track S 生成器（I4），训练循环、各 recipe 与运行登记（I5），评测、统计、资源审计与运行手册（I6）（见“实现步骤”中的 I2–I6 落地记录）。没有训练或评测过任何实验模型，所以还没有任何实验结果；没有任何代码在 GPU 上运行过。凡是 RTX 5090 的数字都是 [估计] 或 TBD，以 G0.5 的实测为准。
- **来源：** 三份文档，都写于 2026-11-02，都不在本仓库中，由 owner 保存：
  - owner 的预注册稿 v1.1（下称“手稿”；本文的“手稿 §N”指它的章节）；
  - 对 v1.0 研究包的对抗式评审 v1.0（下称“评审”；文中的 NOV-、MA-、SCI-、IMPL- 编号和 NEW-01 指它的问题）；
  - 英文实现规格（P0），本文由它压缩而来。
- **优先级：** 本文与 AGENTS.md 冲突时，AGENTS.md 优先。
- **范围：** P0，即手稿 §13 的最小决定性实验集需要的全部代码；这些代码都能先在 CPU 上验证。

## 决定 B-D1

本项目实现为**独立的 PyTorch 包** `fork/bilingual`（已采用，owner 可以推翻）。

- 布局：src 布局，发行名 `surogate-bilingual`，导入包 `surogate_bilingual`，`dependencies = []`。
- GPU 主机：在隔离环境 `fork/.venv-bilingual` 中运行。
- CPU 测试：`fork/dev test bilingual` 在 `FORK_PY` 解释器下运行，只从检出目录导入，不安装任何包。这是 AGENTS.md 第 4 条允许、`fork/tlc` 也采用的做法。
- **不**运行在 Surogate 原生训练器上，原因有二：
  1. 原生训练器只绑定 `token_ids`、`position_ids`、`targets`，没有逐 token 的路由输入；
  2. 本研究需要的算子在原生侧都是新 op：逐 token 的 gather/scatter dispatch、块门控拼接、槽位的 pre/post 混合、通道选择。
- 约 100M 参数时，eager PyTorch 加 bf16 autocast 和 SDPA 在一张 5090 上应当够用 [估计；G0.5 实测]。
- 以后某个结论若需要原生训练器，那是另一个按 ledger 规则登记的 topic。

## 影响代码默认值的 owner 决定

完整的 owner 决定 B-O1 至 B-O19 见手稿 §14.2；下表只列改变代码默认值的几项，其余只影响实验配置或写作。每个默认值只在一处定义，即 `surogate_bilingual.config.DEFAULTS`（I2 落地）；每次运行都记录它实际使用的值。

| ID | config 键 | 默认值 |
|---|---|---|
| B-O3 | `routing.signal` | `"oob"`：路由以逐 token 的 route-id 张量在带外传递，存放在 shard 中。控制标签只作为普通 token 出现在 D_seed 中；用户文本里的字面标签要转义 |
| B-O4 | `model.slot_width` / `model.ffn_width`（族 `slot`） | 789 / 1728；B10-II 的 `slot_width` 为 395 |
| B-O5 | 双路臂的 `routing.gate_policy` | `"dual_const"`，每个 token 上 g = (0.5, 0.5)；`"dual_window"`（w = 32）为次要 |
| B-O9 | `train.expert_decay`、`train.weight_decay` | 专家 `"none"`（wd 0）；种子 wd 0.1；norm 与嵌入永不衰减 |
| B-O10 | `train.global_batch_tokens` 与调度 | 每步 65,536 token（16 × 1024 × accum 4）；warmup 2%；cosine 衰减到峰值的 10% |
| B-O11 | `tokenizer.*` | byte-level BPE，词表 32,768，256 字节初始字母表，normalization `"none"`，特殊 token 沿用 v1.0 的顺序 |
| B-O13 | `train.bridge_variant` | `"c"`：源语言前缀走冻结的源语言专家，损失只算目标段 |

## 仓库规则的落实

- **自有路径。** `fork/` 是 `fork/patches/ledger.toml` 中 `[owned]` 的前缀，所以本包**不需要 ledger topic**，也不修改任何上游文件。接线改动同样只在 fork 自有文件中：`fork/dev`、`fork/tools/` 下的脚本和测试、`fork/docs/` 下的文档、`fork/README.md`、`fork/.gitignore`，以及列在 `[owned].files` 中的 `.github/workflows/fork-ci.yml`。验收：`python3 -I fork/tools/check_divergence.py --strict` 在 ledger 不变的情况下通过。
- **AGENTS.md 第 4 条。** 措辞不改（B-O19）。该条把环境列表委托给 [fork/README.md 的“首次准备环境”](../../README.md#首次准备环境)，I1 只在那张表中加 `fork/.venv-bilingual` 一行。
- **环境。**
  - CPU 测试：`FORK_PY`，默认根目录的 `.venv`。CI 装 torch 2.11.0+cpu 与 `fork/tools/requirements-cpu-lane.txt`，其中已有 numpy、tokenizers、safetensors、pytest 和 pytest-timeout，**没有 scipy**。
  - GPU 主机：`fork/.venv-bilingual`，依次装入 sm_120 的 CUDA 版 torch（G0 记录确切的 wheel）、`fork/bilingual/requirements-gpu.txt`（numpy、tokenizers、safetensors、scipy、pytest、pytest-timeout），再 `pip install -e fork/bilingual --no-deps`。
  - 不要在仓库根目录运行 `uv run` 或 `uv sync`。
- **导入边界。** 包只导入标准库、torch、numpy、tokenizers 和 safetensors。scipy 是可选的，只用于大规模 LAP 求解，导入时要有保护。包从不导入 `surogate`、`surogate._fork`、`surogate.experimental`，也不导入 `fork/bilingual/reference` 中的任何文件。`import surogate_bilingual` 既不导入 torch 也不导入 `surogate*`，子模块按需导入 torch。
- **提交。** 提交需签名；每次提交前运行 `fork/dev map` 与 `fork/dev check`；推送与发布只面向 `CatMeow-233/surogate`。
- **owner 文档。** 手稿与评审**不由 I1 提交**。owner 若要把它们放进仓库，放在 `fork/bilingual/docs/` 下，并且要通过 `fork/tools/check_doc_links.py` 的严格检查：fork 文档属于 F 类，反引号中的每个仓库路径都必须存在。
- **参照代码。** v1.0 代码原样保存在 `fork/bilingual/reference`，说明见 [reference/README.md](../reference/README.md)：
  - 内容：`moe_pilot.py`、`evaluate.py`、`eval_cases.example.jsonl`、`compare.py`、`verify_merge_math.py`、`smoke_results.json`，加上 `MANIFEST.json`（每个文件的 sha256）和说明来源的 `README.md`；
  - 该目录不打包，只由测试按路径加载；
  - `test_compare.py` **不**复制：`fork/tools/gen_maps.py` 拒绝 `FORK_TEST_TREES` 之外的 fork `test_*.py`。它的断言在 I3 移植到 CPU 测试中。

## 模块边界

每个模块只有一项职责；“不得”一栏是禁止它承担的耦合。下表的模块在 I2–I6 中落地。

| 模块 | 职责 | 依赖 | 不得 |
|---|---|---|---|
| `config` | 带类型的冻结 config、默认值、profile、config 哈希 | 标准库 | 读数据；在导入时导入 torch |
| `canonical` | 规范 JSON、sha256、原子写入与替换文件或目录 | 标准库 | 了解任何 schema |
| `ops.*` | 张量算子：norm、rope、attention、FFN 块、融合数学、合并 | torch | 查看路由或名字；分发 token；读文件 |
| `model.backbone` | θ_S：嵌入兼 tied head、attention、norm、族 II 的读出、LP-k 非专家层上的种子 FFN | ops | 持有专家 |
| `model.bank` | 按名字、按专家层存放的专家参数容器 | ops | 计算路由 |
| `model.executor` | 单层内依赖路由的分发：one-hot gather/scatter、双路融合 | ops、routing.plan | 拥有参数；选择路由 |
| `model.lm` | `RoutedLM` = backbone + bank + executor | model.*、routing.plan | 从 token 推出路由 |
| `routing.plan` | 值类型 `RoutePlan` 与其校验 | torch | 了解 tokenizer 或数据 |
| `routing.policies`、`routing.tags` | 由段元数据或完整历史的标签构造 plan | routing.plan | 运行模型 |
| `routing.router` | Router 接口与单调门控函数（训练推迟） | torch | 被 P0 的运行使用 |
| `assembly.hashing` | 内容哈希与种子指纹 | torch、numpy | 了解文件布局 |
| `assembly.packs` | pack 读写（safetensors + JSON）、探针记录 | hashing、canonical | 训练 |
| `assembly.freeze` | 哪些张量可以改变；审计 | hashing | 构造优化器 |
| `assembly.assemble` | 组装 manifest、严格检查、保真度验证 | packs、model | 修改 pack |
| `audit.*` | 参数、MAC 与显存记账；预算行 | model、ops | 修改模型 |
| `data.*` | tokenizer、manifest、划分、去污染、shard、采样器、Track S | tokenizers、numpy | 导入 torch 模型 |
| `train.*` | 唯一的训练循环与各 recipe；状态；调度；目标函数 | model、assembly、data、registry | 在 test 划分上评测 |
| `eval.*` | 评分、NLL/BPB、生成、切换分解、统计、报告 | model、routing、data | 训练 |
| `registry.*` | 运行身份、运行目录、环境、矩阵行 | canonical | 持有模型代码 |

三条横向规则：

- 算子从不看到路由，分发只在 `model.executor` 中进行。这修正了 v1.0 手册中的接口 `InputSlotExpandedFFN.forward(h, route, fusion)`（IMPL-21）。
- 只有 `model.lm` 和 `eval.*` 接受 `RoutePlan`。`train.loop` 经 `routing.policies` 从 shard 的 route id 得到 plan。
- 只读参数一部分的算子（`ops.fusion.channel_select`）经 `ops.reads.declare_partial_read` 告诉参数审计它读了多少元素。这个钩子放在 `ops` 中，所以算子不依赖 `audit`（I3 落地记录第 2 条）。

命令行由 `surogate_bilingual.__main__.COMMANDS` 一张表定义，见 [README 的“命令行”](../README.md#命令行)。表中每一项写明实现它的子模块、入口函数和落地步骤；`verify` 与 `assemble` 同在 `assembly.assemble` 中（入口分别是 `verify_main` 与 `main`）。`bench` 落在 `audit.bench`（英文规格的文件树中没有为它列出模块；I6 落地记录第 1 条）。

## 数据契约摘要

**通用规则。**

- 每个 JSON 文档都有 `format` 字段，形如 `bilingual-<kind>/<major>`；读取方拒绝不认识的 major 版本。
- 哈希是规范 JSON 的小写十六进制 sha256。规范 JSON：UTF-8、键排序、分隔符 `(",", ":")`、不允许 NaN。
- 每次写入都是原子的：先写到同目录下的 `<target>.tmp-<pid>-<rand>`，fsync 文件（目标是目录时也 fsync 目录），再 `os.replace` 到目标。崩溃留下的 `*.tmp-*` 在下次启动时删除。

**格式一览。**

| format | 内容 | 落地步骤 |
|---|---|---|
| `bilingual-reference/1` | `reference/MANIFEST.json`：v1.0 原始文件的 sha256、字节数与在压缩包中的路径 | I1 |
| `bilingual-seed-fingerprint/1` | 种子指纹的哈希输入 | I2 |
| `bilingual-seed/1`、`bilingual-expert/1` | pack 的 `pack.json` | I2 |
| `bilingual-assembly/1` | 组装 manifest，不复制权重 | I2 |
| `bilingual-budget/1` | 预算报告 | I2 |
| `bilingual-fidelity/1` | `verify` 的强制路由保真报告（I2 新增，英文规格中没有） | I2 |
| `bilingual-manifest/1`、`bilingual-split/1` | 语料 manifest、划分文件；由划分物化出的 manifest 另记每个文件所属的划分（`split`）与来源（`derived_from`） | I4 |
| `bilingual-tokenizer/1` | tokenizer 旁边的 `tokenizer.meta.json`：训练输入及其 sha256、manifest 哈希、设置（I4 新增，英文规格中没有） | I4 |
| `bilingual-overlap/1` | 去污染的重叠报告：规则、各规则命中数、被标出的训练文档（I4 新增） | I4 |
| `bilingual-sampler/1` | 采样器的 `state_dict`，I5 写进训练状态的 `extra.json`（I4 新增） | I4 |
| `bilingual-shard/1` | shard 的头 `shard.json` | I4 |
| `bilingual-item/1`、`bilingual-datacard/1` | Track S 的题目（每行一项）与数据卡；数据卡分两种 `kind`：`track_s`（生成器）与 `d_seed`（D_seed 混合） | I4 |
| `bilingual-state/1` | 训练状态的 `extra.json`；另记可训练张量的名字与三个文件的 sha256（I5 新增） | I5 |
| `bilingual-schedule/1` | 状态中的调度器记录：步数、warmup 步数、总步数与学习率参数（I5 新增，英文规格中没有） | I5 |
| `bilingual-env/1` | `environment.json`：环境指纹（I5 新增，英文规格只列了字段） | I5 |
| `bilingual-run/1`、`bilingual-metrics/1` | 运行记录 `run.json`、每步指标 `metrics.jsonl`；指标另有 `targets` 与 `kl` 两个字段（I5 新增） | I5 |
| `bilingual-pred/1`、`bilingual-result/1` | 每项 × 每臂的预测、结果列表（`status` 默认为“未运行”）；结果文件另记模型来源、题目文件与 `items_sha256`、路由与评分设置（I6 落地记录第 13 条） | I6 |
| `bilingual-nll/1`、`bilingual-switch/1` | `eval-nll` 的 2×2 矩阵（带标签）与 unigram 基线；`switch` 的逐题分解与按方向的汇总（I6 新增） | I6 |
| `bilingual-stats/1` | `stats` 的输出：`kind` 为 power、pilot（G1′ 门槛与讨厌参数）、futility 或 confirm（I6 新增） | I6 |
| `bilingual-bench/1`、`bilingual-resource/1`、`bilingual-g0/1` | G0.5 扫描的每行（header、point、choice）；预算加实测资源的报告；GPU 测试写出的 G0 记录（I6 新增） | I6 |

**内容哈希与种子指纹（修正 IMPL-14）。** `content_hash(state)` 对 `sorted(state)` 中的每个名字，依次把下面的字节送入 sha256：

```text
name.encode("utf-8") + b"\0" + str(dtype).encode() + b"\0" + json(list(shape)).encode() + b"\0" + raw_bytes
```

`raw_bytes` 是 `t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()`，对 bf16 同样适用。哈希只取决于张量，与 pickle 容器、step 计数或 dict 类型无关；大端主机直接报错。种子指纹是规范 JSON `{"format": "bilingual-seed-fingerprint/1", "config_hash", "backbone", "seed_ffn", "tokenizer_sha256"}` 的 sha256，其中 backbone 状态覆盖 θ_S：嵌入（兼 tied head）、attention 权重、norm；族 `slot` 另含各层读出 `Q` 与 `b`；LP-k 另含非专家层上冻结的种子 FFN。

**pack（修正 IMPL-13）。** 一个 pack 是一个目录，含 `pack.json`、`weights.safetensors`（种子：`backbone.*` 与 `ffn.*`；专家：只有专家层的 `ffn.*`）、`probe.safetensors`（探针批的 logits 行，fp32）和只在种子 pack 中的 `tokenizer.json` 副本。张量键名：`backbone.emb.weight`；`backbone.layers.{l}.norm_attn.weight`、`backbone.layers.{l}.attn.{q,k,v,o}.weight`、`backbone.layers.{l}.norm_ffn.weight`；族 `slot` 另有 `backbone.layers.{l}.readout.Q.weight` 与 `backbone.layers.{l}.readout.b`；`backbone.final_norm.weight`；`ffn.{l}.{gate,up,down}.weight`（swiglu）或 `ffn.{l}.{R,W}.weight`（slot）。`pack.json` 记录名字、族、config 与其哈希、种子指纹、专家层、各文件哈希、训练元数据、`depends_on`、探针参数、代码提交与是否 dirty。探针批由 `torch.Generator().manual_seed(probe.seed)` 生成，形状 `[2, max_seq]`，id 在非特殊词表上均匀分布；`probe.safetensors` 只存位置 0、1、512、1023 处的 logits（`[2, 4, V]` fp32，pilot 词表下约 1 MiB），前向用种子骨干加本专家的独立模型。Bridge 变体 (c) 与 T1′ 副本在 `depends_on` 中列出 `{name, expert_hash, path}`。

**组装 manifest（修正 IMPL-04）。** 记录种子 pack、各槽位（名字、路径、`expert_hash`）、融合方式、config 哈希、创建时间与代码提交。`assemble` 对下列每种失败抛出不同的 `AssemblyError`：pack 的名字与槽位名不同；槽位名重复；同一 pack 用了两次（`expert_hash` 重复）；`seed_fingerprint`、`config_hash` 或 `family` 不一致；权重键集合与期望的专家键不完全相等；记录的文件哈希与文件不符。槽位名取自 manifest，任何匹配 `[a-z][a-z0-9_]*` 的标识符都允许（例如普通的第三个专家 `t3`）。`load_assembly` 重新校验每个哈希。

**路由 plan。** `RoutePlan.gates` 是 float32 张量 `[B, T, E]`（fp64 测试也可用 float64），每行在单纯形上，P0 中至多两个非零项。one-hot 行走 Top-1 gather/scatter 路径；两项非零的行走模型指定的融合（swiglu 为 `output`/`concat`，slot 为 `pre`/`post`）。路由因果性：`routing.policies` 构造的 plan 只用位置 t 时已知的段元数据给 token t 分配路由；`R_out` 下答案段的路由是问题的语言，在答案开始前就已确定。shard 存 route id 而不是门控，门控由 `apply_gate_policy` 给出。

**语料 manifest 与划分（SCI-10）。** spec 中每个来源一个 `[[source]]`，字段：`name`、`kind`（`zh | en | parallel | codeswitch | synthetic`）、`files`（glob）、`url`、`revision`、`licence`（SPDX 风格 id，必填）、`licence_verified`（默认 false）、`downloaded`、`filters`。manifest 记录每个文件的路径、sha256、字节数、文档数，以及整份 manifest 的 `manifest_sha256`。划分文件记录 `{划分名: [doc_key, …]}`、种子、比例与聚类规则：普通文档按 NFKC、casefold、空白折叠后的文本哈希聚类；平行句对按 `pair_id`；合成文档按 `entity_id`，同一实体的所有文档落在同一划分。

**shard（修正 IMPL-07）。** 目录中有头 `shard.json` 与四个小端文件：`tokens.bin`（词表 ≤ 65535 时为 uint16，否则 uint32）、`doc_offsets.bin`（uint64，n_docs + 1 个偏移）、`routes.bin`（每 token 一个 uint8，取值是 `route_names` 的下标，255 表示未设）、`loss_mask.bin`（每 token 一个 uint8，1 表示该 token 是预测目标，即位置 t − 1 的 logits 被训练去预测 `tokens[t]`）。每个文档以 `<eos>` 结尾。头中记录划分、dtype、tokenizer/manifest/划分的 sha256、`route_names`、token 与文档数、各文件哈希、`tag_counts`、`mixed_window_fraction` 与 `eos_between_docs`。读取方拒绝 dtype 与词表大小不符、`tokenizer_sha256` 不是本运行的、或文件哈希不符的 shard。

**Track S 题目与数据卡（SCI-01、SCI-08、SCI-11）。** 每行一项，字段包括 `id`、`family`（E1、E2、E3par、E3ctx、E4）、`variant`、`direction`、`split`、`template_ids`、`segments`（每段的语言、角色与文本）、`answer_lang`、`candidates`、`gold_index`、`distractor_index`、`chance`、`knowledge_source`、`structured`、`difficulty`、`generator`。参数式题目（E1、E2、E3-par）的 `knowledge_source` 是事实所在的分区 `"zh"` 或 `"en"`，只供 `R_src` 使用；E4 另有 `slots` 与 `reference`，用于生成评分。数据卡记录生成器版本与种子、每个文件的 sha256、按族/划分/方向的计数、每个事实的曝光次数、各划分的模板 id 集合（必须互不相交）、实体分区、泄漏检验统计量和 D_seed 的构成比例。

**训练状态（修正 IMPL-02）。** 每个状态目录含 `trainable.safetensors`（只有可训练张量）、`optimizer.pt`（以 `weights_only=True` 加载）、`rng_torch.pt`（torch CPU 与 CUDA 的 RNG）和 `extra.json`（`run_id`、step、token 数、`frozen_hashes`、调度器、采样器游标、Python 与 numpy 的 RNG）。保留最近两个状态，另加每 `keep_every` 步一个。恢复时拒绝 `run_id` 不同的状态，并检查 `frozen_hashes` 与现有冻结张量一致；采样器游标与 RNG 在取第一个批次**之前**恢复。

**运行登记（修正 IMPL-21）。** 运行目录默认是运行根目录下的 data/runs（I5 落地记录第 1 条），其中的 `index.json` 把运行名映射到唯一的 `run_id`；每个运行目录有 `run.json`、`environment.json`（Python、torch、CUDA/cuDNN、驱动、GPU、CPU、线程数、git 提交与是否 dirty）、`metrics.jsonl`，以及状态、pack 和评测子目录。`run_id` 是规范 JSON `{config_hash, data_manifest_sha256, tokenizer_sha256, seed_fingerprint, code_commit, recipe, init_seed, data_seed}` 的 sha256 的前 16 个十六进制字符。运行名已绑定到另一个 `run_id` 时拒绝创建；同名同 id 则恢复。`code_dirty: true` 时拒绝运行，除非设置了 `registry.allow_dirty`，此时记录该标志。

**指标与预算。** `metrics.jsonl` 每个记录步一行：step、token 数、loss、lr、grad_norm、tokens_per_s、step_s、`peak_alloc_gib`、`peak_reserved_gib`、`dev` 与时间；CPU 上 CUDA 字段为 `null`。预算报告的每行：`id`、`P_total`、`P_active`、`P_update`、`weight_macs`、`macs_T1024`、`within_active_budget`、`within_update_budget`、`declared_in_budget`；I2 另加 `label`、`family`、`method`、`standalone_P_total`、`P_active_dual` 与 `note`（见 I2 落地记录）。

## 开关

| 键 | 取值 | 默认值 | 由谁决定 | 说明 |
|---|---|---|---|---|
| `model.family` | `swiglu`、`slot` | 按行 | 手稿 §7 | |
| `model.ffn_width` / `model.slot_width` | 整数 | 1728 / 789 | B-O4 | B10-II：`slot_width` 395 |
| `model.expert_layers` | `null` 或列表 | `null`（全部层） | 手稿 §7 的 LP 行 | LP-4 = [0..3] ∪ [20..23] |
| `model.gqa` | `sdpa_gqa`、`repeat` | `sdpa_gqa` | G0.5 | 后端不支持 GQA 时退回 `repeat`；RoPE 表按 (T, device, dtype) 缓存（IMPL-10） |
| `routing.signal` | `oob`、`tags` | `oob` | B-O3 | 选 `tags` 时所有 shard 遵循同一种标签约定 |
| `routing.gate_policy` | `one_hot`、`dual_const`、`dual_window` | Top-1 行 `one_hot`，双路行 `dual_const` | B-O5 | |
| `routing.g` / `routing.window` | 浮点 / 整数 | (0.5, 0.5) / 32 | B-O5 | |
| `fusion` | `output`、`concat`、`pre`、`post` | 按行 | 手稿 §7 | `concat` 用于 G0（B7） |
| `train.activation_checkpointing` | 布尔 | `false` | G0.5 | v1.0 无法关闭它（IMPL-09） |
| `train.global_batch_tokens` | 整数 | 65,536 | B-O10 | 必须等于 microbatch × seq × accum |
| `train.microbatch` / `train.seq` / `train.accum` | 整数 | 16 / 1024 / 4 | B-O10 与 G0.5 | |
| `train.lr` / `train.warmup_frac` / `train.min_lr_frac` | | 按 recipe / 0.02 / 0.10 | B-O10 | lr 网格见手稿 §8 |
| `train.weight_decay` | 浮点 | 种子 0.1；norm 与嵌入为 0 | B-O9 | |
| `train.expert_decay` | `none`、`to_zero`、`to_seed` | `none` | B-O9 | `to_seed` 是解耦的：w −= lr·λ·(w − w_seed) |
| `train.anchor_kl` | 浮点 | 0.0 | P1 | 确认性的 B8/B9 专家不用它 |
| `train.grad_clip` | 浮点 | 1.0 | | |
| `train.dtype` | `bf16`、`fp32` | CUDA 上 bf16 autocast，CPU 上 fp32 | | 参数始终为 fp32（每参数 16 字节） |
| `train.bridge_variant` | `a`、`b`、`c` | `c` | B-O13 | 损失只算目标段 |
| `data.window` | `packed`、`doc_aligned` | `packed`（文档间以 `<eos>` 分隔） | IMPL-07 | `doc_aligned` 推迟 |
| `data.seed_mix` | 比例 | 0.80 / 0.12 / 0.05 / 0.03 | B-O7 | 自然文本 / Track S / 混语与平行 / 带标签文档 |
| `tokenizer.normalization` | `none`、`nfkc` | `none` | B-O11 | |
| `eval.length_norm` | `sum`、`mean`、`auto` | `auto`（候选 token 长度全部相等时 `sum`，否则 `mean`） | 手稿 §9.1 | |
| `registry.allow_dirty` | 布尔 | `false` | | 记入 `run.json` |
| `data.seed_mix_parallel_fraction` | 0–1 | 0.5 | [建议]（I4 新增） | `codeswitch_parallel` 的 token 中由拼接平行对占的份额，其余为混语文本 |
| `data.seed_tags` | `start`、`switch`、`bridge` | `true`、`true`、`false` | B-O3（I4 新增） | 带标签文档在开头与每个语言切换点放标签；bridge 标签默认不进 D_seed |
| `data.min_mixed_window_fraction` | 浮点 | 0.01 | [建议]（I4 新增） | D_seed shard 在 `train.seq` 下的混语窗口比例下限，低于它时 shard 写入方拒绝 |
| `data.split_fractions` | 按文档种类的比例 | 见 `config.DEFAULTS` | [建议]（I4 新增） | 七个集合的文档级划分比例 |
| `data.decontam_rules` | 规则列表 | `en_word13`、`zh_han25`、`token13` | 手稿 §6.1（I4 新增） | |
| `model.profile` | profile 名或不设 | 不设 | （I5 新增） | 只用于行配置：先套 profile，再套文件自己的 model 键；不是 `ModelConfig` 的字段 |
| `train.adam_betas` / `train.adam_eps` | 两个浮点 / 浮点 | (0.9, 0.999) / 1e-8 | [建议]（I5 新增） | torch 与 v1.0 的默认值；手稿没有规定 |
| `run.*` | 见 `config.DEFAULTS["run"]` | 见左 | （I5 新增） | 运行名、recipe 及其参数、matrix 行、重复号、两个种子、steps 或 tokens、shard、tokenizer、种子 pack、其他 pack 与路由映射、device、保存/保留/记录的间隔；后三者与 device 不进入运行身份 |
| `eval.autocast` / `eval.max_new` / `eval.lang_share` / `eval.n_boot` / `eval.switch_k` | `none`、`bf16` / 整数 / 0–1 / 整数 / 整数 | `none` / 160 / 0.9 / 10,000 / 16 | [建议]；n_boot 与 k 来自手稿 §9.1–9.2（I6 新增） | 评测默认 fp32；E4 每阶段的生成上限；语言检查的文字系统份额；聚类 bootstrap 的重抽样次数；切换后峰值的 token 数 |
| `eval.alpha_branch` / `eval.sesoi` / `eval.noninferiority` / `eval.g1p` / `eval.power` / `eval.replicates` | 见 `config.DEFAULTS["eval"]` | 0.0125 / H_A 5pp、H_B 3pp、H_C 5pp / E1 2pp、BPB 0.02 / 机会 0.125 + 20pp / 0.8 / 3–6 | B-O8、手稿 §9.3、§9.5（I6 新增） | 都是 [建议]，pilot 揭盲前在签名提交中锁定（手稿 §9.7）；只供 `eval.stats` 读取，不进入运行身份 |

## 实现步骤

通用规则：

- 每个步骤由一个或多个签名提交落地。每次提交前 `fork/dev map`、`fork/dev check` 和 `fork/dev test bilingual` 都要通过。
- 下列每个验收条件都是 CPU 测试中一个有名字的 pytest 用例，标“手工”的除外。
- fp64 测试用 `tiny` profile；检查逐位相等的 CPU 测试调用 `torch.set_num_threads(1)`。
- **套件预算：** 整个 CPU 套件在 CI 的 CPU lane 上不超过 90 秒，单个测试不超过 20 秒；每个测试的硬上限是 `FORK_TEST_TIMEOUT` = 120 秒。

### I1：脚手架与仓库接线

交付物（都在 `fork/bilingual` 下）：`pyproject.toml`、`requirements-gpu.txt`、`README.md`；`fork/bilingual/docs` 下的 spec.md、runbook.md 与 measurements.md；`fork/bilingual/src/surogate_bilingual/__init__.py` 与 `fork/bilingual/src/surogate_bilingual/__main__.py`；`fork/bilingual/tests/cpu/conftest.py` 与 `fork/bilingual/tests/cpu/test_scaffold.py`；以及带 `MANIFEST.json` 与 `README.md` 的 `fork/bilingual/reference`。

仓库接线：`fork/dev` 的 `test bilingual`（case 分支、help 行、usage）；`fork/tools/gen_maps.py` 的 `FORK_TEST_TREES`；`fork/tools/check_doc_links.py` 中 `module_exists` 的根目录；`fork/tools/sync_upstream.py` 中 `checks()` 的套件列表；`fork/tools/tests/test_dev_timeouts.py` 的参数表；`.github/workflows/fork-ci.yml` 的 `Bilingual CPU tests` 步骤（`timeout-minutes: 15`）；`fork/README.md` 的环境表、测试命令行、代码位置表和文档列表；`fork/docs/build-and-release.md` 的套件表与超时段落；`fork/docs/sync.md` 的 CI job 列表与第 8 步；`fork/.gitignore`。

| 验收 | 内容 | 检查方式 |
|---|---|---|
| I1-A1 | `fork/dev test bilingual` 在 `FORK_PY` 下退出码为 0；额外参数原样传给 pytest | 手工运行；参数传递由 `test_dev_timeouts.py` 的 bilingual 用例检查（`fork/dev test tools`） |
| I1-A2 | `fork/dev help` 列出 `test bilingual`；未知套件以非零退出码结束，usage 中含 `bilingual` | `test_fork_dev_help_lists_the_suite`、`test_an_unknown_suite_exits_with_a_usage_that_names_the_suite` |
| I1-A3 | `fork/dev map` 不改动 `fork/docs/maps`；`fork/dev check` 与 `check_divergence.py --strict` 退出码为 0；ledger 无改动 | 手工，提交前运行 |
| I1-A4 | `git check-ignore -q fork/.venv-bilingual` 成功 | `test_environment_and_build_outputs_are_ignored`；该条在目录尚不存在时也成立，所以 `fork/.gitignore` 中这一行不带结尾斜杠 |
| I1-A5 | (a) 导入本包后 `sys.modules` 中没有 `surogate`、`surogate.*` 或 `torch`；(b) pyproject 的名字、`where`、`include`、`dependencies` 与 `requires-python`；(c) 本目录在每个发布 wheel 的打包目录之外；(d) `src/` 中没有对 `surogate`、`surogate._fork`、`surogate.experimental` 或 `reference` 的导入；(e) `reference/` 与 `MANIFEST.json` 一致且没有 `test_*.py` | `test_scaffold.py` 中同名的各用例；(d) 另有一个植入违规的对照用例，确认扫描能查出每一类越界 |
| I1-A6 | `fork/bilingual/src` 在 `sys.path` 最前面，`surogate_bilingual.__file__` 位于其下，否则会话报错 | `conftest.py`；`test_tests_import_the_checkout` |
| I1-A7 | CI 运行新步骤；CPU lane 上套件耗时少于 90 秒 | 手工，记入 [实测记录](measurements.md) |
| I1-A8 | `fork/dev check` 的严格链接检查覆盖每个新 markdown 文件；反引号路径与 `python -m surogate_bilingual` 能解析 | `fork/dev check` |
| I1-A9 | AGENTS.md 无改动；`fork/README.md` 的“首次准备环境”表含 `fork/.venv-bilingual` | 手工 `git diff`；`test_environment_table_lists_the_bilingual_venv` |

### I2：模型核心、路由、组装、冻结、哈希、参数审计

交付物：`config`、`canonical`、`ops` 中的 norm/rope/attention/ffn、`model.*`、`routing` 中的 plan/policies/tags、`assembly.*`、`audit` 中的 params/flops/budget，以及测试中做一步优化所需的训练桩。

- **I2-A1（与 v1.0 模型的一致性）：** 按路径加载 `fork/bilingual/reference/moe_pilot.py`，用 smoke 几何与名字 `['zh','en','bridge']` 构造它的 `RoutedLM`，把权重映射到 `swiglu` 族的 `RoutedLM`；fp32 CPU logits 在两种情况下 max-abs ≤ 1e-6：每个专家的强制路由；batch 2、T ∈ {128（= max_seq）, 37} 的随机混合路由。
- **I2-A2（dispatch 对稠密参照，IMPL-06）：** `tiny` 与 `smoke` profile，batch ≥ 2，T ∈ {max_seq, 一个奇数长度}，各行 plan 不同（随机 one-hot；双路：swiglu 的 `output`/`concat`，slot 的 `pre`/`post`；one-hot 与双路混合）；`Executor` 与 `dense_reference_ffn` 在 fp32 下 ≤ 1e-6、fp64 下 ≤ 1e-12。
- **I2-A3（NEW-01 回归）：** 在 `torch.autocast("cpu", dtype=torch.bfloat16)` 下每条 executor 路径都能运行；强制路由的输出与同一 autocast 下的单专家模型**逐位**相等。
- **I2-A4（MA-14）：** 有 ≥ 2 个专家时 `plan=None` 抛 `ValueError`；恰好一个专家时允许。
- **I2-A5（因果性）：** fp64、plan 从完整 plan 切片时，改动位置 ≥ t₀ 的 token 不改变位置 < t₀ 的 logits（逐位）。
- **I2-A6（数值，MA-13）：** RoPE 对复指数参照 fp32 ≤ 1e-6、fp64 ≤ 1e-12，角度表在提升后的 dtype 中构造；RMSNorm 在 fp64 下对闭式 ≤ 1e-14；GQA 的 `sdpa_gqa` 对 `repeat` 在 fp32 下 ≤ 1e-6（torch 不支持 `enable_gqa` 时跳过并写明原因）。
- **I2-A7：** 开关 activation checkpointing 的梯度在 fp32 下相差 ≤ 1e-7；每个冻结参数的 `.grad is None`。
- **I2-A8（路由，MA-09）：** `plan_from_tags` 在完整历史上运行，切片得到窗口 plan；回归 v1.0 的缺陷（标签在 31 个 token 之前、窗口 20）：窗口内重算得到默认路由，新路径保持标签指定的路由；`R_out`、`R_src`、`R_wrong`、`R_random` 对给定 rng 种子是确定的；`dual_const` 的行和为 1 且恰有两个非零项；`RoutePlan.validate` 拒绝不在单纯形上的行、负项、多于两个非零项、形状错误和未知名字。
- **I2-A9（哈希）：** `content_hash` 与 dict 顺序及非张量元数据无关；单个元素（fp32 与 bf16）、dtype、形状或键名改变时它都改变；safetensors 往返后不变。
- **I2-A10（pack 与组装，IMPL-04）：** 种子与专家 pack 读写往返后每个哈希不变；`assemble` 对上文“组装 manifest”中的每种失败（另含篡改的权重文件、缺键、多键）抛出不同的 `AssemblyError`；`t3` 这样的名字被接受；从 manifest 加载的模型在强制路由下复现各专家 pack 记录的探针行（fp32 CPU max-abs ≤ 1e-5，期望差为 0.0）。
- **I2-A11（冻结）：** 用专家 recipe 的 `FreezePlan` 在随机数据上做一步 AdamW：允许集合之外的张量内容哈希不变；优化器的参数集合按 `untyped_storage().data_ptr()` 比较等于允许集合；冻结张量 `.grad is None`。
- **I2-A12（精确预算，`pilot` profile，meta device）：** 数字来自手稿 §5.1；within-budget 标志只在 B6/B7 与族 II 的双路行为 `False`。

  | 量 | 精确值 |
  |---|---|
  | shared | 32,530,944 |
  | path | 63,700,992 |
  | single | 96,231,936 |
  | ZH+EN | 159,932,928 |
  | ZH+EN+Bridge | 223,633,920 |
  | 31 条路径 | 2,007,261,696 |
  | 族 II single / dual / P_update | 96,222,720 / 138,639,360 / 42,416,640 |
  | B10-II dual / standalone | 96,276,480 / 75,041,280 |
  | B10-I / 其 standalone | 96,231,936 / 64,381,440 |
  | B10-ch | 96,340,608 |
  | B11 r=128 / r=153 | 99,377,664 / 99,992,064 |
  | LP-4 P_update / P_total | 21,233,664 / 117,465,600 |
  | 2B 外推（词表 65,536，f=1328，每层 512×40 的 router）total / active | 2,008,015,360 / 98,755,072 |

- **I2-A13（执行路径上的 P_active，IMPL-21）：** 在 `TorchFunctionMode` 下运行并统计不重复的 storage，切片权重（例如 `channel_select`）用算子声明的元素数：单路径 = shared + path；ZH/EN 混合批为并集 shared + 2 条路径；tied head 只计一次；slot 族的双路 plan 为 shared_II + 2 个专家。
- **I2-A14（MAC）：** 在 `smoke` profile 与 `pilot` meta 模型上做 T = 8 的前向，用 `FlopCounterMode` 统计 Linear 与嵌入头的 FLOPs，必须恰好等于 2 × 解析权重 MAC × token 数（pilot 单路径每 token 96,206,848）；双路与单路的权重 MAC 之比四舍五入到四位小数为 1.6621。
- **I2-A15：** 预算报告等于 I2 提交的 golden 文件 budget_pilot.json。

#### I2 落地记录

**状态：** 已完成（CPU）。下表 15 条验收都是 `fork/bilingual/tests/cpu` 中有名字的用例，全部通过；没有在 GPU 上运行，CI 也还没有运行这些用例（见 [实测记录](measurements.md)）。`assemble`、`verify`、`audit` 三个命令随本步骤落地。

| 验收 | 检查方式（`fork/bilingual/tests/cpu` 下的用例） |
|---|---|
| I2-A1 | `test_model.py` 的 `test_i2_a1_logits_match_the_v10_model_on_forced_and_mixed_routes`（T = 128 与 37）；C1 上实测最大差 4.8e-7，来源见下文偏离 1 |
| I2-A2 | `test_i2_a2_executor_matches_the_dense_reference`：tiny 与 smoke、fp32 与 fp64、四种融合、one-hot/双路/混合 plan，逐层比较；`test_i2_a2_model_logits_match_a_dense_model`：整模型对稠密 executor |
| I2-A3 | `test_i2_a3_every_dispatch_path_runs_under_bf16_autocast`；对照用例 `test_i2_a3_the_v10_dispatch_is_the_regression_it_guards` 表明 v1.0 的 `_ffn` 在 CPU bf16 autocast 下确实报错，即 NEW-01 已在 CPU 上复现，CUDA 上仍未验证 |
| I2-A4 | `test_i2_a4_a_multi_expert_model_requires_a_plan` |
| I2-A5 | `test_i2_a5_later_tokens_never_change_earlier_logits`；同时检查窗口前向用完整 plan 的切片时与完整前向的前缀一致（≤ 1e-12） |
| I2-A6 | `test_numerics.py` 的四个 `test_i2_a6_*` 用例 |
| I2-A7 | `test_i2_a7_checkpointing_leaves_gradients_unchanged_and_frozen_grads_none`（两个族、四种融合，混合 plan） |
| I2-A8 | `test_routing.py` 的 `test_i2_a8_*`；窗口回归用例直接调用 v1.0 的 `route_from_tags`，确认它在窗口内回落到 zh |
| I2-A9 | `test_hashing.py` 的 `test_i2_a9_*` |
| I2-A10 | `test_assembly.py` 的 `test_i2_a10_*`：12 种失败各对应一个不同的 `AssemblyError` 子类，另有改名副本与未知 major 版本两例；`t3` 被接受；三个槽位的探针差都是 0.0 |
| I2-A11 | `test_freeze.py` 的 `test_i2_a11_*`（两个族；另有植入违规的对照用例） |
| I2-A12 | `test_audit.py` 的 `test_i2_a12_*`：上表全部数字，以及 `budget_v11.json` 的 17 行逐项比较 |
| I2-A13 | `test_i2_a13_*`；算子声明部分读取的机制用一个切片权重的玩具模块检查，`channel_select` 本身在 I3 落地 |
| I2-A14 | `test_i2_a14_*`：smoke 上两个族、单路与双路；pilot meta 模型上单路 96,206,848 MAC/token，比值 1.6621 |
| I2-A15 | `test_i2_a15_budget_report_equals_the_golden_file`；golden 文件是 `fork/bilingual/tests/cpu/golden/budget_pilot.json`，由 `audit --profile pilot --out` 生成 |

**实现中的决定与对英文规格的偏离（owner 可以推翻）：**

1. **RoPE 角度表在 float64 中构造**，cos/sin 再转成提升后的 dtype。规格写的是“角度表在提升后的 dtype 中构造”，但 fp32 角度表在 T = 128 时与精确值差 4.9e-6（T = 1024 时约 4e-5），达不到 I2-A6 的 fp32 ≤ 1e-6。代价是与 v1.0（fp32 角度表）的 logits 不再逐位相同：I2-A1 实测差 4.8e-7，仍在 1e-6 之内。
2. **`ops.fusion` 的一部分提前到 I2：** executor 需要的四种融合（`mix_outputs`、`stack_swiglu` 与 `concat_forward`、`slot_pre`、`slot_post`），因为 I2-A2 要求双路 dispatch 与稠密参照一致。交叉连接、通道选择、零初始化 adapter 及全部 fp64 恒等式测试仍在 I3。
3. **`RoutedLM` 的 `fusion` 默认为 None，即所属族的默认融合**（swiglu 为 `output`，slot 为 `post`）。规格签名中的默认值 `"output"` 不属于 slot 族；slot 族不接受 `output`，它在数学上就是 `post`，同一个臂不用两个名字。
4. **`RoutePlan.gates` 也接受 float64**，供 fp64 测试与 I3 的恒等式使用；默认 plan 与由 shard 生成的 plan 都是 float32。
5. **LP-k 的 config 哈希。** 种子 pack 的 config 中 `expert_layers` 为 null；专家 pack 的 config 只能在 `expert_layers` 上与种子不同（否则 `ConfigMismatchError`），同一组装中的专家之间要求 `config_hash` 完全相同（否则 `ConfigHashMismatchError`）。LP-k 时种子与专家的 `config_hash` 必然不同，所以规格中“每个 pack 的 config_hash 相同”按此细化；专家 pack 另记 `seed_config_hash`。
6. **组装的检查顺序**是：同一专家占两个槽位（`expert_hash` 重复，`PackReusedError`）先于名字是否一致。否则 `zh=zh 包 en=zh 包` 会先报名字不符；按这个顺序，改过 `pack.json` 中名字的副本同样以 `PackReusedError` 拒绝。规格之外另有：`InvalidSlotNameError`、`FusionError`、`FormatError`（含未知 major 版本）、`ConfigMismatchError`，以及写专家 pack 时骨干与种子不同的 `BackboneMismatchError`。
7. **manifest 中的路径相对 manifest 所在目录保存**，组装结果可以与 pack 一起移动。
8. **探针**总是在 CPU 上以 fp32 运行独立模型（种子骨干加该专家），与训练所在的设备无关；`pack.json` 的 `probe` 另记 `"device": "cpu"`。保真报告是新格式 `bilingual-fidelity/1`；bf16 CUDA 的容差 0.05 与 argmax 一致率 99.5% 只是 [建议]，G0 运行前锁定。
9. **路由策略的解读。** `R_src` 把整道题（提示与答案）都路由到 `knowledge_source` [假设]，owner 也可以改为只路由答案段；`R_random` 每段抽一次；`R_wrong` 在每一段上交换 zh 与 en。
10. **预算行**另含 `label`、`family`、`method`、`standalone_P_total`、`P_active_dual` 与 `note`。`method` 为 `executed` 时 P_active 来自执行计数；B10-ch 的块 router 与通道选择、B11 的 adapter、2B 外推的 router 在 P0 中没有前向（P1），这几行为 `structural`，这些部分按声明计数。B10-II 的槽宽取 ⌈789/2⌉ = 395（B-O4）。
11. **与手稿 §5.1 的一处不一致，需要 owner 确认：** B10-ch 的权重 MAC 本实现为 96,315,520，手稿表与 `budget_v11.json` 为 96,206,848。后者没有计入 rank-8 块 router 的 24 × 8 × (512 + 54) = 108,672 MAC，而这些 router 参数已经计入了它的 P_active。本实现计入这部分，与 B11 计入 adapter MAC 的做法一致。owner 若同意，手稿表的这一格改为 96,315,520，否则改代码。
12. **2B 外推行的 `P_update` 为 null**：v1.0 §5.2 的组织方式（每层 40 个小 FFN 加每层 router）不是整路径专家，P_update 没有定义。
13. **训练桩**只有 `train.recipes`（`SeedRecipe` 与 `ExpertRecipe` 的可训练集合）与 `train.objectives`（masked CE）。`train.loop` 没有落地，所以 `train` 命令仍以退出码 2 结束，并指向 I5。
14. **`config.load_toml` 只覆盖 `DEFAULTS` 的各节**：行文件的 stage 表与运行身份字段在 I5 加入，`extra_tables` 参数为此预留。

**仍未验证：** CUDA 上的全部行为（bf16 autocast 下的 executor、SDPA 后端、容差）；CI 上的这些用例与 CPU lane 上的套件耗时。

### I3：融合算子与合并（复用 `verify_merge_math.py` 的 fp64 参照测试）

交付物：`ops.fusion`、`ops.merge`、`routing.router` 中的 `topk_softmax_gates`，以及对应的测试。**移植测试：** 按路径加载 `fork/bilingual/reference/verify_merge_math.py`，调用 `main(tmp_path / "out.json")`，必须报告 `all_assertions_passed: true`，以此表明 owner 的文件仍然成立。除另有说明外，都在 fp64 下以 TOL 1e-10 运行。

- **I3-C1（拼接，MA-07）：** 常数 g 折进 W_down 时拼接等于加权 Top-2；逐 token 的单纯形块门控下 `concat_forward` 等于 `mix_outputs`（≤ 1e-12）；反例各断言误差 > 1e-3：静态拼接对逐 token 门控、不带门控的 down 偏置、输入侧门控、每专家的 post-norm；每专家的 pre-norm γ 折进权重列时 ≤ 1e-12；fp32 下拼接与混合相差 ≤ 1e-5（容差，不是逐位相等）。
- **I3-C2：** 隐藏通道置换保持专家函数不变。
- **I3-C3（交叉连接，MA-06）：** 零连接等于拼接，one-hot 端点精确；常数 g 下 `cross_link_lowrank` 等于重参数化 G_i + g_j·C_ij·G_j 后的拼接（≤ 1e-12）；秩 r 的 C = AB^T 等于带门控的 LoRA ΔG_i = g_j·A·(B^T G_j)（≤ 1e-12）；满秩参数公式 24·2·2·f² 在 pilot 几何下为 286,654,464，秩 r 的成本是每个投影、方向和层 2·f·r。
- **I3-C4（pre/post，MA-03、MA-04、NOV-13）：** one-hot 端点处 pre = post = 独立模型，差恰为 0.0；相同专家在 10 个随机单纯形 g 下对 `output`、`concat`、`pre`、`post` 都复现种子（≤ 1e-12），g = (1, 1) 时不复现（断言误差 > 0.1，记录在案的反例）；逐元素 SiLU 的 Jensen 界 |φ(Σg a) − Σg φ(a)| ≤ ¼·g₁g₂·Δa² 在 10⁵ 个随机对上成立，小 δ 时在 a = ±δ 附近达到界的比值 ≥ 0.99；sup|SiLU″| = 0.5 位于 x = 0（[−10, 10] 上步长 1e-4 的网格，≤ 1e-6），SiLU″ 的零点在 ±2.3994（±1e-3），x = 3 处 SiLU″ < 0；tanh 的紧常数为 4/(3√3)（≤ 1e-6）；φ 线性且 g = (1, 1) 时 post − pre = Q·b（≤ 1e-12）。
- **I3-C5（线性 P 的塌缩，MA-02）：** P 线性（z = Ru）、g 为常数单纯形时，`slot_pre` 等于第一层为 Σ g_i W_i R_i 的单个 MLP（≤ 1e-12）；逐张量合并减乘积合并等于 g₁g₂(W₁−W₂)(R₁−R₂)（≤ 1e-12）；g 逐 token 变化时，最小二乘意义下最好的固定第一层留下 > 0.1 的相对残差。
- **I3-C6：** B8 与 B9 参数量相等；用一次 Q 矩阵乘计算的 `slot_post` 等于逐分支的 Q 求和（≤ 1e-12）。
- **I3-C7：** `channel_select` 全选 ZH 得 ZH、全选 EN 得 EN；各选一半等于显式选出的网络；形状钩子断言不物化全宽的中间量。
- **I3-C8：** 两个宽 f/2 的 SwiGLU 专家在常数 g 下拼接，等于由它们堆叠的行构成的稠密宽 f SwiGLU（≤ 1e-12），即 B10-I 恒等式。
- **I3-C9：** `ZeroInitAdapter` 初始化时恰好加 0.0，一次反向后其 up 投影的梯度非零。
- **I3-C10（合并，NOV-6、IMPL-19）：** 相同专家的 `weight_average` 返回该专家；λ = 0 的 `task_arithmetic` 返回种子，λ = 1 且一个专家时返回该专家；density 1、一个专家、λ = 1 的 `ties` 返回该专家；p = 0 的 `dare` 等于 task arithmetic，固定 `rng_seed` 时确定，且无偏（2000 次抽样的均值在 task arithmetic 的 3 SE 之内）；专家 B 是 A 每层随机置换的结果时，`rebasin_align` 精确恢复置换，对齐后的平均等于 A（≤ 1e-12），相同专家时返回恒等置换、`identity_share = 1.0` 并标记为“等价于平均”；numpy 匈牙利算法在 200 个 n ≤ 6 的随机代价矩阵上等于穷举，在 20 个 n = 64 的矩阵上等于 scipy（没有 scipy 时跳过，在 `.venv-bilingual` 中运行）；`solve_lap(solver="numpy")` 拒绝 n > 256 并提示用 scipy。
- **I3-C11（Router 单调性，MA-11）：** 用 `topk_softmax_gates`、旧 logits 由冻结子矩阵单独做矩阵乘得到时，加入一个专家列后，所有不选新专家的 token 的门控在 fp32 与 fp64 下逐位不变；Switch 式原始概率改变 > 99% 的 token（记录在案的反例）。
- **I3-C12：** 移植秩约束 ridge（v1.0 检查 7）与 JS 分解（检查 8），检查 8 的说明限定为与上下文无关的预测器。
- **I3-C13：** 预算行带 `within_budget` 与 `declared_in_budget`，只对声明在预算内的行断言在预算内。
- **I3-C14（MA-10、SCI-17、IMPL-16）：** `compare.py` 中的恒等式改为标注“恒等式检查”的单元测试，永不作为结果报告；`test_compare.py` 的断言在这里移植。

#### I3 落地记录

**状态：** 已完成（CPU）。移植测试与 I3-C1 至 I3-C14 都是 `fork/bilingual/tests/cpu` 中有名字的用例，全部通过；只有 I3-C10 的 scipy 对拍在没有 scipy 的环境中跳过，本机另用放在临时目录中的 scipy 运行过一次（见 [实测记录](measurements.md)）。没有在 GPU 上运行，CI 也还没有运行这些用例。I3 不增加命令。这些用例检查的都是 [已知数学事实] 或基线的定义，不是实验结果。

| 验收 | 检查方式（`fork/bilingual/tests/cpu` 下的用例） |
|---|---|
| 移植测试 | `test_verify_merge_math_port.py` 的 `test_i3_port_verify_merge_math_main_reports_every_assertion_passed`：按路径加载 owner 的文件，调用 `main`，读回它写出的 JSON |
| I3-C1 | `test_fusion.py` 的五个 `test_i3_c1_*`：常数 g 折进 W_down；逐 token 块门控 ≤ 1e-12；四个反例（静态拼接、不带门控的 down 偏置、输入侧门控、每专家 post-norm，另验证随块门控的偏置 ≤ 1e-12）；pre-norm γ 折进权重列；fp32 容差 1e-5 |
| I3-C2 | `test_i3_c2_*`：族 I 的通道置换经 `merge.permute_channels` 完成；族 II 的槽位置换另有一例 |
| I3-C3 | `test_i3_c3_*`：零连接等于拼接，one-hot 端点逐位等于独立专家，连接改变双路输出（v1.0 检查 4）；常数 g 下等于重参数化后的拼接、秩 r 时等于带门控的 LoRA（都 ≤ 1e-12）；参数计数 286,654,464、每个投影/方向/层 2·f·r = 3456·r，并与算子因子张量的非对角元素数一致 |
| I3-C4 | `test_i3_c4_*`：one-hot 端点差恰为 0.0；相同专家复现种子在 tiny 模型的 logits 上检查（四种融合，各 10 组逐 token 的单纯形门控 ≤ 1e-12；slot 族的 b 取随机值，因为初始化时 b = 0，偏置约定测不出来）；g = (1, 1) 时四种融合的相对误差都 > 0.1，post 与 output 恰为种子的 2 倍；SiLU″ 的闭式先与 autograd 对拍，再在网格上检查 sup = 0.5、零点 ±2.3994、x = 3 处为负、最小值 −0.037；tanh 的紧常数；Jensen 界在 10⁵ 对上成立（舍入余量 1e-13），δ = 1e-2 与 1e-3 时比值 ≥ 0.99；线性 φ 下 post − pre = Q·b，单纯形上两者相等 |
| I3-C5 | `test_i3_c5_*`：含 v1.0 检查 1（输入复制在线性时坍缩为求和）；交叉项的符号见下文偏离 1；逐 token 门控下最好的固定第一层留下 > 0.1 的相对残差 |
| I3-C6 | `test_i3_c6_*`：tiny 与 pilot（meta device）上 pre 与 post 的 P_total、P_active 相等，pilot 双路为 138,639,360；用 `TorchFunctionMode` 计数，`slot_post` 与 `slot_pre` 各只做一次 Q 矩阵乘 |
| I3-C7 | `test_i3_c7_*`：形状钩子记录每个 torch 函数输出的形状，断言没有 [N, ≥ f] 的中间量；对照（先算整个专家再掩码）被同一个钩子查出；`p_active` 只计入选中的通道（3·d·f，v1.0 检查 6 的计数）；pilot 几何下 54 块选 27 块时声明的元素数乘以层数等于 B10-ch 行的通道计数 63,700,992；非法选择报错 |
| I3-C8 | `test_i3_c8_two_half_width_experts_concatenated_are_one_dense_swiglu` |
| I3-C9 | `test_i3_c9_zero_init_adapter_adds_exactly_zero_and_trains_its_up_projection`；另检查初始化时 down 的梯度为 0、参数为 2·d·r |
| I3-C10 | `test_merge.py` 的 `test_i3_c10_*`：另有 TIES 的手算例、DARE 与键顺序无关、合并拒绝不一致的输入、族 II 的 Re-Basin、numpy Hungarian 在 50 个有同值的整数矩阵上与穷举的最优值相等、n = 256 时找回植入的置换，以及 `solve_lap` 的各种拒绝 |
| I3-C11 | `test_router_monotone.py` 的 `test_i3_c11_*`：fp32 与 fp64、k = 1 与 2；新专家在 5%–95% 的 token 上胜出，所以检查不是空的；选了新专家的 token 的旧门控确实改变；k ≤ 2 的门控是合法的 `RoutePlan` |
| I3-C12 | `test_verify_merge_math_port.py` 的 `test_i3_c12_*`：秩约束 ridge 的满秩解等于闭式 ridge 解，各秩的目标差等于谱尾，与 owner 的 `ridge_rank_delta` 一致，随机的秩 r 扰动不会更好；JS 分解在 2 类（v1.0 的 p、q）与 7 类上成立 |
| I3-C13 | `test_i3_c13_*`：只有声明在预算内的行被断言在预算内，超预算的恰好是 B6/B7、II-dual、II-n512-dual；v1.0 检查 11 的 shared、path、各行总数与代码计数一致，v1.0 的两专家通道选择布局就是 B10-ch 行（96,340,608） |
| I3-C14 | `test_fusion.py` 中带 `identity_check` 标记的四个 `test_i3_c14_identity_check_*`，含 v1.0 `test_compare.py` 的三条断言；`fork/dev test bilingual -m identity_check` 只运行它们 |

**实现中的决定与对英文规格的偏离（owner 可以推翻）：**

1. **I3-C5 交叉项的符号。** 英文规格写的是“逐张量合并减乘积合并等于 g₁g₂(W₁−W₂)(R₁−R₂)”。由 g₁ + g₂ = 1 推得 W̄R̄ − Σ gᵢWᵢRᵢ = −g₁g₂(W₁−W₂)(R₁−R₂)，方向相反。测试断言正确的方向（乘积合并减逐张量合并等于 g₁g₂(W₁−W₂)(R₁−R₂)），并断言规格写的方向不成立。评审 MA-02 与手稿 §4.3 第 2 条只说“相差交叉项”，没有写方向，不需要改。
2. **部分读取的声明钩子移到 `ops.reads`。** I2 把 `declare_partial_read` 放在 `audit.params` 中；`channel_select` 若从那里导入，`ops` 就依赖了 `audit`，违反模块边界（`ops` 只依赖 torch）。现在钩子在 `ops.reads`，`audit.params` 在一次前向期间安装接收端，并为已有的调用者重新导出 `declare_partial_read`。
3. **`rebasin_align` 按规格返回（对齐后的状态, 恒等比例）。** 每层的置换由 `rebasin_permutations` 给出，报告用的标签由 `alignment_label` 给出（恒等比例为 1.0 时是 `"equivalent to averaging"`）；规格签名之外另加可选参数 `solver`。
4. **合并的细节**（规格没有规定，均为 [建议]）：合并只接受专家 FFN 状态（`ffn.*` 键），遇到 `backbone.*` 键报错；在 float64 中计算，再转回输入的 dtype；`weight_average` 的权重必须在单纯形上；`ties` 按张量修剪（常见实现的做法；论文对整个 task vector 修剪，全局版本没有实现），保留 ⌈density·n⌉ 个元素，同值时按稳定排序取前者；`dare` 每个张量的掩码来自独立的生成器，种子由 (rng_seed, 专家序号, 键名) 的 sha256 派生，所以与 dict 顺序无关。
5. **Re-Basin 也支持族 II：** 可置换的是 n 个槽位（R 的行、W 的列），Q 与 b 共享，不参与置换。规格只写了族 I 的相似度公式。
6. **numpy Hungarian** 是带势函数的 O(n³) 最短增广路算法，在列上向量化。`solver="auto"` 有 scipy 时用 scipy，否则用 numpy，而 numpy 拒绝 n > 256，所以 pilot 的 n = 1728 在没有 scipy 的环境中报错并提示 scipy。
7. **`cross_link_lowrank` 的因子布局：** A、B 都是 [2, E, E, f, r]（投影 gate/up、接收专家 i、发送专家 j、通道、秩），对角线 i = j 忽略；连接读取发送专家自己的 pre-activation，与 v1.0 的 `cross_mix` 相同；全秩连接写成 A = C、B = I。参数计数由 `cross_link_params` 给出。
8. **`channel_select` 的选择对所有 token 相同**，每个专家至多出现一次；B10-ch 的逐 token 块 router 在 P1。
9. **B11 的预算行改用 `ZeroInitAdapter` 构造**（I2 中是两层无偏置 Linear），参数与 MAC 不变，golden 文件不变。
10. **算子不校验门控。** 单纯形由 `RoutePlan.validate` 保证；算子接受 g = (1, 1) 这样被禁止的门控，只是为了让测试展示它破坏了什么（MA-04）。
11. **`identity_check` 标记**注册在 `fork/bilingual/pyproject.toml` 的 pytest 配置中，用于 `compare.py` 的恒等式（I3-C14）。
12. **`topk_softmax_gates`** 先取 top-k，再只对选中的 logit 做 softmax；第 k 名上的同值由 `torch.topk` 决定，不在不变性之内。`Router` 只是 Protocol，训练推迟。
13. **v1.0 各项检查的移植位置：** 检查 1–6 在 `test_fusion.py`，检查 7、8、11 在 `test_verify_merge_math_port.py`，检查 9 在 `test_router_monotone.py`，检查 10 由 I2-A11（`test_freeze.py`：经优化器作用于 nn.Module，并对每个冻结张量做内容哈希）覆盖。
14. **scipy 对拍在 GPU 主机上的运行方式。** 英文规格说 tests/gpu 也在 `fork/.venv-bilingual` 中以 CPU 运行这一对拍；tests/gpu 要到 I6 才有，所以本步骤改为在 [运行手册](runbook.md#g0-环境与-gpu-不变量) G0 第 1 步的判据中要求：CPU 套件在该环境中没有跳过的用例，`test_i3_c10_numpy_hungarian_equals_scipy` 必须运行并通过。

**仍未验证：** 这些算子在 CUDA 上的行为；CI 上的这些用例与 CPU lane 上的套件耗时；scipy 对拍尚未在 `fork/.venv-bilingual` 中运行。

### I4：数据

交付物：`data` 下的 tokenizer、manifest、split、decontam、shards、sampler 与 `data.synth.*`，Track S 生成规格 track_s.toml，以及带小型 fixture 的测试。

- **I4-D1（tokenizer，IMPL-08、IMPL-03）：** 在含 zh、en、emoji、CJK 扩展 B、全角标点与制表符的 fixture 语料上训练（`initial_alphabet=ByteLevel.alphabet()`）；编码产生 0 个 `<unk>`，500 个随机 UTF-8 字符串（含 4 字节字符）与每个标签字符串精确往返；特殊 token id 恰为 `<unk>`=0、`<eos>`=1、`<|zh|>`=2、`<|en|>`=3、`<|bridge|>`=4；`"，"` 与 `","` 保持为不同 token；`escape_user_text("用<|en|>和<|zh|>标记")` 不含特殊 id；训练拒绝不在 manifest 的 `seed_train` 划分中的输入文件。
- **I4-D2（manifest）：** 每个文件有 sha256、字节数与文档数；缺 `licence` 报错，`licence_verified` 默认 false 并在数据卡中列出；`manifest_sha256` 跨运行、跨文件列举顺序稳定。
- **I4-D3（划分，SCI-10）：** 相同种子给出逐字节相同的划分文件；近重复、平行句对、合成文档分别按上文的聚类规则落在同一划分；七个集合两两不相交；10k 个文档上每个划分的比例与目标相差在 ±1% 之内。
- **I4-D4（去污染，SCI-10）：** 英文 13 词 n-gram 查出植入的副本；中文按汉字的 25 字 n-gram 查出植入的中文副本，而按空白分词的 13-gram 查不出（回归）；token 13-gram 两者都查出；过滤恰好删除植入的文档。
- **I4-D5（shard）：** token、偏移、路由与损失掩码往返；篡改的文件在打开时哈希检查失败，tokenizer 哈希不符被拒绝；dtype 从头中读取（词表 ≤ 65535 为 uint16，否则 uint32）；`tag_counts` 与 `mixed_window_fraction` 等于在 fixture 上的穷举重算；标签占比 s 的种子 shard 中每个标签 id 至少出现声明的最少次数。
- **I4-D6（采样器，IMPL-02、IMPL-07）：** 相同种子给出相同批次；k 个批次后取 `state_dict` 再加载，继续得到的批次与不中断的运行完全一致；一个 epoch 内覆盖范围中每个位置恰好作一次目标（不重叠的块，每 epoch 随机偏移）；shard 的最后一个 token 可以作目标（回归 v1.0 的 `randint` 上界）；路由与损失掩码与 token 同步切片。
- **I4-D7（Track S，SCI-01、SCI-08、SCI-11）：** 相同（版本，种子）给出相同的输出字节，每个文件的 sha256 记入数据卡；`checker.py` 由 `structured` 字段独立重算 100% 题目的标准答案并与生成器一致，并确认渲染文本含有答案所需的事实；各族内模板 id 集合在 train、dev、test 之间互不相交，每族 ≥ 100 个 test 模板（完整规模）、每族每方向 ≥ 2,000 道 test 题，实体分区互不相交，test 中的事实对不在任何训练文档中共现；属性值与名字及分区独立（20k 次抽样的卡方 p ≥ 0.001）；E3-par 的泄漏检验：对 `same_city` 与 `month_sum_mod12`，P(gold | 任一单个事实) 是均匀的（卡方 p ≥ 0.001），故意泄漏的对照“谁出生更早”被标出（单事实的 Bayes 正确率 ≥ 0.7，p < 1e-6）；E3-ctx 去掉任一段后标准答案在候选上均匀（逐题结构检查）；E2 与 E3-ctx 的每个候选集都含正确值的另一语言形式，标签平衡在 ±1% 之内；每个事实的曝光次数恰好等于声明值。
- **I4-D8（D_seed，SCI-02）：** 构成比例与 `data.seed_mix` 相差在 ±0.5% 之内；含混语文档与拼接的平行文档；fixture 的混合窗口比例超过声明的最小值（v1.0 的打包方式为 2.6e-7）。

#### I4 落地记录

**状态：** 已完成（CPU）。I4-D1 至 I4-D8 都是 `fork/bilingual/tests/cpu` 中有名字的用例，全部通过；没有在 GPU 主机上运行，CI 也还没有运行这些用例（见 [实测记录](measurements.md)）。`tokenizer`、`manifest`、`split`、`decontam`、`shard`、`synth` 六个命令随本步骤落地。本步骤只生成数据与检查数据，没有训练，所以仍然没有任何实验结果；Track S 的泄漏检验与独立性检验是对**生成器设计**的检查，不是关于模型的结论。

| 验收 | 检查方式（`fork/bilingual/tests/cpu` 下的用例） |
|---|---|
| I4-D1 | `test_data_tokenizer.py` 的七个 `test_i4_d1_*`：fixture 语料含 zh、en、emoji、CJK 扩展 B、全角标点与制表符；特殊 id 为 v1.0 顺序，normalizer 为空；500 个随机字符串（含 4 字节字符与 emoji）及每个标签字符串 0 个 `<unk>`、精确往返；`"，"` 与 `","` 不同；对验收中那句含两个字面标签的中文，`escape_user_text` 的结果不含特殊 id（对照：v1.0 的编码方式把两个字面标签都变成控制 id）；拒绝 `zh_train` 的文件、manifest 中没有的文件和 sha256 不符的文件；同样的输入训练两次逐字节相同 |
| I4-D2 | `test_data_manifest.py` 的五个 `test_i4_d2_*`：每个文件的 sha256、字节数与文档数（`.txt` 以空行分隔文档）；缺 `licence` 或非 SPDX 风格的 id 报错且不写文件；`licence_verified` 默认 false，未核实的来源列在 `unverified_licences`，D_seed 数据卡照抄（`test_seed_mix.py`）；文件列举顺序反过来、整个语料搬到别处时 `manifest_sha256` 不变，改一个字节就变；篡改的 manifest 或文件被拒绝 |
| I4-D3 | `test_data_split.py` 的五个 `test_i4_d3_*`：同一种子的划分文件逐字节相同；全角字母、大小写、空白不同的近重复，平行句对的两侧，同一实体的合成文档，在 5 个种子下都落在同一划分；七个集合两两不相交，每个文档恰好出现一次，物化出的文件与划分一致；10k 个文档上每个划分与目标相差 ≤ 1%；去污染删掉的文档只从训练划分中去掉 |
| I4-D4 | `test_data_decontam.py` 的五个 `test_i4_d4_*`：英文 13 词规则恰好查出植入的英文副本；植入的 40 个汉字的中文副本，按空白分词的 13-gram 查不出（回归），汉字 25-gram 恰好查出；token 13-gram 两者都查出；三条规则合用时过滤恰好删除两篇植入文档；n-gram 哈希与穷举的集合交一致 |
| I4-D5 | `test_data_shards.py` 的五个 `test_i4_d5_*`：token、偏移、路由、损失掩码往返，用户文本中的字面标签仍是文本；篡改 `routes.bin`、tokenizer 哈希不符、未知 major 版本都被拒绝；词表 ≤ 65535 时为 uint16，否则 uint32（用 70,000 词的替身 tokenizer），dtype 从头中读取，与词表不符时拒绝；`tag_counts` 与 `mixed_window_fraction` 在 seq ∈ {3, 8, 64} 下等于穷举重算，分块扫描与块大小无关；标签占比 0.03 的种子 shard 中每个标签不少于声明的最少次数，去掉带标签文档后写入方拒绝，且不留下目录 |
| I4-D6 | `test_data_sampler.py` 的五个 `test_i4_d6_*`（含参数化，共 11 个用例）：同一种子同样的批次；k ∈ {0, 1, 9, 13} 个批次后取 `state_dict`，新建的采样器加载后与不中断的运行完全一致，种子或 token 数不同时拒绝；四种 (n_tokens, seq) 下每个 epoch 覆盖范围内每个位置恰好作一次目标；最后一个 token 可以作目标，而 v1.0 的 `random_batch` 在 200 次抽样中最多只到倒数第二个（按路径调用 owner 的文件）；路由按输入位置、损失掩码按目标位置与 token 同步切片 |
| I4-D7 | `test_track_s.py` 的十七个 `test_i4_d7_*`，细节见下 |
| I4-D8 | `test_seed_mix.py` 的五个 `test_i4_d8_*`：各成分的 token 比例与 `data.seed_mix` 相差 ≤ 0.5%（由返回的文档独立重算），Track S 整体保留，没有文档用两次，自然文本按 token 中英平衡；含按文字切分为 zh/en 段的混语文档和拼接的平行文档；带标签文档在开头和每个切换点有标签；seq 64 的 D_seed shard 混语窗口比例高于声明的下限 0.01 的 10 倍，而同一批文档按 v1.0 方式打包（先全部 zh 再全部 en）低于下限的一半，且不超过只跨一条 zh/en 接缝的上界；池子不够时报错 |
| 命令行 | `test_data_cli.py`：manifest → split assign → decontam → split materialize → tokenizer train/check → synth → shard（D_zh 与 `--seed-mix` 的 D_seed）经 `python -m surogate_bilingual` 走一遍；`--dry-run` 不写任何文件；被拒绝的输入以退出码 1 结束并给出原因 |

I4-D7 的各项：

- **确定性：** 同一（版本、规格、profile、种子）再生成一次，每个文件逐字节相同；数据卡记录每个文件的 sha256 与规格文件的 sha256；换种子则不同；输出目录已存在时拒绝。
- **标准答案：** `checker.py` 不调用生成器，只用闭集词表、人物表与文本：E1/E2/E3-par 由人物表重算，E3-ctx 完全由渲染文本解析（每个事实句恰有一个组号和一个颜色，每条规则恰有一个颜色和一个星期），E4 逐阶段重算槽位；ci 构建的 512 道题与完整规模的 40,000 道题都 100% 一致。参数式题目所需的事实由训练传记的文本扫描确认确实写出。另有一个用例逐族篡改题目（含只改 E3-ctx 事实句中的颜色、不改结构化字段），检查器全部查出。
- **不相交与规模：** 各族的模板 id（以及组成它们的每个部件）在 train、dev、test 之间不相交；注册表拒绝跨划分重复的文本、紧贴汉字的拉丁占位符和含词表值的 E3-ctx 模板。完整规模：注册表中 test 模板数为 E1 120、E2 120、E3-par 100、E3-ctx 104、E4 远多于 100；实际生成完整规模的 test 题目，16 个（族，变体，方向）每个 2,000 道，各族用到的 test 模板都 ≥ 100。人物分区互不相交、大小与声明一致，题目只问对应划分的留出人物；test 的人物对不在任何训练文档中同时被提到（按文本中的名字），组号在题目之间、题目与训练文本之间都不重复。
- **独立性与泄漏：** χ² 生存函数对已知分位数 ≤ 1e-6；20k 次抽样中 6 个属性分别对分区、名字两个部件的 18 个 χ² 检验 p ≥ 0.001；3000 × 3000 人的池子上各 6000 道 E3-par：`same_city` 与 `month_sum` 对 A 或 B 任一单个事实 p ≥ 0.001，单事实 Bayes 正确率不超过机会水平 + 0.05，标签计数相差 ≤ 1；“谁出生更早”对照单事实 Bayes 正确率 ≥ 0.7、p < 1e-6，被标出。
- **E3-ctx 去段检验：** 每道题结构上检查规则是 7 个颜色到 7 个星期候选的双射、所有事实颜色都在其中，所以去掉事实段时答案在 7 个候选上均匀，去掉规则段时（规则是随机双射）同样均匀；另把一道题的规则删到 5 条，检验报错。
- **候选与标签：** 1,600 道 E2 与 1,600 道 E3-ctx-cross 中每个候选集都含正确值的另一语言形式，8 个候选互不相同；正确位置与 E3-ctx 的正确星期各自与均匀份额相差 ≤ 1%。
- **曝光：** 不经检查器、直接从文本数出的每个事实的曝光次数恰好等于声明值（P_seed 每种语言 5 次共 10 次，P_zh、P_en 各 6 次，P_par 平行对 4 次；完整规模为 100、100、100、20），每人每种语言至少 5 种不同的传记。
- **词表：** 每个词表内外两种语言的形式都互不相同，没有一个形式属于两个属性词表；“陶艺家”这类包含另一属性值（“陶艺”）的形式由联合匹配处理（见下文第 13 条）。

**实现中的决定与对英文规格的偏离（owner 可以推翻）：**

1. **E3-ctx 的规则条数固定为 K = 7。** 手稿 §6.2 把规则条数（3–7）列为难度轴，又要求去掉任一段后答案均匀、机会水平为 1/(K+1)、K = 7。三者不能同时成立：规则少于 7 条时，只看规则段就能把答案缩到规则中出现的那几个星期（不看事实段的正确率 1/n > 1/8）。本实现让规则成为 7 个颜色到 7 个星期的随机双射，难度轴保留干扰实体数（0–3）和段落顺序；规格要求 `ctx_categories ≠ 7` 时构建报错并说明原因。另一种做法是 K 随规则条数变化、逐题记录机会水平，但这会让 G1′ 的门槛（机会水平 12.5% + 20 个百分点）失去固定的基准。**需要 owner 决定。**
2. **E3-par 的 `knowledge_source` 为 null**，另加 `knowledge_sources: {"a": "zh", "b": "en"}`。英文规格说 E3-par 的 `knowledge_source` 取 zh 或 en，但它的两个事实分别在 P_zh 与 P_en 中；`R_src` 对 E3-par 需要按片段路由的规则，留给 I6 与 owner。
3. **题目格式的补充字段：** `answer_sep`（zh 为空串，en 为一个空格，评分时接在候选前，与训练时 QA 文档的写法一致）；E4 的 `stages`（每个阶段的答案语言、槽位与参照译文，顶层 `slots`、`reference` 是最后一个阶段的）和 `text` 为 null 的 `answer` 段（回切题在生成的答案后用换行接下一段）；E3-par 的结构化字段有 `label` 与人名顺序 `order`；E3-ctx 的结构化字段用英文形式作颜色与星期的键；题目 id 含划分，例如 `E3ctx-cross-zh.en.zh-test-000123`。
4. **模板是组合的。** 一个角色的模板 id 是各部件 id 用 `+` 连起来（例如问题 = 外框 + 问句），“每族 ≥ 100 个 test 模板”按组合后的 id 计；每个部件池按划分不相交，同一角色内同一文本也不出现在两个划分中（比规格更严，跨属性也不行）。模板 id 由在列表中的位置导出，所以模板列表只能追加。外框（frame）由 E1、E2、E3-par、E3-ctx 共用。
5. **曝光次数按传记计：** 每篇传记写出全部 6 个事实，一个事实的曝光次数就是写出它的传记篇数，按分区与语言声明（P_seed 中、英文各 50，P_zh、P_en 各 100，P_par 的训练人物每人 10 对平行传记，即 20 次）。QA 文档另计，不算曝光，篇数在数据卡中按族列出。
6. **人物的分组。** P_zh、P_en 各一半为 `qa_train`（其 QA 进入专家训练数据），另一半留出，再按 `eval_dev_fraction` 分为 dev 与 test 人物，E1、E2、E3-par 只问留出的人物。P_par 一半的人物进入 D_parallel，另一半只用于 E4 的 dev/test，所以 E4 的 test 人物从不出现在训练数据中（手稿只说“对象是 P_par 的传记和模板句”）。D_seed 另含 P_seed 人物对的 E3-par 格式样本（`seed_par_samples`）[建议]：手稿没有规定 E3-par 的训练格式，没有它，“是/否”与余数两种答法在种子中从未出现。
7. **划分按配额分配**而不是按哈希分桶：各类文档的簇按 sha256(种子, 簇键) 排序后依次填满各划分的配额，每个划分与目标相差不超过一个簇的大小；代价是语料变化时边界上的文档会移动（语料变了 manifest 也就变了，划分按（manifest，种子）重算）。各类文档的默认比例是 [建议]，在 `DEFAULTS["data"]["split_fractions"]` 中。
8. **物化步骤。** `split materialize` 把每个划分的文档写成 `<out>/<划分>/<来源>.jsonl`，并写出一份由划分导出的 manifest，其中每个文件带 `split`；tokenizer 只接受这份 manifest 中 `seed_train` 的文件，并明确拒绝 calib、dev、test。tokenizer 在全部 `seed_train` 文档上训练，它们是 D_seed 混合的超集（混合按 token 比例对自然文本降采样），不含任何留出文本。
9. **去污染的顺序与实现。** token 规则需要 tokenizer，所以顺序是：词规则与汉字规则（划分后、物化前）→ 物化 → tokenizer → token 规则（写 shard 前，`shard --exclude`）。n-gram 以 64 位哈希比较，碰撞只会多删、不会漏删；英文的词是 NFKC + casefold 后按空白切分、去掉首尾标点；n-gram 不跨段。报告格式 `bilingual-overlap/1` 是新增的。
10. **shard 的头**另记 `vocab_size`、`min_tag_counts`、`mixed_window_fraction.floor`、`inputs`（输入文件及其 sha256）与 `extra`（D_seed 的数据卡）。`mixed_window_fraction` 取所有起点的窗口（即均匀随机起点看到的比例，与 v1.0 的 2.6e-7 可比）；`ShardReader` 必须显式传入运行的 `tokenizer_sha256` 与 `vocab_size`，传 None 只用于离线查看，且记录在 `checks` 中。
11. **采样器的窗口**是 seq 个输入加 1 个目标偏移；`routes` 取输入位置（计算该位置的专家），`loss_mask` 取目标位置。状态 `bilingual-sampler/1` 只有 epoch 与 index 两个游标，另存 n_tokens、seq、种子与 offset 用于校验。
12. **D_seed 混合**（新模块 `data.mixture`，英文规格的文件树中没有）：Track S 成分整体保留（否则预注册的曝光次数会变），总量 T 由它反推；自然文本按 token 中英平衡；`codeswitch_parallel` 一半是拼接的平行对、一半是混语文本 [建议]；带标签文档先取混语文档，再交替取中英自然文本；标签的最少次数就是混合时插入的次数，交给 shard 写入方核对。新增的四个默认值见上文“开关”。
13. **属性值的联合匹配。** 中文里两个属性的词表有包含关系（职业“摄影师”含爱好“摄影”，“陶艺家”含“陶艺”），在完整规模题目的第一次检查中，检查器因此误报了 157 道 E4 题（90 道的参照译文、67 道的源文）。`values.find_attributes` 对所有属性词表做联合的最左最长匹配后解决；I6 对生成文本做槽位评分时也必须这样匹配。
14. **词表内容**（均为 [建议]）：城市是 206 个真实地名（构建取前 n 个）；雇主由 20 个前缀与 10 个行业组合而成，不对应真实公司；英文职业不带冠词，模板也不在值前放 a/an；中文的“翻译”改为“口译员”（interpreter），免得与 E4 指令中的“翻译”混淆；`month_sum` 以一月为 1。
15. **新模块 `data.docs`**：整个数据管线共用的文档模型（文本段与标签段，各带语言、角色、路由与损失标志）和 JSON Lines 读写；文档读入时以 manifest 中的来源名为准。
16. **`routing.tags.escape_user_text` 改为调用 `data.tokenizer.encode_user_text`**，两处只有一份实现；因此 `routing.tags` 依赖 `data.tokenizer`（后者不导入 torch）。
17. **数据卡**分两种：Track S 的（`kind: track_s`，含检查器结果、曝光、泄漏与独立性统计量、模板集合、标签平衡、人物分区）与 D_seed 混合的（`kind: d_seed`）。数据卡中的泄漏统计量是在本次构建自己的题目上算的；ci 规模下题目太少，检验力很低，有检验力的版本在测试中用大池子运行。任何一项检查失败时构建报错，不写数据卡。
18. **CLI 的子命令：** `split` 有 `assign` 与 `materialize`，`tokenizer` 有 `train` 与 `check`；`--dry-run` 挂在子命令上。

**仍未验证与未实现：**

- 没有下载或处理任何自然语料：中英文自然文本、平行数据与混语文本的来源与许可都待核实（B-O12），数据卡把未核实的来源列出来。手稿 §6.1 的平行数据清洗（语言识别、长度比、对齐分数阈值、人工抽查 200 对）与领域分类器检查**没有实现**，`decontam` 只做去重之外的重叠过滤。
- Track S 的 test 题目没有再对训练文档做 n-gram 去污染：模板跨划分不相交、人物与组号都是新的，但这一点只由构造保证，没有用 `decontam` 跑过。
- 完整规模的 Track S 只在本机 C1 上构建过一次（见 [实测记录](measurements.md)），GPU 主机上没有运行；shard 写入与 tokenizer 训练在完整规模上的耗时与内存没有测（混语窗口比例的扫描按块进行，但每个 token 的语言码在写入时保存在内存中，十亿 token 约 1 GB）。
- CI 上的这些用例与 CPU lane 上的套件耗时。

### I5：训练

交付物：`train.*`、`registry` 中的 env 与 run、运行配置文件（pilot 行与 tiny 冒烟配置），以及种子、专家（共享部分冻结，可选 anchor KL）、joint-wide B12a、Bridge（a | b | c）、target-copy T1′ 与 router 桩的 recipe。配套：bf16 autocast、梯度裁剪、AdamW 参数组、warmup + cosine、解耦的向种子衰减、带优化器状态的 checkpoint、原子恢复、运行登记、环境指纹。验收在 CPU、`tiny`/`smoke`、单线程下进行：

- **I5-T1（IMPL-01）：** 相同 config 与种子的两个子进程得到相同的初始内容哈希和逐位相同的 20 步 loss 轨迹；`init_seed` 不同时哈希不同。
- **I5-T2（IMPL-02）：** 不中断的 10 步与“5 步、中断、恢复再 5 步”结束时，可训练权重、优化器状态、调度器状态、采样器游标与 loss 轨迹逐位相等。
- **I5-T3：** 在临时写入之后、`os.replace` 之前注入故障，之前的状态仍可加载，临时目录在下次启动时删除；保留策略为最近两个加每 `keep_every` 步一个。
- **I5-T4：** `run_id` 不同或冻结哈希不同时恢复报错。
- **I5-T5：** `tiny` profile 下种子 recipe 在固定批次上 300 步内把 loss 降到初始值的 0.1 倍以下。
- **I5-T6：** 专家 recipe 20 步后，骨干与其他专家的内容哈希不变，优化器只持有允许的参数；`anchor_kl = 0.0` 时轨迹与只用 CE 的训练逐位相等。
- **I5-T7（SCI-12、IMPL-09）：** 梯度为零时 `expert_decay = "to_seed"` 每步把 w − w_seed 恰好乘以 (1 − lr·λ)（fp64 ≤ 1e-12）；`"none"` 不改变权重；种子 recipe 中 norm 与嵌入在不衰减组。
- **I5-T8：** step 0、warmup 末、中点和最后一步的学习率等于 warmup + cosine 的闭式（≤ 1e-12）。
- **I5-T9（B12a）：** 由种子构造的宽 2f FFN（堆叠的行，`down` 乘 ½）复现种子的 logits（fp32 ≤ 1e-6）。
- **I5-T10（Bridge、T1′；SCI-06、IMPL-13）：** 只有 `loss_mask == 1` 的位置有贡献（改动被掩掉位置的标签，loss 与梯度逐位不变）；变体 (c) 把前缀路由到冻结的源语言专家（在 plan 上检查），并在 `depends_on` 中记录 ZH/EN 专家的哈希；T1′ 训练的是**副本**，原 pack 的哈希不变，每个运行更新 ≤ 63,700,992 个参数。
- **I5-T11：** 训练循环中开关 activation checkpointing 的梯度相差 ≤ 1e-7。
- **I5-T12（NEW-01）：** 种子、专家与 Bridge recipe 的 3 步 bf16 autocast CPU 冒烟，loss 有限。
- **I5-T13：** `metrics.jsonl` 每行符合 `bilingual-metrics/1`，CPU 上 CUDA 字段为 `null`。
- **I5-T14：** `run_id` 确定；运行名重用但 `run_id` 不同时拒绝；工作区 dirty 时拒绝，除非设置 `allow_dirty`（并记录）；`environment.json` 含每个必需的键。
- **I5-T15：** `RouterRecipe` 抛 `NotImplementedError` 并指明是推迟项。
- **I5-T16：** `microbatch × seq × accum ≠ global_batch_tokens` 的 config 在训练开始前被拒绝。

#### I5 落地记录

**状态：** 已完成（CPU）。I5-T1 至 I5-T16 都是 `fork/bilingual/tests/cpu` 中有名字的用例，全部通过；没有在 GPU 上运行，CI 也还没有运行这些用例（见 [实测记录](measurements.md)）。`train`、`env`、`registry` 三个命令随本步骤落地。测试中的训练用 `tiny` profile 和一个恰好 64 个 id 的替身 tokenizer（byte-level BPE 至少要 261 个 id，比 tiny 的词表大），只检查实现是否符合设计；其中的 loss 数值不是实验结果。

| 验收 | 检查方式（`fork/bilingual/tests/cpu` 下的用例） |
|---|---|
| I5-T1 | `test_train_loop.py` 的 `test_i5_t1_two_processes_give_the_same_init_and_loss_trace`：两个单线程子进程经命令行各跑 20 步，`run_id`、初始内容哈希与 20 个 loss 逐位相同；本进程按同一 config 建的初始化与之相同，`init_seed` 改为 99 时不同 |
| I5-T2 | `test_i5_t2_killed_and_resumed_equals_uninterrupted`（种子与专家两个 recipe）：子进程在第 5 步的状态写完后以 `os._exit` 退出（相当于 kill -9），本进程恢复再跑 5 步；可训练权重、优化器状态（每个张量与参数组）、调度器状态、采样器游标与 10 步 loss 轨迹都与不中断的 10 步逐位相同，`metrics.jsonl` 中每步恰好一行 |
| I5-T3 | `test_i5_t3_a_crash_before_replace_leaves_the_previous_state_loadable`：子进程在第 10 步的保存中、临时目录写完之后、`os.replace` 之前退出，留下 `step_00000010.tmp-*`；第 5 步的状态照常加载，下次启动删除临时目录并跑完；`test_i5_t3_retention_keeps_the_last_two_and_every_keep_every` 检查保留策略 |
| I5-T4 | `test_i5_t4_resume_refuses_another_run_id_and_changed_frozen_tensors`：`run_id` 不同时 `StateError`；状态中记录的冻结哈希与现有模型不同时 `FreezeError`，`run.json` 记为 failed；另有 `test_i5_t4_a_tampered_state_file_is_refused`（文件 sha256），运行名绑定另一个 id 时由运行登记拒绝（I5-T14） |
| I5-T5 | `test_i5_t5_the_seed_recipe_fits_a_fixed_batch`：恰好 seq + 1 个 token 的 shard 只有一个窗口，每个批次都相同；300 步内最低 loss 低于初始值的 0.1 倍（C1 上第 60 步首次低于，初始 4.19） |
| I5-T6 | `test_train_recipes.py` 的 `test_i5_t6_an_expert_run_changes_only_its_expert`：EN 在混合 shard 上训练、冻结的 ZH pack 处理 zh token；20 步后骨干内容哈希等于种子 pack 记录的值，ZH 的哈希等于其 pack 的值，优化器持有的 storage 恰好是 EN 的参数，pack 的 `depends_on` 记录 ZH；`test_i5_t6_anchor_kl_zero_is_bitwise_ce_only_training`：与测试中另写的只用 CE 的训练循环逐位相同（loss 轨迹与最终权重）；`anchor_kl = 0.5` 时第 1 步的 KL 恰为 0.0（专家起点就是种子 FFN），之后为正，轨迹改变 |
| I5-T7 | `test_train_schedule.py`：fp64 下梯度为零、按调度的 lr 走 6 步，`to_seed` 每步把 w − w_seed 乘以 (1 − lr·λ)（≤ 1e-12），AdamW 自身的衰减为 0；`none` 不改变权重，`to_zero` 等于 AdamW 的解耦衰减；两个族的种子 recipe 中 norm、嵌入（兼 tied head）与族 II 的 b 都在不衰减组，族 II 的 Q 在衰减组 |
| I5-T8 | `test_i5_t8_lr_at_the_four_points_equals_the_closed_form`：N = 1001、W = 20，第 0、19、20、510、1000 步与测试中另写的闭式相差 ≤ 1e-12 |
| I5-T9 | `test_i5_t9_the_width_2f_expert_built_from_the_seed_reproduces_the_seed`：max-abs ≤ 1e-6；pilot（meta device）上 B12a 更新两条路径，127,401,984 个参数；`test_i5_t9_b12a_trains_and_packs_a_wide_expert`：训练、写 pack、组装、`verify` 通过 |
| I5-T10 | `test_i5_t10_only_loss_mask_positions_contribute`：改动被掩掉位置的标签后 loss 与梯度逐位不变；`test_i5_t10_variant_c_routes_the_prefix_to_the_frozen_source_expert`：在 plan 上检查前缀走 zh/en、目标段走 bridge、损失目标都不在前缀中，并检查 (a)、(b) 的映射；`test_i5_t10_variant_c_records_the_zh_en_hashes`：`depends_on` 记录 ZH/EN 的哈希与相对路径，ZH+EN+Bridge 组装后 `verify` 通过；`test_i5_t10_target_copy_trains_a_copy_and_leaves_the_original`：原 pack 逐字节不变，副本的哈希不同，`depends_on` 有 source 与 origin，`p_update` 恰为一条路径，pilot 上为 63,700,992；含另一方向前缀的窗口被拒绝 |
| I5-T11 | `test_i5_t11_checkpointing_leaves_the_loop_gradients_unchanged`（种子与 Bridge (c)）：≤ 1e-7，冻结参数没有梯度 |
| I5-T12 | `test_i5_t12_three_bf16_autocast_steps_on_the_cpu_have_finite_loss`（种子、专家、Bridge）：精度记为 bf16，参数仍为 fp32 |
| I5-T13 | `test_i5_t13_every_metrics_line_validates_and_cuda_fields_are_null_on_cpu`：每行通过 `check_line`，CPU 上 CUDA 字段为 null，并有多字段、缺字段、布尔当整数、NaN、未知版本五个反例；`test_metrics_truncate_drops_later_steps_and_a_torn_line` |
| I5-T14 | `test_registry.py` 的五个 `test_i5_t14_*`：`run_id` 确定且覆盖每个身份字段；同名不同 id（含换了提交）被拒绝；dirty 时拒绝，设置 `allow_dirty` 后 `run.json` 记录 `code_dirty` 与 `allow_dirty`；训练循环同样拒绝；`environment.json` 含全部必需键 |
| I5-T15 | `test_train_recipes.py` 的 `test_i5_t15_the_router_recipe_is_deferred`：构造即抛 `NotImplementedError`，信息指明 Section 8；命令行以退出码 1 结束，不写任何文件 |
| I5-T16 | `test_train_loop.py` 的 `test_i5_t16_inconsistent_batch_arithmetic_is_refused_before_training`：读 config 时报错；命令行以退出码 1 结束，不建运行目录 |
| 其他 | `pilot_rows.toml` 的一致性（60 个运行、种子互不相同、每个专家行的种子 pack 是同一重复的种子行、每个重复的 token 预算等于手稿 §13）与各行在 pilot 几何上的 P_update；Bridge 布局与 `shard --bridge`；`--dry-run` 不写文件；跑完的运行再次调用时幂等；LP-k；dev/test shard、缺 pack、seq 超过 max_seq 都被拒绝 |

**实现中的决定与对英文规格的偏离（owner 可以推翻）：**

1. **运行目录在 data/runs 下。** 英文规格把运行放在仓库根目录的 runs 中。根目录的 `.gitignore`（上游文件）不忽略 runs，而 `fork/dev sync` 的干净检查（`git status --porcelain`）把未跟踪文件也算进去，所以根目录的 runs 会挡住同步。默认的运行目录因此是运行根目录（`train --root`，默认工作目录）下的 data/runs，由已有的 `data/**` 规则忽略；`--runs` 可以放到别处。`code_dirty` 只看 `fork/bilingual/src` 下的改动，与运行输出无关。这解决了 I4 时运行手册记下的待定问题。
2. **行配置的格式。** `[stage.NAME]` 表中放各节的子表（`run`、`model`、`train`…），逐键覆盖文件顶层的同名节；`replicates = [0, 1, 2]` 把阶段展开成 `NAME-r0`…，字符串中的 `{r}` 换成重复号，`init_seed`/`data_seed` 的列表按位置取；命令行的 `--set 节.键=值`（TOML 语法）与 `--name` 用于调参台账的单次试验。新增的 `[run]` 节与 `model.profile` 见上文“开关”。
3. **运行身份。** 进入 config 哈希的是：实际构建的模型几何（B12a 为 2f 宽）、train 节、`gqa`、`data.window`、recipe 及其参数、总步数、解析后的精度、shard 头的 sha256 与所读 pack 的哈希。device、保存/保留/记录间隔不进入。融合方式与门控策略也不进入：训练的 plan 都是 one-hot，它们只在组装与评测时作用于双路行，所以 B8 与 B9-II 评测的是同一组检查点。按规格 `code_commit` 在 `run_id` 中，所以一个运行不能跨提交恢复，换了提交要另起运行名。
4. **调度的细节**（规格只给了闭式的形状）：W = round(warmup_frac·N)（warmup_frac > 0 时至少 1）；第 0 步的 lr 为 P/W，不是 0；最后一步为下限。
5. **参数组。** 种子：除嵌入以外的矩阵衰减（含族 II 的 Q），norm、嵌入与 b 不衰减。专家类 recipe 不看 `train.weight_decay`，只按 `train.expert_decay`：`to_zero` 是 AdamW 自身的解耦衰减（系数 `expert_decay_lambda`），`to_seed` 在 AdamW 步之前用同一步的 lr 施加。T1′ 的 `to_seed` 锚点是种子 FFN（字面读法），owner 也可能更想要原 pack。
6. **AdamW 的 betas 与 eps**（新增 `train.adam_betas`、`train.adam_eps`）：取 torch 与 v1.0 的默认值 (0.9, 0.999)、1e-8 [建议]；手稿没有规定。
7. **损失的归一化。** 一个优化器步内所有 micro batch 的损失目标求和，再除以目标总数，所以每个目标的权重相同；没有任何目标的步报错。`metrics.jsonl` 的 `loss` 是每个目标的目标函数值（开锚点时含 anchor_kl·KL），另加 `targets` 与 `kl` 两个字段。
8. **KL 锚点**：参照是同一几何的种子模型（θ_S 加种子 FFN，LP-k 亦然），在同一批 token 上以 no_grad 前向；`anchor_kl = 0` 时根本不建参照，所以“只用 CE”按构造成立，并另用独立的 CE 循环核对。recipe 自带的 `anchor_kl` 必须与 `train.anchor_kl` 相同，否则拒绝。
9. **Bridge (c) 的第一个目标 token** 由源语言前缀的最后一个位置预测，该位置走源语言专家，所以这一项计入 loss，但没有可训练参数能降低它；这与推理时 R_out 的因果路由一致。
10. **Bridge 的 shard 布局**（新模块 `data.parallel`，`shard --bridge`）：源语言段的路由是其语言、loss 0，目标段的路由是 bridge、loss 1，`<eos>` 随目标段。T1′ 把 bridge 路由映射到目标语言专家的副本，需要单一方向的 shard（`--direction zh>en`）；窗口中出现另一方向的前缀时拒绝，不猜。T3′ 是专家名为 t3 的 Bridge recipe，训练数据由 `--unpair-seed` 在每个方向内按种子打乱配对、没有一对保持原样；T3″ 是专家名为 t3 的普通专家 recipe，用 D_zh ∪ D_en 的 shard（`shard --split zh_train,en_train`），token 数由运行的 `tokens` 对齐。T3′、T3″ 的数据定义是本实现的读法，需要 owner 确认。
11. **B12a**：`check_extends` 放宽为“族 I、每层都有专家时 `ffn_width` 可以与种子不同”，pack 记录宽几何，与普通专家组装时因 config 哈希不同而被拒绝。手稿只为族 I 列了 B12a，族 II 的版本没有实现。
12. **带上下文专家的专家 recipe**：`ExpertRecipe(expert, anchor_kl, context, routes)` 训练一个专家，同时让冻结的上下文 pack 处理按 `run.routes` 路由给它们的 token，`depends_on` 记为 context。Bridge (c) 与 T1′ 用的是同一套机制。
13. **pack 的 `depends_on`** 每项另记 `role`（source、origin、context），路径改为相对新 pack 目录的相对路径，便于整体搬移。
14. **`code_identity` 移到 `registry.env`**，pack 与组装 manifest 从那里取。
15. **状态的补充**：`extra.json` 另记可训练张量的名字和三个文件的 sha256，加载时核对；`trainable.safetensors` 按参数名存；优化器参数组带 `names` 与 `group`。
16. **恢复的补充**：`metrics.jsonl` 在恢复时原子地截到所恢复的步，崩溃留下的半行被丢弃；每次启动记一份环境指纹（首次为 `environment.json`，恢复时为 `environment.step_N.json`）；被拒绝的恢复把 `run.json` 记为 failed；`train --stop-after N` 跑到第 N 步、写状态后返回（状态仍为 running），`--no-resume` 拒绝继续已有状态的运行。
17. **P_update** 记入 `run.json`，并标出是否在 100M 的更新预算内；不强制，B12a 按设计超出，如实记录。
18. **一个运行只允许一个进程。** 没有加锁；同一运行目录上同时启动两个进程的结果未定义。

**仍未验证与未实现：**

- 没有 GPU：CUDA 上的 bf16 路径、峰值显存字段、CUDA 上恢复的容差（G0 的 g4、g5）与吞吐都没有运行。逐位确定只在单线程 CPU 上成立。
- 没有用自然语料建过 tokenizer 与 shard，`pilot_rows.toml` 的路径、token 预算与 lr 都是 [建议]，没有运行过；对 pilot 行的 `train --dry-run` 要等数据准备好才能运行。
- pilot 规模下每次写状态要对约 1 GB 的文件做 sha256 与 fsync，耗时没有测。
- CI 尚未运行这些用例；GPU 测试目录要到 I6 才有。

### I6：评测、统计与 RTX 5090 运行手册

交付物：`eval.*`、`registry.matrix`、完整的 [运行手册](runbook.md)、GPU 测试（无 CUDA 时跳过）与运行手册测试。

- **I6-E1（排序，SCI-11）：** 返回固定 logits 的桩模型下，选出的下标等于手算的和或均值的 argmax；构造 `sum` 与 `mean` 结论相反的例子，按声明翻转；只给候选 token 打分，prompt 是共享前缀。
- **I6-E2（评分，SCI-11、IMPL-15）：** 对标准答案 '43'，`extract_number` 拒绝 '143' 与 '430'；对 '25' 拒绝 '2025'；对 ['2024', '3'] 拒绝 '13 … 2024'；接受 '答案是43' 与 'The answer is 43.'；锚点列表为空时给 `None`（不适用），绝不给 `True`；缺失的专家产生带原因的跳过记录；记录 `items_sha256`。
- **I6-E3（生成，MA-09）：** 批量 greedy 解码等于逐项 greedy（token 相同，fp32 logits ≤ 1e-6）；路由轨迹等于完整历史的 plan；标签在窗口之外的例子保持标签指定的路由。
- **I6-E4（NLL 与切换，MA-08）：** BPB = nats / (ln 2 · UTF-8 字节数)（在构造的 logits 上检查）；2×2 矩阵带标签；无切换的题目上 M 恰好等于 R_new；有切换的题目上分解返回 {M, R_new, R_old}，在专家不同的小模型中 M ≠ R_new（> 1e-6）；`spike_after_switch(k=16)` 恰好平均每次切换后的前 k 个目标 token。
- **I6-E5（统计，SCI-07、SCI-09）：** `t_cdf` 在已知分位点上误差 ≤ 1e-5：(2.919986, df 2) → 0.05，(4.302653, df 2) → 0.025，(2.015048, df 5) → 0.05，(6.205, df 2) → 0.0125，(3.163, df 5) → 0.0125；`paired_t_one_sided` 等于手算；聚类 bootstrap 固定种子时确定，在模拟的聚类数据上 400 次模拟的覆盖率在 [0.92, 0.98] 内，耗时 ≤ 20 秒；`tost` 在构造的例子上给出正确判定；`graph_gatekeeper` 复现手稿 §9.3 图形的八种 p 值情形的手推表（含 α 转移）；三类判定函数在构造的例子上给出优于、等价、不确定；存在重复间 SD 时，重复级 t 检验在零假设下 4000 次模拟的一类错误率在 0.05 ± 0.02 之内；`power_paired_t` 随 R 与 δ 单调，MDE 随 R 单调。
- **I6-E6：** 重新生成的预算与资源报告等于 I2 的 golden 文件；CPU 上实测的 GPU 字段为 `null`，状态为“TBD”。
- **I6-E7：** 运行手册测试取出 [运行手册](runbook.md) 中每个标为 `bash runbook` 的代码块，对其中每条 `python -m surogate_bilingual …` 加 `--dry-run` 在 `tiny` fixture 上运行，必须以 0 退出且不在 `tmp_path` 之外写任何东西；建环境（`venv`、`pip`）与 `pytest` 的行只检查格式；运行手册通过严格链接检查，输出路径都在 `runs/` 与 `data/` 下，它们不是仓库中已跟踪的顶层目录，所以检查器不把它们当作仓库路径。
- **I6-E8：** `registry.matrix` 对手稿 §7 的每个 ID 都有一行及其状态（`minimal-primary`、`minimal-secondary`、`P1`、`deferred`、`G0`），结果默认为“未运行”；`python -m surogate_bilingual matrix` 打印该表。

#### I6 落地记录

**状态：** 已完成（CPU）。I6-E1 至 I6-E8 都是 `fork/bilingual/tests/cpu` 中有名字的用例，全部通过；没有在 GPU 上运行，CI 也还没有运行这些用例（见 [实测记录](measurements.md)）。GPU 测试在 `fork/bilingual/tests/gpu` 中（12 个用例），本机没有 CUDA，全部跳过；以 `BILINGUAL_GPU_TESTS_ON_CPU=1` 在 CPU 上演练过测试代码本身（10 个通过，2 个只能在 CUDA 上运行的跳过），这不是 G0 的结果。`eval-tasks`、`eval-nll`、`switch`、`stats`、`bench`、`matrix` 六个命令随本步骤落地，`audit` 另加 `--resources`。测试中的评测在 tiny 模型、替身 tokenizer 与构造的题目上运行，统计检验在构造或模拟的数据上运行；它们检查的是实现是否符合设计，其中的数值都不是实验结果。

| 验收 | 检查方式（`fork/bilingual/tests/cpu` 下的用例） |
|---|---|
| I6-E1 | `test_eval_score.py` 的三个 `test_i6_e1_*`：返回固定 bigram logits 的桩模型下，sum、mean 与 auto 选出的下标等于手算的 argmax（候选长度不同时 auto 为 mean，相同时为 sum）；log p 为 −2 的单 token 候选对每 token −0.9 的三 token 候选，sum 选前者、mean 选后者；桩模型看到的每一行都以 prompt 开头、只含自己的候选，改动预测 prompt 内部 token 的 logits 行不改变分数，改动预测第一个候选 token 的那一行才改变；prompt 超过 max_seq 时裁剪左侧，分数不变 |
| I6-E2 | `test_i6_e2_extract_number_uses_digit_boundaries`（规格中的全部例子，另有全角数字与非数字的金标准）；`test_i6_e2_slot_scoring_is_exact_and_language_aware`（名字按整词；“21987年”不算 1987 年；“摄影师”不算爱好“摄影”；英文 “19870”、“photographers” 不算；语言检查去掉名字后按文字系统）；`test_i6_e2_a_missing_expert_produces_a_skip_record_with_its_reason`（ZH+Bridge 模型上的英文题与英文规则段被跳过并写明缺少的专家；R_src 在 E3-ctx 与 E3-par 上跳过并写明原因；结果行中被跳过的题计数、不计分）；`test_i6_e2_items_sha256_is_recorded`；`test_i6_e2_eval_tasks_command_writes_predictions_and_results`（命令行全流程：`--dry-run` 不写文件；预测与结果的格式；聚类 CI；不静默覆盖；test 划分在锁定前被拒绝；连续的 gold 边际） |
| I6-E3 | `test_eval_generate.py`：`test_i6_e3_batched_greedy_equals_per_item_greedy`（one-hot 与 dual_const 两种门控，4 个 prompt，其中一个超过 max_seq；token 相同、每步 fp32 logits 差 ≤ 1e-6；路由轨迹等于最终序列的完整历史 plan）；`test_i6_e3_a_tag_beyond_the_window_keeps_the_tagged_route`（标签在窗口之外时轨迹全部是 en，第一步 logits 与强制 en 的窗口相同（≤ 1e-6），与 v1.0 式的窗口重算（回落到 zh）相差 > 1e-4）；长度不对的 plan 被拒绝 |
| I6-E4 | `test_eval_nll.py`：构造 logits 上的 BPB 与手算 nats/(ln 2 · 字节) 一致，窗口按 stride 前进时每个目标只算一次且值不变；2×2 矩阵带行、列与说明，混合语言的文档跳过并计数；无切换的题上 M、R_new、R_old **完全相等**，有切换的题返回三者且 \|M − R_new\| > 1e-6，给出翻译后的历史时另有 L_ref；`spike_after_switch(k=16)` 恰好平均每次切换后的前 16 个目标 token（遇到下一次切换即停，非目标不计入 k）；`eval-nll` 与 `switch` 的命令行（种子 pack 一行、组装两行；单专家组装被 `switch` 拒绝） |
| I6-E5 | `test_eval_stats.py`：五个已知分位点 ≤ 1e-5，另有 df = 1、2、3 的闭式（≤ 1e-12）与 `t_ppf`；`paired_t_one_sided` 对 [1, 2, 3, 4] 给出 t = √15 与 df = 3 闭式的 p；聚类 bootstrap 同种子相同、换种子不同、标签类型无关；400 次模拟的覆盖率在 C1 上为 0.945，耗时约 0.5 s；TOST 的构造例子（含恰在区间边界两侧的界值）；三类判定（显著但低于 SESOI 时不判为优于）；八种 p 值情形的守门手推表（含 α 转移的数额，并用反向的假设顺序检查结果与拒绝顺序无关）；4000 次模拟、R = 3、重复间 SD 0.05、Bernoulli 题目下一类错误率在 C1 上为 0.0505；功效随 R 与 δ 单调、δ = 0 时等于 α、MDE 随 R 单调、与 40,000 次模拟相差 ≤ 0.01；重复数规则；G1′ 门槛与无效性规则 |
| I6-E6 | `test_eval_report.py` 的 `test_i6_e6_the_resource_report_carries_the_golden_budget_and_tbd_gpu_fields`：资源报告中的预算（本会话从模块树重算）等于 golden 文件，实测字段全部为 null、状态 TBD；没有 CUDA 时给出的测量值也不采用；`test_i6_e6_audit_writes_the_resource_report`（`audit --resources`） |
| I6-E7 | `test_runbook.py`：每个 `bash runbook` 块的每一行都有已知类别（否则失败），包命令都已实现，`pytest` 行的目录存在，环境行只针对 `fork/.venv-bilingual`；运行手册通过严格链接检查；整本手册按顺序演练：每条包命令先 `--dry-run`（退出码 0，运行根目录中一个文件都不改），再以 tiny 规模真正运行（57 条包命令中的 55 条；`bench` 与 `verify --device cuda` 在没有 CUDA 的主机上只 dry-run），最后检查检出目录的 fork/bilingual 与 data 没有变化 |
| I6-E8 | `test_i6_e8_the_matrix_has_every_ms7_row_with_its_status_and_unrun_results`：手稿 §7 的 42 个 ID 与状态逐项相等，结果都是“未运行”，命令行打印全部行；有训练的行所列阶段都在 pilot_rows.toml 中；`test_i6_e8_the_matrix_shows_recorded_results` |
| 其他 | 显存记账：开 checkpointing 时 autograd 保存的字节数少于不开；CPU 上的 bench 点、选择规则与命令（没有 CUDA 时拒绝，`--device cpu` 时运行）；`stats` 的 pilot、futility、power、confirm 子命令（H_B 的前提不满足时记为“在这一分歧水平下无法检验”）；`test_scaffold.py` 改为检查全部命令已落地，“尚未落地”的路径用临时加入的命令检查 |

**实现中的决定与对英文规格的偏离（owner 可以推翻）：**

1. **新增模块。** 英文规格的文件树之外另有 `eval.items`（题目到 token 段、plan 与哈希）、`eval.models`（评测用模型的加载、臂的预设与合并）和 `audit.bench`（G0.5，规格没有给它模块）；`audit.memory` 在规格的文件树中，I2 没有交付，本步骤补上。`resource_report` 在 `eval.report` 中，`audit --resources` 在命令行这一层延迟导入它，`audit` 的计数代码不依赖 `eval`。
2. **BOS 约定：** 每个 prompt 与每篇评测文档以 `<eos>` 开头（训练时除 shard 的第一篇外，每篇文档都跟在 `<eos>` 之后，IMPL-15），它的路由取第一段的语言 [建议]。
3. **候选的切分：** 问题段与 `answer_sep + 候选` 一起编码（训练时的 QA 文档就是这样一段），检查问题段的 token 是共享前缀；若 tokenizer 在边界上合并，改为单独编码候选并在记录中标 `boundary_merge`（track_s.toml 的模板都以冒号或问号结尾，不会出现）。
4. **语言检查：** 去掉名字后，答案语言的文字系统在汉字与拉丁字母中占 ≥ 90% [建议]；规格的 fastText 语言识别在推迟项中。
5. **E4 的两个指标：** `slot_acc` 是语言正确的阶段中答对的槽位占全部槽位的比例；`correct`（结果中为 `all_slots`）要求每个阶段的每个槽位都对且语言都对。手稿说“每个槽位都要精确匹配，并且语言正确”，没有说是槽位级还是题目级；两者都记录，`stats confirm` 的 H_C 增益用 `slot_acc`。**需要 owner 决定。**
6. **生成：** 到 `<eos>` 或每阶段 160 个 token 为止 [建议]，不在换行处停；回切题的第二阶段在第一阶段的生成之后接换行和回译指令。没有 KV cache，每步重算最后 max_seq 个 token 的窗口。
7. **聚类：** 排序题按问题模板（外框加问句的组合 id）聚类，E4 按全部模板 id 的组合 [建议]。
8. **R_src：** E3-ctx 没有知识来源，E3-par 的两个事实在两个分区中，两者都按“不能评分”记为跳过并写明原因；这就是 I4 落地记录第 2 条留下的问题在本实现中的答案（另一种做法是按片段路由，需要 owner 决定）。运行手册在 E2 上评测 B4-src。
9. **守门与判定：** 图中的“拒绝”是 p ≤ 该假设当前的 α；三类判定叠加在上面，从未获得 α 的假设不能判为优于。H_B 的前提用 dev 预测的不一致率（各重复的平均）检查，不满足时 H_B 记为“在这一分歧水平下无法检验”，在图中按未拒绝处理，所以分支 1 不会把 α 转给 H_C（保守；手稿没有规定）。H_C 是交并检验：p = max(增益的 p, 各非劣部分的 p)。
10. **t 分布与功效：** 不依赖 scipy：t 的 CDF 用不完全 beta 函数的连分式，0 附近改用另一种形式以免精度损失，分位点用二分；功效默认对 χ² 做数值积分（确定、单调），也可以模拟；SD 为 0 的样本按均值的符号给 p = 0 或 1。
11. **bootstrap：** 百分位 CI，簇按第一次出现的顺序编号，所以结果与簇标签的类型无关；只作重复内的描述。
12. **G1′ 门槛与无效性规则：** 门槛取三个臂中各重复平均最高的一个；无效性规则用重复 0 在 dev E3-ctx-cross 上的 (B12a − B4) 与 (B6 − B4)。
13. **结果文件：** `results.json` 是 `{"format": "bilingual-result/1", ..., "results": [...]}`，而不是规格写的裸列表，因为每个 JSON 文档都要有 `format` 字段；另记模型来源（种子指纹、各专家哈希、融合方式及是否被覆盖）、题目文件的 sha256、`items_sha256`、路由与评分设置、代码提交。评测默认 fp32（`--autocast none`）[建议]。
14. **测试集的保护：** `eval-tasks` 与 `switch` 默认只评测 dev，遇到 test 题目时拒绝，除非显式加 `--allow-test`（手稿 §9.7 的锁定之后）。
15. **臂的预设与合并：** 手稿 §7 的行 ID 给出默认的路由、门控、融合与合并方式（`eval.models.ARMS`），每个都可以用参数覆盖；B3a/B3b/B3c 与 B3-II 在组装上以 `--merge` 在内存中合并，λ 与 TIES 的密度按调参台账在 dev 上选；DARE、RegMean/Fisher 与 Re-Basin 的臂（P1）没有接到命令行。
16. **资源字段：** `nvidia-smi` 一项是计时结束时的 memory.used（缓存分配器保留着峰值时的预留），不是轮询得到的峰值；MFU 用 6N 约定，N 为每 token 的权重 MAC，冻结骨干的 expert recipe 因此被低估。没有 CUDA 时 `bench` 拒绝运行，除非显式给 `--device cpu`（只供测试）。
17. **GPU 测试 (g1) 的随机初始化检查：** 在随机初始化的 pilot 几何上只断言 max-abs，argmax 一致率只记录：未训练模型的 logits 几乎相等，CPU 演练中 bf16 autocast 的 max-abs 为 0.022，argmax 一致率却只有 0.875。argmax 判据只对训练过的 pack 有意义（tiny 训练的 pack 与 G3 的 `verify`）。手稿 §9.4 的探针是均匀随机 token，训练好的模型在这样的输入上也可能有接近的 logits；这一判据是否合适，**需要 owner 在 G0 之前确认**。
18. **运行手册测试的演练：** 规格只要求 `--dry-run`；后面的命令要读前面命令的输出，所以测试在 dry-run 之后以 tiny 规模真正运行每条命令，替换的参数都列在 `test_runbook.py` 的 `TINY` 与 `TRAIN_TINY` 中；fixture 语料只列 5 篇 Track S 种子文档，好让 300 篇自然文本填满 D_seed 中 80% 的份额。运行手册的输出路径按 I5 的决定在 data/runs 下（规格写的是 runs/）。
19. **默认值的位置：** 评测与统计的默认值（上表 `eval.*` 的新行）都在 `config.DEFAULTS["eval"]` 中；它们不进入训练运行的身份。
20. **对照矩阵：** T0（主（参照））按 minimal-primary 记；每行另记训练它的 pilot_rows.toml 阶段。
21. **G1 的固定生成集：** [g1_prompts.jsonl](../examples/g1_prompts.jsonl) 是 8 个 [建议] 的提示（中英各 4 个），`eval-tasks --prompts` 记录输出与粗略的语言检查，判断要人工看。

**仍未验证与未实现：**

- 没有 GPU：tests/gpu 的 (g1)–(g6) 与 scipy 对拍、`bench`、CUDA 上的 bf16 容差与 SDPA 后端都没有运行。
- 没有实现：RtS（recall-then-switch）的评测协议；B9-II+aff（在 dev 上拟合的仿射参数）；T1′ 的评测需要一种组装命名（T1′ 的副本 pack 与原专家同名，不能和原专家组装在一起）；H_C 的 BPB 非劣部分还没有接入 `stats confirm`（需要每个重复 T2 与 T0 的 NLL 文件），目前只接了 E1；手稿 §9.1 的中介量（每层 \|a_ZH − a_EN\|、\|a\| < 2.40 的比例、权重漂移、CKA）与实体/数字保持率；GLMM 敏感性分析（推迟项）。
- 评测的耗时（没有 KV cache 的生成、对每个候选的前向）在 pilot 几何上没有测；手稿 §13 的“评测约折合 1B token”仍是 [估计]。
- CI 尚未运行这些用例；本机 C1 上整个 CPU 套件约 50 s（见实测记录），CI 的 CPU lane 上的 90 秒预算待测。

## 推迟项（不属于 P0）

| 项目 | 需要什么 | 阶段 |
|---|---|---|
| 学习式 Router 的训练（B5；IMPL-12、MA-11、NOV-9） | 在 R-early（r_t = Router(u_{0,t})）、R-late（r_{t+1} = Router(h_t^L)，以 oracle 路由 teacher forcing，并测量 exposure bias）与由 LM head 发出标签之间选一种；门控用 `topk_softmax_gates`；router 参数放在单路径 3,768,064 的余量之内；手稿文献 [7] 式的对齐只用于有可训练表示的中间层 | P1 |
| 修复阶梯 R0–R5 | 在 D_calib 上校准 router 或边界、B11 adapter、anchor-KL 重训、跨路径校准；每一步与 R0 在相同校准 token 与 GPU 时下比较，报告 Pareto | P1 |
| Bridge T0–T5 矩阵的运行 | I5 的 recipe 已有；按手稿 §13 运行；T4/T5 需要双路 + Bridge 的 executor | P1 |
| KV cache（IMPL-17） | 感知路由的 cache；单 token 步 `is_causal=False`；多 token 步（分块 prefill）用 `torch.nn.attention.bias.causal_lower_right`；显式位置偏移；对完整重算的 CPU 测试（含路由切换，fp32 ≤ 1e-5）；依赖 I2 的完整历史 plan | P1 |
| FX 协同训练、ST 拼接、B12b、B8-lin、B10-I/B10-ch/B10-II 的种子、LP-2/LP-6 | FX：双路训练 recipe；ST：双流 executor（约 2 倍 attention FLOPs 与 KV）；其余：各自的种子 | P1 |
| 自然语言轨的 LCB 指标 | fastText 语言识别模型（核实许可）；按字符数的中文行规则；自建 zh→en 题目 | G1-NL 之后 |
| GLMM 敏感性分析 | `.venv-bilingual` 中的 statsmodels 或同等工具；Kenward-Roger/Satterthwaite 自由度 | 分析 |
| 第二个规模（约 400M–1B） | 在一张 5090 上是否可行 TBD | 主会扩展 |
| `torch.compile`、融合或分块 CE | 放在开关后面，并有 CPU 等价测试 | G0.5 之后 |

## 已知限制与风险

- **本机没有 GPU。** CUDA 行为都未经测试：bf16 容差、sm_120 上的 SDPA 后端、kernel 的确定性，以及 NEW-01 这种失败本身（把握很高，但仍未验证）。G0 检查这些。
- **CPU 环境。** CPU lane 没有 scipy：大规模 LAP 求解与 scipy 交叉检查只在 `.venv-bilingual` 中运行。
- **统计。** 基于 statsmodels 的混合模型推迟，确认性检验不依赖它们。
- **逐位相等。** 逐位检查只在单线程 CPU 上成立；CUDA 上每个相等检查都是声明的容差。
- **Track S。** Track S 测的是种子已经学过的格式的保持与组合（手稿 §6.2），不能支持“组合能力由组装涌现”的主张。
- **预算。** 3 个重复可能不足以给 H_C 足够的功效；扩展规则与成本见手稿 §9.5 与 §13。
