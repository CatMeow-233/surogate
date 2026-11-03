# 运行手册：备料、tokenizer 交接与 RTX 5090 上的分词器比较

本手册把 owner 的决定（原文）「分詞器得對比效果要不然就搞另類神經網絡分詞 語料用minicpm的ultradata那些的」落到可执行的步骤：语料取自 UltraData 家族（[fork/corpora](../../corpora/README.md)），两个项目各自的 tokenizer 按训练后的效果（等 FLOPs 的 bits per byte）比较，H-Net 1-stage（学习式动态分块，即“另類神經網絡分詞”）作为参与比较的候选臂。

**当前状态：本手册的步骤全部未运行。** 本机没有 GPU、也放不下语料（约 6 GB 空闲）：备料在 owner 主机上做，凡 RTX 5090 的数字都是 [估计] 或 TBD，以 G0.5-ts 实测为准。每一步的结果按主机、提交、日期记入 [实测记录](measurements.md)。判据不成立时记录实测值并停下，不放宽阈值；失败的种子不丢弃，试过的每个配置都计入判定。

## 0. 前提

**环境。** 每个包的命令只在它自己的环境中运行（AGENTS.md 第 4 条），环境的建法见 [fork/README.md 的“首次准备环境”](../../README.md#首次准备环境)：

| 步骤 | 环境 |
|---|---|
| 备料（`python -m surogate_corpora`） | `fork/.venv-corpora`，需要网络 |
| TLC 的去污染、tokenizer、shard、原生训练与评测 | `fork/.venv-tlc`（[TLC 运行手册的 G0](../../tlc/docs/runbook.md#g0-环境)） |
| 双语的 manifest、划分、去污染、tokenizer | `fork/.venv-bilingual` |
| 研究包的全部命令（`python -m surogate_tokstudy`） | `fork/.venv-tokstudy` |

**目录。** 研究的输入与输出放在检出目录的 `data/tokstudy/` 下（根目录的 `.gitignore` 忽略 `data/**`，可以是指向其他磁盘的符号链接）。提交进仓库的臂登记 [fork/tokstudy/arms/tlc.toml](../arms/tlc.toml) 与 [fork/tokstudy/arms/bilingual.toml](../arms/bilingual.toml) 按相对路径指向 `data/tokstudy/<项目>/tok/<臂>/tokenizer.json`。双语项目的语料根目录沿用它自己的运行手册：`data/corpus`。

**签字。** 下列决定未签字之前按默认执行；标“必须签”的不签，对应的步骤就不能开始。

| 决定 | 默认 [建议] | 卡住的步骤 |
|---|---|---|
| TLC 的 Python 来源与许可政策（必须签） | Stack-v3-L2 python，`permissive` 且非 vendor（profile `tlc-python-permissive`）；备选 `tlc-python-all`（加 `no_license`）或 Code-L2 消融 | 第 1 步 |
| 双语 zh 来源池（B-O12，必须签） | 网页池 CCI3 + ChineseWebText + IndustryCorpus（`bilingual-zh-web`）；备选加 Tele（`bilingual-zh-apache`）或全部 8 个来源 | 第 1 步 |
| 双语平行数据（B-O12，必须签） | UN PC en-zh（UltraData 之外的例外，`bilingual-parallel-unpc`） | Bridge 数据（不在本研究内） |
| 三张表的阈值、预算、种子与扩展规则（必须签） | [prereg/tlc.toml](../prereg/tlc.toml)、[prereg/bilingual.toml](../prereg/bilingual.toml)、原生 TOK-AB 表的 [prereg/tlc-T.toml](../prereg/tlc-T.toml) 中的 [建议] 值（研究 §8.4(a)–(e)、§9.2 的阈值；M6：\|Δ\| < 3σ_Δ 时补一个种子、至多 3 个；M10：与参照相差 < 2% 时各补 3 点 LR 再判；双语按阶段的种子：判定阶段 B0 3、B4 3、按初筛排名的第一名 3、第二名 2），签在 `[signature]`，在任何 pilot 结果出来之前；孪生门的 [prereg/tlc-twin.toml](../prereg/tlc-twin.toml)、[prereg/bilingual-twin.toml](../prereg/bilingual-twin.toml) 一并签 | 第 5、6 步 |
| TLC 流的混合 | prereg 的 `[mixture]` 是 py 0.7 / en 0.3，按参照臂 N0 的 token 计（`[stream] measure = "N0"`，研究 M9 与 §3.3：70/30 按 token 定，由一个参照 tokenizer 定下文档清单，所有臂共用）；按第 1 步的备料预算约等于按字节 63/37 [估计]。另一读法是按字节 0.63/0.37（`measure = "bytes"`，依赖默认 freqtrim 的估计字节/token），签字时可改 | 第 6 步 |
| 双语研究期间的去污染 | 文档集固定为 `en_word13 + zh_han25`（不依赖 tokenizer）；B-O11 冻结之后恢复 `token13` | 第 2 步 |
| Track S 长度归一的单位 | 研究中按字节（`eval-tasks --length-unit byte`；本包的 `probe` 只按字节） | 第 7 步 |

## 1. 备料（owner 主机，`fork/.venv-corpora`）

每个 profile 先 `plan --dry-run`（读 footer 与元数据列，什么都不写，打印估计的保留字节与传输字节），再 `plan`、`fetch`（可中断续跑）、`verify`，最后用 `compose` 合成项目的语料 spec。预算是保留的文本字节 [估计，研究 §3.4、§4.5]：

| profile | `--budget-bytes` | 估计传输 |
|---|---|---|
| `tlc-python-permissive` | 2,600,000,000（2.3–2.8 GB） | 15.5–19.5 GB parquet（permissive 只占字节的 5.6–5.7%） |
| `tlc-english` | 1,500,000,000（1.4–1.6 GB） | 0.82–0.93 GB |
| `bilingual-zh-web` | 2,500,000,000（1.95–2.5 GB，3 个重复共用一个池） | 3.6–4.5 GB |
| `bilingual-en` | 2,900,000,000（2.6–2.9 GB） | 1.5–1.65 GB |
| `bilingual-parallel-unpc` | 420,000,000（约 0.1B token） | 约 0.2 GB |

```bash
C=fork/.venv-corpora/bin/python
D=data/tokstudy/corpora
$C -m surogate_corpora plan --profile tlc-python-permissive --budget-bytes 2600000000 --out "$D/plan-tlc-python.json" --dry-run
$C -m surogate_corpora plan --profile tlc-python-permissive --budget-bytes 2600000000 --out "$D/plan-tlc-python.json"
$C -m surogate_corpora fetch --plan "$D/plan-tlc-python.json" --out "$D/tlc-python"
$C -m surogate_corpora verify "$D/tlc-python"
$C -m surogate_corpora plan --profile tlc-english --budget-bytes 1500000000 --out "$D/plan-tlc-english.json"
$C -m surogate_corpora fetch --plan "$D/plan-tlc-english.json" --out "$D/tlc-english"
$C -m surogate_corpora verify "$D/tlc-english"
$C -m surogate_corpora compose --project tlc --export "$D/tlc-python" --export "$D/tlc-english" \
    --seed 0 --val-fraction 0.01 --out "$TLC_RUNS/corpus/corpus.toml"
```

- `compose` 打印按种类的保留文本字节与份额：上表的预算给出约 63/37（按字节），即按 N0 的 token 约 70/30 [估计]。研究的流（第 6 步）按 prereg 的 `[mixture]` 以 N0 的 token 计数，取最稀缺的领域决定总量：按字节的份额与 63/37 相差太多时，被丢掉的是多余领域的文档，原生 TOK-AB 的 shard 却会全部用上，所以这时先调预算重做计划，再建流。
- TLC 语料 spec 的位置决定英文文档的键（`来源名/相对路径:行号`），也就决定了划分：spec 一旦写定，只能连同它引用的目录整体移动。合成绑定程序与去污染报告按 [TLC 运行手册 G0 第 4 步](../../tlc/docs/runbook.md#g0-环境) 加进这份 spec。
- 双语：先按双语运行手册生成 Track S（`synth --out data/corpus/track_s`），把 Track S 的 D_seed 成分写成一个 `[[source]]` 文件（`name = "track-s-seed"`、`kind = "synthetic"`、`files = ["track_s/seed/*.jsonl"]`、`licence = "LicenseRef-generated"`、`licence_verified = true`），再导出到语料根目录下并合成：

```bash
C=fork/.venv-corpora/bin/python
B=data/corpus
$C -m surogate_corpora plan --profile bilingual-zh-web --budget-bytes 2500000000 --out "$B/plan-zh.json"
$C -m surogate_corpora fetch --plan "$B/plan-zh.json" --out "$B/ufw-zh-web"
$C -m surogate_corpora plan --profile bilingual-en --budget-bytes 2900000000 --out "$B/plan-en.json"
$C -m surogate_corpora fetch --plan "$B/plan-en.json" --out "$B/ufw-en"
$C -m surogate_corpora plan --profile bilingual-parallel-unpc --budget-bytes 420000000 --out "$B/plan-unpc.json"
$C -m surogate_corpora fetch --plan "$B/plan-unpc.json" --out "$B/unpc"
$C -m surogate_corpora compose --project bilingual --root "$B" --export "$B/ufw-zh-web" --export "$B/ufw-en" \
    --export "$B/unpc" --include "$B/track_s.toml" --out "$B/spec.toml"
```

- 每个导出各跑一次 `verify`（同上，这里从略）；`compose` 本身也逐个重算，任何一个字节变了都拒绝。
- 平行数据的语言识别、长度比与对齐清洗还没有实现（双语 I4 落地记录），UN PC 导出一律 en 在前。B-O12 的 en 与 zh 按 `score` 分位数匹配也还没有工具：先记录两边的 score 分布（导出行的 `meta.score`）。
- 记入实测记录：每个计划的 `plan_sha256`、估计与实际的保留字节和传输字节、`compose` 打印的份额、内容过滤器（去重、非拉丁比例）比元数据估计少保留了多少（决定以后 `margin` 取多大）。

## 2. 各项目自己的去污染、划分与 tokenizer

**TLC（`fork/.venv-tlc`）。** 按 [TLC 运行手册 G0 第 4、5 步](../../tlc/docs/runbook.md#g0-环境) 加合成来源、去污染、建 T3 套件。然后为原生 TOK-AB 的每个臂建一个 tokenizer（词表 16,384，都只在去污染后的 train 划分上；臂与判定规则见 [TLC data.md 的“Tokenizer”](../../tlc/docs/data.md#tokenizer)）：

```bash
PY=fork/.venv-tlc/bin/python
T=data/tokstudy/tlc/tok
$PY -m surogate_tlc.data.tokenizer train "$TLC_RUNS/corpus/corpus.toml" "$T/T0" --pretokenizer qwen2
$PY -m surogate_tlc.data.tokenizer freqtrim "$TLC_RUNS/corpus/corpus.toml" "$T/T1" --regex qwen --teacher "$QWEN/tokenizer.json" \
    --teacher-repo Qwen/Qwen2.5-Coder-1.5B-Instruct --teacher-revision "$QWEN_REV" --teacher-licence apache-2.0
$PY -m surogate_tlc.data.tokenizer freqtrim "$TLC_RUNS/corpus/corpus.toml" "$T/T2" --regex fine-a --teacher "$QWEN/tokenizer.json" \
    --teacher-repo Qwen/Qwen2.5-Coder-1.5B-Instruct --teacher-revision "$QWEN_REV" --teacher-licence apache-2.0
$PY -m surogate_tlc.data.tokenizer train "$TLC_RUNS/corpus/corpus.toml" "$T/T3" --pretokenizer python-digit
$PY -m surogate_tlc.data.tokenizer freqtrim-check "$T/T1" --teacher "$QWEN/tokenizer.json"
$PY -m surogate_tlc.data.tokenizer freqtrim-check "$T/T2" --teacher "$QWEN/tokenizer.json"
```

- 教师文件只在 owner 主机上取到自己的目录中，按 sha256 识别（研究用的是 revision `2e1fd397…`、sha256 前缀 `c0382117`）；`fine-a` 与 `fine-b` 在 TOK-AB 之前定一种。T3 只在 CPU 上测得的 BPM 跨边界率 ≤ 5% 时才进入 GPU 运行（第 3 步的 `align`）。
- 每个用 std::regex 回退的臂（T2、T3）先按 TLC 运行手册 G0 第 5 步过原生一致性检查（`tokenizer check --limit 10000`）。
- 原生 TOK-AB 每臂建自己的 shard，行长按字节对齐：压缩最差的臂用 512 个 token，其余臂按第 3 步实测的字节/token 取更短的 8 的倍数（[TLC training.md 的“评测”](../../tlc/docs/training.md#评测)）。

**双语（`fork/.venv-bilingual`）。** 按 [双语运行手册 G1′ 的数据准备](../../bilingual/docs/runbook.md#g1-可测性-pilot) 跑 `manifest → split assign → decontam（不给 --tokenizer，即 en_word13 + zh_han25） → split materialize`；研究期间的文档集就是这份派生 manifest，不再跑 `token13`。然后在 `seed_train` 上训练 B0–B3：

```bash
PY=fork/.venv-bilingual/bin/python
T=data/tokstudy/bilingual/tok
$PY -m surogate_bilingual tokenizer train --manifest data/splits/manifest.json --vocab 32768 --pretokenizer bytelevel --out "$T/B0/tokenizer.json"
$PY -m surogate_bilingual tokenizer train --manifest data/splits/manifest.json --vocab 16384 --pretokenizer bytelevel --out "$T/B1/tokenizer.json"
$PY -m surogate_bilingual tokenizer train --manifest data/splits/manifest.json --vocab 32768 --pretokenizer qwen2-digits --out "$T/B2/tokenizer.json"
$PY -m surogate_bilingual tokenizer train --manifest data/splits/manifest.json --vocab 32768 --pretokenizer han-isolate --out "$T/B3/tokenizer.json"
$PY -m surogate_bilingual tokenizer check "$T/B0/tokenizer.json"
```

每个都跑 `tokenizer check`（0 个 `<unk>`、0 个往返失败）。

## 3. CPU 初筛（`fork/.venv-tokstudy`）

内在指标与教师对齐只用于淘汰明显被支配的臂和解释结果，不作判定依据（[README 的“命令行”](../README.md#命令行)）。用去污染后的 train 样本（导出分片即可，它们的正文字段是 `text`）：

```bash
K=fork/.venv-tokstudy/bin/python
$K -m surogate_tokstudy intrinsic --project tlc --tokenizer T0=data/tokstudy/tlc/tok/T0/tokenizer.json@"$SHA_T0" \
    --tokenizer T1=data/tokstudy/tlc/tok/T1/tokenizer.json@"$SHA_T1" --bytes bytes \
    --docs py=data/tokstudy/corpora/tlc-python/parts/stack3-py-00000.jsonl --docs en=data/tokstudy/corpora/tlc-english/parts/ufw-en-00000.jsonl \
    --out data/tokstudy/tlc/intrinsic.json
$K -m surogate_tokstudy align --teacher "$QWEN/tokenizer.json@$QWEN_SHA" --tokenizer T3=data/tokstudy/tlc/tok/T3/tokenizer.json@"$SHA_T3" \
    --docs py=data/tokstudy/corpora/tlc-python/parts/stack3-py-00000.jsonl --out data/tokstudy/tlc/align.json
```

- 读数：每臂每领域的字节/token（第 4 步的 FLOP 匹配与原生 TOK-AB 的行长都用它），TLC 各臂对 Qwen 教师的跨边界率与非恒等全空白行占比（冻结的 tokenizer 超过 0.5% 时 BPM 默认开启全空白行屏蔽 [建议]）。

## 4. 交接：项目的 tokenizer 进入臂登记

研究包只按路径与 sha256 读取项目建好的 `tokenizer.json`（[README 的“tokenizer 交接”](../README.md#tokenizer-交接)）。

1. 把 TOK-AB 当时领先臂的 TLC tokenizer 目录复制（或链接）到 `data/tokstudy/tlc/tok/leader`；双语 B0–B3 已在第 2 步写到登记指向的位置。
2. 检查并写入 sha256。`--pin` 只改每个子词臂那一行的 `sha256`，同时核对项目自己的记录：TLC 的 `tokenizer_manifest.json` 必须记着这份 corpus spec 的 sha256 与去污染记录、训练于 train 划分；双语的 `tokenizer.meta.json` 必须记着这份派生 manifest 的 `manifest_sha256`、训练于 `seed_train`；两边都不能有 normalizer。

   ```bash
   K=fork/.venv-tokstudy/bin/python
   $K -m surogate_tokstudy arms fork/tokstudy/arms/tlc.toml --pin --tlc "$TLC_RUNS/corpus/corpus.toml" --record data/tokstudy/tlc/handoff.json
   $K -m surogate_tokstudy arms fork/tokstudy/arms/bilingual.toml --pin --bilingual data/splits/manifest.json --record data/tokstudy/bilingual/handoff.json
   ```

3. H-Net 臂的 FLOP 匹配。登记中 N1、B4 的 `[arm.model.hnet]` 是按假设的字节/token（TLC 3.8、双语 3.5）匹配的 [估计]；用第 3 步实测的参照臂字节/token（按流的混合加权）重算，并把 `ratio` 改成同一个值：

   ```bash
   K=fork/.venv-tokstudy/bin/python
   $K -m surogate_tokstudy flops fork/tokstudy/arms/tlc.toml --k-train 2048 --bytes-per-token N0="$C_N0" --match N1 --reference N0
   $K -m surogate_tokstudy flops fork/tokstudy/arms/bilingual.toml --k-train 4096 --bytes-per-token B0="$C_B0" --match B4 --reference B0
   ```

   训练开始时 `train` 在流的前 2,000 篇上实测 c_arm 并记入 `run.json`；与这里用的值相差超过 3% 时，在参照臂第一次运行之后、H-Net 臂第一次运行之前重新匹配。
4. 提交两个登记文件与签了字的 prereg（同一个签名提交）。之后改动任何一个都改变臂的 sha256 或 prereg 的 sha256：评测集与运行登记会拒绝旧的运行；prereg 的改动要 `train --amend "<原因>"`（已有运行的衰减分支用 `branch --amend "<原因>"`；评测只接受运行开始时的 prereg 或之后被锁接受的 prereg）。改了 F_target、调度、FLOP 口径或预算点的 prereg 会给出另一个运行 id，旧运行不能在它之下继续，只能重跑。没签字的 prereg 能训练（stderr 警告），但只能进 `verdict --draft`。

## 5. G0-ts、G0.5-ts 与孪生门（GPU）

1. **G0-ts。** sm_120 可见，记录 torch wheel。`triton` 路线：mamba-ssm 2.3.2.post1 的 `__init__` 无条件导入 CUDA 扩展，`MAMBA_SKIP_CUDA_BUILD=TRUE` 装出的包用不了（[README 的“模型与内核”](../README.md#模型与内核)）；要么在 sm_120 上完整编译 mamba-ssm，要么用 `torch` 后端并记录。用 `triton` 时，`kernels.ssd(backend="triton")` 在 fp32 下前向与反向都必须与 `torch` 后端相对差 ≤ 1e-4。bf16 autocast 下重复前向的 H-Net 边界相同。本包的 GPU 测试目录还没有建（S17）。
2. **G0.5-ts。** 对 N0、N1、N2、B0、B4、B5 在计划的形状、微批 {4, 8, 16} 下测字节/秒、FLOP/秒与峰值显存（`bench` 命令属于 S17，未落地；在它之前用 `train` 的 `metrics.jsonl` 中的 `bytes_per_s`、`flops_per_s` 与峰值显存）。H-Net 臂低于其子词参照的 0.25 倍、或双语臂放不下微批 × K_train 时停下并报告。重算预算（下文）。
3. **孪生门。** TW-1（A0 孪生的 logits 与 `ref_mamba2.py` 一致）已在 CPU 测试中通过。TW-2：N0 用原生的调度、TOK-AB 的数据与预算在本包中训练，Python 与英文 BPB 与原生 A0（`python -m surogate_tlc eval … --dump-nll`）相对差 ≤ 1%。TW-3：B0 经 `surogate_bilingual.train.loop`（cosine、1 个种子、0.4B）与本包中同调度的 B0，每种语言相差 ≤ 1%。门不过时该表只能内部比较（判定标为 internal-only），由 owner 决定。
   - **配方。** 孪生运行用自己的 prereg（[prereg/tlc-twin.toml](../prereg/tlc-twin.toml)、[prereg/bilingual-twin.toml](../prereg/bilingual-twin.toml)：`[schedule] kind = "cosine"`，即原生与双语生产调度按 FLOP 进度的版本；预算点只有 1.0）和自己的运行根目录（prereg 锁按表与运行根目录记，孪生的 prereg 不能与判定用的 prereg 共用一个根目录）。TW-2 的 prereg 没有 `[mixture]`：流读原生 train 划分的全部文档，与原生 shard 相同；F_target、K_train 与 warmup 是 [待定]，按原生领先臂那次运行的训练 token 数、shard 行长（`seq_len` × 实测字节/token）与 `warmup_steps / max_steps` 填好再签字。TW-3 的混合与判定用的相同（B0 token 计的中英平衡）。

     ```bash
     K=fork/.venv-tokstudy/bin/python
     TW=data/tokstudy/twin-runs
     $K -m surogate_tokstudy stream --prereg fork/tokstudy/prereg/tlc-twin.toml --tlc "$TLC_RUNS/corpus/corpus.toml" --out data/tokstudy/tlc/twin-stream
     $K -m surogate_tokstudy train --arms fork/tokstudy/arms/tlc.toml --arm N0 --seed 0 --stream data/tokstudy/tlc/twin-stream \
         --prereg fork/tokstudy/prereg/tlc-twin.toml --runs "$TW" --lr "$LR_NATIVE"
     $K -m surogate_tokstudy eval score "$TW/tlc-N/N0-s0-$RUN_ID" --evalset data/tokstudy/tlc/evalset --out data/tokstudy/tlc/twin/N0.nll
     # the native side: the TOK-AB run of the same tokenizer, scored by TLC's own eval
     PY=fork/.venv-tlc/bin/python
     $PY -m surogate_tlc eval "$TLC_RUNS/models/T_LEADER" "$TLC_RUNS/data" --dump-nll data/tokstudy/tlc/twin/native.nll --arm T_LEADER --corpus "$TLC_RUNS/corpus/corpus.toml"
     ```

     双语同理：`--prereg fork/tokstudy/prereg/bilingual-twin.toml --arm B0 --measure fork/tokstudy/arms/bilingual.toml:B0`，另一边是双语运行手册的生产循环（0.4B、1 个种子、同一个 tokenizer），用 `eval-tasks`/`eval` 的 `--dump-nll` 写同一种 NLL 文件 [待定：双语生产循环的按文档 NLL 输出未落地，先比较两边各自报告的按语言 BPB]。比较两边每个领域的 BPB，相对差 ≤ `twin_rel_max`（1%），结果按主机、提交、日期记入实测记录。
   - TW-2 的两边评测上下文不同：原生按 512 token 的窗口给整篇文档打分，本包按评测集的公共字节前缀（K_eval）一次打分；差值里含有这部分 [待定]，记录两边各自的文档数与字节数。

## 6. 训练流、评测集与运行

```bash
K=fork/.venv-tokstudy/bin/python
$K -m surogate_tokstudy stream --prereg fork/tokstudy/prereg/tlc.toml --tlc "$TLC_RUNS/corpus/corpus.toml" \
    --measure fork/tokstudy/arms/tlc.toml:N0 --out data/tokstudy/tlc/stream
$K -m surogate_tokstudy eval prefix --arms fork/tokstudy/arms/tlc.toml --prereg fork/tokstudy/prereg/tlc.toml --tlc "$TLC_RUNS/corpus/corpus.toml" --out data/tokstudy/tlc/evalset
$K -m surogate_tokstudy stream --prereg fork/tokstudy/prereg/bilingual.toml --bilingual data/splits/manifest.json \
    --measure fork/tokstudy/arms/bilingual.toml:B0 --out data/tokstudy/bilingual/stream
$K -m surogate_tokstudy eval prefix --arms fork/tokstudy/arms/bilingual.toml --prereg fork/tokstudy/prereg/bilingual.toml \
    --bilingual data/splits/manifest.json --out data/tokstudy/bilingual/evalset
```

- **评测集只用自然文本。** `eval prefix` 默认去掉生成来源的留出文档：TLC spec 中 `generated = true` 的合成绑定程序，双语 kind 为 `synthetic` 的 Track S 成分（研究 §3.5：评测是同分布的 Stack-v3 留出仓库与自然文本）；训练流照常包含它们。清单的 `source.generated_excluded` 记下被排除的来源名；`--include-generated` 只用于另作的读数，不进判定。
- **LR 扫描（M10）。** 每个结构族（子词、平坦字节、H-Net）3 个学习率 × 10% 的 F_target × 1 个种子，之后固定；H-Net 另扫 `--lambda-outer` ∈ {1, 2, 3}。扫描的运行是另一个运行 id（`--lr` 进入配置哈希），计入判定的试过的配置数。挑战臂与参照相差 < 2% 时各补 3 点再判。
- **tlc-N。** N0 × 3、N1 × 3、N2 × 1，F_target 为 A0 在 0.8B token 处（1.642e17），每个运行取 0.25、0.5 两个衰减分支。
- **bilingual-B。** 初筛：B0 × 3、B1–B4 × 2、B5 × 1，`--stage screen`（F_target = F_dec，停在 0.5 的稳定检查点，取 0.25、0.5 两个分支）。判定：B0 与至多两个入围臂（B4 只要不被淘汰就入围），以 `--stage decision` 继续同一个运行到 F_target。

```bash
K=fork/.venv-tokstudy/bin/python
R=data/tokstudy/runs
$K -m surogate_tokstudy train --arms fork/tokstudy/arms/tlc.toml --arm N1 --seed 0 --stream data/tokstudy/tlc/stream \
    --prereg fork/tokstudy/prereg/tlc.toml --runs "$R" --lr "$LR_HNET" --lambda-outer "$LAMBDA"
$K -m surogate_tokstudy branch "$R/tlc-N/N1-s0-$RUN_ID" --at 0.5
$K -m surogate_tokstudy eval score "$R/tlc-N/N1-s0-$RUN_ID" --point 0.5 --evalset data/tokstudy/tlc/evalset --out data/tokstudy/tlc/nll/N1-s0-E0.5.nll
$K -m surogate_tokstudy train --arms fork/tokstudy/arms/bilingual.toml --arm B2 --seed 0 --stream data/tokstudy/bilingual/stream \
    --prereg fork/tokstudy/prereg/bilingual.toml --runs "$R" --lr "$LR_SUBWORD" --stage screen
```

- 原生 TOK-AB 的每个检查点用 `python -m surogate_tlc eval MODEL DATA --dump-nll FILE --arm T1 --corpus "$TLC_RUNS/corpus/corpus.toml"` 写同一种 `tokstudy-nll/1`（表 `tlc-T`、后端 `native`）；按笔记规则 11，它永不与 `tlc-N` 同表。
- **臂的增减。** N0b（原生第二名的孪生）与 N1-4（4+4 的 H-Net）是条件臂：加进登记会改变表的臂集合，评测集要重建（公共前缀随臂集合变）：`eval score` 拒绝臂集合（id 与 `arm_sha256`）与运行当前登记不同的评测集，所以每个运行都要在新评测集上重新打分（训练不必重跑：臂的 sha256 不因别的臂改变）；`verdict` 拒绝同一张表里来自不同评测集（`evalset_sha256` 不同）的 NLL 文件。

## 7. 判定与冻结

**判定**（`verdict`，S16；规则见 [README 的“统计与判定”](../README.md#统计与判定)）。每张表一次，读该表签了字的 prereg 与全部 NLL 文件；没签字的 prereg 被拒绝（`--draft` 只给草稿）。

```bash
K=fork/.venv-tokstudy/bin/python
R=data/tokstudy/runs
# 孪生门（第 5 步）：TW-2 / TW-3 的记录
$K -m surogate_tokstudy verdict twin --prereg fork/tokstudy/prereg/tlc-twin.toml --tokstudy data/tokstudy/tlc/twin/N0.nll \
    --production data/tokstudy/tlc/twin/native.nll --out data/tokstudy/tlc/twin.json
# tlc-N：N1 晋升与否（证据文件写 owner 的集成签字；吞吐从 --runs 的 metrics 读）
$K -m surogate_tokstudy verdict --prereg fork/tokstudy/prereg/tlc.toml --nll data/tokstudy/tlc/nll/*.nll --runs "$R" \
    --twin data/tokstudy/tlc/twin.json --evidence data/tokstudy/tlc/evidence.toml --out data/tokstudy/tlc/verdict.json
# 双语：先初筛，再用初筛的 shortlist 判定（Track S 探针每个运行一份）
$K -m surogate_tokstudy probe "$R/bilingual-B/B2-s0-$RUN_ID" --bilingual-items data/corpus/track_s/items/E1.dev.jsonl --bilingual-items data/corpus/track_s/items/E3ctx.dev.jsonl --out data/tokstudy/bilingual/probe/B2-s0.probe
$K -m surogate_tokstudy verdict --prereg fork/tokstudy/prereg/bilingual.toml --stage screen --nll data/tokstudy/bilingual/nll/*E0.5.nll \
    --runs "$R" --twin data/tokstudy/bilingual/twin.json --out data/tokstudy/bilingual/screen.json
$K -m surogate_tokstudy verdict --prereg fork/tokstudy/prereg/bilingual.toml --stage decision --nll data/tokstudy/bilingual/nll/*.nll \
    --probe data/tokstudy/bilingual/probe/*.probe --runs "$R" --twin data/tokstudy/bilingual/twin.json \
    --evidence data/tokstudy/bilingual/evidence.toml --out data/tokstudy/bilingual/verdict.json
# 原生 TOK-AB（tlc-T）：TLC 的 eval --dump-nll 文件；BPM 跨边界率来自第 3 步的 align
$K -m surogate_tokstudy verdict --prereg fork/tokstudy/prereg/tlc-T.toml --nll data/tokstudy/tlc/native-nll/*.nll \
    --evidence data/tokstudy/tlc/native-evidence.toml --out data/tokstudy/tlc/tok-ab.json
```

证据文件（TOML 或 JSON）只放本包的文件里没有的东西：`integration_signoff`（owner 接受集成清单）、`throughput`（不给 `--runs` 时的吞吐比）、`bpm_cross_boundary`（各 own-BPE 臂的跨边界率）、`probe_improved`、`evbind_improved`、`lr_resweep_done`（M10 补扫过的臂）、`shortlist`（双语判定阶段：初筛结果的 `decision.shortlist`）。判定结果为 `extend` 时按其理由补种子（M6）、补 LR 点（M10）或延长（走向缩小），再重跑判定；`tie` 归参照。

**冻结（交接的反方向）。** 研究包从不写进项目；获胜臂的 tokenizer 本来就是项目自己建的文件，冻结是项目的步骤，`handoff.json` 记下它必须有的 sha256：

| 项目 | 冻结什么 | 怎么核对 | 截止 |
|---|---|---|---|
| TLC（子词之间） | 原生 TOK-AB 的胜者（G-TOK-a，[TLC data.md 的“Tokenizer”](../../tlc/docs/data.md#tokenizer)） | 四个 P0 行的模型目录与 shard 都由这一个 tokenizer 目录建成：shard manifest 与 `tlc_meta.json` 记着同一个 `tokenizer_sha256` | Phase 1b 之前 |
| TLC（神经臂） | N1 只在 prereg 的五个条件都满足、且 owner 选定集成路线（原生新 op，或全部行改到 PyTorch）之后晋升；在那之前 TLC 仍用子词，G-TOK-b 不关闭 | `verdict` 的结果（S16） | Phase 1b 之前 |
| 双语 | `verdict` 选出的臂（默认 B0）的 `tokenizer.json`：复制到双语运行手册使用的 `data/tokenizer/tokenizer.json`，之后每个运行都用它；B-O11 冻结后恢复 `token13` 去污染 | `sha256sum data/tokenizer/tokenizer.json` 等于 `data/tokstudy/bilingual/handoff.json` 中该臂的 sha256；它进入每个运行 id 与种子指纹 | G1′ 与 S_I 之前 |
| 双语（B4 胜出） | 分块器属于 θ_S、在种子训练结束时冻结，`hnet.py`、`mamba2.py` 整文件移入 `fork/bilingual`，导入边界按 owner 的选项改 | spec 级改动，owner 签字 | 仅在 B4 胜出时 |

## 预算 [估计]

GPU 小时 = Σ 运行 × 1.075 × F_target /(209.5e12 × 实测 MFU)/ 3600，再加评测（实测）。研究 §11 给出 tlc-N 约 10–40 GPU 小时、bilingual-B（标准方案）约 24–82 GPU 小时；原生 TOK-AB 加 F3 约 3–32 GPU 小时（TLC 自己的预算）。G0.5-ts 之后按实测的 MFU 重算，记入实测记录。

| 步骤 | 结果 | 主机 | 提交 | 日期 |
|---|---|---|---|---|
| 第 1 步：备料与 `compose` | 未运行 | | | |
| 第 2 步：去污染、划分、各臂 tokenizer | 未运行 | | | |
| 第 3 步：CPU 初筛 | 未运行（20 行样本上的软件检查见 [README 的实测记录](../README.md#实测记录本地不进-ci)） | | | |
| 第 4 步：`arms --pin`、FLOP 匹配、签字提交 | 未运行 | | | |
| 第 5 步：G0-ts、G0.5-ts、TW-2、TW-3 | 未运行 | | | |
| 第 6 步：LR 扫描、两张表 | 未运行 | | | |
| 第 7 步：判定与冻结 | 未落地（S16） | | | |
