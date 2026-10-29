# 语料、tokenizer 与数据 shard

相关模块：`surogate_tlc.data.corpus`（语料 spec、文档、train/val 划分与 provenance）、`surogate_tlc.data.decontam`（对 HumanEval、MBPP、EvalPlus 去污染）、`surogate_tlc.data.synth`（合成绑定任务）、`surogate_tlc.data.tokenizer`（训练 tokenizer，以及与原生 tokenizer 的一致性检查）、`surogate_tlc.data.shards`（BIN.TOK v3 shard 与 manifest）。探针套件的条目格式和 T3 自然代码套件见 [training.md 的“绑定探针”](training.md#绑定探针)。这些模块都不需要原生扩展，只有 tokenizer 的 `check` 命令会加载原生 tokenizer；每个写文件的工具都是 `python -m surogate_tlc.data.<模块>`。

## 语料

语料由一个 TOML spec 描述：顶层是划分参数，每个 `[[source]]` 是一个来源及其 provenance。

```toml
seed = 0               # 划分用的种子
val_fraction = 0.01    # 进入 val 的文档比例
decontam = "decontam.json"   # 可选：去污染报告，相对 spec 所在目录（见下文“去污染”）

[[source]]
name = "stack-python"  # 各来源不同名
kind = "code"          # code 或 text
files = ["python/*.jsonl"]   # glob，相对 spec 所在目录，每个至少匹配一个文件
url = "https://huggingface.co/datasets/..."
revision = "..."       # 可选
license = "..."
downloaded = 2026-10-20
stage = "M1"           # 可选
generated = false      # 可选：fork 工具生成的来源（合成绑定程序）写 true
filters = ["..."]      # 下载之后、入库之前做过的过滤；可以为空列表
```

- 缺少必需键、出现未知键、glob 匹配不到文件时报错。
- `.jsonl` 文件每行一个文档，正文在 `text` 字段；可选的 `group`（例如仓库名）让同组的文档落在同一划分。其他文件整个是一个文档。
- 文档的键（`来源名/文件` 或 `来源名/文件:行号`）与 manifest 中的文件路径都相对 spec 所在目录，spec 目录之外的文件写成以 .. 开头的相对路径（运行手册中合成来源的 train.jsonl 就在 spec 目录之外）。键决定划分（没有 `group` 时）、shard 文件的分配和去污染报告的条目，所以整个目录树换个位置（另一台主机、改名的 `$TLC_RUNS`）得到同样的划分和逐字节相同的 shard；只移动 spec 而不移动它引用的文件则不然。
- 入库规则：换行按 Python 的通用换行统一为 `\n`（`\r\n` 和单独的 `\r`），之后为空的文档和不是合法 UTF-8 的文档（包括 JSON 中单独的代理项转义，如 `"\ud800"`）被丢弃并计数。后续的标签按 UTF-8 字节偏移对齐时，看到的就是模型训练的文本。
- 划分：sha256(seed, group) 按 2⁶⁴ 的比例小于 `val_fraction` 的文档进 val；没有 `group` 的文档自成一组。tokenizer 只用 train 划分训练，shard 按同一划分写出。
- `generated = true` 的来源照常参与 tokenizer 和 shard，但 T3 自然代码套件不从它取探针项：T3 测的是自然代码，合成程序有自己的 T1、T2。
- 文本文档中 markdown 代码块的判定只有一处实现（`surogate_tlc.data.corpus.fences`）：去掉缩进后以 ``` 开头的行打开代码块，代码块内恰为 ``` 的行关闭它，文档结束时仍未关闭的代码块在末尾关闭；行只在 `\n` 处结束（U+2028 等不算）。shard 的框定和去污染的代码扫描都用它，所以去污染当作代码扫描的，正是训练时框成代码的部分。
- 配比（70% Python、30% 英文）在准备输入文件时决定；shard manifest 按来源记录实际的 token 数和 train 中的占比，供核对。

## 去污染

```bash
python -m surogate_tlc.data.decontam CORPUS.toml BENCHMARK_DIR CORPUS_DIR/decontam.json --plant 40 --jobs 8
```

在训练 tokenizer 之前，`surogate_tlc.data.decontam` 按四条规则为语料的每个文档生成候选：

| 规则 | 内容 | 结果 |
|---|---|---|
| `code13` | 基准解答的信息性 13-gram 中，在文档里出现的比例（覆盖率）。代码先做标识符抽象：标识符记为 ID，字符串与数字字面量记为 STR、NUM，去掉注释和 docstring。13-gram 中至少 4 个是词（关键字、内建名、ID）才算信息性，这样只含字面量的列表不会处处命中。信息性 13-gram 少于 15 个的条目不建索引 | ≥ 0.5 标记，≥ 0.8 删除 |
| `text8` | 基准描述（MBPP 的 text，HumanEval 的 docstring）的小写词 8-gram 覆盖率，条目至少有 5 个 8-gram | ≥ 0.5 标记，≥ 0.8 删除 |
| `ast` | 文档中某个函数的 AST 在 α-重命名、字面量归一、去掉 docstring、装饰器与注解（带注解的赋值变成普通赋值）之后与基准函数相同（`surogate_tlc.data.decontam.ast_hash`）。α-重命名覆盖绑定的名字：变量、形参、`except ... as`、`import ... as`、`global`/`nonlocal`，以及调用同一段代码中定义的函数、具名 lambda 或类（其 `__init__`）时的关键字实参名（它们指向形参）；其他关键字实参名（`sorted(key=...)`、`print(end=...)`）是 API 名，保持不变，所以只改局部变量名不改变哈希。属性名和被导入的模块名不变。少于 40 个节点的函数会处处相同，不比较 | 删除 |
| `name` | 文档定义的函数与基准的独特入口函数同名：至少 10 个字符、带下划线、不是 main、solve 之类的通用名 | 只标记 |

- 只有高精度的证据才删除文档：`ast` 命中，或 `code13`、`text8` 的覆盖率达到删除阈值。误删会悄悄减少训练数据，误标只多一次人工查看，所以只被标记的文档留在语料中，供抽样人工审计。候选按函数和 n-gram 生成，删除以整个文档为单位。
- 文本文档按文本扫描，其中的 markdown 代码块（`fences()`）另按代码扫描。
- 基准文件：`BENCHMARK_DIR` 中所有 `*.jsonl`、`*.jsonl.gz`、`*.json`。支持两种格式：HumanEval 与 HumanEval+（`prompt` 是签名加 docstring，`canonical_solution` 是函数体），MBPP、MBPP+ 与 sanitized MBPP（完整代码加描述）。MBPP 要放入全部 974 个 id（`mbpp.jsonl`，含 prompt 划分 1–10）：要报告 MBPP+，这是前提。来源与许可：HumanEval 来自 openai/human-eval（MIT），MBPP 来自 google-research（CC-BY-4.0），HumanEval+ 与 MBPP+ 来自 evalplus 的 release（Apache-2.0）。
- 报告 `decontam.json`（`tlc_decontam_version` 为 3：关键字实参的 α-重命名改成只认本段代码中定义的被调函数，spec 目录之外的文件的键也改成了相对路径，旧报告的 AST 哈希与键不再可比，语料 spec 拒绝读取）的内容：
  - 版本，语料 spec 的 sha256，每个基准文件的 sha256 和条目数，各规则及其阈值；
  - 各来源扫描的文档数；
  - `removed` 与 `flagged`：每个文档的键、来源、正文的 sha256 和全部命中；
  - 从中抽取的 20 个待审计文档；
  - 植入校准的结果。
- 使用：在语料 spec 顶层写 `decontam = "decontam.json"`。此后训练 tokenizer 和构建 shard 都丢弃报告中删除的文档，按来源记作 `decontaminated`；文档正文的 sha256 与报告不符时报错，要求重新扫描。tokenizer 与 shard 的 manifest 都在 `corpus.decontam` 中记下报告的路径、sha256 和删除数。扫描本身忽略 spec 中的这个键。
- 植入校准（G6）：`--plant 40` 取 40 个基准解答，做 α-重命名（标识符改成 `zz_` 加倒序名，属性名不改），追加到随机抽取的自然代码文档末尾。宿主不取 `generated = true` 的来源：合成程序短、从不命中基准、总能解析，用作宿主会压低记下的误报率与标记率，抬高 `ast` 的召回（它们照常被扫描）。报告给出删除规则的召回率、标记的召回率和各规则的召回数，未改动的宿主文档被删除（误报）和被标记的比例，以及宿主取自哪些来源（`host_sources`）。在库代码上的一次校准见 [measurements.md](measurements.md)；英文网页上 `text8` 的误报要在 G0 用实际语料的报告抽样核对。

## 合成绑定任务

```bash
python -m surogate_tlc.data.synth OUT_DIR --seed 0 --per-task 4000 --train-chars 2000000
```

`surogate_tlc.data.synth` 生成静态 Python 子集上的绑定程序。程序只用赋值、函数定义与调用和 `assert`，值都是不透明的 4 字符字符串字面量（如 `'k3f9'`），没有算术，所以测的是绑定而不是执行代码的能力。程序以读出行 `assert EXPR == 'value'` 结束：生成时先执行程序，取得 EXPR 的值并写进读出行，再执行一次核验，不靠人工标注。每个函数定义之后紧接着调用它，所以函数内的读出看到的正是它前面文本中的绑定。

- **T1 值追踪**：一条长 `hops` 的拷贝链把字面量传给被读的名字，链被重绑 `overwrites` 次（换新的字面量），另有 `live − hops` 个名字绑定干扰字面量，随机分散在链的前后。作用域模式如下：

  | 模式 | 结构 |
  |---|---|
  | `flat` | 全在模块级 |
  | `shadow-in` | `depth` 层嵌套函数各自重绑该名字，在最内层读；链是被遮蔽的外层值 |
  | `shadow-out` | 同上，但在这些调用之后回到模块级读 |
  | `closure` | 链在一个外层函数中，`depth` 层嵌套函数读它（闭包） |
  | `shadow-closure` | 外层函数中先有 `depth` 层函数遮蔽该名字，之后另一个闭包读外层的值 |
  | `call-pos`、`call-kw` | 值经一个三参数函数的形参（按位置或按关键字）传入，再从返回值读出 |

  网格：`live` ∈ {2, 4, 8, 16, 24, 32, 64}，`hops` ∈ {1, 2, 3, 4}（不超过 `live`），`overwrites` ∈ {0, 1, 2, 3, 4}，作用域模式另有 `depth` ∈ {1, 2}。`live` 的 2–16 是主层、24–64 是压力层；`overwrites` 取实验笔记与绑定显著性研究各自要求的层级的并集。
- **T2 形实参对应**：读出 `assert f(实参) == ` 的值，即被调函数返回的那个形参收到的值。除被调函数外，还有 `distractors` 个形参同名、顺序不同的函数，以及 0–3 个不传入的无关绑定。调用模式：`positional`；`keyword`（按形参顺序）；`mixed`（前几个按位置，其余按关键字）；`default`（后几个形参有默认值，调用时省略一部分）；`names-reordered`（模块变量与形参同名，按另一顺序按位置传入）；`keyword-reordered`。网格：`formals` ∈ {2, 3, 4, 6}，`distractors` ∈ {0, 1, 2}，`actuals` ∈ {literal, name}。
- **划分**：训练流与评测项在四个轴上互不相交，这样评测上的提升不可能来自见过题目的表面形式。

  | 轴 | 训练流 | 只在评测中出现 |
  |---|---|---|
  | 模板族（表面写法） | T1 的 `assign` 与 `unpack`（相邻的字面量绑定合成元组赋值）；T2 的 `return` 与 `local`（经局部变量返回） | T1 的 `annotated`（`name: str = ...`）；T2 的 `lambda` |
  | 作用域模式（组合格） | 其余模式 | `shadow-closure`（闭包内的遮蔽）和 `keyword-reordered`（关键字重排的调用）。其中的原子（遮蔽、闭包、关键字传参、重排）都在训练流中出现 |
  | 标识符池 | `seen`：36 个代码词素，以及它们两两组合中的约 80%（共 1,062 个） | `novel`：其余 234 个组合，即由已见词素组成的新串；`rare`：只用 j、k、q、v、w、x、z 的 3–4 字母串（2,744 个），训练流中没有这样的标识符 |
  | 字面量 | sha256 桶 1–3 | sha256 桶 0，形状相同 |

  此外，任何训练程序的 AST（`ast_hash`，与去污染的 `ast` 规则相同，调用程序中定义的函数时的关键字实参名也参与 α-重命名）都不与评测项相同：撞上就在同一格内重抽，最多 20 次，仍然撞上就放弃这一格。`check()` 从训练流的文本重新计算每个程序的 AST 哈希（不论它有没有对应的记录），逐项核验：训练流中没有留出的模板族、模式、标识符或字面量，没有评测项的 AST，dev 与 test 没有相同的程序；有违反时抛出列出全部违反的 `SplitLeak`。`build()` 把核验结果（含撞上训练 AST 的评测项数，应为 0）写进 manifest。`tlc_synth_version` 为 2：关键字实参名加入 α-重命名之后，版本 1 的训练流与评测项之间有 AST 相同的程序。
- **评测项**：每个任务每半至少 `--per-task` 项，网格的每格项数相同。每项随机取模板族和标识符池，所以每项记着自己被留出的轴（`strata.heldout`）。dev 与 test 是同一分布的两半，程序不重复：dev 用于选层，test 只在预注册的里程碑读取。
- **训练流** `train.jsonl`：每行一个文档，由连续若干程序组成一个约 1,600 字符的模块，程序之间空一行；它像自然源文件一样定义并调用多个函数，按 512 token 切窗时大多落在一行内。整个模块再执行一次核验。T1 与 T2 交替出现，格在训练网格上均匀抽取，只用 `seen` 池和训练字面量。它作为一个 `kind = "code"`、`generated = true` 的来源加进语料 spec，与其他来源一起训练 tokenizer、构建 shard。配比（暂定 5%）由 `--train-chars` 决定，约为其余来源的字符数 × 0.05 / 0.95。P0 用固定配比，没有 ramp；按 A0 的成功率标定配比属于 Phase 1。
- 输出：`dev.jsonl`、`test.jsonl`（`surogate_tlc.data.items.Item`）、`train.jsonl` 和 `manifest.json`。manifest 记录网格、留出的轴、各文件的 sha256、训练流中与评测 AST 相撞后重抽和放弃的次数，以及 `check()` 的结果。同样的参数得到逐字节相同的文件。规模与用时见 [measurements.md](measurements.md)；P0 的 512 token 放不下 `live` ≥ 32 的大部分 T1 项，探针把它们计为 `too_long`，每格都放得进 2048 token。
- 有待 owner 定的两点：
  - 留出的组合格（`shadow-closure`、`keyword-reordered`）现在 dev 与 test 两半都有：实验笔记一处说它们只出现在 test，另一处又要求 dev/test 是同一分布的两半。只放 test 时，dev 上就不能按这些格选层。
  - 实验笔记的交换/置换子集（变量值互换后读出）还没有实现，绑定显著性研究中的以下部分也没有：标识符末片与同函数其他标识符共享的分层、改名孪生对照、值打乱对照、按 seed 的锁定步数，以及 copy-5 等过滤后的 NLL 类别。

## Tokenizer

```bash
python -m surogate_tlc.data.tokenizer train CORPUS.toml OUT_DIR --pretokenizer qwen2 --vocab-size 16384
```

- 16384 个 id 的字节级 BPE，没有 normalizer，decode(encode(x)) == x。特殊 token 的 id 固定为 0–4：`<|pad|>`、`<|bos|>`、`<|eos|>`、`<|code_start|>`、`<|code_end|>`，其中前三个与 `hf.default_config()` 的 pad、bos、eos 一致。语料不够学满词表时报错。
- 预分词由 `--pretokenizer` 选择，默认 `qwen2`：

  | 名称 | 规则 | 出处 | 原生 tokenizer 的实现 |
  |---|---|---|---|
  | `qwen2`（默认） | Qwen2 的 split 正则：GPT-2 一族，字母和数字用 Unicode 类 `\p{L}`、`\p{N}`，每个数字单独成片 | 教师引导训练的备忘录：与 Qwen 子集词表配用，数字切分与 Qwen 教师一致 | 手写 splitter |
  | `gpt2` | GPT-2 正则，连续数字成一片 | 实验笔记的候选 | 手写 splitter |
  | `python` | 实验笔记的 pyaware2 改用 `\p{L}`、`\p{N}`：标识符（含 `_`）、常见的多字符运算符、换行连同其后的缩进、整个数字（含小数和指数）各成一片 | 实验笔记的候选 | std::regex 回退 |

  - 还没有选定，owner 在 A0 之前定下，四行共用同一个 tokenizer。实验笔记的选择规则看 Python 加英文的总 token 数、跨词素率和多角色率：比较总 token 数时，用各候选分别建 shard，对比 manifest 中每个来源的 `characters` 与 `real_tokens`；跨词素率和多角色率要用 P1 的对齐与角色标注来算。
  - 另两个候选没有实现，同样待 owner 定：实验笔记的第三个候选"严格按 Python 词法切分"不是正则预分词，原生 tokenizer 推理时做不到；教师引导训练的备忘录要的 Qwen 子集词表（按 TLC 语料频率从 Qwen 的 BPE 中选 16,384 个 token，对 merge 父项封闭，让 Qwen 教师能直接在学生的 token 上打分）不是训练出来的 BPE，要另写一个构建器，`train_tokenizer` 做不出来。选它的话，要在 A0 之前加上这个构建器。
  - `model_type` 为 `tlc` 时，原生 tokenizer 从 `tokenizer.json` 读取 split 正则，没有时用 GPT-2 的。`qwen2` 与 GPT-2 的正则各有手写实现；其他正则（包括 `python`）走 std::regex 回退，用之前必须通过下面的一致性检查。CPU 的 pin 测试钉住这几点。
- 编码：文档正文按普通文本编码，其中的特殊 token 字符串不会变成特殊 id（原始文本永远不产生特殊 token）；prompt 按带特殊 token 的方式编码。
- chat template：`<|bos|>` 加各条消息的内容，不要求生成提示时最后加 `<|eos|>`。prompt 就是文档的开头，与 shard 的框定一致。
- 输出目录：`tokenizer.json`、`tokenizer_config.json` 和 `tokenizer_manifest.json`（配方、`tokenizer_sha256`、语料 spec 的 sha256、各来源的 provenance 和参与训练的文档数）。tokenizer 的身份是 `tokenizer.json` 的 sha256（`surogate_tlc.data.tokenizer.tokenizer_sha256()`），shard 的 manifest 也记录它。同样的语料与参数训练出逐字节相同的文件。
- 一致性检查：`compare()` 比较本 tokenizer 与原生 tokenizer 的特殊 id 和词表大小，并逐段比较普通编码、带特殊 token 的编码，以及两侧对普通编码 id 的解码。文本是固定的边界用例（`EDGE_CASES`：CRLF、单独的 CR、tab、CJK 标识符、emoji、f-string、字符串中的 U+2028、特殊 token 字符串等）加语料样本。CPU 测试用替身检查 harness 能发现各类差异；与真实原生 tokenizer 的比较在 GPU 主机上进行（G0 的 `check` 命令与 `fork/tlc/tests/gpu/test_data_native.py`）。原生 tokenizer 在 CPU 上编译需要 CMake 拉取的头文件依赖（fmt、nlohmann/json、minja），fork CI 的 CPU lane 没有它们，所以这一比较放在有原生构建的主机上；上游同步时，pin 测试盯住 split 正则的名单和手写 splitter 覆盖的正则。

## 数据 shard

```bash
python -m surogate_tlc.data.shards CORPUS.toml TOKENIZER_DIR OUT_DIR --seq-len 512 --train-files 1
```

- 输出 `train-00000.bin` 等、`val-00000.bin`，最后写 `manifest.json`。`OUT_DIR` 必须不存在或为空；同样的输入得到逐字节相同的输出。构建失败时删掉自己写的文件，修好之后同一条命令可以重跑；被强行终止的构建留下隐藏的 `.*.bin.tmp`，“is not empty” 的报错列出它们，删除 `OUT_DIR` 后重跑。命令结束时打印各划分的统计，以及单进程构建（编码、框定、写出）所用的 CPU 秒和每 CPU 秒的真实 token 数，即数据流水线的单核速率（G0.5）。
- 格式：BIN.TOK v3（1024 字节的头、int32 token、int32 position、按位存储的 mask），头中标为 non-overlapping。写出的文件与上游 `TokenizedDataFileWriter(masking=True, non_overlapping=True)` 的逐字节相同（CPU 测试）；上游模块 import 时就需要原生扩展，所以 fork 自己写。读取时文件头用上游的 `surogate.distill.sidecar.read_token_shard_header`。
- 每行一个文档，不跨文档打包：`mamba_ssm_scan` 在打包的文档之间不复位状态。A1、A1h 也这样做，保证各行的数据语义相同。
  - 框定：代码文档为 `<|bos|> <|code_start|> 正文 <|code_end|> <|eos|>`；英文文档为 `<|bos|> 正文 <|eos|>`，其中 `fences()` 的 fence 行换成 `<|code_start|>`、`<|code_end|>`。
  - 窗口：超过 `seq_len` 的文档切成 `seq_len` 个 token 的窗口，相邻窗口共享一个 token，因此除文档的首个 token 外，每个 token 恰好作一次目标。每个窗口是一行，末尾用 `<|pad|>` 补齐；每行的 position 为 0..seq_len−1。
  - mask：输入位置 i 的 mask 位为 1，当且仅当它的目标（第 i+1 个 token）是同一行中的真实 token。原生 DataLoader 对 non-overlapping 文件恰好给出这些行，并把每行最后一个位置的目标置为 −100（`surogate_tlc.data.shards.IGNORE`）。
- ValidTokenCount：有 n 个真实 token 的行有 n−1 个有效目标。manifest 按 shard 和按划分记录 `rows`、`tokens`、`real_tokens`、`valid_targets`（一个 epoch 中 ValidTokenCount 之和）和 `pad_fraction`；每个 shard 另记各来源占的行区间（`sources`：文档按来源依次写出，每个来源在一个文件中是连续的一段），`eval` 据此按来源给出 loss。按来源还记录文档数、字符数、行数、真实 token 数、train 中的 token 占比、丢弃计数和 provenance（URL、revision、license、下载日期、过滤规则、是否生成，以及每个输入文件的大小与 sha256）；`rules` 列出入库与框定规则。
- 身份：数据的身份是 `manifest.json` 的 sha256（`surogate_tlc.data.shards.manifest_sha256()`），与 `tokenizer_sha256` 一起写进运行记录、checkpoint 的 `tlc_meta.json` 和评测输出。训练驱动在开始前核对 shard 的 `seq_len` 等于训练配置的 `sequence_len`，manifest 中的 tokenizer sha256 等于模型目录中 `tokenizer.json` 的 sha256（`check_tokenizer()`）。
- 原生 DataLoader 依次读取各文件，只在文件内打乱行，所以 train 文档按 hash 分到 `--train-files` 个文件，每个文件都是各来源的随机样本。单个文件少于 2³¹ 个 token（含 pad），所以每约 15 亿个真实 train token 用一个文件；超出时构建在写到上限处报错。
- `surogate_tlc.data.shards.served_rows()` 按原生 DataLoader 的读取规则给出每行的输入、目标与 position。CPU 测试用它核对上面的窗口与 ValidTokenCount；GPU 主机上的测试核对它与原生 DataLoader 一致，并核对从保存的位置恢复的 DataLoader 逐位接着读下去。
- padding 浪费（`pad_fraction`）超过约 25% 时，再考虑实现文档复位并改用打包 shard，所有行同时切换。
- 上下文：P0 用 512。正式训练用 2048 还是 512 由 owner 定：用 2048 时 shard 以 `--seq-len 2048` 重建，并把 `surogate_tlc.hf.CONTEXT_LENGTH` 改为 2048（config 检查、tokenizer 的 `model_max_length` 和 shard 的默认行长都取自它）。
