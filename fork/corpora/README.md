# fork/corpora：固定版本的语料适配器

本包把 Hugging Face Hub 上固定 revision 的 UltraData（以及 UN PC）parquet shard 按 HTTP Range 逐个 row group 读出、过滤，导出为 TLC 与双语项目各自的语料格式，并写下 provenance。owner 的决定（原文）：「分詞器得對比效果要不然就搞另類神經網絡分詞 語料用minicpm的ultradata那些的」；本包负责其中“语料用 UltraData”的数据侧。

它是叶子包：不导入任何项目，项目也不导入它（两个项目的 `test_scaffold.py` 各有一条检查）。数据只经它写出的文件流向项目：JSONL 分片、`[[source]]` spec 片段、provenance 清单。去污染不在这里做：每个项目在训练 tokenizer 之前用自己的 `decontam` 模块做。

**真实数据行永不提交。** 测试用的 parquet 是测试运行时按登记的 schema 写出、内容标为 `SYNTHETIC FIXTURE` 的合成文件；Hub 由测试线程里的本地 `http.server` 代替。`verify-samples` 读到的真实行只在本地使用，报告只留计数。

## 状态

| 步骤 | 内容 | 状态 |
|---|---|---|
| S0 | 包骨架、`fork/dev test corpora`、CI、导入边界测试 | 已落地 |
| S1 | 来源登记 `sources`、Hub 客户端 `hub`、Range 文件 `rangefile` | 已落地 |
| S2 | parquet footer 与 row group `parquet`、计划 `plan` | 已落地 |
| S3 | 过滤器与注释器 `filters`、profile `profiles` | 已落地 |
| S4 | 导出与 provenance `export`、检查 `verify` | 已落地 |
| S5 | 可续跑的抓取循环 `run`、命令行、样本校验 | 已落地 |
| 备料 | 在 owner 主机上按各 profile 的预算 `plan` → `fetch` → `verify`，再把 spec 片段贴进项目 spec | 未运行 |

工作流的 S1（“骨架加语料适配器”）对应实现规格的 S0 到 S5：S0 单独一个提交，S1–S5 一个提交。

## 目录

| 模块 | 职责 |
|---|---|
| `surogate_corpora.canonical` | canonical JSON、sha256、`corpora-<name>/<major>` 格式检查、原子写入（与双语包同一约定，各自一份） |
| `surogate_corpora.sources` | 固定来源登记：repo、40 位 revision、文件范围、parquet schema、读取的列、许可块、允许的 profile |
| `surogate_corpora.hub` | Hub 客户端：按 revision 列出文件（tree API，分页），HEAD resolve 不跟随重定向并核对大小、ETag、xet hash、commit；token 只发给 endpoint 的主机 |
| `surogate_corpora.rangefile` | 可 seek 的只读文件，按块的 HTTP Range 读取，带小缓存；校验 206 与 Content-Range；签名 URL 过期或 401/403 时重新 resolve 一次；从不写盘 |
| `surogate_corpora.parquet` | footer（row group、逐列字节、schema）、按列投影读取一个 row group、扁平行（文本列为 bytes）、打开文件的 LRU |
| `surogate_corpora.plan` | `corpora-plan/1`：按 rank key 选 row group，元数据列惰性估计保留字节，到预算为止 |
| `surogate_corpora.filters` | 纯过滤器（许可、vendor、score、来源、UTF-8、非空、非拉丁字母比例）、精确去重、简繁标记计数 |
| `surogate_corpora.profiles` | 每种用途的来源、有序过滤器、注释器与写出器；把一个 profile 作用于一个来源的 `Pipeline` |
| `surogate_corpora.export` | TLC、双语、平行三种写出器，分片、spec 片段、署名清单 |
| `surogate_corpora.run` | `fetch`：按计划顺序处理 unit，每个 unit 后 fsync 并记入 `done.jsonl`；续跑时截回最后记录的状态 |
| `surogate_corpora.verify` | `verify_export`（重算分片哈希与计数）与 `verify_samples`（datasets-server 样本行对登记 schema 与导出） |

## 环境

所有 `python -m surogate_corpora …` 命令都在 `fork/.venv-corpora` 中运行（AGENTS.md 第 4 条）；备料需要网络，在 owner 主机上进行：

```bash
python3.12 -m venv fork/.venv-corpora
fork/.venv-corpora/bin/pip install -r fork/corpora/requirements.txt
fork/.venv-corpora/bin/pip install -e fork/corpora --no-deps
```

代理与 CA 取自环境（urllib 读 `HTTPS_PROXY`、`no_proxy`，默认 SSL 上下文读 `SSL_CERT_FILE`）。设置了 `HF_TOKEN` 时它只发给 `huggingface.co`，从不发给重定向指向的 CDN 主机；这些数据集匿名可读，不需要 token。

CPU 测试 `fork/dev test corpora` 用 `FORK_PY`（默认根目录 `.venv`），只从检出目录导入 `fork/corpora/src`（导出测试还用检出目录中的 TLC 与双语语料读取器读回结果），不安装本包；pyarrow 由 `fork/tools/requirements-cpu-lane.txt` 中的 `datasets` 带入，版本不固定，所以测试只比较解码后的值与导出字节，从不比较 parquet 字节。

## 命令行

```bash
fork/.venv-corpora/bin/python -m surogate_corpora sources
fork/.venv-corpora/bin/python -m surogate_corpora plan --profile tlc-python-permissive --budget-bytes 2600000000 --out "$CORPORA/plan-tlc-python.json"
fork/.venv-corpora/bin/python -m surogate_corpora fetch --plan "$CORPORA/plan-tlc-python.json" --out "$CORPORA/tlc-python"
fork/.venv-corpora/bin/python -m surogate_corpora verify "$CORPORA/tlc-python"
fork/.venv-corpora/bin/python -m surogate_corpora verify-samples ROWS.json --source ufw-zh --profile bilingual-zh-web --out "$CORPORA/check-zh"
```

- `plan --dry-run` 与 `fetch --dry-run` 只打印计划的保留字节与传输字节估计，什么都不写（`plan --dry-run` 仍要读 footer 与元数据列）。
- `--param name=value` 设 profile 参数：Python profile 的 `non_latin_max`（默认 0.10，TLC 在 Phase 0 冻结），`tlc-english` 的 `score_min`（默认不设）。
- `--allow-unlicensed` 是 owner 标志，只有数据来源消融 `tlc-python-unlicensed-ablation`（UltraData-Code-L2，没有逐文件许可字段）需要它。
- 每一种拒绝都以退出码 1 结束并写明原因；用法错误为 2。

## 来源与 profile

| 来源 | repo @ revision | 文件 | 允许的 profile |
|---|---|---|---|
| `stack3-py` | `openbmb/UltraData-Stack-v3-L2` @ `113743ad…` | python 45 个 | `tlc-python-permissive`、`tlc-python-all` |
| `ufw-en` | `openbmb/Ultra-FineWeb` @ `02c85641…` | en 2048 个 | `tlc-english`、`bilingual-en` |
| `ufw-zh` | 同上 | zh 256 个 | `bilingual-zh-web`、`bilingual-zh-apache`、`bilingual-zh-all` |
| `code-l2-py` | `openbmb/UltraData-Code` @ `8ce985cf…` | L2 py 119 个 | `tlc-python-unlicensed-ablation`（需 `--allow-unlicensed`） |
| `code-l3-py` | 同上 | L3 py 147 个 | 无：模型合成，任何预训练 profile 都拒绝 |
| `unpc-en-zh` | `Helsinki-NLP/un_pc` @ `f0eca5ee…`（parquet 转换分支 `refs/convert/parquet`） | en-zh 10 个 | `bilingual-parallel-unpc`（UltraData 之外的例外，需 owner 同意） |

| profile | 过滤器，按顺序 |
|---|---|
| `tlc-python-permissive` | 元数据：`licence_permissive`、`not_vendor`；内容：`utf8_valid`、`exact_dedup`、`non_latin_letter_ratio_max(x)` |
| `tlc-python-all` | owner 备选：去掉 `licence_permissive`（含 `no_license` 文件） |
| `tlc-python-unlicensed-ablation` | `utf8_valid`、`exact_dedup`、`non_latin_letter_ratio_max(x)` |
| `tlc-english` | 可选 `score_min(q)`；`utf8_valid`、`exact_dedup` |
| `bilingual-zh-web` | `source_in(CCI3, ChineseWebText, IndustryCorpus)`；`utf8_valid`；注释器 `script_stats` |
| `bilingual-zh-apache` | 网页池加 Tele |
| `bilingual-zh-all` | 全部 8 个上游来源 |
| `bilingual-en` | `utf8_valid`、`exact_dedup` |
| `bilingual-parallel-unpc` | 两侧 `utf8_valid`、`non_empty`；语言识别、长度比与对齐清洗留在双语项目 |

许可：每个来源的许可块写明发布层许可、上游许可与核实程度。双语的许可 id 都能被 `surogate_bilingual.data.manifest` 的 `LICENCE_ID` 接受（测试检查），`licence_verified` 一律为 false，只有 owner 读过条款后才改。Ultra-FineWeb zh 的每个上游来源（Tele、CCI3、ChineseWebText、IndustryCorpus、SkyPile、WanJuan、WuDao、MiChao）各有一个分片系列和一条 spec 片段，各自带许可 id。

## 数据契约

**计划 `corpora-plan/1`。** 每个文件的每个 row group 是一个 unit，rank key = sha256(`seed\0repo\0revision\0path\0rg`)，按 rank key 升序取，直到估计保留字节达到 `预算 × (1 + margin)`；所以计划只比目标多出不到一个 unit，单元来自所有文件，绝不取“开头的行”（Ultra-FineWeb zh 的行按来源成簇，文件开头全是 Tele）。估计只看 profile 的元数据过滤器：Stack-v3 是通过的行的 `size_bytes` 之和，Ultra-FineWeb 是 content 列未压缩字节乘以通过的行的比例，没有元数据过滤器的 profile 是 content 列未压缩字节；去重、UTF-8、非拉丁比例等内容过滤由 `margin` 覆盖。计划还记录 profile 的完整定义，代码里的 profile 改了之后旧计划被拒绝；文件的 schema 与登记的不同也被拒绝。同样的输入给出逐字节相同的计划文件。

**导出目录。**

- `parts/<名字>-NNNNN.jsonl`：每行一个 canonical JSON 对象，每个分片至多 256 MiB。文本就是来源的字节，不做任何换行、Unicode 或空白规范化（测试覆盖 CRLF、单独的 CR、制表符、U+2028、全角字符、emoji、CJK 扩展 B、NUL）。
- TLC 代码行 `{"group": repo_name, "id": uuid, "meta": {"path", "commit", "spdx", "unit"}, "text"}`，英文行 `{"id": "<文件名>:<rg>:<行>", "meta": {"unit", "score"}, "text"}`，不设 group；`surogate_tlc.data.corpus` 只读 `text` 与 `group`。
- 双语行 `{"id", "meta": {"upstream_source", "score", "unit", "script"}, "text"}`，平行行 `{"id", "pair_id", "segments": [{"lang": "en", …}, {"lang": "zh", …}], "meta"}`；只用双语读取器认识的键，从不写它拒绝的 `group`。
- `spec_fragment.toml`：要贴进项目 spec 的 `[[source]]` 表，列出确切的分片路径（相对导出目录）、url、revision、许可、`downloaded` 与 `filters`（每个过滤器的 `名字(参数)`，最后是 `corpora export <plan_sha256>`）；双语条目带 `licence_verified = false`。
- `attribution.jsonl`（代码导出）：每个仓库一行，`{"repo", "commit_ids", "spdx", "files", "bytes"}`。
- `provenance.json`（`corpora-export/1`）：计划 sha、profile、代码 commit 与 dirty 标志、python 与 pyarrow 版本、下载日期；每个 unit 的文件大小、观察到的 ETag 与 xet hash、行数、读入与保留的行、按过滤器的丢弃数、取回字节；每个分片的路径、sha256、字节与文档数；合计。
- `done.jsonl`：续跑清单。中断后重跑跳过已完成的 unit（不会重读），把分片截回最后记录的状态，从已写出的分片重建去重集合与署名清单；同一计划、profile 与代码给出逐字节相同的分片，中断与否都一样。续跑会重新打开文件，所以 provenance 里取回的字节（footer）可能与一次跑完的不同。

## 落地记录（2026-11-03，本机无 GPU，CPU 与网络只读）

CPU 测试：`fork/dev test corpora` 117 项通过，约 10 秒（4 vCPU 容器，FORK_PY 的 pyarrow 26.0.0）；在 `fork/.venv-corpora`（pyarrow 21.0.0）中同一套测试只有平行布局一项失败，原因是那里没有双语 `data.parallel` 需要的 numpy（该项只在测试中导入）。

对真实数据的本地核对（`fork/.venv-corpora`，pyarrow 21.0.0；读到的行与导出用后即删，这里只留计数）[CPU实测]：

| 核对 | 结果 |
|---|---|
| datasets-server 20 行，Ultra-FineWeb en，`bilingual-en` 与 `tlc-english` | schema 一致；20 篇，文本 101,541 字节，导出字节等于输入字节 |
| 同上 zh，`bilingual-zh-all` | 20 篇，86,238 字节，全部进入 `ufw-zh-tele` 分片；以繁体为主的 0 篇；NFKC 会改变 13/20 篇，导出一篇也没改 |
| 同上 zh，`bilingual-zh-web` | 保留 0 篇（20 篇都因 `source_in` 丢弃）：开头 20 行全是 Tele，这就是 unit 不能取开头行的原因 |
| UltraData-Code-L2 py 20 行，`tlc-python-permissive` | 拒绝：该 profile 依赖的许可与 vendor 列在这个来源里不存在 |
| 同上，`tlc-python-unlicensed-ablation` | 不带 `--allow-unlicensed` 拒绝；带上后 20 篇，153,269 字节，每篇都有 `group = repo_name` |
| UltraData-Code-L3 py，任何预训练 profile | 拒绝（模型合成） |
| Stack-v3 python，datasets-server 偏移 0、1M、5M、12M、24M 各 100 行，`tlc-python-permissive` | 嵌套 schema 一致；各块 no_license 占 94/95/97/95/96%（合计 477/500），保留的 23 篇都是 permissive 且非 vendor，导出字节等于输入字节 |
| UN PC：真实 `plan`（预算 1 MB）与 `fetch` | 10 个 footer 共 11.76 MB（每个约 1.2 MB，每个文件 1746 个约 1000 行的 row group）；选 4 个 unit，抓取 4,000 对、1,088,252 文本字节，`verify` 干净 |
| Stack-v3：真实 `plan --dry-run`（预算 30 MB） | 45 个文件、405 个 row group；3 个 unit 估计保留 35.3 MB，估计传输 273 MB，元数据取回 5.3 MB：保留字节约为传输的 13% |

与实现规格的出入：

- `requirements.txt` 固定 pyarrow 21.0.0：探针环境实际装的是 21.0.0，规格写的 22.0.0 与之不符。
- `RangeFile` 默认块 64 KiB，而不是 8 MiB：每次读取的缺失块合并成一个请求（单个请求至多 16 MiB），块越大只会在两端多取；UN PC 的 row group 约 160 KB，8 MiB 的块要多取约 50 倍。
- 文件列表用 tree API（`/api/datasets/<repo>/tree/<sha>/<目录>`，分页），不用 `/revision/<sha>?blobs=true`：只有前者给出 xet hash，而且只列一个目录（Ultra-FineWeb 的整库列表有 19.5 MB）。
- UN PC 在登记里直接固定 parquet 转换分支的完整 sha `f0eca5eea7dad233a4b59de0479f9374d7a3500f`（本次从 API 读到），列名 `translation.en`、`translation.zh` 读自 footer；规格原打算在计划时再读。
- Ultra-FineWeb 的 `score` 在 parquet 里是字符串（datasets-server 显示为 float32）：登记分别记两种类型，`score_min` 按数值比较，导出的 `meta.score` 是 JSON 数字。
- 每个 profile 的内容过滤器都以 `utf8_valid` 开头（不是合法 UTF-8 的文本写不成 JSON 文本）；平行 profile 的顺序是先 `utf8_valid` 后 `non_empty`。
- 平行文档的 id 是 `unpc-en-zh-<文件名>:<rg>:<行>`：规格的 `unpc:<rg>:<行>` 在 10 个文件之间会重复。
- spec 片段的 `files` 列出确切的分片路径，而不是 glob。
- 计划另记 `profile_def`、`params`、`allow_unlicensed`、每个 row group 的 `pass_compressed`（内容遍历要读的列的压缩字节），估计另记 `units`、`units_total`、`budget_reached`；provenance 带自身的 `provenance_sha256`；`verify-samples` 要求新的 `--out` 目录并写出只含计数的 `sample_report.json`。

仍未核实或未做：

- 全规模备料没有跑过：Ultra-FineWeb en 的计划要读 2048 个 footer（每个约 64 KB，约 4,000 个请求；Hub 对 resolve 的限额是每 5 分钟 3,000 次，客户端遇 429 按 Retry-After 退避），耗时与传输量都未测。
- 内容过滤器（去重、非拉丁比例约 5.5%）让实际保留字节低于元数据估计，需要多大的 `margin` 要在 owner 主机上实测。
- 峰值内存未测：一次在内存中保留一个 row group（Stack-v3 的 content 列未压缩约 315 MB [估计]：压缩约 120 MB 乘以 footer 给出的未压缩/压缩比 2.62），再按 4096 行一批转成 Python 对象。
- 许可条款（TeleChat 社区许可、SkyPile、WanJuan、WuDao、MiChao）没有读原文，`licence_verified` 保持 false。
- UN PC 不记录原文语言，导出一律 en 在前，所以双语的 `data.parallel` 只会看到 en>zh 方向；zh>en 的 Bridge 数据怎么构造由双语项目决定。
