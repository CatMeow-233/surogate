# fork/corpora：固定版本的语料适配器

本包把 Hugging Face Hub 上固定 revision 的 UltraData（以及 UN PC）parquet shard 按 HTTP Range 逐个 row group 读出、过滤，导出为 TLC 与双语项目各自的语料格式，并写下 provenance。owner 的决定（原文）：「分詞器得對比效果要不然就搞另類神經網絡分詞 語料用minicpm的ultradata那些的」；本包负责其中“语料用 UltraData”的数据侧。

它是叶子包：不导入任何项目，项目也不导入它（两个项目的 `test_scaffold.py` 各有一条检查）。数据只经它写出的文件流向项目：JSONL 分片、`[[source]]` spec 片段、provenance 清单。真实数据行永不提交；测试用的 parquet 是测试运行时按真实 schema 写出、内容标为 `SYNTHETIC FIXTURE` 的合成文件。

## 状态

| 步骤 | 内容 | 状态 |
|---|---|---|
| S0 | 包骨架、`fork/dev test corpora`、CI、导入边界测试 | 已落地 |
| S1–S5 | 来源登记、Hub 客户端、Range 文件、parquet 与计划、过滤与 profile、导出与 provenance、抓取循环与命令行 | 未落地；命令行在它们落地前以退出码 2 报出所属步骤 |

## 环境

所有 `python -m surogate_corpora …` 命令都在 `fork/.venv-corpora` 中运行（AGENTS.md 第 4 条）；备料需要网络，在 owner 主机上进行：

```bash
python3.12 -m venv fork/.venv-corpora
fork/.venv-corpora/bin/pip install -r fork/corpora/requirements.txt
fork/.venv-corpora/bin/pip install -e fork/corpora --no-deps
```

CPU 测试 `fork/dev test corpora` 用 `FORK_PY`（默认根目录 `.venv`），只从检出目录导入 `fork/corpora/src`，不安装本包；pyarrow 由 `fork/tools/requirements-cpu-lane.txt` 中的 `datasets` 带入，版本不固定，所以测试只比较解码后的值与导出字节，从不比较 parquet 字节。

## 导入边界

`surogate_corpora` 只导入标准库与 pyarrow，pyarrow 只在函数内导入（`import surogate_corpora` 不加载 pyarrow）；不导入 torch、`surogate`、`surogate_tlc`、`surogate_bilingual`、`surogate_tokstudy`。`tests/cpu/test_scaffold.py` 按 AST 检查这一点，并用埋入的违规样例确认检查本身有效。
