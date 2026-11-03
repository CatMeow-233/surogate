# fork/tokstudy：分词器研究

本包做 TLC 的分词器决定 O-10 与双语可组合专家的分词器决定 B-O11 的“效果对比”：在同一个 PyTorch 后端上，以相同的训练 FLOPs 训练子词、平坦字节与 H-Net（学习式动态分块，即“另類神經網絡分詞”）各臂，按语言、按领域比较 bits-per-byte，并按预注册的阈值给出判定。owner 的决定（原文）：「分詞器得對比效果要不然就搞另類神經網絡分詞 語料用minicpm的ultradata那些的」。

分工：子词与子词之间的 TLC 比较仍在原生训练器上做（TLC 自己的 TOK-AB）；本包的 TLC 表只放 N0（原生领先臂的 PyTorch 孪生）、N1（H-Net）与 N2（平坦字节），永不与原生的行同表。双语的全部臂 B0–B5 都在本包中经真实的双语 trunk 训练。生产用的 tokenizer 构建器留在各自的项目里；本包只按路径与 sha256 读取它们产出的 `tokenizer.json`。语料由 [fork/corpora](../corpora/README.md) 备好，经各项目自己的去污染与划分后才进入本包。

## 状态

本机没有 GPU：凡 RTX 5090 的数字都是 [估计] 或 TBD，以 G0.5-ts 实测为准。

| 步骤 | 内容 | 命令 | 状态 |
|---|---|---|---|
| S0 | 包骨架、`fork/dev test tokstudy`、CI、导入边界测试 | — | 已落地 |
| S8 | 臂登记与编解码器 | `arms` | 未落地 |
| S9 | 文档、分帧、训练流 | — | 未落地 |
| S10 | 内在指标与教师对齐（只作 CPU 初筛，不作判定依据） | `intrinsic`、`align` | 未落地 |
| S11–S12 | SSD/EMA/卷积 kernel、Mamba-2、A0 孪生、H-Net 与 LM 包装 | — | 未落地 |
| S13 | FLOPs 与 H-Net 配置匹配 | `flops` | 未落地 |
| S14 | 训练循环（按 FLOP 进度的 WSD 与衰减分支） | `train`、`branch` | 未落地 |
| S15 | 评测（公共字节前缀上的 BPB）与探针 | `eval`、`probe` | 未落地 |
| S16 | 统计、预注册与判定 | `verdict` | 未落地 |
| S17 | GPU：G0-ts、G0.5-ts、kernel 一致性、孪生门 | `bench` | 未运行 |

未落地的命令以退出码 2 报出所属步骤（`python -m surogate_tokstudy` 列出全部命令）。

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
| 测试 | 另可按路径加载 `fork/tlc/prototypes/ref_mamba2.py`，并导入 `surogate_tlc.ir`、两个项目的 tokenizer 与 shard 模块 | — |

`tests/cpu/test_scaffold.py` 按 AST 检查这张表，并用埋入的违规样例确认检查本身有效；两个项目的 `test_scaffold.py` 各有一条检查确认它们从不导入本包或 `surogate_corpora`。
