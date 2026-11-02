# v1.0 原始文件（参照）

本目录逐字节原样保存 owner 于 2026-11-02 上传的研究包 `zh_en_merge_loss_comparison_v1.zip`（v1.0）中的六个文件。它们是 owner 自己的工作，只作为新包的参照：不修改、不打包，`surogate_bilingual` 不导入它们，只有测试按文件路径加载它们。每个文件的 sha256、字节数和在压缩包中的路径记在 [MANIFEST.json](MANIFEST.json) 中，压缩包本身的 sha256 也记在那里；[test_scaffold.py](../tests/cpu/test_scaffold.py) 检查本目录的文件与清单一致，且其中没有 `test_*.py`。

| 文件 | 压缩包中的位置 | 内容 | 新包中的用途 |
|---|---|---|---|
| [moe_pilot.py](moe_pilot.py) | `language_baseline/` | v1.0 的 `RoutedLM` pilot：24 层、d=512、GQA 8Q/2KV、词表 32768、SwiGLU 1728、tied head；子命令 seed、expert、assemble、verify、generate、tokenizer、pack | I2 按路径加载它，把权重映射到新的 `RoutedLM`，检查两者的 logits 一致 |
| [evaluate.py](evaluate.py) | `language_baseline/` | v1.0 的评测脚本 | I6 移植评分规则时的参照 |
| [eval_cases.example.jsonl](eval_cases.example.jsonl) | `language_baseline/` | v1.0 的评测样例 | 同上 |
| [verify_merge_math.py](verify_merge_math.py) | 根目录 | fp64 代数检查：拼接等于加权和、置换不变性、交叉连接、pre/post 的 Jensen 差界、稀疏通道选择、秩约束 ridge、无条件合并的 JS 下界、带掩码的 router、冻结权重的梯度、参数预算 | I3 按路径加载它并调用 `main`，要求报告全部断言通过；它的各项检查也移植成新包的单元测试 |
| [compare.py](compare.py) | 根目录 | 合成回归对照 | I3 把其中的恒等式改写为标注“恒等式检查”的单元测试；它们不是实验结果 |
| [smoke_results.json](smoke_results.json) | 根目录 | `compare.py` 冒烟运行的输出 | 只作记录：它是恒等式检查的输出，不是双语模型的结果 |

## 没有收入的文件

- `test_compare.py`：`fork/tools/gen_maps.py` 不允许 fork 中出现测试树之外的 `test_*.py`（`fork/dev check` 会失败），它的断言在 I3 移植到新包的 CPU 测试中。
- 文稿与说明：`research_manuscript.md`、`full_experiment_manual.md`、`EXPERIMENT_PROTOCOL.md`，以及包根目录与 `language_baseline/` 中的两个 `README.md`。它们是 owner 的文档，I1 不提交；owner 若要把它们（或修订后的 v1.1 稿）放进仓库，放在 `fork/bilingual/docs` 下，并且要通过 `fork/dev check` 的严格链接检查。

## 规则

- 不要编辑本目录中的文件。`fork/bilingual/pyproject.toml` 的 `[tool.ruff]` 排除了本目录，所以 `fork/dev fmt` 不会改写它们。
- 评审在 v1.0 代码中指出的缺陷在新包中修正，不在这里修改。例如评审判断 CUDA bf16 autocast 下 `_ffn` 的 `index_add` 会因 fp32 与 bf16 不一致而报错（NEW-01）；这一条在 CUDA 上尚未验证，由 I2 的 CPU autocast 回归测试与 G0 的 GPU 检查覆盖。
- 换用 owner 的新版文件时，作为一次单独的提交同时更新文件与 `MANIFEST.json`，并在提交说明中写明来源。
