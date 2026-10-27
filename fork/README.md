# Fork 维护入口

本仓库 `CatMeow-233/surogate` 是 [invergent-ai/surogate](https://github.com/invergent-ai/surogate) 的 fork。上游代码保持原有目录布局；fork 自有内容集中在 `fork/` 下，对上游文件的每一处修改都登记在 [patches/ledger.toml](patches/ledger.toml) 中。开发准则见 [AGENTS.md](../AGENTS.md)。

## 代码在哪里

| 领域 | 原生代码（C++/CUDA） | Python | 入口 |
|---|---|---|---|
| 训练 | `csrc/src/{runtime,kernels,recipes,utilities,config,shims}`，绑定 `csrc/src/binding/binding.cpp` 与 `csrc/src/binding/py_train.cpp` | `surogate/{train,dsl,core,grpo,dpo,distill}` | 扩展模块 `_surogate`：`NB_MODULE` 位于 `csrc/src/binding/binding.cpp:197` |
| 服务 | `csrc/src/serve`（CMake 段落 `csrc/CMakeLists.txt:885-2023`），绑定 `csrc/src/binding/py_serve*.cpp` | `surogate/serve` | 扩展模块 `_surogate_serve`：`NB_MODULE` 位于 `csrc/src/binding/py_serve.cpp:234` |
| 共享 | `csrc/src/tokenizer`，`csrc/src/serve/core/unified_memory.h` | | |
| CLI | | `surogate/cli` | 命令 `surogate` = `surogate.cli.main:cli_main`（`pyproject.toml:92`） |
| 实验训练 | 无 | 独立项目 `fork/experimental`：代码 `fork/experimental/src/surogate/experimental`，测试 `fork/experimental/tests`，文档 `fork/experimental/docs`，示例 `fork/experimental/examples`，打包 `fork/experimental/pyproject.toml` 与 `fork/experimental/build_release.py` | `python -m surogate.experimental.train`，发布工作流 `.github/workflows/experimental-release.yml` |

## Fork 改了什么

`git diff -M --name-status $(git merge-base HEAD upstream/main) HEAD` 中的每一条都必须能在 ledger 中找到：

- **A（新增）**：fork 自有路径，列在 `[owned]` 中（`fork/` 及其他尚在上游目录中的 fork 文件）。
- **M（修改上游文件）**：归属某个 `[[topic]]`（T1–T9），每个 topic 记录提交、开关、涉及文件、测试和冲突历史。
- **D（删除）/ R（重命名）**：分别列在 `[[deletion]]` 与 `[[relocation]]` 中；重命名必须是 100% 相似度。

检查命令（只用标准库，读取已提交的 HEAD）：

```bash
python3 -I fork/tools/check_divergence.py --strict
```

未登记的改动会使其以退出码 1 失败并列出路径；`--strict` 还会报告过期条目（topic 文件已与上游一致、登记路径已不存在等）。实现见 [tools/check_divergence.py](tools/check_divergence.py)。

GRPO 补丁（T6–T8）调用的 fork 代码位于 `surogate/_fork`（随原生 wheel 发布），上游文件中只保留调用它的几行。这些补丁的 CPU 测试位于 `fork/patches/tests/grpo`，用 `python -m pytest -q fork/patches/tests` 运行；`tests/` 与上游逐字节一致。

## 文档

- [docs/build-and-release.md](docs/build-and-release.md)：每个产物的构建命令、输出位置和发布工作流，版本号和依赖版本固定在哪里，以及 `fork/dev` 的测试命令。日常操作从 `fork/dev help` 开始。
- [docs/performance.md](docs/performance.md)：fork 的性能改动、测量方法和开关，包括 LoRA 融合输入梯度和 GRPO 周期计时指标。
- [native/README.md](native/README.md)：fork 原生代码（C++ 与 CMake）的目录、`csrc/CMakeLists.txt` 中两行挂接点的约定，以及 OpenMP 接线测试的运行方法。
- [docs/upstream-errata.md](docs/upstream-errata.md)：上游已知的失效引用、过时配置和测试缺陷；fork 不修改这些上游文件，只记录绕行方法。

## 规则

1. fork 自有文件放在 `fork/` 下，例外只能是 ledger `[owned]` 中列出的路径。
2. 对上游文件的每一处修改都属于一个 ledger topic；新增或删除修改时在同一个提交里更新 ledger。
3. 同步上游只用 `fork/dev sync`：merge，不 rebase，也不用 GitHub 的 “Sync fork”。
4. 实验代码只在 `fork/` 下的隔离虚拟环境 `.venv-experimental` 中运行；不要在仓库根目录执行 `uv run` 或 `uv sync`。
5. 提交、推送和 Release 只面向 `CatMeow-233/surogate`，不向上游推送或创建 PR。
