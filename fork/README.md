# Fork 维护入口

本仓库 `CatMeow-233/surogate` 是 [invergent-ai/surogate](https://github.com/invergent-ai/surogate) 的 fork。上游代码保持原有目录布局；fork 自有内容集中在 `fork/` 下，对上游文件的每一处修改都登记在 [patches/ledger.toml](patches/ledger.toml) 中。开发准则见 [AGENTS.md](../AGENTS.md)。

第一次在本地使用时运行 `fork/dev setup`（配置上游远程和合并选项，并打印需要在 GitHub 上完成的设置），再按下文“首次准备环境”创建 Python 环境，之后从 `fork/dev help` 开始。

## 常用命令

所有维护操作都通过 `fork/dev` 完成；它会先打印底层命令再执行，可以在任意目录下运行，参数中的相对路径相对于当前目录。

| 要做的事 | 命令 |
|---|---|
| 查看全部命令和产物 | `fork/dev help` |
| 提交前检查 | `fork/dev map`（重新生成地图和 ledger 表格），然后 `fork/dev check`（ledger、地图、文档链接和 make 目标） |
| 构建训练扩展或服务引擎 | `fork/dev build train`、`fork/dev build serve` |
| 测试 | `fork/dev test cpu`、`patches`、`experimental`、`tlc`、`openmp` 不需要 GPU；`native`、`gpu` 在 GPU 主机上运行。各自的内容和所需环境见 [测试命令](docs/build-and-release.md#测试命令) |
| 从检出目录运行实验训练器 | `fork/dev exp-train <配置>` |
| 发布实验版本 | 按 [发布实验版本](docs/build-and-release.md#发布实验版本) 的清单操作 |
| 发布原生 wheel（RTX 50 系列） | 修改 `fork/native/release/VERSION` 并推送，见 [发布原生版本](docs/build-and-release.md#发布原生版本) |
| 同步上游（每周一次） | `fork/dev sync --dry-run` 查看报告，`fork/dev sync` 合并并开 PR，见 [docs/sync.md](docs/sync.md) |

每条命令对应的产物、构建目录和发布方式见 [docs/build-and-release.md](docs/build-and-release.md)。fork CI（`.github/workflows/fork-ci.yml`）在每次推送时运行同样的 `fork/dev` 命令。

## 首次准备环境

`fork/dev test experimental` 和 `fork/dev exp-train` 自己创建并使用隔离的 `fork/.venv-experimental`，只需要 uv。其余 Python 命令（`fork/dev test cpu`、`fork/dev test patches`、`fork/dev test tlc`、同步的检查步骤）默认使用仓库根目录的 `.venv`，可以用 `FORK_PY` 指向别的解释器。

- GPU 主机：按上游 [README.md](../README.md) 中 “For development” 的步骤创建 `.venv`（`uv venv --python 3.12`，然后 `uv pip install -e .`，会完整构建原生扩展）。之后修改训练侧 C++ 时用 `fork/dev build train`，它只重建 `_surogate` 并把 `.so` 复制进 `.venv`。
- 只用 CPU：与 fork CI 相同，不构建原生扩展。无法访问 download.pytorch.org 时改装 PyPI 的 `torch==2.11.0`，并在运行 `fork/dev test experimental` 时设置 `FORK_TORCH_INDEX=pypi`。

```bash
python3.12 -m venv .venv
.venv/bin/pip install 'torch==2.11.0+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r fork/tools/requirements-cpu-lane.txt
```

TLC 包 `fork/tlc` 在 GPU 主机上另用带原生构建的 `fork/.venv-tlc`，见 [tlc/README.md 的“环境”](tlc/README.md#环境)。

## 代码在哪里

| 领域 | 原生代码（C++/CUDA） | Python | 入口 |
|---|---|---|---|
| 训练 | `csrc/src/{runtime,kernels,recipes,utilities,config,shims}`，绑定 `csrc/src/binding/binding.cpp` 与 `csrc/src/binding/py_train.cpp` | `surogate/{train,dsl,core,grpo,dpo,distill}` | 扩展模块 `_surogate`：`csrc/src/binding/binding.cpp` 中的 `NB_MODULE(_surogate, m)` |
| 服务 | `csrc/src/serve`（CMake 中是 `csrc/CMakeLists.txt` 里 `if(SUROGATE_BUILD_SERVE)` 开始的段落），绑定 `csrc/src/binding/py_serve*.cpp` | `surogate/serve` | 扩展模块 `_surogate_serve`：`csrc/src/binding/py_serve.cpp` 中的 `NB_MODULE(_surogate_serve, m)`；引擎可执行文件见 [maps/native.md](docs/maps/native.md) 的“产物入口” |
| 共享 | `csrc/src/tokenizer`（训练与服务各编译一次），`csrc/src/serve/core/unified_memory.h`（训练侧也包含它，文件列表见 [maps/native.md](docs/maps/native.md) 的“接缝”） | | |
| 命令行 | | `surogate/cli` | 命令 `surogate` = `surogate.cli.main:cli_main`（根目录 `pyproject.toml` 的 `[project]` 表中的 `scripts`），子命令表 `COMMAND_MAPPING` 在 `surogate/cli/main.py` |
| fork：GRPO 补丁调用的代码 | | `surogate/_fork`（随原生 wheel 发布） | 由 `surogate/grpo/trainer.py` 在函数内导入（ledger T6、T7） |
| fork：原生代码 | `fork/native`：CMake 挂接、头文件、C++ 测试 | | `csrc/CMakeLists.txt` 中紧跟 `surogate-common` 的 `target_link_libraries` 之后的两行挂接点，见 [native/README.md](native/README.md) |
| TLC 实验 | 无 | 独立 fork 包 `fork/tlc`：代码 `fork/tlc/src/surogate_tlc`，测试 `fork/tlc/tests/cpu` 与 `fork/tlc/tests/gpu`，纯 PyTorch 原型 `fork/tlc/prototypes`（不打包），打包 `fork/tlc/pyproject.toml`（不发布，也不进原生 wheel） | editable 安装进 `fork/.venv-tlc` 后运行 `python -m surogate_tlc <子命令>`（训练、评测、模型目录与数据），示例配置在 `fork/tlc/examples`；CPU 测试 `fork/dev test tlc`；说明与 GPU 运行手册见 [tlc/README.md](tlc/README.md) |
| 实验训练 | 无 | 独立项目 `fork/experimental`：代码 `fork/experimental/src/surogate/experimental`，测试 `fork/experimental/tests`，文档 `fork/experimental/docs`，示例 `fork/experimental/examples`，打包 `fork/experimental/pyproject.toml` 与 `fork/experimental/build_release.py` | `python -m surogate.experimental.train` 或脚本 `surogate-experimental-train`（`fork/experimental/pyproject.toml` 的 `[project.scripts]`）；从检出目录运行用 `fork/dev exp-train`；发布工作流 `.github/workflows/experimental-release.yml` |

更细的位置由 `fork/tools/gen_maps.py` 从当前代码生成：

- [maps/python.md](docs/maps/python.md)：各子包的文件数与作用、命令行子命令到模块的对应、`python -m` 入口、子包之间的导入关系、动态导入、依赖目录深度的路径。
- [maps/native.md](docs/maps/native.md)：`csrc/src` 各目录属于哪个引擎、全部 CMake 目标及其 `OUTPUT_NAME` 和所在文件、头文件根目录、训练与服务之间的接缝、一个服务模型目标分布的 8 处位置。
- [maps/tests.md](docs/maps/tests.md)：三棵测试树、pytest 标记、`tests/` 各目录的测试文件数与 GPU/slow 标记、`csrc/src/testing/serve` 中的 Python 文件。
- [maps/docs.md](docs/maps/docs.md)：每个 markdown 文件的分类。

地图只记录有什么、在哪里（文件、导入、CMake 目标、分类），不含行号和行数，所以只改文件内容不会让它们过期；新增、删除或移动文件，或修改 ledger 之后运行 `fork/dev map`。同步时刷新提交会自动重新生成它们。

上游新增 `surogate/` 或 `csrc/src` 下的顶层目录、或新增无法归类的 markdown 文件时，`fork/dev check` 会失败，提示在 `fork/tools/gen_maps.py` 的分类表中补一条规则；脚本按固定路径读取的上游文件（例如 `surogate/cli/main.py`、`csrc/CMakeLists.txt`）被移动或删除时，它同样失败并提示修改脚本中的路径。

## 文档分类

仓库中的 markdown 文件分为四类（逐个文件的归类见 [maps/docs.md](docs/maps/docs.md)）：

- **U 上游用户与开发者文档**：`README.md`、`docs/` 下的指南与参考（目录页 `docs/index.md`）、`examples/` 的说明。fork 不修改它们。
- **N 上游工程笔记**：开发日志、跟踪清单、设计稿和测量记录，例如 `csrc/src/serve/PATCHES.md`、`surogate/serve/BENCHMARKS.md`。原先在仓库根目录的笔记已原样移到 `docs/upstream-notes/`，见 [其说明](../docs/upstream-notes/README.md)。
- **P 来源与授权说明**：vendored 代码和 NOTICE 文件，不移动、不修改。
- **F fork 自有文档**：`fork/` 下的文档、`AGENTS.md`、`docs/experimental/training.md`（指向新位置的跳转页）和 `docs/upstream-notes/README.md`。

`fork/tools/check_doc_links.py` 对 F 类文档做严格检查：链接、反引号中的仓库路径和代码块中的路径都必须存在。其他文档只在出现 [doc_links_baseline.txt](tools/doc_links_baseline.txt) 之外的新失效链接时失败；上游已有的失效链接记录在 [docs/upstream-errata.md](docs/upstream-errata.md)。

fork 文档引用上游代码时写文件和符号名（函数、CMake 目标或 `if()` 段落、配置表），不写行号：上游的每次修改都会让行号漂移，而检查只能确认文件存在。确实需要行号的文档（例如 [upstream-errata.md](docs/upstream-errata.md)）要写明行号对应的上游提交。

## Fork 改了什么

`git diff -M --name-status $(git merge-base HEAD upstream/main) HEAD` 中的每一条都必须能在 ledger 中找到：

- **A（新增）**：fork 自有路径，列在 `[owned]` 中（`fork/` 及少数必须放在其他位置的 fork 文件）。
- **M（修改上游文件）**：归属某个 `[[topic]]`（T1–T9），每个 topic 记录提交、开关、涉及文件、测试和冲突历史。
- **D（删除）/ R（重命名）**：分别列在 `[[deletion]]` 与 `[[relocation]]` 中；重命名必须是 100% 相似度。

ledger 的表格形式见 [patches/README.md](patches/README.md)（由 ledger 生成）。检查命令（只用标准库，比较 merge base 与工作区，已暂存和未暂存的改动都算，未跟踪的文件在 `git add` 之前不计；`fork/dev check` 会运行它）：

```bash
python3 -I fork/tools/check_divergence.py --strict
```

未登记的改动会使其以退出码 1 失败并列出路径；`--strict` 还会报告过期条目（topic 文件已与上游一致、登记路径已不存在等）。实现见 [tools/check_divergence.py](tools/check_divergence.py)。

GRPO 补丁（T6–T8）调用的 fork 代码位于 `surogate/_fork`，上游文件中只保留调用它的几行。这些补丁的 CPU 测试位于 `fork/patches/tests/grpo`，用 `fork/dev test patches` 运行；`tests/` 与上游逐字节一致。

## 文档

- [docs/sync.md](docs/sync.md)：同步上游的步骤、GitHub 设置清单（仓库所有者操作）、按 topic 的冲突处理方法，以及冲突面的测量依据。
- [docs/build-and-release.md](docs/build-and-release.md)：每个产物的构建命令、输出位置和发布工作流，`fork/dev` 的测试命令，版本号和依赖版本固定在哪里，以及发布实验版本的清单。
- [patches/README.md](patches/README.md)：ledger 的表格形式，包括每个 topic 的文件、测试和冲突记录。
- [docs/performance.md](docs/performance.md)：fork 的性能改动、测量方法和开关，包括 LoRA 融合输入梯度和 GRPO 周期计时指标。
- [native/README.md](native/README.md)：fork 原生代码（C++ 与 CMake）的目录、`csrc/CMakeLists.txt` 中两行挂接点的约定，以及 OpenMP 接线测试的运行方法。
- [docs/upstream-errata.md](docs/upstream-errata.md)：上游已知的失效引用、过时配置和测试缺陷；fork 不修改这些上游文件，只记录绕行方法。
- [tlc/README.md](tlc/README.md)：TLC 包的目录、命令行、模型目录与 checkpoint 契约、初始化与 Mamba-2 数值参照、语料 spec、去污染、合成绑定任务、tokenizer 与数据 shard、训练驱动（运行配置、OOM 策略、运行记录）、评测（loss 与绑定探针）、GPU 预算和 RTX 5090 运行手册。
- [experimental/README.md](experimental/README.md) 与 [experimental/docs/training.md](experimental/docs/training.md)：实验训练器的安装、配置和发布产物。

## 规则

1. fork 自有文件放在 `fork/` 下，例外只能是 ledger `[owned]` 中列出的路径。
2. 对上游文件的每一处修改都属于一个 ledger topic；新增或删除修改时在同一个提交里更新 ledger。提交前运行 `fork/dev map` 和 `fork/dev check`。
3. 同步上游只用 `fork/dev sync`：merge，不 rebase，也不用 GitHub 的 “Sync fork”。
4. 实验代码只在 `fork/` 下的隔离虚拟环境中运行：实验训练器 `fork/experimental` 用 `.venv-experimental`（不含原生扩展）；TLC 包 `fork/tlc` 需要原生训练器，以 editable 方式装进带原生构建的 `.venv-tlc`；TLC 的纯 PyTorch 原型 `fork/tlc/prototypes` 不打包，在 `.venv-experimental` 或任一 CPU torch 环境中运行。它们的测试用 `fork/dev test experimental` 和 `fork/dev test tlc`（CPU，不需要原生扩展）。不要在仓库根目录执行 `uv run` 或 `uv sync`。
5. 提交、推送和 Release 只面向 `CatMeow-233/surogate`，不向上游推送或创建 PR。
