# 上游勘误

本文记录上游代码中已知的失效引用、过时配置和测试缺陷。fork 不修改这些上游文件：每处修复都会成为每次同步时的永久冲突来源，而 AGENTS.md 又不允许向上游提 PR。这里只记录现象、证据和绕行方法。数字和行号对应上游 `282f9901`（2026-10-23）；同步之后可以用各节给出的命令重新核对。发现新的上游缺陷时，在本文追加一节，写明现象、证据（文件:行号或提交）和绕行方法。

## 已改为本地保存的 design 笔记

上游在 `e15524d1`（2026-08-31，“chore: design notes are local-only”）删除了仓库中剩余的 design 笔记（另有几篇在更早的提交中删除），并在 `.gitignore:41` 忽略该目录。代码、构建文件和上游工程笔记中仍有 66 处引用，分布在 44 个文件中，指向 12 个不同的笔记；`docs/` 下的用户文档不引用它们。列出全部引用（排除 fork 自己的文档，例如本页）：

```bash
git grep -nE 'design/[A-Za-z0-9_.-]+\.md' -- ':!fork'
```

其中 11 个可以从历史中取回（行尾注释为引用次数），`MODELS.md` 从未被跟踪：

```bash
git show e15524d1^:design/INFERENCE.md                   # 23
git show a04e61e3^:design/buffer-runtime-v4.md           # 16
git show e15524d1^:design/serve-engine-plan.md           # 7
git show e15524d1^:design/serve-engine-multiarch.md      # 6
git show e15524d1^:design/serve-engine-flash-next.md     # 3
git show a04e61e3^:design/capture-safe-runtime-plan.md   # 3
git show e15524d1^:design/unified-train-serve.md         # 2
git show e15524d1^:design/35B_VLLM_KERNELS.md            # 2
git show 3037a3db^:design/tid-baked-dispatch.md          # 1
git show e15524d1^:design/p0-port-spike.md               # 1
git show e15524d1^:design/NVFP4_ROUTED_EXPERTS.md        # 1
# design/MODELS.md（1 处引用）从未被跟踪，无法取回
```

## 被忽略的 study 目录

`.gitignore:20` 忽略 study 目录，上游历史中从未跟踪过其中的文件。除 `.gitignore` 和 `.dockerignore` 外，仍有 38 处引用，分布在 17 个文件中，无法在仓库内解析（同样排除 fork 文档）：

```bash
git grep -n 'study/' -- ':!.gitignore' ':!.dockerignore' ':!fork'
```

## 已删除的 NInfer 署名文件

`csrc/src/serve/PATCHES.md:4-6` 仍以链接形式指向同目录下的 NOTICE，说明服务引擎的第三方来源和许可证。上游在 `8d1b9cc4`（2026-09-02）删除了 NOTICE 和 LICENSE.ninfer。取回：

```bash
git show 8d1b9cc4^:csrc/src/serve/NOTICE
git show 8d1b9cc4^:csrc/src/serve/LICENSE.ninfer
```

fork 是否需要恢复其中的 Apache-2.0 署名属于法律判断，由所有者决定；默认不恢复。

## 失效的文档链接

在 `282f9901` 上，仓库中 33 个本地 Markdown 链接失效：

- 23 个 `docs/` 页面以 Docusaurus 时代的页脚 “Back to docs index” 结尾，链接到已不存在的 `index.mdx`。上游在 `7fee9b13`（2026-05-08）删除了 Docusaurus 站点，`3902e260`（2026-05-15）恢复的扁平页面以 `docs/index.md` 为目录。列出：`git grep -n 'index.mdx' -- docs`。
- 其余 10 个（文件:行号与失效目标）：

```text
docs/reference/config.md:198              guides/multi-node.md   （应为 ../guides/multi-node.md）
csrc/src/serve/PATCHES.md:6               NOTICE                 （已在 8d1b9cc4 删除）
csrc/src/testing/serve/README.md:5        ../AGENTS.md
csrc/src/testing/serve/README.md:6        ../docs/maintainer/op-development.md
csrc/src/testing/serve/bench/README.md:7  ../docs/maintainer/op-development.md
surogate/serve/tools/README.md:7          ../README.md
surogate/serve/tools/README.md:16         ../convert/qwen3_6_27b/
surogate/serve/tools/README.md:17         ../convert/qwen3_8_27b/
surogate/serve/tools/README.md:18         ../convert/qwen3_6_35b_a3b/
surogate/serve/tools/README.md:86         ../bench/README.md
```

`csrc/src/testing/serve` 下的 3 个链接按 NInfer 原仓库的目录布局编写，目标在本仓库中从未存在；`surogate/serve/tools/README.md` 的 5 个链接指向 `surogate/serve` 下并不存在的目录。

## Makefile 中失效的目标

- `Makefile:275`（`build-tests`）和 `Makefile:280`（`test-unit`）使用 CMake 目标 `unit-tests`，但 `csrc/CMakeLists.txt` 没有定义它，因此 `Makefile:293` 的默认目标 `test: test-unit` 也会失败。需要原生测试时，直接构建现有目标，例如 `cmake --build <build-dir> --target integration-tests dsl-run-state-tests`，再运行 `ctest --test-dir <build-dir>`。
- `Makefile:299`（`regression-smoke`）运行 `uv run pytest -q tests/test_regression_baseline_runner.py tests/test_moe_monitor.py --no-gpu`，这两个测试文件已在 `7a864d06`（2026-04-30）删除。
- `Makefile:30` 的 `.PHONY` 列出 `wheel` 和 `export-checkpoint`，但两者都没有规则：`make wheel` 只输出 “Nothing to be done for 'wheel'”，不会构建 wheel。实际可用的是 `wheel-cu130` 和 `wheel-dev`。

## 过时的排除路径

vendored 代码位于 `csrc/src/third_party/`，但两处排除项仍写作旧路径：

- `.pre-commit-config.yaml:30` 中 clang-format 的排除项是 `^(csrc/third_party/|csrc/build/|build/)`，匹配不到 vendored 目录。执行 `pre-commit run --all-files`，或提交改动过的 vendored 文件时，clang-format 会重排其中的 C/C++/CUDA 文件（当前 214 个）。只想格式化改动过的行时用 `git clang-format`。
- `pyproject.toml:178` 中 ruff 的 `extend-exclude` 写作 `"csrc/third_party"`。该目录下目前没有 Python 文件，所以暂时没有影响。

## `.vscode/launch.json` 指向缺失的示例

`.vscode/launch.json:22` 和 `.vscode/launch.json:33` 的调试配置分别使用 `${workspaceFolder}/examples/sft/qwen3vl-lora-bf16.yaml` 和 `${workspaceFolder}/examples/pt/qwen3-bf16.yaml`，这两个文件都不存在。

## 安装文档仍声称支持 CUDA 12.8

`docs/getting-started/installation.md:14` 和 `docs/getting-started/installation.md:45` 写的是支持 CUDA 12.8 及以上的 12.x。实际上服务引擎需要 CUDA 13.0 或更高：CUDA 12.8 和 12.9 把每个 block 的静态共享内存限制在 48 KB，无法编译 W8 和 NVFP4 GEMM（`csrc/CMakeLists.txt:885-896`），版本低于 13.0 时服务目标被禁用。发布的 wheel 也只有 CUDA 13 一种（`.github/workflows/wheel.yml:41-44`）。

## 测试：MagicMock 泄漏使同一会话中的结果依赖顺序

`tests/grpo/test_failure_exit_status.py:36`：未构建 `_surogate` 扩展时，该模块把一个模块级 `MagicMock` 写入 `sys.modules["surogate._surogate"]`，并且从不移除。在同一个 pytest 会话中，之后收集的模块都会导入这个 mock：能收集到的模块变多，部分测试的结果翻转；`tests/train/test_padded_truncation.py` 中的 `test_loader_*` 会在 `while loader.has_next(1)` 里死循环（MagicMock 恒为真），直到内存耗尽。绕行方法：每个测试文件单独用一个进程运行，或至少让该文件在独立进程中运行；在单会话运行中用 `--deselect` 排除这两个 `test_loader_*` 测试也能避免内存耗尽。
