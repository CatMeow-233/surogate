# 构建与发布

本页说明仓库中每个产物由哪条命令构建、输出在哪里、由哪个工作流发布到哪里，版本号和依赖版本固定在哪些文件中，以及如何发布实验版本。日常命令都通过 `fork/dev` 调用（`fork/dev help` 列出全部子命令）；`fork/dev` 会先打印每条底层命令再执行，可以直接复制底层命令单独运行。

本页按文件和符号名（make 目标、CMake 选项、工作流步骤）指向上游代码，不写行号；需要具体位置时运行文中给出的 `git grep` 命令。

## 产物

| 产物 | `fork/dev` 命令 | 底层命令 | 构建目录 | 输出 | 工作流 | 触发 | 发布到 | fork 上 |
|---|---|---|---|---|---|---|---|---|
| 训练扩展 `_surogate` 与 `libsurogate-common.so` | `fork/dev build train` | `make wheel-dev`（只构建 CMake 目标 `_surogate`） | `csrc/build` | 两个 `.so` 复制到 `surogate/` 与 `.venv/lib/python3.12/site-packages/surogate/` | `wheel.yml`（作为原生 wheel 的一部分） | `v*` tag | invergent-ai 的 GitHub Release | 禁用 |
| 服务引擎：`surogate-engine`、`surogate-engine-cli`、`surogate-embed`、`surogate-stt`、`surogate-tts`、`_surogate_serve`、`libsinfer.so` | `fork/dev build serve` | `make serve-build` | `csrc/build-serve` | 可执行文件留在构建目录；`_surogate_serve*.so` 与 `libsinfer.so` 复制到 `surogate/` 与 `.venv` | `wheel.yml` | `v*` tag | 同上 | 禁用 |
| 训练加服务 | `fork/dev build all` | `make build-all`（= `make build` + `make serve-build`） | `csrc/build`、`csrc/build-serve` | 同上两行 | — | — | — | — |
| 只构建语音识别 `surogate-stt`（不需要 CUDA） | `fork/dev build stt` | `make serve-stt-build` | `csrc/build-stt` | `surogate-stt` | `wheel.yml`（打进原生 wheel） | `v*` tag | 同上 | 禁用 |
| 只构建语音合成 `surogate-tts`（不需要 Torch 和 CUDA） | `fork/dev build tts` | `make serve-tts-build` | `csrc/build-tts` | `surogate-tts` 及两个前端测试 | `wheel.yml`（打进原生 wheel） | `v*` tag | 同上 | 禁用 |
| 量化工具 `llama-quantize` | `fork/dev build quantizer` | `make quantizer` | `build/quantizer` | `surogate/serve/_llama_cpp/bin/llama-quantize` | `wheel.yml`（打进原生 wheel） | `v*` tag | 同上 | 禁用 |
| 原生 wheel `surogate-<版本>+cu130-cp312-abi3-manylinux_2_39_{x86_64,aarch64}` | 无 | 本地等价命令 `make wheel-cu130` | scikit-build 临时目录 | `dist/*.whl`（auditwheel 修复后） | `wheel.yml` | `v*` tag | invergent-ai 的 GitHub Release（附 `install.sh`） | 禁用 |
| Docker 镜像（训练加服务、语音） | 无 | `docker build -f Dockerfile.cu130` / `Dockerfile-speech.cu130`（从已发布的 wheel 安装，不复制仓库文件） | — | 镜像 | `docker.yml` | “Build Wheels” 在 `v*` tag 上成功后（`workflow_run`） | `ghcr.io/invergent-ai/surogate` | 禁用 |
| jackalope 终端界面二进制 | 无 | 在 `jackalope/` 下用 bun 构建（见 `.github/workflows/jackalope.yml`） | `jackalope/dist` | 5 个平台的二进制 | `jackalope.yml` | 推送到 `main` 且改动 `jackalope/**`，或 `jackalope-v*` tag | invergent-ai 的 GitHub Release（`jackalope-latest` 或 `jackalope-v*`） | 禁用 |
| `surogate-experimental`（纯 Python wheel、sdist、示例 zip） | `fork/dev release experimental [--tag experimental-vX]` | `python3 fork/experimental/build_release.py` | 临时目录 | `build/experimental-release/artifacts`：wheel、sdist、`experimental-examples.zip`、`README.md`、`build-info.json`、`SHA256SUMS` | `.github/workflows/experimental-release.yml` | `experimental-v*` tag（手动触发时只构建和检查，不发布） | `CatMeow-233/surogate` 的预发布 Release | **启用** |
| fork 原生 wheel `surogate-<版本>+cu130-cp312-abi3-manylinux_2_39_x86_64`（训练加服务，GPU 代码只含 sm_120a） | 无（在 GitHub 上手动运行 Native Wheel Release） | `fork/native/release/build-wheel.sh`，在 `nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04` 容器中运行 | `build/cp312-abi3-linux_x86_64`（`pyproject.toml` 的 `build-dir = "build/{wheel_tag}"`，scikit-build-core 填入 wheel tag） | `wheelhouse/`：wheel、`build-info.json`、`SHA256SUMS` | `.github/workflows/native-release.yml` | 只能手动运行（`workflow_dispatch`）：在 `main` 上勾选 `publish` 才发布，否则只构建 | `CatMeow-233/surogate` 的预发布 Release | **启用** |

“fork 上”一列是目标状态：上游的三个发布工作流 “Build Wheels”、“Build Docker Images” 和 “jackalope” 应在 GitHub 设置中禁用，因为它们使用 fork 没有的 Blacksmith runner，并把镜像推到写死的 `ghcr.io/invergent-ai/surogate`。禁用是仓库所有者在 GitHub 上的操作，`fork/dev setup` 会打印检查清单，`fork/dev sync` 在推送前检查它们确实已禁用；在完成之前，这些工作流在 fork 上仍然是启用的。fork 不修改上游工作流文件，所以同步不会带来冲突。

实验训练器的使用方法见 [fork/experimental/docs/training.md](../experimental/docs/training.md)。从检出目录运行实验训练器或它的测试时，用 `fork/dev exp-train <配置>` 和 `fork/dev test experimental`：两者都在 `fork/` 下的隔离虚拟环境 `.venv-experimental` 中运行，因为已安装的原生 `surogate` 包会遮蔽 `surogate.experimental` 命名空间。

## 测试命令

本表与 `fork/dev help` 是测试命令的唯一说明，其他 fork 文档只引用它们。Python 环境的准备方法见 [fork/README.md 的“首次准备环境”](../README.md#首次准备环境)。

| 命令 | 运行内容 | 需要 |
|---|---|---|
| `fork/dev test cpu` | `fork/tools/test_cpu.sh`：fork 的补丁测试（`fork/patches/tests`），以及上游 `tests/` 与 `csrc/src/testing/serve` 的 pytest，分 7 个进程运行，结果与 [known_failures.txt](../tools/known_failures.txt) 比较 | `FORK_PY` 指向装有 torch 和 [requirements-cpu-lane.txt](../tools/requirements-cpu-lane.txt) 的解释器（默认 `.venv/bin/python`） |
| `fork/dev test experimental` | 在 `fork/` 下的 `.venv-experimental` 中运行 `fork/experimental/tests` | uv；无法访问 download.pytorch.org 时设 `FORK_TORCH_INDEX=pypi` |
| `fork/dev test patches` | `fork/patches/tests` | `FORK_PY`（默认 `.venv/bin/python`），需要 [requirements-cpu-lane.txt](../tools/requirements-cpu-lane.txt)（其中有 pytest-timeout） |
| `fork/dev test tools` | `fork/tools/tests`：同步脚本、`check_divergence.py`、`fork/dev` 的超时与参数传递、`csrc/CMakeLists.txt` 挂接点的 configure 检查，都在临时 git 仓库中运行；以及发布检查 `fork/tools/release_guard.py`，对着一个假的 GitHub CLI 运行 | `FORK_PY`，需要 pytest 和 pytest-timeout；git；挂接点用例需要 CMake 和主机 C++ 编译器，缺少时跳过 |
| `fork/dev test tlc` | `fork/tlc/tests/cpu`：TLC 包的 CPU 测试，不需要原生扩展。GPU 测试 `fork/tlc/tests/gpu` 按 [TLC 运行手册](../tlc/docs/runbook.md) 在 GPU 主机上运行 | `FORK_PY`（默认 `.venv/bin/python`），需要 CPU torch 和 [requirements-cpu-lane.txt](../tools/requirements-cpu-lane.txt) |
| `fork/dev test openmp` | OpenMP 接线测试和 CPU 优化器的线程数契约（`fork/native/include/surogate_fork/omp_team.h`），构建目录 `build/fork-openmp` | 主机 C++ 编译器、OpenMP、CMake；不需要 GPU 或 CUDA |
| `fork/dev test release` | `fork/native/release/test_plan.sh`：在临时目录中建立合成的上游仓库和 fork 仓库，检查 `plan.sh` 的每个发布条件、`build-wheel.sh` 的版本检查和两者共用的 `fork/tools/release_gate.sh`，以及 `native-release.yml` 在构建之前和发布之前调用 `plan.sh` | git、bash；不访问网络 |
| `fork/dev test native` | 先 `make configure`，再加 `-DBUILD_TESTS=ON` 重新配置 `csrc/build`，构建并用 ctest 运行 `dsl-weight-loader-tests`、`dsl-run-state-tests`、`integration-tests`（fork 的 C++ 用例编译在后两个里，见 [native/README.md](../native/README.md)） | CUDA 构建环境；GPU 用例无 GPU 时跳过 |
| `fork/dev test gpu` | `test native`，再 `make test-py`、`make serve-check`、`make grpo-test` | GPU 主机；`make grpo-test` 需要 `SUROGATE_TEST_MODEL` |

`fork/dev test cpu`、`patches`、`tlc`、`tools` 中每个用例最多运行 `FORK_TEST_TIMEOUT` 秒（默认 120；最慢的 fork 用例约 10 秒，最慢的上游 CPU 用例约 17 秒），超时的用例由 pytest-timeout 按名字报告失败，例如丢失 T7 进程内传递后一直等待批次的 GRPO 用例。`FORK_PY` 缺少 pytest-timeout 时命令直接报错。`patches`、`tlc`、`tools`、`experimental` 后面的参数原样传给 pytest，例如 `fork/dev test patches -k handoff`。`fork/dev test openmp` 在 configure 时还按 `csrc/CMakeLists.txt` 中的挂接行检查 `surogate-common` 会链接 OpenMP 并得到 fork 头文件目录（见 [native/README.md 的挂接约定](../native/README.md#挂接约定)）。

`fork/dev test native` 替代坏掉的 `make test`（见下文“上游构建的已知问题”）。`fork/dev fmt` 需要 `ruff` 和 `git clang-format`（clang-format 自带的 git 子命令）。

`known_failures.txt` 每行一个 pytest node id，格式为 `nodeid  # 原因; first seen at upstream <sha>`。只有在 `upstream/main` 上同样失败的用例才能列入；由 fork 引起的失败必须修复。列表中已经通过的条目会被报告为过期（stale），但不会让运行失败，此时把该行删除即可。

## 发布实验版本

`surogate-experimental` 的版本号写死在 `fork/experimental/pyproject.toml` 中，按 PEP 440 的规范写法书写（例如 `0.1.0a4`，不能写成 `0.1.0.a4` 或 `0.1.0a04`，`build_release.py` 拒绝其他写法），tag 必须是 `experimental-v<版本>`。推送这个 tag 会运行 `.github/workflows/experimental-release.yml`，它重新测试、构建并发布预发布 Release。手动触发（包括在 `experimental-v*` tag 上手动运行）只构建和检查，不发布。上传与发布按 [一个版本只发布一次](#一个版本只发布一次) 的步骤进行：这个 tag 已有 Release（包括草稿）时不上传，文件先传到草稿，逐个校验后才发布；要修正已发布的版本只能发布新版本号。在 `main` 上按顺序操作：

1. 更新版本号：两次发布之间，`fork/experimental/pyproject.toml` 中是下一个版本加 `.dev0`（例如 `0.1.0a4.dev0`），这样从检出目录构建的 wheel 不会与已发布的 wheel 同名。把它改为新版本（去掉 `.dev0`；忘记去掉时，第 4 步的 `--tag` 检查因版本不符而失败），再用 `git grep -n '<上一个发布版本>' -- fork/experimental` 列出其余写有版本号的地方（`fork/experimental/README.md` 与 `fork/experimental/docs/training.md` 中的下载文件名，以及 `fork/experimental/RELEASE_NOTES.md`），全部改为新版本。`training.md` 的“Differences from <上一个发布版本>”一节写的是检出目录与上一个发布版本的差异，发布时删除它，需要保留的内容并入发布说明。
2. 改写 `fork/experimental/RELEASE_NOTES.md`：它就是 Release 的正文。
3. 在 GPU 主机上运行 `fork/dev test gpu`。整理后的 fork 原生 C++ 用例（`fork/native/tests`）、GRPO 周期计时指标和 T10 的 tied LM head 修复只在没有 GPU 的环境中验证过编译和 CPU 行为，下一个实验 tag 之前必须在 GPU 上通过一次（T10 还要通过 TLC 运行手册 G2 的 `test_g2_tied_head_trains_as_the_embedding`，并做 [upstream-errata.md](upstream-errata.md#全量训练时-tied-的-lm-head-停在初始值fork-已修复ledger-t10) 验证段中 Gemma 3 的全量训练检查：这两项测试都不经过 Gemma 3 的路径）；结果记在对应 topic（T1–T4、T6、T10）的 `conflict_history` 中。
4. 运行 `fork/dev test experimental` 和 `fork/dev release experimental --tag experimental-v<新版本>`，确认测试通过、产物能构建。
5. 提交并推送 `main`，等 fork CI 通过。
6. 只推送这一个 tag：

   ```bash
   git tag experimental-v<新版本>
   git push origin experimental-v<新版本>
   ```

   不要用 `git push --tags`，也不要推送 `v*` 或 `jackalope-v*` tag（见下文“版本号”）。
7. 发布之后，把 `fork/experimental/pyproject.toml` 改为下一个版本加 `.dev0`（例如发布 `0.1.0a4` 之后改为 `0.1.0a5.dev0`）并提交。此后改变用户可见行为（配置项、默认值、checkpoint 格式、数据约定）的提交，同时在 `training.md` 中新建或更新“Differences from <新版本>”一节。

## 发布原生版本

fork 的原生 wheel 由 `.github/workflows/native-release.yml` 在 GitHub 免费 runner（4 核、16 GB）上构建：工作流腾出磁盘、加 swap，然后在 CUDA 13.1 构建镜像中运行 `fork/native/release/build-wheel.sh`。脚本照搬上游 `wheel.yml` 的 `wheel` job（系统包、Rust、`set_cuda_version_tag.py`、构建目标、auditwheel 排除项和产物检查），区别是训练和服务端的 GPU 代码只编译所选架构（默认 `120a`，即 RTX 50 系列），版本号取自 `fork/native/release/VERSION`，并且不发布 `install.sh`。同步报告在上游修改 `wheel.yml` 时会提示把改动移植到这个脚本。

工作流只能手动运行：推送（包括合并到 `main`）既不构建也不发布。手动运行要求工作流文件已经在默认分支 `main` 上。一次运行能否发布由 `fork/native/release/plan.sh` 判断，条件是：

1. 运行在 `main` 上，并勾选了 `publish`。
2. `VERSION` 是不带本地部分的 PEP 440 版本，按规范写法书写（数字不带前导零，例如 `1.5.12.post1`，不能写成 `1.5.12.post01` 或 `1.5.12post1`：它们与 `1.5.12.post1` 是同一个版本，wheel 文件名相同，却对应另一个还不存在的 tag），并且形如 `<上游版本>.postN`。上游版本是被构建的提交历史中最新的上游发布 tag `vX.Y.Z`。`fork/dev sync` 合并的是上游 `main`，不一定正好是这个 tag，所以 Release 说明另外写明 fork 合并的上游提交（被构建的提交与上游 `main` 的 merge base，与 `check_divergence.py` 的定义相同）以及它比该发布多出的提交数，例如 “at `<sha>` (v1.5.12 + 3 commits)”。
3. tag `native-v<版本>` 还不存在。已发布的版本不会被重新构建或替换。
4. 被构建的提交仍在 `main` 的历史中，并且 `.github/workflows` 与 `main` 的当前提交相同：GitHub 不允许 `GITHUB_TOKEN` 在工作流文件与默认分支不同的提交上创建 tag。

plan job 在构建之前检查全部条件，不满足时运行失败，不会开始数小时的构建。release job 在发布前一刻按远端的最新状态再检查条件 1、3、4 和版本格式，然后按 [一个版本只发布一次](#一个版本只发布一次) 的步骤上传到草稿、校验文件后发布，tag 在发布时创建。草稿不创建 tag，所以 plan 看不到失败的运行留下的草稿，release job 在上传之前拒绝它。同一版本的 release job 依次执行，后一个会看到前一个创建的 tag 或 Release 并拒绝发布；勾选 `publish` 的运行也会排队，后一个运行在前一个结束后开始，然后在 plan 中停止；GitHub 只保留一个排队中的运行，再运行一次会取消之前排队的那个。

发布步骤（要发布的代码已经合并进 `main`，并且 fork CI 已经通过）：

1. 确认 `fork/native/release/VERSION`。wheel 的版本为 `<版本>+cu130`，tag 为 `native-v<版本>`。基于上游 `vX.Y.Z` 的第一个 fork 版本是 `X.Y.Z.post1`，之后每次发布把 N 加一，例如基于上游 v1.5.12 时第一个版本是 `1.5.12.post1`；同步到新的上游发布后，下一个版本从新的 `.post1` 开始。需要修改时，像其他改动一样把修改合并进 `main`。在本地运行 `fork/native/release/plan.sh build` 会打印版本、上游发布与合并的上游提交，以及发布运行会拒绝的原因（只读取远端，不需要 GitHub 权限；本地只写入 `FETCH_HEAD` 和取回的对象，不移动任何分支或远程跟踪分支）。
2. 在 GitHub 上打开 Actions → Native Wheel Release → Run workflow，“Use workflow from” 选 `main`，勾选 `publish`，然后运行。不需要也不要手动推送 `native-v*` tag。
3. 构建期间不要把修改 `.github/workflows` 的提交（例如上游同步）合并进 `main`，否则 release job 按条件 4 拒绝发布，需要在新的 `main` 上重新运行。
4. 冷构建需要数小时。构建步骤超时或运行被取消时，工作流先停止构建容器，再把已经编译完成的结果存进 Actions 缓存；这次运行中 ccache 没有写入或使用任何条目（例如在编译之前就被停止）时不保存，以免这个几乎为空的条目在下次恢复时被选中。在 GitHub 上对这次运行选择 “Re-run all jobs”（推荐，plan 会在重新构建之前检查条件）或 “Re-run failed jobs”：新的尝试恢复同一提交最新的缓存，只编译缺少的部分，被停止时正在编译的单元重新编译。两种重新运行都会在发布前重新检查，tag 或 Release（包括草稿）已经存在时拒绝发布。需要改代码才能修复时，把修复合并进 `main` 后重新运行工作流；这个版本还没有发布时不用改 `VERSION`。
5. release job 在发布之前失败（上传中断、文件校验失败，或发布请求被拒绝）时，Release 还是草稿，tag 还不存在，这个版本没有公开过。按 [一个版本只发布一次](#一个版本只发布一次) 删除草稿，然后：在原运行开始后 30 天内，对它选择 “Re-run failed jobs”，重用它的 wheel 产物，不重新构建；超过 30 天后 GitHub 不再允许重新运行这次运行（Actions 产物的保留期更长，但工作流只在原运行的新尝试中使用它），或者发布因条件 4 被拒绝时，在 `main` 上重新运行工作流并勾选 `publish`，它会重新构建。草稿从未公开，所以 `VERSION` 可以不改（plan 仍按条件 2 检查它）。Release 一旦发布，它的文件在发布前已经全部校验过；之后发现问题时不删除重发，把 `VERSION` 改成下一个版本再发布。
6. 构建在没有 GPU 的 runner 上完成，发布前没有在 GPU 上运行过。安装后在 GPU 主机上运行 `fork/dev test gpu`。

不勾选 `publish` 时，工作流可以在任意分支上运行，只构建 wheel（版本 `0.0.0.dev0`）并上传为 Actions 产物，不创建 tag；plan job 以 notice 列出发布运行会拒绝的原因。`archs` 输入可以改架构，例如 `89;120a`，每多一个架构，构建时间大致按比例增加。

GitHub 重新运行一次运行时，使用的是原运行提交上的工作流定义，不是当前 `main` 上的。在改为只能手动发布的提交 `042a070b` 之前创建的运行（由推送触发，没有 `publish` 输入、发布前的再检查和 `overwrite_files: false`）不得重新运行，无论 “Re-run all jobs” 还是 “Re-run failed jobs”。已知还在可重新运行期限内的是已取消的运行 40634738095（推送到功能分支的 `8e91b723` 触发，2026-10-28 创建，plan 已经选定 `native-v1.5.9.post1`，GitHub 允许重新运行到 2026-11-27 前后）：重新运行会在数小时的构建之后，按旧定义尝试在未合并的提交上发布这个版本号错误的 Release，挡住它的只剩 GitHub 不允许 `GITHUB_TOKEN` 在工作流文件不同的提交上创建 tag 这一条服务端限制，而这一条没有实际验证过。仓库所有者应删除这次运行（fork 的工具不改 GitHub 上的东西）：

```bash
gh run delete 40634738095 -R CatMeow-233/surogate
```

## 一个版本只发布一次

两个发布工作流（`native-release.yml` 与 `experimental-release.yml`）用同样的三步发布，由 `fork/tools/release_guard.py` 把关：

1. 上传之前，`release_guard.py absent` 要求这个 tag 还没有任何 Release，已发布的和草稿都算。只有能推送的 token 才看得到草稿，release job 的 `GITHUB_TOKEN` 有 `contents: write`。
2. softprops/action-gh-release 以 `draft: true` 创建草稿并上传全部文件，不替换已有文件（`overwrite_files: false`）。不设 `draft` 时，它先发布预发布 Release 再逐个上传，上传中断时已经公开的文件就成了一个不完整的 Release。草稿只有仓库协作者看得到。
3. `release_guard.py publish` 下载草稿上的每个文件，要求文件集合与本次运行的产物完全相同、每个文件都已上传完毕并且 SHA-256 一致，然后才把草稿改为已发布（仍是预发布，不标记为 latest）。原生版本的 tag 在这一步创建；实验版本的 tag 是推送的，保持不动。任何一项不满足时 job 失败，草稿保持未发布。

规则：同一个版本号不会以不同的字节发布两次。构建产物带时间戳，重新构建同一提交也会得到不同的 SHA-256，因此：

- 已发布的 Release 不替换、不补传，也不删除后用同一版本号重发。已发布的版本有问题时，用新版本号发布。
- release job 在发布之前失败时，留下的只是一个从未公开的草稿。删除它之后再重新发布（草稿存在时 `absent` 拒绝任何新的上传）；因为这个版本号没有公开过，重新构建后仍可以用它：

  ```bash
  gh release delete <tag> -R CatMeow-233/surogate --yes   # 不加 --cleanup-tag：实验版本的 tag 要保留
  ```

- 不经草稿发布、上传中途失败的 Release（例如在网页上手动创建，或由使用草稿之前的工作流定义创建）已经公开了一部分文件。只能用原运行的产物补完它：从原运行的 Artifacts 下载，按其中的 `SHA256SUMS` 校验，再用 `gh release upload` 上传缺少的文件；原运行的产物已过期或不完整时，保留这个 Release，把版本号改成下一个再发布。

`fork/dev test tools` 中的 `fork/tools/tests/test_release_guard.py` 对着一个假的 GitHub CLI 检查这些规则；`fork/dev test release` 与 `fork/experimental/tests/test_release_workflow.py` 检查两个工作流按这个顺序调用它。

## 版本固定在哪里

升级下列依赖时，用对应的命令找出所有位置并一起修改，否则各处会悄悄不一致。

| 内容 | 查找命令或位置 |
|---|---|
| torch 2.11.0 | `git grep -n 'torch==2\.11'`：上游的构建依赖（`pyproject.toml` 的 `[build-system]`）、`.github/scripts/set_cuda_version_tag.py`、`install.sh`、两个 Dockerfile，以及 fork 的工作流、`fork/dev` 的 `exp_venv` 和 `fork/README.md`。`fork/tools/requirements-cpu-lane.txt` 中的 `triton` 版本要与该 torch 在 PyPI 上的 Linux wheel 所要求的一致 |
| NVIDIA 运行时 wheel（`nvidia-cuda-runtime` 等） | `git grep -n 'nvidia-cuda-runtime=='`：`install.sh` 和两个 Dockerfile |
| CUDA 架构列表 | `.github/workflows/wheel.yml` 构建矩阵中的 `archs`、`Makefile` 的 `SERVE_CUDA_ARCHS`、`csrc/CMakeLists.txt` 的 `_SUROGATE_SERVE_DEFAULT_ARCHS` |
| auditwheel 排除的库 | `Makefile` 的 `build_wheel` 宏与 `.github/workflows/wheel.yml` 中 `auditwheel repair` 的 `--exclude` 列表；Makefile 缺少 `libcudart.so.12`、`libcublas.so.12`、`libcublasLt.so.12` 三项 |
| 仓库名 `invergent-ai/surogate` | `git grep -n 'invergent-ai/surogate' -- ':!fork' ':!docs' ':!*.md'`：`install.sh`、`surogate/cli/_jackalope.py`、`pyproject.toml`、两个 Dockerfile、`.github/workflows/docker.yml`、`jackalope/src` |
| 实验包的版本与依赖 | `fork/experimental/pyproject.toml`（静态版本号，见“发布实验版本”）、`fork/experimental/requirements-ci.txt` |

这些上游文件中的仓库名不要改成 fork：`install.sh` 和 `surogate/cli/_jackalope.py` 下载的是上游发布的 wheel 和二进制。fork 的原生 wheel 用它的 Release 说明中的 `uv pip install` 命令安装。

## 版本号

原生包的版本号来自四套互不相同的机制：

1. setuptools_scm（`pyproject.toml` 的 `[tool.setuptools_scm]`）。fork 在其中的 `git_describe_command` 加了 `--match v[0-9]*`（ledger T9），让 `experimental-v*` tag 不参与原生版本计算。
2. `.github/scripts/set_cuda_version_tag.py` 原地改写 `pyproject.toml`，写入静态版本 `<版本>+cu130`；不在 tag 上时它截掉 `git describe` 输出中的提交距离，因此手动构建会沿用上一次发布的版本号。
3. `.github/workflows/docker.yml` 的 “Determine version” 步骤根据触发它的 tag 自行计算镜像版本。
4. `install.sh` 的 `VERSION_OVERRIDE` 由 `.github/workflows/wheel.yml` 的 “Pin version in install.sh” 步骤用 sed 填入。

fork 上没有 `v*` tag（`git ls-remote --tags origin` 只有 `experimental-v*` 和 `native-v*`），所以用 `make` 在 fork 上构建的原生包版本号是回退值 `0.0.1` 或上游最近一次发布的版本号。`native-release.yml` 的发布运行则把 `fork/native/release/VERSION` 中的版本写入 `pyproject.toml`，只构建的运行用 `0.0.0.dev0`（见“发布原生版本”）。

规则：

- 只推送 `experimental-v*` tag；`native-v*` tag 由 `native-release.yml` 创建。不要推送 `v*` 或 `jackalope-v*` tag，它们会触发上游的发布工作流。
- 不要运行 `git push --tags`：它会把从上游取到的 tag 一起推到 fork。
- `experimental-v0.1.0a1`、`a2`、`a3` 指向 `main` 上的提交，所以同步上游时只能 merge，不能 rebase。

## 上游构建的已知问题

fork 不修改这些上游文件，只绕开它们；完整列表见 [upstream-errata.md](upstream-errata.md)。

- `make test`、`make test-unit`、`make test-integration`、`make test-all` 和 `make build-tests` 都会失败：`build-tests` 构建的 CMake 目标 `unit-tests` 已被上游删除。改用 `fork/dev test native`。
- `make wheel` 什么也不做：`wheel` 只出现在 `Makefile` 的 `.PHONY` 中，没有对应规则，但 `make help` 仍然列出它。原生 wheel 用 `make wheel-cu130`。
- uv 的缓存键（`pyproject.toml` 中 `[tool.uv]` 的 `cache-keys`）包含 `surogate/**/*.py`：任何 Python 文件的修改都会让 `uv run` 或 `uv sync` 重新构建 CUDA 扩展，冷构建需要数小时；而 C++ 修改反而不会触发重建。因此不要在仓库根目录执行 `uv run` 或 `uv sync`，测试用 `.venv/bin/python -m pytest`（Makefile 本身也这样做）。
- 配置服务引擎需要 cargo（`csrc/cmake/serve_llguidance.cmake`）、FFmpeg 开发库和 libcurl；`pyproject.toml` 的 `[tool.scikit-build.cmake.define]` 设置了 `SUROGATE_REQUIRE_SERVE=ON`，所以构建 wheel 时 CUDA 低于 13.0 会直接报错。
- `make build` 实际上也会构建服务引擎，与 `Makefile` 中 `build-all` 上方的注释不符：默认开启 `SUROGATE_BUILD_SERVE` 时，配置出的 Ninja 构建图中默认目标 `all` 包含 `libsinfer.so` 和 `src/serve/tts/all`（只做了 configure，没有实际构建）。`fork/dev build train` 因此用 `make wheel-dev`，它只构建 `_surogate`。
- `make regression-smoke` 运行的两个测试文件已被上游删除。

## `scripts/` 与 `benchmarks/` 的分类

这两个目录完全属于上游，fork 不在其中添加文件（实验发布脚本已移到 `fork/experimental/build_release.py`）。“调用方”指在仓库中引用该文件的地方，用 `git grep -n <文件名>` 可以重新核对。

| 文件 | 类别 | 调用方 |
|---|---|---|
| `scripts/upcycle_moe.py` | 用户工具：把稠密模型扩成 MoE | `docs/getting-started/quickstart-sft.md` |
| `scripts/serve_prompt_parity.py` | 测试依赖：引擎与 vLLM 的 prompt 一致性 | `tests/serve/test_prompt_parity.py` 按路径加载 |
| `scripts/serve_speech_reference.py` | 测试基准生成：NeMo 参考张量 | `csrc/src/testing/serve/test_speech_model.cpp` 的文件头注释 |
| `scripts/gen-unicode-data.py` | 代码生成器 | `csrc/src/tokenizer/unicode_data.cpp` 的文件头注释 |
| `scripts/kv_admission_e2e.py` | 服务端到端工具（`--kv-admission`） | 无自动调用 |
| `scripts/vllm_max_tokens_smoke.py` | 服务与 vLLM 的对照冒烟测试 | 无自动调用 |
| `scripts/chat.py` | 开发工具：gradio 聊天界面 | 无 |
| `scripts/check_expert_divergence.py` | 开发工具：MoE 扩展后的健康检查 | 无 |
| `scripts/dump_model_structure.py` | 开发工具：打印 HF 模型结构 | 无 |
| `scripts/generate_stubs.sh` | 可能过时：使用 pybind11-stubgen，而扩展模块是 nanobind | 无 |
| `scripts/bench_buffer_runtime_v4.py` | 基准门禁脚本 | 无 |
| `scripts/force_capture_mini_harness.sh` | 一次性验证脚本，引用的设计稿已不在仓库中 | 无 |
| `scripts/loss_diff_harness.sh` | 一次性损失回归脚本 | `scripts/force_capture_mini_harness.sh` 的注释 |
| `scripts/outer_capture_grad_diff_harness.py` | 一次性梯度对比脚本 | 无 |
| `scripts/quantize_fp8.py` | 一次性：写死模型的 FP8 量化 | 无 |
| `benchmarks/bench_glm_kda.py` | GLM KDA 基准 | `examples/sft/glm/README.md` |
| `benchmarks/bench_glm_memory.py` | GLM 显存基准 | `docs/guides/long-context.md` |
| `benchmarks/benchmark_sft.sh` | SFT 吞吐基准 | `docs/reference/benchmarks.md` |
| `benchmarks/benchmark_pt.sh` | 预训练吞吐基准 | 无 |
| `benchmarks/run-unsloth.py`、`benchmarks/run-unsloth-moe.py` | 与 Unsloth 的对照运行 | 无 |
| `benchmarks/profile.md` | 保存为 `.md` 的 nsys 命令片段，不是文档 | 无 |
