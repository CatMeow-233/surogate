# 构建与发布

本页说明仓库中每个产物由哪条命令构建、输出在哪里、由哪个工作流发布到哪里，以及版本号和依赖版本固定在哪些文件中。日常命令都通过 `fork/dev` 调用（`fork/dev help` 列出全部子命令）；`fork/dev` 会先打印每条底层命令再执行，可以直接复制底层命令单独运行。

## 产物

| 产物 | `fork/dev` 命令 | 底层命令 | 构建目录 | 输出 | 工作流 | 触发 | 发布到 | fork 上 |
|---|---|---|---|---|---|---|---|---|
| 训练扩展 `_surogate` 与 `libsurogate-common.so` | `fork/dev build train` | `make wheel-dev`（只构建 CMake 目标 `_surogate`） | `csrc/build` | 两个 `.so` 复制到 `surogate/` 与 `.venv/lib/python3.12/site-packages/surogate/` | `wheel.yml`（作为原生 wheel 的一部分） | `v*` tag | invergent-ai 的 GitHub Release | 禁用 |
| 服务引擎：`surogate-engine`、`surogate-engine-cli`、`surogate-embed`、`_surogate_serve`、`libsinfer.so` | `fork/dev build serve` | `make serve-build` | `csrc/build-serve` | 可执行文件留在构建目录；`_surogate_serve*.so` 与 `libsinfer.so` 复制到 `surogate/` 与 `.venv` | `wheel.yml` | `v*` tag | 同上 | 禁用 |
| 训练加服务 | `fork/dev build all` | `make build-all`（= `make build` + `make serve-build`） | `csrc/build`、`csrc/build-serve` | 同上两行 | — | — | — | — |
| 语音识别 `surogate-stt`（不需要 CUDA） | `fork/dev build stt` | `make serve-stt-build` | `csrc/build-stt` | `surogate-stt` | `wheel.yml`（打进原生 wheel） | `v*` tag | 同上 | 禁用 |
| 语音合成 `surogate-tts`（不需要 Torch 和 CUDA） | `fork/dev build tts` | `make serve-tts-build` | `csrc/build-tts` | `surogate-tts` 及两个前端测试 | `wheel.yml`（打进原生 wheel） | `v*` tag | 同上 | 禁用 |
| 量化工具 `llama-quantize` | `fork/dev build quantizer` | `make quantizer` | `build/quantizer` | `surogate/serve/_llama_cpp/bin/llama-quantize` | `wheel.yml`（打进原生 wheel） | `v*` tag | 同上 | 禁用 |
| 原生 wheel `surogate-<版本>+cu130-cp312-abi3-manylinux_2_39_{x86_64,aarch64}` | 无 | 本地等价命令 `make wheel-cu130` | scikit-build 临时目录 | `dist/*.whl`（auditwheel 修复后） | `wheel.yml` | `v*` tag | invergent-ai 的 GitHub Release（附 `install.sh`） | 禁用 |
| Docker 镜像（训练加服务、语音） | 无 | `docker build -f Dockerfile.cu130` / `Dockerfile-speech.cu130`（从已发布的 wheel 安装，不复制仓库文件） | — | 镜像 | `docker.yml` | “Build Wheels” 在 `v*` tag 上成功后（`workflow_run`） | `ghcr.io/invergent-ai/surogate` | 禁用 |
| jackalope 终端界面二进制 | 无 | 在 `jackalope/` 下用 bun 构建（见 `.github/workflows/jackalope.yml`） | `jackalope/dist` | 5 个平台的二进制 | `jackalope.yml` | 推送到 `main` 且改动 `jackalope/**`，或 `jackalope-v*` tag | invergent-ai 的 GitHub Release（`jackalope-latest` 或 `jackalope-v*`） | 禁用 |
| `surogate-experimental`（纯 Python wheel、sdist、示例 zip） | `fork/dev release experimental [--tag experimental-vX]` | `python3 fork/experimental/build_release.py` | 临时目录 | `build/experimental-release/artifacts`：wheel、sdist、`experimental-examples.zip`、`README.md`、`build-info.json`、`SHA256SUMS` | `.github/workflows/experimental-release.yml` | `experimental-v*` tag（手动触发时只构建和检查，不发布） | `CatMeow-233/surogate` 的预发布 Release | **启用** |

“fork 上”一列是目标状态：上游的三个发布工作流 “Build Wheels”、“Build Docker Images” 和 “jackalope” 应在 GitHub 设置中禁用，因为它们使用 fork 没有的 Blacksmith runner，并把镜像推到写死的 `ghcr.io/invergent-ai/surogate`。禁用是仓库所有者在 GitHub 上的操作，`fork/dev setup` 会打印检查清单；在完成之前，这些工作流在 fork 上仍然是启用的。fork 不修改上游工作流文件，所以同步不会带来冲突。

实验训练器的使用方法见 [fork/experimental/docs/training.md](../experimental/docs/training.md)。从检出目录运行实验训练器或它的测试时，用 `fork/dev exp-train <配置>` 和 `fork/dev test experimental`：两者都在 `fork/` 下的隔离虚拟环境 `.venv-experimental` 中运行，因为已安装的原生 `surogate` 包会遮蔽 `surogate.experimental` 命名空间。

## 测试命令

| 命令 | 运行内容 | 需要 |
|---|---|---|
| `fork/dev test cpu` | `fork/tools/test_cpu.sh`：fork 的 GRPO 补丁测试，以及上游 `tests/` 与 `csrc/src/testing/serve` 的 pytest，分 7 个进程运行，结果与 [known_failures.txt](../tools/known_failures.txt) 比较 | `FORK_PY` 指向装有 torch 和 [requirements-cpu-lane.txt](../tools/requirements-cpu-lane.txt) 的解释器（默认 `.venv/bin/python`） |
| `fork/dev test experimental` | 在 `fork/` 下的 `.venv-experimental` 中运行 `fork/experimental/tests` | uv；无法访问 download.pytorch.org 时设 `FORK_TORCH_INDEX=pypi` |
| `fork/dev test patches` | `fork/patches/tests` | `FORK_PY`（默认 `.venv/bin/python`） |
| `fork/dev test openmp` | OpenMP 接线测试，构建目录 `build/fork-openmp` | 主机 C++ 编译器、OpenMP、CMake |
| `fork/dev test native` | 先 `make configure`，再加 `-DBUILD_TESTS=ON` 重新配置 `csrc/build`，构建并用 ctest 运行 `dsl-weight-loader-tests`、`dsl-run-state-tests`、`integration-tests` | CUDA 构建环境；GPU 用例无 GPU 时跳过 |
| `fork/dev test gpu` | `test native`，再 `make test-py`、`make serve-check`、`make grpo-test` | GPU 主机；`make grpo-test` 需要 `SUROGATE_TEST_MODEL` |

`fork/dev test native` 替代坏掉的 `make test`（见下文“上游构建的已知问题”）。

`known_failures.txt` 每行一个 pytest node id，格式为 `nodeid  # 原因; first seen at upstream <sha>`。只有在 `upstream/main` 上同样失败的用例才能列入；由 fork 引起的失败必须修复。列表中已经通过的条目会被报告为过期（stale），但不会让运行失败，此时把该行删除即可。

## 版本固定在哪里

升级下列依赖时要同时修改所有位置，否则各处会悄悄不一致。

| 内容 | 位置 |
|---|---|
| torch 2.11.0 | `pyproject.toml:101`（构建依赖）、`.github/scripts/set_cuda_version_tag.py:31`、`install.sh:85`、`Dockerfile.cu130:68`、`Dockerfile-speech.cu130:73`；fork 自有的两处：`.github/workflows/experimental-release.yml:32` 和 `fork/dev` 中的 `exp_venv` |
| NVIDIA 运行时 wheel（`nvidia-cuda-runtime==13.1.80` 等） | `install.sh:87`、`Dockerfile.cu130:70`、`Dockerfile-speech.cu130:75` |
| CUDA 架构列表 | `.github/workflows/wheel.yml:65` 与 `:76`、`Makefile:66-72`、`csrc/CMakeLists.txt:121-125` |
| auditwheel 排除的库 | `Makefile:191-205` 与 `.github/workflows/wheel.yml:215-232`；Makefile 缺少 `libcudart.so.12`、`libcublas.so.12`、`libcublasLt.so.12` 三项 |
| 仓库名 `invergent-ai/surogate` | `install.sh:13`、`surogate/cli/_jackalope.py:28`、`pyproject.toml:15`、`Dockerfile.cu130:20,23,80`、`Dockerfile-speech.cu130:35,38`、`.github/workflows/docker.yml:127-130`、`jackalope/src/links.ts:7`、`jackalope/src/modal.ts:27,40` |
| 实验包的版本与依赖 | `fork/experimental/pyproject.toml`（静态版本号）、`fork/experimental/requirements-ci.txt` |

这些上游文件中的仓库名不要改成 fork：`install.sh` 和 `surogate/cli/_jackalope.py` 下载的是上游发布的 wheel 和二进制，fork 不发布这些产物。

## 版本号

原生包的版本号来自四套互不相同的机制：

1. setuptools_scm（`pyproject.toml:104-108`）。fork 在 `:108` 加了 `--match v[0-9]*`（ledger T9），让 `experimental-v*` tag 不参与原生版本计算。
2. `.github/scripts/set_cuda_version_tag.py` 原地改写 `pyproject.toml`，写入静态版本 `<版本>+cu130`；不在 tag 上时它截掉 `git describe` 的提交距离（`:62-71`），因此手动构建会沿用上一次发布的版本号。
3. `.github/workflows/docker.yml:84-103` 根据触发它的 tag 自行计算镜像版本。
4. `install.sh` 的 `VERSION_OVERRIDE` 由 `.github/workflows/wheel.yml:337-341` 用 sed 填入。

fork 上没有 `v*` tag（`git ls-remote --tags origin` 只有 `experimental-v*`），所以在 fork 上构建的原生包版本号是回退值 `0.0.1` 或上游最近一次发布的版本号。`surogate-experimental` 的版本号写死在 `fork/experimental/pyproject.toml` 中，`--tag` 必须等于 `experimental-v<版本>`。

规则：

- 只推送 `experimental-v*` tag。不要推送 `v*` 或 `jackalope-v*` tag，它们会触发上游的发布工作流。
- 不要运行 `git push --tags`：它会把从上游取到的 tag 一起推到 fork。
- `experimental-v0.1.0a1`、`a2`、`a3` 指向 `main` 上的提交，所以同步上游时只能 merge，不能 rebase。

## 上游构建的已知问题

fork 不修改这些上游文件，只绕开它们；完整列表见 [upstream-errata.md](upstream-errata.md)。

- `make test`、`make test-unit`、`make test-integration`、`make test-all` 和 `make build-tests` 都会失败：`Makefile:275` 构建的 CMake 目标 `unit-tests` 已被上游删除。改用 `fork/dev test native`。
- `make wheel` 什么也不做：`wheel` 只出现在 `.PHONY`（`Makefile:30`）中，没有对应规则，但 `make help`（`:348`）仍然列出它。原生 wheel 用 `make wheel-cu130`。
- uv 的缓存键（`pyproject.toml:157-162`）包含 `surogate/**/*.py`：任何 Python 文件的修改都会让 `uv run` 或 `uv sync` 重新构建 CUDA 扩展，冷构建需要数小时；而 C++ 修改反而不会触发重建。因此不要在仓库根目录执行 `uv run` 或 `uv sync`，测试用 `.venv/bin/python -m pytest`（Makefile 本身也这样做）。
- 配置服务引擎需要 cargo（`csrc/cmake/serve_llguidance.cmake:2`）、FFmpeg 开发库和 libcurl；`pyproject.toml:153` 设置了 `SUROGATE_REQUIRE_SERVE=ON`，所以构建 wheel 时 CUDA 低于 13.0 会直接报错。
- `make build` 实际上也会构建服务引擎，与 `Makefile:35-38` 的注释不符：默认开启 `SUROGATE_BUILD_SERVE` 时，配置出的 Ninja 构建图中默认目标 `all` 包含 `libsinfer.so` 和 `src/serve/tts/all`（只做了 configure，没有实际构建）。`fork/dev build train` 因此用 `make wheel-dev`，它只构建 `_surogate`。
- `make regression-smoke`（`Makefile:299`）运行的两个测试文件已被上游删除。

## `scripts/` 与 `benchmarks/` 的分类

这两个目录完全属于上游，fork 不在其中添加文件（实验发布脚本已移到 `fork/experimental/build_release.py`）。“调用方”指在仓库中引用该文件的地方。

| 文件 | 类别 | 调用方 |
|---|---|---|
| `scripts/upcycle_moe.py` | 用户工具：把稠密模型扩成 MoE | `docs/getting-started/quickstart-sft.md:173` |
| `scripts/serve_prompt_parity.py` | 测试依赖：引擎与 vLLM 的 prompt 一致性 | `tests/serve/test_prompt_parity.py` 按路径加载 |
| `scripts/serve_speech_reference.py` | 测试基准生成：NeMo 参考张量 | `csrc/src/testing/serve/test_speech_model.cpp:2` |
| `scripts/gen-unicode-data.py` | 代码生成器 | `csrc/src/tokenizer/unicode_data.cpp:1` |
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
| `benchmarks/bench_glm_kda.py` | GLM KDA 基准 | `examples/sft/glm/README.md:162` |
| `benchmarks/bench_glm_memory.py` | GLM 显存基准 | `docs/guides/long-context.md:120` |
| `benchmarks/benchmark_sft.sh` | SFT 吞吐基准 | `docs/reference/benchmarks.md:109-112` |
| `benchmarks/benchmark_pt.sh` | 预训练吞吐基准 | 无 |
| `benchmarks/run-unsloth.py`、`benchmarks/run-unsloth-moe.py` | 与 Unsloth 的对照运行 | 无 |
| `benchmarks/profile.md` | 保存为 `.md` 的 nsys 命令片段，不是文档 | 无 |
