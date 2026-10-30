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

## 全量训练时 tied 的 LM head 停在初始值（fork 已修复：ledger T10）

本节是例外：fork 修改了上游文件，修复记在 [ledger](../patches/ledger.toml) 的 T10。行号对应上游 `282f9901`，这些文件在 fork 的 T10 之前与它逐字节相同。

现象：`tie_word_embeddings` 为 true 的模型（Qwen3 0.6B/1.7B/4B、Llama 3.2 1B/3B、Gemma 4、LFM2、LFM2-MoE/VL、Qwen3.5 小模型、TLC 各行）做原生全量训练（`lora: false`）时，前向读的 LM head 一直停在加载或初始化时的值，只有 embedding 被更新。Gemma 3 的文本模型症状不同：官方 config 被读成不 tie，head 作为第二个矩阵训练，导出抛错（见证据最后一条）。上游的示例 `examples/training/full-finetune.yaml`、`examples/training/multi-gpu.yaml`、`examples/pt/qwen3.yaml`、`examples/pt/qwen3-continue.yaml` 都在其中；默认的 LoRA 训练不受影响。

证据：

- IR 中 `lm_head` 是单独的可训练参数（`surogate/dsl/modules/embedding.py` 的 `LMHead` 注册自己的 `weight`），前向 `fused_lm_head_loss` 按名字读它。
- `csrc/src/runtime/dsl/dsl_param_store.cpp:203-204` 只在 `freeze_base`（LoRA）且没有 weight manager 时让 `lm_head` 共用 embedding 的存储；全量训练时它另有存储，加载时从 embedding 复制一份（`dsl_weight_loader.cpp:388-415`），`from_scratch` 时另取一份随机初始化（`dsl_model_weights.cpp:79-115`）。weight manager（ZeRO-3、`master_dtype` 与模型 dtype 不同、`offload_master`、`cpu_training`）为两者各建一份 master（`dsl_weight_manager.cpp:280` 起的 `allocate_weights`）。
- 它的梯度就是 embedding 的梯度缓冲（`dsl_grad_store.cpp:104-113`，两部分梯度相加）。GPU 优化器跳过已见过的梯度指针：`adamw_8bit_optimizer.cpp:237-241`、`adamw_optimizer.cpp:224`/`460`、`normuon_optimizer.cpp:317`/`542`。名字按字典序排列，`embedding` 先到，所以只有 embedding 被更新。
- checkpoint 的 `weights.shard_*`（`dsl_param_store.cpp:473`、`dsl_weight_manager.cpp:1192`）和 `model.safetensors`（`dsl_model_weights.cpp:567`、`576`）写出停在初始值的 `lm_head.weight`，`config.json` 仍是 `tie_word_embeddings: true`；遵守 tie 的使用者（vLLM、`surogate serve`）因此用的是训练后的 embedding，和训练时的网络不同。Gemma 3（`lm_head` 映射到 embedding 的 HF 名）导出时第二次写 `model.embed_tokens.weight` 而抛错（`utilities/safetensors.cpp:497`）；LFM2-MoE/VL 的 `tied_to` 映射在导出时抛 “unsupported HF export mapping”（`dsl_model_weights.cpp:562`）。
- `cpu_training` 加 `adamw_8bit` 是例外：`step_cpu_streaming`（`adamw_8bit_optimizer.cpp:734-786`）不去重，两份 master 拿到同一个梯度、同步更新；但两个名字各有一份主机梯度（`dsl_grad_store.cpp:797-806`），CPU 上的梯度范数（`cpu_adamw.cpp:106`）把 tied 梯度算了两次，裁剪与设备路径不同。
- 来历：`2580d6cc`（2026-01-28）把旧模块化实现移植到 DSL 时只带过来梯度别名和指针去重，没有带过来共用存储；`4ac6a3d8`（2026-02-03）之后 DSL 是唯一的训练路径。上游没有对应的 issue 或测试：全量训练的 GPU 测试多用 `tie_word_embeddings=False` 的小模型。
- Gemma 3 的文本模型（`Gemma3ForCausalLM`，如 gemma-3-270m/1b；`Gemma3TextModel`，如 embeddinggemma-300m）：官方 `config.json` 不写 `tie_word_embeddings`（HF 的 `Gemma3TextConfig` 默认为 True；仓库对 gemma-3-270m-it 的记录见 `tests/serve/test_gemma3_convert.py`），上游的 Gemma 3 声明（`surogate/dsl/models/gemma3.py`）也不带这个键。`load_pretrained_config` 只在 config.json 有这个键时设置它（`csrc/src/config/pretrained_config.cpp:316-323`，默认 false），`apply_declared_config_defaults`（`dsl_model.cpp:512-524`）只在声明带这个键时补上，所以原生的 `TiedWordEmbeddings` 为 false，上面的梯度别名与存储共用都不发生。声明的映射却把 `lm_head` 指向 `model.embed_tokens.weight`：加载时它是 embedding 的副本，训练时作为第二个矩阵独立更新，而 HF 与 vLLM 按 tie 加载，用的是另一个网络；导出时第二次写 `model.embed_tokens.weight` 而抛错（`utilities/safetensors.cpp:497`），全量训练在第一次保存 checkpoint 时就中止（`runtime/training/checkpoint.cpp:123` 调用 `export_weights`）。config.json 明确写了 `tie_word_embeddings: true` 的 Gemma 3 与上面列出的模型相同。多模态的 `Gemma3ForConditionalGeneration`（4B/12B/27B）没有原生 DSL 模型，原生训练不支持它。

修复（T10）：全量训练时 tied 的 `lm_head` 就是 embedding——参数存储（没有 weight manager 时）或 weight manager（让 `lm_head` 的 master 与 work 都是 embedding 的）共用一份存储；一次更新（两部分梯度之和）就是前向读的 head。无法共用（形状或 dtype 不同、没有 `embedding` 参数）的可训练 tied head 在构建时报错。`weights.shard_*` 和 `model.safetensors` 只写 embedding（没有 `lm_head.weight`，与 HF 保存 tied 模型的方式一致；head 不共用存储的 QLoRA 和带 weight manager 的 LoRA 加载时，由 `dsl_weight_loader.cpp` 中 `DslWeightLoader::load_direct` 的 tied 回退从 embedding 回填，pin 检查这段上游代码）；旧 checkpoint 中停在初始值的 `lm_head` 在加载时被忽略。`cpu_training` 下共用的 master 只做一次 CPU 更新，tied 梯度只有一份主机副本，范数只算一次。Gemma 3 的声明补上 `tie_word_embeddings = True`（`surogate/dsl/models/gemma3.py`）：config.json 不写这个键时，原生训练按声明取 true，与 HF 取类默认值一致，T10 对官方 Gemma 3 配置才生效；config.json 明确写了的仍以它为准。LoRA 与 QLoRA 的行为不变，只是 Gemma 3 的 LoRA 训练现在也像其他 tied 模型一样，在没有 weight manager 时让冻结的 head 共用 embedding 的存储（数值不变，少一份 V×C）。代价：tied 全量训练与 LoRA 的别名一样不再把参数绑定进 persistent arena（参数留在各自的分配中，总量少了一份 V×C）。

未修复，改为拒绝（读代码并在 CPU 上重放，GPU 上未确认）：dispatch-PP 的全量训练（`lora: false`，SFT 与 GRPO 都经过 `dispatch_pp_train_step_multigpu`）在步末逐阶段从各自的 GPU 读梯度（阶段 `s` 在 GPU `s % ngpu`，`py_train.cpp`）：`embedding` 取自阶段 0 的 GPU，`lm_head` 取自损失阶段的 GPU；`dispatch_pp_write_grads`（`dsl_model_execution.cpp`）再把两者先后 memcpy 进 GPU 0 上同一块别名缓冲，后写的 `lm_head` 覆盖先写的 `embedding`。两个阶段在同一块 GPU 上（`(num_stages-1) % ngpu == 0`，包括单 GPU）时两次读到的都是两部分梯度之和，结果正确；否则 tied 矩阵只得到 LM head 的梯度，embedding 查表的梯度丢失，原生代码也不报错。例：28 层（Qwen3-0.6B/1.7B）在 2 GPU 上设 `SUROGATE_DISPATCH_STAGE_BLOCKS=14`，或用默认阶段大小跑 4、8 GPU；36 层（Qwen3-4B）跑 3 GPU；16 层（Llama 3.2 1B）跑 2、4 GPU。28 层跑 2、3 GPU 与 36 层跑 2、4、8 GPU 不受影响。这在 T10 之前就存在，而且上游在任何配置下都不更新 head；T10 之后只有上述配置丢失 embedding 一侧的梯度。dispatch-PP 在每块 GPU 上建完整副本，T10 的构建期拒绝不会触发；上游的 `docs/guides/dispatch-pp.md` 把 LoRA 写为支持的模式，全量训练照样运行（M=1），没有警告。正确的修复要按（GPU，梯度指针）去重，并把来自不同 GPU 的部分相加，不是一两行的改动，fork 没有改原生代码，而是拒绝这些配置：SFT 与 GRPO 的 trainer 构建 IR 之后调用 `surogate/_fork/dispatch_pp_tied_head.py` 中的 `refuse_lost_tied_gradient`；`parallelism: dispatch_pp`、`lora: false`、模型按原生的规则 tie（`config.json` 顶层的 `tie_word_embeddings`，没有这个键时取 DSL 声明，LFM2 恒为 tie），且 `(num_stages-1) % gpus != 0` 时抛 `ValueError`，在创建原生 trainer 之前停下。报错写明损失阶段所在的 GPU，并给出能让阶段 0 与损失阶段都在 GPU 0 上的 `SUROGATE_DISPATCH_STAGE_BLOCKS`、较少的 GPU 数，或 `lora: true`。例：28 层在 4 GPU 上可改用 `SUROGATE_DISPATCH_STAGE_BLOCKS=6`（5 个阶段）或 3 GPU；16 层在 4 GPU 上没有可用的阶段大小，只能用 3 GPU 或 LoRA。另外，`cpu_training` 配 `adamw` 或 `normuon` 没有流式分支，块参数的梯度在 GPU 上为空，与 tie 无关。

验证：在没有 GPU 的环境中为 sm_120a 编译了包含改动头文件的全部 87 个上游翻译单元和新的 C++ 测试；源码 pin `fork/patches/tests/native/test_tied_lm_head.py`（`fork/dev test patches`）在未修复的源码上失败；其中 `test_official_gemma3_configs_resolve_as_tied` 用真实的 DSL 为 gemma-3-270m-it 与 embeddinggemma-300m 的官方 config 编译 IR，检查声明带 `tie_word_embeddings: true`，`test_a_missing_tie_resolves_to_the_declaration` 检查原生代码在 config 不写这个键时取声明的值，`cpu_adamw.cpp` 中 CPU 梯度范数的指针去重也由 pin 检查；`fork/patches/tests/native/test_dispatch_pp_tied_head.py` 在真实的阶段规划上把拒绝条件与原生梯度收集的重放逐一比对（1–40 层、1–8 GPU、全部阶段大小），用真实的 IR 检查 tie 的判定，并 pin 住这段重放所依据的原生代码（`dispatch_pp_train_step_multigpu` 的梯度收集、`dispatch_pp_read_block_grads`、`dispatch_pp_write_grads`）。行为只能在 GPU 上确认：`fork/dev test native` 中的 `dsl-run-state-tests '[tied-head]'`，以及 TLC 运行手册 G2 的 `test_g2_tied_head_trains_as_the_embedding`。
