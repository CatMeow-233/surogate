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
- Gemma 3 的文本模型（`Gemma3ForCausalLM`，如 gemma-3-270m/1b；`Gemma3TextModel`，如 embeddinggemma-300m）：官方 `config.json` 不写 `tie_word_embeddings`（HF 的 `Gemma3TextConfig` 默认为 True；仓库对 gemma-3-270m-it 的记录见 `tests/serve/test_gemma3_convert.py`。这是记录下来的事实，离线测试不核对它：pin `test_the_published_gemma3_configs_still_omit_the_tie` 只在本机缓存了这两个 gated 模型时核对），上游的 Gemma 3 声明（`surogate/dsl/models/gemma3.py`）也不带这个键。`load_pretrained_config` 只在 config.json 有这个键时设置它（`csrc/src/config/pretrained_config.cpp:316-323`，默认 false），`apply_declared_config_defaults`（`dsl_model.cpp:512-524`）只在声明带这个键时补上，所以原生的 `TiedWordEmbeddings` 为 false，上面的梯度别名与存储共用都不发生。声明的映射却把 `lm_head` 指向 `model.embed_tokens.weight`：加载时它是 embedding 的副本，训练时作为第二个矩阵独立更新，而 HF 与 vLLM 按 tie 加载，用的是另一个网络；导出时第二次写 `model.embed_tokens.weight` 而抛错（`utilities/safetensors.cpp:497`），全量训练在第一次保存 checkpoint 时就中止（`runtime/training/checkpoint.cpp:123` 调用 `export_weights`）。config.json 明确写了 `tie_word_embeddings: true` 的 Gemma 3 与上面列出的模型相同。多模态的 `Gemma3ForConditionalGeneration`（4B/12B/27B）没有原生 DSL 模型，原生训练不支持它。

修复（T10）：全量训练时 tied 的 `lm_head` 就是 embedding——参数存储（没有 weight manager 时）或 weight manager（让 `lm_head` 的 master 与 work 都是 embedding 的）共用一份存储；一次更新（两部分梯度之和）就是前向读的 head。无法共用（形状或 dtype 不同、没有 `embedding` 参数）的可训练 tied head 在构建时报错。`weights.shard_*` 和 `model.safetensors` 只写 embedding（没有 `lm_head.weight`，与 HF 保存 tied 模型的方式一致；head 不共用存储的 QLoRA 和带 weight manager 的 LoRA 加载时，由 `dsl_weight_loader.cpp` 中 `DslWeightLoader::load_direct` 的 tied 回退从 embedding 回填，pin 检查这段上游代码）；旧 checkpoint 中停在初始值的 `lm_head` 在加载时被忽略。`cpu_training` 下共用的 master 只做一次 CPU 更新，tied 梯度只有一份主机副本，范数只算一次。Gemma 3 的声明补上 `tie_word_embeddings = True`（`surogate/dsl/models/gemma3.py`）：config.json 不写这个键时，原生训练按声明取 true，与 HF 取类默认值一致，T10 对官方 Gemma 3 配置才生效；config.json 明确写了的仍以它为准。LoRA 与 QLoRA 的行为不变，只是 Gemma 3 的 LoRA 训练现在也像其他 tied 模型一样，在没有 weight manager 时让冻结的 head 共用 embedding 的存储（数值不变，少一份 V×C）。代价：tied 全量训练与 LoRA 的别名一样不再把参数绑定进 persistent arena（参数留在各自的分配中，总量少了一份 V×C）。

未修复，改为拒绝（读代码并在 CPU 上重放，GPU 上未确认）：dispatch-PP 的全量训练（`lora: false`，SFT 与 GRPO 都经过 `dispatch_pp_train_step_multigpu`）在步末逐阶段从各自的 GPU 读梯度（阶段 `s` 在 GPU `s % ngpu`，`py_train.cpp`）：`embedding` 取自阶段 0 的 GPU，`lm_head` 取自损失阶段的 GPU；`dispatch_pp_write_grads`（`dsl_model_execution.cpp`）再把两者先后 memcpy 进 GPU 0 上同一块别名缓冲，后写的 `lm_head` 覆盖先写的 `embedding`。两个阶段在同一块 GPU 上（`(num_stages-1) % ngpu == 0`，包括单 GPU）时两次读到的都是两部分梯度之和，结果正确；否则 tied 矩阵只得到 LM head 的梯度，embedding 查表的梯度丢失，原生代码也不报错。例：28 层（Qwen3-0.6B/1.7B）在 2 GPU 上设 `SUROGATE_DISPATCH_STAGE_BLOCKS=14`，或用默认阶段大小跑 4、8 GPU；36 层（Qwen3-4B）跑 3 GPU；16 层（Llama 3.2 1B）跑 2、4 GPU。28 层跑 2、3 GPU 与 36 层跑 2、4、8 GPU 不受影响。这在 T10 之前就存在，而且上游在任何配置下都不更新 head；T10 之后只有上述配置丢失 embedding 一侧的梯度。dispatch-PP 在每块 GPU 上建完整副本，T10 的构建期拒绝不会触发；上游的 `docs/guides/dispatch-pp.md` 把 LoRA 写为支持的模式，全量训练照样运行（M=1），没有警告。正确的修复要按（GPU，梯度指针）去重，并把来自不同 GPU 的部分相加，不是一两行的改动，fork 没有改原生代码，而是拒绝这些配置：SFT 与 GRPO 的 trainer 构建 IR 之后调用 `surogate/_fork/dispatch_pp_tied_head.py` 中的 `refuse_lost_tied_gradient`；`parallelism: dispatch_pp`、`lora: false`、模型按原生的规则 tie（`config.json` 顶层的 `tie_word_embeddings`，没有这个键时取 DSL 声明，LFM2 恒为 tie），且 `(num_stages-1) % gpus != 0` 时抛 `ValueError`，在创建原生 trainer 之前停下。报错写明损失阶段所在的 GPU，并给出能让阶段 0 与损失阶段都在 GPU 0 上的 `SUROGATE_DISPATCH_STAGE_BLOCKS`、较少的 GPU 数，或 `lora: true`。例：28 层在 4 GPU 上可改用 `SUROGATE_DISPATCH_STAGE_BLOCKS=6`（5 个阶段）或 3 GPU；16 层在 4 GPU 上没有可用的阶段大小，只能用 3 GPU 或 LoRA。另外，`cpu_training` 配 `adamw` 或 `normuon` 没有流式分支，块参数的梯度在 GPU 上为空，与 tie 无关。

验证：在没有 GPU 的环境中为 sm_120a 编译了包含改动头文件的全部 87 个上游翻译单元和新的 C++ 测试；源码 pin `fork/patches/tests/native/test_tied_lm_head.py`（`fork/dev test patches`）在未修复的源码上失败；其中 `test_official_gemma3_configs_resolve_as_tied` 用真实的 DSL 为 gemma-3-270m-it 与 embeddinggemma-300m 的官方 config 编译 IR，检查声明带 `tie_word_embeddings: true`，`test_a_missing_tie_resolves_to_the_declaration` 检查原生代码在 config 不写这个键时取声明的值，`cpu_adamw.cpp` 中 CPU 梯度范数的指针去重也由 pin 检查；`fork/patches/tests/native/test_dispatch_pp_tied_head.py` 在真实的阶段规划上把拒绝条件与原生梯度收集的重放逐一比对（1–40 层、1–8 GPU、全部阶段大小），用真实的 IR 检查 tie 的判定，并 pin 住这段重放所依据的原生代码（`dispatch_pp_train_step_multigpu` 的梯度收集、`dispatch_pp_read_block_grads`、`dispatch_pp_write_grads`）。行为只能在 GPU 上确认：`fork/dev test native` 中的 `dsl-run-state-tests '[tied-head]'`，以及 TLC 运行手册 G2 的 `test_g2_tied_head_trains_as_the_embedding`。这两个测试都不经过 Gemma 3 的路径：C++ 测试手工构建图并直接设置 `TiedWordEmbeddings`，没有 IR、config.json 与声明的解析，也不导出；TLC 的模型总是写明 `tie_word_embeddings`，head 映射到 `lm_head.weight`，而不是 embedding 的 HF 名。Gemma 3 部分要单独在 GPU 上确认（[build-and-release.md](build-and-release.md#发布实验版本) 第 3 步，以及同步第 11 步的 GPU 复测）：以 `examples/training/full-finetune.yaml` 为底，`model` 换成 `google/gemma-3-270m-it`（官方 config.json，不写这个键），设 `max_steps: 2`、`save_steps: 1`。期望：训练、每个 checkpoint 和最终导出都不抛错（修复前第一次保存 checkpoint 就抛 `Tensor model.embed_tokens.weight has already been written`）；checkpoint 与最终输出的 `model.safetensors` 中没有 `lm_head.weight`，`model.embed_tokens.weight` 只有一份；checkpoint 的 `config.json` 写 `tie_word_embeddings: true`（最终输出目录的 `config.json` 复制自官方文件，不写这个键，HF 按类默认值 tie）。结果记在 T10 的 `conflict_history` 中。

## 打包 SFT 的整步 CUDA graph 在没有行内边界的微步上跨行或跨文档注意（fork 已修复：ledger T17）

本节同样是例外：fork 修改了上游文件。行号对应上游 `282f9901`，这些行在 fork 的修复之前与它逐字节相同。

触发：`sample_packing: true`，`use_cuda_graphs: true`（`sft_config.py` 的默认值），文档掩码开启（默认），模型能整步捕获：不是 MoE/EP，不是 QLoRA 或预量化，没有 `offload_master`，也不是带打包的 Gated DeltaNet（这些情况由 Python 侧关掉 graph，或走 eager 分支）。打包器（`surogate/train/tokenize.py` 的 `_write_packed_vectorized`）把文档首尾相接再切成行，不截断；长文档跨行时，后一行的位置号接着前一行，行内没有重置。

背景：带文档捕获整步 graph 时（`py_train.cpp` 的 `train_step_graphed`），`enable_doc_masking_pad_to_max` 为每个微步在 `mCuSeqlensCpu` 中留一段 cap+1 个偏移。每个捕获的前向从自己那一段做一次 H2D 拷贝，注意力按 `num_docs = cap`、`max_seqlen = B*T` 捕获（`graph_executor.cpp:2382-2391`）；每次 launch 之前重写各段，空位填 `B*T`。

三处缺陷：

1. 带文档捕获之后，某一步的 GA×B 行都没有行内重置：`py_train.cpp:1600-1606` 把每个微步的列表换成 `{0, B*T}`，即一个跨越所有行的文档。FA2（Hs ≤ 256）上第 b 行因果地注意第 0..b-1 行。这通常是同一长文档的前文：eager 在行首截断上下文，所以 graph 与 eager 不一致；文档恰好从行首开始时才是跨文档泄漏。mem_eff（Hs > 256，例如 Gemma 4 的全局层）每个文档只计算 `min(max_seqlen, T)` 个查询（`backend_mem_eff.cpp:120`，`kernel_forward.h:318-319`），输出预先清零（`backend_mem_eff.cpp:100`），第 1..B-1 行没有注意力输出。eager 对同一批数据的路径是 `compute_doc_masking` 返回空，然后 `clear_doc_masking`，按行稠密注意（`causal_lm_execution_profile.cpp:224-228`）。`py_train.cpp:1640-1646` 与 `1741-1746` 的 `cu.size() <= 1` 分支是死代码：每行都会结束自己的最后一个文档，列表至少有 B+1 项。验证时的蒙特卡罗（B=2，T=2048，GA=4，按块连续的微步）给出整步没有重置的概率：GSM8K 一类为 0，中位长度 1k 的对话混合为 0.07%，中位 4k 为 7.7%，中位 16k 为 45%。B=1 不受影响。
2. 捕获时某个微步没有行内重置：它的前向调用 `apply_doc_masking`，后者调用 `clear_doc_masking`，于是这个微步的注意力按稠密捕获（cuDNN），也不记录它那一段的拷贝。之后同一个微步带文档（不超过 cap，所以不重新捕获，`py_train.cpp:1598`）的步回放的仍是稠密注意力，行内的文档互相注意。同一模型的蒙特卡罗：捕获的那一步混有两种微步的概率在中位 1k、4k、16k 时分别为 45%、88%、55%；之后该槽位中带文档的微步分别占 86%、47%、18%，直到下一次重新捕获。`mamba_conv1d.cpp:119-131` 读同一份 cu_seqlens，打包的 Mamba 混合模型的卷积状态可能同样跨行或跨文档（未验证）。
3. 第一个带文档的步出现在一次不带文档的捕获之后（例如第 0 步整步都在长文档内部）：这一步因 `captured_doc_cap <= 0` 重新捕获，此前不跑 warmup，也没有 eager 调用分配过设备上的 `mCuSeqlensGpu`，`set_doc_masking` 于是在已经开始的捕获中调用 `cudaMalloc`（`graph_executor.cpp:2340-2349`）。CUDA 把 `cudaMalloc` 列为捕获期间可能不安全的调用（只有 relaxed 模式允许），而 `train_step_graphed` 用 `cudaStreamCaptureModeThreadLocal`，所以这一步预计以 CUDA 错误中止。这一处由 fork 的 CPU 重放发现，GPU 上未确认。

修复（T17，始终开启）：

- `train_step_graphed` 的扫描和回放决定移到 `fork/native/include/surogate_fork/doc_mask_staging.h`（`scan_micro_step_documents`、`plan_captured_step`）。带文档捕获的 graph 回放每个不超过 cap 的步；没有行内重置的微步写入它的按行列表 `{0, T, …, B*T}`，即 eager 的几何。B 个文档总能放进 cap：带文档的捕获至少见过一个有 B+1 个文档的微步，cap 不小于这个数。`{0, B*T}` 的合成与两处死分支已删除。
- `apply_doc_masking` 在 `compute_doc_masking` 返回空时，先调用新的 `IGraphExecutor::set_row_doc_masking_for_capture`。只有在 pad 模式下且主流正在捕获时，它才按每行一个文档设置 varlen，所以每个捕获的微步都读自己那一段。eager 调用（warmup、`validate()`）仍按行稠密，与上游相同。代价是：带文档的 graph 中没有行内重置的微步也按填充后的 varlen 几何运行，和同一 graph 中带文档的微步一样，而不再用 cuDNN 稠密注意。
- 不带文档的捕获之前，`train_step_graphed` 调用 `enable_doc_masking_pad_to_max(0, …)` 关掉 pad 模式，各段的存储保持不动（读它们的 graph 已经销毁）。这样的 graph 不读任何一段，它的微步保持稠密。
- `enable_doc_masking_pad_to_max` 在捕获之前把 `mCuSeqlensGpu` 扩到至少 cap+1 项，捕获期间不再分配。

验证（没有 GPU）：`fork/patches/tests/native/test_doc_mask_staging.py`（`fork/dev test patches`）把 `graph_executor.cpp` 中 GraphExecutor 的文档掩码函数、`causal_lm_execution_profile.cpp` 的 `compute_doc_masking` 与 `apply_doc_masking` 按源码原样编译，CUDA 由主机桩代替：捕获中的拷贝只记录，每次回放重新执行，与 CUDA graph 相同；捕获中 `cudaMalloc` 直接失败。测试用 fork 的函数重放 `train_step_graphed` 的文档掩码步骤，对每个微步把回放时注意力读到的几何和 eager 的几何换算成 (query, key) 对的集合，按 FA2 与 mem_eff 两种语义比较；40 组随机的多步序列也在其中。两个对照把上游的选择放回去（`{0, B*T}` 合成；捕获时 `clear_doc_masking`），必须显示缺陷。用上游 `488e5f00` 的 executor 与 profile 源码重跑：第 2 处缺陷的测试失败，第 3 处的测试在捕获中 `cudaMalloc`，40 组随机序列中 32 组失败（不一致，或在捕获中 `cudaMalloc`）；py_train 也换回上游逻辑时，40 组中 26 组在 FA2 与 mem_eff 上都不一致，另有 10 组在捕获中 `cudaMalloc`，跑完的各组中 402 个微步有 77 个不一致。`fork/native/tests/test-fork-doc-mask-staging.cpp`（Catch2 `[doc-mask]`，编进 `dsl-run-state-tests`，只用主机）在本机用 Catch2 的 amalgamated 源码编译运行通过。`py_train.cpp`、`binding.cpp`、`graph_executor.cpp`、`causal_lm_execution_profile.cpp`、`dsl_model_execution.cpp` 与该测试按 sm_120a 配置的编译命令编译通过。kernel 的行为只能在 GPU 上确认（5090，Qwen3-0.6B LoRA，B=2，GA=2，`sample_packing`，直接用手工构造的数组调用 `train_step_graphed`，学习率 0，与 `use_cuda_graphs: false` 的 trainer 比较每一步的 loss 和梯度范数；修复后应在 bf16 噪声内相等，修复前不相等）：

- 第 1 处：第 0 步每行都是短文档；第 1 步每行都在同一个长文档内部（位置号跨行连续），另做一个在 t=T 处恰好重置位置的变体。在 Hs > 256 的模型（Gemma 4 的全局层）上再做一次，看第 1..B-1 行是否没有输出。
- 第 2 处：第 0 步 ms0 为短文档、ms1 在一个长文档内部；第 1 步两个微步都是短文档；比较第 1 步。
- 第 3 处：第 0 步两个微步都在长文档内部，第 1 步为短文档。修复前第 1 步的捕获预计报 CUDA 错误；修复后第 1 步与 eager 相等。
- GPU 主机上 `fork/dev test native` 也运行 `dsl-run-state-tests '[doc-mask]'`。

## 整步 graph 的 pad 模式不会关闭，超过上限的 eager 调用丢掉尾部文档（fork 已加保护：ledger T18）

本节同样是例外：fork 修改了上游文件。行号对应上游 `282f9901`。`enable_doc_masking_pad_to_max`（`graph_executor.cpp:2240-2255`）设置 `mDocMaskingPadMaxDocs` 之后，再没有代码把它清零：`clear_doc_masking`、`reset_capture` 和重置 graph 都不动它。此后的 eager 调用（`validate()`，形状改变后重新捕获之前的 warmup）也走 pad 分支：`set_doc_masking` 只拷贝前 `min(count, cap+1)` 个偏移（2280），并把 `num_docs` 固定为 cap（2389）。文档比 cap 多时，从 `cu[cap]` 到 `total_q` 的查询不属于任何文档：FA2 不写它们，输出缓冲保留旧内容；mem_eff 中它们为零。这些 token 上的评估 loss 是无意义的值，不只是少了掩码。

目前发布的流程走不到这里：SFT 的评估集总是由 `_write_padded_vectorized` 写成每行一条、位置为 `arange(T)`，没有行内边界；GRPO、DPO、KD 与多模态训练都关掉了 graph。用户自己放进 `output_dir` 的打包 `eval*.bin`，或打包评估（roadmap 的 R12），会走到这里。R12 以本保护为前提。

保护（T18，始终开启）：

- pad 模式下文档数超过 cap、且主流不在捕获中时，`set_doc_masking` 按精确的几何运行（真实的 `num_docs` 与 `max_seqlen`），偏移经 fork 自己的主机与设备缓冲（`mForkExactCuSeqlensCpu`、`mForkExactCuSeqlensGpu`，只在 eager 中使用，可以增长）。捕获的 graph 引用的各段和 `mCuSeqlensGpu` 既不改写，也不重新分配。不超过 cap 的 eager 调用仍走上游的 pad 分支：结果正确，只是按填充后的几何运行。
- 在捕获中超过 cap 时不能抛异常（捕获还开着），于是打印一次错误并保留上游的截断。`train_step_graphed` 用同一次扫描确定 cap，不会走到这里。
- `stage_cu_seqlens_for_micro_step` 在不捕获时收到超过 cap 的列表就抛异常：`train_step_graphed` 在写入之前已经重新捕获，所以这只能是调用方的错误。
- T17 在不带文档的捕获之前关掉 pad 模式；带文档捕获之后 pad 模式保持开启，因为 graph 读那些段。

验证：`fork/patches/tests/native/test_doc_mask_staging.py` 中 `test_eval_with_more_documents_than_the_cap_runs_exact_geometry`（cap 4 的 graph 之后用 8 个文档评估：注意力覆盖与 eager 相同，各段的存储不变，下一步的回放仍与 eager 相同）、`test_exact_geometry_leaves_the_buffers_a_graph_holds_alone`、`test_staging_more_documents_than_the_cap_throws_outside_a_capture`；用上游源码重跑时前一个失败（只覆盖前 8 个 token）。GPU 检查（未做；只有 R12 或手工打包的评估文件才有意义）：每个微步不超过 4 个文档时捕获，然后用每个微步 8 个文档的 batch 调用 `trainer.validate`，与 `use_cuda_graphs: false` 的 trainer 比较评估 loss，修复后应在 bf16 噪声内相等。

## MoE 路由统计在重计算回放和分块训练的 KV sweep 中重复累加（fork 已修复：ledger T19）

本节同样是例外：fork 修改了上游文件。行号对应上游 `282f9901`。`moe_topk.cpp:158-192`（`dispatch_moe_topk` 中的统计块）没有判断 `mInReplay`。`recompute: true`（默认）时，反向对每一层调用 `replay_layer_forward`（`compiled_ops_execute.cpp:135`、`339-361`），它重新执行该层除损失 op 以外的全部前向 op，其中包括 MoETopK，统计 kernel 于是把每一层再加一次（`moe_losses.cu:384-394`、`532-542`、`688-698`：每次调用都给 slot 4 的层数加 1）。`get_moe_stats`（`dsl_run_state.cpp:1434-1458`）原样返回 slot 0 与 1，其余各项除以 slot 4。所以只有两个求和项 `aux_loss` 与 `z_loss` 报告为 2 倍；其余 8 个平均值（`expert_utilization`、`load_imbalance`、`active_experts`、最大与最小占比、`load_cv`、`router_entropy`、`router_confidence`）的层数也翻倍，在浮点舍入内不变。这条路径只用于监控，不影响梯度。`aux_loss` 与 `z_loss` 是一步中计入统计的每次前向在所有 MoE 层上的和，不是平均：不分块时为 GA 个微步，分块时为每个微步的每个块。

分块训练（`sequence_chunks > 1`）即使 `recompute: false` 也重复计数。它只能 eager 运行，允许 MoE（`sft_config.py:1122-1132` 只要求 batch 为 1、`lora_dropout: 0`，并关掉 graph）。`train_step_graphed` 的分块调度对每个微步 j 先做阶段 A（KV sweep：按块从左到右前向，序号 `micro_base + c`，`py_train.cpp:1402`、`1413`），再做阶段 B（从右到左对同一块重新前向，然后反向，序号 `micro_base + (eff_chunks - 1 - c)`，`1421`、`1431`）。两次前向都执行 MoETopK，统计只在序号为 0 的前向（`begin_step`）中清零。j = 0 时阶段 B 的第一次前向序号为 0，清掉阶段 A 累加的值；j ≥ 1 时两次前向的序号都大于 0，每个块计两次。于是 `aux_loss`、`z_loss` 与 slot 4 的层数一起偏大；平均值不再是各块等权的平均，第一个微步的权重只有其余微步的一半（各微步路由相同时数值不变）。`recompute: true` 时回放再加一次：上游中 j = 0 的每块计 2 次，j ≥ 1 的每块计 3 次。上游已经用 `ChunkPackMeta::kv_sweep` 标记阶段 A，并在其中跳过损失 op（`fused_lm_head_loss.cpp:243`、`cross_entropy_loss.cpp:27`），`model.h:75-78` 的注释写的就是同样的重复计数；统计块没有判断这个标记。`step()` 的分块调度 `step_chunked`（`py_train.cpp:687-752`，多模态训练循环经 `step()` 进入，`surogate/train/trainer.py:1128`）阶段 A 调用 `forward_no_save`，pack 由 `chunk_pack_meta_from_positions` 生成，不设 `kv_sweep`（`py_train.cpp:719-727`），重复计数的方式相同。

roadmap 还说整步 warmup 也会重复计数，这不对：微步 0 的前向调用 `begin_step()`（`graph_executor.cpp:1957-1958`），它最后调用 `reset_moe_stats()`，warmup 累加的值在 Python 读取统计之前就被清掉。因此 T20 去掉 warmup 不改变任何报告的统计值。

修复：回放中，以及 `train_step_graphed` 分块调度的阶段 A 中，跳过统计块（`(mInReplay || sequence_chunk_kv_sweep()) ? nullptr : mRunState.moe_stats_device()`）。阶段 B 对同一块的路由与阶段 A 相同（上游的 `reuse_ep` 依赖这一点，`model.h:71-74`），所以每个块在每个微步计一次。这也省去回放与阶段 A 中每个 MoE 层一次单 CTA 的统计 kernel。

未修复：`step_chunked` 的阶段 A 不设 `kv_sweep`，T19 不覆盖它，多模态的分块 MoE 训练在第一个之后的每个微步仍计两次。在那里设置 `kv_sweep` 会让阶段 A 同时跳过损失 op，改的是统计以外的行为（这条路径的损失是否同样重复计数，fork 没有核实），所以 T19 不改它。上游给这个阶段 A 设置 `kv_sweep` 时，pin `test_step_chunked_does_not_mark_its_kv_sweep` 失败，届时更新本节。

验证：源码 pin `fork/patches/tests/native/test_moe_stats_replay.py`（`fork/dev test patches`）检查守卫；回放在 `mInReplay` 为真时分派层内的 op，MoETopK 不被跳过；`kv_sweep` 只标记 `train_step_graphed` 分块调度的阶段 A，阶段 B 换用不带它的新 pack，`set_sequence_chunk` 按 pack 设置或清除它；`step_chunked` 不设 `kv_sweep`；`get_moe_stats` 的求和与平均；微步 0 清零。测试中还有一个按 pin 住的调度为每个（微步，块）计数的模型，守卫条件从源码读取，覆盖 1–4 个微步、每个微步 1–4 块、`recompute` 开与关，要求每块恰好计一次。在只判断 `mInReplay` 的 T19 第一版上，守卫 pin 与 6 个多微步的模型用例失败；在上游源码上 9 项失败。`moe_topk.cpp` 按 sm_120a 配置的编译命令编译通过。GPU 检查（未做）：

- 重计算：MoE SFT（例如 OLMoE-1B-7B，BF16 LoRA），固定 seed 与 batch，分别用 `recompute: true` 与 `false` 记录 `aux_loss` 与 `z_loss`：修复前两者之比为 2.0，修复后为 1.0；`expert_utilization` 与 `router_entropy` 在四次运行中都相同。
- 分块：同一模型，`sequence_chunks: 2`、`per_device_train_batch_size: 1`、`recompute: false`、学习率 0，每个微步喂同一行，分别用 GA 1 与 GA 2 记录 `aux_loss`：修复后 GA 2 与 GA 1 之比为 2.0，修复前为 3.0；`router_entropy` 在两种 GA 下相同。

## cpu_training 的全量训练不保存 CPU AdamW 的矩（fork 已修复：ledger T21）

本节同样是例外：fork 修改了上游文件，修复记在 [ledger](../patches/ledger.toml) 的 T21。行号对应上游 `282f9901`；`upstream/main` 的 `82d42c72` 中这三个文件没有变化。

现象：`cpu_training: true`、`lora: false`、`optimizer: adamw_8bit`（默认优化器，也是唯一有 CPU 步的优化器）的训练从 checkpoint 恢复时，CPU AdamW 的一阶矩和二阶矩从零重新开始，偏差校正却沿用恢复的步数，恢复后的更新比不中断的训练大数倍；同时 GPU 上多分配一份永远不用的 8-bit 优化器状态（Qwen3-8B 约 17.4 GB，0.6B 约 1.3 GB），此后每个 checkpoint 都把它写进 `adam.m`/`adam.v`。没有报错，也没有警告。SFT 默认 `resume_from_checkpoint: true`（`surogate/core/config/sft_config.py:359`），GRPO、DPO 和 Ray 的恢复也都经过同一个 `load_checkpoint`。

证据：

- `csrc/src/runtime/optimizers/adamw_8bit_optimizer.cpp:162` 的 `CPUAdamWState cpu_state`（`cpu_adamw.h`，两个 pageable 的 `std::vector<float>`）只有 `step_cpu_streaming` 使用（`:672` 起，`:728` 第一次 resize）；`momentum_container()`/`variance_container()`（`:197-203`）返回的 8-bit 容器在 `mState1` 为空时什么也不列出（`:47`），而 `step_cpu_streaming` 从不调用 `init_state`。所以 `checkpoint.cpp:86-89` 写出的 `adam.m`/`adam.v` 没有任何张量。
- 恢复时 `load_checkpoint`（`runtime/training/checkpoint.cpp:243`）调用 `DslModel::prepare_optimizer_for_checkpoint_load`（`dsl_model_weights.cpp:378-387`，没有优化器时建一个 `ADAMW_8BIT`），再到 `AdamW8BitOptimizer::prepare_for_checkpoint_load`（`adamw_8bit_optimizer.cpp:313`）→ `init_state`。`cpu_training` 只强制打开 `offload_master` 与 `offload_grads`（`binding/binding.cpp:557-559`），不打开 `offload_optimizer`，所以 `alloc_kind` 是 `ON_DEVICE`（`:269`）：每参数 2 字节的状态加每 32 个元素一个 FP16 scale，8.19B 参数时为 16.4 GB + 1.0 GB。trainer 构造时 run state 已经分配好，这份状态叠加在它之上；`on_restore_checkpoint`（`:305`）随后把容器指向它，`step()` 仍走 `step_cpu_streaming`，状态从不使用也不释放。
- `SafeTensorsReader::load_tensors`（`utilities/safetensors.cpp:287-295`）只读文件和容器都有的名字，缺少的名字不报错（在 CPU 上复现：从只有文件头的文件加载，目标中的哨兵值不变）。
- `SafeTensorWriter::write_tensor`（`:518`）与 `write_raw`（`:565`）对每个张量做 `cudaMemcpy(..., cudaMemcpyDeviceToHost)`：源是 pageable 主机内存时这在 GPU 主机上属于未定义行为，在没有驱动的主机上直接失败（`cudaErrorInsufficientDriver`）。即使把这份状态交给容器，现有的写入器也写不了它。
- 步长（按 `cpu_adamw_step` 的 FP32 运算模拟，β1 0.9、β2 0.999）：在第 50/500/2000 步恢复后，第一次更新是 0.71/1.99/2.94 倍学习率的符号步；与不中断的训练相比，噪声主导的梯度下第一步的 RMS 比为 3.1/8.5/12.9 倍，高信噪比梯度下比值在恢复后约 12-16 步达到 1.6/4.2/6.1 倍；在第 2000 步恢复时，第 100 步仍是 3.0 倍，第 500 步 1.5 倍。

修复（T21）：`fork/native/src/cpu_adamw_checkpoint.cpp` 持有 CPU 步与恢复共用的布局（`build_layout`：按 `param_names()` 的顺序，跳过已见过的 master，即 T10 原先放在 `step_cpu_streaming` 中的 tied 去重），并把矩以格式 v1 交给 checkpoint：每个 master 一个 `cpu_adamw.m.<参数名>` / `cpu_adamw.v.<参数名>`（FP32，形状同 master），每个文件另有 `cpu_adamw.format`（INT32 [1]，值为 1）。`cpu_training` 下 `prepare_for_checkpoint_load` 只布置主机上的矩，在 `init_state` 之前返回，不再分配设备上的 8-bit 状态；`load_checkpoint` 在读每个优化器文件之前调用 `require_complete`，文件缺少任何一个应有的名字就拒绝；`on_restore_checkpoint` 拒绝其他格式版本。写入器对 pageable 主机内存直接 memcpy（`surogate_fork/pageable_host.h`），pinned 与 write-combined 内存仍走 `cudaMemcpy`，权重 shard 与导出不变。新训练的 CPU 步不变（同样的槽位、偏移和 kernel）；从 T21 的 checkpoint 恢复时沿用保存的矩，偏差校正的步数仍由 trainer 传入（`step + 1`）。格式、开关、磁盘开销和兼容规则见 [performance.md 的“cpu_training 的 AdamW 状态 checkpoint”](performance.md#cpu_training-的-adamw-状态-checkpoint)。

旧 checkpoint：上游和不带 T21 的 fork 写出的 `cpu_training` checkpoint 中没有矩（没有张量，或者是此后每次保存写下的无用 8-bit 状态），无法补救。用带 T21 的构建恢复它们会报错并说明原因；设 `SUROGATE_FORK_CPU_ADAMW_ALLOW_MISSING_STATE=1` 后照旧恢复（矩为零，与上游相同，打印警告），之后的 checkpoint 带有矩。

未修复（相邻问题，只读了代码）：`cpu_training` 配 `adamw` 或 `normuon` 没有 CPU 路径（块参数的梯度在 GPU 上为空，T10 一节末尾已记），它们的 checkpoint 也不带优化器状态；它们恢复时 `DslModel::prepare_optimizer_for_checkpoint_load` 建的 `ADAMW_8BIT` 现在布置主机上的矩（8B 时 65.5 GB，第一步换成配置的优化器时释放），并且因文件里没有矩而拒绝，除非设上面的开关。应在配置阶段拒绝这种组合，不属于本修复。

验证：`fork/patches/tests/native/test_cpu_adamw_checkpoint.py`（`fork/dev test patches`）用主机编译器从生产源码（`utilities/safetensors.cpp`、`cu_file_fallback.cpp`、`cu_file_common.cpp`、`tensor.cpp`、`dtype.cpp`、`utils.cpp`）和 `fork/native/src/cpu_adamw_checkpoint.cpp` 构建 `fork/native/tests/host/cpu_adamw_checkpoint.cpp`，链接 libcudart，不用 GPU：带 tied 别名的布局与上游的步顺序一致；保存、恢复后矩逐字节相同；只有文件头的旧文件和只有 8-bit 状态的文件被拒绝，开关打开后以零恢复；参数集不同、格式版本不同、形状或 dtype 不同时拒绝；保存开关关闭时不写矩但恢复仍读。需要 CUDA 工具包（`CUDA_HOME`）以及 fmt 与 nlohmann_json 的头文件（`FORK_FC_DEPS` 或 `csrc/build/_deps`），没有时跳过并说明原因（fork CI 中跳过，只运行源码 pin）。同一程序换成上游的 `safetensors.cpp` 时，写入在 `:518` 以 `cudaErrorInsufficientDriver` 失败。三个上游文件和 fork 源文件都按 sm_120a 的编译命令编译通过。

GPU 上未确认，在 RTX 5090 上这样检查（`fork/dev build train` 之后）。第 1-4 步的运行都以 `examples/training/cpu-offload.yaml` 为底，设 `model: Qwen/Qwen3-8B`、`max_steps: 30`、`save_steps: 10`，各用一个新的 `output_dir`（`checkpoint_dir` 默认与它相同）。warmup（`warmup_ratio: 0.15`）与默认的线性衰减都按 `max_steps` 计算（`surogate/train/trainer.py:606-620`），所以恢复用的 checkpoint 必须来自同样 `max_steps: 30` 的运行：`max_steps: 12` 的运行只有 1 步 warmup，第 0-9 步的学习率之和是 30 步运行的 0.83 倍，第 10 步时权重与矩已经不同，不论矩是否恢复，接着跑都会偏离不中断的运行。trainer 只在 `step > start_step` 且 `step % save_steps == 0` 时保存（`trainer.py:1463`），没有结束时的 checkpoint。每个带矩的 checkpoint 约 98 GB（矩 65.5 GB，权重 shard 与导出约 32.8 GB），四次运行共约 460 GB。

1. 参考运行 R，不中断。`R/step_00000010/adam.m.shard_000_of_001.safetensors` 的文件头应列出 `cpu_adamw.format` 和每个 master 一个 `cpu_adamw.m.*`，文件约 32.8 GB，`adam.v` 同样（修复前这两个文件只有约 60 字节的文件头）。
2. 噪声基线 N：同样的配置再运行一次，另设环境变量 `SUROGATE_FORK_CPU_ADAMW_SAVE_STATE=0`。这个变量只决定 checkpoint 写不写矩，训练与 R 相同；`N/step_00000010` 的 `adam.m`/`adam.v` 只有文件头。
3. 恢复运行 C（修复）：新建一个空目录，用 `cp -al R/step_00000010 <目录>/` 放进 R 的第 10 步 checkpoint（硬链接，不占磁盘；恢复只读这些文件），以这个目录为 `output_dir`，设 `resume_from_checkpoint: true`、`debug_memory_breakdown: true` 运行。日志写明从第 10 步恢复；分配器的分项中没有 `AdamW8bit_OptState`（修复前约 16.2 GiB），`nvidia-smi` 的峰值与 R 相同，日志中没有 `[surogate fork T21]` 警告。
4. 阳性对照 P：按第 3 步再建一个目录，放进 `N/step_00000010`（没有矩）。不设开关时恢复报错，信息中提到 `SUROGATE_FORK_CPU_ADAMW_ALLOW_MISSING_STATE`；设为 1 后打印 `[surogate fork T21] WARNING` 并恢复，矩从零开始，偏差校正的步数仍从 11 算起（trainer 传入 `step + 1`），这正是修复前恢复的行为（只是不再分配设备状态）。
5. 以 R 为准比较 C、P 与 N：第 10-29 步的 loss 与 grad norm；`step_00000020/weights.shard_000_of_001.safetensors` 的权重；C 与 P 的 `step_00000020/adam.v.shard_000_of_001.safetensors` 中全部 `cpu_adamw.v.*` 的相对 L2 差 ‖X−R‖/‖R‖（N 没有写矩）。期望：C 与 R 的差别不超过 N 与 R 的差别，P 与 R 的差别明显更大。按 `cpu_adamw_step` 的 FP32 运算在这条学习率路径上模拟（两边梯度相同，第 10 步把一边的矩清零）：第 20 步 P 的 v 与 R 相差约 50%（只累积了 10 步，不是 20 步），m 相差 26-33%，权重之差是 R 在这 20 步中离开初始权重的距离的 8-41%；C 的 v 差应比 P 小至少一个数量级。P 与 R 的差别不大于 N 与 R 的差别时，这个检查在该硬件上分辨不出修复，记下数字，不算通过。R 与 N 逐位相同而 C 与 R 不同时，恢复还丢了矩以外的状态，记下差别和 P 的数字。
6. 用修复前的构建写出的 checkpoint 恢复：报错并提到 `SUROGATE_FORK_CPU_ADAMW_ALLOW_MISSING_STATE`；设为 1 后恢复并打印警告。
7. 在 GPU 主机上运行 `fork/dev test patches -k cpu_adamw_checkpoint`（设 `CUDA_HOME`，构建过 `csrc/build` 时自动找到 `_deps`）：这时 `is_pageable_host` 走 `cudaPointerGetAttributes` 成功的分支。
8. 可选：`Qwen/Qwen3-14B` 重复第 1、3 步，修复前在恢复时 OOM，修复后不分配设备状态。
