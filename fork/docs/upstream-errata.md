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

未修复（相邻问题，只读了代码）：`cpu_training` 配 `adamw` 或 `normuon` 没有 CPU 路径（块参数的梯度在 GPU 上为空，T10 一节末尾已记），它们的 checkpoint 也不带优化器状态；它们恢复时 `DslModel::prepare_optimizer_for_checkpoint_load` 建的 `ADAMW_8BIT` 现在布置主机上的矩（8B 时 65.5 GB，第一步换成配置的优化器时释放），并且因文件里没有矩而拒绝，除非设上面的开关。应在配置阶段拒绝这种组合，不属于本修复。另外，换训练方式的恢复只在一个方向上被拒绝：`cpu_training` 从 GPU 训练的 checkpoint 恢复时被拒绝；GPU 训练（不设 `cpu_training`）从 `cpu_training` 的 checkpoint 恢复时不检查（`require_complete` 只检查 T21 的容器），读取器跳过 `cpu_adamw.*`，`init_state` 初始化的 8-bit 状态（矩为零）留在原处，`on_restore_checkpoint` 把它标为已初始化，没有警告，与上游从任何 `cpu_training` checkpoint 恢复时相同。

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

## cuDNN 注意力忽略 softmax_scale，Gemma3-27B 的全局层按 1/sqrt(128) 训练（fork 已修复：ledger T22）

本节同样是例外：fork 修改了上游文件。行号对应上游 `282f9901`；这几个文件在 `488e5f00` 与上游 `82d42c72` 中相同。

缺陷：flash_attention op 的 `softmax_scale` 为 0 时表示 kernel 默认的 1/sqrt(Hs)，非零时按原值使用（`graph_compiler.h:333-337`）。前向与反向的派发都把它放进 `AttentionParams::softmax_scale`（`flash_attention.cpp:475`、`681`），反向 op 从前向复制这个属性（`:932`）；mem_eff、flash_varlen、kvprefix 与 custom 都使用它。cuDNN 后端不用：`CuDNNAttention::supports()`（`backend_cudnn.cpp:35-63`）不检查它，`forward()`、`backward()` 不传递它，`attention_cudnn.cpp` 在前向与反向都把 `1/sqrtf(HS)` 写进 cuDNN 的 `attn_scale`（`:330`、`:394`）。注册表取第一个支持的后端，优先级为 kvprefix 110（只用于序列分块）> cuDNN 100 > mem_eff 95 > flash_varlen 90（`attention_backend.h:32-38`）。因此因果、BF16、Hs ≤ 128 且为 8 的倍数、没有滑动窗口、`cu_seqlens` 为空的 op 总是按 1/sqrt(Hs) 计算，与 op 声明的比例无关。

触发：

- 文本版 Gemma3-27B（`Gemma3ForCausalLM`，head_dim 128，`query_pre_attn_scalar` 168）。DSL 发出 `168**-0.5`（`surogate/dsl/blocks/gemma3.py:259-271`），HF 的 `modeling_gemma3.py` 同样用 `query_pre_attn_scalar**-0.5`。62 层中每 6 层有一个全局层，共 10 层走 cuDNN；滑动层（窗口 1024）cuDNN 不接受，由 flash_varlen 按正确的比例计算。cuDNN 用的是 0.0883883，不是 0.0771517，logits 放大 1.14564 倍，相当于 softmax 温度 0.873。前向与反向用同一个错误的比例，梯度与错误的前向自洽：训练的是另一个网络，与 HF、vLLM 推理时的网络不同，导出的 adapter 与 GRPO 训练端的 logprob 也随之偏离。
- 只发生在没有行内文档边界的微步上，这时 `compute_doc_masking` 返回空，`cu_seqlens` 为空：不打包的 SFT、每行只有一个（截断或填充的）文档的打包批次、评估（`_write_padded_vectorized` 写的行）、`doc_masking: false`，以及运行时 decode 路径的 prefill（`flash_attention.cpp` 的 prefill 调用 `select(params).forward`，之后的 paged decode 用正确的比例，同一进程内 prefill 与 decode 不一致）。带行内边界的微步走 flash_varlen 或 mem_eff，比例正确，所以同一次训练会按批次内容混用两种比例。
- 官方的 `google/gemma-3-27b-*` 是 `Gemma3ForConditionalGeneration`，没有原生 DSL 模型（见上文 T10 一节）。走到这里的是 `Gemma3ForCausalLM` 的 27B 检查点：Google 发布的纯文本 `google/medgemma-27b-text-it`（模型卡用 `AutoModelForCausalLM` 加载，直接走这条路径；多模态的 `google/medgemma-27b-it` 是 `Gemma3ForConditionalGeneration`），以及文本版导出（`Gemma3ForCausalLM.save_pretrained`、社区的无视觉导出）。MedGemma 的 config.json 在本环境取不到（huggingface.co 不可达）：`Gemma3ForCausalLM`、head_dim 128 与 `query_pre_attn_scalar` 168 来自模型卡和一份第三方转录（GitHub `xynexus/hipfire` 的 `crates/hipfire-arch-gemma3/src/config.rs`，注明 2026-06-19 核对过 config.json），要在能访问 Hugging Face 的主机上用 `AutoConfig.from_pretrained("google/medgemma-27b-text-it")` 确认。其他检查点按 config 判断：`architectures` 为 `Gemma3ForCausalLM`、head_dim ≤ 128 且是 8 的倍数、`query_pre_attn_scalar` 不等于 head_dim 的都受影响。不受影响：Gemma 3 的 270M/1B/4B/12B 与 EmbeddingGemma（head_dim 256，超过 cuDNN 的上限 128）、已发布的 Gemma 4（256/512）、DeepSeek V4（比例等于 1/sqrt(d)，且都是滑动窗口）。Gemma 4 风格的配置若有 Hs ≤ 128 的全注意力层，`softmax_scale=1.0` 会被换成 1/sqrt(Hs)；只有合成配置能走到这里。
- 已有的绕行只覆盖窄情况：序列分块模式（kvprefix）；`MoeRolloutParity` 把 `cudnn_handle` 置空（`flash_attention.cpp:490`，注释写明 cuDNN 的封装固定了比例）；带行内边界的步。

影响的估计（CPU 上的模拟，真实权重下的幅度未知）：GQA 因果注意力，Hq 32、Hkv 16、Hs 128，T=256，RMS 归一化的 Q/K 增益 1–2 时，每个受影响层注意力输出的相对 L2 误差为 8.8–15.8%，每行平均 KL 为 0.009–0.025 nats。

修复（T22，始终开启）：

- `attention_forward_cudnn` 与 `attention_backward_cudnn`（`attention_cudnn.cpp` 的 BF16 版本、`attention_custom.cu` 的 FP32 与 Tensor 重载、`attention_kernels.h` 中的声明）增加 `float scale` 参数，约定与 flash_varlen、custom 的封装相同：0 取 1/sqrt(HS)，非零按原值使用。`CuDNNAttention` 把 `p.softmax_scale` 传给前向、反向，以及分块反向（`AttBwdChunks > 1`）的每一块。
- `attn_scale` 本来就是 pass-by-value 的图张量，每次 `execute()` 从 variant pack 读取，图缓存只以形状为键：不重建图，workspace 大小与后端选择都不变，这 10 层仍走 cuDNN。
- 没有声明比例的 op 取 `1.f / sqrtf(HS)`，HS ≤ 4096 时与上游前向的 `1.0 / sqrtf(HS)` 逐位相同（逐个核对过），所以不声明比例的模型结果不变。
- 没有改动：`flash_attention.cpp` 中 `MoeRolloutParity` 仍把 `cudnn_handle` 置空；它的注释（“cuDNN's wrapper fixes the softmax scale”）已经过时，parity 模式的行为不变。

checkpoint 与恢复：T22 不改变 checkpoint、adapter 与优化器状态的格式，旧文件照常加载，但同一组权重算出的结果变了。T22 之前训练的上述 Gemma3-27B 文本模型的 checkpoint 或 adapter（LoRA 或全量）在 T22 之后恢复训练时，从恢复的那一步起 10 个全局层按 `168**-0.5` 计算（logits 为之前的 0.873 倍）：没有行内文档边界的微步，loss 与 logprob 在恢复处跳变，而优化器的矩是对着旧网络累积的。原生评估（验证集总是逐行填充、不打包：`surogate/train/tokenize.py` 对 `validation` 传 `packing=False`）与 decode 的 prefill 对同一 checkpoint 给出的数值也与 T22 之前不同。T22 之前导出的 adapter 是对着另一个网络训练的，在 HF、vLLM 中本来就与训练时不一致；T22 之后原生评估与 prefill 用的才是 HF、vLLM 的比例。跨越修复比较 loss 曲线时只比较带行内文档边界的打包微步：它们修复前后都走 flash_varlen，比例不变。没有开关能恢复旧的比例；要按旧行为跑完一次训练，只能用 T22 之前的提交构建的原生 wheel。

验证（没有 GPU）：`fork/patches/tests/native/test_cudnn_softmax_scale.py`（`fork/dev test patches`）把 `backend_cudnn.cpp` 的 `CuDNNAttention`、`attention_cudnn.cpp` 的 BF16 封装、`attention_custom.cu` 的 FP32 与 Tensor 重载，以及 `attention_backend.h` 的 `AttentionParams`、`AttentionBackend` 与优先级按源码原样编译；CUDA、cuDNN 与 cuDNN frontend 的图由主机桩代替，桩图在每次 `execute()` 时记录 variant pack 中的 `attn_scale`。测试用真实的 DSL 编译 Gemma3-27B 的文本配置，把 62 个 flash_attention op 逐个交给探针：不打包时 10 个全局层选中 cuDNN，前向与反向收到的比例必须等于 op 声明的 `168**-0.5`；打包时 cuDNN 拒绝；Hs 为 8–128 且不声明比例时与上游的值相同；1.0 按原值使用；分块反向的每一块用同一个比例；FP32 重载也传递比例。另外 pin 住修复依赖的上游代码：两个方向的派发都复制 `softmax_scale`，反向 op 继承这个属性，两个图构建函数中的 `attn_scale` 是 pass-by-value，缓存键不含比例。用上游源码（`488e5f00`）运行时，Gemma3-27B 的测试失败（收到 `0x1.6a09e6p-4`，即 1/sqrt(128)；声明的是 `0x1.3c0ca4p-4`），显式比例与分块反向的测试也失败，FP32 探针无法编译。GPU 检查用的转储比较脚本 `fork/patches/tests/native/t22_attention_dump_check.py` 也在同一个测试文件中检查：用 torch 的 SDPA 代替 kernel，按两种比例产生原生布局的 bf16 转储，脚本必须分别报告 `declared` 与 `default`；另外 pin 住 `flash_attention.cpp` 中转储的名字、文件格式，以及它们位于 kernel 调用之后。改动的 3 个翻译单元和另外 7 个包含 `attention_kernels.h` 的翻译单元按 sm_120a 配置的编译命令编译通过。

GPU 检查（未做）：在 5090 上用 transformers 构建一个随机权重的 `Gemma3ForCausalLM`：`Gemma3TextConfig(hidden_size=512, intermediate_size=1024, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=128, query_pre_attn_scalar=168, sliding_window=64, layer_types=["sliding_attention"] * 5 + ["full_attention"], vocab_size=4096)`，`torch.manual_seed(0)`；把第 5 层的 `self_attn.q_norm.weight` 与 `k_norm.weight` 设为 1.0（增益 2，让 logits 足够大，随机初始化下温度的差别才看得出），转成 bf16 后 `save_pretrained`。按 `tests/test_onboarding_gemma4.py` 的方式建原生 trainer（`PretrainedConfig.from_pretrained`、`compile_model_for_hf`、`use_cuda_graphs=False`、LoRA rank 8），设 `SUROGATE_DEBUG_ATTN_SELECT=1`、`SUROGATE_DEBUG_ATTENTION_RUNTIME=1`、`SUROGATE_DEBUG_ATTENTION_DUMP_LAYER=5` 与 `SUROGATE_DEBUG_DUMP_DIR=<目录>`：`flash_attention.cpp` 把第 5 层 attention 的输入与输出按 float32 写进这个目录（`attn_live.layer5.<名字>.bin` 与记录形状的 `.json`，每次调用覆盖上一次）。判据是 kernel 本身的输出，不是 loss：`fork/patches/tests/native/t22_attention_dump_check.py` 从转储的 qkv（与 d_out）按 float64 计算因果 GQA 注意力在声明比例 `168**-0.5` 与默认比例 1/sqrt(128) 下的前向与解析反向，报告 kernel 的 out、lse（自然对数）与 d_qkv 离哪一个参考近；每一项都只与声明比例一致时以 0 退出，否则以 1 退出。下面的数值是 CPU 上用同一个模型、以 torch 的 SDPA 代替 kernel 模拟的结果。结果（日期、提交、脚本输出）记在 T22 的 `conflict_history` 中。

1. 前向：B=1、T=256，一条随机 token 序列，`trainer.validate(inputs, targets)`（`targets` 为左移一位的 `inputs`，末位 -100），然后运行 `python3 fork/patches/tests/native/t22_attention_dump_check.py <目录> --layer 5 --hq 4 --hkv 2 --scale 0.07715167498104596`。期望：`[ATTN] dir=fwd layer=5` 一行是 `backend=cudnn`、`packed=0`，其余各层是 `flash_varlen`；修复后脚本对 out 与 lse 都报告 `declared`、以 0 退出：out 与声明比例参考的相对 L2 约 0.2%（阈值 2%），与默认比例参考约 7%；lse 与声明比例参考的最大差约 1e-6 nats（阈值 0.05），与默认比例参考约 2 nats（5 条随机序列中 1.9–2.3）。修复前两边对调，脚本报告 `default`、以 1 退出。
2. 反向：清空目录，对同一批数据做一次 `trainer.step`（一个微步），再运行同一条命令；目录中多了 `qkv_bwd`、`d_out` 与 `d_qkv`，脚本也比较 d_qkv。期望：`[ATTN] dir=bwd layer=5` 是 `backend=cudnn`；修复后 d_qkv 与声明比例参考的相对 L2 在阈值 3% 以内（模拟约 0.2%；把 P 与 dS 舍入为 bf16 后约 0.3%），与默认比例参考约 14%，脚本以 0 退出；修复前报告 `default`。
3. loss 只作粗查：把第 1 步的 loss 与 HF（`attn_implementation="eager"`，bf16，`labels=inputs`）比较。它分辨不了修复前后：在这个模型上（loss 约 8.4），缺陷只让 loss 变化 1.6e-4–8.7e-4（5 条随机序列，fp32），而 HF 自己的 bf16 与 fp32 之差可达 1.05e-3，sdpa 与 eager 之差可达 3.8e-4。两者相差超过 1e-2 时另有问题。早先的写法以 loss 为判据：修复前的差“远大于 bf16 噪声”在这个模型上不成立；比较两种文档布局训练一步后的 loss，其差距没有量化过。两者都已由上面两步代替。

## 交叉熵 kernel 按 int 计算行偏移，一个 LM-head 分块超过 2^31 个 logits 时越界读写（fork 已修复：ledger T23）

本节同样是例外：fork 修改了上游文件。行号对应上游 `282f9901`；`fused_classifier.cu` 在 `488e5f00` 与上游 `82d42c72` 中相同。

缺陷：`csrc/src/kernels/fused_classifier.cu` 中四个交叉熵 kernel 用 int 计算 logits 与 dlogits 的下标 `row * P + i`（行号、`P` 与 `i` 都是 int），共 7 处：

- `chunked_cross_entropy_forward_kernel`（V > 65536）：`:680` 读每个词表分块的 logits 求 LSE，`:699` 读目标 logit。
- `chunked_cross_entropy_backward_kernel`：`:768` 给 target 为 -100 的行写零，`:788` 写梯度。
- `cross_entropy_forward_kernel`（V ≤ 65536）：`:598` 读目标 logit。
- `cross_entropy_backward_kernel`：`:626`、`:653` 两处写。

同一文件中两个反向 kernel 的读取（`:640`、`:775`）、`prepare_softmax_blockwide3`、`fused_classifier_kernel5`、`argmax_correct_kernel`、`extract_logprobs_kernel`、`scale_logits_rows_kernel`、softcap 与 `kd_*`/`candidate_*` kernel 已经用 int64 的行基址。nvcc 13.1 为 compute_120a 生成的 PTX 中，这 7 处是 `mul.lo.s32`/`mad.lo.s32`（PTX 规定按 32 位回绕）再 `mul.wide.s32` 成字节偏移：先回绕，再符号扩展。

触发：这些 kernel 一次处理一个 LM-head 分块的行（`fused_lm_head_loss.cpp` 的 SFT 路径 `:477`、`:729`，GRPO/DPO 默认的行压缩路径 `:976`、`:1207`，以及 `cross_entropy_loss.cpp:58`、`:118`），logits 缓冲区是 `{B*T/lmhead_chunks, V}`（`dsl_run_state.cpp:618-621`）。分块行数 × V 超过 2^31 时回绕：V=151936（Qwen2.5/3）从 14,135 行起，128256（Llama 3）从 16,744 行起，201088（GPT-OSS）从 10,680 行起，248320（Qwen3.5）从 8,649 行起，262144 从 8,193 行起；V=65536（LFM2，走非分块 kernel）从 32,769 行起。`surogate/core/config/sft_config.py:726-759` 的自动分块：全量训练按约 256 行一块、最多 32 块；LoRA 默认不自动分块，只在 LoRA+recompute 时取第一个能整除 BT 的块数（fp8_hybrid 依次试 2、4、8，其他 recipe 试 8、4、2），都不能整除或 `recompute: false` 时保持 1 块。GRPO 与 DPO 沿用这段逻辑，并默认走行压缩路径，分块行数是该块中的有效行数。V=151936 时可达的配置：

- fp8_hybrid LoRA+recompute，BT ≥ 28,270（2 块），例如 `docs/guides/long-context.md` 开头的配置（`sequence_len: 32768`、`recompute: true`、batch 1）配 fp8_hybrid：两块各 16,384 行，第 14,134 行起共 13.7% 的元素回绕。`sequence_len` 不写时取 `max_model_len`（`sft_config.py:708-709`；Qwen3 为 40,960），batch 1 即每块 20,480 行。
- LoRA 且 `recompute: false`，BT ≥ 14,135（1 块），例如 4096×4、8192×2、16384×1。
- bf16/nvfp4 LoRA+recompute，BT ≥ 113,080（8 块），例如 131,072 个 token（`docs/guides/long-context.md` 的表中列出的长度）。
- BT 不能被候选块数整除时退回 1 块。全量训练的自动分块取能整除 BT 的最大 2 的幂（最多 32）：BT 是 32 的倍数时，约 45 万个 token 以下不触发；BT 含 2 的因子少时阈值随之降低（奇数 BT 只有 1 块，从 14,135 起触发）；关闭自动分块（`auto_lmhead_chunks: false`）或显式给 `lmhead_chunks` 时按实际块数计算。

只有 1 块的 fp8_hybrid LoRA 配置（上面第 2、4 项中 `recipe: fp8_hybrid` 的那些）在反向还要让同一个分块经过 `quant.cu` 的 `abs_max` 与 E5M2 量化，那两个 kernel 的循环下标同样是 int，而且在比 2^31 略少的元素数上就回绕（V=151936 时 5090 上从 14,121 行起），见[下一节](#fp8-abs_max-与量化-kernel-的循环下标是-int一次处理约-231-个元素时越界读写fork-已修复ledger-t23)。只改本节的 7 处不能让这些配置正确运行；T23 同时修复了两处。

影响（读代码与 CPU 模拟，GPU 上未确认）：回绕的前向读取让损失与保存的 LSE（反向用它）取到错误的值，或触发非法地址。反向是原地运算（`dlogits` 就是 `logits`）：读取正确，写入错位，回绕行的真实位置留着原始 logits，dX、dW（以及 LoRA）的 GEMM 把它们当作梯度；错位的写入落在 logits 缓冲区之前至多 2^31 个元素处（BF16 为 4 GiB），分块超过 2^32 个元素时落在缓冲区内更早的行上。那里是不是别的分配、结果是静默破坏还是 `cudaErrorIllegalAddress`，取决于 GPU 上的地址布局，未确认。

修复（T23，始终开启，没有开关）：这 7 处在乘法前把行号转成 `int64_t`（`static_cast<int64_t>(row) * P + i`），与同文件中 `kd_*` 版本和反向读取的写法相同。只有地址计算改变：为 compute_120a 编译修复前后的 PTX，这四个 kernel（各有 float 与 BF16 两个实例）之外的 29 个入口逐字节相同；这 8 个入口的浮点指令序列（操作码、立即数与源码行）相同；按数据流追踪全局读写的地址，修复前有 20 处经过含 `P` 的 32 位乘法，修复后为 0。分块行数 × V ≤ 2^31 的形状读写相同的元素、做相同的浮点运算，结果不变。

验证（没有 GPU）：`fork/patches/tests/native/test_cross_entropy_row_offsets.py`（`fork/dev test patches`）把这四个 kernel、`prepare_softmax_blockwide3`、`SoftmaxParams`、`maybe_softcap_value` 以及 `kernels.h` 的两个分块常量按源码原样编译成主机 C++，用 `-fwrapv`，int 溢出按 PTX 的方式回绕。每次调用运行一个 block、一个线程（`blockReduce` 退化为恒等），这个线程访问 GPU 上整个 block 访问的全部元素。logits 缓冲区是一段 `PROT_NONE` 的地址保留（行数 × V 个元素，下方另留 2^31 个元素），只有被测的行可读写，越界访问立即触发 SIGSEGV，探针报告元素偏移。float 与 BF16 两种实例各跑：V=151936 的第 14,134 行（前向的全部分块；反向的全部 block，target 分别为普通 token 与 -100）；V=100000 第 21,474 行的目标 logit（分块 0 的循环都在 2^31 以内，只有 `:699` 越界）；V=65536 的第 32,768 行（非分块的前向与反向）；另有 28,270 行时的末行（超过 2^32，回绕后落在缓冲区内）。在上游源码（`488e5f00`）上这 15 个用例都在回绕的访问处出错（元素偏移 -2^31 附近，或缓冲区内更早的行），修复后与 float64 参考一致；不回绕的最后一行（V=151936 的第 14,133 行、V=65536 的第 32,767 行）在修复前后都通过。用同一个探针分别编译上游与修复后的 kernel，36 个不回绕的用例输出逐字节相同。`fused_classifier.cu` 按 sm_120a 的编译命令编译通过（PTX 与 SASS）。

GPU 检查（未做）：在 5090 上运行。

1. kernel 级：BF16 logits `{16384, 151936}`（4.64 GiB），随机值，target 随机，其中若干行为 -100。调用 `chunked_cross_entropy_forward`，再用它的 LSE 调用原地的 `chunked_cross_entropy_backward`（`dloss` 为 1/16384），在 `compute-sanitizer --tool memcheck` 下运行，与 torch float32 的 `log_softmax`/`softmax` 参考比较每行的 loss、LSE 与 dlogits。期望：修复前 memcheck 报告第 14,134 行起的越界读写（或出现非法地址错误），或这些行与参考不符；修复后 memcheck 干净，结果在 bf16 误差内与参考一致，第 14,134 行之前的行与修复前逐位相同。非分块 kernel 同样检查：V=65536、32,769 行，`fused_cross_entropy_forward` 与 `fused_cross_entropy_backward`（4.3 GB）。
2. 端到端：Qwen3-0.6B，LoRA，`recipe: fp8_hybrid`，`recompute: true`，`per_device_train_batch_size: 1`，`sequence_len: 32768`（日志出现 `Auto-setting lmhead_chunks=2`），训练 1 步；同一配置加 `lmhead_chunks: 4`（每块 8,192 行，不回绕）作对照，比较 loss 与 grad norm。期望：修复前出现非法地址错误，或 loss、grad norm 与对照不一致；修复后两者在 bf16 误差内相同。显存估计约 8 GB（logits 4.64 GiB），32 GB 的卡放得下（估计，未确认）。2 块时 dX 走 BF16，这一项不经过 `quant.cu`；只有 1 块的 fp8_hybrid 配置由下一节的 GPU 检查覆盖。

## FP8 abs_max 与量化 kernel 的循环下标是 int，一次处理约 2^31 个元素时越界读写（fork 已修复：ledger T23）

本节同样是例外：fork 修改了上游文件。行号对应上游 `282f9901`；`quant.cu`、`fused_lm_head_loss.cpp`、`causal_lm_execution_profile.cpp` 与 `graph_executor_weight_cache.cpp` 在 `488e5f00` 与上游 `82d42c72` 中相同。

缺陷：`csrc/src/recipes/fp8_hybrid/kernels/quant.cu` 的 `reduce_abs_max_kernel`（`:81`）与三个 `quantize_with_abs_max_kernel`（FP32→BF16 `:138`、→INT8 `:169`、→FP8 E4M3/E5M2 `:213`）用 grid-stride 循环遍历 `long N` 个元素，循环下标却是 int：`for (int i = vec * (blockIdx.x * blockDim.x + threadIdx.x); i < N; i += vec * blockDim.x * gridDim.x)`，vec 是 16 字节中的元素数（BF16 为 8，FP32 为 4）。nvcc 13.1 为 compute_120a 生成的 PTX 用 `add.s32` 推进下标，再 `cvt.s64.s32` 与 64 位的 `N` 比较：下一个下标超过 2^31 − 1 的线程回绕到约 −2^31，`i < N` 仍然成立，于是继续读 `in + i`（量化 kernel 还写 `out + i`）。`N > 2^31` 时任何 int 都小于 `N`，循环不会结束，直到访问到非法地址；`N ≤ 2^31` 时循环会结束，但回绕的线程要在缓冲区下方读写多轮：按每轮的起点逐轮模拟 5090 上 BF16 的循环，14,121 行（V=151936）时 261,120 个线程中有 11,696 个回绕，最多在缓冲区下方迭代 2,057 次；14,134 行时有 258,592 个回绕，最多 228,220 次。同一文件的 `quantize_with_delayed_scale_kernel`（`:666`）已经用 `long` 下标。

阈值：两个启动函数启动“每 SM 最大线程数 × SM 数”个线程，步长 S = vec × 线程数；N 是 vec 的倍数、并且 N ≥ 2^31 − S + vec 时，读最后一个向量的线程会回绕。RTX 5090（170 SM × 1536 线程，BF16 输入）S = 2,088,960，阈值为 2,145,394,696 个元素：V=151936 时从 M = 14,121 行起，V=128256 时从 16,728 行起，比上一节交叉熵 kernel 的 14,135、16,744 行略低。H100（132 SM × 2048 线程）S = 2,162,688。

触发：`fused_lm_head_loss.cpp` 的 `lm_head_dx_matmul`（`:185`，SFT 路径 `:802`、GRPO/DPO 行压缩路径 `:1271`）找到转置的 FP8 lm_head 缓存时，先对分块中全部 M × V 个 dlogits 调用 `abs_max`（`:206`），再用 `quantize_with_abs_max` 量化成 E5M2（`:207`），然后做 FP8 的 dX GEMM。这个缓存只在以下条件都满足时存在：`recipe: fp8_hybrid`；`lmhead_chunks ≤ 1`（`causal_lm_execution_profile.cpp:113-117` 只在这时把 `lm_head` 放进反向的 FP8 缓存组）；lm_head 冻结且不是流式缓冲，即 LoRA（`graph_executor_weight_cache.cpp:306-316`）；没有设置 `SUROGATE_DISABLE_FP8_LMHEAD` 或 `SUROGATE_DISABLE_FP8_LMHEAD_T`；缓存（V × C 字节）不超过 `SUROGATE_FP8_LMHEAD_T_MAX_MIB`（默认 512 MiB，`:379-419`），或者设置了 `SUROGATE_FORCE_FP8_LMHEAD_T`。Llama-3.1-8B 的缓存为 501 MiB，Qwen3-0.6B/1.7B/4B 为 148/297/371 MiB，Qwen3-8B 为 594 MiB（超过默认上限）。因此 V=151936 时，fp8_hybrid LoRA 在 `recompute: false` 且 BT ≥ 14,121 时，或 `recompute: true` 而 BT 为奇数（自动分块退回 1 块）且 BT ≥ 14,121 时，会在交叉熵之外再经过这里。Qwen3-0.6B/1.7B/4B 是 tied 模型，它们的 `lm_head` 在 LoRA 下也是冻结参数，按代码应同样被缓存，没有在 GPU 上确认。量化 kernel 的其他调用点（fp8_hybrid 的 GEMM 输入、FP8 权重缓存）遇到同样大的张量时也会回绕，fork 没有逐一确认哪些配置能达到这个大小。

影响（读代码、PTX 与 CPU 模拟，GPU 上未确认）：`N > 2^31` 时，`abs_max` 从 dlogits 下方约 4 GiB（BF16）处开始读，循环要么停在非法地址（`cudaErrorIllegalAddress`），要么一直读下去，训练步不再结束；量化 kernel 写到 E5M2 临时缓冲下方至多 2 GiB 处。N 略小于 2^31 时循环会结束，但 amax 混入了缓冲区之外的值，量化还写坏了缓冲区之前的内存；回绕多轮的线程也拖长了 kernel。

修复（T23，始终开启，没有开关）：四个循环的下标改为 `long`（`for (long i = ...)`），与 `quantize_with_delayed_scale_kernel` 相同。为 compute_120a 编译修复前后的 PTX：20 个入口中 11 个逐字节相同；改变的 9 个正是这四个 kernel 的全部实例，循环从 `add.s32` 加 `cvt.s64.s32` 变为 `add.s64`，浮点、转换与访存指令的序列（不含寄存器）相同。不回绕的形状中，每个线程按相同顺序访问相同的元素，结果不变。

验证（没有 GPU）：`fork/patches/tests/native/test_fp8_quant_grid_stride.py`（`fork/dev test patches`）把 `reduce_abs_max_kernel`、FP8 的 `quantize_with_abs_max_kernel`、`fp8_scale_from_amax`、两个启动函数中计算 block 与 grid 的两行，以及 `utils.h` 的 FP8 常量按源码原样编译成主机 C++（`-fwrapv`），运行真实启动配置中的一个线程。输入与输出各是一段 `PROT_NONE` 的地址保留（N 个元素，下方另留 2^31 个元素），只打开这个线程应当访问的向量（从起点起每隔 S 个元素一个，按 64 位计算），其他访问立即触发 SIGSEGV，探针报告偏移。探针记录每次读写的偏移，与这份列表逐一比较；abs-max 与输入比较；写出的字节与输入比较。FP8 转换是桩：scale 为 1，写出的字节就是整数输入，所以检查的是读写哪些元素，不是舍入。用例：N = 16,384 × 151,936（> 2^31）时的第一个和最后一个线程，`abs_max` 测 BF16 与 FP32 输入、5090 与 H100 的启动配置，量化测 E5M2←BF16、E4M3←BF16、E4M3←FP32；N = 14,121 × 151,936（< 2^31，但读最后一个向量的线程回绕）。在上游源码（`488e5f00`）上这 16 个用例都在回绕后的第一次读取处出错（输入偏移约 −2^31，例如 5090 上 BF16 的第 0 号线程在 1,029 次正常读取之后读偏移 −2,145,427,456），修复后全部通过；N = 14,120 × 151,936 时读最后一个向量的线程不回绕，修复前后都通过。INT8 与 FP32→BF16 两个变体的循环由同一处改动加宽（PTX 中同样变为 64 位），探针不运行它们：它们的循环体还有与下标无关的问题，INT8 版本存回一个没有赋值的向量，FP32→BF16 版本每 4 个元素写 8 个 BF16，后 4 个没有赋值，并与相邻线程的元素重叠。fork 没有改这两处，也没有确认它们是否有调用路径。`quant.cu` 按 sm_120a 的编译命令编译通过（PTX 与 SASS）。

GPU 检查（未做）：在 5090 上运行。

1. kernel 级：BF16 输入 `{16384, 151936}`（4.64 GiB），随机值；E5M2 输出 2.32 GiB。仿照 `csrc/src/testing/kernels/test-fp8-quant.cpp` 调用 `abs_max`，再用它的结果调用 `quantize_with_abs_max`（E5M2←BF16），在 `timeout 600 compute-sanitizer --tool memcheck` 下运行。期望：修复前 memcheck 报告输入缓冲区下方的越界读、非法地址错误，或者 kernel 不结束而超时；修复后 memcheck 干净，amax 等于 torch 的 `x.abs().max()`，反量化后的输出与 torch 参考在 E5M2 的一个最小单位内一致。再用 `{14121, 151936}` 跑一次（小于 2^31，修复前同样越界）。
2. 端到端：Qwen3-0.6B，LoRA，`recipe: fp8_hybrid`，`recompute: true`，`per_device_train_batch_size: 1`，`sequence_len: 16383`（BT 为奇数，日志中不出现 `Auto-setting lmhead_chunks`，保持 1 块；如果有地方要求偶数长度而报错，改用 `recompute: false`、`sequence_len: 16384`，显存未估计），训练 1 步；同一配置加环境变量 `SUROGATE_DISABLE_FP8_LMHEAD_T=1`（dX 走 BF16，不经过 `quant.cu`）作对照。期望：修复前出现非法地址错误或训练步不结束；修复后两次的 loss 相同（前向相同），grad norm 只差 E5M2 dlogits 带来的量化误差（估计相对差在 1e-2 以内，未确认）。两次 grad norm 逐位相同时说明没有走 FP8 dX 路径（例如 tied 的 head 没有进入缓存），检查不成立，改用 untied 的模型（如 Qwen3-8B 加 `SUROGATE_FP8_LMHEAD_T_MAX_MIB=1024`，显存未估计）。交叉熵 kernel 也处理这个 16,383 行的分块，所以这一项同时检查上一节的修复。

在修复之前，受影响的配置可以设 `SUROGATE_DISABLE_FP8_LMHEAD_T=1`，或用 `lmhead_chunks ≥ 2`，避开这里；交叉熵 kernel 的回绕还要另外满足上一节的条件。

## NVFP4 随机舍入与 Four Over Six 块比例的缺陷（fork 已修复：ledger T24）

本节同样是例外：fork 修改了上游文件。行号对应上游 `282f9901`；`csrc/src/recipes/nvfp4/` 在 `488e5f00` 与上游 `82d42c72` 中相同。三处缺陷都在 NVFP4 recipe 的默认设置下出现（`fp4_backend: cutlass`、`fp4_four_over_six: true`，`surogate/core/config/sft_config.py:407-408`；`use_cuda_graphs: true`，`:414`）。数值都来自 CPU 上逐运算重放 fp32 算术的模拟（`fork/patches/tests/native/test_nvfp4_4o6.py`；`rcp.approx` 按正确舍入的倒数模拟，二者相差不超过 1 ulp），GPU 上未确认。

缺陷 1（SR）：`quantize_block_4o6_stochastic`（`quant_nvfp4_4o6_sm100.cu`）给缩放后的值加上固定的 `(b/256 - 0.5) * 0.5`（`:338`，b 是 8 位随机字段，噪声在 [-0.25, 0.25)），再用 `cvt.rn.satfinite.e2m1x2` 舍入到最近。E2M1 的间距在 [0, 2) 是 0.5，在 [2, 4) 是 1，在 [4, 6] 是 2，所以只有 [0, 2) 近似无偏：2.4 向上的概率是 76/256 而不是 0.4（期望 2.297），2.2 与 4.6 从不向上，2.8 与 5.4 总是向上；偏差最大为间距的 25%（[2, 4)）和 38%（[4, 6)），[0, 2) 上也有 -1/1024 的均值偏差。另外，值用未舍入的 `6/amax` 或 `4/amax` 编码（`:325`、`:336`），GEMM 却按存下的 E4M3 比例解码（`:353-357`），每块相差 sf/d，均方根 2.7%、范围 0.94–1.06，这是与种子无关的逐块偏差：只修正舍入时还剩 2.6e-2；改用按 RN 舍入的存储比例编码时 amax 会饱和，剩 0.8e-2–1.3e-2。随机数是每线程一个 LCG（`:503-508`），低字节只取决于种子与块号的低 8 位。触发：每个带权重梯度的稠密 matmul 的 wgrad 操作数 dout^T（`nvfp4_recipe.cpp:967-977`，经 `matmul.cpp:912` 与 `matmul_swiglu.cpp:511`），即全量训练；LoRA/QLoRA 冻结的基础权重（`skip_weight_grad`）、`fp4_backend: cudnn`、`fp4_four_over_six: false`（此时 wgrad 用标准的 Philox SR）与 MoE 专家（BF16）不经过这里。在 32×4096 的类 dout^T 张量（行比例对数正态，测试模块的 `dout_like(1234, ...)`）上，对随机字段取精确期望后 ||E[x̂]-x||/||x|| 为 5.3e-2（高斯）、4.2e-2（t(3)）、2.0e-2（0.2% 的 token 大 3e4 倍）；对种子取平均时停在 0.053 附近（4/16/64/256 个种子：0.070、0.058、0.054、0.053），说明是偏差而不是噪声。上游文档 `docs/guides/precision-and-recipes.md:119` 列出的 `no_fp4_stochastic_rounding` 没有任何代码读取，无法关掉 SR。

缺陷 2（RN）：`quantize_block_4o6`（前向的输入与权重、wgrad 的 inp^T、FP4 权重缓存与流式权重）有同样的编码/解码不一致，并且把比例下限钳在 1/128（`:263`）：块比例低于 1/128（block amax < 全局 amax × 2^-15）的块按自身 amax 编码、按 1/128 解码，被放大 1/(128 d) 倍。在上面的离群 token 张量上，这些块（占 92%）的最小二乘增益为 1.98，相对误差 1.93；标准 NVFP4 kernel 用解码后的比例编码（`quant_nvfp4_cutlass.cu` 中的 `actual_decode`），4/6 论文（arXiv:2512.02010 表 2）也用 `fp8_e4m3(delta)` 编码。

缺陷 3（图）：SR 的种子是主机值（`matmul.cpp:862`、`matmul_swiglu.cpp:483`，来自 `std::minstd_rand{42}`，`graph_executor.h:461`），作为 kernel 参数在捕获时固定。DSL 训练把前向与反向捕获一次（`py_train.cpp:1657`），之后每步重放，所以捕获之后每一步、每个元素用同一份噪声：每个位置的舍入成为固定阈值，对步取期望不再等于输入（固定字段 b 时，位置的平均误差为 (0.5 - b/256) 个间距）。这对 4/6 的 SR 和标准 Philox SR（`quantize_nvfp4_stochastic_cutlass_kernel` 与 `_auto_kernel`，dgrad 的 dout，LoRA 也经过）都成立。

修复（T24，默认开启；三个开关都在进程内第一次调用时读取）：

- SR（`SUROGATE_NVFP4_4O6_SR_LEGACY=1` 恢复上游 kernel）：`quant_nvfp4_4o6.cu` 的 `quantize_nvfp4_4o6_stochastic_cutlass_auto_scale` 在 `abs_max` 之后调用 `fork/native/src/nvfp4_4o6.cu` 的 `quantize_4o6_stochastic`。4/6 的选择不变（块的确定函数，不影响无偏）；存储比例取不小于 d·(1-2^-16) 的最小 E4M3 码（`e4m3_code_no_saturation`，256 与 384 保持精确），钳在 [2^-9, 448]，用存下的比例编码（`global_encode / sf`），因此没有值饱和；每个值按所在间距 s 向下取整到网格，以 k/256 的概率进一位，k = round_even(256·frac)（arXiv:2504.20634 的 SRC），每个值的 |E[q]-y| ≤ s/512，网格上的值精确，平均偏差为 0；随机数是每 16 个值一次 Philox4x32-10（计数器 (块号, 代数, 0, 0)，密钥 (种子, 0x346F3653)），16 个字节各给一个值。张量上的精确期望偏差降到 4.1e-4（高斯）、4.0e-4（t(3)）、4.0e-4（离群），种子平均按 1/sqrt(N) 下降（4/16/64/256 个种子：0.072、0.036、0.018、0.009）。代价是单个种子的误差从 0.107 升到 0.142，这是无偏 SR 本身的方差。论文 2512.02010 附录 A.3 中“按 RN 选择再做 SR”的无偏做法（与这里相同）相对普通 NVFP4 没有训练收益，有偏的 4/6-on-SR 反而有，所以收敛效果未知，按结果记录。
- RN（`SUROGATE_NVFP4_4O6_RN_LEGACY=1` 恢复上游 kernel）：`quantize_nvfp4_4o6_cutlass_auto_scale` 与 `_from_amax` 调用 `quantize_4o6`：两个候选各自用 RN 得到的 E4M3 比例编码，并按 GEMM 实际的解码（sf/global_encode）计算误差来选择，比例钳在 [2^-9, 448]。相对误差：高斯 9.03e-2 降到 8.73e-2，t(3) 8.98e-2 到 8.80e-2，离群 1.98e-2 到 1.27e-2；离群张量中低于 1/128 的块的增益从 1.98 回到 0.99。这改变所有 NVFP4 运行（包括 LoRA）的前向数值，需要在 GPU 上对比 loss。
- 两个 4/6 kernel 多出的算术：每 16 个值一次 10 轮的 Philox（代替 4 步 LCG），SR 每块 1 次、RN 每块 4 次 IEEE 除法；寄存器数与上游相同（40），耗时没有在 GPU 上测量（见下文 GPU 检查第 4 项）。
- 图（`SUROGATE_NVFP4_SR_GRAPH_LEGACY=1` 恢复上游行为）：`fork/native/include/surogate_fork/sr_graph_generation.cuh` 给包含它的翻译单元一个设备计数器；计数器以及读它、加它的函数（`generation`、`curand_offset`、`advance`）都在匿名命名空间中，是内部链接：外部链接的 inline `advance` 在主机编译器不内联时（-O0，CMake 的 Debug 构建）会被链接器合并成一份，只有一个翻译单元的计数器在增加，另一个单元的 SR 又用回捕获时的噪声。启动方在 `cudaStreamIsCapturing` 为真时让 kernel 读计数器（4/6 SR 放进 Philox 计数器，标准 SR 作为 `curand_init` 的 offset `gen << 34`），并在 kernel 之后排一个单线程 kernel 把计数器加一，所以每次重放都抽到新的噪声。不在捕获中时既不读也不加：标准 SR kernel 的结果与上游逐位相同，4/6 SR 用代数 0，eager 运行和 checkpoint 恢复的可重现性不变。计数器不进 checkpoint：图模式下恢复之后的噪声与不中断的运行不同（上游在图模式下本来就固定）。代价是每个被捕获的 SR 启动多一个单线程 kernel。图修复只在 fork 的 4/6 SR kernel 和标准 SR kernel 中：设 `SUROGATE_NVFP4_4O6_SR_LEGACY=1` 时运行的上游 4/6 SR kernel 不接受代数，图重放时 wgrad 的 4/6 操作数总是复用捕获时的噪声，`SUROGATE_NVFP4_SR_GRAPH_LEGACY` 只还影响标准 SR。三个开关因此不是相互独立的：只设 `SUROGATE_NVFP4_4O6_SR_LEGACY=1` 的 A/B 在图模式下同时去掉了 wgrad 操作数上的图修复。

没有修复、另行记录：

- `fp4_backend: cudnn` 的 SR kernel（`quant_fp4.cu` 的 `quantize_fp4_block_stochastic_kernel` 与 `_auto_kernel`）同样在图中复用噪声。
- 标准 SR kernel 的 `curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, ...)` 不含 `blockIdx.y`，同一行块中相隔 128 列的元素得到相同的随机数（读代码得出，未在 GPU 上确认）：不引入偏差，但让这些元素的舍入误差相关。改它会改变 eager 模式的结果，fork 没有动。
- RHT 的种子（`nvfp4_recipe.cpp:345`）与其他由 `mRngSeedFn` 取种子的 op 在图中同样固定；RHT 只要求同一 GEMM 的两个操作数用同一个符号模式，固定不算错误。
- 上游 4/6 kernel 文件 `quant_nvfp4_4o6_sm100.cu` 没有改动，作为 `*_LEGACY` 路径保留；`no_fp4_stochastic_rounding` 的文档行没有改（fork 不修改上游用户文档）。

验证（没有 GPU）：`fork/patches/tests/native/test_nvfp4_4o6.py`（`fork/dev test patches`）用 numpy 逐运算模拟上游与 fork 的 kernel（E2M1 的 RN、E4M3 的 RN satfinite、LCG 与字节布局、Philox、4/6 的 MSE 选择与 nvcc 收缩的乘加、CUTLASS 的解码 q·sf·amax/1536），检查上面的数字；`fork/native/tests/host/nvfp4_4o6_check.cpp` 把 kernel 调用的 `surogate_fork/nvfp4_4o6_arith.h` 用主机编译器编译：Random123 的 Philox 已知答案、每个 E4M3 值及其相邻值与中点的 RN 编码、不饱和码是满足下限的最小码、SR 对 2^21 个点的 256 个字节取期望不超过 s/512、半步格点上平均偏差为 0，并以上游的抖动为反例；找到 CUDA 头文件时，还逐个比较 [2^-11, 512] 中每个 float 的 E4M3 编码与 `cuda_fp8.h` 的 `__nv_cvt_float_to_fp8`，以及 10 万个计数器上的 Philox 与 `curand_Philox4x32_10`；模拟与头文件对同一批请求逐值相同。源码 pin 检查上游启动函数在开关之后转给 fork kernel、fork kernel 调用的是检查过的算术、上游 4/6 kernel（模拟所依据的行）没有被改、fork kernel 复制或必须与之一致的上游代码没有变（`quant_nvfp4_4o6_sm100.cu` 去掉注释后的代码摘要，4/6 的 GEMM alpha 与流式权重中的 1536）、标准 SR kernel 接受代数，以及文档中的开关名就是代码读取的名字。在 `488e5f00` 的源码上，主机检查无法编译（头文件不存在），pin 失败，上游 kernel 的模拟给出上面的偏差。fork kernel、`quant_nvfp4_4o6.cu`、`quant_nvfp4_cutlass.cu` 和 GPU 测试按 sm_120a 的编译命令编译通过（fork 的 4/6 kernel 与标准 SR kernel 都用 40 个寄存器，与上游相同），fork kernel 也为 sm_89 与 sm_100a 编译通过。

GPU 检查（未做；5090，`fork/dev test native` 之后运行 `csrc/build/integration-tests '[nvfp4-4o6]'`，即 `fork/native/tests/test-fork-nvfp4-4o6.cu`）：

1. SR：32×4096 的高斯 BF16 张量（行比例对数正态，固定种子），`quantize_nvfp4_4o6_stochastic_cutlass_auto_scale` 跑 1024 个种子，按 `cutlass_scale_index` 取块比例、按 q·sf·amax/1536 反量化求平均。期望前两个种子的 FP4 码与块比例与主机模型（同一头文件）至少 99.9% 相同，||E[x̂]-x||/||x|| 在 1024 个种子时 ≤ 6e-3，256 个种子的值至少是它的 1.6 倍。设 `SUROGATE_NVFP4_4O6_SR_LEGACY=1` 时该用例跳过；要看上游的数字，可以临时去掉跳过，期望约 5e-2 且不随种子数下降。
2. RN：同样形状的高斯与离群张量，`quantize_nvfp4_4o6_cutlass_auto_scale` 的码与比例与主机模型至少 99.9% 相同，离群张量中低于 1/128 的块的最小二乘增益在 1±0.03 内。
3. 图：把 4/6 SR 与标准 SR 的量化捕获进一个图，重放 3 次，每次的 FP4 码都与上一次不同，块比例相同；同一种子的两次 eager 调用结果相同。设 `SUROGATE_NVFP4_SR_GRAPH_LEGACY=1` 时期望重放结果相同；设 `SUROGATE_NVFP4_4O6_SR_LEGACY=1` 时 4/6 的码在重放之间相同（上游 kernel），标准 SR 的码仍按图开关变化。Debug 构建上也运行一次（`make build-tests BUILD_TYPE=Debug BUILD_DIR=csrc/build-debug`，然后 `csrc/build-debug/integration-tests '[sr-graph]'`）：上面的链接问题只在主机代码不内联时出现，Release 构建看不到。
4. 训练 A/B：Qwen3-0.6B NVFP4 全量训练 500 步，比较默认、`SUROGATE_NVFP4_4O6_SR_LEGACY=1 SUROGATE_NVFP4_4O6_RN_LEGACY=1 SUROGATE_NVFP4_SR_GRAPH_LEGACY=1`、`fp4_four_over_six: false` 与 bf16 的 loss 曲线；单独看 4/6 SR kernel 的修复时，用 `use_cuda_graphs: false` 比较默认与只设 `SUROGATE_NVFP4_4O6_SR_LEGACY=1`（eager 时图开关不起作用；图模式下只设它还会去掉 wgrad 操作数上的图修复，见上）。再对 LoRA（只受 RN 与图两项影响）做同样的比较，并用 nsys 记录 SR 量化 kernel 与新增单线程 kernel 的耗时。结果记在 T24 的 `conflict_history` 中。
5. 端到端的图噪声：33b73381 的提交说明中写的检查（Qwen3-0.6B NVFP4 LoRA，第 2 步与第 3 步用同一批数据，比较 dout_fp4 的哈希，修复前相同）不能说明缺陷：LoRA 的适配器在两步之间被优化器更新，基础 matmul 的 dout 与它的 FP4 码即使噪声固定也不同。改为只让噪声可能不同：`learning_rate: 0`（参数不变）、每步同一个序列、`use_cuda_graphs: true`，在第 2 步与第 3 步（两次重放）记录反向中第一个用 SR 量化的 dgrad 操作数（如最后一个 decoder 层 `down_proj` 的 dout）的 BF16 输入与 FP4 码的哈希（临时在 `nvfp4_recipe.cpp` 的 `backward_matmul_cutlass` 中把两者复制到步后读取的缓冲区）。BF16 输入两步必须相同（不同时说明还有别的不确定性，检查无效）；FP4 码在默认设置下不同，设 `SUROGATE_NVFP4_SR_GRAPH_LEGACY=1` 时相同。全量训练再对 wgrad 的 4/6 操作数（dout^T 的 FP4 码）做同样的比较，LoRA 不量化它（`skip_weight_grad`）。

## GPT-OSS 预量化 checkpoint 的 LoRA 训练跳过冻结专家的输入梯度（fork 已修复：ledger T25）

本节同样是例外：fork 修改了上游文件。行号对应上游 `282f9901`；这两个文件在 `488e5f00` 与上游 `82d42c72` 中相同。

缺陷：`dispatch_moe_grouped_gemm_gate_up_backward`（`csrc/src/runtime/ops/moe_grouped_gemm_gate_up.cpp:1043-1056`）与 `dispatch_moe_grouped_gemm_down_backward`（`moe_grouped_gemm_down.cpp:902-914`）在 `lora_enabled && is_lora_only_mode() && is_prequantized() && Architecture == GPT_OSS` 时设 `skip_base_backward`：不做冻结专家的 dgrad GEMM，把 `d_input` 清零，之后只加上分组 LoRA 的 dX。`is_lora_only_mode()` 就是 `lora_enabled()`（`dsl_model_execution.cpp:419-426`），所以在预量化的 GPT-OSS checkpoint 上做任何 LoRA 训练都会走到这里，上游示例 `examples/sft/gpt-oss/gptoss-lora-mxfp4.yaml`（`openai/gpt-oss-20b`，MXFP4）也在其中。GPT-OSS 专家的计算顺序是 gate_up → bias → `gpt_oss_moe_act` → down → bias（`surogate/dsl/modules/moe.py` 的 `GptOssMoEExperts`）。down 反向的 `d_input` 就是 `d_act`，激活的反向由它算出 `d_gate_up`，因此：

- 同一层专家 gate_up 的 LoRA 梯度只来自 down LoRA 一项。down LoRA 的 B 初始为 0（`lora_weights_manager.cpp:383-389`），第一步时这一项恰好为 0，之后与正确梯度也几乎正交。
- MoE 分支向注意力层的 LoRA 和更早的层只传回这一项，冻结专家的雅可比完全缺失；残差流照常传回。
- 只有最后一个 MoE 层的 down LoRA 梯度是精确的。

来历：`7a8d5fee`（2026-02-12，GPT-OSS 支持）加入这个跳过，同时提供 `SUROGATE_PREQUANT_BASE_GRAD=1` 恢复 dgrad，说明当时知道它是近似。`1181c715`（2026-03-31，“stability improvements”，同一提交还删掉了几个调试用的环境变量）删去了这个变量，跳过从此无条件生效，提交说明、文档和测试都没有解释。这不是实现上的限制：两个反向 op 照样解析反量化后的 BF16 专家权重（`resolve_tensor(op.inputs[2])`，预量化的 MXFP4 已经是 DSL 布局，见 `dsl_qlora_pipeline.cpp:1954-1957`），其他架构的预量化 MoE 和 BF16 checkpoint 上的 GPT-OSS 都在同样的缓冲上做同样的 dgrad。上游的 `tests/test_prequant_mxfp4_gpt_oss.py` 只检查 loss 下降；`tests/test_onboarding_gpt_oss_moe.py` 与 HF 的梯度比较用的不是预量化模型。`docs/reference/benchmarks.md` 中 gpt-oss-20B 的吞吐是带着这个跳过测的。

影响的估计（CPU 上 float64 的模拟，真实权重下的幅度未知）：一个 GPT-OSS 专家，T=256、C=512、M=256、r=16，B=0 时 gate_up LoRA 的 B 梯度与输入梯度都恰好为 0（正确值不为 0）；|B| 约 2e-4 与 1e-2 时 gate_up LoRA 的 A、B 梯度与正确值的余弦在 -0.01 到 0.09 之间；经过 MoE 分支的输入梯度相对误差为 1.0；down LoRA 的梯度余弦为 1.0。两层、8 个专家取 top-2、注意力投影与每个专家的 gate_up/down 都挂 LoRA 时，第一步两层专家 gate_up LoRA 的梯度都为 0，注意力 LoRA 的余弦为 0.70 / 0.80，第一层专家 down LoRA 为 0.86；B 不为 0 时第一层注意力 LoRA 的余弦为 -0.15，专家 gate_up LoRA 为 0.51 / 0.65。

修复（T25，始终开启）：两处条件再要求 `surogate_fork::gpt_oss_prequant_skip_expert_dgrad_requested()`（`fork/native/include/surogate_fork/gpt_oss_prequant_dgrad.h`），未设置开关时为 false，所以默认计算冻结专家的 dgrad，走的是其他预量化 MoE 模型已经在用的分支（包括其前面 EP/LLEP 的权重指针准备）。`SUROGATE_GPT_OSS_PREQUANT_SKIP_EXPERT_DGRAD` 设为空串和 `0` 以外的值时恢复上游的跳过（约定与原来的 `SUROGATE_PREQUANT_BASE_GRAD` 相同），每个进程只读一次，打开时在 stderr 打印一行说明梯度是近似的。代价（未在 GPU 上测量）：每个 MoE 层多两个 BF16 分组 dgrad GEMM，约为前向专家 FLOP 的一半；路线图按 GPT-OSS-20B 前向加重放的专家时间 10.9 ms/层估计约 +5 ms/层，这也改变路线图 #27 中 W-C 方案的上界。

验证（没有 GPU）：`fork/patches/tests/native/test_gpt_oss_prequant_dgrad.py`（`fork/dev test patches`）从两个源文件中截取 `skip_base_backward` 的定义原文，与开关头文件和真实的 `PretrainedConfig::ArchitectureId` 一起编译成探针，对全部输入组合和开关取值运行：开关未设置、为空或为 `0` 时从不跳过，打开时只在四个条件都成立时跳过；另外检查这个变量只用于选择清零还是 dgrad 的那个分支。最后一个测试用 float64 模拟一个挂 gate_up 与 down LoRA 的 GPT-OSS 专家，按探针为“LoRA + 预量化 + GPT-OSS、开关未设置”选出的分支计算梯度，与 torch autograd 比较。用上游源码（`488e5f00`）运行时探针选择跳过，默认分支的测试失败，模拟的 gate_up LoRA B 梯度与输入梯度相对误差为 1。两个翻译单元按 sm_120a 配置的编译命令编译通过。

GPU 检查（未做）：在 5090 上（`ep_size: 1`）用 transformers 构建一个随机权重的两层 GPT-OSS（例如 `GptOssConfig(hidden_size=512, intermediate_size=256, num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2, head_dim=64, num_local_experts=8, num_experts_per_tok=2, vocab_size=4096)`），专家导出为 MXFP4（与 `tests/test_prequant_mxfp4_gpt_oss.py` 的做法相同），再把同一组权重反量化后另存一份 BF16 checkpoint。LoRA 设置同 `gptoss-lora-mxfp4.yaml`（q/k/v/o 与 gate_up_proj/down_proj，rank 16），同一个 seed，同一批数据，各做一次反向，读 `get_lora_gradients`。

1. 预量化模型、开关未设置：每层专家 gate_up LoRA 的 B 梯度范数不为 0；所有 LoRA 梯度与 BF16 checkpoint 的结果（以及 HF + PEFT 在反量化权重上的结果）的余弦大于 0.99。
2. 同一模型设 `SUROGATE_GPT_OSS_PREQUANT_SKIP_EXPERT_DGRAD=1`：第一步专家 gate_up LoRA 的 B 梯度为 0，stderr 有一行开关说明（这是修复前的行为）。
3. 运行 `tests/test_prequant_mxfp4_gpt_oss.py`，记录开关 0/1 时 `gptoss-lora-mxfp4.yaml` 的步时间；EP 大于 1 时的权重指针准备在 GPT-OSS 上没有跑过 dgrad，另外用 `ep_size: 2` 重复第 1 步。结果记在 T25 的 `conflict_history` 中。
