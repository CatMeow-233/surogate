# fork 原生代码

本目录存放 fork 自有的 C++ 与 CMake 代码。上游原生构建 `csrc/CMakeLists.txt` 只通过一个两行的挂接点引入这里的内容，因此 fork 在上游文件中的改动保持最小，上游在 `csrc/` 中新增或修改文件时也不会与 fork 文件冲突。

## 目录

| 路径 | 内容 | ledger topic |
|---|---|---|
| [cmake/fork_native.cmake](cmake/fork_native.cmake) | 挂接入口：定义 `surogate_fork_configure_common()`，并在 `csrc` 目录处理完毕后为测试目标和服务端的 `sinfer_ops` 补充设置 | T5、T16 |
| [cmake/training_openmp.cmake](cmake/training_openmp.cmake) | `surogate_enable_training_openmp()`：把 `OpenMP::OpenMP_CXX` 链接到训练库 | T5 |
| [cmake/cpu_step_kernels.cmake](cmake/cpu_step_kernels.cmake) | `surogate_fork_add_cpu_step_kernels()`：把 `src/cpu_step_kernels.cpp` 加入目标，只对这个文件加 `-fno-math-errno -ffp-contract=off`；`surogate-common` 与 OpenMP 契约测试都用它 | T14、T15 |
| [include/surogate_fork/](include/surogate_fork/) | fork 头文件，源码中写作 `#include "surogate_fork/<文件名>"`：`moe_weight_grad_policy.h`、`moe_weight_grad.h`（T3）、`omp_team.h`（T5，CPU 优化器各并行区的线程数和定序的范数求和；T14 的 serial group）、`host_grad_overwrites.h`（T13，micro-step 0 的 D2H 覆盖了哪些主机梯度，以及恢复上游 memset 的开关 `SUROGATE_FORK_HOST_GRAD_MEMSET`）、`cpu_step_kernels.h`（T14、T15，`cpu_training` 主机端步的向量化循环）、`gqa_fork_shapes.h`（T16，fork 注册的 GQA 形状及其路由开关）、`doc_mask_staging.h`（T17、T18，整步 graph 的文档掩码几何：每个微步的文档扫描、回放决定、捕获时按行 varlen、pad 模式超限的路由）、`full_step_warmup.h`（T20，整步 warmup 是否运行）、`cpu_adamw_checkpoint.h`（T21，`cpu_training` 的 CPU AdamW 矩的布局、checkpoint 容器与加载检查）、`pageable_host.h`（T21，写入器判断 pageable 主机内存）、`nvfp4_4o6.h`（T24，Four Over Six kernel 的入口与开关）、`nvfp4_4o6_arith.h`（T24，E4M3 比例码、E2M1 随机舍入与 Philox，主机与设备共用）、`sr_graph_generation.cuh`（T24，CUDA 图重放时的 SR 代数）、`gpt_oss_prequant_dgrad.h`（T25，GPT-OSS 预量化 LoRA 跳过冻结专家 dgrad 的开关） | T3、T5、T13–T16、T17、T18、T20、T21、T24、T25 |
| [src/](src/) | 上游代码调用的 fork 源文件，由 `surogate_fork_configure_common()` 编译进 `surogate-common`；`moe_weight_grad.cpp`（T3，MoE 基础专家权重梯度）、`cpu_step_kernels.cpp`（T14、T15，主机梯度加法与 CPU AdamW）、`cpu_adamw_checkpoint.cpp`（T21，`cpu_training` 的 CPU AdamW 矩写进 checkpoint）、`nvfp4_4o6.cu`（T24，NVFP4 Four Over Six 的 RN 与 SR kernel） | T3、T14、T15、T21、T24 |
| [serve/](serve/) | 服务引擎的 fork 源文件，由 `_surogate_fork_attach_serve()` 加入 `sinfer_ops`；目前是 fork 注册的三个 GQA 形状的 small-T 解码单元（`gqa_attention_decode_gqa128_{32q8,40q8,32q32}.cu`），开关见 [performance.md](../docs/performance.md#服务端-gqa-形状注册) | T16 |
| [openmp-contract/](openmp-contract/) | 独立的 OpenMP 回归测试，只需主机 C++ 编译器和 OpenMP：接线（`check.cpp`）、线程数契约（`team_check.cpp`）与 `cpu_training` 主机端步的契约（`cpu_step_check.cpp`） | T5、T13–T15 |
| [bench/cpu-optimizer/](bench/cpu-optimizer/) | `run.sh` 用训练库的编译命令编译 `cpu_adamw.cpp`（带与不带 OpenMP），按 NVML 的方式绑定 rank 线程，测 `cpu_training` 的优化器步耗时，结果记在 [performance.md](../docs/performance.md#cpu-优化器并行)；`step.sh` 用同样的方式编译 `cpu_adamw.cpp` 与 `src/cpu_step_kernels.cpp`，测步开始的清零、主机梯度加法和 AdamW 在 T13–T15 前后的耗时并逐位比较结果，记在 [performance.md](../docs/performance.md#cpu_training-主机端步) | T5、T13–T15 |
| [tests/](tests/) | fork 的 C++ 测试用例，编译进上游的 `dsl-run-state-tests`（T1、T10、T17）和 `integration-tests`（T2–T4、T24）；`test-fork-gqa-shapes.cpp` 是单独的服务端测试 `sinfer_gqa_fork_shapes_test`（T16） | T1–T4、T10、T16、T17、T24 |
| [tests/host/](tests/host/) | 不进上游测试可执行文件的主机程序：`cpu_adamw_checkpoint.cpp` 由 `fork/patches/tests/native/test_cpu_adamw_checkpoint.py` 用主机编译器从生产源码构建，链接 libcudart，不用 GPU；`stubs.cpp` 替代它不链接的两个符号；`nvfp4_4o6_check.cpp` 由 `fork/patches/tests/native/test_nvfp4_4o6.py` 用主机编译器构建，检查 T24 kernel 调用的算术，不需要 CUDA（有 CUDA 头文件时多两项对照） | T21、T24 |
| [release/](release/) | `build-wheel.sh` 在 CUDA 构建镜像中构建 fork 的原生 wheel；`VERSION` 是发布版本号；`plan.sh` 判断一次运行能否发布，并在发布前再检查一次，`test_plan.sh` 是它的测试（`fork/dev test release`）。由 `.github/workflows/native-release.yml` 使用，只有在 `main` 上手动运行并勾选 `publish` 才发布，推送不会发布；文件先传到草稿，由 `fork/tools/release_guard.py` 校验后才发布。见 [发布原生版本](../docs/build-and-release.md#发布原生版本) | — |

各 topic 的提交、开关和冲突历史见 [ledger](../patches/ledger.toml)。

## 挂接约定

`csrc/CMakeLists.txt` 中 fork 只有两行（ledger T5），位于 `surogate-common` 的 `target_link_libraries` 之后：

```cmake
include(${CMAKE_CURRENT_SOURCE_DIR}/../fork/native/cmake/fork_native.cmake)
surogate_fork_configure_common(surogate-common)
```

- `surogate_fork_configure_common(<target>)` 为目标启用训练用的 OpenMP，把 `fork/native/include` 加为 PRIVATE 头文件目录，并把 `src/` 中的源文件加入目标。头文件目录不是 PUBLIC，因此链接 `surogate-common` 的其他目标看不到它。
- `_surogate_fork_attach_tests()` 通过 `cmake_language(DEFER)` 在 `csrc` 目录处理完毕时运行，因为测试目标定义在挂接点之后。它为 `integration-tests` 和 `dsl-run-state-tests` 加上同一个头文件目录，并用 `target_sources()` 把 `tests/` 中的源文件加入这两个目标；每个目标都先用 `if(TARGET ...)` 判断，未开启 `BUILD_TESTS` 时 configure 不受影响。
- `_surogate_fork_attach_binding()` 同样延迟运行，为训练扩展模块 `_surogate`（`PYTHON_BINDING` 构建）加上 PRIVATE 的 fork 头文件目录：`py_train.cpp` 包含两个只有头文件的 fork 文件（T17 的 `doc_mask_staging.h`、T20 的 `full_step_warmup.h`）。
- `_surogate_fork_attach_serve()` 同样通过 `cmake_language(DEFER)` 运行，只在服务端构建中存在 `sinfer_ops` 时生效：把 `serve/` 中的解码单元加入 `sinfer_ops`，并为它加上同一个 PRIVATE 头文件目录（T16 的 wrapper 与 prompt launcher 包含 `surogate_fork/gqa_fork_shapes.h`）；开启服务端测试时，它也为上游的 `sinfer_gqa_attention_test` 加上这个头文件目录（其注册契约检查包含同一个头文件），再按 `sinfer_gqa_attention_test` 的设置构建 `sinfer_gqa_fork_shapes_test`，加入 `serve-tests`，并登记开关为 0 和 1 的 ctest 条目。
- 训练和服务端的头文件根目录中都没有名为 `surogate_fork` 的目录，所以 fork 头文件不会遮蔽上游头文件，上游头文件也不会遮蔽 fork 头文件。
- 原生构建因此依赖 `csrc/` 之外的 `fork/native/`：只复制 `csrc/` 的构建方式会在 configure 时找不到 `fork_native.cmake` 而失败。

两处自动检查这个约定，丢掉任何一行或 `fork_native.cmake` 中的调用都会让 fork CI 失败：

- `fork/dev check`：ledger 的 `[[exact_diff]]` 列出这两行，`check_divergence.py` 要求 `git diff -U0 <merge base> -- csrc/CMakeLists.txt` 改动的行恰好是它们（按文件检查时，只要还有一行改动，文件就仍是登记过的 M）。
- `fork/dev test openmp`：configure 时 [openmp-contract/hook_contract.cmake](openmp-contract/hook_contract.cmake) 从 `csrc/CMakeLists.txt` 读出挂接行，从 `surogate_fork_configure_common(<target>)` 取得目标名，确认挂接行位于 `add_library(<target> ...)` 之后，在一个不编译的同名替身目标上按原样执行，再检查目标链接了 `OpenMP::OpenMP_CXX` 并有 `fork/native/include` 头文件目录。同一个项目的 ctest 再确认 `OpenMP::OpenMP_CXX` 确实以多线程运行。替身只执行挂接行：上游改名 `surogate-common` 时这个检查不需要修改，上游自己链接 OpenMP 时要删去其中的 OpenMP 检查（见 [sync.md](../docs/sync.md) 冲突处理手册中的 T5）。

## 新增 fork 原生代码的规则

1. fork 的 C++、CUDA 和 CMake 文件放在 `fork/native/` 下，不要在 `csrc/src` 中新建文件。头文件放在 `include/surogate_fork/`；上游代码调用、需要编译进 `surogate-common` 的源文件放在 `src/`，并在 `surogate_fork_configure_common()` 中用 `target_sources()` 加入，这样 fork 测试调用的就是生产代码所用的同一个函数。服务引擎的源文件放在 `serve/`，在 `_surogate_fork_attach_serve()` 中加入对应的 `sinfer_*` 目标。
2. 需要接入上游目标时，修改 `fork_native.cmake`（例如在 `_surogate_fork_attach_tests()` 中对已有目标调用 `target_sources()` 或 `target_include_directories()`），不要在 `csrc/CMakeLists.txt` 中增加行。
3. 对上游源文件本身的修改（例如 `#include` 一个 fork 头文件）仍属于某个 ledger topic，并在同一提交中更新 ledger。

## 运行 OpenMP 接线测试

运行 `fork/dev test openmp`（不需要 GPU、CUDA 编译器或 CUDA 头文件）。测试用 `surogate_enable_training_openmp()` 构建一个主机共享库，检查它确实以两个 worker 运行，并得到与串行参考完全一致的整数结果。如果运行时把线程上限设为 1（例如 `OMP_THREAD_LIMIT=1`），测试以退出码 77 报告跳过，ctest 将其列为 Skipped；`OMP_NUM_THREADS` 不影响结果，因为测试用 `num_threads(2)` 显式请求两个线程。

同一个库还编译了 `include/surogate_fork/omp_team.h`（与 `surogate-common` 中的 `cpu_adamw.cpp` 用同一个 OpenMP 辅助函数 `surogate_enable_training_openmp()`；该测试不设构建类型，不使用训练库的优化级别），`training-openmp-team` 检查：低于阈值的循环串行；按 NVML 方式绑定的线程，其线程数不超过自身掩码中的 CPU 数，且每个 team 线程都在调用线程的掩码内；共用 CPU 的线程平分 CPU，掩码不相交的线程不平分，线程退出或空闲超过共享窗口（它两次并行区之间最长空闲间隔的 2 倍，至少 1 秒）后不再计入；`ordered_block_sum` 在任何线程数下的结果都与串行的分块顺序逐位相同。需要更多 CPU 的检查在 CPU 不足时打印 skipped，其余检查照常运行。`cpu-step-contract`（`cpu_step_check.cpp`，ledger T13–T15）链接同一个库，库中还有生产用的 `src/cpu_step_kernels.cpp`，用它自己的编译选项并按训练库 Release 构建的 `-O3` 编译，因此测的是向量化后的循环：把 `DslGradStore` 的主机端协议（micro-step 0 的覆盖、之后的累加、覆盖缺口、比缓冲短或从中间开始的拷贝、共用的缓冲）与上游的 memset 路径逐字节对比，`SUROGATE_FORK_HOST_GRAD_MEMSET=1` 时也对比，并确认这时记录器不清零、开关只读取一次；主机梯度加法与 AdamW 和上游循环逐位比较（FP32 结果中 NaN 的 payload 除外；参考循环按 `-O3`、默认的浮点收缩编译，与训练库中的 `cpu_adamw.cpp` 相同，基线 ISA 有 FMA 时只确认 AdamW 开关被拒绝）；serial group 中的主机函数与 rank 线程互不减半线程数，主机函数给 rank 线程留一个 CPU。构建目录 `build/fork-openmp` 已被根目录的 `.gitignore` 忽略。

## 运行 fork 的 C++ 测试

训练侧的用例（T1–T4、T10）原先追加在上游测试文件里，现在单独成文件，但仍编译进同一个测试可执行文件（`dsl-run-state-tests`、`integration-tests`），所以测试名和 ctest 条目都没有变化。辅助函数（例如 `require_cuda`、`DeviceTensor`）是上游测试文件中同名函数的副本，放在匿名命名空间里，避免与上游副本冲突。

这些测试需要 CUDA 构建环境，涉及 GPU 的用例在没有 GPU 时跳过。在 GPU 主机上运行 `fork/dev test native`：它沿用 `make configure` 的参数（CUDA、NCCL 等）并加上 `-DBUILD_TESTS=ON` 配置 `csrc/build`，再构建并用 ctest 运行这两个可执行文件（以及 `dsl-weight-loader-tests`）。

T16 的 `test-fork-gqa-shapes.cpp` 不同：它是服务端构建（`csrc/build-serve`）中单独的可执行文件 `sinfer_gqa_fork_shapes_test`。fork 登记了三个新的 ctest 条目：`sinfer_gqa_fork_shapes_test` 与 `sinfer_gqa_fork_shapes_on_test` 以开关 0 和 1 运行它，`sinfer_gqa_attention_fork_shapes_on_test` 以开关 1 运行上游的 `sinfer_gqa_attention_test`。`fork/dev test native` 不构建也不运行它们。用 `make serve-test-build` 构建，再 `cd csrc/build-serve && ctest -R '^sinfer_gqa_(attention|fork_shapes)'` 运行；`fork/dev test gpu` 的 `make serve-check` 也会运行它们。

之后只运行 fork 的用例（这些标签只出现在 fork 的测试中）：

```bash
csrc/build/dsl-run-state-tests '[dispatch-pp],[tied-head],[doc-mask]'
csrc/build/integration-tests '[fused-dx],[wgrad-zero],[recipe],[nvfp4-4o6]'
```
