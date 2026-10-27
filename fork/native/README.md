# fork 原生代码

本目录存放 fork 自有的 C++ 与 CMake 代码。上游原生构建 `csrc/CMakeLists.txt` 只通过一个两行的挂接点引入这里的内容，因此 fork 在上游文件中的改动保持最小，上游在 `csrc/` 中新增或修改文件时也不会与 fork 文件冲突。

## 目录

| 路径 | 内容 | ledger topic |
|---|---|---|
| [cmake/fork_native.cmake](cmake/fork_native.cmake) | 挂接入口：定义 `surogate_fork_configure_common()`，并在 `csrc` 目录处理完毕后为测试目标补充设置 | T5 |
| [cmake/training_openmp.cmake](cmake/training_openmp.cmake) | `surogate_enable_training_openmp()`：把 `OpenMP::OpenMP_CXX` 链接到训练库 | T5 |
| [include/surogate_fork/](include/surogate_fork/) | fork 头文件，源码中写作 `#include "surogate_fork/<文件名>"` | T3 |
| [openmp-contract/](openmp-contract/) | 独立的 OpenMP 接线回归测试，只需主机 C++ 编译器和 OpenMP | T5 |
| [tests/](tests/) | fork 的 C++ 测试用例，编译进上游的 `dsl-run-state-tests`（T1）和 `integration-tests`（T2–T4） | T1–T4 |

各 topic 的提交、开关和冲突历史见 [ledger](../patches/ledger.toml)。

## 挂接约定

`csrc/CMakeLists.txt` 中 fork 只有两行（ledger T5），位于 `surogate-common` 的 `target_link_libraries` 之后：

```cmake
include(${CMAKE_CURRENT_SOURCE_DIR}/../fork/native/cmake/fork_native.cmake)
surogate_fork_configure_common(surogate-common)
```

- `surogate_fork_configure_common(<target>)` 为目标启用训练用的 OpenMP，并把 `fork/native/include` 加为 PRIVATE 头文件目录。目录不是 PUBLIC，因此链接 `surogate-common` 的其他目标看不到它。
- `_surogate_fork_attach_tests()` 通过 `cmake_language(DEFER)` 在 `csrc` 目录处理完毕时运行，因为测试目标定义在挂接点之后。它为 `integration-tests` 加上同一个头文件目录，并用 `target_sources()` 把 `tests/` 中的源文件加入 `integration-tests` 和 `dsl-run-state-tests`；每个目标都先用 `if(TARGET ...)` 判断，未开启 `BUILD_TESTS` 时 configure 不受影响。
- 训练和服务端的头文件根目录中都没有名为 `surogate_fork` 的目录，所以 fork 头文件不会遮蔽上游头文件，上游头文件也不会遮蔽 fork 头文件。
- 原生构建因此依赖 `csrc/` 之外的 `fork/native/`：只复制 `csrc/` 的构建方式会在 configure 时找不到 `fork_native.cmake` 而失败。

检查上游文件中的 fork 改动仍然只有这两行：

```bash
git diff -U0 upstream/main -- csrc/CMakeLists.txt   # 只有一个 hunk，恰好新增 2 行
```

## 新增 fork 原生代码的规则

1. fork 的 C++、CUDA 和 CMake 文件放在 `fork/native/` 下，不要在 `csrc/src` 中新建文件。头文件放在 `include/surogate_fork/`。
2. 需要接入上游目标时，修改 `fork_native.cmake`（例如在 `_surogate_fork_attach_tests()` 中对已有目标调用 `target_sources()` 或 `target_include_directories()`），不要在 `csrc/CMakeLists.txt` 中增加行。
3. 对上游源文件本身的修改（例如 `#include` 一个 fork 头文件）仍属于某个 ledger topic，并在同一提交中更新 ledger。

## 运行 OpenMP 接线测试

在仓库根目录执行（不需要 GPU、CUDA 编译器或 CUDA 头文件）：

```bash
cmake -S fork/native/openmp-contract -B build/fork-openmp
cmake --build build/fork-openmp
ctest --test-dir build/fork-openmp --output-on-failure
```

测试用 `surogate_enable_training_openmp()` 构建一个主机共享库，检查它确实以两个 worker 运行，并得到与串行参考完全一致的整数结果。如果运行时把线程上限设为 1（例如 `OMP_THREAD_LIMIT=1`），测试以退出码 77 报告跳过，ctest 将其列为 Skipped；`OMP_NUM_THREADS` 不影响结果，因为测试用 `num_threads(2)` 显式请求两个线程。构建目录 `build/fork-openmp` 已被根目录的 `.gitignore` 忽略。

## 运行 fork 的 C++ 测试

`tests/` 中的用例原先追加在上游测试文件里，现在单独成文件，但仍编译进同一个测试可执行文件，所以测试名和 ctest 条目都没有变化。辅助函数（例如 `require_cuda`、`DeviceTensor`）是上游测试文件中同名函数的副本，放在匿名命名空间里，避免与上游副本冲突。

这些测试需要 CUDA 构建环境，涉及 GPU 的用例在没有 GPU 时跳过。配置时沿用 `make configure` 的参数（CUDA、NCCL 等），再加上 `-DBUILD_TESTS=ON`：

```bash
cmake -S csrc -B csrc/build -DBUILD_TESTS=ON
cmake --build csrc/build --target dsl-run-state-tests integration-tests
ctest --test-dir csrc/build -R '^(dsl-run-state-tests|integration-tests)$' --output-on-failure
```

只运行 fork 的用例（这些标签只出现在 fork 的测试中）：

```bash
csrc/build/dsl-run-state-tests '[dispatch-pp]'
csrc/build/integration-tests '[fused-dx],[wgrad-zero],[recipe]'
```
