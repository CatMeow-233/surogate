# 实测记录

开发过程中测得的数字，只代表当时的代码、数据和机器。每条注明主机、代码所在的提交和日期；代码改动后数字可能过期，需要时按“做法”一栏重测，并新增一行，不改写旧行。

## 主机

- **C1：** 本仓库开发用的 CPU 容器。4 个 vCPU（Intel Xeon @ 2.10GHz），16 GiB 内存，无 GPU；Python 3.12.3，torch 2.11.0（CUDA 构建，只在 CPU 上运行），pytest 9.1.1，pytest-timeout 2.4.0，没有 scipy。
- **RTX 5090 主机：** 尚无记录。按 [运行手册的 G0](runbook.md#g0-环境与-gpu-不变量) 建好环境后，在这里补充 GPU 型号、驱动、torch 的确切 wheel 与 CUDA 版本。

## CPU 测试与 CI

| 内容 | 结果 | 做法 | 主机 | 提交 | 日期 |
|---|---|---|---|---|---|
| I1 CPU 套件（I1-A7 的本机部分） | 21 个用例全部通过；`fork/dev test bilingual` 连续三次的墙钟时间 1.3 s、0.8 s、0.7 s（冷缓存的第一次约 11 s）；最慢的用例约 0.06 s（导入检查的子进程） | `FORK_PY=… fork/dev test bilingual`，`--durations` | C1 | I1（本表所在的提交） | 2026-11-02 |
| I2 CPU 套件 | 133 个用例全部通过（I1 的 21 个，I2 的 112 个）；`fork/dev test bilingual` 连续三次的墙钟时间 12.3 s、11.7 s、11.4 s；最慢的是 pilot 预算报告的 fixture，约 4.4 s（在 meta device 上建 18 行的模型并执行前向），其次是写 pack 的 fixture，约 1.4 s | `FORK_PY=… fork/dev test bilingual`，`--durations` | C1 | I2（本行所在的提交） | 2026-11-02 |
| I2 与 v1.0 的 logits 差（I2-A1） | smoke 几何、fp32、单线程：强制路由与随机混合路由下最大差 4.8e-7（T = 128）与 3.6e-7（T = 37），容差 1e-6；差来自 RoPE 角度表（v1.0 用 fp32，本包用 float64，见 spec 的 I2 落地记录第 1 条） | 一次性脚本（未入库），按测试中的构造打印每种路由的最大差 | C1 | I2 | 2026-11-02 |
| I3 CPU 套件 | 206 个用例：205 个通过，1 个跳过（`test_i3_c10_numpy_hungarian_equals_scipy`，本机没有 scipy）；I3 新增 73 个。`fork/dev test bilingual` 连续三次的墙钟时间 15.7 s、14.3 s、15.2 s；I3 中最慢的用例是 DARE 的 2000 次抽样（约 0.6 s）和 pilot meta 模型上 B8/B9 的计数（约 0.5 s） | `FORK_PY=… fork/dev test bilingual`，`--durations` | C1 | I3（本行所在的提交） | 2026-11-02 |
| I3 的 scipy 对拍（I3-C10 的跳过部分） | 用 `uv pip install --target` 把 scipy 1.18.1 放进会话临时目录（没有装进任何环境），只对这一次运行加 `PYTHONPATH`：整个套件 206 个用例全部通过，没有跳过。另用一次性脚本（未入库）比较：300 个随机矩阵（n 在 2–99 之间，一半是有同值的整数矩阵）上 numpy 与 scipy 的最优值全部相等；n = 64、128、256 的随机矩阵上置换相同，numpy 用时 0.005 s、0.017 s、0.048 s；scipy 求解 n = 1728 用时 0.20 s。`fork/.venv-bilingual` 中尚未运行 | 见左栏 | C1 | I3 | 2026-11-02 |
| v1.0 `compare.py` 的 `run(0, 12)`（恒等式检查，不是结果） | 与评审 MA-10 的复算一致：`wrong_route`、`weight_average` 与 `smoke_results.json` 逐位相同，`top2_average` 与 `2n_concat` 互换了最后一位 fp32 ulp，`2n_top2_max_abs_diff` 为 2.4e-7（文件中为 3.0e-7）。所以该文件来自没有记录的 `--steps 12 --seeds 0`；没有写成测试，因为逐位结果依赖 torch 版本与 CPU kernel | 一次性脚本（未入库），按路径加载 `fork/bilingual/reference/compare.py` 后调用 `run(0, 12)` | C1 | I3 | 2026-11-02 |
| CI 步骤 `Bilingual CPU tests`（I1-A7；I1 至 I3 的用例） | **未运行**：I1、I2、I3 的提交都没有推送，fork CI 尚未执行这一步；90 秒的预算在 CI 的 CPU lane 上待测 | 推送后看 `fork-ci.yml` 中 `fork-tests` job 的这一步 | — | — | — |

## GPU

没有记录。G0 之前这里为空；G0 至 G3 的结果按 [运行手册](runbook.md) 中各门槛的记录表填写，每行注明主机、提交和日期。
