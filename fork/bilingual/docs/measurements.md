# 实测记录

开发过程中测得的数字，只代表当时的代码、数据和机器。每条注明主机、代码所在的提交和日期；代码改动后数字可能过期，需要时按“做法”一栏重测，并新增一行，不改写旧行。

## 主机

- **C1：** 本仓库开发用的 CPU 容器。4 个 vCPU（Intel Xeon @ 2.10GHz），16 GiB 内存，无 GPU；Python 3.12.3，torch 2.11.0（CUDA 构建，只在 CPU 上运行），pytest 9.1.1，pytest-timeout 2.4.0，没有 scipy。
- **RTX 5090 主机：** 尚无记录。按 [运行手册的 G0](runbook.md#g0-环境与-gpu-不变量) 建好环境后，在这里补充 GPU 型号、驱动、torch 的确切 wheel 与 CUDA 版本。

## CPU 测试与 CI

| 内容 | 结果 | 做法 | 主机 | 提交 | 日期 |
|---|---|---|---|---|---|
| I1 CPU 套件（I1-A7 的本机部分） | 21 个用例全部通过；`fork/dev test bilingual` 连续三次的墙钟时间 1.3 s、0.8 s、0.7 s（冷缓存的第一次约 11 s）；最慢的用例约 0.06 s（导入检查的子进程） | `FORK_PY=… fork/dev test bilingual`，`--durations` | C1 | I1（本表所在的提交） | 2026-11-02 |
| I1 CI 步骤 `Bilingual CPU tests`（I1-A7） | **未运行**：本步骤的提交没有推送，fork CI 尚未执行这一步；90 秒的预算在 CI 的 CPU lane 上待测 | 推送后看 `fork-ci.yml` 中 `fork-tests` job 的这一步 | — | — | — |

## GPU

没有记录。G0 之前这里为空；G0 至 G3 的结果按 [运行手册](runbook.md) 中各门槛的记录表填写，每行注明主机、提交和日期。
