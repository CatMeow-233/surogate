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
| I4 CPU 套件 | 270 个用例：269 个通过，1 个跳过（同 I3，没有 scipy）；I4 新增 64 个。`fork/dev test bilingual` 的墙钟时间 30.4 s、31.7 s、28.0 s（pytest 报告的时间）；I4 中最慢的用例是完整规模的 test 题目生成与检查（约 6.1 s），其次是 Track S 的确定性检查（约 1.5 s，构建两次 ci 规模）与命令行全流程（约 1.2–1.7 s） | `FORK_PY=… fork/dev test bilingual`，`--durations` | C1 | I4（本行所在的提交） | 2026-11-02 |
| Track S 完整规模构建（`--profile full --seed 20261102`） | 墙钟 5 分 31 秒（user 4 分 55 秒），输出 1.1 GB（seed、zh、en 三个成分各约 344 MB 的 JSON Lines）；训练文档：D_seed 54.8 万篇（传记 40 万、QA 4.8 万、E3-par 格式 2 万、E3-ctx 格式 8 万），D_zh、D_en 各 46.4 万篇，D_parallel 1 万对；题目 4 万道（每族每变体每方向 test 2,000、dev 500）。数据卡的 8 项检查全部为 true；文本扫描得到的曝光次数恰为声明值（100、100、100、20）；E3-par 单事实泄漏检验 p：same_city 的 A 城市 0.287、B 城市 0.0041，month_sum 的 A 月份 0.741、B 月份 0.593（门槛 0.001。B 城市的 0.0041 高于门槛但偏低；按构造两种标签下 B 的城市都均匀，另用 20 个种子按同样规模重做 same_city 题目，A、B 的 p 值没有异常：B 有 1/20 低于 0.05，A 为 0/20）；属性独立性的 18 个检验最小 p = 0.117；“谁出生更早”对照的单事实 Bayes 正确率 0.750，χ² = 3331（df 9），p 在双精度下下溢为 0 | 一次性脚本（未入库）调用 `build_track_s`，输出在会话临时目录，没有入库 | C1 | I4 | 2026-11-02 |
| 完整规模题目的生成与独立检查（种子 1，只生成题目） | 4 万道题生成 2.6 s，检查器 3.0 s，全部一致。在改用属性词表的联合匹配之前，同一检查误报了 157 道 E4 题（“摄影师”“陶艺家”被同时读成爱好“摄影”“陶艺”），见 spec 的 I4 落地记录第 13 条 | 一次性脚本（未入库） | C1 | I4 | 2026-11-02 |
| 运行手册 G1′ 数据准备的命令序列（合成的临时语料，不是自然语料） | 按 [运行手册](runbook.md#g1-可测性-pilot) 的顺序逐条运行（Track S 用 ci 规模），全部以退出码 0 结束；D_seed shard 的成分比例最大误差 2.2e-5，seq 1024 下混语窗口比例 1.0（临时语料的文档很短，自然语料会低得多）。语料太小时 `shard --seed-mix` 以退出码 1 报告自然文本 token 不足，没有写出 shard | 临时语料由一次性脚本生成（未入库） | C1 | I4 | 2026-11-02 |
| I5 CPU 套件 | 319 个用例：318 个通过，1 个跳过（同 I3，没有 scipy）；I5 新增 49 个。`fork/dev test bilingual` 的墙钟时间 40.9 s、39.2 s（pytest 报告的时间）。I5 中最慢的是 `test_train_loop.py` 的子进程 fixture，约 5.3–5.5 s（五个单线程子进程同时启动：I5-T1 的两次完整运行、I5-T2 的两次被杀的运行、I5-T3 的一次崩溃，各自导入 torch），其次是 I5-T5 的 300 步（约 1.7–1.9 s）；I5 的其余用例都在 1 s 以内。训练用的 tiny 模型在单线程下每步约 5–10 ms | `FORK_PY=… fork/dev test bilingual`，`--durations` | C1 | I5（本行所在的提交） | 2026-11-02 |
| I5-T5 的收敛步数（tiny profile，固定批次，lr 峰值 3e-3，warmup 10%） | 初始 loss 4.186，第 60 步首次低于 0.1 倍；同一设置下 lr 1e-3 为第 154 步，1e-2 为第 30 步。这是实现检查（模型能拟合一个固定批次），不是实验结果 | 一次性脚本（未入库），按测试中的设置调用 `train` | C1 | I5 | 2026-11-02 |
| I6 CPU 套件 | 370 个用例：369 个通过，1 个跳过（同 I3，没有 scipy）；I6 新增 51 个。`fork/dev test bilingual` 的墙钟时间 53.2 s、48.5 s、51.6 s（pytest 报告的时间）。I6 中最慢的是运行手册的演练（约 8 s：57 条包命令各做一次 `--dry-run`，其中 55 条再以 tiny 规模真正运行），其次是评测 fixture 的组装与加载（约 1 s）和 bootstrap 覆盖率的 400 次模拟（约 0.5 s） | `FORK_PY=… fork/dev test bilingual`，`--durations` | C1 | I6（本行所在的提交） | 2026-11-02 |
| I6 统计检验的校准（模拟，不是结果） | 聚类 bootstrap（60 簇 × 8 题，簇效应 SD 0.3，2000 次重抽样）400 次模拟的 95% CI 覆盖率 0.945；重复级单侧 t 检验在零假设下（R = 3，重复间 SD 0.05，300 道 Bernoulli 题）4000 次模拟的一类错误率 0.0505（α = 0.05）；每重复差的 SD 为 0.03、δ = 5pp、单侧 α = 0.0125 时，R = 3、4、5、6 的功效为 0.206、0.414、0.622、0.781（数值积分；R = 5 的 40,000 次模拟为 0.624），所以这一 SD 下 R = 6 仍不足 0.8 | 与 `test_eval_stats.py` 相同的设置，一次性脚本（未入库）打印数值 | C1 | I6 | 2026-11-02 |
| GPU 测试代码的 CPU 演练（`BILINGUAL_GPU_TESTS_ON_CPU=1`，**不是 G0 结果**） | 12 个用例：10 个通过，2 个只能在 CUDA 上运行的跳过（(g3) SDPA 后端与 scipy 对拍）。CPU bf16 autocast 下：tiny 训练的 pack 的探针 max-abs 为 0.0026（zh）与 0.0030（en），argmax 一致率 1.0；fp32 下差恰为 0.0；随机初始化的 pilot 几何 max-abs 为 0.022（swiglu）与 0.017（slot），swiglu 的 argmax 一致率只有 0.875（未训练模型的 logits 几乎相等，见 spec 的 I6 落地记录第 17 条）；CPU 上 (g4)、(g5) 的 loss 相对差为 0.0 | `BILINGUAL_GPU_TESTS_ON_CPU=1 BILINGUAL_G0_RECORD=… pytest fork/bilingual/tests/gpu` | C1 | I6 | 2026-11-02 |
| CR1 CPU 套件（代码评审的修正之后） | 395 个用例：394 个通过，1 个跳过（同 I3，没有 scipy）；CR1 新增 25 个。`fork/dev test bilingual` 连续两次的墙钟时间 51.7 s、50.9 s。最慢的仍是运行手册的演练（约 8 s），其次是 pilot 预算报告的 fixture（约 5.4 s，多了 B12a 一行）与完整规模题目的生成与检查（约 5.4 s） | `FORK_PY=… fork/dev test bilingual`，`--durations` | C1 | CR1（本行所在的提交） | 2026-11-02 |
| CI 步骤 `Bilingual CPU tests`（I1-A7；I1 至 I6 与 CR1 的用例） | **未运行**：这些提交都没有推送，fork CI 尚未执行这一步；90 秒的预算在 CI 的 CPU lane 上待测（本机 C1 上约 51 s） | 推送后看 `fork-ci.yml` 中 `fork-tests` job 的这一步 | — | — | — |

## GPU

没有记录。G0 之前这里为空；G0 至 G3 的结果按 [运行手册](runbook.md) 中各门槛的记录表填写，每行注明主机、提交和日期。
