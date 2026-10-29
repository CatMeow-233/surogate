# 实测记录

开发过程中在 CPU 容器上测得的数字，只代表当时的代码、数据和机器。每条注明主机、代码所在的提交和日期；代码改动后数字可能过期，需要时按“做法”一栏重测，并新增一行而不是改写旧行。GPU 上的数字按 [运行手册](runbook.md) 记进 `$TLC_RUNS/env/`，不写在这里。

主机 C1：本仓库开发用的 CPU 容器，4 个共享 vCPU，无 GPU，Python 3.12，torch 2.11（CPU 运行），重负载用 `nice -n 10`。

| 内容 | 结果 | 做法 | 主机 | 提交 | 日期 |
|---|---|---|---|---|---|
| G1(b) 的 `tolerance` | `init_seed=1` 的 A0 约为 0.045；A0、batch 2、512 token 单线程约 6 秒 | `prototypes/ref_mamba2.py`，见 [model.md](model.md#g1b-的-goldens) | C1 | 9b2392a2（P0-I4） | 2026-10-29 |
| 原生 tokenizer 一致性 | 三个预分词候选各比较 295 段文本（640 万字符，含边界用例和 CRLF 版本），全部一致；不代替 G0 | 用原生 tokenizer 源码编译的 CPU 程序，在本机的 stdlib 与仓库文档上训练三个候选 | C1 | 5339d1dc（P0-I5） | 2026-10-29 |
| 原生 DataLoader 读出的行 | 与 `served_rows()` 逐行一致 | 用原生 `dataloader.cpp` 编译的 CPU 程序读取本机语料建出的 shard | C1 | 5339d1dc（P0-I5） | 2026-10-29 |
| `pad_fraction` | 约 4.6% | 本机 stdlib 与仓库文档按整文件切窗，行长 512 | C1 | 5339d1dc（P0-I5） | 2026-10-29 |
| 去污染校准 | 基准 HumanEval、HumanEval+、MBPP 全部 974 个、MBPP+，共 1,680 条；宿主为本机 CPython stdlib 和一个 site-packages 中随机抽取的 2,000 个非测试文件（24.2 MB）。删除规则在宿主上没有误报；`code13` ≥ 0.5 标记 11 个文件（0.55%，覆盖率都低于 0.6），`name` 标记 4 个；40 个植入删除 27 个（`code13` 27 个，`ast` 20 个），漏掉的 13 个都是 2–4 行的短解答，信息性 13-gram 少于 15 个、AST 不到 40 个节点（通用短函数的功能等价不算泄漏）；不改名、直接复制时删除 29/40；单进程约 0.75 MB/CPU 秒 | `python -m surogate_tlc.data.decontam ... --plant 40` | C1 | 082a42da（P0-I7）；之后 `ast` 规则的 α-重命名改过（加入 except-as、import-as、global/nonlocal，以及调用本段代码中定义的函数时的关键字实参名），宿主也改成只取非生成的来源，召回与误报都要在 G0 重测 | 2026-10-29 |
| T3 套件的 floor | stdlib 建的套件（val 占 30%，1,100 项，平均 5.3 个候选）：chance 26.9%，`recent` 16.4%，`frequent` 22.5%。实验笔记引用的 stdlib 数字（27.8%、25.3%、31.7%）包括只有一个候选的读取，两者不能直接比较 | `python -m surogate_tlc.data.natural` 后对 floor 作统计 | C1 | 082a42da（P0-I7） | 2026-10-29 |
| 合成绑定任务的默认规模 | 每半 8,382 项（T1 4,290、T2 4,092），训练流 2,000,112 字符（1,027 个文档、4,088 个程序），用时约 49 秒；最长的 prompt 2,501 个字符；与评测 AST 相撞后放弃的抽样 17 次（T1 的 `flat` 与 `shadow-out` 中 `live` 很小、结构只有几种的格），`eval_items_with_a_training_ast` 为 0 | `python -m surogate_tlc.data.synth OUT --seed 0`（默认参数） | C1 | P0-fix（本表所在的提交） | 2026-10-29 |
| G1、G2 参照套件 | G1.1–G1.7 在 fp64 与 fp32 下通过、五种变异全部检出，约 2.3 CPU 秒；G2.1–G2.3 约 4.7 CPU 秒（G2.3：下限 1.85 nats，完整模型 0.0016，消融模型 1.86） | `python -I -B fork/tlc/prototypes/g1_suite.py`、`g2_suite.py`，单线程 | C1 | P0-fix（本表所在的提交） | 2026-10-29 |
