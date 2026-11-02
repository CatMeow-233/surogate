# 双语独立专家：冻结骨干上的中英 FFN 专家组合

本目录是 owner 的预注册研究稿《独立训练的中英专家能否在冻结骨干上组合：非线性前后专家混合、语言切换与翻译 Bridge 的预注册受控研究》（v1.1，2026-11-02，下称“手稿”；手稿与其评审不在本仓库中，由 owner 保存）的 P0 实现：独立的纯 PyTorch 包 `surogate_bilingual`。

研究的问题 [待实验]：从同一个双语种子分叉、在冻结的共享骨干（嵌入兼 tied head、attention、norm）上**各自独立训练**的中文 FFN 专家与英文 FFN 专家，组装后在没有任何联合训练的条件下能否完成跨语言组合；在同一组冻结的专家检查点上，“先混合、再经共享冻结的非线性读出”（pre）与“先读出、再混合”（post）是否不同；独立专家相对单一双语专家要付出多少切换代价与模块化代价。通道拼接等于 Top-2 输出求和是已知恒等式 [已知数学事实]，只作零假设与数值检查，不算贡献。

**当前状态：实现步骤 I1–I3 已完成（只在 CPU 上验证）。** 包里有模型核心、路由、组装、冻结审计与参数预算（见下文“模型、路由与组装”），以及融合算子、同初始化合并与 router 门控（见下文“融合、合并与 router 门控”）；还没有数据、训练循环与评测代码，所以没有任何实验结果。凡涉及 RTX 5090 的吞吐、显存和时长，都是 [估计] 或 TBD，在 [运行手册](docs/runbook.md) 的 G0、G0.5 实测并记入 [实测记录](docs/measurements.md) 之前，不得写成结果。证据层级沿用手稿：[假设]、[建议]、[待实验]、[已知数学事实]、[已有论文结果]、[估计]。

## 决定 B-D1（已采用，owner 可以推翻）

本项目是独立的 PyTorch 包，**不使用** Surogate 的原生训练器：

- 原生训练器只绑定 `token_ids`、`position_ids`、`targets` 三个输入，没有逐 token 的路由输入；
- 本研究需要的算子在原生侧都要新增：逐 token 的 gather/scatter dispatch、块门控拼接、非线性前与非线性后的槽位混合、通道选择；
- 约 100M 参数的模型在一张 5090 上用 eager PyTorch、bf16 autocast 和 SDPA 应当够用 [估计；G0.5 实测]。

以后若要把某个结论移到原生训练器上，按 ledger 规则另立 topic。决定的完整记录见 [docs/spec.md](docs/spec.md#决定-b-d1)。

本包位于 `fork/` 之下（ledger `[owned]` 的前缀），不修改任何上游文件，所以不需要 ledger topic。

## 目录

| 路径 | 内容 |
|---|---|
| [pyproject.toml](pyproject.toml) | 包 `surogate-bilingual`，src 布局，`dependencies = []`，不发布。`[tool.ruff]` 沿用仓库根目录的设置，只多排除 `reference/`，免得 `fork/dev fmt` 改写 owner 的原始文件 |
| [requirements-gpu.txt](requirements-gpu.txt) | GPU 主机环境 `fork/.venv-bilingual` 的依赖；torch 按 [运行手册的 G0](docs/runbook.md#g0-环境与-gpu-不变量) 另装 |
| [src/surogate_bilingual/](src/surogate_bilingual/) | 包代码。`import surogate_bilingual` 不导入 torch，也不导入任何 `surogate` 模块；子模块按需导入 torch |
| [docs/](docs/) | 决定与数据契约、运行手册和实测记录，见下文“文档” |
| [tests/cpu/](tests/cpu/) | CPU 测试，用 `fork/dev test bilingual` 运行；`golden/` 中是 pilot 几何的预算报告；`-m identity_check` 只运行恒等式检查 |
| [reference/](reference/) | owner 的 v1.0 代码与数据，逐字节原样保存，sha256 固定在 `MANIFEST.json` 中；不打包，包代码不导入，只供测试按路径加载，见 [reference/README.md](reference/README.md) |

后续步骤还会加入运行配置的 examples 目录（I4、I5）和 GPU 测试目录（I6）。

本包不进任何 wheel：根目录 `pyproject.toml` 的原生 wheel 只打包 `surogate/`，实验 wheel 只打包 `fork/experimental/src` 中的 `surogate.experimental*`。`tests/cpu/test_scaffold.py` 从这两个 pyproject 读出打包目录并检查本包不在其中，还检查：

- 导入本包不会加载 torch 或 `surogate`；
- 包代码只导入标准库、torch、numpy、tokenizers、safetensors，scipy 只能在 `try`/`except ImportError` 中导入；
- 包代码不导入 `surogate`、`surogate._fork`、`surogate.experimental`，也不按路径加载任何文件（`reference/` 只能这样被加载）；
- `reference/` 中的文件与 `MANIFEST.json` 一致，且其中没有 `test_*.py`。

## 环境

- **CPU 测试：** `fork/dev test bilingual` 用 `FORK_PY` 指定的解释器，默认是仓库根目录的 `.venv/bin/python`，需要 CPU 版 torch 和 `fork/tools/requirements-cpu-lane.txt`，准备方法见 [fork/README.md 的“首次准备环境”](../README.md#首次准备环境)。测试只把 `fork/bilingual/src` 放到 `sys.path` 最前面，从检出目录导入本包，不安装任何包；`surogate_bilingual` 若从别处导入（已安装的副本、另一份检出），`conftest.py` 让整个会话报错。CPU 测试环境没有 scipy。
- **GPU 主机（RTX 5090）：** 专用隔离环境 `fork/.venv-bilingual`，依次装入 sm_120 的 CUDA 版 torch（记录确切的 wheel）、`requirements-gpu.txt` 和本包（editable，`--no-deps`），步骤见 [运行手册的 G0](docs/runbook.md#g0-环境与-gpu-不变量)。不要在仓库根目录运行 `uv run` 或 `uv sync`。

## 命令行

在 GPU 主机上用 `fork/.venv-bilingual/bin/python -m surogate_bilingual 命令 ...` 运行，不带参数时列出命令。命令由 `surogate_bilingual.__main__.COMMANDS` 一张表定义：每个命令对应一个子模块及其入口函数，并且都接受 `--dry-run`（只解析参数、检查输入，不在临时目录之外写任何东西）。模块还没落地的命令以退出码 2 结束，并说明它在哪个步骤实现。

| 命令 | 作用 | 实现步骤 |
|---|---|---|
| `env` | 写出 `environment.json`（环境指纹） | I5 |
| `tokenizer` | 训练或检查 tokenizer | I4 |
| `manifest` | 生成语料 manifest | I4 |
| `split` | 按文档划分数据集合 | I4 |
| `decontam` | 重叠报告与过滤 | I4 |
| `shard` | 写出 shard | I4 |
| `synth` | 生成 Track S | I4 |
| `train <row.toml> [--stage S] [--dry-run]` | 按行配置运行一次训练；已有状态时恢复 | I5 |
| `assemble` | 写出组装 manifest | I2 |
| `verify` | 保真度报告 | I2 |
| `eval-tasks` | 候选排序与生成评分 | I6 |
| `eval-nll` | NLL、BPB 与 2×2 矩阵 | I6 |
| `switch` | 切换代价分解 | I6 |
| `audit` | 预算与参数报告 | I2 |
| `stats` | 由结果做确认性检验 | I6 |
| `bench` | G0.5 吞吐与显存扫描 | I6 |
| `matrix` | 对照矩阵各行的状态 | I6 |
| `registry` | 列出或查看运行 | I5 |

## 模型、路由与组装（I2）

以下都只在 CPU 上验证过，CUDA 上的行为待 G0 检查。模块边界与数据契约见 [docs/spec.md](docs/spec.md)，验收与偏离见其中的 I2 落地记录。

- **配置**（`config`）：所有默认值都在 `DEFAULTS` 一张表中；`tiny`、`smoke`（= v1.0 的 smoke 几何）、`mech8`、`pilot` 四个 profile；`config_hash` 是规范 JSON 的 sha256。
- **模型**（`model`、`ops`）：`RoutedLM` 由冻结的共享骨干（嵌入兼 tied head、attention、norm；族 II 另含各层共享读出 Q、b；LP-k 另含非专家层的种子 FFN）、按名字存放的专家库和 executor 组成。两个族：`swiglu`（族 I，B4/B6/B7）与 `slot`（族 II，B4-II/B8/B9）。有两个以上专家时必须传入 `RoutePlan`。只有 executor 读取 plan：全部 token 走同一专家时直接调用该专家，与单专家模型逐位相同；one-hot 行走 gather/scatter；两项非零的行按模型的融合方式（`output`、`concat`、`pre`、`post`）计算。输出缓冲区与输入同 dtype，所以 bf16 autocast 下不再出现 v1.0 的 NEW-01 错误。
- **路由**（`routing`）：`RoutePlan` 是逐 token 的单纯形门控（P0 中每行至多两个非零项）；`plan_from_segments` 给出 `R_out`、`R_src`、`R_wrong`、`R_random`；`apply_gate_policy` 给出 `dual_const` 与 `dual_window`；`plan_from_tags` 在完整历史上计算后再切片，修正了 v1.0 的窗口截断问题（MA-09）。
- **组装**（`assembly`）：种子 pack 与专家 pack（safetensors 加 `pack.json`，记录内容哈希、种子指纹与探针 logits）；`assemble` 写出不复制权重的 manifest，对名字互换、同一专家占两个槽位、种子不同、config 或族不同、缺键、多键、文件被改等情况各报一种错误；`verify` 在强制路由下复现每个专家 pack 的探针（CPU fp32 期望差为 0.0）；`FreezePlan` 按 storage 审计优化器，并对每个冻结张量做内容哈希。
- **审计**（`audit`）：`p_total`、`p_update` 与执行路径上的 `p_active`（在 `TorchFunctionMode` 下记录前向实际读到的参数），解析 MAC 与 `FlopCounterMode` 交叉检查，以及手稿 §5.1 的预算行。

预算报告在 meta device 上从模块树重算，不分配权重内存，几秒内完成：

```bash
PYTHONPATH=fork/bilingual/src "${FORK_PY:-.venv/bin/python}" -m surogate_bilingual audit --profile pilot
```

它与手稿 §5.1 和评审的 `budget_v11.json` 逐项一致，只有 B10-ch 的权重 MAC 例外：本实现计入了块 router 的 108,672 MAC，手稿表没有计入，需要 owner 确认（spec 的 I2 落地记录第 11 条）。改动预算代码后，用同一命令加 `--out fork/bilingual/tests/cpu/golden/budget_pilot.json` 重新生成 golden 文件，并在提交说明中写明数字为什么变。

## 融合、合并与 router 门控（I3）

以下都只在 CPU 上以 fp64 参照测试检查过；它们检查的是 [已知数学事实] 与基线的定义，不是实验结果。验收与偏离见 [docs/spec.md 的 I3 落地记录](docs/spec.md#i3-落地记录)。

- **融合**（`ops.fusion`）：executor 用到的 `mix_outputs`（B6）、块门控拼接 `concat_forward`（B7）、`slot_pre`（B8）与 `slot_post`（B9，只做一次 Q 矩阵乘）；另有三个 P0 不运行的算子：`cross_link_lowrank`（v1.0 的交叉连接，只用来检查它在常数 g 下就是拼接的重参数化，即带门控的 LoRA，MA-06）、`channel_select`（先取出选中通道的权重再投影，不产生全宽中间量，并向参数审计声明读了多少元素；B10-ch 的布局）和 `ZeroInitAdapter`（B11，up 投影初始化为零）。算子不看路由，也不校验门控；单纯形由 `RoutePlan` 保证。
- **合并**（`ops.merge`，只作用于专家 FFN 状态）：`weight_average`（B3a；族 II 的 B3-II）、`task_arithmetic`（B3b）、`ties`（B3c）、`dare`（B3d），以及 Re-Basin：每层对通道相似度做一次精确指派（`solve_lap`：有 scipy 时用 scipy，否则用 numpy Hungarian，后者只接受 n ≤ 256，所以 pilot 的 n = 1728 需要 `fork/.venv-bilingual` 中的 scipy）；置换为恒等时标为“等同于平均”（NOV-6）。
- **router 门控**（`routing.router`）：`topk_softmax_gates` 先取 top-k、再只对选中的 logit 做 softmax，于是加入新专家时，没有选中它的 token 门控逐位不变（MA-11）；学习式 router 的训练推迟。
- **v1.0 的检查**：`verify_merge_math.py` 按原样运行并要求全部断言通过，它的各项检查移植并加强为单元测试；`compare.py` 的恒等式成为带 `identity_check` 标记的单元测试（恒等式检查），永不作为结果报告，可以单独运行：

```bash
fork/dev test bilingual -m identity_check
```

## 实现步骤

每个步骤由一个或多个签名提交落地；提交前 `fork/dev map`、`fork/dev check` 和 `fork/dev test bilingual` 都要通过。各步骤的交付物和验收条件见 [docs/spec.md 的“实现步骤”](docs/spec.md#实现步骤)。

| 步骤 | 内容 | 状态 |
|---|---|---|
| I1 | 脚手架与仓库接线 | 已完成（CPU）；CI 尚未运行，见 [实测记录](docs/measurements.md) |
| I2 | 模型核心、路由、组装、冻结、哈希、参数审计 | 已完成（CPU）；CI 尚未运行，GPU 上未运行 |
| I3 | 融合算子与合并，fp64 参照测试（复用 v1.0 的 `verify_merge_math.py`） | 已完成（CPU）；CI 尚未运行，GPU 上未运行 |
| I4 | 数据：tokenizer、manifest、划分、去污染、shard、采样器、Track S | 未开始 |
| I5 | 训练：单一训练循环、各 recipe、状态与恢复、运行登记 | 未开始 |
| I6 | 评测、统计与 RTX 5090 运行手册 | 未开始 |

## 文档

- [docs/spec.md](docs/spec.md)：决定记录（B-D1、影响代码默认值的 owner 决定）、模块边界、数据契约摘要、开关、实现步骤与验收条件、推迟项和已知限制。
- [docs/runbook.md](docs/runbook.md)：RTX 5090 上 G0–G3 的步骤、判据与记录表（全部未运行）。
- [docs/measurements.md](docs/measurements.md)：注明主机、提交和日期的实测记录。
- [reference/README.md](reference/README.md)：v1.0 原始文件的来源、用途与保存规则。
