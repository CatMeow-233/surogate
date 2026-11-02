# 运行手册（RTX 5090，G0–G3）

命令都在 GPU 主机的仓库根目录、环境 `fork/.venv-bilingual` 中运行。门槛按顺序进行：某一步以非零退出码结束、或判据不成立，就停止，保存输出，不进入下一个门槛；判据不成立时记录实测值，不要放宽阈值。每个结果都按主机、提交和日期记入 [实测记录](measurements.md)。本手册中的时长、吞吐和显存数字在实测之前都是 [估计]。

**当前状态（截至步骤 I6）：本手册中的门槛全部未运行。** 全部命令都已落地（I2–I6），但没有在 GPU 主机上运行过，所以下文的结果表都是“未运行”。

**命令块的检查。** 标为 `bash runbook` 的代码块由 `fork/bilingual/tests/cpu/test_runbook.py` 逐条检查（I6-E7）：

- 每条 `python -m surogate_bilingual …` 先加 `--dry-run` 运行，必须以 0 退出，且不写任何文件；
- 之后的命令要读前面命令的输出，所以测试接着以 tiny 规模真正运行它：Track S 用 ci 规模，模型用 `tiny` profile，训练只跑 2 步，评测只取前几道题；只有 CUDA 才能运行的命令（`bench`、`verify --device cuda`）只做 dry-run；
- 建环境（`venv`、`pip`）、`python -c` 与 `pytest` 的行只检查格式。

所以 CPU 测试通过说明命令的参数、文件路径和命令之间的衔接是对的；它不说明 GPU 上的任何行为。

运行输出都在 data/runs 下（I5 决定，见 [spec 的 I5 落地记录](spec.md#i5-落地记录)第 1 条）：根目录的 `.gitignore` 已忽略 data，所以运行输出不出现在 `git status` 中，不会挡住 `fork/dev sync`；`code_dirty` 只看 `fork/bilingual/src` 下的改动。每个运行的 `run_id` 包含代码提交，换了提交的运行不能恢复，要另起运行名；所以一个阶段开始之前先提交，运行期间不要在 GPU 主机上改动 `fork/bilingual/src`。评测输出的目录约定：

| 目录 | 内容 |
|---|---|
| data/runs/_assembly | 组装 manifest（不复制权重）与保真报告 |
| data/runs/_eval/g1p/rN/ARM | G1′ 的 dev 评测，每个臂一个目录：`predictions.jsonl`（`bilingual-pred/1`）与 `results.json`（`bilingual-result/1`） |
| data/runs/_eval/main/rN/ARM | G1–G3 与正式运行的评测，同上；NLL 与切换分解是同目录下的 JSON 文件 |
| data/runs/_decisions | G1′ 门槛、无效性规则等决定（`bilingual-stats/1`） |

## G0 环境与 GPU 不变量

约 1 小时 [估计]。

1. **建环境并跑 CPU 测试。** 先在 download.pytorch.org 上找到支持 sm_120（RTX 5090）的 CUDA 版 torch 索引，填进 `TORCH_INDEX`。依赖文件见 [requirements-gpu.txt](../requirements-gpu.txt)；本包以 editable 方式装入、不带依赖，免得解析器替换已装的 torch。

   ```bash runbook
   TORCH_INDEX=   # 支持 sm_120 的 CUDA 索引，例如 https://download.pytorch.org/whl/cu130；记入实测记录
   python3.12 -m venv fork/.venv-bilingual
   fork/.venv-bilingual/bin/pip install torch --index-url "$TORCH_INDEX"
   fork/.venv-bilingual/bin/pip install -r fork/bilingual/requirements-gpu.txt
   fork/.venv-bilingual/bin/pip install -e fork/bilingual --no-deps
   fork/.venv-bilingual/bin/pip freeze > fork/.venv-bilingual/requirements.lock
   fork/.venv-bilingual/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
   fork/.venv-bilingual/bin/python -m pytest -q -p no:cacheprovider fork/bilingual/tests/cpu
   ```

   判据：倒数第二条输出的计算能力为 `(12, 0)`；CPU 测试全部通过，并且没有跳过的用例：在 GPU 主机上 scipy 已装，CPU lane 上跳过的 `test_i3_c10_numpy_hungarian_equals_scipy`（numpy Hungarian 对 scipy 的对拍）在这里必须运行并通过；pilot 的 Re-Basin（n = 1728）也只能在这里用 scipy 求解。把 torch 的确切版本（含 `+cu` 后缀）、CUDA 版本、驱动版本（`nvidia-smi`）和 `requirements.lock` 的 sha256 记入实测记录。`fork/.venv-bilingual` 被 `fork/.gitignore` 忽略，所以其中的锁文件不会让工作区变 dirty。

2. **环境指纹。** 写出 Python、torch、CUDA/cuDNN、驱动（`nvidia-smi`）、GPU、CPU、线程数、库版本、git 提交与是否 dirty；读不到的字段为 null，不猜。每个训练运行开始和每次恢复时也各记一份。

   ```bash runbook
   fork/.venv-bilingual/bin/python -m surogate_bilingual env --out data/runs/_env/environment.json
   ```

   判据：`cuda`、`cudnn`、`driver`、`gpu` 都不为 null，`gpu` 的计算能力为 12.0。

3. **GPU 不变量。** GPU 测试在 `fork/bilingual/tests/gpu` 中，没有 CUDA 时整体跳过。`BILINGUAL_G0_RECORD` 指定 (g3) 与 (g6) 的记录写到哪里。

   ```bash runbook
   BILINGUAL_G0_RECORD=data/runs/_g0/g0.json fork/.venv-bilingual/bin/python -m pytest -q -rs -p no:cacheprovider fork/bilingual/tests/gpu
   ```

   判据（每一条是 GPU 测试中的一个用例，全部通过、没有跳过）：

   | 检查 | 判据 |
   |---|---|
   | (g1) 保真度 | 在探针批上，CUDA bf16 autocast 路径的强制路由输出对 fp32 独立模型满足手稿 §9.4 声明的容差（默认 max-abs ≤ 0.05、argmax 一致率 ≥ 99.5%，运行前锁定）；关闭 TF32 的 CUDA fp32 路径 ≤ 1e-5。测试用 tiny 训练出的 pack 与随机初始化的 pilot 几何各检查一次；训练好的 pilot pack 在 G3 的 `verify` 中再检查 |
   | (g2) NEW-01 | 每条 dispatch 路径（强制、one-hot 混合、双路、双路与 one-hot 混合；两个族、四种融合）都能在 CUDA autocast 下运行，强制路由与单专家模型逐位相等 |
   | (g3) SDPA 后端 | 在 `sdpa_kernel` 中依次强制 flash、efficient、cudnn、math，各带 `enable_gqa`；记录 sm_120 支持哪些后端；模型前向在只允许融合后端（不含 math）时能运行，即没有静默回退。没有任何融合后端接受 `enable_gqa` 时测试失败，默认改为 `model.gqa = "repeat"` |
   | (g4) 可重复性 | 相同种子的两次 20 步运行，loss 相对差 ≤ 1e-3；不假设逐位确定 |
   | (g5) 恢复 | CUDA 上 5 步、中断、恢复再 5 步，与不中断的 10 步在同一容差内一致 |
   | (g6) 记录 | 驱动、torch 构建、计算能力与 `PEAK_TFLOPS`（dense BF16、fp32 累加，按 NVIDIA 规格表填写；待核实的值为 209.5）写入记录文件，且都不为 null |
   | scipy | 在该环境中 numpy Hungarian 与 scipy 在 n = 64 的 20 个矩阵上一致（I3-C10 的 scipy 部分） |

| 项目 | 结果 | 主机 | 提交 | 日期 |
|---|---|---|---|---|
| 第 1 步：环境与 CPU 测试 | 未运行 | | | |
| 第 2 步：环境指纹 | 未运行 | | | |
| (g1)–(g6)、scipy 对拍 | 未运行 | | | |

## G0.5 吞吐与显存

约 1–2 小时 [估计]。对每个（族、recipe、seq、microbatch、checkpointing）组合，在 pilot 几何的随机初始化模型上先跑 `--warmup` 步、再计时 `--steps` 步；`seed` 训练全部参数，`expert` 冻结骨干、只训练一个专家。每个点一行 JSON，不放得下的点记为 oom，最后一行是选择。

```bash runbook
fork/.venv-bilingual/bin/python -m surogate_bilingual bench --family swiglu,slot --recipe seed,expert --seq 256,512,1024 \
  --microbatch 1,4,8,16,32 --checkpointing off,on --warmup 10 --steps 30 --out data/runs/_bench/g05.jsonl
```

- **记录：** tokens/s 与每步时间；`max_memory_allocated`、`max_memory_reserved` 与计时结束时 `nvidia-smi` 的 memory.used（缓存分配器保留着峰值时的预留，所以它是峰值时进程的占用加 CUDA context）；各 SDPA 后端是否可用；MFU = 6·N_matmul·tokens/s / PEAK（N_matmul 为每 token 的权重 MAC；冻结的权重不算权重梯度，所以 expert recipe 的 MFU 低估了 GPU 实际做的计算）。
- **决定：** 文件最后的 choice 行按（族、recipe）给出 seq 1024 下能凑满 `global_batch_tokens` = 65,536、不 OOM、tokens/s 最高的（microbatch, accum, checkpointing）。把选定的组合写进 [pilot_rows.toml](../examples/pilot_rows.toml) 的 `[train]` 节并提交（它们进入运行身份，所以不要用 `--set` 临时改），再用实测的 tokens/s 重算手稿 §13 的预算，重新锁定 token 预算。

[估计]，未实测：

| 设置 | 显存 | 吞吐 |
|---|---|---|
| 16×1024，不开 checkpointing | 约 21–23 GiB | 约 60–100k tok/s |
| 32×1024，开 checkpointing | 约 19–21 GiB | 约 45–75k tok/s |
| v1.0 默认（1×512，开 checkpointing） | — | 约 15–25k tok/s |

| 项目 | 结果 | 主机 | 提交 | 日期 |
|---|---|---|---|---|
| 吞吐与显存网格 | 未运行 | | | |
| 选定的（microbatch, accum, checkpointing） | 未运行 | | | |

## G1′ 可测性 pilot

`mech8` profile，3 个重复，约 3–11 小时 [估计]。先由 Track S 生成规格生成数据（`synth`），再按 pilot 行配置运行 `g1p-*` 各阶段（`train`）：每个重复（r0–r2）种子 0.3B、ZH 与 EN 各 0.1B、B2 0.2B、B12a 0.2B token。

**数据准备（CPU 即可）。** 先生成 Track S，再处理自然语料。自然语料的来源与许可要 owner 先选定并核实（B-O12），在此之前只能运行第一条命令。语料目录中的 spec 由 owner 编写：每个来源一个 `[[source]]`（字段见 [spec 的“语料 manifest 与划分”](spec.md#数据契约摘要)），其中 Track S 的 D_seed 成分作为 `kind = "synthetic"` 的来源列入（files 为 `track_s/seed/*.jsonl`），这样它进入 `seed_train`，tokenizer 也在它上面训练；D_zh、D_en、D_parallel 的 Track S 成分不进 manifest，写 shard 时用 `--add` 加入。自然语料来源的 glob 不要匹配到 `track_s/` 下的文件。

```bash runbook
PY=fork/.venv-bilingual/bin/python
$PY -m surogate_bilingual synth --spec fork/bilingual/examples/track_s.toml --seed 20261102 --out data/corpus/track_s
$PY -m surogate_bilingual manifest --spec data/corpus/spec.toml --root data/corpus --out data/manifest.json
$PY -m surogate_bilingual split assign --manifest data/manifest.json --seed 20261102 --out data/split.json
$PY -m surogate_bilingual decontam --manifest data/manifest.json --split data/split.json --out data/overlap_text.json
$PY -m surogate_bilingual split materialize --manifest data/manifest.json --split data/split.json --exclude data/overlap_text.json --out data/splits
$PY -m surogate_bilingual tokenizer train --manifest data/splits/manifest.json --out data/tokenizer/tokenizer.json
$PY -m surogate_bilingual tokenizer check data/tokenizer/tokenizer.json
$PY -m surogate_bilingual decontam --manifest data/splits/manifest.json --rules token13 --tokenizer data/tokenizer/tokenizer.json --out data/overlap_token.json
$PY -m surogate_bilingual shard --manifest data/splits/manifest.json --split seed_train --tokenizer data/tokenizer/tokenizer.json --route seed --seed-mix --exclude data/overlap_token.json --out data/shards/seed
$PY -m surogate_bilingual shard --manifest data/splits/manifest.json --split zh_train --tokenizer data/tokenizer/tokenizer.json --route zh --add data/corpus/track_s/zh/track_s.jsonl --exclude data/overlap_token.json --out data/shards/zh
$PY -m surogate_bilingual shard --manifest data/splits/manifest.json --split en_train --tokenizer data/tokenizer/tokenizer.json --route en --add data/corpus/track_s/en/track_s.jsonl --exclude data/overlap_token.json --out data/shards/en
$PY -m surogate_bilingual shard --manifest data/splits/manifest.json --split zh_train,en_train --tokenizer data/tokenizer/tokenizer.json --add data/corpus/track_s/zh/track_s.jsonl data/corpus/track_s/en/track_s.jsonl --exclude data/overlap_token.json --out data/shards/zh_en
```

判据：`synth` 以退出码 0 结束并写出 `datacard.json`（任何一项检查失败时它报错、不写数据卡）；数据卡的 `status` 全部为 true；`tokenizer check` 报告 0 个 `<unk>` 与 0 个往返失败；D_seed shard 的头中 `extra.seed_mix.max_share_error` ≤ 0.005，`mixed_window_fraction.value` 高于下限 0.01。zh_en shard 是 D_zh ∪ D_en，供 B2、B12a（和 T3″）用。Bridge 各臂的平行 shard 用 `shard --bridge` 写（源语言段走其语言的路由、loss 0，目标段走 bridge 路由、loss 1）：T2 用两个方向都有的 parallel，T1′ 用 `--direction zh>en` 与 `--direction en>zh` 的 parallel_zh_en、parallel_en_zh，T3′ 用 `--unpair-seed` 的 parallel_unpaired（路径见 pilot_rows.toml）；Bridge 的运行属于 P1，不在 G0–G3 之内。把 `datacard.json`、`data/split.json`、tokenizer 与各 shard 头的 sha256 记入实测记录；test 文件的 sha256 在任何 test 评测之前写入签名的预注册锁定提交（手稿 §9.7）。完整规模的 `synth` 在本机 C1 上约 5.5 分钟、1.1 GB [CPU 实测，见实测记录]；自然语料各步在完整规模上的耗时没有测过。

| 项目 | 结果 | 主机 | 提交 | 日期 |
|---|---|---|---|---|
| Track S 完整构建（种子 20261102）与数据卡 sha256 | 未运行（GPU 主机）；C1 上的一次构建见实测记录 | | | |
| 自然语料的来源与许可核实（B-O12） | 未开始 | | | |
| tokenizer、各 shard 与它们的 sha256 | 未运行 | | | |

**pilot 运行。** 行配置在 [pilot_rows.toml](../examples/pilot_rows.toml) 中，每个阶段一个运行，`-r0`…`-r2` 是重复。每个阶段先 `--dry-run`（读入并校验全部输入，不写文件），再运行；中断后用同一条命令恢复。r1、r2 把下面的 `r0` 换掉即可。

```bash runbook
PY=fork/.venv-bilingual/bin/python
$PY -m surogate_bilingual train fork/bilingual/examples/pilot_rows.toml --list
$PY -m surogate_bilingual train fork/bilingual/examples/pilot_rows.toml --stage g1p-seed-r0
$PY -m surogate_bilingual train fork/bilingual/examples/pilot_rows.toml --stage g1p-zh-r0
$PY -m surogate_bilingual train fork/bilingual/examples/pilot_rows.toml --stage g1p-en-r0
$PY -m surogate_bilingual train fork/bilingual/examples/pilot_rows.toml --stage g1p-b2-r0
$PY -m surogate_bilingual train fork/bilingual/examples/pilot_rows.toml --stage g1p-b12a-r0
$PY -m surogate_bilingual registry list
```

调参台账（手稿 §9.6）：每个训练臂另起两个 lr 试验，例如 `--stage g1p-seed-r0 --set train.lr=3e-4 --name g1p-seed-r0-lr3e-4`，只在 dev 上选。每个运行的 `run.json` 记录 config、`run_id`、P_update 与状态，`metrics.jsonl` 每步一行（tokens/s、step 时间、`max_memory_allocated` 与 `max_memory_reserved` 的峰值）。

**pilot 评测（dev）。** 组装（B2、B12a 是单槽位组装，B4、B6 是 ZH+EN 组装），在 dev 的 E3-ctx-cross 上评测 B0、B2、B12a、B4 与 B6，然后由三个重复的预测得出门槛与讨厌参数。评测默认 fp32（`--autocast none`），候选按手稿 §9.1 的规则排序（等长时求和，否则逐 token 平均）。

```bash runbook
PY=fork/.venv-bilingual/bin/python
ITEMS=data/corpus/track_s/items/E3ctx.dev.jsonl
$PY -m surogate_bilingual assemble --seed data/runs/g1p-seed-r0/packs/seed --slot bi=data/runs/g1p-b2-r0/packs/bi --out data/runs/_assembly/g1p-b2-r0.json
$PY -m surogate_bilingual assemble --seed data/runs/g1p-seed-r0/packs/seed --slot joint=data/runs/g1p-b12a-r0/packs/joint --out data/runs/_assembly/g1p-b12a-r0.json
$PY -m surogate_bilingual assemble --seed data/runs/g1p-seed-r0/packs/seed --slot zh=data/runs/g1p-zh-r0/packs/zh --slot en=data/runs/g1p-en-r0/packs/en --out data/runs/_assembly/g1p-zh-en-r0.json
$PY -m surogate_bilingual eval-tasks --model data/runs/g1p-seed-r0/packs/seed --arm B0 --replicate 0 --items $ITEMS --variant cross --out data/runs/_eval/g1p/r0/B0
$PY -m surogate_bilingual eval-tasks --model data/runs/_assembly/g1p-b2-r0.json --arm B2 --replicate 0 --items $ITEMS --variant cross --out data/runs/_eval/g1p/r0/B2
$PY -m surogate_bilingual eval-tasks --model data/runs/_assembly/g1p-b12a-r0.json --arm B12a --replicate 0 --items $ITEMS --variant cross --out data/runs/_eval/g1p/r0/B12a
$PY -m surogate_bilingual eval-tasks --model data/runs/_assembly/g1p-zh-en-r0.json --arm B4 --replicate 0 --items $ITEMS --variant cross --out data/runs/_eval/g1p/r0/B4
$PY -m surogate_bilingual eval-tasks --model data/runs/_assembly/g1p-zh-en-r0.json --arm B6 --replicate 0 --items $ITEMS --variant cross --out data/runs/_eval/g1p/r0/B6
$PY -m surogate_bilingual stats pilot --pred data/runs/_eval/g1p --out data/runs/_decisions/g1p.json
```

`stats pilot` 读入目录下所有重复的预测：`gate` 给出门槛，`nuisance` 给出各臂对（B12a−B4、B2−B4、B6−B4）每个重复的差、重复间 SD、不一致率与手稿 §9.5 规则给出的重复数 R（SESOI 取 [建议] 默认值 5pp）。

- **通过：** 在 **dev** 上，B0、B2、B12a 中最好的一个（按三个重复的平均）在 E3-ctx-cross 上的正确率 ≥ 随机水平（12.5%）+ 20 个百分点。
- **输出：** 讨厌参数（不一致率、题目方差、重复间 SD、发散情况）；按手稿 §9.5 的规则定出的重复数 R（`stats power --sd … --sesoi …` 可对其他 SESOI 重算）；在签名的预注册锁定提交中锁定 SESOI 与界值（手稿 §9.7）。pilot 只估计讨厌参数，不估计效应量。
- **失败：** 重新设计 Track S（例如增加曝光次数或种子 token），不开始确认性运行。

| 项目 | 结果 | 主机 | 提交 | 日期 |
|---|---|---|---|---|
| E3-ctx-cross dev 正确率（B0 / B2 / B12a，r0–r2） | 未运行 | | | |
| 重复数 R 与锁定的 SESOI | 未运行 | | | |

## G1 种子 S_I，重复 0

0.8B token，约 2.2–7.4 小时 [估计]。dev 文档是划分物化出的 data/splits/dev 下的文件；混合语言的文档（平行对、混语文本）不进 2×2 矩阵，结果中记为 `skipped_mixed_docs`。

```bash runbook
PY=fork/.venv-bilingual/bin/python
$PY -m surogate_bilingual train fork/bilingual/examples/pilot_rows.toml --stage seed-I-r0
$PY -m surogate_bilingual eval-nll --model data/runs/seed-I-r0/packs/seed --docs data/splits/dev/*.jsonl --unigram-shard data/shards/seed --out data/runs/_eval/main/r0/B0.nll.json
$PY -m surogate_bilingual eval-tasks --model data/runs/seed-I-r0/packs/seed --arm B0 --replicate 0 --items data/corpus/track_s/items/E3ctx.dev.jsonl --variant mono --out data/runs/_eval/main/r0/B0-mono
$PY -m surogate_bilingual eval-tasks --model data/runs/seed-I-r0/packs/seed --arm B0 --replicate 0 --prompts fork/bilingual/examples/g1_prompts.jsonl --out data/runs/_eval/main/r0/B0-gen
```

- **通过：** 两种语言的 dev BPB 都低于 unigram 基线（`B0.nll.json` 中 `matrix.bpb` 对 `unigram.bpb`；unigram 基线用种子 shard 的加一平滑频率 [建议]）；E3-ctx-mono 的 dev 正确率 ≥ 随机水平 + 20 个百分点；目标语言的短自然语言生成不是完全随机的（固定生成集 [g1_prompts.jsonl](../examples/g1_prompts.jsonl) 是 [建议] 的 8 个提示，owner 可以换；输出记录在 `B0-gen/predictions.jsonl`，`lang_ok` 是按文字系统的粗检查，判断要人工看）。
- **记录：** 学习曲线（`metrics.jsonl`）与固定生成集的输出。

| 项目 | 结果 | 主机 | 提交 | 日期 |
|---|---|---|---|---|
| dev BPB（zh / en）与 unigram 基线 | 未运行 | | | |
| E3-ctx-mono dev 正确率 | 未运行 | | | |
| 固定生成集 | 未运行 | | | |

## G2 专家 ZH-I 与 EN-I，重复 0

各 0.2B token，约 1.1–3.7 小时 [估计]。

```bash runbook
PY=fork/.venv-bilingual/bin/python
$PY -m surogate_bilingual train fork/bilingual/examples/pilot_rows.toml --stage zh-I-r0
$PY -m surogate_bilingual train fork/bilingual/examples/pilot_rows.toml --stage en-I-r0
$PY -m surogate_bilingual assemble --seed data/runs/seed-I-r0/packs/seed --slot zh=data/runs/zh-I-r0/packs/zh --slot en=data/runs/en-I-r0/packs/en --out data/runs/_assembly/zh-en-I-r0.json
$PY -m surogate_bilingual eval-nll --model data/runs/_assembly/zh-en-I-r0.json --docs data/splits/dev/*.jsonl --out data/runs/_eval/main/r0/zh-en.nll.json
$PY -m surogate_bilingual eval-tasks --model data/runs/seed-I-r0/packs/seed --arm B0 --replicate 0 --items data/corpus/track_s/items/E1.dev.jsonl --out data/runs/_eval/main/r0/B0-E1
$PY -m surogate_bilingual eval-tasks --model data/runs/_assembly/zh-en-I-r0.json --arm B1 --replicate 0 --items data/corpus/track_s/items/E1.dev.jsonl --out data/runs/_eval/main/r0/B1-E1
```

`B1-E1` 在 ZH+EN 组装上按 R_out 路由：E1 的题目只有一种语言，所以每道题整题走它语言的专家，与该专家单独运行完全相同（H1）。

- **通过：** 骨干哈希不变（写 pack 时与种子 pack 比较，不同则拒绝写出）；E1 的 dev 保持（`B1-E1` 对 `B0-E1`）；2×2 BPB 矩阵（`zh-en.nll.json` 的行是强制路由的专家，列是评测语言）中每个专家在自己语言上的 BPB ≤ B0（`B0.nll.json`）；每个 pack 都记录了探针。

| 项目 | 结果 | 主机 | 提交 | 日期 |
|---|---|---|---|---|
| 骨干哈希、E1 保持、2×2 BPB 矩阵 | 未运行 | | | |

## G3 组装与 dev 评测，重复 0

不超过 1 小时 [估计]，B2 与 B12a 的训练另计。

- **组装与验证：** 组装 ZH+EN（G2 已写出 manifest）并运行 `verify`；通过条件为 CUDA 上的 G0 容差。
- **dev 评测：** 在 dev 的 E3-ctx-cross 上评测 B0、B4、B4-wrong、B4-rand、B4-src 与 B6；B2 与 B12a 训练完成后加入。B4-src 只对有单一知识来源的参数式题目有定义，在 E3-ctx 上每道题都记为跳过（带原因），所以它在 E2 上另评。在 E4 的 dev 子集上做切换代价分解（M、R_new、R_old，以及切换后前 16 个目标 token 的 NLL 峰值）。

```bash runbook
PY=fork/.venv-bilingual/bin/python
ITEMS=data/corpus/track_s/items/E3ctx.dev.jsonl
OUT=data/runs/_eval/main/r0
ZHEN=data/runs/_assembly/zh-en-I-r0.json
$PY -m surogate_bilingual verify $ZHEN --device cuda --autocast bf16 --out data/runs/_assembly/zh-en-I-r0.fidelity.json
$PY -m surogate_bilingual eval-tasks --model data/runs/seed-I-r0/packs/seed --arm B0 --replicate 0 --items $ITEMS --variant cross --out $OUT/B0
$PY -m surogate_bilingual eval-tasks --model $ZHEN --arm B4 --replicate 0 --items $ITEMS --variant cross --out $OUT/B4
$PY -m surogate_bilingual eval-tasks --model $ZHEN --arm B4-wrong --replicate 0 --items $ITEMS --variant cross --out $OUT/B4-wrong
$PY -m surogate_bilingual eval-tasks --model $ZHEN --arm B4-rand --replicate 0 --items $ITEMS --variant cross --out $OUT/B4-rand
$PY -m surogate_bilingual eval-tasks --model $ZHEN --arm B6 --replicate 0 --items $ITEMS --variant cross --out $OUT/B6
$PY -m surogate_bilingual eval-tasks --model $ZHEN --arm B4-src --replicate 0 --items data/corpus/track_s/items/E2.dev.jsonl --out $OUT/B4-src-E2
$PY -m surogate_bilingual eval-tasks --model $ZHEN --arm B4 --replicate 0 --items data/corpus/track_s/items/E2.dev.jsonl --out $OUT/B4-E2
$PY -m surogate_bilingual switch --model $ZHEN --items data/corpus/track_s/items/E4.dev.jsonl --out $OUT/switch.json
$PY -m surogate_bilingual train fork/bilingual/examples/pilot_rows.toml --stage b2-I-r0
$PY -m surogate_bilingual train fork/bilingual/examples/pilot_rows.toml --stage b12a-I-r0
$PY -m surogate_bilingual assemble --seed data/runs/seed-I-r0/packs/seed --slot bi=data/runs/b2-I-r0/packs/bi --out data/runs/_assembly/b2-I-r0.json
$PY -m surogate_bilingual assemble --seed data/runs/seed-I-r0/packs/seed --slot joint=data/runs/b12a-I-r0/packs/joint --out data/runs/_assembly/b12a-I-r0.json
$PY -m surogate_bilingual eval-tasks --model data/runs/_assembly/b2-I-r0.json --arm B2 --replicate 0 --items $ITEMS --variant cross --out $OUT/B2
$PY -m surogate_bilingual eval-tasks --model data/runs/_assembly/b12a-I-r0.json --arm B12a --replicate 0 --items $ITEMS --variant cross --out $OUT/B12a
$PY -m surogate_bilingual stats futility --pred $OUT --replicate 0 --out data/runs/_decisions/futility_r0.json
$PY -m surogate_bilingual matrix --results data/runs/_eval/main
```

- **无效性规则（手稿 §9.6，B-O18）：** `futility_r0.json` 记录 (B12a − B4) 与 (B6 − B4) 在 dev 上的值和决定：两者都小于 SESOI_A / 2 时分支 1 判为“无效性停止（不检验）”，不训练族 II，分支 2 照常进行。
- **test 集：** 在手稿 §9.7 的预注册锁定之前不碰 test 集：`eval-tasks` 与 `switch` 默认只评测 dev，遇到 test 题目时拒绝，除非显式加 `--allow-test`。锁定之后的确认性检验是 `stats confirm`（重复级单侧配对 t 检验与手稿 §9.3 的图形化守门，H_B 的不一致率前提用 dev 预测检查）。
- **对照矩阵：** 最后一条打印手稿 §7 的全部行；有结果的行列出记录到的指标，其余为“未运行”。

| 项目 | 结果 | 主机 | 提交 | 日期 |
|---|---|---|---|---|
| 组装保真度（CUDA） | 未运行 | | | |
| dev E3-ctx-cross 各臂 | 未运行 | | | |
| 切换代价分解（E4 dev） | 未运行 | | | |
| 无效性决定 | 未运行 | | | |

## GPU 预算 [估计]

来自手稿 §13，按 100k / 60k / 30k tok/s 推算，每 1B token 约 2.78 / 4.63 / 9.26 小时；G0.5 实测之后重算。

| 范围 | token | GPU 小时 | GPU-day |
|---|---:|---:|---:|
| 最小决定性集合（G1′ pilot、3 个重复 × 4.0B、评测） | 14.2B | 39.4 / 65.7 / 131.5 | 1.6 / 2.7 / 5.5 |
| G0 与 G0.5 | — | — | 约 0.2 |
