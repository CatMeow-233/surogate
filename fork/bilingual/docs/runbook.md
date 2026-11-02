# 运行手册（RTX 5090，G0–G3）

命令都在 GPU 主机的仓库根目录、环境 `fork/.venv-bilingual` 中运行。门槛按顺序进行：某一步以非零退出码结束、或判据不成立，就停止，保存输出，不进入下一个门槛；判据不成立时记录实测值，不要放宽阈值。每个结果都按主机、提交和日期记入 [实测记录](measurements.md)。本手册中的时长、吞吐和显存数字在实测之前都是 [估计]。

**当前状态（截至步骤 I4）：本手册中的门槛全部未运行。** 现在能运行的是 G0 的第 1 步（建环境、跑 CPU 测试）和 G1′ 之前的数据准备（`synth` 与数据管线的各命令，I4）。`assemble`、`verify`、`audit` 三个命令已在 I2 落地，但前两个需要的种子与专家 pack 要等 I5 的训练才有；其余命令（`env`、`bench`、`train` 与各评测命令）在步骤 I5–I6 落地，在此之前 `python -m surogate_bilingual 命令` 以退出码 2 结束并说明命令在哪个步骤实现。步骤 I6 补全本手册，届时可运行的命令块会由测试以 `--dry-run` 逐条检查。

运行输出的位置还没有定：英文规格把运行目录放在仓库根目录的 runs 下，而根目录的 `.gitignore` 只忽略 data 目录、不忽略 runs，未跟踪的运行输出会让工作区变 dirty（`code_dirty`）。I5 落地运行登记时一并决定（例如放在 `fork/bilingual` 下并由 `fork/.gitignore` 忽略，或经环境变量放到仓库之外）。

## G0 环境与 GPU 不变量

约 1 小时 [估计]。

1. **建环境并跑 CPU 测试（现在可运行）。** 先在 download.pytorch.org 上找到支持 sm_120（RTX 5090）的 CUDA 版 torch 索引，填进 `TORCH_INDEX`。依赖文件见 [requirements-gpu.txt](../requirements-gpu.txt)；本包以 editable 方式装入、不带依赖，免得解析器替换已装的 torch。

   ```bash
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

2. **环境指纹（I5 落地后）。** 计划中的命令为 `fork/.venv-bilingual/bin/python -m surogate_bilingual env --out <运行目录>/_env/environment.json`，写出 Python、torch、CUDA/cuDNN、驱动、GPU、CPU、线程数、git 提交与是否 dirty。

3. **GPU 不变量（I6 加入 GPU 测试后）。** 判据：

   | 检查 | 判据 |
   |---|---|
   | (g1) 保真度 | 在探针批上，CUDA bf16 autocast 路径的强制路由输出对 fp32 独立模型满足手稿 §9.4 声明的容差（默认 max-abs ≤ 0.05、argmax 一致率 ≥ 99.5%，运行前锁定）；关闭 TF32 的 CUDA fp32 路径 ≤ 1e-5 |
   | (g2) NEW-01 | 每条 dispatch 路径都能在 CUDA autocast 下运行 |
   | (g3) SDPA 后端 | 在 `sdpa_kernel` 中依次强制 flash、efficient、cudnn、math，各带 `enable_gqa`；记录 sm_120 支持哪些后端，并断言选定的默认后端运行时没有静默回退，否则默认改为 `model.gqa = "repeat"` |
   | (g4) 可重复性 | 相同种子的两次 20 步运行，loss 相对差 ≤ 1e-3；不假设逐位确定 |
   | (g5) 恢复 | CUDA 上的恢复在同一容差内往返 |
   | (g6) 记录 | 驱动、torch 构建与 `PEAK_TFLOPS`（dense BF16、fp32 累加，按 NVIDIA 规格表填写；待核实的值为 209.5） |

| 项目 | 结果 | 主机 | 提交 | 日期 |
|---|---|---|---|---|
| 第 1 步：环境与 CPU 测试 | 未运行 | | | |
| 第 2 步：环境指纹 | 未运行（I5） | | | |
| (g1)–(g6) | 未运行（I6） | | | |

## G0.5 吞吐与显存

约 1–2 小时 [估计]。命令在 I6 落地，计划的形式为 `python -m surogate_bilingual bench --family swiglu,slot --seq 256,512,1024 --microbatch 1,4,8,16,32 --checkpointing off,on --warmup 10 --steps 30`，结果写成 JSON Lines。

- **记录：** tokens/s 与每步时间；`max_memory_allocated`、`max_memory_reserved` 与 `nvidia-smi` 的峰值；使用的 SDPA 后端；MFU = 6·N_matmul·tokens/s / PEAK。
- **决定：** 选出能凑满 `global_batch_tokens` = 65,536、不 OOM、tokens/s 最高的（microbatch, accum, checkpointing）组合，再用实测的 tokens/s 重算手稿 §13 的预算，重新锁定 token 预算。

[估计]，未实测：

| 设置 | 显存 | 吞吐 |
|---|---|---|
| 16×1024，不开 checkpointing | 约 21–23 GiB | 约 60–100k tok/s |
| 32×1024，开 checkpointing | 约 19–21 GiB | 约 45–75k tok/s |
| v1.0 默认（1×512，开 checkpointing） | — | 约 15–25k tok/s |

| 项目 | 结果 | 主机 | 提交 | 日期 |
|---|---|---|---|---|
| 吞吐与显存网格 | 未运行（I6） | | | |
| 选定的（microbatch, accum, checkpointing） | 未运行 | | | |

## G1′ 可测性 pilot

`mech8` profile，3 个重复，约 3–11 小时 [估计]。先由 Track S 生成规格生成数据（`synth`，I4），再按 pilot 行配置运行 `g1p-*` 各阶段（`train`，I5）：每个重复（r0–r2）种子 0.3B、ZH 与 EN 各 0.1B、B2 0.2B、B12a 0.2B token。

**数据准备（I4，CPU 即可，命令已落地）。** 先生成 Track S，再处理自然语料。自然语料的来源与许可要 owner 先选定并核实（B-O12），在此之前只能运行第一条命令。语料目录中的 spec 由 owner 编写：每个来源一个 `[[source]]`（字段见 [spec 的“语料 manifest 与划分”](spec.md#数据契约摘要)），其中 Track S 的 D_seed 成分作为 `kind = "synthetic"` 的来源列入（files 为 `track_s/seed/*.jsonl`），这样它进入 `seed_train`，tokenizer 也在它上面训练；D_zh、D_en、D_parallel 的 Track S 成分不进 manifest，写 shard 时用 `--add` 加入。自然语料来源的 glob 不要匹配到 `track_s/` 下的文件。

```bash
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
```

判据：`synth` 以退出码 0 结束并写出 `datacard.json`（任何一项检查失败时它报错、不写数据卡）；数据卡的 `status` 全部为 true；`tokenizer check` 报告 0 个 `<unk>` 与 0 个往返失败；D_seed shard 的头中 `extra.seed_mix.max_share_error` ≤ 0.005，`mixed_window_fraction.value` 高于下限 0.01。D_parallel shard（Bridge 用）的路由与损失掩码由 I5 的 Bridge recipe 决定，在 I5 补上。把 `datacard.json`、`data/split.json`、tokenizer 与各 shard 头的 sha256 记入实测记录；test 文件的 sha256 在任何 test 评测之前写入签名的预注册锁定提交（手稿 §9.7）。完整规模的 `synth` 在本机 C1 上约 5.5 分钟、1.1 GB [CPU 实测，见实测记录]；自然语料各步在完整规模上的耗时没有测过。

| 项目 | 结果 | 主机 | 提交 | 日期 |
|---|---|---|---|---|
| Track S 完整构建（种子 20261102）与数据卡 sha256 | 未运行（GPU 主机）；C1 上的一次构建见实测记录 | | | |
| 自然语料的来源与许可核实（B-O12） | 未开始 | | | |
| tokenizer、各 shard 与它们的 sha256 | 未运行 | | | |

**pilot 运行（I5 落地后）：**

- **通过：** 在 **dev** 上，B0、B2、B12a 中最好的一个在 E3-ctx-cross 上的正确率 ≥ 随机水平（12.5%）+ 20 个百分点。
- **输出：** 讨厌参数（不一致率、题目方差、重复间 SD、发散情况）；按手稿 §9.5 的规则定出的重复数 R；在签名的预注册锁定提交中锁定 SESOI 与界值（手稿 §9.7）。
- **失败：** 重新设计 Track S（例如增加曝光次数或种子 token），不开始确认性运行。

| 项目 | 结果 | 主机 | 提交 | 日期 |
|---|---|---|---|---|
| E3-ctx-cross dev 正确率（B0 / B2 / B12a，r0–r2） | 未运行 | | | |
| 重复数 R 与锁定的 SESOI | 未运行 | | | |

## G1 种子 S_I，重复 0

0.8B token，约 2.2–7.4 小时 [估计]。

- **通过：** 两种语言的 dev BPB 都低于 unigram 基线；E3-ctx-mono 的 dev 正确率 ≥ 随机水平 + 20 个百分点；目标语言的短自然语言生成不是完全随机的。
- **记录：** 学习曲线与一组固定的生成样例。

| 项目 | 结果 | 主机 | 提交 | 日期 |
|---|---|---|---|---|
| dev BPB（zh / en）与 unigram 基线 | 未运行 | | | |
| E3-ctx-mono dev 正确率 | 未运行 | | | |

## G2 专家 ZH-I 与 EN-I，重复 0

各 0.2B token，约 1.1–3.7 小时 [估计]。

- **通过：** 骨干哈希不变；E1 的 dev 保持；2×2 BPB 矩阵中每个专家在自己语言上的 BPB ≤ B0；每个 pack 都记录了探针。

| 项目 | 结果 | 主机 | 提交 | 日期 |
|---|---|---|---|---|
| 骨干哈希、E1 保持、2×2 BPB 矩阵 | 未运行 | | | |

## G3 组装与 dev 评测，重复 0

不超过 1 小时 [估计]。

- **组装与验证：** 组装 ZH+EN 并运行 `verify`；通过条件为 CUDA 上的 G0 容差。
- **dev 评测：** 在 dev 的 E3-ctx-cross 上评测 B0、B4、B4-wrong、B4-random、B4-src 与 B6；B2 与 B12a 训练完成后加入。在 E4 的 dev 子集上做切换代价分解。
- **无效性规则（手稿 §9.6）：** 决定记录在运行目录的 `_decisions/futility_r0.json` 中。
- **test 集：** 在手稿 §9.7 的预注册锁定之前不碰 test 集。

| 项目 | 结果 | 主机 | 提交 | 日期 |
|---|---|---|---|---|
| 组装保真度（CUDA） | 未运行 | | | |
| dev E3-ctx-cross 各臂 | 未运行 | | | |
| 无效性决定 | 未运行 | | | |

## GPU 预算 [估计]

来自手稿 §13，按 100k / 60k / 30k tok/s 推算，每 1B token 约 2.78 / 4.63 / 9.26 小时；G0.5 实测之后重算。

| 范围 | token | GPU 小时 | GPU-day |
|---|---:|---:|---:|
| 最小决定性集合（G1′ pilot、3 个重复 × 4.0B、评测） | 14.2B | 39.4 / 65.7 / 131.5 | 1.6 / 2.7 / 5.5 |
| G0 与 G0.5 | — | — | 约 0.2 |
