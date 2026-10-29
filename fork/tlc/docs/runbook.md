# 运行手册（RTX 5090，G0–G3）与 GPU 预算

命令都在 GPU 主机的仓库根目录运行。门槛按顺序进行：某一步的命令以非零退出码结束、或判据不成立，就停止，把输出和运行记录中的 `failure_report` 存进 `$TLC_RUNS/env/`，不进入下一个门槛；A0 没有通过 G1 或 G2 时不进入后续阶段。标 [P] 的阈值是暂定的：不成立时记录实测值并停下，不要放宽阈值。GPU 测试单独运行，不与 `fork/tlc/tests/cpu` 在同一次 pytest 中收集：CPU 测试会把仓库根目录放到 `sys.path` 最前面，GPU 测试则用 `fork/.venv-tlc` 的导入路径。

```bash
set -o pipefail                  # `pytest ... | tee` 保留 pytest 的退出码
export TLC_RUNS=/data/tlc        # 所有产物的根目录，放在仓库之外
PY=fork/.venv-tlc/bin/python
mkdir -p "$TLC_RUNS/env" "$TLC_RUNS/corpus" "$TLC_RUNS/goldens" "$TLC_RUNS/benchmarks"
```

语料 spec `$TLC_RUNS/corpus/corpus.toml` 由 owner 按 [data.md 的“语料”](data.md#语料)准备（来源表、70/30 配比、合成数据占比）。去污染用的基准文件放在 `$TLC_RUNS/benchmarks`：`HumanEval.jsonl.gz`、`mbpp.jsonl`（全部 974 条）、`HumanEvalPlus.jsonl.gz`、`MbppPlus.jsonl.gz`，见 [data.md 的“去污染”](data.md#去污染)。

## G0 环境

1. 原生扩展。根目录 `.venv` 按 [fork/README.md 的“首次准备环境”](../../README.md#首次准备环境) 创建（只需一次；首次完整构建需要数小时），然后用本检出重建训练扩展。CMake 按本机 GPU 自动选择 `120a`；之后每次同步上游或修改训练侧 C++，都重新运行 `fork/dev build train`。

   ```bash
   uv venv --python 3.12
   uv pip install -e .
   fork/dev build train
   cuobjdump --list-elf surogate/libsurogate-common.so | grep -o 'sm_[0-9]*a*' | sort -u
   ```

   预期最后一条只输出 `sm_120a`。

2. `fork/.venv-tlc`。依赖与根目录 `.venv` 相同，并锁定成文件；`surogate` 经 `.pth` 文件从本检出导入，所以 `fork/dev build train` 复制进 `surogate/` 的扩展就是 `.venv-tlc` 加载的扩展，不需要第二次原生构建，也不会留下过期的副本。

   ```bash
   uv venv --python 3.12 fork/.venv-tlc
   uv pip freeze -p .venv --exclude-editable > "$TLC_RUNS/env/requirements.lock"
   uv pip install -p fork/.venv-tlc -r "$TLC_RUNS/env/requirements.lock" pytest
   uv pip install -p fork/.venv-tlc --no-deps -e fork/tlc
   echo "$PWD" > fork/.venv-tlc/lib/python3.12/site-packages/surogate-checkout.pth
   ```

   根目录 `.venv` 的 torch 若来自 download.pytorch.org（锁文件中的版本带 `+cu…`），第三条命令要加上对应的 `--index-url`。

3. 环境指纹与 GPU。

   ```bash
   $PY -c 'import json; from surogate_tlc.train import report; print(json.dumps(report.environment(), indent=2))' > "$TLC_RUNS/env/environment.json"
   $PY -m pytest -c fork/tlc/pyproject.toml fork/tlc/tests/gpu/test_gpu_environment.py
   ```

   预期 2 个测试通过：`surogate._surogate` 与 `surogate_tlc` 都来自本检出，GPU 的计算能力为 (12, 0)，指纹中提交、原生扩展的 sha256、驱动版本齐全。每个运行记录都带同样的指纹。

4. 合成数据、去污染与 T3 套件。依次运行，每条之后按说明改 spec：

   ```bash
   $PY -m surogate_tlc.data.synth "$TLC_RUNS/suites/synth" --seed 0 --train-chars "$SYNTH_CHARS"
   $PY -m surogate_tlc.data.decontam "$TLC_RUNS/corpus/corpus.toml" "$TLC_RUNS/benchmarks" "$TLC_RUNS/corpus/decontam.json" --plant 40 --jobs 8
   $PY -m surogate_tlc.data.natural "$TLC_RUNS/corpus/corpus.toml" "$TLC_RUNS/suites/natural"
   ```

   - `synth`：`SYNTH_CHARS` 约为其余来源的字符数 × 0.05 / 0.95（暂定的 5%）。然后在 spec 中加一个来源：`name = "synth-binding"`、`kind = "code"`、`generated = true`、`files = ["../suites/synth/train.jsonl"]`，`url` 写生成命令与 manifest 的 sha256，`license` 写“generated”。预期 manifest 的 `disjoint.eval_items_with_a_training_ast` 为 0。
   - `decontam`：然后在 spec 顶层加 `decontam = "decontam.json"`。把输出中的 `calibration`（召回率与误报率）连同 `audit_sample` 中各文档的人工审计结论记进 `$TLC_RUNS/env/decontam.txt`，作为 G6 的去污染记录。
   - `natural`：T3 的 dev 与 test，取自去污染之后语料 val 划分中的自然代码；`generated = true` 的合成来源不出探针项，manifest 的 `corpus.sources` 列出实际取用的来源。

5. tokenizer、模型目录与 shard。

   ```bash
   $PY -m surogate_tlc.data.tokenizer train "$TLC_RUNS/corpus/corpus.toml" "$TLC_RUNS/tokenizer" --pretokenizer qwen2
   for row in A0 A0-split A1 A1h; do
     $PY -m surogate_tlc init "$row" "$TLC_RUNS/models/$row" --tokenizer "$TLC_RUNS/tokenizer" --seed 1
   done
   $PY -m surogate_tlc.data.tokenizer check "$TLC_RUNS/models/A0" "$TLC_RUNS/corpus/corpus.toml" --limit 10000
   $PY -m pytest -c fork/tlc/pyproject.toml fork/tlc/tests/gpu/test_data_native.py
   $PY -m surogate_tlc.data.shards "$TLC_RUNS/corpus/corpus.toml" "$TLC_RUNS/tokenizer" "$TLC_RUNS/data" --seq-len 512 | tee "$TLC_RUNS/env/shards.json"
   ```

   - 预分词和词表还没有选定（[data.md 的“Tokenizer”](data.md#tokenizer)）。owner 在 A0 之前定下，四行共用同一个 tokenizer 和同一个 `--seed`；A0 与 A0-split 用同一个种子，得到同一份权重（G1(d)）。
   - `check`：预期输出 `consistent on N texts`，退出码 0；有差异时停止，不建 shard。`test_data_native.py`：预期 5 个测试通过（三个预分词候选的原生一致性、原生 DataLoader 读出的行，以及从保存的位置恢复的 DataLoader 逐位接着读下去）。
   - `data.shards` 输出各划分的 `pad_fraction`，以及单核构建速率（`real_tokens_per_cpu_second`，G0.5 的数据流水线速率）；`pad_fraction` 超过约 25% 时记下来，pilot 照常进行。

6. kernel 与图。

   ```bash
   $PY -m pytest -c fork/tlc/pyproject.toml fork/tlc/tests/gpu/test_gates.py -k g0
   $PY -m pytest -c fork/tlc/pyproject.toml fork/tlc/tests/gpu/test_w_twice.py -s
   ```

   - `-k g0`：预期 3 个测试通过（A0、A1、A1h）：在 2 × 512 token 上做一次前向与反向，每个可训练参数的梯度都存在、有限且不全为零。它在 TLC 的图中跑过 BF16 GEMM、selective scan 的前向与反向（A0、A1h）和 head_dim 64 的 flash attention（A1、A1h），也完成两项附带检查：没有 `BlockSchema` 的 `TLCMambaFFNBlock` 被 buffer plan 接受，A1h 中带 MLP 的 `A` 层能运行。A0-split 的模型级第 1 层由 G1(d) 检查。执行沙箱的检查属于 P5-a，不在 P0 的 G0 中。
   - `test_w_twice.py`：一个权重在同一层中被两次使用的 toy 图（`fork/tlc/tests/gpu/twice_probe.py`），为后续的共享权重深度循环摸底。它在子进程中做一次前向与反向，把结果（`ok`、`nan`、`garbage` 即梯度与 PyTorch 的余弦低于 0.99、`error` 或 `crash`）写进 `$TLC_RUNS/env/w-twice.json`；任何结果测试都通过，P0 预期它失败，记录本身就是交付物。

## G0.5 吞吐

```bash
( set -e  # a subshell: the first failing command stops the loop, not the login shell
for row in A0 A1; do
  for microbatch in 2 8 32; do
    for stage in throughput throughput-eager; do
      TLC_ROW=$row $PY -m surogate_tlc train fork/tlc/examples/p0.yaml --stage "$stage" --microbatch "$microbatch"
    done
  done
done
$PY -m surogate_tlc.train.report "$TLC_RUNS/train" --peak-tflops "$PEAK_TFLOPS" | grep throughput | tee "$TLC_RUNS/env/throughput.jsonl"
)
```

- `PEAK_TFLOPS` 是 GPU 规格表中 dense bf16（fp32 累加）的峰值。每次更新 16,384 token 不变，microbatch 2、8、32（GA 相应为 16、4、1），各跑 30 步、开和关 CUDA graph。报告每行给出 tokens/s、MFU、峰值显存和是否用 CUDA graph；tokens/s 只在相同 microbatch 之间比较，每行注明所用档位。
- 数据流水线的单核速率取自第 5 步 `data.shards` 的输出（`real_tokens_per_cpu_second`，包括 tokenizer 编码）。
- 还没有做的：kernel 计数（要用 nsys 另行剖析）、上下文 2048 的那一半（等 owner 定正式训练的上下文，并放开 `surogate_tlc.hf.CONTEXT_LENGTH`）。

## G1 因果性

```bash
$PY -I -B fork/tlc/prototypes/ref_mamba2.py "$TLC_RUNS/models/A0" "$TLC_RUNS/goldens/A0.npz" --batch 2 --seq-len 512 --seed 0
$PY -m pytest -c fork/tlc/pyproject.toml fork/tlc/tests/gpu/test_gates.py -k g1 -s | tee "$TLC_RUNS/env/g1.txt"
```

第一条在 CPU 上生成 G1(b) 的 goldens（见 [model.md 的“数值参照”](model.md#数值参照)）。参照实现本身的因果性与路径一致性（G1.1–G1.8）由 CPU 测试检查，不在 GPU 上重复。`-s` 让测试打印实测的误差（JSON 行）。

| 选择 | 行 | 判据 |
|---|---|---|
| `-k g1n1` | A0、A1、A1h | 把位置 p 起的 token 全部改掉，位置 p 之前的 logprobs 逐位不变，p ∈ {1, 2, 7, 127, 128, 129, 511}（chunk 128 的两侧）；p 之后的 logprobs 确有变化 |
| `-k g1n3` | A0、A1、A1h | 只保留位置 129 之前的目标，改动之后的 token，参数梯度的变化不超过同一输入重复一次的差异的 2 倍（原生反向用浮点 atomicAdd，重复运行未必逐位相同） |
| `-k g1b` | A0 | goldens 记录的权重 sha256 与模型目录一致；原生 logprobs 与参照的最大绝对差不超过 goldens 中的 `tolerance` |
| `-k g1c` | A1；A0 为负对照 | 两个文档打包在一行（第二个文档的 position 从 0 重新开始），与各自单独打分比较：A1 最大绝对差 ≤ 2e-2 [P]；A0 的第一个文档 ≤ 2e-2，第二个文档的差异 > 1e-3，因为 `mamba_ssm_scan` 不在文档之间复位，这正是 Mamba 行每行只放一个文档的原因 |
| `-k g1d` | A0、A0-split | 两者的 `init_seed` 相同；logprobs 最大绝对差不超过 goldens 的 `tolerance`；梯度按 `split_param_name()` 改名后一一对应，最小余弦 ≥ 0.999 [P] |

G1(d) 不成立时，模型级第 1 层的方案不能承载 P2/P3 的 c_t 与槽写入，改走 fork IR 后处理 pass。打包行的文档隔离靠每行一个文档保证（shard 的构建规则，CPU 测试）。

## G2 数学与梯度

```bash
$PY -m pytest -c fork/tlc/pyproject.toml fork/tlc/tests/gpu/test_gates.py -k g2 -s | tee "$TLC_RUNS/env/g2.txt"
```

| 测试 | 判据 |
|---|---|
| `test_g2_initial_loss`（A0、A1、A1h） | 初始权重下随机 token 的交叉熵在 ln V + var(logits)/2 ± 0.1 之内，logits 的标准差 < 1。var(logits) 由权重估计：初始化时 final norm 的输出每个 token 均方为 1，所以它是宽度 × LM head（tied 时即 embedding）的均方；原生只返回目标的 log-probability，拿不到完整的 logits |
| `test_g2_gradients_point_where_the_reference_does` | A0 第 1 层与第 12 层的每个参数：原生 bf16 梯度与参照实现（分块形式，fp32 autograd，2 × 128 token）的梯度余弦 ≥ 0.99 [P] |
| `test_g2_overfits_a_small_alphabet_batch`（A0、A1） | 一个只含 64 个 token id 的固定 batch（2 × 512，其 H(next \| current) ≥ 1.5），200 步 AdamW（学习率 1e-3）后 loss < 0.05 [P]。只看当前 token 的模型到不了这个下限，所以拟合它说明序列混合有效；消融 mixer 的对照在 CPU 参照上做（`prototypes/g2_suite.py`），原生图无法消融 |
| `test_g2_resume_reproduces_the_next_steps_and_the_logits` | A0 的 smoke 阶段跑两次：一次不中断，一次从第 5 步的 checkpoint 恢复；第 6–10 步的 loss 相差 ≤ 1e-3（原生反向用浮点 atomicAdd，不要求逐位相等），两次的最终权重在 goldens 上的 logprobs 相差不超过 G1(b) 的 `tolerance` [P]。再把第 5 步的 checkpoint 导出到一个用另一个 seed 初始化的源目录：导出目录的 logprobs 与 checkpoint 逐位相同，tied embedding 只存一份（没有 `lm_head.weight`） |

梯度全部有限由 G0 的第 6 步检查；固定 batch 在真实数据上的过拟合是 G3 的 overfit 阶段。

## G3 基线

```bash
( set -e  # a subshell: the first failing command stops the loop, not the login shell
for row in A0 A1 A1h; do
  for stage in smoke overfit pilot; do
    TLC_ROW=$row $PY -m surogate_tlc train fork/tlc/examples/p0.yaml --stage "$stage"
    $PY -m surogate_tlc sample "$TLC_RUNS/train/$row/$stage" "$TLC_RUNS/data" --out "$TLC_RUNS/train/$row/$stage/samples.json" > /dev/null
  done
  for microbatch in 2 8; do
    TLC_ROW=$row $PY -m surogate_tlc train fork/tlc/examples/p0.yaml --stage pilot --microbatch "$microbatch"
  done
  $PY -m surogate_tlc eval "$TLC_RUNS/train/$row/pilot" "$TLC_RUNS/data" --out "$TLC_RUNS/train/$row/pilot/eval-val.json"
  for suite in synth natural; do
    $PY -m surogate_tlc probe "$TLC_RUNS/train/$row/pilot" "$TLC_RUNS/data" "$TLC_RUNS/suites/$suite/dev.jsonl" \
      --out "$TLC_RUNS/train/$row/pilot/probes-$suite-dev.json" > /dev/null
  done
done
$PY -m surogate_tlc.train.report "$TLC_RUNS/train" --peak-tflops "$PEAK_TFLOPS" | tee "$TLC_RUNS/train/report.jsonl"
$PY -m pytest -c fork/tlc/pyproject.toml fork/tlc/tests/gpu/test_probes.py fork/tlc/tests/gpu/test_gates.py -k "probe or g3" -s | tee "$TLC_RUNS/env/g3.txt"
)
```

每行依次跑 10 步冒烟、100 步过拟合和 300 步 pilot（microbatch 32），每个阶段结束时在固定 prompt 上写出 greedy 样例；再在每次更新 16,384 token 不变的前提下用 microbatch 2 和 8 重跑 pilot（运行目录 `pilot-mb2`、`pilot-mb8`）；`eval` 给出验证 loss、按来源和种类的 loss 与 unigram 基线；`probe` 在两个 dev 套件上打分；`report` 汇总全部运行。判据：

- 每条命令退出码为 0，`status` 为 completed，loss 曲线中没有 NaN 或 Inf。
- smoke：第 1 步的 loss 接近 G2 的初始 loss。
- overfit：在固定 batch 上，第 100 步的 loss 低于第 1 步的一半 [P]。
- pilot：`eval` 输出 `below_unigram: true`。
- 样例：10、100、300 步各有 `samples.json`（固定 prompt、生成的 token 与文本）；`-k g3` 通过，即采样用的 trainer（零初始化的 rank 1 LoRA）给出的 next-token 分布与 `compute_logprobs` 一致，最大绝对差 ≤ 2e-2 [P]。
- 探针：`test_probes.py` 通过，即 A0 在合成 dev 套件的短项上，原生的候选得分与参照实现（bf16 存储）之差不超过 G1(b) 的 `tolerance` × 计分 token 数，参照的领先幅度超过误差界时两者选出同一个候选。pilot 规模下准确率接近 chance 是预期的，这一步只检验探针流水线；test 半不在 G3 读取。
- 记录每行在 microbatch 2、8、32 下的 tokens/s、MFU 与峰值显存，以及每次更新的有效目标数，然后填写下文的 GPU 预算。
- 这些结果是 pilot：G3 只认证流水线，不认证模型质量。未通过 G3 时不进入 Phase 1。

## GPU 预算

按 G3 的实测填写。tokens/s 取自 `python -m surogate_tlc.train.report` 的输出（名义 token，含 pad；按 `valid_targets_per_update / tokens_per_update` 可折算成有效 token）。

GPU 小时 = Σ 种子数 × 名义 token 数 ÷ 实测 tokens/s × (1 + 评测与保存开销) ÷ 3600。

- 关键比较至少 3 个种子，pilot 可以只用 1 个。
- tokens/s 只在相同 microbatch 之间比较，每行注明所用档位。

| 行 | 阶段 | 种子数 | 名义 token（每个种子） | B × GA | 实测 tokens/s | 评测与保存开销 | GPU 小时 |
|---|---|---|---|---|---|---|---|
| A0 | 300 步 pilot | 1 | 4,915,200 | 32 × 1 | 待 G3 | 待 G3 | 待 G3 |
| A1 | 300 步 pilot | 1 | 4,915,200 | 32 × 1 | 待 G3 | 待 G3 | 待 G3 |
| A1h | 300 步 pilot | 1 | 4,915,200 | 32 × 1 | 待 G3 | 待 G3 | 待 G3 |
| A0、A1、A1h | 正式训练 | 3 | 待定 | 待 G3 | 待 G3 | 待 G3 | 待 G3 |
