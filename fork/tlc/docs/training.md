# 训练驱动与评测

相关模块：`surogate_tlc.train.config`（运行配置）、`surogate_tlc.train.driver`（训练驱动）、`surogate_tlc.train.report`（环境指纹与运行摘要）、`surogate_tlc.native`（原生训练器的构建）、`surogate_tlc.eval.loss`、`surogate_tlc.eval.probes`、`surogate_tlc.eval.samples`，以及探针条目 `surogate_tlc.data.items`、启发式 `surogate_tlc.data.floors`、自然代码套件 `surogate_tlc.data.natural`。

## 训练驱动

`surogate_tlc.train.driver.train()`（命令 `train`）直接调用 `SurogateTrainerWrapper(config, train_files, eval_files).train()`（`surogate/train/trainer.py`），不经过上游的 `sft` 命令。上游 `sft` 先 tokenize 数据集，再用 `train_with_oom_recovery`（`surogate/train/sft.py`）包住训练：OOM 时它把 microbatch 减半，GA 达到 16 以后不再加倍，于是每次更新的 token 数悄悄减半，只留一条 warning。TLC 的数据是 fork 写的 shard，各行比较时每次更新的名义 token 数固定。

### 运行配置

运行配置是上游的 SFT 配置文件（`SFTConfig` 读取的键）加一张 `tlc` 表，由 `surogate_tlc.train.config` 读取。唯一的示例 [examples/p0.yaml](../examples/p0.yaml) 供所有行使用：模型目录、运行目录和运行名取自环境变量 `TLC_ROW`，其余设置各行相同（CPU 测试核对），所以 A0、A1、A1h 在同样的超参数下比较。字符串中的 `${VAR}` 从环境替换，变量未设置时报错。

```yaml
tlc:
  data: ${TLC_RUNS}/data      # data.shards 的输出目录
  tokens_per_update: 16384    # 每次更新的名义 token = microbatch × GA × sequence_len × gpus
  stages:                     # 用 --stage 选择，覆盖 tlc 表之外的键
    smoke: {max_steps: 10, save_steps: 5}
    overfit: {max_steps: 100, fixed_batch: true}
```

- `gradient_accumulation_steps` 由 `tokens_per_update` 和 microbatch 推出，配置里不写；写了而且不一致就报错。示例的 microbatch 是 32 行 × 512 token、GA 为 1，不开 recompute（梯度检查点）：2 × 512 的 microbatch 受 kernel launch 开销限制。`--microbatch B` 改 microbatch，GA 随之调整。
- 示例的阶段：`smoke`（10 步，第 5 步的 checkpoint 供 G2 的恢复检查）、`overfit`（固定 batch 上 100 步，恒定学习率）、`pilot`（300 步，4,915,200 名义 token）、`throughput` 与 `throughput-eager`（G0.5：各 30 步，开或关 CUDA graph）。
- 示例不用 weight decay：原生 AdamW 对每个参数都做 decay，包括 Mamba-2 参照实现排除的 `A_log`、`dt_bias`、`D`（见 [model.md 的“初始化”](model.md#初始化)）。
- 运行目录是 `<output_dir>/<阶段>`，给出 `--microbatch B` 时是 `<output_dir>/<阶段>-mb<B>`，所以 microbatch 扫描中的各次运行不会接着别人的 checkpoint 训练。checkpoint、上游日志和运行记录 `tlc_run.json` 都在运行目录中。
- 驱动替配置设定下列值，配置写了别的值就拒绝：`lora: false`、`sample_packing: false`、`row_packing: false`、`from_scratch: false`（`surogate_tlc.native.FIXED`）。上游默认开 LoRA 和 tokenize 期打包；`from_scratch` 会用原生初始化覆盖 fork 写的权重。
- 不能出现的键：`datasets`、`validation_datasets`（数据只来自 `tlc.data`），`distillation`（P0 没有蒸馏），`checkpoint_dir`（上游不读它，checkpoint 总在 `output_dir` 中，写了只会误导）。`max_steps` 必须是正整数：每次运行的长度固定；运行没有时长或磁盘上限。
- 训练前检查（`surogate_tlc.train.config.problems()`，有任何问题就不开始，退出码 2）：模型目录满足契约且带权重；`sequence_len` 等于 shard 的行长，且不超过 `max_position_embeddings`；manifest 中 tokenizer 的 sha256 等于模型目录中 `tokenizer.json` 的 sha256；数据目录中有 `train-*.bin`。
- 训练文件是 `tlc.data` 中全部的 `train-*.bin`，评测文件是全部的 `val-*.bin`。阶段设 `fixed_batch: true` 时，驱动把第一个 train 文件的前 `tokens_per_update / sequence_len` 行（示例中 32 行）原样写成运行目录中的 `fixed-batch.bin`，每一步都训练这一批行。
- 每次更新中非 pad 的有效目标数（ValidTokenCount 的期望值）写进运行记录：固定 batch 时是这批行的精确值，否则按 manifest 中 train 划分的 `valid_targets / tokens` 折算。上游训练器不按步记录 ValidTokenCount。

### 恢复

再次运行同一条命令时，上游从运行目录中最新的 checkpoint 继续（`resume_from_checkpoint: true`）。运行已经完成、`max_steps` 也没变时，驱动直接返回运行记录，不再调用训练器：否则上游会从最后保存的一步起把末段重训一遍，并覆盖最终导出。恒定学习率的阶段加大 `max_steps` 时照常恢复。要从头重训，删除运行目录或换一个阶段、`--microbatch`。驱动只在运行的身份不变时允许恢复，否则拒绝（退出码 2）并列出变了的项：

- 初始权重、config.json、tokenizer 和数据 manifest 的 sha256；
- 训练设置（`surogate_tlc.train.config.settings()` 的 sha256）：除下列键以外的全部上游设置，加上学习率调度的总步数（恒定学习率时为空）、`tokens_per_update` 和是否固定 batch。可以改的键只影响保存、日志、评测的频率和运行长度：`output_dir`、`max_steps`、`save_steps`、`save_total_limit`、`eval_steps`、`logging_steps`、`log_gpu_util`、`run_name`、`resume_from_checkpoint`。`max_steps` 同时决定非恒定学习率调度的长度，所以对 cosine 等调度改它会被拒绝。microbatch 与 GA 也在设置之中：checkpoint 中数据加载器的位置以 microbatch 为单位。

### 失败

任何失败（OOM 也一样）都立即停止，运行记录中写 `failure_report`：错误类型、消息与 traceback、当时的 microbatch 与 GA、显存（`torch.cuda.mem_get_info`）、最后一个 checkpoint，OOM 时还有补救办法（用更小的 `--microbatch` 开一个新运行：每次更新的 token 数不变，运行目录不同）。驱动不在同一个运行中自动降档：降档会改变运行的身份，而且降档后的运行与别的行比较时要另行说明。已写完的 checkpoint 照常收尾，可以用作模型目录。

识别 OOM：消息含 "out of memory" 或单词 "OOM" 的 `RuntimeError`（原生分配器的 "Cuda OOM when allocating tensor ..."，CUDA 的 out of memory），以及 `MemoryError`（原生 activation 栈的 `std::bad_alloc`）。

### 运行记录与收尾

运行记录 `tlc_run.json`（`tlc_run_version` 2）：

- `identity`：初始权重、config.json、tokenizer、数据 manifest 和训练设置的 sha256；`settings` 是训练设置本身，拒绝恢复时据此指出变了的键。
- `attempts`：每次调用（第一次和每次恢复）的 microbatch、GA、名义 token 数、配置文件的 sha256、从哪一步恢复、环境指纹、上游日志的路径、状态，以及完成时的摘要。
  - 环境指纹（`surogate_tlc.train.report.environment()`）：提交、工作区是否有未提交的改动、Python 与各包的版本、原生扩展的路径与 sha256 及其旁边 `libsurogate-common.so`（运行时与 kernel）的路径与 sha256、GPU 名称、计算能力、显存和驱动版本。结果因此能追溯到产生它的代码和机器；扩展只是一层绑定，只改 kernel 时它可能逐字节不变，两次 kernel 构建靠共享库的 sha256 区分（两者都由 `fork/dev build train` 写出）。
  - 摘要：loss 曲线、首末 loss、验证 loss、按每步时间中位数折算的 tokens/s（名义 token，含 pad）、峰值显存、block schema 统计。驱动为每次调用设置环境变量 `SUROGATE_REGRESSION_ARTIFACT`，上游训练器据此写出 `regression-<序号>.json`；摘要取自它和上游日志。
- `status`（running、completed、failed）、`failure_report`、`data`（文件、固定 batch 的统计、每次更新的有效目标数）、`sft`（实际使用的上游设置）。

收尾：训练完成后，驱动对运行目录中每个完整的 step 目录执行 `finalize_dir()`，再对运行目录本身（上游的最终导出）执行一次；训练失败时只收尾已有的 checkpoint。`tlc_meta.json` 另记 `tokenizer_sha256`、`data_manifest_sha256`、`shard_version`（shard 格式版本，即 `surogate_tlc.data.shards.SHARD_VERSION`）、`kind`（`checkpoint` 或 `run`）和 `run`：写出该目录的那次调用（`attempt` 与它的 `resumed_from_step`）、训练设置的 sha256，以及那次调用的配置文件 sha256、提交和工作区是否有改动。step 目录归写出它的调用，即起点低于该步的最后一次调用（上游只保存恢复点之后的步），所以换了提交或改了配置文件之后恢复，不会把前面调用写出的 checkpoint 改记到新的调用名下；被强行终止、没来得及收尾的调用写出的 checkpoint，也由下一次调用按原来的调用收尾。运行记录中没有哪次调用写出的 step 目录（例如从别的运行复制进来）保持原样。运行目录本身记最后一次调用。因此运行目录和其中的 step 目录都可以直接用作 `model_dir`；`export` 保留 step 目录 `tlc_meta.json` 中的这些字段。

`python -m surogate_tlc.train.report DIR [--peak-tflops X]` 把 DIR 下每个运行记录打印成一行 JSON：阶段、状态、调用次数（`attempts`）与最后一次调用的 `resumed_from_step`、microbatch、GA、是否用 CUDA graph、有效目标数、步数范围、首末 loss、验证 loss、tokens/s、峰值显存、前向 FLOPs/token（`surogate_tlc.ir.flops_per_token`），给出 GPU 的 dense bf16 峰值时还有 MFU（tokens/s × 3 × 前向 FLOPs/token ÷ 峰值）。loss 与验证 loss 接起所有调用：后一次调用的点替换它恢复点之后的点，所以失败后恢复的运行仍从第 1 步读起。完成的调用的 loss 取自摘要（regression artifact，步数从 1 起），失败的调用没有 artifact，取自它的上游日志的 step 行（步数从 0 起，pin 测试钉住）。吞吐与显存取自最后一次调用。G0.5、G3 的记录和 [运行手册](runbook.md) 中的 GPU 预算都取自它。

## 评测

`python -m surogate_tlc eval MODEL DATA`（`surogate_tlc.eval.loss.evaluate()`）按原生 DataLoader 的读法（`served_rows()`）取 `DATA` 的 val 划分，用原生 `compute_logprobs(use_lora=False)` 打分；loss 是有效目标上的平均负对数概率。基线是 train 划分中目标 token 的 add-one 平滑 unigram 分布下的 loss：连 token 频率都没学到的模型谈不上训练过，所以 G3 要求验证 loss 低于它（输出中的 `below_unigram`）。模型目录的 tokenizer 必须与 shard 的一致。输出另有行数、有效目标数、困惑度、tokenizer 与数据 manifest 的 sha256，以及按来源（`by_source`）和按来源种类（`by_kind`：`code`、`text`）的有效目标数、loss 与困惑度；`by_kind.text` 的困惑度就是英文 PPL。每行的来源取自 shard manifest 中各来源的行区间。

按角色分层的 loss 需要 P1 的角色标签，自然代码中指向局部变量的标识符的 NLL、α-重命名不变性也在 P0 之后，见 [README 的“P0 之后”](../README.md#p0-之后)。

### 绑定探针

```bash
python -m surogate_tlc.data.natural CORPUS.toml OUT_DIR --limit 4000      # T3 的 dev 与 test
python -m surogate_tlc probe MODEL DATA SUITE.jsonl --out FILE             # 打分
```

主指标是行为式的绑定准确率：每个探针项（`surogate_tlc.data.items.Item`）有一个 prompt 和若干候选；模型给每个候选打分，正确的候选得分最高即为答对。各行按同一标准计分，不读模型的内部状态，所以 Mamba、attention 与混合行可以直接比较。

- **T1、T2**：[data.md 的“合成绑定任务”](data.md#合成绑定任务)中的合成程序。prompt 到读出行的 `== ` 为止，候选是 prompt 中出现的全部字符串字面量，chance 为候选数的倒数。
- **T3 自然代码**：`surogate_tlc.data.natural` 从语料 val 划分的代码文档中（`generated = true` 的来源除外），取函数内对局部变量的读取作为探针项。
  - 该名字在函数文本中此前已有绑定。绑定从 Python 实际执行它的位置起算：赋值在语句结束之后，循环变量在可迭代对象之后。
  - 候选是此前出现过的局部名（至少 2 个），答案是被读的名字。
  - 只用静态子集中的函数：没有 global、nonlocal，没有对名字的 del，没有 eval、exec、locals、getattr 之类的反射。
  - AST（α-重命名后）与 train 划分中某个函数相同的函数不用。
  - 每个文档最多取 8 项；dev 与 test 按文档的 group 分开。manifest 记录取了哪些来源。
- **打分**（`surogate_tlc.eval.probes`）：
  - 每个候选的文本是 prompt + 候选 + 尾部（候选之后到行末），按 shard 的方式编码：文档开头加 `<|bos|> <|code_start|>`，正文中的特殊 token 字符串按普通文本处理。
  - 得分是一段 token 的对数概率之和：从共享上下文之后的第一个 token 起，到包含尾部第一个字符的 token 为止。因此候选按"在此处结束的整词"计分。
  - 有 token 跨过 prompt 末尾时（例如 GPT-2 式预分词把空格和后面的字母合成一个 token），共享上下文在最早的那个 token 之前结束。各候选的上下文 token 不一致的项不计分（`context`）。
  - 合成项放不进模型的上下文时不计分（`too_long`），因为截断可能切掉被问的绑定；自然代码项保留最后 seq_len 个 token，与 shard 的窗口一致。
  - 有候选得分不是有限值时（例如发散的 checkpoint）不计分（`nonfinite`）：对 NaN 取 max 会选中第一个候选，对 T1、T2 而言就是 `line1` 启发式。
  - 原生打分与 `eval` 相同，都用 `compute_logprobs(use_lora=False)`；每行的长度是 `DATA` 中 shard 的行长，每批 `--batch` 行。
- **启发式 floor**：准确率只有超出文本层面的捷径才有意义，所以每项都带着各启发式的预测。T1、T2 的启发式来自一个不看作用域的文本读者（`surogate_tlc.data.floors`）：

  | 启发式 | 预测 |
  |---|---|
  | `recent` | prompt 中最后一个字面量 |
  | `line1` | prompt 中第一个字面量 |
  | `same_name` | 把所有作用域当作一个名字空间，沿每个名字在文本中最近的绑定解析读出的表达式；调用中的关键字实参绑定被调函数中同名的形参 |
  | `first_bound` | 同上，但沿最早的绑定解析 |
  | `stale` | 读者在该表达式最后一次变化之前给出的值 |
  | `position` | 被调函数返回的形参按实参的位置对应，关键字实参也按书写顺序计 |

  T3 的启发式：`recent`（最近出现的局部名）和 `frequent`（此前读取次数最多的局部名）。
- **汇总**（`surogate_tlc.eval.probes.summarize`，按任务给出）：
  - 准确率、chance、每个启发式的准确率；不计分的项按原因计入 `skipped`。
  - 按格（T1：模式、live、hops、overwrites、depth；T2：模式、formals、distractors、actuals）给出 `floor` 和 `shortcut_floor`。`floor` 是该格最好的启发式；`shortcut_floor` 只在该格中不恒对的启发式里取最好的。恒对的启发式就是该格的正确算法，例如没有遮蔽时的 `same_name`、只有一层遮蔽时 `shadow-out` 的 `stale`、没有重绑时的 `first_bound`、按位置调用时的 `position`。
  - `above_floor` 与 `above_shortcut_floor`：每个计分项的对错（1 或 0）减去所在格的 floor，再取平均。
  - `disagreement`：所有不恒对的启发式都答错的项（启发式分歧子集）。
  - `stale_rate`：选中了读者给出的上一个值、而它又不是答案的比例（只跟踪名字、不跟踪重绑的模型会选它）。
  - 同样的指标还按每个分层键、被留出的轴（`heldout`：family、pattern、pool 或 none）、两套距离桶和被问标识符的 token 片数（1、2、3、≥4）分组给出。距离是答案所在绑定到 prompt 末尾的 token 数：`distance`（≤128、128–512、512–2048、>2048）覆盖后续阶段的 2048 上下文，`distance_512`（≤64、65–256、257–512、>512）细分 P0 的 512 上下文。
- 输出：`probe` 打印 `summary`，以及模型、套件、tokenizer 和数据 manifest 的 sha256 与行长；`--out` 另写每项的结果（状态、选择、各候选得分、距离和片数）。

### 生成样例

`python -m surogate_tlc sample MODEL DATA`（`surogate_tlc.eval.samples`）在固定的 prompt（`PROMPTS`：四个代码开头、两个英文开头）上做 greedy 生成，最多 `--max-new` 个 token，遇到 `<|eos|>` 或 `<|code_end|>` 停止。原生 `next_token_logits` 只在带常驻 LoRA 权重的 trainer 上运行，所以采样用的 trainer 带一个 rank 1 的 LoRA（`surogate_tlc.native.SAMPLING_LORA`，挂在 MLP 的 down 投影上）：原生 LoRA 的 B 初始化为零，适配后的模型在初始权重上就是原模型（pin 测试钉住这一点，G3 在 GPU 上核对它的 next-token 分布与 `compute_logprobs` 一致）。训练之后不一定：带 LoRA 的 trainer 把 tied embedding 当作 head，不读 checkpoint 中的 `lm_head.weight`，见 [model.md 的“已知问题：tied LM head”](model.md#已知问题tied-lm-head)。Mamba 还没有原生的解码 cache，每生成一个 token 都重算整行；G3 只有几个 prompt，这个开销可以接受。prompt 按 shard 的方式框定：代码前加 `<|bos|> <|code_start|>`，英文前加 `<|bos|>`。
