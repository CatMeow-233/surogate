# 同步上游

本文说明如何把上游 `invergent-ai/surogate` 的 `main` 合入本 fork：一次性设置、每次同步做什么、遇到冲突或检查失败时怎么处理，以及这套做法的测量依据。同步由 `fork/tools/sync_upstream.py` 完成，入口是 `fork/dev sync`，可以在仓库内任意目录运行。

## 速查

```bash
fork/dev sync --dry-run    # 只报告，不改动任何东西
fork/dev sync              # 合并到同步分支，归位笔记，刷新 ledger 和地图，检查，推送分支并开 PR
fork/dev sync --continue   # 解决冲突或修复检查失败并提交之后，从归位步骤继续
fork/dev sync --finish     # PR 的必需检查全部通过后，把 main 快进到同步分支并推送
```

退出码：0 完成；2 需要手动解决冲突；3 预检失败；4 检查失败；1 其他错误（例如某条 git 命令失败）。

节奏：每周一次。上游最近 30 天有 78 次合入，约每周 18 次；按下文“测量依据”，每周同步预计不会遇到内容冲突。

## 规则

- 只用 merge，不 rebase。`experimental-v0.1.0a1`、`a2`、`a3` 三个 tag 指向 `main` 上的 `5341277f`、`1b61b74f`、`20026e80`，rebase 会让它们脱离 `main` 的历史。合并提交保留上游父提交，下一次同步的 merge base 因此始终准确。
- 不用 GitHub 的 “Sync fork”：它直接更新 `main`，不经过同步分支和 fork CI；有冲突时它只提供丢弃 fork 提交的选项。
- 同步 PR 不用 squash 或 rebase 合并（仓库设置中应关闭这两种方式），否则会丢掉上游父提交。PR 通过检查后用 `fork/dev sync --finish` 快进 `main`。
- 不推送 `v*` 或 `jackalope-v*` tag，不执行 `git push --tags`：这些 tag 会触发上游的发布工作流。fork 的工具获取上游时一律带 `--no-tags`。
- `merge.directoryRenames` 保持未设置（git 默认为 `conflict`）。上游重命名一个目录、而 fork 直接在其中添加过文件时，合并报告 `CONFLICT (file location)` 并停下，而不是悄悄移动 fork 文件。fork 自有的子目录（例如 `surogate/_fork`）git 检测不到，由同步脚本检查，见“目录重命名”。
- `rerere.enabled=true`，`rerere.autoUpdate=false`：git 记住以前的冲突解决方式，再次遇到同样的冲突时写回工作区，但不自动暂存，每次都要人工确认。

## 一次性设置

每个本地克隆运行一次 `fork/dev setup`。它添加 `upstream` 远程（如果没有），把 upstream 的推送地址设为 `DISABLED-see-AGENTS.md`，开启 rerere、关闭 rerere 自动暂存，取消 `merge.directoryRenames`，执行 `git fetch --no-tags upstream main`，最后打印下面的 GitHub 设置清单；`gh` 已登录时还会显示仓库当前的合并设置和工作流状态。

检查本地设置：

```bash
git push --dry-run upstream HEAD            # 必须失败
git config --get rerere.enabled             # true
git config --get rerere.autoUpdate          # false
git config --get merge.directoryRenames     # 无输出
```

### GitHub 设置（仓库所有者操作）

fork 的工具不修改 GitHub 设置，以下各项由仓库所有者完成。

1. 禁用上游的三个发布工作流。它们使用 fork 没有的 Blacksmith runner，并向写死的 `ghcr.io/invergent-ai/surogate` 和上游 Release 发布（详见 [build-and-release.md](build-and-release.md)）。fork 不修改这些工作流文件，所以禁用只能在设置中完成。

   ```bash
   gh workflow disable "Build Wheels" -R CatMeow-233/surogate
   gh workflow disable "Build Docker Images" -R CatMeow-233/surogate
   gh workflow disable "jackalope" -R CatMeow-233/surogate
   gh workflow list -R CatMeow-233/surogate --all
   ```

   最后一条命令应显示这三个为 `disabled_manually`，“Experimental Trainer Release” 和 “Fork CI” 为 `active`。

2. 关闭 squash 与 rebase 合并，保留 merge commit，并把仓库描述指向维护入口：

   ```bash
   gh repo edit CatMeow-233/surogate --enable-squash-merge=false --enable-rebase-merge=false --enable-merge-commit \
     --description "Fork of invergent-ai/surogate. Maintainers: start at fork/README.md"
   gh repo view CatMeow-233/surogate --json squashMergeAllowed,rebaseMergeAllowed,description
   ```

3. fork CI 在 `main` 上第一次全部通过后，把 `check`、`experimental`、`fork-tests`、`openmp-contract` 设为 `main` 的必需检查：

   ```bash
   cat > protect.json <<'EOF'
   {"required_status_checks":{"strict":false,"contexts":["check","experimental","fork-tests","openmp-contract"]},"enforce_admins":false,"required_pull_request_reviews":null,"restrictions":null}
   EOF
   gh api -X PUT repos/CatMeow-233/surogate/branches/main/protection --input protect.json
   ```

   `upstream-cpu-lane` 暂不设为必需：它在工作流中是 `continue-on-error: true`。`fork/tools/known_failures.txt` 是在没有 GPU 的容器中生成的；等 CI 上两次相同的运行确认它（或按 CI 上传的 JUnit 结果重新生成）之后，去掉 `continue-on-error` 并把它加入必需检查。

fork CI（`.github/workflows/fork-ci.yml`）在每个分支的每次推送时运行，各 job 执行的都是 `fork/dev` 命令：`check` 运行 `fork/dev check`（先获取上游 `main`），`experimental` 运行 `fork/dev test experimental` 和 `fork/dev release experimental`（只构建，不发布），`fork-tests` 运行 `fork/dev test patches`，`openmp-contract` 运行 `fork/dev test openmp`，`upstream-cpu-lane` 运行 `fork/tools/test_cpu.sh` 并上传 JUnit 结果。

## 每次同步做什么

| 步骤 | 内容 | 失败时 |
|---|---|---|
| 1. 预检 | 当前分支是 `main`，工作区干净，并且与 `origin/main` 一致（先执行 `git fetch --no-tags origin main`）；upstream 推送地址已禁用；rerere 已开启，`merge.directoryRenames` 未设置；`check_divergence.py --strict` 通过；`origin` 上没有 `v*` 或 `jackalope-v*` tag。`--dry-run` 不要求在 `main` 上，也不访问 `origin`；`--local` 不访问 `origin`。 | 退出码 3，列出每一项问题 |
| 2. 获取 | `git fetch --no-tags upstream main:refs/remotes/upstream/main`。给出 `--to <提交>` 时不获取，直接合并该提交（只同步到上游某个较早的提交时，先手动获取上游，再 `fork/dev sync --to <sha>`）。 | 退出码 1 |
| 3. 报告 | 新提交数；用 `git merge-tree` 预演的冲突路径及其所属 ledger 条目；触及 ledger 中文件的上游提交，按 topic 分组；上游新增的根目录 `.md` 及其去向；`csrc/src` 与 `surogate/` 下新增的顶层目录。`--dry-run` 到此结束。报告同时写入 `.git/fork-sync-report.md`，作为 PR 正文。 | — |
| 4. 分支与合并 | `git switch -c sync/<yyyymmdd>-<sha 前 8 位>`，然后 `git merge --no-ff -m "sync: merge upstream/main <sha> (<n> commits)"`。 | — |
| 5. 自动解决 | 只自动处理一种冲突：上游删除了一个已移到 `docs/upstream-notes/` 的笔记（索引中有 base 与 ours 两个 stage，没有上游 stage）。脚本 `git rm` 该文件，并删除 ledger 中对应的 `[[relocation]]`。另外检查 `[owned]` 中每个路径所在的上游目录在合并后是否还存在（见“目录重命名”）。 | 其他冲突或目录消失时停下：退出码 2，列出文件及其 ledger topic |
| 6. 归位提交 | 提交 “sync: re-home upstream root notes”。本次合并新增到根目录的 `.md` 文件，如果匹配 ledger `[root_rehome]` 的 `patterns`、不匹配 `never`，且文件名没有出现在 `tooling_refs` 所列的文件中（出现即说明工具按路径引用它），就 `git mv` 到 `docs/upstream-notes/` 并登记 `[[relocation]]`；其余的只报告，由你决定。归位是单独的提交，合并提交保持纯合并；下一次同步时 merge base 已含根目录路径，上游对该文件的修改会跟随重命名合入。 | — |
| 7. 刷新提交 | 提交 “sync: refresh ledger and maps”：`[sync].last_synced` 设为合并的上游提交；`gen_maps.py --write` 重新生成地图；`check_doc_links.py --update-baseline` 把上游文档中新出现的失效链接加入基线、删去上游已修复的条目，变化逐条写在提交说明里。 | — |
| 8. 检查 | `fork/dev check`；不带 `--fast` 时还运行 `fork/dev test patches`、`fork/dev test experimental` 和 `fork/dev test cpu`。 | 退出码 4 |
| 9. 发布 | `git push -u origin <同步分支>`，`gh pr create -R CatMeow-233/surogate --base main`，PR 正文是第 3 步的报告加上同步分支的提交列表。`--local` 跳过此步。 | 退出码 1 |
| 10. 完成 | PR 的必需检查通过后运行 `fork/dev sync --finish`：先用 `gh pr checks --required` 确认，再 `git switch main`、`git merge --ff-only <同步分支>`、`git push origin main`。 | 检查未全部通过时退出码 4 |
| 11. GPU 复测 | 报告中出现 T1–T5 或 T6–T8 的上游提交时，在 GPU 主机上运行 `fork/dev test gpu`，并在该 topic 的 `conflict_history` 末尾记一行（日期、上游 sha、结果）。 | — |

`--continue` 在同步分支上重新执行第 6–9 步，已经完成的部分会跳过，所以修复之后可以反复运行。`--local` 与 `--fast` 用于演练：前者不访问 `origin`、不推送也不开 PR（之后用 `fork/dev sync --finish --local` 在本地快进），后者跳过第 8 步中的测试套件。

放弃一次同步：

```bash
git merge --abort                       # 仍处于冲突状态时
git switch main
git branch -D sync/<yyyymmdd>-<sha>
```

在移动更多上游路径（新增 `[[relocation]]`）之前：用 `git log --diff-filter=R -M --since='60 days ago' upstream/main -- <目录>` 确认上游近期没有重命名，用 `git grep` 确认没有代码按字符串引用该路径，再按下文的演练方法重放一遍。

## 冲突处理手册

通用做法：以上游版本为基础，把 fork 的改动按下面各节重新加上。

```bash
git diff $(git merge-base HEAD MERGE_HEAD) HEAD -- <文件>   # fork 对该文件做了什么
git log --oneline HEAD..MERGE_HEAD -- <文件>                # 上游改了什么
```

解决后运行该 topic 的测试，在 ledger 中该 topic 的 `conflict_history` 末尾记一行（日期、上游 sha、冲突位置、做法），然后 `git add`、`git commit --no-edit`、`fork/dev sync --continue`。如果上游已经实现了同样的改动，就删去 fork 的版本，并把该文件从 topic 的 `files` 中移除（文件不再是 M 时，`check_divergence.py --strict` 会报告过期条目）。

各 topic 的文件、提交和测试以 [ledger](../patches/ledger.toml) 为准（表格形式见 [patches/README.md](../patches/README.md)）。下面的“回放”是 90/30/14 天窗口内上游修改与 fork 改动重叠的冲突块数（见“测量依据”）。

### T1 dispatch-PP 批量读取边界张量

- 文件：`csrc/src/binding/py_train.cpp`、`csrc/src/runtime/executor/graph_executor.{h,cpp}`、`csrc/src/runtime/executor/graph_executor_utils.{h,cpp}`。
- 改动：graph executor 增加 `read_named_bytes_batch()`；`py_train.cpp` 的 dispatch-PP 函数中 5 处成对的 `read_named_bytes()` 调用改为一次批量调用。
- 回放：`py_train.cpp` 1/0/0。
- 解决：保留上游代码，再把相邻成对读取边界张量的两次 `read_named_bytes()` 合并为一次 `read_named_bytes_batch()`。上游修改 `read_named_bytes()` 的签名或语义时，同步修改批量版本。
- 测试：`fork/native/tests/test-fork-dispatch-pp-host-copy.cpp`（Catch2 标签 `[dispatch-pp]`，编译进 `dsl-run-state-tests`），在 GPU 主机上运行 `fork/dev test native`。

### T2 LoRA 融合输入梯度（`SUROGATE_FUSE_LORA_DX=1`，默认关闭）

- 文件：`csrc/src/runtime/lora/lora_model_utils.h`（约 150 行新增，位于文件中部）、`csrc/src/runtime/lora/lora_slice_dispatch.h`。
- 回放：无冲突。
- 解决：开关关闭时走的路径必须与上游一致。上游改写 LoRA 反向分发时，先采用上游版本，再把融合分支接回开关判断之后；不满足融合条件时仍回退到上游的逐切片实现。
- 测试：`fork/native/tests/test-fork-lora-fused-dx.cu`（`[fused-dx]`，在 `integration-tests` 中）；性能验证方法见 [performance.md](performance.md)。

### T3 MoE backward scratch 与条件清零

- 文件：`csrc/src/runtime/ops/moe_grouped_gemm.cpp`。
- 改动：包含 `fork/native/include/surogate_fork/moe_weight_grad_policy.h`（写作 `#include "surogate_fork/moe_weight_grad_policy.h"`）；基础专家权重梯度只在 `moe_weight_grad_needs_zero()` 判定需要时清零；backward 不再分配未使用的 FP8 输入量化缓冲。
- 回放：2/0/0。
- 解决：保留上游，重新加上 include 行和清零条件。只有在上游 backward 仍不使用 `ctx.inp_quant` 时才删去那段缓冲分配。
- 测试：`fork/native/tests/test-fork-moe-grouped-gemm.cu`（`[wgrad-zero]`）。

### T4 FP8 与 matmul 临时张量

- 文件：`csrc/src/recipes/fp8_hybrid/fp8_hybrid_recipe.cpp`、`csrc/src/runtime/ops/matmul.cpp`。
- 改动：已缓存转置权重时，FP8 backward 不再在线量化一份未转置的权重；matmul backward 不再为跳过的权重梯度分配临时张量，传给 recipe 的 `dweight` 可以为空。
- 回放：`matmul.cpp` 1/0/0。
- 解决：保留上游，重新应用这两处。如果上游的 recipe 不再接受空的 `dweight`，就恢复上游的临时张量写法，并在 `conflict_history` 中记录。
- 测试：`fork/native/tests/test-fork-fp8-quant.cpp`（`[recipe]`）。

### T5 OpenMP 与原生挂接点

- 文件：`csrc/CMakeLists.txt` 中的两行挂接（`include(.../fork_native.cmake)` 和 `surogate_fork_configure_common(surogate-common)`），紧跟在 `surogate-common` 的 `target_link_libraries` 之后。约定见 [native/README.md](../native/README.md)。
- 回放：无冲突。该文件 90 天内有 147 次上游提交，但挂接点所在区域没有改动。
- 解决：上游移动这一区域时，把两行放回 `surogate-common` 定义之后。上游改名 `surogate-common` 时，同时修改挂接行的参数。上游自己为训练库链接 OpenMP 时，从 `fork/native/cmake/fork_native.cmake` 删去 `surogate_enable_training_openmp()` 调用，并更新 T5。
- 测试：`fork/dev test openmp`；GPU 主机上 `fork/dev test native`（确认 fork 的测试仍编进上游目标）。

### T6–T8 GRPO

三个 topic 共用 `surogate/grpo/trainer.py`。回放：`trainer.py` 6/0/0，`packer.py` 1/0/0，`data.py` 与 `batch.py` 无冲突。90 天窗口中 `trainer.py` 的 6 处是：T8 的 `_find_sample_boundaries`；`_log_step` 的指标字典（fork 只在其中保留 `**getattr(self, "_fork_cycle_metrics", {})` 一行）；`train()` 中 T6 的 3 处计时点；1 处 `.ravel()`。fork 的逻辑放在 `surogate/_fork`，上游文件中只保留调用它的几行；fork 的 import 写在函数内部，不要移到模块顶部的 import 区，上游经常修改那里。

- T6 周期计时（`trainer.py`、`packer.py`）：`train()` 记录 publish、pack、batch_wait、batch_receive 各阶段的耗时；每次调用 `_log_step` 前设置 `self._fork_cycle_metrics`，并调用 `surogate/_fork/grpo_cycle_timing.py` 中的 `log_cycle_summary()` 打印单独一行；`packer.py` 的等待循环只在没有批次时 sleep。上游改写主循环时以上游为准，再把计时点放回对应阶段。测试：`fork/patches/tests/grpo/test_trainer_cycle_timing.py`、`fork/patches/tests/grpo/test_packer_wait.py`。
- T7 进程内 micro-batch 传递（`data.py`、`packer.py`、`trainer.py`）：`GRPODataLoader` 的关键字参数 `micro_batch_receiver`（注解写成字符串，不导入该名字），`packer.py` 中 `BasePacker`、`SinglePacker` 与 `setup_grpo_packer` 的关键字参数 `micro_batch_sender`，以及 `trainer.py` 的 `_setup_data` 中对 `surogate/_fork/grpo_handoff.py` 的局部导入。不要为注解在 `data.py` 的 import 区加导入：整理前唯一一处 1/1/1 冲突就在那里。测试：`fork/patches/tests/grpo/test_inprocess_micro_batches.py`。
- T8 向量化批次准备（`batch.py`、`trainer.py`）：`prepare_batch` 用切片代替逐个 `pop(0)`；`_find_sample_boundaries` 向量化；14 处 `.flatten()` 改为 `.ravel()`。这些只影响性能；上游重写了这些函数时可以直接采用上游版本、去掉 fork 的这部分，并在 `conflict_history` 中记录。测试：`fork/patches/tests/grpo/test_batch_distribution.py`、`fork/patches/tests/grpo/test_microbatch_preparation.py`。
- 以上测试都用 `fork/dev test patches` 运行。GPU 主机上 `fork/dev test gpu` 包含 `make grpo-test`，指标 JSONL 中应有 `train/cycle_duration_ms` 等计时字段。

### T9 setuptools_scm 只认 `v*` tag

- 文件：`pyproject.toml` 的 `[tool.setuptools_scm]` 中一行 `git_describe_command`（`--match v[0-9]*`）。
- 解决：上游修改该表时保留这一行；没有它，`experimental-v*` tag 会成为原生包的版本来源。
- 测试：`fork/experimental/tests/test_release_version.py`（`fork/dev test experimental`）。

### 移动过的上游笔记与删除的文件

- 上游修改已移动的笔记：没有冲突，修改合入 `docs/upstream-notes/` 中的文件。
- 上游删除已移动的笔记：rename/delete 冲突，第 5 步自动处理。
- 上游把已移动的笔记改名到别处：rename/rename 冲突，需要手动处理。通常接受上游的新位置：`git rm` fork 的副本，`git add` 上游的新路径，并删去对应的 `[[relocation]]`；新位置仍在根目录时，第 6 步会再次归位。
- 上游重新修改 fork 已删除的 `RELEASE_v1.4.0.md`：modify/delete 冲突。确认仍没有引用后 `git rm`；如果上游重新启用了它，就保留文件并删去 `[[deletion]]`。

### 目录重命名

上游重命名或删除一个含有 fork 文件的目录时，分两种情况：

- fork 文件直接位于该目录中（例如 `.github/workflows/` 下的两个 fork 工作流）：git 报告 `CONFLICT (file location)` 并给出建议的新位置，第 5 步停下。
- fork 自有的子目录位于该目录中（`surogate/_fork/`、`docs/upstream-notes/`、`docs/experimental/`）：git 不报告冲突，而是把子目录留在旧位置（在 git 2.43 上验证，`merge.directoryRenames` 设为 `conflict` 或 `true` 都一样）。因此同步脚本对 `[owned]` 中的每个路径找出它所在的上游目录（例如 `surogate/_fork/` 所在的 `surogate/`），该目录在上游消失时，报告中列为需要手动处理，合并后以退出码 2 停下。

处理方法相同：把 fork 文件 `git mv` 到上游的新位置，更新 ledger 的 `[owned]`、topic 中的路径、引用这些路径的 fork 文档和 `fork/tools/gen_maps.py` 的分类表，运行 `fork/dev map`，提交，再 `fork/dev sync --continue`。fork 改过的上游文件会随 git 的重命名检测移到新位置，但 ledger 中登记的路径要一起改，否则 `check_divergence.py` 会失败。

### 检查失败（退出码 4）

- `gen_maps.py` 报告 “unclassified directory”、“unclassified subpackage” 或 “unclassified markdown file”：在 `fork/tools/gen_maps.py` 的 `CSRC_ENGINES`、`PY_ROLES` 或 `DOC_RULES` 中补一条规则，再运行 `fork/dev map`。
- `check_doc_links.py` 报告 fork 文档中的问题：`文件:行号` 超出了文件长度，或引用的上游路径已不存在。按上游的新内容更新 fork 文档。上游文档中新出现的失效链接已由刷新提交加入基线。
- `check_divergence.py --strict` 报告 “is not modified (M)”：上游已经包含了该改动，从 topic 中删去该文件。报告 “not owned” 或 “outside the ledger”：合并结果带入了未登记的差异（通常是解决冲突时留下的改动），登记它或撤销它。
- `make -n` 探测失败：上游改名或删除了 `fork/dev` 调用的 make 目标，更新 `fork/dev` 和 [build-and-release.md](build-and-release.md)。
- 测试失败：如果上游测试在上游提交本身也失败，就把它加入 `fork/tools/known_failures.txt` 并注明该 sha；由 fork 造成的失败必须先修复，同步才能继续。

修复后在同步分支上提交，再运行 `fork/dev sync --continue`。

## 测量依据

### 冲突面（反向回放）

把 fork 的当前状态与上游 90、30、14 天前的提交（固定为 `14ad7e86`、`14b9069a`、`9a717fb2`）做三方合并预演（`git merge-tree`，以 `282f9901` 为 base），得到的是该窗口内上游修改与 fork 改动重叠的冲突块数：

| 文件 | 整理前（`b5af2f8e`） | 整理后 |
|---|---|---|
| `surogate/grpo/trainer.py` | 8/2/0 | 6/0/0 |
| `surogate/grpo/data.py` | 1/1/1 | 0/0/0 |
| `surogate/grpo/packer.py` | 1/0/0 | 1/0/0 |
| `csrc/src/binding/py_train.cpp` | 1/0/0 | 1/0/0 |
| `csrc/src/runtime/ops/matmul.cpp` | 1/0/0 | 1/0/0 |
| `csrc/src/runtime/ops/moe_grouped_gemm.cpp` | 2/0/0 | 2/0/0 |

整理后，按每周（以及每两周）一次同步，预计没有内容冲突；拖到 90 天才同步，预计要手动解决约 11 处。fork 改过的上游文件从 22 个减到 16 个，位于上游目录中的 fork 文件从 10 个减到 0 个。

### 正向演练

在一个共享克隆中，从 `558ab695`（2026-09-11）开始：先执行当时已经存在的笔记操作（移动 `TODO.md`、`TODOv2.md`、`glm-dsa-kernels.md`，删除 `RELEASE_v1.4.0.md`），复制 fork 自有的全部文件，ledger 的 `last_synced` 设为起点并去掉 topic（fork 对上游文件的改动是在更晚的上游上写的，放到起点上本身就会冲突）。然后对 `558ab695..282f9901` 的 295 个 first-parent 合入逐个运行 `sync_upstream.py --local --fast --to <合入>` 和 `--finish --local`，最后在 `282f9901` 之上加 5 个人造的上游提交。结果：

- 295 次同步全部没有冲突，没有一次需要手动处理；每次约 6 秒。
- `DESIGN-kv-admission.md` 在 2026-10-19 的合入 `18ef8bcb`、`DESIGN-host-kv-tier.md` 在 2026-10-23 的合入 `282f9901` 被归位到 `docs/upstream-notes/`。每次同步后根目录的 markdown 都只有 `AGENTS.md` 和 `README.md`。
- 每次同步后 `check_divergence.py --strict` 都通过。
- 每次同步都在第 8 步的 `fork/dev check` 处以退出码 4 结束，这是演练的设置造成的：复制过去的 fork 文档描述的是今天的代码（行号、后来才加入的脚本），而演练中没有 T1–T9 的改动。到最后一个合入时只剩两项：`fork/README.md` 引用的服务端 CMake 段落行号范围到第 2023 行，而没有 T5 两行挂接的 `csrc/CMakeLists.txt` 只有 2021 行；上游 2026-10-16 在 `docs/` 下加入的 `grpo_routing_replay.md` 没有 `DOC_RULES` 规则（真实的 fork 已把它移到 `docs/upstream-notes/`）。后者正是新文档分类检查应有的反应。
- 人造提交：(a) 修改 `TODO.md`：没有冲突，修改落在 `docs/upstream-notes/TODO.md`。(b) 删除 `TODO.md`：1 个 rename/delete 冲突，自动解决，文件和对应的 `[[relocation]]` 一起删除。(c) 根目录新增 `CHANGELOG.md`：在 never 列表中，不移动，报告。(d) 根目录新增 `DESIGN-x.md` 并在 `Makefile` 中引用：不移动，报告 “named in Makefile”。(e) 把 `surogate/` 改名为 `python/surogate/`：git 没有报告任何冲突，`surogate/_fork` 留在了旧位置；同步脚本的目录检查在报告和合并后都指出了它，并以退出码 2 停下（见“目录重命名”）。

重放方法（例如在新增 `[[relocation]]` 之前）：

```bash
git clone -q --shared . ../rehearse && cd ../rehearse
git remote add upstream /nonexistent && git remote set-url --push upstream DISABLED-see-AGENTS.md
git config rerere.enabled true && git config rerere.autoUpdate false
git fetch -q <原仓库路径> refs/remotes/upstream/main:refs/remotes/upstream/main
git checkout -B main <起点>
# 在起点上重做要验证的移动，复制 fork 自有文件，ledger 的 last_synced 设为起点、
# 删去起点上不成立的条目，然后提交
for L in $(git rev-list --first-parent --reverse <起点>..upstream/main); do
    python3 -I fork/tools/sync_upstream.py --local --fast --to "$L"; rc=$?
    [ "$rc" -eq 0 ] || [ "$rc" -eq 4 ] || break    # 2：需要手动解决
    python3 -I fork/tools/sync_upstream.py --finish --local
    python3 -I fork/tools/check_divergence.py --strict || break
done
```
