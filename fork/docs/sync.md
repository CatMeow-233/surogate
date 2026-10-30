# 同步上游

本文说明如何把上游 `invergent-ai/surogate` 的 `main` 合入本 fork：一次性设置、每次同步做什么、遇到冲突或检查失败时怎么处理，以及这套做法的测量依据。同步由 `fork/tools/sync_upstream.py` 完成，入口是 `fork/dev sync`，可以在仓库内任意目录运行。

## 速查

```bash
fork/dev sync --dry-run    # 只报告，不改动任何东西
fork/dev sync              # 合并到同步分支，归位笔记，刷新 ledger 和地图，检查，推送分支并开 PR
fork/dev sync --continue   # 解决冲突或修复检查失败并提交之后，从归位步骤继续
fork/dev sync --finish     # 同步分支当前提交（即 PR 的 head）上四个必需检查都通过后，把 main 快进到这个提交并推送
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

   `fork/dev sync --finish` 用 `gh api repos/CatMeow-233/surogate/commits/<sha>/check-runs` 检查同样的四个 job（`fork/tools/sync_upstream.py` 中的 `REQUIRED_CHECKS`）在要推送的那个提交上是否通过，所以设置分支保护之前也能完成同步，而且 `enforce_admins` 为 `false` 时所有者推送的也只能是检查过的提交；修改必需检查时两处一起改。`upstream-cpu-lane` 暂不设为必需：它在工作流中是 `continue-on-error: true`。`fork/tools/known_failures.txt` 是在没有 GPU 的容器中生成的；等 CI 上两次相同的运行确认它（或按 CI 上传的 JUnit 结果重新生成）之后，去掉 `continue-on-error` 并把它加入必需检查。

fork CI（`.github/workflows/fork-ci.yml`）在每个分支的每次推送时运行，各 job 执行的都是 `fork/dev` 命令：`check` 运行 `fork/dev check`（先获取上游 `main`）和 `fork/dev test release`，`experimental` 运行 `fork/dev test experimental` 和 `fork/dev release experimental`（只构建，不发布），`fork-tests` 运行 `fork/dev test patches`、`fork/dev test tlc` 和 `fork/dev test tools`（每个用例最多 120 秒，卡住的用例按名字报告失败，而不是等到 job 超时），`openmp-contract` 运行 `fork/dev test openmp`，`upstream-cpu-lane` 运行 `fork/tools/test_cpu.sh` 并上传 JUnit 结果。

## 每次同步做什么

| 步骤 | 内容 | 失败时 |
|---|---|---|
| 1. 预检 | 当前分支是 `main`，工作区干净，并且与 `origin/main` 一致（先执行 `git fetch --no-tags origin main`）；upstream 推送地址已禁用；rerere 已开启，`merge.directoryRenames` 未设置；`check_divergence.py --strict` 通过；`origin` 上没有 `v*` 或 `jackalope-v*` tag；`gh api` 读到的工作流状态中，上游的 `wheel.yml`、`docker.yml`、`jackalope.yml` 都不是 `active`（`jackalope.yml` 在推送到 `main` 时就会发布 Release）。`--dry-run` 不要求在 `main` 上，也不访问 `origin` 和 GitHub；`--local` 同样不访问。 | 退出码 3，列出每一项问题 |
| 2. 获取 | `git fetch --no-tags upstream main:refs/remotes/upstream/main`。给出 `--to <提交>` 时不获取，直接合并该提交；它必须是已获取的 `upstream/main` 的祖先，否则退出码 3（只同步到上游某个较早的提交时，先手动获取上游，再 `fork/dev sync --to <sha>`）。 | 退出码 1 |
| 3. 报告 | 新提交数；用 `git merge-tree` 预演的冲突路径及其所属 ledger 条目；触及 ledger 中文件的上游提交，按 topic 分组；上游新增的根目录 `.md` 及其去向；`csrc/src` 与 `surogate/` 下新增的顶层目录；上游新增的工作流（检查它的触发条件，会发布的就在 GitHub 上禁用）。`--dry-run` 到此结束；否则报告写入 `.git/fork-sync-report.md`，作为 PR 正文的开头。 | — |
| 4. 分支与合并 | `git switch -c sync/<yyyymmdd>-<sha 前 8 位>`，然后 `git merge --no-ff -m "sync: merge upstream/main <sha> (<n> commits)"`。 | — |
| 5. 自动解决 | 只自动处理一种冲突：上游删除了一个已移到 `docs/upstream-notes/` 的笔记（索引中有 base 与 ours 两个 stage，没有上游 stage）。脚本 `git rm` 该文件，并删除 ledger 中对应的 `[[relocation]]`。另外检查 `[owned]` 中每个路径所在的上游目录在合并后是否还存在（见“目录重命名”）。 | 其他冲突或目录消失时停下：退出码 2，列出文件及其 ledger topic |
| 6. 归位提交 | 提交 “sync: re-home upstream root notes”。本次合并新增到根目录的 `.md` 文件，如果匹配 ledger `[root_rehome]` 的 `patterns`、不匹配 `never`，文件名没有出现在 `tooling_refs` 所列的文件中（出现即说明工具按路径引用它），没有其他 `.md`/`.mdx` 文件用链接指向它，它自己的相对链接从 `docs/upstream-notes/` 出发也仍然到达原来的文件（链接按 `check_doc_links.py` 的语法解析；指向已移动笔记或指向笔记自身的链接移动后不变，其余相对链接都会失效，或者到达别的文件，例如 `README.md` 会指向 fork 的 `docs/upstream-notes/README.md`；fork 不改上游文档，无法修复这些链接），就 `git mv` 到 `docs/upstream-notes/` 并登记 `[[relocation]]`；其余的留在根目录并在报告中写明原因，由你决定：留在根目录的文件要在 `fork/tools/gen_maps.py` 的 `DOC_RULES` 中分类（例如 `("N", "RELEASE_v1.6.0.md")`）并运行 `fork/dev map`，否则第 8 步的 `fork/dev check` 报告 “unclassified markdown file”。归位是单独的提交，合并提交保持纯合并；下一次同步时 merge base 已含根目录路径，上游对该文件的修改会跟随重命名合入。 | — |
| 7. 刷新提交 | 提交 “sync: refresh ledger and maps”：`[sync].last_synced` 设为合并的上游提交；`gen_maps.py --write` 重新生成地图；`check_doc_links.py --update-baseline` 把上游文档中新出现的失效链接加入基线、删去上游已修复的条目，变化逐条写在提交说明里。新的失效链接由 fork 造成时不自动加入基线：链接指向 fork 移动（`[[relocation]]` 的 `from`）或删除（`[[deletion]]`）的上游路径，或者链接写在 fork 移动过的笔记中、从笔记的上游位置出发能到达一个存在的路径。这时刷新撤销自己的改动并停下，见“移动过的上游笔记与删除的文件”。 | 新的失效链接由 fork 造成时退出码 4 |
| 8. 检查 | `fork/dev check`；不带 `--fast` 时还运行 `fork/dev test patches`、`fork/dev test experimental`、`fork/dev test tlc`、`fork/dev test tools` 和 `fork/dev test cpu`，它们需要准备好的 `.venv`（或 `FORK_PY`）、uv，以及 download.pytorch.org（无法访问时设 `FORK_TORCH_INDEX=pypi`），见 [fork/README.md 的“首次准备环境”](../README.md#首次准备环境)。 | 退出码 4 |
| 9. 发布 | `git push -u origin <同步分支>`；PR 还没有打开时 `gh pr create -R CatMeow-233/surogate --base main`，正文（`.git/fork-sync-pr.md`）是第 3 步的报告加上同步分支的提交列表；PR 已经打开时（例如 `--continue` 之后）推送就更新了它，不再新建。`--local` 跳过此步。 | 退出码 1 |
| 10. 完成 | PR 的检查通过后运行 `fork/dev sync --finish`。它先 `git fetch origin main <同步分支>`，要求本地 `HEAD` 等于 `origin/<同步分支>` 和 PR 的 head（`gh pr view --json headRefOid`），并且包含 `origin/main`；再用 `gh api .../commits/<HEAD>/check-runs` 确认 `check`、`experimental`、`fork-tests`、`openmp-contract` 在这个提交上都已通过（同名检查重跑过时以最新一次为准）；然后 `git switch main`、`git merge --ff-only <HEAD>`、`git push origin <HEAD>:refs/heads/main`。推送的总是检查过的那个提交：本地还没推送的提交、比 PR head 旧的本地分支都不会进入 `main`。 | 任一条件不满足时退出码 4，列出原因或未通过的 job，`main` 不变 |
| 11. GPU 复测 | 报告中出现 T1–T5、T6–T8 或 T10 的上游提交时，在 GPU 主机上运行 `fork/dev test gpu`，并在该 topic 的 `conflict_history` 末尾记一行（日期、上游 sha、结果）。 | — |

`--continue` 在同步分支上重新执行第 6–9 步，已经完成的部分会跳过，所以修复之后可以反复运行。它按分支名 `sync/<日期>-<sha 前 8 位>` 找同步合并：同步分支第一父链上第二父提交以这 8 位开头、并且是 `upstream/main` 祖先的最新合并；找不到（例如 `git merge --abort` 之后）或该提交不在上游历史中时停下，不会把别的合并当作同步合并。

同步期间 `main` 有了新提交（例如合并了别的 PR）时，在同步分支上 `git merge origin/main`，提交后运行 `fork/dev sync --continue`，等新的检查通过后再 `--finish`；`--finish` 在 `origin/main` 不是同步分支祖先时拒绝推送。`--local` 与 `--fast` 用于演练：前者不访问 `origin`、不推送也不开 PR（之后用 `fork/dev sync --finish --local` 在本地快进），后者跳过第 8 步中的测试套件。

放弃一次同步：

```bash
git merge --abort                       # 仍处于冲突状态时
git switch main
git branch -D sync/<yyyymmdd>-<sha>
```

在移动更多上游路径（新增 `[[relocation]]`）之前：用 `git log --diff-filter=R -M --since='60 days ago' upstream/main -- <目录>` 确认上游近期没有重命名，用 `git grep` 确认没有代码按字符串引用该路径，再按下文的正向演练方法演练一遍。

## 冲突处理手册

通用做法：以上游版本为基础，把 fork 的改动按下面各节重新加上。

```bash
git diff $(git merge-base HEAD MERGE_HEAD) HEAD -- <文件>   # fork 对该文件做了什么
git log --oneline HEAD..MERGE_HEAD -- <文件>                # 上游改了什么
```

解决后运行该 topic 的测试，在 ledger 中该 topic 的 `conflict_history` 末尾记一行（日期、上游 sha、冲突位置、做法），然后 `git add`、`git commit --no-edit`、`fork/dev sync --continue`。如果上游已经实现了同样的改动，就删去 fork 的版本，并把该文件从 topic 的 `files` 中移除（文件不再是 M 时，`check_divergence.py --strict` 会报告过期条目）。

各 topic 的文件、提交和测试以 [ledger](../patches/ledger.toml) 为准（表格形式见 [patches/README.md](../patches/README.md)）。下面的“反向回放”是把 fork 与上游 90、30、14 天前的提交做三方合并预演时的冲突块数（见“测量依据”）。

### T1 dispatch-PP 批量读取边界张量

- 文件：`csrc/src/binding/py_train.cpp`、`csrc/src/runtime/executor/graph_executor.{h,cpp}`、`csrc/src/runtime/executor/graph_executor_utils.{h,cpp}`。
- 改动：graph executor 增加 `read_named_bytes_batch()`；`py_train.cpp` 的 dispatch-PP 函数中 5 处成对的 `read_named_bytes()` 调用改为一次批量调用。
- 反向回放：`py_train.cpp` 1/0/0。
- 解决：保留上游代码，再把相邻成对读取边界张量的两次 `read_named_bytes()` 合并为一次 `read_named_bytes_batch()`。上游修改 `read_named_bytes()` 的签名或语义时，同步修改批量版本。
- 测试：`fork/native/tests/test-fork-dispatch-pp-host-copy.cpp`（Catch2 标签 `[dispatch-pp]`，编译进 `dsl-run-state-tests`），在 GPU 主机上运行 `fork/dev test native`。

### T2 LoRA 融合输入梯度（`SUROGATE_FUSE_LORA_DX=1`，默认关闭）

- 文件：`csrc/src/runtime/lora/lora_model_utils.h`（约 150 行新增，位于文件中部）、`csrc/src/runtime/lora/lora_slice_dispatch.h`。
- 反向回放：无冲突。
- 解决：开关关闭时走的路径必须与上游一致。上游改写 LoRA 反向分发时，先采用上游版本，再把融合分支接回开关判断之后；不满足融合条件时仍回退到上游的逐切片实现。
- 测试：`fork/native/tests/test-fork-lora-fused-dx.cu`（`[fused-dx]`，在 `integration-tests` 中）；性能验证方法见 [performance.md](performance.md)。

### T3 MoE backward scratch 与条件清零

- 文件：`csrc/src/runtime/ops/moe_grouped_gemm.cpp`。
- 改动：包含 `fork/native/include/surogate_fork/moe_weight_grad.h`（写作 `#include "surogate_fork/moe_weight_grad.h"`）；`CompiledExecutor::backward_moe_base_weights()` 从形状检查开始的部分（dtype 映射、清零、逐 expert 的 `cublasGemmEx` 循环）移到 `fork/native/src/moe_weight_grad.cpp` 的 `dsl::moe_base_weight_grad()`，由 `fork_native.cmake` 编译进 `surogate-common`，原处只留一次调用；其中清零改为只在 `moe_weight_grad_needs_zero()` 判定需要时执行（上游在不累加时总是清零）；backward 不再分配未使用的 FP8 输入量化缓冲。
- 反向回放：2/0/0（移动循环前后相同）。
- 解决：保留上游，重新加上 include 行，再把 `ensure_output_tensor()` 与 `store_tensor()` 之间的代码换回 `accumulate` 的计算和对 `moe_base_weight_grad()` 的调用。上游修改了移走的那些行（形状检查、dtype、GEMM 参数或循环）时，冲突出现在这个 hunk：把上游的修改移植到 `fork/native/src/moe_weight_grad.cpp`，并确认每个有 token 的 expert 仍由一次 beta 为零的 GEMM 写满整个 rows×cols 块；做不到时把清零条件改回上游的 `!accumulate`。只有在上游 backward 仍不使用 `ctx.inp_quant` 时才删去那段缓冲分配。
- 测试：`fork/patches/tests/native/test_moe_weight_grad.py`（CPU，`fork/dev test patches`）确认 `backward_moe_base_weights()` 只通过 `moe_base_weight_grad()` 写 dW、fork 源文件编译进库、GPU 测试调用的正是这个函数。`fork/native/tests/test-fork-moe-grouped-gemm.cu`（`[wgrad-zero]`，在 `integration-tests` 中，GPU 主机上运行 `fork/dev test native`）直接调用 `moe_base_weight_grad()`，覆盖：dW 预填 NaN、每个 expert 都有 token 且不累加（相当于 `cpu_training` 流式模式下不清零就复用的梯度槽）；BF16 输入配 BF16 或 FP32 的 dW；dY 全为零的 expert；跨层复用同一个梯度槽；按 tile 截断的 offsets。上游修改 `moe_grouped_gemm.cpp` 后（第 11 步），这些用例检查的就是生产代码。

### T4 FP8 与 matmul 临时张量

- 文件：`csrc/src/recipes/fp8_hybrid/fp8_hybrid_recipe.cpp`、`csrc/src/runtime/ops/matmul.cpp`。
- 改动：已缓存转置权重时，FP8 backward 不再在线量化一份未转置的权重；matmul backward 不再为跳过的权重梯度分配临时张量，传给 recipe 的 `dweight` 可以为空。
- 反向回放：`matmul.cpp` 1/0/0。
- 解决：保留上游，重新应用这两处。如果上游的 recipe 不再接受空的 `dweight`，就恢复上游的临时张量写法，并在 `conflict_history` 中记录。
- 测试：`fork/native/tests/test-fork-fp8-quant.cpp`（`[recipe]`）。

### T5 OpenMP、原生挂接点与 CPU 优化器的线程数

- 文件：`csrc/CMakeLists.txt` 中的两行挂接（`include(.../fork_native.cmake)` 和 `surogate_fork_configure_common(surogate-common)`），紧跟在 `surogate-common` 的 `target_link_libraries` 之后。约定见 [native/README.md](../native/README.md)。`csrc/src/runtime/optimizers/cpu_adamw.cpp`：包含 `surogate_fork/omp_team.h`；5 个逐元素循环的 pragma 前一行是 `const surogate_fork::OmpTeam team(n);`，pragma 末尾加 `num_threads(team.threads) if (team.threads > 1)`；`cpu_gradient_norm_squared` 的两个 `reduction(+ : param_norm)` 循环改为调用 `surogate_fork::ordered_block_sum`。原因与测量见 [performance.md](performance.md#cpu-优化器并行)。
- 反向回放：无冲突。`csrc/CMakeLists.txt` 90 天内有 147 次上游提交，但挂接点所在区域没有改动；上游最后一次修改 `cpu_adamw.cpp` 是 2026-04-19（`a7ebaa56`），早于 90 天窗口。
- 解决：上游移动挂接区域时，把两行放回 `surogate-common` 定义之后。上游改名 `surogate-common` 时，同时修改挂接行的参数。上游自己为训练库链接 OpenMP 时，从 `fork/native/cmake/fork_native.cmake` 删去 `surogate_enable_training_openmp()` 调用，并更新 T5。`cpu_adamw.cpp` 以上游为准，再重新加上 include、每个区域前的 `OmpTeam` 和两个子句；上游新增的 `#pragma omp parallel` 区域同样处理（pin 会因区域个数变化而失败）。上游改写范数循环时，保留它跳过非有限值和指针去重的逻辑（T10 依赖去重，T5 的 pin 检查去重这一行），只把求和换回 `ordered_block_sum`。上游自己为这些区域确定线程数时，比较两种做法，再决定是否删去 fork 的版本。
- 测试：`fork/dev test openmp`（`training-openmp-contract` 和线程数契约 `training-openmp-team`）；`fork/dev test patches`（源码 pin `fork/patches/tests/native/test_cpu_adamw_openmp.py`）；GPU 主机上 `fork/dev test native`（确认 fork 的测试仍编进上游目标）。

### T6–T8 GRPO

三个 topic 共用 `surogate/grpo/trainer.py`（T10 在其中也有一处调用，见 T10）。反向回放：`trainer.py` 6/0/0，`packer.py` 1/0/0，`data.py` 与 `batch.py` 无冲突。90 天窗口中 `trainer.py` 的 6 处是：T8 的 `_find_sample_boundaries`；`_log_step` 的指标字典（fork 只在其中保留 `**getattr(self, "_fork_cycle_metrics", {})` 一行）；`train()` 中 T6 的 3 处计时点；1 处 `.ravel()`。fork 的逻辑放在 `surogate/_fork`，上游文件中只保留调用它的几行；fork 的 import 写在函数内部，不要移到模块顶部的 import 区，上游经常修改那里。

- T6 周期计时（`trainer.py`、`packer.py`）：`train()` 记录 publish、pack、batch_wait、batch_receive 各阶段的耗时；每次调用 `_log_step` 前设置 `self._fork_cycle_metrics`，并调用 `surogate/_fork/grpo_cycle_timing.py` 中的 `log_cycle_summary()` 打印单独一行；`packer.py` 的等待循环只在没有批次时 sleep。上游改写主循环时以上游为准，再把计时点放回对应阶段。测试：`fork/patches/tests/grpo/test_trainer_cycle_timing.py`、`fork/patches/tests/grpo/test_packer_wait.py`。
- T7 进程内 micro-batch 传递（`data.py`、`packer.py`、`trainer.py`）：`GRPODataLoader` 的关键字参数 `micro_batch_receiver`（注解写成字符串，不导入该名字），`packer.py` 中 `BasePacker`、`SinglePacker` 与 `setup_grpo_packer` 的关键字参数 `micro_batch_sender`，以及 `trainer.py` 的 `_setup_data` 中对 `surogate/_fork/grpo_handoff.py` 的局部导入。不要为注解在 `data.py` 的 import 区加导入：整理前唯一一处 1/1/1 冲突就在那里。测试：`fork/patches/tests/grpo/test_inprocess_micro_batches.py`。
- T8 向量化批次准备（`batch.py`、`trainer.py`）：`prepare_batch` 用切片代替逐个 `pop(0)`；`_find_sample_boundaries` 向量化；14 处 `.flatten()` 改为 `.ravel()`。这些只影响性能；上游重写了这些函数时可以直接采用上游版本、去掉 fork 的这部分，并在 `conflict_history` 中记录。测试：`fork/patches/tests/grpo/test_batch_distribution.py`、`fork/patches/tests/grpo/test_microbatch_preparation.py`。
- 以上测试都用 `fork/dev test patches` 运行。GPU 主机上 `fork/dev test gpu` 包含 `make grpo-test`，指标 JSONL 中应有 `train/cycle_duration_ms` 等计时字段。

### T9 setuptools_scm 只认 `v*` tag

- 文件：`pyproject.toml` 的 `[tool.setuptools_scm]` 中一行 `git_describe_command`（`--match v[0-9]*`）。
- 解决：上游修改该表时保留这一行；没有它，`experimental-v*` tag 会成为原生包的版本来源。
- 测试：`fork/experimental/tests/test_release_version.py`（`fork/dev test experimental`）。

### T10 tied 的 LM head 在全量训练时共用 embedding

- 文件：`csrc/src/runtime/dsl/dsl_param_store.{h,cpp}`（`storage_alias` 的条件不再要求 `freeze_base`；无法共用的可训练 tied head 报错；`iterate_tensors` 跳过别名；`set_weight_manager` 核对两边一致）、`csrc/src/runtime/dsl/dsl_weight_manager.{h,cpp}`（`DslWeightManagerConfig::tie_lm_head`、`DslWeightEntry::tied_alias`、`allocate_weights` 中 `lm_head` 取 embedding 的 master 与 work、`iterate_tensors` 跳过）、`csrc/src/runtime/dsl/dsl_model_weights.cpp`（导出与 `resolve_tied_params` 跳过别名）、`csrc/src/runtime/dsl/dsl_grad_store.cpp`（`allocate_streaming_buffers` 中共用设备梯度的名字共用主机副本）、`csrc/src/runtime/optimizers/adamw_8bit_optimizer.cpp`（`step_cpu_streaming` 每个 master 只更新一次）、`surogate/dsl/models/gemma3.py`（`_build_gemma3_model` 中一行 `tie_word_embeddings = True`：官方 Gemma 3 config 不写这个键，没有这行时原生训练把它读成不 tie，T10 不生效）、`surogate/train/trainer.py` 与 `surogate/grpo/trainer.py`（`__init__` 中构建 IR 之后的一处局部导入和对 `surogate/_fork/dispatch_pp_tied_head.py` 中 `refuse_lost_tied_gradient(config, ir_json)` 的调用：dispatch-PP 全量训练在阶段 0 与损失阶段不在同一块 GPU 上时拒绝 tied 模型）。缺陷与证据见 [upstream-errata.md](upstream-errata.md#全量训练时-tied-的-lm-head-停在初始值fork-已修复ledger-t10)。
- 解决：每处改动都很小，上游改写这些函数时以上游为准，再把对应的几行放回；源码 pin 按函数体定位，会指出缺了哪一处。上游自己修好 tied head（例如在 IR 中让 head 直接读 embedding，或让参数存储在全量训练时也共用存储）时，删掉 fork 的对应改动和 pin，从 ledger 删除 T10。dispatch-PP 全量训练在阶段 0 与损失阶段不在同一块 GPU 上时，tied 矩阵只得到 LM head 的梯度（上游已有，见 errata）；fork 没有修原生代码，而是由 `refuse_lost_tied_gradient` 拒绝这些配置。两个 trainer 的调用必须留在构建 IR 之后、创建原生 trainer 之前（pin 检查这个顺序）。`test_dispatch_pp_tied_head.py` 还 pin 住 `py_train.cpp` 中 `dispatch_pp_train_step_multigpu` 的梯度收集与 `dsl_model_execution.cpp` 中的 `dispatch_pp_read_block_grads`、`dispatch_pp_write_grads`：上游改了这些函数时这个 pin 失败，按新代码检查测试中的重放；上游把各 GPU 的部分相加后，删掉拒绝、调用和这个测试。
- 依赖：T10 还依赖 fork 没有改动的上游代码，pin 同样检查。`lm_head` 的梯度就是 embedding 的梯度缓冲；各个 GPU 优化器、设备上的梯度范数和 `cpu_training` 的梯度范数对同一个梯度指针只处理一次（T10 让两个名字共用一份主机梯度后，CPU 范数靠它只算一次）；config 不写 `tie_word_embeddings` 时取声明的值；`lm_head.weight` 缺失时从 embedding 加载（T10 的导出没有 `lm_head.weight`，QLoRA 和带 weight manager 的 LoRA 不共用 head 的存储，靠这个回退加载这些导出）；dispatch-PP 的拒绝按原生的梯度收集方式判断。上游去掉去重时，共用的矩阵会被更新两次，或 tied 梯度在裁剪用的范数中算两次。这些代码按同步报告能否发现分两类（pin 失败，或报告中出现 T10 的上游提交时，按第 11 步做 GPU 复测）：
  - 在 T10 的 `files` 中、fork 没有改动的函数：`dsl_grad_store.cpp` 的 `DslGradStore` 构造函数（梯度别名）、`adamw_8bit_optimizer.cpp` 的 `init_state`（指针去重）。第 3 步的报告把上游对这两个文件的提交归到 T10。
  - 不在 T10 的 `files` 中：`adamw_optimizer.cpp` 与 `normuon_optimizer.cpp` 的 `step`/`step_graph`，`dsl_model.cpp` 的 `calculate_gradient_norm` 与 `apply_declared_config_defaults`，`cpu_adamw.cpp` 的 `cpu_gradient_norm_squared`，`pretrained_config.cpp` 的 `load_pretrained_config`，`ir.cpp` 的 `load_ir_from_json`，`dsl_weight_loader.cpp` 中 `DslWeightLoader::load_direct` 的 tied 回退，`dsl_model_execution.cpp` 的 `dispatch_pp_read_block_grads` 与 `dispatch_pp_write_grads`，以及 `py_train.cpp` 的 `dispatch_pp_train_step_multigpu`。报告不把上游对它们的提交归到 T10（`py_train.cpp` 属于 T1，它的提交归到 T1），第 8 步的 `fork/dev test patches` 中 pin 失败才是信号。
- 测试：`fork/patches/tests/native/test_tied_lm_head.py` 与 `fork/patches/tests/native/test_dispatch_pp_tied_head.py`（`fork/dev test patches`，不需要 GPU）；GPU 主机上 `fork/dev test native`（`dsl-run-state-tests '[tied-head]'`）和 TLC 运行手册 G2 的 `test_g2_tied_head_trains_as_the_embedding`。这两个测试不经过 Gemma 3 的路径；第 11 步的 GPU 复测还要做 [errata](upstream-errata.md#全量训练时-tied-的-lm-head-停在初始值fork-已修复ledger-t10) 验证段中 Gemma 3 的全量训练检查。

### 移动过的上游笔记与删除的文件

- 上游修改已移动的笔记：没有冲突，修改合入 `docs/upstream-notes/` 中的文件。
- 上游删除已移动的笔记：rename/delete 冲突，第 5 步自动处理。
- 上游把已移动的笔记改名到别处：rename/rename 冲突，需要手动处理。通常接受上游的新位置：`git rm` fork 的副本，`git add` 上游的新路径，并删去对应的 `[[relocation]]`；新位置仍在根目录时，第 6 步会再次归位。
- 上游重新修改 fork 已删除的 `RELEASE_v1.4.0.md`：modify/delete 冲突。确认仍没有引用后 `git rm`；如果上游重新启用了它，就保留文件并删去 `[[deletion]]`。
- 上游文档新增指向已移动笔记（或已删除文件）旧路径的链接，或者上游在已移动的笔记中新增相对链接（链接从 `docs/upstream-notes/` 解析，因而失效）：第 7 步以退出码 4 停下并列出这些链接。通常把笔记移回根目录：`git mv docs/upstream-notes/<文件> <文件>`，删去 ledger 中对应的 `[[relocation]]`（已删除的文件则恢复它并删去 `[[deletion]]`），在 `fork/tools/gen_maps.py` 的 `DOC_RULES` 中给根目录的这个文件分类，运行 `fork/dev map`。确实要接受失效链接时，自己把第 7 步列出的行（`<文档>\t<链接目标>`）加入 `fork/tools/doc_links_baseline.txt`。然后提交并 `fork/dev sync --continue`。第 6 步不会移动有链接指向、或者自己带相对链接的新笔记，所以这种情况只发生在以前移动或删除的路径上。
- 上游在已移动的笔记中新增的相对链接从 `docs/upstream-notes/` 出发恰好到达另一个存在的文件时（例如 `README.md` 到达 fork 的 `docs/upstream-notes/README.md`），链接没有失效，第 7 步发现不了。第 3 步的报告在“relocated or deleted upstream notes”下列出修改已移动笔记的上游提交，检查它们新增的链接；处理方法同上。

### 目录重命名

上游重命名或删除一个含有 fork 文件的目录时，分两种情况：

- fork 文件直接位于该目录中（例如 `.github/workflows/` 下的两个 fork 工作流）：git 报告 `CONFLICT (file location)` 并给出建议的新位置，第 5 步停下。
- fork 自有的子目录位于该目录中（`surogate/_fork/`、`docs/upstream-notes/`、`docs/experimental/`）：git 不报告冲突，而是把子目录留在旧位置（在 git 2.43 上验证，`merge.directoryRenames` 设为 `conflict` 或 `true` 都一样）。因此同步脚本对 `[owned]` 中的每个路径找出它所在的上游目录（例如 `surogate/_fork/` 所在的 `surogate/`），该目录在上游消失时，报告中列为需要手动处理，合并后以退出码 2 停下。

处理方法相同：把 fork 文件 `git mv` 到上游的新位置，更新 ledger 的 `[owned]`、topic 中的路径、引用这些路径的 fork 文档和 `fork/tools/gen_maps.py` 的分类表，运行 `fork/dev map`，提交，再 `fork/dev sync --continue`。fork 改过的上游文件会随 git 的重命名检测移到新位置，但 ledger 中登记的路径要一起改，否则 `check_divergence.py` 会失败。

### 检查失败（退出码 4）

- `gen_maps.py` 报告 “unclassified directory”、“unclassified subpackage” 或 “unclassified markdown file”：在 `fork/tools/gen_maps.py` 的 `CSRC_ENGINES`、`PY_ROLES` 或 `DOC_RULES` 中补一条规则，再运行 `fork/dev map`。
- `gen_maps.py` 报告 “not found; upstream moved or deleted it” 或 “no COMMAND_MAPPING”：上游移动或改写了脚本按固定路径解析的文件（例如 `surogate/cli/main.py`、`csrc/CMakeLists.txt`）。把 `fork/tools/gen_maps.py` 中的路径或解析方式改成上游的新写法，再运行 `fork/dev map`。
- `check_doc_links.py` 报告 fork 文档中的问题：引用的上游路径已不存在。按上游的新内容更新 fork 文档。上游文档中新出现的失效链接已由刷新提交加入基线。
- `check_divergence.py --strict` 报告 “is not modified (M)”：上游已经包含了该改动，从 topic 中删去该文件。报告 “not owned” 或 “outside the ledger”：合并结果带入了未登记的差异（通常是解决冲突时留下的改动），登记它或撤销它。
- `check_divergence.py` 报告 “is not exactly the [[exact_diff]] lines”：该文件与上游的差异必须恰好是 ledger 中 `[[exact_diff]]` 列出的行（目前是 `csrc/CMakeLists.txt` 的两行挂接），按文件检查的 M 条目发现不了丢失的一行。按输出中的 missing/extra 恢复挂接行；有意修改挂接行时（例如上游改名 `surogate-common` 后修改参数），同时更新 `[[exact_diff]]` 的 `lines`。`fork/dev test openmp` 在 configure 时也会检查挂接点（报错以 “fork hook contract” 开头），见 [native/README.md 的挂接约定](../native/README.md#挂接约定)。
- `check_divergence.py` 报告 “[sync].last_synced ... is not an upstream commit”：`last_synced` 只能是上游提交。改成本次同步合并的上游提交（同步合并的第二父提交）。
- `make -n` 探测失败：上游改名或删除了 `fork/dev` 调用的 make 目标，更新 `fork/dev` 和 [build-and-release.md](build-and-release.md)。
- 测试失败：如果上游测试在上游提交本身也失败，就把它加入 `fork/tools/known_failures.txt` 并注明该 sha；由 fork 造成的失败必须先修复，同步才能继续。

修复后在同步分支上提交，再运行 `fork/dev sync --continue`。

## 测量依据

整理 fork 时用两种方法测量了冲突面：把 fork 与上游 90、30、14 天前的提交做三方合并预演（反向回放），以及从 `558ab695`（2026-09-11）起逐个同步上游的 295 次 first-parent 合入（正向演练，全部没有手动冲突，两个新的根目录笔记被自动归位）。结论：每周同步一次，预计不会遇到内容冲突；拖到 90 天才同步，预计要手动解决约 11 个冲突块。各 topic 的数字记在 ledger 的 `conflict_history` 中，表格见 [patches/README.md](../patches/README.md)。

### 正向演练方法

新增 `[[relocation]]` 或修改同步脚本之前，在一个共享克隆中重演一段上游历史：

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

复制到起点上的 fork 文档描述的是今天的代码，所以演练中第 8 步的 `fork/dev check` 可能以退出码 4 结束（例如引用了起点上还不存在的文件）；循环把它当作可以继续的结果，只在需要手动解决冲突时停下。
