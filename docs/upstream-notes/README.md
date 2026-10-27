# 上游工程笔记

本目录存放上游 `invergent-ai/surogate` 的工程笔记（TODO 清单、设计稿、专题说明）。fork 把它们从仓库根目录和 `docs/` 原样移到这里，内容一字未改，因此上游之后对这些文件的修改会在同步时跟随重命名合入。

上游代码和文档只以文件名提到这些笔记（例如注释中的 DESIGN-kv-admission.md），没有按路径链接，所以按文件名搜索仍能找到它们。

每次同步时，上游新放到根目录的同类笔记会按 [ledger](../../fork/patches/ledger.toml) 中 `[root_rehome]` 的规则移到这里，并登记为 `[[relocation]]`；已有的移动记录也在其中。本文件由 fork 维护，其余文件属于上游。
