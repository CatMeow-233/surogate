# TLC 原型

纯 PyTorch 的原型与数值参照（oracle），例如 Mamba-2 的逐 token 递归参照及其 fp32 goldens。

- 不打包：`fork/tlc/pyproject.toml` 只打包 `fork/tlc/src` 下的 `surogate_tlc`，这里的文件不进任何 wheel。
- 不需要原生扩展：在 `fork/.venv-experimental` 或任一装有 CPU torch 的环境中运行。
- `surogate_tlc` 不导入这里的模块（它们不随包安装）；测试可以直接运行它们，或读取它们生成的 goldens。

## 文件

| 文件 | 内容 |
|---|---|
| [ref_mamba2.py](ref_mamba2.py) | A0、A0-split 的 Mamba-2 数值参照，读取模型目录并写出 G1(b) 的 fp32 goldens（`.npz`）。逐 token 递归、分块 SSD 和逐 token 解码三种求值顺序；可选 bf16 存储模拟。用法和判据见 [../README.md 的“数值参照”](../README.md#数值参照g1b) |
