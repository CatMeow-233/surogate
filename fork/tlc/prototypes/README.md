# TLC 原型

纯 PyTorch 的原型与数值参照（oracle），例如 Mamba-2 的逐 token 递归参照及其 fp32 goldens。

- 不打包：`fork/tlc/pyproject.toml` 只打包 `fork/tlc/src` 下的 `surogate_tlc`，这里的文件不进任何 wheel。
- 不需要原生扩展：在 `fork/.venv-experimental` 或任一装有 CPU torch 的环境中运行。
- `surogate_tlc` 不导入这里的模块（它们不随包安装）；测试可以直接运行它们，或读取它们生成的 goldens。
