# TLC 原型

纯 PyTorch 的原型与数值参照（oracle）：Mamba-2 的参照实现、它的 fp32 goldens，以及在它上面运行的 G1、G2 参照套件。

- 不打包：`fork/tlc/pyproject.toml` 只打包 `fork/tlc/src` 下的 `surogate_tlc`，这里的文件不进任何 wheel。
- 不需要原生扩展，也不 import `surogate_tlc`：在 `fork/` 下的隔离环境中运行，只有 CPU 时用 `fork/.venv-experimental`（`fork/dev test experimental` 创建，已有 torch、numpy 与 safetensors），GPU 主机上用 `fork/.venv-tlc`；用 `-I -B`（不把当前目录放上 `sys.path`，也不写字节码）。CPU 测试 `test_reference_gates.py` 经 `fork/dev test tlc` 用 `FORK_PY` 从检出目录加载它们，属于规则 4 的 CPU 测试例外。
- `surogate_tlc` 不导入这里的模块（它们不随包安装）；测试可以直接运行它们，或读取它们生成的 goldens。

## 文件

| 文件 | 内容 |
|---|---|
| [ref_mamba2.py](ref_mamba2.py) | A0、A0-split 的 Mamba-2 数值参照，读取模型目录（或直接接收权重）并写出 G1(b) 的 fp32 goldens（`.npz`）。逐 token 递归、分块 SSD、prefill 加逐 token 解码三种求值顺序；可选 fp64、bf16 存储模拟和按 position 的文档复位。用法和判据见 [../docs/model.md 的“数值参照”](../docs/model.md#数值参照) |
| [g1_suite.py](g1_suite.py) | G1.1–G1.7 在一个两层小模型上的检查（fp64 与 fp32），以及 G1.8 的五种植入错误，每种都必须让某项检查失败 |
| [g2_suite.py](g2_suite.py) | G2.1–G2.3：梯度、初始 loss 公式，以及小字母表 batch 上完整模型与消融 mixer 的模型的过拟合对照 |

`fork/.venv-experimental/bin/python -I -B fork/tlc/prototypes/g1_suite.py`（`g2_suite.py` 同样）打印每项结果，有失败时退出码为 1；CPU 测试 `fork/tlc/tests/cpu/test_reference_gates.py` 运行同样的检查。
