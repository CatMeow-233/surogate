"""Tensor operators: norm, RoPE, attention, FFN blocks and fusion math. Operators never see a route or a name;
dispatch lives in model.executor (docs/spec.md, "模块边界")."""
