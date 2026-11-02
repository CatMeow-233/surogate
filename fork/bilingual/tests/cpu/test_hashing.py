"""Content hashes (docs/spec.md, I2-A9; IMPL-14): they depend on tensor names, dtypes, shapes and values only."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from surogate_bilingual.assembly.hashing import content_hash, seed_fingerprint, tensor_hashes


def state(dtype=torch.float32):
    g = torch.Generator().manual_seed(0)
    return {"b.weight": torch.randn(3, 4, generator=g).to(dtype), "a.weight": torch.randn(5, generator=g).to(dtype)}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_i2_a9_hash_is_order_free_and_changes_with_any_element(dtype):
    s = state(dtype)
    h = content_hash(s)
    assert content_hash(dict(reversed(list(s.items())))) == h
    for name in s:
        for index in range(s[name].numel()):
            changed = {k: v.clone() for k, v in s.items()}
            flat = changed[name].view(-1)
            flat[index] = flat[index] + 1  # one element; exact in bf16 too (|x| < 128)
            assert content_hash(changed) != h, (name, index)


def test_i2_a9_hash_ignores_metadata_and_containers():
    s = state()
    h = content_hash(s)

    class Odd(dict):
        step = 3052

    assert content_hash(Odd(s)) == h
    assert content_hash({k: v.clone().requires_grad_() for k, v in s.items()}) == h
    assert content_hash({k: torch.nn.Parameter(v.clone()) for k, v in s.items()}) == h
    assert content_hash({"a.weight": s["a.weight"], "b.weight": s["b.weight"].t().contiguous().t()}) == h  # layout
    with pytest.raises(TypeError, match="tensors only"):
        content_hash({**s, "step": 3})


def test_i2_a9_hash_changes_with_dtype_shape_and_name():
    s = state()
    h = content_hash(s)
    assert content_hash({**s, "a.weight": s["a.weight"].double()}) != h
    assert content_hash({**s, "b.weight": s["b.weight"].reshape(4, 3)}) != h
    assert content_hash({"a.weight": s["a.weight"], "b.weights": s["b.weight"]}) != h
    zero = {"x": torch.zeros(2)}
    assert content_hash(zero) != content_hash({"x": torch.zeros(2, dtype=torch.int32)})  # same bytes, other dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_i2_a9_hash_survives_a_safetensors_round_trip(tmp_path, dtype):
    s = state(dtype)
    save_file(s, str(tmp_path / "w.safetensors"))
    assert content_hash(load_file(str(tmp_path / "w.safetensors"))) == content_hash(s)


def test_tensor_hashes_and_seed_fingerprint():
    s = state()
    per = tensor_hashes(s)
    assert per == {name: content_hash({name: s[name]}) for name in s}
    inputs = ("c" * 64, "b" * 64, "f" * 64, "t" * 64)
    a = seed_fingerprint(*inputs)
    assert a == seed_fingerprint(*inputs)
    for i in range(4):  # every input enters the fingerprint
        assert seed_fingerprint(*inputs[:i], "0" * 64, *inputs[i + 1 :]) != a, i
