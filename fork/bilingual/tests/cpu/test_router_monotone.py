"""Adding an expert to a router (docs/spec.md, I3-C11; MA-11). Router training is deferred; this fixes the gate
form. Bitwise checks run on one CPU thread (the conftest fixture).
"""

import pytest
import torch
from torch.nn import functional as F

from surogate_bilingual.routing.plan import RoutePlan
from surogate_bilingual.routing.router import topk_softmax_gates

N, FEATURES, E_OLD = 4096, 32, 3


def router(dtype):
    """Features, the frozen old router and one new column whose bias makes it win on part of the tokens."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, FEATURES, generator=g, dtype=dtype)
    w_old = torch.randn(E_OLD, FEATURES, generator=g, dtype=dtype) / FEATURES**0.5
    w_new = torch.randn(1, FEATURES, generator=g, dtype=dtype) / FEATURES**0.5
    logits_old = F.linear(x, w_old)  # the frozen sub-matrix in its own matmul
    logits_new = F.linear(x, w_new) - 0.5
    return logits_old, logits_new


def switch_raw(logits: torch.Tensor, k: int) -> torch.Tensor:
    """The forbidden Switch-style form: softmax over every logit, top k of the raw probabilities, no renormalising."""
    values, index = torch.topk(torch.softmax(logits, dim=-1), k, dim=-1)
    return torch.zeros_like(logits).scatter(-1, index, values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("k", [1, 2])
def test_i3_c11_tokens_that_do_not_choose_the_new_expert_keep_bitwise_gates(dtype, k):
    logits_old, logits_new = router(dtype)
    before = topk_softmax_gates(logits_old, None, k)
    after = topk_softmax_gates(logits_old, logits_new, k)
    chose_new = after[:, E_OLD] != 0
    assert 0.05 < float(chose_new.double().mean()) < 0.95  # the new expert wins somewhere, so the check is not empty
    stay = ~chose_new
    assert torch.equal(after[stay, :E_OLD], before[stay])
    assert not bool((after[chose_new, :E_OLD] == before[chose_new]).all(-1).any())  # the others do move
    for gates in (before, after):  # rows on the simplex with k nonzero entries: valid P0 plans for k <= 2
        assert bool(((gates != 0).sum(-1) == k).all())
        names = tuple(f"e{i}" for i in range(gates.shape[-1]))
        RoutePlan(names, gates.reshape(1, N, -1).to(torch.float32), "topk").validate()


@pytest.mark.parametrize("k", [1, 2])
def test_i3_c11_switch_style_raw_probabilities_change_almost_every_token(k):
    """The documented negative: without renormalising over the selected logits, adding an expert changes the
    denominator of every token's gates."""
    logits_old, logits_new = router(torch.float32)
    before = switch_raw(logits_old, k)
    after = switch_raw(torch.cat([logits_old, logits_new], dim=-1), k)
    changed = (after[:, :E_OLD] != before).any(-1)
    assert float(changed.double().mean()) > 0.99


def test_topk_softmax_gates_refuses_bad_arguments():
    logits_old, logits_new = router(torch.float32)
    for k in (0, E_OLD + 2):
        with pytest.raises(ValueError):
            topk_softmax_gates(logits_old, logits_new, k)
    with pytest.raises(ValueError):
        topk_softmax_gates(logits_old, logits_new[:-1], 1)
