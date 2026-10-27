"""Layer selection preserves gradient flow and explicit optimizer-state semantics."""

import copy
import json
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from surogate.experimental.train.layers import LayerSampler


class _Model(nn.Module):
    def __init__(self, layers=4, frozen_base=False, tied_head=False):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(layers)])
        self.head = nn.Linear(4, 8, bias=False)
        if tied_head:
            self.head.weight = self.embedding.weight
        if frozen_base:
            for layer in self.model.layers:
                layer.weight.requires_grad_(False)

    def forward(self, ids):
        value = self.embedding(ids)
        for layer in self.model.layers:
            value = torch.tanh(layer(value))
        return self.head(value)


def test_frozen_decoder_blocks_still_pass_gradients_to_embedding():
    torch.manual_seed(7)
    model = _Model(frozen_base=True)
    initially_frozen = [layer.weight for layer in model.model.layers]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    sampler = LayerSampler(model, 1, 2, seed=17)
    python_rng, torch_rng = random.getstate(), torch.random.get_rng_state().clone()
    report = sampler.on_step(0, optimizer)
    model(torch.tensor([[1, 2, 3]])).square().mean().backward()
    assert model.embedding.weight.grad is not None and model.embedding.weight.grad.abs().sum() > 0
    assert model.head.weight.grad is not None
    for index, layer in enumerate(model.model.layers):
        assert not layer.weight.requires_grad and layer.weight.grad is None
        assert layer.bias.requires_grad == (index in sampler.selected_layers)
        assert (layer.bias.grad is not None) == (index in sampler.selected_layers)
    assert all(not parameter.requires_grad for parameter in initially_frozen)
    assert report["trainable_parameter_count"] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert random.getstate() == python_rng and torch.equal(torch.random.get_rng_state(), torch_rng)


@pytest.mark.parametrize("policy", ["drop", "cpu"])
def test_inactive_state_is_dropped_or_retained_on_cpu(policy):
    model = _Model()
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.tensor([[1, 2]])).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    original = {id(parameter): copy.deepcopy(state) for parameter, state in optimizer.state.items()}
    sampler = LayerSampler(model, 1, 1, state_policy=policy, seed=5)
    before = sampler.optimizer_state_bytes(optimizer)
    report = sampler.on_step(0, optimizer)
    for index, layer in enumerate(model.model.layers):
        for parameter in layer.parameters():
            if policy == "drop" and index not in sampler.selected_layers:
                assert parameter not in optimizer.state
            else:
                for key, value in original[id(parameter)].items():
                    torch.testing.assert_close(optimizer.state[parameter][key], value)
                    assert optimizer.state[parameter][key].device.type == "cpu"
    assert model.embedding.weight in optimizer.state and model.head.weight in optimizer.state
    assert report["optimizer_state_bytes"] == sampler.optimizer_state_bytes(optimizer)
    assert (report["optimizer_state_bytes"] < before) == (policy == "drop")
    assert report["state_variant"] == ("reset_inactive_moments" if policy == "drop" else "retain_inactive_on_cpu")


def test_json_roundtrip_restores_subset_rng_and_flags_before_optimizer_load():
    model = _Model(tied_head=True)
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = LayerSampler(model, 2, 3, seed=19)
    sampler.on_step(0, optimizer)
    sampler.on_step(1, optimizer)
    saved = json.loads(json.dumps(sampler.state_dict()))
    expected = []
    for step in range(2, 15):
        sampler.on_step(step, optimizer)
        expected.append(sampler.selected_layers)
    restored_model = _Model(tied_head=True)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters())
    restored = LayerSampler(restored_model, 2, 3, seed=19)
    restored.load_state_dict(saved)
    assert restored.selected_layers == tuple(saved["selected_layers"])
    assert restored_model.head.weight is restored_model.embedding.weight
    for index, layer in enumerate(restored_model.model.layers):
        assert all(parameter.requires_grad == (index in restored.selected_layers) for parameter in layer.parameters())
    actual = []
    for step in range(2, 15):
        restored.on_step(step, restored_optimizer)
        state = restored.state_dict()
        restored.on_step(step, restored_optimizer)
        assert restored.state_dict() == state  # repeated boundary does not consume RNG
        actual.append(restored.selected_layers)
    assert actual == expected


def test_cross_block_and_external_sharing_is_rejected_without_toggling_flags():
    model = _Model()
    model.model.layers[1].weight = model.model.layers[0].weight
    flags = [parameter.requires_grad for parameter in model.parameters()]
    with pytest.raises(ValueError, match="shared across decoder"):
        LayerSampler(model, 1, 1)
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    model = _Model()
    model.outside_alias = model.model.layers[0].weight
    with pytest.raises(ValueError, match="outside modules"):
        LayerSampler(model, 1, 1)


@pytest.mark.parametrize("destination", ["other_decoder", "outside"])
@pytest.mark.parametrize("source_trainable,alias_trainable", [(True, True), (True, False), (False, True)])
def test_distinct_parameter_storage_aliases_across_domains_are_rejected(
    destination,
    source_trainable,
    alias_trainable,
):
    model = _Model()
    source = model.model.layers[0].weight
    source.requires_grad_(source_trainable)
    alias = nn.Parameter(source.detach(), requires_grad=alias_trainable)
    assert source is not alias
    assert source.untyped_storage().data_ptr() == alias.untyped_storage().data_ptr()
    if destination == "other_decoder":
        model.model.layers[1].weight = alias
    else:
        model.outside_alias = alias
    flags = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
    before = source.detach().clone()
    with pytest.raises(ValueError, match="shared parameter storage across decoder ownership domains"):
        LayerSampler(model, 1, 1)
    assert {id(parameter): parameter.requires_grad for parameter in model.parameters()} == flags
    torch.testing.assert_close(source, before, rtol=0, atol=0)


def test_storage_guard_checks_all_frozen_domains_before_a_trainable_alias():
    model = _Model()
    source = model.model.layers[0].weight
    source.requires_grad_(False)
    model.model.layers[1].weight = nn.Parameter(source.detach(), requires_grad=False)
    # The outside head is traversed after both frozen decoder aliases.
    model.head.weight = nn.Parameter(source.detach())
    with pytest.raises(ValueError, match="shared parameter storage"):
        LayerSampler(model, 1, 1)
    assert not source.requires_grad and not model.model.layers[1].weight.requires_grad
    assert model.head.weight.requires_grad


def test_initially_frozen_cross_domain_storage_remains_frozen():
    model = _Model()
    source = model.model.layers[0].weight
    source.requires_grad_(False)
    model.model.layers[1].weight = nn.Parameter(source.detach(), requires_grad=False)
    model.outside_alias = nn.Parameter(source.detach(), requires_grad=False)
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = LayerSampler(model, 1, 1)
    sampler.on_step(0, optimizer)
    assert not source.requires_grad
    assert not model.model.layers[1].weight.requires_grad
    assert not model.outside_alias.requires_grad


def test_same_block_storage_and_empty_parameters_keep_baseline_behavior():
    model = _Model()
    layer = model.model.layers[0]
    layer.storage_alias = nn.Parameter(layer.weight.detach())
    model.model.layers[0].empty = nn.Parameter(torch.empty(0))
    model.model.layers[1].empty = nn.Parameter(torch.empty(0))
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = LayerSampler(model, 1, 1)
    sampler.on_step(0, optimizer)
    assert layer.storage_alias.requires_grad == layer.weight.requires_grad
    assert layer.weight.requires_grad == (0 in sampler.selected_layers)


def test_explicit_path_and_complete_optimizer_candidates_are_required():
    model = _Model()
    model.blocks = model.model.layers
    del model.model.layers
    with pytest.raises(ValueError, match="layer_path"):
        LayerSampler(model, 1, 1)
    sampler = LayerSampler(model, 1, 1, layer_path="blocks")
    incomplete = torch.optim.AdamW(model.blocks[0].parameters())
    with pytest.raises(ValueError, match="all initially trainable"):
        sampler.on_step(0, incomplete)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_checkpoint_rejects_changed_initial_mask_and_invalid_selection():
    model = _Model()
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = LayerSampler(model, 1, 1)
    sampler.on_step(0, optimizer)
    state = sampler.state_dict()
    changed = _Model(frozen_base=True)
    with pytest.raises(ValueError, match="initial trainable"):
        LayerSampler(changed, 1, 1).load_state_dict(state)
    fresh = LayerSampler(_Model(), 1, 1)
    bad = copy.deepcopy(state)
    bad["selected_layers"] = [9]
    with pytest.raises(ValueError, match="invalid selected"):
        fresh.load_state_dict(bad)


@pytest.mark.parametrize("policy", ["drop", "cpu"])
def test_resumed_training_matches_uninterrupted_with_tied_parameters(policy):
    torch.manual_seed(5)
    initial = _Model(tied_head=True).state_dict()

    def build():
        model = _Model(tied_head=True)
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        sampler = LayerSampler(model, 2, 2, state_policy=policy, seed=17)
        return model, optimizer, sampler

    def train(parts, start, end):
        model, optimizer, sampler = parts
        for step in range(start, end):
            optimizer.zero_grad(set_to_none=True)
            sampler.on_step(step, optimizer)
            model(torch.tensor([[1, 2, 3]])).square().mean().backward()
            optimizer.step()

    continuous = build()
    train(continuous, 0, 7)
    first = build()
    train(first, 0, 3)
    model_state = copy.deepcopy(first[0].state_dict())
    optimizer_state = copy.deepcopy(first[1].state_dict())
    sampler_state = json.loads(json.dumps(first[2].state_dict()))
    resumed = build()
    resumed[0].load_state_dict(model_state)
    resumed[2].load_state_dict(sampler_state)  # Flags are restored before loading optimizer state.
    resumed[1].load_state_dict(optimizer_state)
    train(resumed, 3, 7)
    assert resumed[0].head.weight is resumed[0].embedding.weight
    assert resumed[2].state_dict() == continuous[2].state_dict()
    for key, value in continuous[0].state_dict().items():
        torch.testing.assert_close(resumed[0].state_dict()[key], value, rtol=0, atol=0)
    for actual_parameter, expected_parameter in zip(resumed[0].parameters(), continuous[0].parameters(), strict=True):
        actual, expected = resumed[1].state.get(actual_parameter), continuous[1].state.get(expected_parameter)
        if expected is None:
            assert actual is None
        else:
            assert actual.keys() == expected.keys()
            for key in actual:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device to verify actual state transfer")
def test_cpu_policy_moves_real_cuda_state_to_host():
    model = _Model().cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.tensor([[1, 2]], device="cuda")).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sampler = LayerSampler(model, 1, 1, state_policy="cpu")
    sampler.on_step(0, optimizer)
    for index, layer in enumerate(model.model.layers):
        if index not in sampler.selected_layers:
            assert all(value.device.type == "cpu" for p in layer.parameters() for value in optimizer.state[p].values())
