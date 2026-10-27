"""Experimental layer sampling; drop policy resets inactive optimizer moments."""

from __future__ import annotations

import random

import torch
from torch import nn

_LAYER_PATHS = (
    "model.layers",
    "model.decoder.layers",
    "transformer.h",
    "transformer.blocks",
    "gpt_neox.layers",
    "base_model.model.model.layers",
    "base_model.model.model.decoder.layers",
    "base_model.model.transformer.h",
    "base_model.model.gpt_neox.layers",
)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value if value.device.type == "cpu" else value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def _state_bytes(value, seen, counts):
    if isinstance(value, torch.Tensor):
        if id(value) not in seen:
            seen.add(id(value))
            counts[value.device.type] = counts.get(value.device.type, 0) + value.numel() * value.element_size()
    elif isinstance(value, dict):
        for item in value.values():
            _state_bytes(item, seen, counts)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _state_bytes(item, seen, counts)


class LayerSampler:
    """Sample trainable decoder blocks only at optimizer-step boundaries.

    Initially frozen parameters (including PEFT base weights) stay frozen;
    trainable parameters outside decoder blocks stay enabled. Shared parameters
    crossing decoder-block boundaries, or shared with an outside module, are
    rejected because requires_grad cannot express separate ownership. Distinct
    parameters sharing storage across those domains are also rejected when
    either is initially trainable; freezing an alias would not prevent writes
    through its trainable counterpart. Storage sharing within one domain, or
    between initially frozen parameters only, preserves its existing semantics.

    ``drop`` is a reset variant: inactive blocks lose optimizer history.
    ``cpu`` retains history on CPU; the optimizer must restore active state
    lazily (the experimental rank-scaled optimizer supports this).
    """

    def __init__(self, model, active_layers, interval, state_policy="drop", layer_path=None, seed=42):
        if not _integer(active_layers) or active_layers < 1:
            raise ValueError("active_layers must be a positive integer")
        if not _integer(interval) or interval < 1:
            raise ValueError("interval must be a positive integer")
        if state_policy not in ("drop", "cpu"):
            raise ValueError("state_policy must be 'drop' or 'cpu'")
        if not _integer(seed):
            raise ValueError("seed must be an integer")
        self.model = model
        self.active_layers = active_layers
        self.interval = interval
        self.state_policy = state_policy
        self.seed = seed
        self._rng = random.Random(seed)
        self.layer_path, self.layers = self._discover(model, layer_path)
        if active_layers > len(self.layers):
            raise ValueError("active_layers exceeds the number of decoder layers")
        self._parameters = list(model.parameters())
        self._initial_trainable = {id(parameter): parameter.requires_grad for parameter in self._parameters}
        self._owners = {}
        for index, layer in enumerate(self.layers):
            for parameter in layer.parameters():
                previous = self._owners.setdefault(id(parameter), index)
                if previous != index:
                    raise ValueError("parameters shared across decoder blocks are unsupported for layer sampling")
        names = list(model.named_parameters(remove_duplicate=False))
        for name, parameter in names:
            owner = self._owners.get(id(parameter))
            if owner is not None and not name.startswith(f"{self.layer_path}.{owner}."):
                raise ValueError("parameters shared between decoder blocks and outside modules are unsupported")
        self._check_storage_aliases()
        self._signature = [
            [name, list(parameter.shape), str(parameter.dtype), self._initial_trainable[id(parameter)]]
            for name, parameter in model.named_parameters()
        ]
        if not any(self._initial_trainable[key] for key in self._owners):
            raise ValueError("decoder layers contain no initially trainable parameters")
        self.selected_layers = ()
        self._last_step = None
        self._last_selection_step = None

    def _check_storage_aliases(self):
        storages = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.numel() or parameter.is_meta:
                continue
            storage = (parameter.device, parameter.untyped_storage().data_ptr())
            owner = self._owners.get(id(parameter))
            trainable = self._initial_trainable[id(parameter)]
            aliases = storages.setdefault(storage, [])
            for previous_name, previous_owner, previous_trainable in aliases:
                if owner != previous_owner and (trainable or previous_trainable):
                    raise ValueError(
                        "layer sampling does not support shared parameter storage across decoder ownership domains: "
                        f"{previous_name}, {name}"
                    )
            aliases.append((name, owner, trainable))

    @staticmethod
    def _discover(model, requested):
        for path in (requested,) if requested is not None else _LAYER_PATHS:
            try:
                module = model.get_submodule(path)
            except (AttributeError, KeyError):
                continue
            if isinstance(module, nn.ModuleList) and len(module):
                # PEFT wrappers may resolve a delegated shortcut that differs
                # from the registered prefix used by named_parameters().
                canonical = next(name for name, candidate in model.named_modules() if candidate is module)
                return canonical, module
        raise ValueError("cannot discover a nonempty decoder ModuleList; set layer_path explicitly")

    @property
    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self._parameters if parameter.requires_grad)

    def _apply_selection(self):
        active = set(self.selected_layers)
        for parameter in self._parameters:
            owner = self._owners.get(id(parameter))
            if owner is not None:
                enabled = self._initial_trainable[id(parameter)] and (
                    self._last_selection_step is None or owner in active
                )
                parameter.requires_grad_(enabled)
                if not enabled:
                    parameter.grad = None

    def optimizer_state_bytes(self, optimizer):
        counts = {}
        _state_bytes(optimizer.state, set(), counts)
        return sum(counts.values())

    def on_step(self, step, optimizer):
        if not _integer(step) or step < 0 or (self._last_step is not None and step < self._last_step):
            raise ValueError("layer sampling steps must be nonnegative and nondecreasing")
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        if any(enabled and key not in optimized for key, enabled in self._initial_trainable.items()):
            raise ValueError("optimizer must include all initially trainable parameters before layer sampling")
        if self._last_selection_step is None or (step % self.interval == 0 and self._last_selection_step != step):
            self.selected_layers = tuple(sorted(self._rng.sample(range(len(self.layers)), self.active_layers)))
            self._last_selection_step = step
            self._apply_selection()
        self._last_step = step
        for parameter in list(optimizer.state):
            if id(parameter) in self._owners and not parameter.requires_grad:
                if self.state_policy == "drop":
                    del optimizer.state[parameter]
                else:
                    optimizer.state[parameter] = _cpu_state(optimizer.state[parameter])
        counts = {}
        _state_bytes(optimizer.state, set(), counts)
        return {
            "selected_layers": list(self.selected_layers),
            "trainable_parameter_count": self.trainable_parameter_count,
            "optimizer_state_bytes": sum(counts.values()),
            "optimizer_state_cpu_bytes": counts.get("cpu", 0),
            "optimizer_state_cuda_bytes": counts.get("cuda", 0),
            "state_policy": self.state_policy,
            "state_variant": "reset_inactive_moments" if self.state_policy == "drop" else "retain_inactive_on_cpu",
        }

    def _contract(self):
        return {
            "layer_path": self.layer_path,
            "num_layers": len(self.layers),
            "active_layers": self.active_layers,
            "interval": self.interval,
            "state_policy": self.state_policy,
            "seed": self.seed,
            "parameters": [[name, list(shape), dtype, trainable] for name, shape, dtype, trainable in self._signature],
        }

    def state_dict(self):
        version, internal, gauss = self._rng.getstate()
        return {
            "version": 1,
            **self._contract(),
            "selected_layers": list(self.selected_layers),
            "last_step": self._last_step,
            "last_selection_step": self._last_selection_step,
            "rng_state": [version, list(internal), gauss],
        }

    def load_state_dict(self, state):
        if state.get("version") != 1 or any(state.get(key) != value for key, value in self._contract().items()):
            raise ValueError(
                "layer sampling checkpoint does not match model/initial trainable parameters/configuration"
            )
        selected = state.get("selected_layers")
        last, selection_step = state.get("last_step"), state.get("last_selection_step")
        if (
            not isinstance(selected, list)
            or any(not _integer(index) or not 0 <= index < len(self.layers) for index in selected)
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("invalid selected layers in layer sampling checkpoint")
        if selection_step is None:
            valid = last is None and not selected
        else:
            valid = (
                _integer(last)
                and _integer(selection_step)
                and 0 <= selection_step <= last
                and len(selected) == self.active_layers
            )
        if not valid:
            raise ValueError("invalid layer sampling checkpoint step/selection")
        try:
            version, internal, gauss = state["rng_state"]
            restored_rng = random.Random()
            restored_rng.setstate((version, tuple(internal), gauss))
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError("invalid layer sampling RNG state") from error
        self._rng = restored_rng
        self.selected_layers = tuple(sorted(selected))
        self._last_step, self._last_selection_step = last, selection_step
        self._apply_selection()
