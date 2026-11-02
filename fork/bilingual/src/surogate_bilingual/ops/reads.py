"""How an operator that reads only part of a parameter tells the parameter audit how many elements it read (IMPL-21).

`audit.params.p_active` records every parameter passed to a torch function. An operator that slices a parameter
(`ops.fusion.channel_select` gathers rows and columns of an expert) passes the whole Parameter into the slicing call,
so the audit would count all of it; the operator therefore calls `declare_partial_read(param, n)` with the number of
elements it actually reads. The audit installs a sink with `partial_read_sink` for the length of one forward; outside
an audit a declaration does nothing.

The hook lives in `ops` so that operators depend on torch only and never import `audit` (docs/spec.md, "模块边界").
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

Sink = Callable[[torch.Tensor, int], None]

_SINK: contextvars.ContextVar[Sink | None] = contextvars.ContextVar("bilingual_partial_reads", default=None)


def declare_partial_read(param: torch.Tensor, n_elements: int) -> None:
    """Called by an operator that reads only `n_elements` of `param` (no-op outside an audit)."""
    sink = _SINK.get()
    if sink is not None:
        sink(param, int(n_elements))


@contextmanager
def partial_read_sink(sink: Sink) -> Iterator[None]:
    """Route every `declare_partial_read` in this context to `sink` (used by the audit)."""
    token = _SINK.set(sink)
    try:
        yield
    finally:
        _SINK.reset(token)
