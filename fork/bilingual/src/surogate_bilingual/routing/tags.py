"""Plans from in-band control tags, computed over the FULL token history (MA-09), and the escaping of user text.

A tag id switches the route from its own position onward (the v1.0 convention). v1.0 `generate` recomputed the
routes on the truncated window `ids[-max_seq:]`, so once the last tag scrolled out of the window every token fell
back to the default route. Here the plan is built once over the whole history and callers slice the window out of
it with `RoutePlan.slice`.

`escape_user_text` encodes user text so that a literal "<|en|>" in it is ordinary text, never the special id
(IMPL-03); it needs a `tokenizers.Tokenizer` (tokenizers >= 0.20, `encode_special_tokens`) and delegates to
`data.tokenizer.encode_user_text`, which the shard writer uses too.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ..data.tokenizer import encode_user_text
from .plan import PlanError, RoutePlan

# The control tags of the v1.0 tokenizer and the expert each one selects.
TAG_ROUTES = {"<|zh|>": "zh", "<|en|>": "en", "<|bridge|>": "bridge"}


def tag_ids(tok) -> dict[int, str]:
    """{special id: expert name} for the control tags a tokenizer knows."""
    out = {}
    for text, name in TAG_ROUTES.items():
        token_id = tok.token_to_id(text)
        if token_id is not None:
            out[int(token_id)] = name
    return out


def plan_from_tags(ids: Sequence[int], tag_to_name: Mapping[int, str], names: Sequence[str], default: str) -> RoutePlan:
    """A [1, len(ids), E] one-hot plan over the full id list; a tag selects its expert from its own position on."""
    names = tuple(names)
    if default not in names:
        raise PlanError(f"default route {default!r} is not one of {names}")
    unknown = sorted(set(tag_to_name.values()) - set(names))
    if unknown:
        raise PlanError(f"tags select {unknown}, which are not in {names}")
    current = names.index(default)
    lookup = {int(k): names.index(v) for k, v in tag_to_name.items()}
    routes = []
    for token in ids:
        current = lookup.get(int(token), current)
        routes.append(current)
    return RoutePlan.one_hot(torch.tensor([routes], dtype=torch.long).reshape(1, -1), names, f"tags(default={default})")


def escape_user_text(tok, text: str) -> list[int]:
    """Token ids of user text with every special-token string encoded as plain text. The one implementation is
    `data.tokenizer.encode_user_text`; this name is the routing-side entry point of the API."""
    return encode_user_text(tok, text)
