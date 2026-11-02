"""The Bridge layout of parallel documents (MS §8.4; B-O13; IMPL-13), used by `shard --bridge`.

A parallel document holds two text segments, the source first, in different languages. In the Bridge layout:

- the source segment keeps its language as its route and has loss 0: under variant (c) it runs on the frozen
  source-language expert and is never a target;
- the target segment takes the route "bridge" and loss 1: it runs on the trained expert (the Bridge, a T3' third
  expert, or a T1' copy of the target-language expert, which maps the "bridge" route to itself) and is the only
  span the loss covers. `<eos>` inherits the target segment's route and loss flag.

The direction is (source language)>(target language). T1' trains one direction per run, so `select_direction`
keeps one; T3' needs the same sentences with the pairs broken, which `unpair` does within each direction with a
seeded permutation that leaves no pair in place.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .docs import Doc, DocError, Segment

DIRECTIONS = ("zh>en", "en>zh")
BRIDGE_ROUTE = "bridge"


def _texts(doc: Doc) -> tuple[Segment, Segment]:
    segs = [s for s in doc.segments if s.text is not None]
    if len(segs) != 2 or segs[0].lang is None or segs[1].lang is None or segs[0].lang == segs[1].lang:
        raise DocError(f"{doc.key}: a parallel document needs two text segments in two languages, source first")
    return segs[0], segs[1]


def direction(doc: Doc) -> str:
    source, target = _texts(doc)
    return f"{source.lang}>{target.lang}"


def bridge_layout(doc: Doc) -> Doc:
    source, target = _texts(doc)
    return doc.replace(
        segments=(
            Segment(text=source.text, lang=source.lang, role="source", route=source.lang, loss=False),
            Segment(text=target.text, lang=target.lang, role="target", route=BRIDGE_ROUTE, loss=True),
        ),
        route=None,
    )


def select_direction(docs: Iterable[Doc], wanted: str) -> list[Doc]:
    if wanted not in DIRECTIONS:
        raise DocError(f"direction must be one of {DIRECTIONS}: {wanted!r}")
    return [d for d in docs if direction(d) == wanted]


def unpair(docs: Sequence[Doc], seed: int) -> list[Doc]:
    """The same source and target sentences with every pair broken (T3'): within each direction, target i goes to
    source perm[(k + 1) % n] where perm is a seeded permutation and source i = perm[k], so no target stays with
    its own source (n >= 2). Keys and pair ids get an "unpaired" mark; documents keep their order."""
    out = list(docs)
    by_direction: dict[str, list[int]] = {}
    for i, doc in enumerate(out):
        by_direction.setdefault(direction(doc), []).append(i)
    rng = np.random.default_rng([int(seed), 0x7533])
    for key in sorted(by_direction):
        idx = by_direction[key]
        if len(idx) < 2:
            raise DocError(f"unpair needs at least two {key} documents")
        perm = [idx[int(j)] for j in rng.permutation(len(idx))]
        targets = {perm[(k + 1) % len(perm)]: _texts(out[perm[k]])[1] for k in range(len(perm))}
        for i in idx:
            source, _ = _texts(out[i])
            out[i] = out[i].replace(
                key=f"{out[i].key}#unpaired",
                pair_id=None,
                segments=(source, targets[i]),
                meta={**out[i].meta, "unpaired_seed": int(seed)},
            )
    return out
