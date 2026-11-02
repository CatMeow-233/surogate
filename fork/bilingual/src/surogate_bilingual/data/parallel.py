"""The Bridge layout of parallel documents (MS §8.4; B-O13; IMPL-13), used by `shard --bridge`.

A parallel document holds a source and a target text segment in different languages, the source first; a Track S
document of the E4 translate format also holds an instruction segment between them, in the target language
(data.synth.build). In the Bridge layout:

- the source segment keeps its language as its route and has loss 0: under variant (c) it runs on the frozen
  source-language expert and is never a target;
- the instruction segment (if any) is in the target language and belongs to the target side: it takes the route
  "bridge" with loss 0. At evaluation the answer route of T2/T3' covers the instruction and answer segments of
  translation items, and for T1' R_out sends the target-language instruction to the target slot, which holds the
  copy (eval.items), so training and evaluation agree for every T arm. A source-language token is never on
  "bridge", so a T1' run still refuses a window of the other direction;
- the target segment takes the route "bridge" and loss 1: it runs on the trained expert (the Bridge, a T3' third
  expert, or a T1' copy of the target-language expert, which maps the "bridge" route to itself) and is the only
  span the loss covers. `<eos>` inherits the target segment's route and loss flag.

The direction is (source language)>(target language). T1' trains one direction per run, so `select_direction`
keeps one; T3' needs the same sentences with the pairs broken, which `unpair` does within each direction with a
seeded permutation that leaves no pair in place (the instruction stays with its source).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .docs import Doc, DocError, Segment

DIRECTIONS = ("zh>en", "en>zh")
BRIDGE_ROUTE = "bridge"


def _parts(doc: Doc) -> tuple[Segment, Segment | None, Segment]:
    """(source, instruction or None, target) of a parallel document."""
    segs = [s for s in doc.segments if s.text is not None]
    if len(segs) == 3 and segs[1].role == "instruction":
        source, instruction, target = segs
        if instruction.lang != target.lang:
            raise DocError(f"{doc.key}: the instruction of a parallel document is in the target language")
    elif len(segs) == 2:
        (source, target), instruction = segs, None
    else:
        raise DocError(
            f"{doc.key}: a parallel document holds a source and a target text segment (an instruction between them"
            " is allowed), source first"
        )
    if source.lang is None or target.lang is None or source.lang == target.lang:
        raise DocError(f"{doc.key}: the source and target of a parallel document are in two languages")
    return source, instruction, target


def direction(doc: Doc) -> str:
    source, _, target = _parts(doc)
    return f"{source.lang}>{target.lang}"


def bridge_layout(doc: Doc) -> Doc:
    source, instruction, target = _parts(doc)
    segs = [Segment(text=source.text, lang=source.lang, role="source", route=source.lang, loss=False)]
    if instruction is not None:
        segs.append(
            Segment(text=instruction.text, lang=instruction.lang, role="instruction", route=BRIDGE_ROUTE, loss=False)
        )
    segs.append(Segment(text=target.text, lang=target.lang, role="target", route=BRIDGE_ROUTE, loss=True))
    return doc.replace(segments=tuple(segs), route=None)


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
        targets = {perm[(k + 1) % len(perm)]: _parts(out[perm[k]])[2] for k in range(len(perm))}
        for i in idx:
            source, instruction, _ = _parts(out[i])
            out[i] = out[i].replace(
                key=f"{out[i].key}#unpaired",
                pair_id=None,
                segments=tuple(s for s in (source, instruction, targets[i]) if s is not None),
                meta={**out[i].meta, "unpaired_seed": int(seed)},
            )
    return out
