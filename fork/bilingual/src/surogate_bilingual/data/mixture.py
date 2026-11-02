"""The D_seed mixture (MS §6.1, B-O7; SCI-02): natural text, Track S, code-switched text with concatenated parallel
pairs, and tagged documents, in declared token shares, shuffled by document.

v1.0 packed all of zh.txt before all of en.txt, so a seed window held both languages with probability 2.6e-7,
and its control tags never occurred in the seed shard (SCI-02, IMPL-03). Here:

- The Track S component is taken whole: its exposures per fact are preregistered, so it is never subsampled. The
  total is T = tokens(Track S) / share(track_s), and every other component is drawn to its share of T.
- Natural text is balanced by tokens between zh and en (MS §6.1).
- `codeswitch_parallel` is filled by concatenated parallel pairs ("source + translation" as one document, two
  segments) for `seed_mix_parallel_fraction` of its tokens and by code-switched documents for the rest.
  Code-switched text without language marks is segmented by script (`docs.script_segments`).
- Tagged documents (B-O3): each opens with the tag of its first language and carries the tag of the new language
  at every switch; the tags are ordinary input and target tokens. They are drawn from code-switched documents
  first, then natural documents of either language alternately. `<|bridge|>` is left out unless
  `seed_tags.bridge` is set; then remaining parallel pairs are tagged [<|src|>, source, <|bridge|>, target], and
  those pairs count against the Bridge's parallel-data budget (MS §6.4).
- Every pool document is used at most once, components are drawn greedily from a seeded shuffle until they reach
  their target (overshoot below one document), and the final list is shuffled by document.

The card (`bilingual-datacard/1`, kind `d_seed`) records the declared and achieved token shares, the counts, the
declared minimum count of each tag (handed to the shard writer, which refuses a shard below it) and the sources
whose licence is not verified.
"""

from __future__ import annotations

import zlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ..config import DEFAULTS
from .docs import TAG_OF_LANG, Doc, Segment, script_segments

COMPONENTS = ("natural", "track_s", "codeswitch_parallel", "tagged")


class MixtureError(ValueError):
    pass


def doc_tokens(doc: Doc, tok) -> int:
    from .tokenizer import encode_user_text

    return sum(1 if s.tag is not None else len(encode_user_text(tok, s.text)) for s in doc.segments) + 1  # + <eos>


def pools_from_docs(docs: Iterable[Doc]) -> dict[str, list[Doc]]:
    pools: dict[str, list[Doc]] = {"zh": [], "en": [], "codeswitch": [], "parallel": [], "track_s": []}
    for doc in docs:
        pools["track_s" if doc.kind == "synthetic" else doc.kind].append(doc)
    return pools


def _langs_of(doc: Doc) -> list[Segment]:
    """Text segments with languages: code-switched text without marks is split by script."""
    if doc.kind == "codeswitch" and all(s.lang is None for s in doc.segments if s.text):
        return [seg for s in doc.segments if s.text for seg in script_segments(s.text)]
    return [s for s in doc.segments if s.text]


def concat_parallel(doc: Doc) -> Doc:
    segs = [s for s in doc.segments if s.text]
    if len(segs) != 2 or {s.lang for s in segs} != {"zh", "en"}:
        raise MixtureError(f"{doc.key}: a parallel document needs one zh and one en text segment")
    first = Segment(text=segs[0].text + "\n", lang=segs[0].lang, role="source")
    second = Segment(text=segs[1].text, lang=segs[1].lang, role="target")
    return doc.replace(segments=(first, second), meta={**doc.meta, "mix": "parallel_concat"})


def tag_doc(doc: Doc, *, start: bool, switch: bool, bridge: bool = False) -> Doc:
    segs = _langs_of(doc)
    out: list[Segment] = []
    previous = None
    for i, seg in enumerate(segs):
        if bridge and doc.kind == "parallel" and i == 1:
            out.append(Segment(tag="<|bridge|>"))
        elif seg.lang is not None and (
            (i == 0 and start) or (switch and previous is not None and seg.lang != previous)
        ):
            out.append(Segment(tag=TAG_OF_LANG[seg.lang]))
        out.append(seg)
        previous = seg.lang if seg.lang is not None else previous
    return doc.replace(segments=tuple(out), meta={**doc.meta, "mix": "tagged"})


def _draw(pool: Sequence[Doc], target: float, tok, rng, used: set, name: str, transform=None):
    chosen, tokens = [], 0
    for i in rng.permutation(len(pool)):
        if tokens >= target:
            break
        doc = pool[int(i)]
        if doc.key in used:
            continue
        doc2 = transform(doc) if transform else doc
        used.add(doc.key)
        chosen.append(doc2)
        tokens += doc_tokens(doc2, tok)
    if tokens < target:
        raise MixtureError(f"pool {name!r} holds {tokens} usable tokens; its share needs {target:.0f}")
    return chosen, tokens


def build_seed_mixture(
    pools: Mapping[str, Sequence[Doc]],
    tok,
    *,
    seed: int,
    shares: Mapping[str, float] | None = None,
    parallel_fraction: float | None = None,
    tags: Mapping[str, bool] | None = None,
    sources: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[Doc], dict[str, Any]]:
    data = DEFAULTS["data"]
    shares = dict(shares or data["seed_mix"])
    if set(shares) != set(COMPONENTS) or abs(sum(shares.values()) - 1.0) > 1e-9 or min(shares.values()) <= 0:
        raise MixtureError(f"seed_mix must give positive shares of {COMPONENTS} summing to 1: {shares}")
    parallel_fraction = data["seed_mix_parallel_fraction"] if parallel_fraction is None else parallel_fraction
    tags = {**data["seed_tags"], **(tags or {})}
    rng = lambda name: np.random.default_rng([int(seed), zlib.crc32(name.encode())])  # noqa: E731
    track_s = list(pools.get("track_s", []))
    t_s = sum(doc_tokens(d, tok) for d in track_s)
    if t_s == 0:
        raise MixtureError("the Track S component is empty; D_seed is sized from it")
    total = t_s / shares["track_s"]
    target = {c: shares[c] * total for c in COMPONENTS}
    used: set[str] = set()
    parts: dict[str, list[Doc]] = {"track_s": track_s}
    tokens: Counter = Counter({"track_s": t_s})
    natural: list[Doc] = []
    for lang in ("zh", "en"):
        docs, n = _draw(pools.get(lang, []), target["natural"] / 2, tok, rng(f"natural/{lang}"), used, lang)
        natural += docs
        tokens[f"natural/{lang}"] = n
    parts["natural"] = natural
    tokens["natural"] = tokens["natural/zh"] + tokens["natural/en"]
    par_target = target["codeswitch_parallel"] * parallel_fraction
    par_docs, n_par = _draw(pools.get("parallel", []), par_target, tok, rng("parallel"), used, "parallel",
                            concat_parallel)  # fmt: skip
    cs_docs, n_cs = _draw(pools.get("codeswitch", []), target["codeswitch_parallel"] - n_par, tok, rng("codeswitch"),
                          used, "codeswitch", lambda d: d.replace(segments=tuple(_langs_of(d))))  # fmt: skip
    parts["codeswitch_parallel"] = par_docs + cs_docs
    tokens["codeswitch_parallel"] = n_par + n_cs
    tokens["codeswitch_parallel/parallel"], tokens["codeswitch_parallel/codeswitch"] = n_par, n_cs
    # tagged: code-switched first, then natural alternating by language, then (bridge) parallel
    order = rng("tagged")
    tag_pool = [d for d in pools.get("codeswitch", []) if d.key not in used]
    zh_left = [pools["zh"][int(i)] for i in order.permutation(len(pools.get("zh", [])))]
    en_left = [pools["en"][int(i)] for i in order.permutation(len(pools.get("en", [])))]
    zh_left = [d for d in zh_left if d.key not in used]
    en_left = [d for d in en_left if d.key not in used]
    for i in range(max(len(zh_left), len(en_left))):
        tag_pool += [x[i] for x in (zh_left, en_left) if i < len(x)]
    if tags["bridge"]:
        tag_pool = [d for d in pools.get("parallel", []) if d.key not in used] + tag_pool
    tagged, n_tag = [], 0
    for doc in tag_pool:
        if n_tag >= target["tagged"]:
            break
        t = tag_doc(doc, start=tags["start"], switch=tags["switch"], bridge=tags["bridge"])
        used.add(doc.key)
        tagged.append(t)
        n_tag += doc_tokens(t, tok)
    if n_tag < target["tagged"]:
        raise MixtureError(f"not enough documents left to tag: {n_tag} of {target['tagged']:.0f} tokens")
    parts["tagged"] = tagged
    tokens["tagged"] = n_tag
    tag_min = Counter()
    for doc in tagged:
        for seg in doc.segments:
            if seg.tag is not None:
                tag_min[seg.tag] += 1
    docs = [d for c in COMPONENTS for d in parts[c]]
    docs = [docs[int(i)] for i in rng("shuffle").permutation(len(docs))]
    achieved_total = sum(tokens[c] for c in COMPONENTS)
    achieved = {c: tokens[c] / achieved_total for c in COMPONENTS}
    card = {
        "format": "bilingual-datacard/1",
        "kind": "d_seed",
        "seed": int(seed),
        "declared_shares": shares,
        "achieved_shares": achieved,
        "max_share_error": max(abs(achieved[c] - shares[c]) for c in COMPONENTS),
        "tokens": dict(sorted(tokens.items())),
        "documents": {c: len(parts[c]) for c in COMPONENTS},
        "natural_balance": {"zh": tokens["natural/zh"], "en": tokens["natural/en"]},
        "parallel_fraction": parallel_fraction,
        "tags": dict(tags),
        "min_tag_counts": {t: tag_min.get(t, 0) for t in ("<|zh|>", "<|en|>", "<|bridge|>")},
        "bridge_parallel_documents": sum(1 for d in tagged if d.kind == "parallel"),
        "unverified_licences": sorted(s["name"] for s in sources if not s.get("licence_verified", False)),
    }
    return docs, card
