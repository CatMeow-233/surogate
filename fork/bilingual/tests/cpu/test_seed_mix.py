"""I4-D8: the D_seed mixture (SCI-02, B-O7)."""

import json

import pytest

from surogate_bilingual.config import DEFAULTS
from surogate_bilingual.data import shards as SH
from surogate_bilingual.data.docs import Doc, Segment, iter_file_docs
from surogate_bilingual.data.mixture import (
    MixtureError,
    build_seed_mixture,
    doc_tokens,
    pools_from_docs,
    tag_doc,
)


@pytest.fixture(scope="module")
def mixture(seed_mixture):
    return seed_mixture


def test_i4_d8_mixture_shares_match_the_declared_seed_mix(mixture, data_pipeline):
    pools, docs, card = mixture
    declared = DEFAULTS["data"]["seed_mix"]
    assert card["declared_shares"] == declared
    assert card["max_share_error"] <= 0.005
    for name, share in declared.items():
        assert abs(card["achieved_shares"][name] - share) <= 0.005, name
    # recount independently from the returned documents
    tok = data_pipeline["tok"]
    track_keys = {d.key for d in pools["track_s"]}
    totals = {"natural": 0, "track_s": 0, "codeswitch_parallel": 0, "tagged": 0}
    for d in docs:
        n = doc_tokens(d, tok)
        if d.key in track_keys:
            totals["track_s"] += n
        elif d.meta.get("mix") == "tagged":
            totals["tagged"] += n
        elif d.meta.get("mix") == "parallel_concat" or d.kind == "codeswitch":
            totals["codeswitch_parallel"] += n
        else:
            totals["natural"] += n
    total = sum(totals.values())
    for name, share in declared.items():
        assert abs(totals[name] / total - share) <= 0.005, name
    assert {d.key for d in docs} >= track_keys  # Track S is taken whole (exposures stay exact)
    assert len({d.key for d in docs}) == len(docs)  # no pool document is used twice
    nb = card["natural_balance"]
    assert abs(nb["zh"] - nb["en"]) / (nb["zh"] + nb["en"]) < 0.01  # balanced by tokens
    assert card["unverified_licences"] == ["cs", "par", "zh-nat"]  # licence_verified = false is surfaced


def test_i4_d8_codeswitched_and_concatenated_parallel_documents_are_present(mixture):
    _, docs, card = mixture
    concat = [d for d in docs if d.meta.get("mix") == "parallel_concat"]
    switched = [d for d in docs if d.kind == "codeswitch" and d.meta.get("mix") != "tagged"]
    assert concat and switched
    for d in concat:
        assert [s.lang for s in d.segments] in (["zh", "en"], ["en", "zh"]) and d.segments[0].text.endswith("\n")
    for d in switched:
        assert {"zh", "en"} <= {s.lang for s in d.segments}  # split by script into zh and en runs
    tagged = [d for d in docs if d.meta.get("mix") == "tagged"]
    assert tagged and all(d.segments[0].tag in ("<|zh|>", "<|en|>") for d in tagged)
    assert card["documents"]["codeswitch_parallel"] == len(concat) + len(switched)
    # documents are shuffled: the components interleave
    kinds = [d.meta.get("mix", d.kind) for d in docs[:200]]
    assert len(set(kinds)) >= 3


def test_i4_d8_tags_open_documents_and_mark_every_switch():
    doc = Doc("x", "codeswitch", (Segment(text="我们今天 went to the park 然后回家"),))
    tagged = tag_doc(doc, start=True, switch=True)
    shape = [(s.tag, s.lang) for s in tagged.segments]
    assert shape == [("<|zh|>", None), (None, "zh"), ("<|en|>", None), (None, "en"), ("<|zh|>", None), (None, "zh")]
    assert [s.tag for s in tag_doc(doc, start=True, switch=False).segments if s.tag] == ["<|zh|>"]


def test_i4_d8_mixed_window_fraction_exceeds_the_declared_floor(mixture, data_pipeline, tmp_path):
    _, docs, _ = mixture
    tok = data_pipeline["tok"]
    floor = DEFAULTS["data"]["min_mixed_window_fraction"]
    seq = 64
    path = SH.write_shard(
        tmp_path / "seed",
        docs,
        tok,
        ["zh", "en", "bridge", "seed"],
        "seed_train",
        None,
        tokenizer_sha256=data_pipeline["tokenizer_sha256"],
        default_route="seed",
        seq=seq,
        min_mixed_window_fraction=floor,
    )
    mixed = json.loads((path / "shard.json").read_text())["mixed_window_fraction"]
    assert mixed["value"] > floor and mixed["floor"] == floor
    # the v1.0 packing of the same documents (all zh, then all en, then the rest) is far below the floor
    v10 = sorted(docs, key=lambda d: {"zh": 0, "en": 1}.get(d.kind, 2))
    v10 = [d for d in v10 if d.kind in ("zh", "en")]
    with pytest.raises(SH.ShardError, match="below the declared floor"):
        SH.write_shard(
            tmp_path / "v10",
            v10,
            tok,
            ["seed"],
            "seed_train",
            None,
            tokenizer_sha256="x",
            default_route="seed",
            seq=seq,
            min_mixed_window_fraction=floor,
        )
    path = SH.write_shard(tmp_path / "v10b", v10, tok, ["seed"], "seed_train", None, tokenizer_sha256="x", seq=seq)
    v10_value = json.loads((path / "shard.json").read_text())["mixed_window_fraction"]["value"]
    assert v10_value < floor / 2 and mixed["value"] > 10 * floor
    n = json.loads((path / "shard.json").read_text())["n_tokens"]
    assert v10_value <= (seq - 1) / (n - seq + 1) + 1e-12  # only the windows across the single zh/en seam


def test_i4_d8_a_short_pool_is_refused(data_pipeline, track_s_ci):
    pools = pools_from_docs(data_pipeline["derived"].iter_docs(split="seed_train"))
    pools["track_s"] = list(iter_file_docs(track_s_ci[0] / "seed" / "track_s.jsonl", source=None, kind=None))
    with pytest.raises(MixtureError, match="usable tokens"):
        build_seed_mixture(pools, data_pipeline["tok"], seed=0)
    with pytest.raises(MixtureError, match="positive shares"):
        build_seed_mixture(pools, data_pipeline["tok"], seed=0, shares={"natural": 1.0})
