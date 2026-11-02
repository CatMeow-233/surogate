"""I4-D3: document-level splits and clusters (SCI-10)."""

import json

import numpy as np
import pytest

from surogate_bilingual.data import split as S
from surogate_bilingual.data.docs import Doc, Segment


def _doc(key, text, kind="zh", **kw):
    return Doc(key, kind, (Segment(text=text, lang=kind if kind in ("zh", "en") else None),), **kw)


def test_i4_d3_same_seed_gives_a_byte_identical_split_file(data_pipeline, tmp_path):
    m = data_pipeline["manifest"]
    a = S.split_doc(m.iter_docs(), S.default_fractions(), 1, m.sha256)
    b = S.split_doc(m.iter_docs(), S.default_fractions(), 1, m.sha256)
    S.write_split(tmp_path / "a.json", a)
    S.write_split(tmp_path / "b.json", b)
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert (tmp_path / "a.json").read_bytes() == (data_pipeline["base"] / "split.json").read_bytes()
    c = S.split_doc(m.iter_docs(), S.default_fractions(), 2, m.sha256)
    assert c["splits"] != a["splits"]
    assert a["format"] == "bilingual-split/1" and a["cluster_rule"] == S.CLUSTER_RULE and a["seed"] == 1


def test_i4_d3_near_duplicates_pairs_and_entities_stay_together():
    docs = []
    for i in range(300):
        docs.append(_doc(f"zh/{i}", f"第{i}篇文档的正文"))
    # near-duplicates: equal after NFKC, casefold and whitespace collapse (full-width letters, case, spacing)
    docs.append(_doc("en/a", "Hello   World 42", kind="en"))
    docs.append(_doc("en/b", "ＨＥＬＬＯ world\t４２", kind="en"))
    docs.append(_doc("en/c", "hello world 42 ", kind="en"))
    for i in range(40):
        docs += [
            _doc(f"par/{i}/zh", f"平行句{i}", kind="parallel", pair_id=f"p{i}"),
            _doc(f"par/{i}/en", f"parallel {i}", kind="parallel", pair_id=f"p{i}"),
        ]
        docs += [_doc(f"syn/{i}/{j}", f"entity {i} doc {j}", kind="synthetic", entity_id=f"E{i}") for j in range(3)]
    fractions = {"seed_train": 0.4, "zh_train": 0.1, "en_train": 0.1, "parallel_train": 0.1, "calib": 0.1,
                 "dev": 0.1, "test": 0.1}  # fmt: skip
    for seed in range(5):
        where = {k: name for name, keys in S.assign_splits(docs, fractions, seed).items() for k in keys}
        assert len({where["en/a"], where["en/b"], where["en/c"]}) == 1
        for i in range(40):
            assert where[f"par/{i}/zh"] == where[f"par/{i}/en"]
            assert len({where[f"syn/{i}/{j}"] for j in range(3)}) == 1
    # the clusters themselves
    groups = S.clusters(docs)
    keys = [{docs[i].key for i in g} for g in groups]
    assert {"en/a", "en/b", "en/c"} in keys and {"par/0/zh", "par/0/en"} in keys


def test_i4_d3_seven_sets_are_pairwise_disjoint(data_pipeline):
    splits = data_pipeline["split"]["splits"]
    assert list(splits) == list(S.SPLITS) and len(S.SPLITS) == 7
    sets = {k: set(v) for k, v in splits.items()}
    for a in S.SPLITS:
        for b in S.SPLITS:
            if a < b:
                assert not sets[a] & sets[b], (a, b)
    all_keys = [k for d in data_pipeline["manifest"].iter_docs() for k in [d.key]]
    assert sorted(all_keys) == sorted(k for v in splits.values() for k in v)  # every document exactly once
    with pytest.raises(S.SplitError, match="in both"):
        S.check_disjoint({"dev": ["x"], "test": ["x"]})
    # the materialized corpora hold exactly the split's documents, file by file
    derived = data_pipeline["derived"]
    for name in S.SPLITS:
        got = sorted(d.key for d in derived.iter_docs(split=name))
        assert got == sorted(splits[name]), name


def test_i4_d3_fractions_within_one_percent_on_10k_documents():
    rng = np.random.default_rng(3)
    docs = [_doc(f"d/{i}", f"document {i} " + " ".join(map(str, rng.integers(0, 1000, 5))), kind="en")
            for i in range(10_000)]  # fmt: skip
    fractions = {"seed_train": 0.55, "zh_train": 0.12, "en_train": 0.12, "parallel_train": 0.06, "calib": 0.03,
                 "dev": 0.06, "test": 0.06}  # fmt: skip
    splits = S.assign_splits(docs, fractions, seed=11)
    for name, target in fractions.items():
        assert abs(len(splits[name]) / len(docs) - target) <= 0.01, name
    # per-kind fractions: a zh document never lands in en_train
    mixed = docs[:2000] + [_doc(f"z/{i}", f"中文 {i}") for i in range(2000)]
    splits = S.assign_splits(mixed, S.default_fractions(), seed=1)
    assert not any(k.startswith("z/") for k in splits["en_train"])
    assert not any(k.startswith("d/") for k in splits["zh_train"])
    with pytest.raises(S.SplitError, match="sum to 1"):
        S.assign_splits(docs[:10], {"seed_train": 0.5, "dev": 0.4}, seed=0)


def test_i4_d3_materialize_drops_decontaminated_training_documents_only(data_pipeline, tmp_path):
    m, s = data_pipeline["manifest"], data_pipeline["split"]
    victim_train = s["splits"]["seed_train"][0]
    victim_dev = s["splits"]["dev"][0]
    derived = S.write_split_corpora(m, s, tmp_path / "out", exclude=[victim_train, victim_dev])
    from surogate_bilingual.data.manifest import load_manifest

    d = load_manifest(derived)
    keys = {doc.key for doc in d.iter_docs()}
    assert victim_train not in keys and victim_dev in keys
    info = json.loads(derived.read_text())["derived_from"]
    assert info["dropped_from_training"] == 1 and info["manifest_sha256"] == m.sha256
    assert info["split_sha256"] == S.split_sha256(s)
