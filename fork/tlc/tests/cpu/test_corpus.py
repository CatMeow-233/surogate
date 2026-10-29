"""Corpus spec (surogate_tlc.data.corpus, docs/data.md: 语料): validation, documents, provenance and the split."""

import hashlib
import json
from collections import Counter

import pytest
from surogate_tlc.data import corpus


def _spec(corpus_spec, replace: tuple[str, str] | None = None, append: str = ""):
    """A copy of the fixture spec next to its files, edited."""
    text = corpus_spec.read_text(encoding="utf-8")
    if replace:
        assert replace[0] in text
        text = text.replace(*replace)
    path = corpus_spec.parent / f"edited-{hashlib.sha256((text + append).encode()).hexdigest()[:8]}.toml"
    path.write_text(text + append, encoding="utf-8")
    return path


def test_loads_the_fixture_spec(corpus_spec):
    c = corpus.load_corpus(corpus_spec)
    assert (c.seed, c.val_fraction) == (7, 0.25)
    code, text = c.sources
    assert (code.name, code.kind, len(code.files)) == ("code", "code", 14)
    assert text.provenance == {
        "url": "https://example.invalid/text",
        "revision": "r1",
        "license": "CC-BY-4.0",
        "downloaded": "2026-10-02",
        "stage": "M1",
        "filters": [],
        "generated": False,
    }
    assert code.provenance["downloaded"] == "2026-10-01"  # a TOML date and an ISO string read alike
    assert not code.generated and not text.generated


@pytest.mark.parametrize(
    ("replace", "message"),
    [
        (('license = "MIT"\n', ""), "missing keys \\['license'\\]"),
        (('stage = "M1"', 'stage = "M1"\nnotes = "x"'), "unknown keys \\['notes'\\]"),
        (('kind = "code"', 'kind = "binary"'), "kind must be one of"),
        (('files = ["code/*.py"]', 'files = ["nothing/*.py"]'), "matches no file"),
        (('url = "https://example.invalid/code"', 'url = ""'), "url must be a non-empty string"),
        (("filters = []", 'filters = "none"'), "filters must be a list"),
        (("downloaded = 2026-10-01", 'downloaded = "last week"'), "Invalid isoformat"),
        (("val_fraction = 0.25", "val_fraction = 1"), "val_fraction must be"),
        (("seed = 7", "seed = -1"), "seed must be"),
        (('name = "text"', 'name = "code"'), "distinct names"),
        (('stage = "M1"', 'stage = "M1"\ngenerated = "yes"'), "generated must be true or false"),
    ],
)
def test_rejects_incomplete_or_wrong_specs(corpus_spec, replace, message):
    with pytest.raises(ValueError, match=message):
        corpus.load_corpus(_spec(corpus_spec, replace))


def test_documents_are_normalised_and_drops_counted(corpus_spec):
    c = corpus.load_corpus(corpus_spec)
    dropped = Counter()
    docs = list(corpus.documents(c, dropped))
    assert dropped == {("code", "not_utf8"): 1, ("code", "empty"): 1, ("text", "empty"): 1}
    assert len(docs) == len({d.key for d in docs}) == 27
    assert all(d.text and "\r" not in d.text for d in docs)
    crlf = next(d for d in docs if d.key == "code/code/mod_01.py")
    assert crlf.text.startswith("def scale_items(values, limit=1):\n    total = 0\n")
    assert next(d for d in docs if d.key == "code/code/mod_02.py").text.count("\n") == 6
    assert [d.group for d in docs if d.group.startswith("group:")] == ["group:book"] * 3
    assert docs[-1].key == "text/text.jsonl:15"  # JSONL keys carry the line number


def test_malformed_jsonl_is_an_error(corpus_spec):
    bad = corpus_spec.parent / "bad.jsonl"
    bad.write_text(json.dumps({"body": "no text field"}) + "\n", encoding="utf-8")
    spec = _spec(corpus_spec, ('files = ["text.jsonl"]', 'files = ["bad.jsonl"]'))
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        list(corpus.documents(corpus.load_corpus(spec)))
    bad.unlink()


def test_split_is_by_group_and_stable(corpus_spec):
    c = corpus.load_corpus(corpus_spec)
    docs = list(corpus.documents(c))
    splits = {d.key: corpus.split_of(c, d) for d in docs}
    assert Counter(splits.values()) == {"train": 22, "val": 5}
    assert len({splits[d.key] for d in docs if d.group == "group:book"}) == 1
    # The split depends on (seed, group) only: the same document in another spec lands in the same split.
    for d in docs:
        expected = int.from_bytes(hashlib.sha256(f"7\0{d.group}".encode()).digest()[:8], "little") < 0.25 * 2**64
        assert splits[d.key] == ("val" if expected else "train")


def test_sample_takes_equal_shares(corpus_spec):
    c = corpus.load_corpus(corpus_spec)
    docs = corpus.sample(c, 6)
    assert [d.source.name for d in docs] == ["code"] * 3 + ["text"] * 3


def test_provenance_lists_files_with_hashes(corpus_spec):
    sources = corpus.provenance(corpus.load_corpus(corpus_spec))
    files = {f["path"]: f for f in sources[0]["files"]}
    assert sorted(files) == sorted(f"code/{p.name}" for p in (corpus_spec.parent / "code").iterdir())
    data = (corpus_spec.parent / "code" / "bad.py").read_bytes()
    assert files["code/bad.py"] == {
        "path": "code/bad.py",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    keys = {"name", "kind", "url", "revision", "license", "downloaded", "stage", "filters", "generated", "files"}
    assert set(sources[1]) == keys
