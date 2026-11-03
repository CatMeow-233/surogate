"""Codecs (surogate_tokstudy.codecs, spec step S8): exact token bytes, special ids, refusals. SYNTHETIC tokenizers."""

import json

import numpy as np
import pytest
from surogate_tokstudy import codecs
from surogate_tokstudy.canonical import TokstudyError, sha256_file
from tokenizers import pre_tokenizers

TLC = codecs.TLC_SPECIALS


@pytest.fixture(scope="module")
def trained(tmp_path_factory, bpe_writer, edge_texts):
    """A BBPE learned from the edge texts (GPT-2 regex) with the TLC special tokens."""
    texts = [t * 5 for t in edge_texts] + ["def total(values):\n    return sum(values) + 12345\n"] * 20
    return bpe_writer(tmp_path_factory.mktemp("bbpe") / "tokenizer.json", specials=list(TLC), pre="gpt2",
                      train_on=texts)  # fmt: skip


@pytest.fixture(scope="module")
def zero_merge(tmp_path_factory, bpe_writer):
    """A BPE with the 256 byte tokens, the TLC specials and no merge at all."""
    return bpe_writer(tmp_path_factory.mktemp("zero") / "tokenizer.json", specials=list(TLC))


def test_project_specials_equal_the_projects():
    from surogate_bilingual.data import tokenizer as bilingual
    from surogate_tlc.data import tokenizer as tlc

    assert codecs.TLC_SPECIALS == {t: i for i, t in enumerate(tlc.SPECIAL_TOKENS)}
    assert codecs.BILINGUAL_SPECIALS == {t: i for i, t in enumerate(bilingual.SPECIAL_TOKENS)}


def test_byte_alphabet_is_the_byte_level_one(edge_texts):
    assert set(codecs.BYTE_OF_CHAR) == set(pre_tokenizers.ByteLevel.alphabet())
    assert sorted(codecs.BYTE_OF_CHAR.values()) == list(range(256))
    level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    for text in edge_texts:
        mapped = "".join(p for p, _ in level.pre_tokenize_str(text))
        assert bytes(codecs.BYTE_OF_CHAR[c] for c in mapped) == text.encode()


def _codecs(trained, zero_merge):
    return [
        codecs.SubwordCodec.from_tokenizer_json(*trained, TLC, check=True),
        codecs.SubwordCodec.from_tokenizer_json(*zero_merge, TLC, check=True),
        codecs.ByteCodec(TLC, check=True),
    ]


def test_token_bytes_join_to_the_text(trained, zero_merge, edge_texts):
    for codec in _codecs(trained, zero_merge):
        assert codec.special_ids == TLC
        for text in edge_texts:
            ids, ends = codec.encode_text(text)
            raw = text.encode()
            assert ids.dtype == np.int32 and ends.dtype == np.int64 and ids.size == ends.size
            assert b"".join(codec.token_bytes(int(i)) for i in ids) == raw
            assert ends.tolist() == np.cumsum([len(codec.token_bytes(int(i))) for i in ids]).tolist()
            assert not set(ids.tolist()) & set(TLC.values())  # special-token strings in text stay text


def test_zero_merge_bpe_is_the_byte_codec_renumbered(zero_merge, edge_texts):
    subword = codecs.SubwordCodec.from_tokenizer_json(*zero_merge, TLC)
    byte = codecs.ByteCodec(TLC)
    for text in edge_texts:
        s_ids, s_ends = subword.encode_text(text)
        b_ids, b_ends = byte.encode_text(text)
        assert s_ends.tolist() == b_ends.tolist() == list(range(1, len(text.encode()) + 1))
        assert [subword.token_bytes(int(i)) for i in s_ids] == [byte.token_bytes(int(i)) for i in b_ids]
    assert byte.vocab_size == 261 and byte.encode_text("A")[0].tolist() == [5 + 65]


@pytest.mark.parametrize("which", [0, 1, 2])
def test_a_lone_surrogate_is_refused(trained, zero_merge, which):
    codec = _codecs(trained, zero_merge)[which]
    with pytest.raises(TokstudyError, match="lone surrogate"):
        codec.encode_text("ok \ud800 no")


def test_load_refusals(tmp_path, trained, bpe_writer):
    path, sha = trained
    with pytest.raises(TokstudyError, match="expected"):
        codecs.SubwordCodec.from_tokenizer_json(path, "0" * 64)
    with pytest.raises(TokstudyError, match="special tokens"):
        codecs.SubwordCodec.from_tokenizer_json(path, sha, codecs.BILINGUAL_SPECIALS)
    nfc, nfc_sha = bpe_writer(tmp_path / "nfc" / "tokenizer.json", specials=list(TLC), normalizer="nfc")
    with pytest.raises(TokstudyError, match="normaliser"):
        codecs.SubwordCodec.from_tokenizer_json(nfc, nfc_sha)
    teacher = codecs.SubwordCodec.from_tokenizer_json(nfc, nfc_sha, drop_normalizer=True, check=True)
    assert teacher.normalizer_dropped == {"type": "NFC"} and teacher.identity()["normalizer_dropped"] == {"type": "NFC"}
    decomposed = "café"
    assert teacher.changed_by_dropped_normalizer(decomposed) and not teacher.changed_by_dropped_normalizer("cafe")
    assert len(teacher.encode_text(decomposed)[1]) == len(decomposed.encode())  # the raw bytes, not NFC's
    data = json.loads(path.read_text(encoding="utf-8"))
    for name, edit, message in (
        ("prefix", lambda d: d["pre_tokenizer"].update(add_prefix_space=True), "prefix space"),
        ("whitespace", lambda d: d.update(pre_tokenizer={"type": "Whitespace"}), "no ByteLevel step"),
        ("broken", lambda d: d["model"].update(type="WordLevel"), "not a tokenizer.json"),
    ):
        bad = tmp_path / name / "tokenizer.json"
        bad.parent.mkdir()
        edited = json.loads(json.dumps(data))
        edit(edited)
        bad.write_text(json.dumps(edited), encoding="utf-8")
        with pytest.raises(TokstudyError, match=message):
            codecs.SubwordCodec.from_tokenizer_json(bad, sha256_file(bad))


def test_model_kinds(tmp_path, trained):
    from tokenizers import Tokenizer, models, pre_tokenizers

    word = Tokenizer(models.WordLevel({"a": 0, "<unk>": 1}, unk_token="<unk>"))
    word.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    word.save(str(tmp_path / "word.json"))
    with pytest.raises(TokstudyError, match="WordLevel, not a byte-level BPE"):
        codecs.SubwordCodec.from_tokenizer_json(tmp_path / "word.json", sha256_file(tmp_path / "word.json"))
    data = json.loads(trained[0].read_text(encoding="utf-8"))
    del data["model"]["type"]  # the older layout (GPT-2's tokenizer.json) names no model type
    old = tmp_path / "old" / "tokenizer.json"
    old.parent.mkdir()
    old.write_text(json.dumps(data), encoding="utf-8")
    codec = codecs.SubwordCodec.from_tokenizer_json(old, sha256_file(old), TLC, check=True)
    assert codec.encode_text("def total(values):")[1][-1] == len("def total(values):")


def test_byte_codec_padding_and_specials():
    codec = codecs.ByteCodec(codecs.BILINGUAL_SPECIALS, pad_to=264)
    assert codec.vocab_size == 264 and codec.special_ids == codecs.BILINGUAL_SPECIALS
    assert codec.token_bytes(0) == b"" and codec.token_bytes(5) == b"\x00" and codec.token_bytes(260) == b"\xff"
    with pytest.raises(TokstudyError, match="padding"):
        codec.token_bytes(261)
    with pytest.raises(TokstudyError, match="special ids 0-4"):
        codecs.ByteCodec({"<a>": 0, "<b>": 2})


def test_tokenizer_argument():
    assert codecs.parse_tokenizer_arg("q=dir/a@b/tokenizer.json@" + "a" * 64)[1].as_posix() == "dir/a@b/tokenizer.json"
    with pytest.raises(TokstudyError, match="NAME=PATH@SHA256"):
        codecs.parse_tokenizer_arg("dir/tokenizer.json")
