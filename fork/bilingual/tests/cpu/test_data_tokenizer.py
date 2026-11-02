"""I4-D1: the byte-level BPE tokenizer (B-O11; IMPL-08, IMPL-03)."""

import json

import pytest

from surogate_bilingual.config import SPECIAL_TOKENS
from surogate_bilingual.data import tokenizer as T
from surogate_bilingual.data.tokenizer import encode_user_text


def test_i4_d1_fixture_corpus_holds_every_required_script(data_pipeline):
    text = "".join(d.text for d in data_pipeline["derived"].iter_docs(split="seed_train"))
    assert any("一" <= c <= "鿿" for c in text) and any(c.isascii() and c.isalpha() for c in text)
    for probe in ("😀", "𠀀", "，", "。", "\t"):  # emoji, CJK Extension B (4-byte UTF-8), full-width punctuation, tab
        assert probe in text, probe


def test_i4_d1_special_ids_are_the_v10_order(data_pipeline):
    tok = data_pipeline["tok"]
    assert T.special_ids(tok) == {"<unk>": 0, "<eos>": 1, "<|zh|>": 2, "<|en|>": 3, "<|bridge|>": 4}
    assert tok.normalizer is None  # normalization "none"


def test_i4_d1_no_unk_and_exact_round_trips(data_pipeline):
    tok = data_pipeline["tok"]
    report = T.check_tokenizer(tok, n=500, seed=7)
    assert report["ok"], report
    assert report["unk"] == 0 and report["failures"] == []
    strings = T.random_strings(500, seed=7)
    assert any(any(ord(c) >= 0x20000 for c in s) for s in strings)  # 4-byte UTF-8 (CJK Extension B) was drawn
    assert any(any(0x1F300 <= ord(c) <= 0x1F64F for c in s) for s in strings)  # emoji
    for text in SPECIAL_TOKENS:  # every tag string, as escaped text and as the control token
        assert T.decode(tok, T.encode_user_text(tok, text)) == text
    for tag in ("<|zh|>", "<|en|>", "<|bridge|>"):
        assert tok.encode(tag, add_special_tokens=False).ids == [tok.token_to_id(tag)]


def test_i4_d1_fullwidth_and_ascii_comma_stay_distinct(data_pipeline):
    tok = data_pipeline["tok"]
    assert T.encode_user_text(tok, "，") != T.encode_user_text(tok, ",")
    assert T.decode(tok, T.encode_user_text(tok, "，")) == "，"


def test_i4_d1_escaped_user_text_has_no_special_ids(data_pipeline):
    tok = data_pipeline["tok"]
    text = "用<|en|>和<|zh|>标记"
    ids = encode_user_text(tok, text)
    assert not set(ids) & set(range(len(SPECIAL_TOKENS)))
    assert T.decode(tok, ids) == text
    plain = tok.encode(text, add_special_tokens=False).ids  # the v1.0 encoding turns both into control tags
    assert {2, 3} <= set(plain)


def test_i4_d1_training_refuses_files_outside_seed_train(data_pipeline, tmp_path):
    derived = data_pipeline["derived"]
    other = [derived.root / e["path"] for e in derived.files(split="zh_train")]
    assert other
    with pytest.raises(T.TokenizerError, match="not 'seed_train'"):
        T.train_tokenizer(other, 600, tmp_path / "t.json", derived)
    stray = tmp_path / "stray.jsonl"
    stray.write_text(json.dumps({"text": "未登记的文件"}, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(T.TokenizerError, match="not listed in the manifest"):
        T.train_tokenizer([stray], 600, tmp_path / "t.json", derived)
    seed_file = next(derived.root / e["path"] for e in derived.files(split="seed_train"))
    original = seed_file.read_bytes()
    try:
        seed_file.write_bytes(original + b'{"text": "tampered"}\n')
        with pytest.raises(ValueError, match="sha256 differs"):
            T.train_tokenizer([seed_file], 600, tmp_path / "t.json", derived)
    finally:
        seed_file.write_bytes(original)


def test_i4_d1_training_is_deterministic_and_records_its_inputs(data_pipeline, tmp_path):
    derived = data_pipeline["derived"]
    files = [derived.root / e["path"] for e in derived.files(split="seed_train")]
    path = T.train_tokenizer(files, 1024, tmp_path / "again" / "tokenizer.json", derived)
    assert path.read_bytes() == data_pipeline["tokenizer_path"].read_bytes()
    meta = json.loads(T.meta_path(path).read_text(encoding="utf-8"))
    assert meta["format"] == "bilingual-tokenizer/1" and meta["split"] == "seed_train"
    assert meta["manifest_sha256"] == derived.sha256 and meta["tokenizer_sha256"] == T.tokenizer_sha256(path)
    assert meta["initial_alphabet"] == "bytelevel" and meta["normalization"] == "none"
    assert {i["path"] for i in meta["inputs"]} == {e["path"] for e in derived.files(split="seed_train")}
