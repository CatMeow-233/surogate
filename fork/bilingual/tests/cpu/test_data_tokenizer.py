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


# The tokenizer.json the I4-D1 fixture corpus gave before `tokenizer.pretokenizer` existed (tokenizers 0.23.3, the
# version fork/bilingual/requirements-gpu.txt pins): the default must keep producing exactly this file.
BYTELEVEL_REGRESSION_SHA256 = "ca52c03a796b8368dd3a30f3d1965c0d511682de42c7b3acf1c6b56ab3882687"


def _is_han(c: str) -> bool:
    import unicodedata

    return unicodedata.name(c, "").startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH"))


@pytest.fixture(scope="module")
def arms(data_pipeline, tmp_path_factory):
    """The fixture corpus's tokenizer under each pre-tokenizer (1024 ids, as data_pipeline's)."""
    derived = data_pipeline["derived"]
    files = [derived.root / e["path"] for e in derived.files(split="seed_train")]
    root = tmp_path_factory.mktemp("pretok")
    paths = {
        n: T.train_tokenizer(files, 1024, root / n / "tokenizer.json", derived, pretokenizer=n) for n in T.PRETOKENIZERS
    }
    return {n: (path, T.load_tokenizer(path)) for n, path in paths.items()}


def test_bytelevel_default_is_the_v10_recipe(data_pipeline, arms):
    import tokenizers
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    path, _ = arms["bytelevel"]
    assert path.read_bytes() == data_pipeline["tokenizer_path"].read_bytes()  # the default is bytelevel
    if tokenizers.__version__ == "0.23.3":
        assert T.tokenizer_sha256(path) == BYTELEVEL_REGRESSION_SHA256
    # The recipe as it was written before the option, rebuilt here: byte-identical on any tokenizers version.
    old = Tokenizer(models.BPE(unk_token="<unk>"))
    old.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    old.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=1024,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    derived = data_pipeline["derived"]
    inputs = [(derived.root / e["path"], e) for e in derived.files(split="seed_train")]
    old.train_from_iterator(T._texts(inputs, derived), trainer=trainer)
    assert old.to_str().encode("utf-8") == path.read_bytes()
    meta = json.loads(T.meta_path(path).read_text(encoding="utf-8"))
    assert (meta["pretokenizer"], meta["pretokenizer_pattern"]) == ("bytelevel", None)


@pytest.mark.parametrize("name", ["qwen2-digits", "han-isolate"])
def test_study_pretokenizers_keep_the_d1_contract(arms, name):
    path, tok = arms[name]
    report = T.check_tokenizer(tok, n=500, seed=11)  # special ids, no <unk>, 500 random strings and the tags
    assert report["ok"], report
    assert T.encode_user_text(tok, "，") != T.encode_user_text(tok, ",")
    assert tok.normalizer is None
    meta = json.loads(T.meta_path(path).read_text(encoding="utf-8"))
    assert (meta["pretokenizer"], meta["pretokenizer_pattern"]) == (name, T.PRETOKENIZERS[name])
    data = json.loads(path.read_text(encoding="utf-8"))
    split, byte_level = data["pre_tokenizer"]["pretokenizers"]
    assert split["pattern"] == {"Regex": T.PRETOKENIZERS[name]} and split["behavior"] == "Isolated"
    assert byte_level["use_regex"] is (name == "han-isolate")


def test_qwen2_digits_isolates_every_digit(arms):
    _, tok = arms["qwen2-digits"]
    assert len(T.encode_user_text(tok, "1234567890")) == 10
    pieces = [p for p, _ in tok.pre_tokenizer.pre_tokenize_str("x1 = 2026年3月")]
    assert pieces == ["x", "1", "Ġ=", "Ġ", "2", "0", "2", "6", "å¹´", "3", "æľĪ"]
    for i in range(tok.get_vocab_size()):
        text = T.decode(tok, [i])
        assert not (len(text) > 1 and any(c.isdigit() for c in text)), text


def test_han_isolate_gives_every_han_character_its_own_piece(data_pipeline, arms):
    _, tok = arms["han-isolate"]
    text = "".join(d.text for d in data_pipeline["derived"].iter_docs(split="seed_train"))
    text += "扩展B：𠀀𠀁𪚥，兼容：豈"  # CJK Extension B (4-byte UTF-8) and a compatibility ideograph
    han = [c for c in text if _is_han(c)]
    assert any(ord(c) >= 0x20000 for c in han)
    pieces = tok.pre_tokenizer.pre_tokenize_str(text)
    isolated = [text[a:b] for _, (a, b) in pieces if any(_is_han(c) for c in text[a:b])]
    assert isolated == han  # each Han character alone, in order, none merged with a neighbour
    for i in range(tok.get_vocab_size()):
        assert sum(map(_is_han, T.decode(tok, [i]))) <= 1
    assert T.round_trips(tok, "扩展B：𠀀𠀁𪚥")


def test_unknown_pretokenizer_is_refused(data_pipeline, tmp_path, capsys):
    derived = data_pipeline["derived"]
    files = [derived.root / e["path"] for e in derived.files(split="seed_train")]
    with pytest.raises(T.TokenizerError, match="pretokenizer must be one of"):
        T.train_tokenizer(files, 1024, tmp_path / "t.json", derived, pretokenizer="gpt2")
    assert not (tmp_path / "t.json").exists()
    with pytest.raises(SystemExit):  # the command line offers only the three names
        T.main(["train", "--manifest", str(tmp_path / "m.json"), "--pretokenizer", "gpt2", "--out", str(tmp_path)])
    assert "invalid choice: 'gpt2'" in capsys.readouterr().err
