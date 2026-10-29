"""Tokenizer (surogate_tlc.data.tokenizer, README: "Tokenizer") and its native-consistency harness, on CPU.

The harness runs here against stand-ins for the native tokenizer: one answered by the Hugging Face tokenizer, and
damaged ones it must catch. The comparison with the real native tokenizer is a GPU-host test
(tests/gpu/test_tokenizer_native.py) and, at G0 scale, the `check` command.
"""

import json

import pytest
from surogate_tlc import hf, modeldir
from surogate_tlc.data import corpus
from surogate_tlc.data import tokenizer as tok
from tokenizers import Tokenizer
from transformers import AutoTokenizer


class FakeNative:
    """The native Tokenizer's interface, answered by the Hugging Face tokenizer."""

    def __init__(self, path):
        self.ordinary, self.special = tok.load(path, ordinary=True), tok.load(path)
        ids = tok.special_ids(self.special)
        self.pad_token_id, self.bos_token_id, self.eos_token_id = ids[tok.PAD], ids[tok.BOS], ids[tok.EOS]
        self.vocab_size = self.special.get_vocab_size()

    def encode(self, text, add_special_tokens=False):
        return self.special.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_ordinary(self, text):
        return self.ordinary.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.ordinary.decode(ids, skip_special_tokens=False)


class DropsLastDigit(FakeNative):
    def encode_ordinary(self, text):
        ids = super().encode_ordinary(text)
        return ids[:-1] if any(c.isdigit() for c in text) else ids


class NoSpecialTokens(FakeNative):
    def encode(self, text, add_special_tokens=False):
        return self.encode_ordinary(text)


class TabsToSpaces(FakeNative):
    def decode(self, ids):
        return super().decode(ids).replace("\t", "    ")


@pytest.fixture(scope="module")
def texts(corpus_spec) -> list[str]:
    return [*tok.EDGE_CASES, *(d.text for d in corpus.documents(corpus.load_corpus(corpus_spec)))]


@pytest.fixture(scope="module")
def variants(tmp_path_factory, corpus_spec) -> dict[str, object]:
    """Every pre-tokenizer candidate trained on the fixture corpus."""
    c = corpus.load_corpus(corpus_spec)
    out = tmp_path_factory.mktemp("variants")
    return {name: tok.train_tokenizer(c, out / name, pretokenizer=name, vocab_size=300) for name in tok.PRETOKENIZERS}


def test_recipe(trained_tokenizer):
    data = json.loads((trained_tokenizer / tok.TOKENIZER_FILE).read_text(encoding="utf-8"))
    split, byte_level = data["pre_tokenizer"]["pretokenizers"]
    assert split["pattern"] == {"Regex": tok.PRETOKENIZERS[tok.DEFAULT_PRETOKENIZER]}
    assert (split["behavior"], byte_level["type"], byte_level["use_regex"]) == ("Isolated", "ByteLevel", False)
    assert data["normalizer"] is None and data["decoder"]["type"] == "ByteLevel"
    assert [(t["id"], t["content"], t["special"]) for t in data["added_tokens"]] == [
        (i, token, True) for i, token in enumerate(tok.SPECIAL_TOKENS)
    ]
    assert Tokenizer.from_file(str(trained_tokenizer / tok.TOKENIZER_FILE)).get_vocab_size() == 300
    config = json.loads((trained_tokenizer / tok.CONFIG_FILE).read_text(encoding="utf-8"))
    assert (config["bos_token"], config["eos_token"], config["pad_token"]) == (tok.BOS, tok.EOS, tok.PAD)
    assert config["chat_template"] == tok.CHAT_TEMPLATE and config["add_bos_token"] is False


def test_manifest(trained_tokenizer, corpus_spec):
    manifest = json.loads((trained_tokenizer / tok.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["tokenizer_sha256"] == tok.tokenizer_sha256(trained_tokenizer)
    assert (manifest["pretokenizer"], manifest["vocab_size"]) == ("qwen2", 300)
    assert manifest["corpus"]["spec_sha256"] == corpus.file_sha256(corpus_spec)
    # Trained on the train split only: 22 of the 27 documents.
    c = corpus.load_corpus(corpus_spec)
    train = [d.source.name for d in corpus.documents(c) if corpus.split_of(c, d) == "train"]
    assert len(train) == 22
    assert {s["name"]: s["documents"] for s in manifest["sources"]} == {n: train.count(n) for n in ("code", "text")}
    assert manifest["sources"][0]["dropped"] == {"empty": 1, "not_utf8": 1}
    assert all({"url", "license", "downloaded", "filters", "files"} <= set(s) for s in manifest["sources"])


def test_training_is_deterministic(tmp_path, corpus_spec, trained_tokenizer):
    again = tok.train_tokenizer(corpus.load_corpus(corpus_spec), tmp_path / "again", vocab_size=300)
    for name in (tok.TOKENIZER_FILE, tok.CONFIG_FILE, tok.MANIFEST_FILE):
        assert (again / name).read_bytes() == (trained_tokenizer / name).read_bytes()


def test_refuses_to_overwrite_or_to_fall_short(tmp_path, corpus_spec, trained_tokenizer):
    c = corpus.load_corpus(corpus_spec)
    with pytest.raises(FileExistsError):
        tok.train_tokenizer(c, trained_tokenizer, vocab_size=300)
    with pytest.raises(ValueError, match="too small"):
        tok.train_tokenizer(c, tmp_path / "big", vocab_size=tok.VOCAB_SIZE)


def test_digits_are_single_tokens(trained_tokenizer):
    t = tok.load(trained_tokenizer)
    assert len(t.encode("1234567890").ids) == 10
    pieces = [t.decode([i]) for i in range(t.get_vocab_size())]
    assert all(len(p) == 1 for p in pieces if any(c.isdigit() for c in p))


@pytest.mark.parametrize("name", list(tok.PRETOKENIZERS))
def test_round_trip_and_special_strings(variants, texts, name):
    ordinary, special = tok.load(variants[name], ordinary=True), tok.load(variants[name])
    for text in texts:
        assert ordinary.decode(ordinary.encode(text).ids, skip_special_tokens=False) == text
    ids = tok.special_ids(special)
    text = "end <|eos|> pad <|pad|> code <|code_start|>"
    assert not set(ordinary.encode(text).ids) & set(ids.values())  # never a special id from document text
    assert set(special.encode(text).ids) & set(ids.values()) == {ids[tok.EOS], ids[tok.PAD], ids[tok.CODE_START]}


def test_model_dir_and_chat_template(tmp_path, trained_tokenizer):
    model = modeldir.write_model_dir(tmp_path / "model", hf.default_config(), trained_tokenizer, init_seed=0)
    auto = AutoTokenizer.from_pretrained(model)
    assert (auto.pad_token_id, auto.bos_token_id, auto.eos_token_id) == (0, 1, 2)
    content = "def f(x):\n    return x\n"
    body = auto(content, add_special_tokens=False)["input_ids"]
    messages = [{"role": "user", "content": "x = 1\n"}, {"role": "assistant", "content": content}]
    assert auto.apply_chat_template(messages[1:], return_dict=False) == [1, *body, 2]
    prompt = auto.apply_chat_template(messages[:1], add_generation_prompt=True, return_dict=False)
    assert prompt == [1, *auto("x = 1\n", add_special_tokens=False)["input_ids"]]
    assert auto.apply_chat_template(messages, return_dict=False)[-1] == 2


def test_compare_accepts_a_consistent_tokenizer(trained_tokenizer, texts):
    assert tok.compare(trained_tokenizer, texts, FakeNative(trained_tokenizer)) == []


@pytest.mark.parametrize(
    ("fake", "expected"),
    [
        (DropsLastDigit, "encode_ordinary: "),
        (NoSpecialTokens, "encode: "),
        (TabsToSpaces, "decode (native): "),
    ],
)
def test_compare_reports_differences(trained_tokenizer, texts, fake, expected):
    problems = tok.compare(trained_tokenizer, texts, fake(trained_tokenizer), examples=2)
    summaries = [p for p in problems if " of " in p and "texts differ" in p]
    assert len(summaries) == 1 and summaries[0].startswith(expected), problems
    assert len(problems) <= 3  # one summary line and at most `examples` examples


def test_compare_checks_special_ids_and_vocab(trained_tokenizer):
    native = FakeNative(trained_tokenizer)
    native.bos_token_id, native.vocab_size = 7, 16384
    problems = tok.compare(trained_tokenizer, [], native)
    assert problems == ["bos_token_id: native 7, <|bos|> has id 1", "vocab_size: native 16384, here 300"]
