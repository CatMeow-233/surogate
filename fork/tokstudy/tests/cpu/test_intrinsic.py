"""Intrinsic metrics (surogate_tokstudy.intrinsic, spec step S10) against hand computations on SYNTHETIC inputs."""

import json
import math

import numpy as np
import pytest
from surogate_tokstudy import codecs, intrinsic
from surogate_tokstudy.canonical import TokstudyError
from surogate_tokstudy.docs import EvalDoc, Seg

TLC = codecs.TLC_SPECIALS
# The hand-made tokenizer: no pre-tokenizer split (merges act anywhere), and these merges in rank order.
MERGES = [("Ġ", "="), ("1", "2"), ("(", "x"), ("b", "a")]
PY = "x = 12\nf(x)\nab = 1\n"
# Its tokens, worked out by hand: x | Ġ= | Ġ | 12 | Ċ | f | (x | ) | Ċ | a | b | Ġ= | Ġ | 1 | Ċ
PY_TOKENS = ["x", " =", " ", "12", "\n", "f", "(x", ")", "\n", "a", "b", " =", " ", "1", "\n"]


@pytest.fixture(scope="module")
def hand(tmp_path_factory, bpe_writer):
    path, sha = bpe_writer(tmp_path_factory.mktemp("hand") / "tokenizer.json", MERGES, specials=list(TLC))
    return codecs.SubwordCodec.from_tokenizer_json(path, sha, TLC, check=True)


def doc(text, domain="py", key="d"):
    return EvalDoc(key, domain, (Seg(text=text),))


def test_the_hand_tokenizer_gives_the_hand_tokens(hand):
    ids, _ = hand.encode_text(PY)
    assert [hand.token_bytes(int(i)).decode() for i in ids] == PY_TOKENS


def test_python_metrics_by_hand(hand):
    m = intrinsic.measure(hand, [doc(PY)])["py"]
    assert (m["docs"], m["bytes"], m["chars"], m["tokens"]) == (1, 19, 19, 15)
    assert m["bytes_per_token"] == 19 / 15
    py = m["python"]
    # Lexemes x = 12 f ( x ) ab = 1; only "(x" overlaps two of them ("Ġ=" holds a space, which is no lexeme).
    assert (py["parsed_segments"], py["tokens"], py["cross_lexeme_tokens"]) == (1, 15, 1)
    # Identifiers x, f, x, ab: pieces 1, 1, 1 ("(x"), 2 ("a" "b").
    assert (py["identifiers"], py["identifier_pieces"], py["single_piece_identifiers"]) == (4, 5, 3)
    assert py["cross_lexeme_rate"] == 1 / 15 and py["pieces_per_identifier"] == 5 / 4
    # Digit runs "12" (one token) and "1".
    assert (m["digit_runs"], m["digit_chars"], m["digit_pieces"]) == (2, 3, 2)
    assert m["words"] is None and m["han_chars"] == 0 and m["invalid_utf8_tokens"] == 0
    counts = {"x": 1, " =": 2, " ": 2, "12": 1, "\n": 3, "f": 1, "(x": 1, ")": 1, "a": 1, "b": 1, "1": 1}
    assert m["distinct_tokens"] == len(counts) and m["vocab_utilisation"] == len(counts) / (hand.vocab_size - 5)
    p = np.array(list(counts.values())) / 15
    assert m["renyi_efficiency"] == pytest.approx(math.log((p**2.5).sum()) / (1 - 2.5) / math.log(hand.vocab_size))


def test_renyi_efficiency_closed_form():
    expected = math.log(0.75**2.5 + 0.25**2.5) / (1 - 2.5) / math.log(2)
    assert intrinsic.renyi_efficiency(np.array([3, 1]), 2) == pytest.approx(expected, rel=1e-15)
    assert intrinsic.renyi_efficiency(np.array([1, 1]), 2) == pytest.approx(1.0)  # uniform over |V|: efficiency 1
    assert intrinsic.renyi_efficiency(np.array([0, 0]), 2) is None


def test_bytes_chars_han_words_and_invalid_utf8():
    codec = codecs.ByteCodec(TLC)
    zh = "中文𠀀，好々ab"  # Han: 中 文 𠀀 好 々 (the iteration mark is Script=Han); "，" is not
    en = "Hello  world\tfoo\n"
    m = intrinsic.measure(codec, [doc(zh, "zh"), doc(en, "en")])
    assert (m["zh"]["bytes"], m["zh"]["chars"], m["zh"]["han_chars"]) == (len(zh.encode()), 8, 5)
    assert m["zh"]["han_chars_per_token"] == 5 / len(zh.encode())
    # Each byte of a multi-byte character is its own token, never valid UTF-8 alone; "a", "b" are.
    assert m["zh"]["invalid_utf8_tokens"] == len(zh.encode()) - 2
    assert (m["en"]["words"], m["en"]["words_per_token"]) == (3, 3 / len(en))
    assert m["en"]["python"] is None and m["zh"]["words"] is None
    assert m["en"]["vocab_utilisation"] == len(set(en.encode())) / 256


def test_han_count_matches_the_ideograph_names():
    import unicodedata

    text = "".join(chr(c) for c in (*range(0x4E00, 0x4E40), *range(0x20000, 0x20010), 0xF900, 0x3001, 0x41, 0x3042))
    ideographs = sum(unicodedata.name(c, "").startswith(("CJK UNIFIED", "CJK COMPATIBILITY IDEOGRAPH")) for c in text)
    assert intrinsic.han_chars(text) == ideographs == 0x40 + 0x10 + 1


def test_context_coverage_is_the_upper_median(hand):
    long_doc = " =" * 1100  # 1,100 tokens of 2 bytes under the hand tokenizer
    docs = [doc(long_doc, "en", "a"), doc(" =" * 600 + "x" * 9, "en", "b"), doc("short", "en", "c")]
    m = intrinsic.measure(hand, docs)["en"]
    assert m["context"]["512"] == {"docs": 2, "median_bytes": 1024}
    assert m["context"]["2048"] == {"docs": 0, "median_bytes": None}
    byte = intrinsic.measure(codecs.ByteCodec(TLC), [doc("é" * 1100, "en")])["en"]
    assert byte["context"]["2048"] == {"docs": 1, "median_bytes": 2048}


def test_segments_never_merge_across_a_special(hand):
    joined = intrinsic.measure(hand, [doc("abab")])["py"]
    split = intrinsic.measure(hand, [EvalDoc("d", "py", (Seg(text="ab"), Seg(special="<|eos|>"), Seg(text="ab")))])
    assert joined["tokens"] == 3  # a | ba | b
    assert split["py"]["tokens"] == 4 and split["py"]["bytes"] == 4  # a b, then a b: the special separates them


def test_python_segments_that_do_not_parse_are_counted_not_measured(hand):
    m = intrinsic.measure(hand, [doc("def (\n"), doc(PY)])["py"]["python"]
    assert (m["parsed_segments"], m["unparsed_segments"], m["tokens"]) == (1, 1, 15)


def test_lexemes_of_edge_cases(edge_texts):
    for text in edge_texts:
        spans = intrinsic.lexemes(text)
        if spans is not None:
            assert all(text[s:e] == string for s, e, _, string in spans)
    spans = intrinsic.lexemes("s = f'{x!r:>10} {{y}}'\n")
    assert spans is not None and [t for *_, t in spans][:2] == ["s", "="]


def _write_samples(tmp_path):
    """SYNTHETIC FIXTURE sample files in the two accepted formats."""
    jsonl = tmp_path / "py.jsonl"
    jsonl.write_text("\n".join(json.dumps({"text": t, "id": i}) for i, t in enumerate([PY, "y = 3\n"])) + "\n",
                     encoding="utf-8")  # fmt: skip
    rows = tmp_path / "rows_en.json"
    payload = {"rows": [{"row_idx": i, "row": {"content": t}} for i, t in enumerate(["Hello world.", "Two words"])]}
    rows.write_text(json.dumps(payload), encoding="utf-8")
    return jsonl, rows


def test_command_line(tmp_path, capsys, bpe_writer):
    from surogate_tokstudy.__main__ import main

    path, sha = bpe_writer(tmp_path / "tok" / "tokenizer.json", MERGES, specials=list(TLC))
    jsonl, rows = _write_samples(tmp_path)
    args = ["intrinsic", "--tokenizer", f"hand={path}@{sha}", "--bytes", "bytes", "--project", "tlc", "--check",
            "--docs", f"py={jsonl}", "--docs", f"en={rows}", "--out", str(tmp_path / "out.json")]  # fmt: skip
    assert main(args) == 0
    report = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert report["format"] == "tokstudy-intrinsic/1" and sorted(report["arms"]) == ["bytes", "hand"]
    assert report["arms"]["hand"]["codec"]["sha256"] == sha
    assert [(i["domain"], i["docs"]) for i in report["inputs"]] == [("py", 2), ("en", 2)]
    assert report["arms"]["bytes"]["domains"]["en"]["tokens"] == len("Hello world.Two words")
    assert report["arms"]["hand"]["domains"]["py"]["python"]["identifiers"] == 5
    first = (tmp_path / "out.json").read_bytes()
    assert main(args) == 0 and (tmp_path / "out.json").read_bytes() == first  # deterministic
    out = capsys.readouterr().out
    assert "hand\tpy\t2 docs" in out
    bad = [a.replace(sha, "f" * 64) for a in args]
    assert main(bad) == 1 and "expected" in capsys.readouterr().err
    assert main([*args[:-2], "--field", "body", "--out", str(tmp_path / "x.json")]) == 1
    assert "no string field 'body'" in capsys.readouterr().err


def test_inputs_are_refused_precisely(tmp_path):
    from surogate_tokstudy.docs import parse_docs_arg, plain_documents

    with pytest.raises(TokstudyError, match="DOMAIN=PATH"):
        parse_docs_arg("samples.jsonl")
    with pytest.raises(TokstudyError, match="domain"):
        parse_docs_arg("Py=samples.jsonl")
    other = tmp_path / "x.txt"
    other.write_text("x", encoding="utf-8")
    with pytest.raises(TokstudyError, match=".jsonl or a datasets-server"):
        list(plain_documents(other, "py"))
