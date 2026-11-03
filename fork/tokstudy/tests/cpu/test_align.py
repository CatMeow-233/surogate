"""Student/teacher alignment (surogate_tokstudy.align, spec step S10) on constructed spans and SYNTHETIC tokenizers."""

import json

import numpy as np
import pytest
from surogate_tokstudy import align, codecs
from surogate_tokstudy.canonical import TokstudyError
from surogate_tokstudy.docs import EvalDoc, Seg

ONE, HEAD, INTERIOR, CROSS = range(4)


@pytest.mark.parametrize(
    ("student", "expected"),
    [
        ([2, 4], [ONE, ONE]),  # ab | cd against teacher ab | cd
        ([1, 2, 3, 4], [HEAD, INTERIOR, HEAD, INTERIOR]),  # a b c d: 1:N inside each teacher token
        ([4], [CROSS]),  # abcd covers both teacher tokens whole: a teacher boundary inside it
        ([1, 3, 4], [HEAD, CROSS, INTERIOR]),  # a | bc | d: "bc" straddles the teacher boundary
        ([2, 3, 4], [ONE, HEAD, INTERIOR]),
    ],
)
def test_classes_on_constructed_spans(student, expected):
    assert align.classify(np.array(student), np.array([2, 4])).tolist() == expected


def test_three_letter_teacher_token():
    # teacher abc | d; student a | bc | d: head, interior (ends with the teacher token), one-to-one
    assert align.classify(np.array([1, 3, 4]), np.array([3, 4])).tolist() == [HEAD, INTERIOR, ONE]
    with pytest.raises(TokstudyError, match="different bytes"):
        align.classify(np.array([1, 3]), np.array([2, 4]))


def doc(text, domain="py"):
    return EvalDoc("d", domain, (Seg(text=text),))


def test_whitespace_rows_and_counts(tmp_path, bpe_writer):
    """Teacher: "ĠĠ" is one token; a byte student splits it (two non-identity whitespace rows), a student equal to
    the teacher has none."""
    t_path, t_sha = bpe_writer(tmp_path / "t" / "tokenizer.json", [("Ġ", "Ġ")])
    teacher = codecs.SubwordCodec.from_tokenizer_json(t_path, t_sha, drop_normalizer=True, check=True)
    byte = codecs.ByteCodec(codecs.TLC_SPECIALS, check=True)
    m = align.measure(byte, teacher, [doc("a  b")])["py"]
    assert (m["student_tokens"], m["teacher_tokens"], m["bytes"]) == (4, 3, 4)
    assert (m["one_to_one"], m["head"], m["interior"], m["cross_boundary"]) == (2, 1, 1, 0)
    assert (m["whitespace_tokens"], m["non_identity_whitespace"]) == (2, 2)
    assert m["rates"]["non_identity_whitespace"] == 0.5
    same = align.measure(teacher, teacher, [doc("a  b\tc")])["py"]
    assert same["one_to_one"] == same["student_tokens"] and same["non_identity_whitespace"] == 0
    assert same["whitespace_tokens"] == 2  # "ĠĠ" and the tab


def test_a_coarser_student_crosses(tmp_path, bpe_writer):
    t_path, t_sha = bpe_writer(tmp_path / "t" / "tokenizer.json", [("a", "b")])
    s_path, s_sha = bpe_writer(tmp_path / "s" / "tokenizer.json", [("a", "b"), ("ab", "c")])
    teacher = codecs.SubwordCodec.from_tokenizer_json(t_path, t_sha, drop_normalizer=True)
    student = codecs.SubwordCodec.from_tokenizer_json(s_path, s_sha)
    m = align.measure(student, teacher, [doc("abcab", "en")])["en"]
    # teacher ab | c | ab, student abc | ab
    assert (m["cross_boundary"], m["one_to_one"], m["student_tokens"]) == (1, 1, 2)


def test_teacher_normaliser_is_dropped_and_counted(tmp_path, bpe_writer):
    t_path, t_sha = bpe_writer(tmp_path / "t" / "tokenizer.json", normalizer="nfc")
    teacher = codecs.SubwordCodec.from_tokenizer_json(t_path, t_sha, drop_normalizer=True, check=True)
    byte = codecs.ByteCodec(codecs.TLC_SPECIALS)
    m = align.measure(byte, teacher, [doc("café"), doc("cafe")])["py"]
    assert m["docs_changed_by_teacher_normalizer"] == 1 and m["one_to_one"] == m["student_tokens"]


def test_command_line(tmp_path, capsys, bpe_writer):
    from surogate_tokstudy.__main__ import main

    t_path, t_sha = bpe_writer(tmp_path / "t" / "tokenizer.json", [("Ġ", "Ġ")], normalizer="nfc")
    samples = tmp_path / "py.jsonl"
    samples.write_text(json.dumps({"text": "x  = 1\n"}) + "\n", encoding="utf-8")  # SYNTHETIC FIXTURE
    args = ["align", "--teacher", f"{t_path}@{t_sha}", "--bytes", "bytes", "--docs", f"py={samples}",
            "--out", str(tmp_path / "align.json")]  # fmt: skip
    assert main(args) == 0
    report = json.loads((tmp_path / "align.json").read_text(encoding="utf-8"))
    assert report["format"] == "tokstudy-align/1" and report["teacher"]["normalizer_dropped"] == {"type": "NFC"}
    assert report["students"]["bytes"]["domains"]["py"]["non_identity_whitespace"] == 2
    assert "bytes\tpy\t1:1" in capsys.readouterr().out
    assert main([a.replace(t_sha, "e" * 64) for a in args]) == 1
    assert "expected" in capsys.readouterr().err
