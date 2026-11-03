"""freqtrim (surogate_tlc.data.freqtrim, GAP-18; docs/data.md: Tokenizer) on CPU, against SYNTHETIC teachers.

Two teachers stand in for Qwen, both written by the tests in the format of Qwen2's tokenizer.json: a hand-made one
whose counts and merges are small enough to work out by hand, and conftest's learned one (qwen_teacher). Neither is
a real Qwen file; the notebook's checks on the real one run on the owner host (`freqtrim-check`).
"""

import json

import pytest
from surogate_tlc.data import corpus, freqtrim
from surogate_tlc.data import tokenizer as tok
from tokenizers import Regex, Tokenizer, models, normalizers, pre_tokenizers

REF = freqtrim.TeacherRef("fixture/synthetic-qwen", "0123456789abcdef0123456789abcdef01234567", "apache-2.0")
ALPHABET = sorted(pre_tokenizers.ByteLevel.alphabet())
# The hand-made teacher's merges, in rank order. "12" is a multi-digit token Qwen2's pattern never produces.
MERGES = [
    ("a", "b"),  # 256 ab
    ("c", "d"),  # 257 cd
    ("ab", "c"),  # 258 abc
    ("x", "y"),  # 259 xy
    ("Ġ", "Ġ"),  # 260 ĠĠ
    ("ĠĠ", "Ġ"),  # 261 ĠĠĠ
    ("Ġ", "x"),  # 262 Ġx
    ("ĠĠ", "ĠĠ"),  # 263 ĠĠĠĠ
    ("(", "x"),  # 264 (x
    ("_", "ab"),  # 265 _ab
    ("1", "2"),  # 266 12
]
# Qwen2's pattern with identifiers (letters, "_", digits) and integers kept whole: coarser than Qwen2's.
COARSE_WORDS = (
    tok.PRETOKENIZERS["qwen2"]
    .replace(r"[^\r\n\p{L}\p{N}]?\p{L}+", r"[^\r\n\p{L}\p{N}]?[\p{L}_][\p{L}\p{N}_]*")
    .replace(r"\p{N}|", r"\p{N}+|")
)
# Indentation glued to the next word: coarser than Qwen2's in the context-dependent whitespace.
COARSE_INDENT = r"[ \t]*\p{L}+|" + tok.PRETOKENIZERS["qwen2"]


def write_hand_teacher(path, added=("<|endoftext|>", "<|im_start|>", "<|im_end|>")):
    """SYNTHETIC FIXTURE: a Qwen2-format tokenizer.json with the 256 byte tokens and MERGES."""
    vocab = {c: i for i, c in enumerate(ALPHABET)}
    for a, b in MERGES:
        vocab[a + b] = len(vocab)
    teacher = Tokenizer(models.BPE(vocab=vocab, merges=MERGES))
    teacher.normalizer = normalizers.NFC()
    split = pre_tokenizers.Split(Regex(tok.PRETOKENIZERS["qwen2"]), behavior="isolated")
    teacher.pre_tokenizer = pre_tokenizers.Sequence(
        [split, pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)]
    )
    teacher.add_special_tokens(list(added))
    teacher.save(str(path))
    return path


@pytest.fixture(scope="module")
def hand(tmp_path_factory):
    return freqtrim.load_teacher(write_hand_teacher(tmp_path_factory.mktemp("hand") / "tokenizer.json"), REF)


def subset(teacher, texts, size, regex="qwen"):
    counts = freqtrim.count_tokens(teacher, freqtrim.REGEXES[regex], texts)
    keep, stats = freqtrim.select(teacher, counts, 256 + size)
    return keep - set(ALPHABET), stats


def test_selection_by_count_with_parents_and_skips(hand):
    texts = ["abc\nabc\nabc\ncd\ncd\nxy"]  # abc 3, cd 2, xy 1 (and the newline byte, already in)
    assert subset(hand, texts, 1)[0] == {"cd"}  # abc needs ab too and does not fit: skipped, not a stop
    assert subset(hand, texts, 2)[0] == {"ab", "abc"}
    assert subset(hand, texts, 3)[0] == {"ab", "abc", "cd"}
    keep, stats = subset(hand, texts, 4)
    assert keep == {"ab", "abc", "cd", "xy"}
    assert stats == {
        "byte_alphabet": 256,
        "by_frequency": 3,
        "as_parents": 1,
        "skipped_for_space": 0,
        "observed_teacher_tokens": 4,
    }
    assert subset(hand, ["cd\nxy"], 1)[0] == {"cd"}  # equal counts: the lower teacher id first
    with pytest.raises(freqtrim.FreqtrimError, match="too small"):
        subset(hand, texts, 5)


def test_student_ids_and_merges(hand):
    keep = set(ALPHABET) | {"cd", "ab", "abc"}
    student = freqtrim.build_tokenizer(hand, keep, freqtrim.REGEXES["qwen"])
    vocab = student.get_vocab()
    assert [vocab[t] for t in tok.SPECIAL_TOKENS] == [0, 1, 2, 3, 4]
    ordinary = sorted((i, t) for t, i in vocab.items() if i >= 5)
    assert [t for _, t in ordinary] == sorted(keep, key=hand.vocab.__getitem__)  # teacher id order from id 5
    assert freqtrim.tokenizer_merges(student) == [("a", "b"), ("c", "d"), ("ab", "c")]  # rank order kept
    assert student.encode("abcd", add_special_tokens=False).tokens == ["ab", "cd"]  # (c, d) ranks before (ab, c)
    assert student.encode("abc", add_special_tokens=False).tokens == ["abc"]
    assert json.loads(student.to_str())["normalizer"] is None


def test_counting_follows_the_arm_regex(hand):
    texts = ["(x\n_ab\n12\nab"]
    assert subset(hand, texts, 3, "qwen")[0] == {"(x", "_ab", "ab"}  # Qwen2 glues "(" and "_" to letters
    assert subset(hand, texts, 1, "fine-a")[0] == {"ab"}  # form A: "(" + "x", "_" + "ab"
    assert subset(hand, texts, 2, "fine-b")[0] == {"_ab", "ab"}  # form B keeps "_" + letters
    with pytest.raises(freqtrim.FreqtrimError, match="too small"):
        subset(hand, texts, 2, "fine-a")  # the slots never go to tokens form A cannot produce


def _encode_with(student, pattern, text):
    other = Tokenizer.from_str(student.to_str())
    other.pre_tokenizer = tok.split_pre_tokenizer(pattern)
    return other.encode(text, add_special_tokens=False).tokens


def test_coarser_pretokenizer_is_a_noop_except_whitespace(hand, freqtrim_dirs):
    """The notebook's structural fact: under a Qwen2-regex subset, keeping identifiers and integers whole
    changes nothing, and context-dependent whitespace (indentation + the next word) is the exception."""
    texts = ["x_ab\n12\n(x\nabc\n    x\n"]
    keep, _ = subset(hand, texts, 7)
    assert keep == {"_ab", "ab", "(x", "abc", "ĠĠĠ", "ĠĠ", "Ġx"}  # never ĠĠĠĠ: Qwen2 leaves one space to the word
    student = freqtrim.build_tokenizer(hand, keep | set(ALPHABET), freqtrim.REGEXES["qwen"])
    words = "x_ab = 12 + (x_ab12)\n"
    assert _encode_with(student, COARSE_WORDS, words) == student.encode(words, add_special_tokens=False).tokens
    assert "12" not in keep and student.encode("12", add_special_tokens=False).tokens == ["1", "2"]
    assert student.encode("    x", add_special_tokens=False).tokens == ["ĠĠĠ", "Ġx"]
    assert _encode_with(student, COARSE_INDENT, "    x") == ["ĠĠ", "ĠĠ", "x"]
    learned = tok.load(freqtrim_dirs["qwen"])
    for text in ("def left_idx(x1, y2):\n    return x1 + 12345 * y2\n", "total_count += values[index2] / 300\n"):
        assert _encode_with(learned, COARSE_WORDS, text) == learned.encode(text, add_special_tokens=False).tokens


def test_fixture_builds_pass_the_checks(freqtrim_dirs, qwen_teacher):
    for directory in freqtrim_dirs.values():
        assert freqtrim.check_directory(directory, qwen_teacher) == []
        assert tok.load(directory).get_vocab_size() == 300


def test_digits_one_per_token_although_the_teacher_has_longer_ones(freqtrim_dirs, qwen_teacher):
    teacher = freqtrim.load_teacher(qwen_teacher, REF)
    assert any(len(t) > 1 and t.isdigit() for t in teacher.vocab)  # the cl100k-style learning left some
    for directory in freqtrim_dirs.values():
        student = tok.load(directory)
        assert student.encode("1234567890", add_special_tokens=False).tokens == list("1234567890")
        assert not [t for t in student.get_vocab() if len(t) > 1 and any(c.isdigit() for c in t)]


def test_manifest(freqtrim_dirs, qwen_teacher, corpus_spec, tmp_path):
    manifest = json.loads((freqtrim_dirs["fine-a"] / tok.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert (manifest["kind"], manifest["regex"], manifest["split_pattern"]) == (
        "freqtrim",
        "fine-a",
        freqtrim.REGEXES["fine-a"],
    )
    assert manifest["tokenizer_sha256"] == tok.tokenizer_sha256(freqtrim_dirs["fine-a"])
    teacher = manifest["teacher"]
    assert (teacher["repo"], teacher["revision"], teacher["licence"]) == (REF.repo, REF.revision, REF.licence)
    assert teacher["file_sha256"] == corpus.file_sha256(qwen_teacher) and teacher["normalizer"] == {"type": "NFC"}
    assert manifest["normalizer"] is None
    assert manifest["subset"]["size"] == 295 and 0 < manifest["subset"]["reachable_on_train"] <= 300
    assert manifest["corpus"]["spec_sha256"] == corpus.file_sha256(corpus_spec)
    c = corpus.load_corpus(corpus_spec)
    train = [d.source.name for d in corpus.documents(c) if corpus.split_of(c, d) == "train"]
    assert {s["name"]: s["documents"] for s in manifest["sources"]} == {n: train.count(n) for n in ("code", "text")}
    # The vocabulary identity ignores added tokens; the file sha256 does not.
    data = json.loads(qwen_teacher.read_text(encoding="utf-8"))
    data["added_tokens"] = data["added_tokens"][:1]
    other = tmp_path / "tokenizer.json"
    other.write_text(json.dumps(data), encoding="utf-8")
    loaded = freqtrim.load_teacher(other, REF)
    assert loaded.identity == teacher["vocab_identity"] and loaded.file_sha256 != teacher["file_sha256"]


def test_build_is_deterministic(tmp_path, corpus_spec, qwen_teacher, freqtrim_dirs):
    again = freqtrim.build(corpus.load_corpus(corpus_spec), tmp_path / "again", qwen_teacher, REF, regex="qwen",
                           vocab_size=300)  # fmt: skip
    for name in (tok.TOKENIZER_FILE, tok.CONFIG_FILE, tok.MANIFEST_FILE):
        assert (again / name).read_bytes() == (freqtrim_dirs["qwen"] / name).read_bytes()


def _tamper(directory, tmp_path, edit):
    copy = tmp_path / "tampered"
    copy.mkdir()
    for name in (tok.TOKENIZER_FILE, tok.CONFIG_FILE, tok.MANIFEST_FILE):
        (copy / name).write_bytes((directory / name).read_bytes())
    data = json.loads((copy / tok.TOKENIZER_FILE).read_text(encoding="utf-8"))
    edit(data)
    (copy / tok.TOKENIZER_FILE).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return copy


def _drop_parent(data):
    """Remove a merged parent of the last merge (and the merges naming it), keeping its child."""
    model = data["model"]
    a, b = model["merges"][-1]
    parent = a if len(a) > 1 else b
    model["vocab"] = {t: i for t, i in model["vocab"].items() if t != parent}
    model["merges"] = [m for m in model["merges"] if parent not in m and "".join(m) != parent]


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (lambda d: d["model"]["vocab"].update({"ĠzzqqQ": 299}), "not teacher tokens"),
        (_drop_parent, "lack a merge parent"),
        (lambda d: d["model"]["merges"].reverse(), "merges differ"),
        (lambda d: d.update(normalizer={"type": "NFC"}), "has a normaliser"),
        (lambda d: d["model"]["merges"].append(["Ġzz", "q"]), "does not load"),
    ],
)
def test_checks_find_damage(freqtrim_dirs, qwen_teacher, tmp_path, edit, expected):
    teacher = freqtrim.load_teacher(qwen_teacher, REF)
    problems = freqtrim.check(_tamper(freqtrim_dirs["qwen"], tmp_path, edit), teacher)
    assert any(expected in p for p in problems), problems


def test_teacher_refusals(tmp_path, qwen_teacher):
    sha = corpus.file_sha256(qwen_teacher)
    with pytest.raises(freqtrim.FreqtrimError, match="40-hex"):
        freqtrim.load_teacher(qwen_teacher, freqtrim.TeacherRef("r", "main", "apache-2.0"))
    with pytest.raises(freqtrim.FreqtrimError, match="expected"):
        freqtrim.load_teacher(qwen_teacher, freqtrim.TeacherRef(REF.repo, REF.revision, REF.licence, "0" * 64))
    assert freqtrim.load_teacher(qwen_teacher, freqtrim.TeacherRef(REF.repo, REF.revision, REF.licence, sha))
    data = json.loads(qwen_teacher.read_text(encoding="utf-8"))
    for name, edit, message in (
        ("pattern", lambda d: d["pre_tokenizer"]["pretokenizers"][0]["pattern"].update(Regex=r"\s+"), "verbatim"),
        ("merges", lambda d: d["model"].update(ignore_merges=True), "ignore_merges"),
        ("model", lambda d: d["model"].update(type="Unigram"), "not BPE"),
    ):
        bad = json.loads(json.dumps(data))
        edit(bad)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(freqtrim.FreqtrimError, match=message):
            freqtrim.load_teacher(path, REF)


def test_build_refusals(tmp_path, corpus_spec, qwen_teacher, freqtrim_dirs):
    c = corpus.load_corpus(corpus_spec)
    with pytest.raises(FileExistsError):
        freqtrim.build(c, freqtrim_dirs["qwen"], qwen_teacher, REF, regex="qwen", vocab_size=300)
    with pytest.raises(freqtrim.FreqtrimError, match="regex must be one of"):
        freqtrim.build(c, tmp_path / "x", qwen_teacher, REF, regex="qwen2", vocab_size=300)
    with pytest.raises(freqtrim.FreqtrimError, match="too small"):
        freqtrim.build(c, tmp_path / "y", qwen_teacher, REF, regex="qwen", vocab_size=tok.VOCAB_SIZE)


def test_cli(tmp_path, capsys, corpus_spec, qwen_teacher, freqtrim_dirs):
    args = ["--teacher", str(qwen_teacher), "--teacher-repo", REF.repo, "--teacher-revision", REF.revision]
    args += ["--teacher-licence", REF.licence, "--vocab-size", "300"]
    out = tmp_path / "cli"
    assert tok.main(["freqtrim", str(corpus_spec), str(out), *args, "--regex", "fine-b"]) == 0
    assert json.loads(capsys.readouterr().out)["sha256"] == tok.tokenizer_sha256(freqtrim_dirs["fine-b"])
    assert tok.main(["freqtrim-check", str(out), "--teacher", str(qwen_teacher)]) == 0
    assert capsys.readouterr().out.strip() == "freqtrim checks pass"
    assert tok.main(["freqtrim", str(corpus_spec), str(tmp_path / "bad"), *args, "--regex", "gpt2"]) == 1
    assert "regex must be one of" in capsys.readouterr().err
    tampered = _tamper(out, tmp_path, lambda d: d["model"]["merges"].reverse())
    assert tok.main(["freqtrim-check", str(tampered), "--teacher", str(qwen_teacher)]) == 1
    assert "merges differ" in capsys.readouterr().out
