"""Arm registries (surogate_tokstudy.arms, spec step S8): tokstudy-arms/1, sha256 checks, refusals, the CLI."""

import shutil

import pytest
from surogate_tokstudy import arms, codecs
from surogate_tokstudy.canonical import TokstudyError

REGISTRY = """\
format = "tokstudy-arms/1"
table = "tlc-N"

[[arm]]
id = "N0"
kind = "subword"
tokenizer = {{path = "tok/tokenizer.json", sha256 = "{sha}"}}
model = {{profile = "a0"}}
seeds = [0, 1, 2]
stages = ["decision"]

[[arm]]
id = "N1"
kind = "hnet"
model = {{profile = "a0", encoder_layers = 2}}
seeds = [0, 1, 2]
stages = ["decision"]

[[arm]]
id = "N2"
kind = "bytes"
main = "twin_a0"
specials = "tlc"
seeds = [0]
stages = ["decision"]
"""


@pytest.fixture
def registry(tmp_path, bpe_writer):
    _, sha = bpe_writer(tmp_path / "tok" / "tokenizer.json", specials=list(codecs.TLC_SPECIALS))
    path = tmp_path / "tlc.toml"
    path.write_text(REGISTRY.format(sha=sha), encoding="utf-8")
    return path, sha


def test_load_and_codecs(registry):
    path, sha = registry
    loaded = arms.load_arms(path)
    assert [(a.id, a.kind, a.main, a.project) for a in loaded] == [
        ("N0", "subword", "twin_a0", "tlc"),
        ("N1", "hnet", "twin_a0", "tlc"),
        ("N2", "bytes", "twin_a0", "tlc"),
    ]
    assert loaded[0].tokenizer.sha256 == sha and loaded[0].seeds == (0, 1, 2)
    assert all(a.specials == codecs.TLC_SPECIALS for a in loaded)
    subword, hnet, byte = (arms.codec_for(a) for a in loaded)
    assert isinstance(subword, codecs.SubwordCodec) and subword.special_ids == codecs.TLC_SPECIALS
    assert hnet is None and isinstance(byte, codecs.ByteCodec) and byte.vocab_size == 261


def test_arm_sha256_follows_content_not_location(registry, tmp_path, bpe_writer):
    path, _ = registry
    first = [arms.arm_sha256(a) for a in arms.load_arms(path)]
    moved = tmp_path / "moved"
    shutil.copytree(path.parent / "tok", moved / "tok")
    shutil.copy(path, moved / "tlc.toml")
    assert [arms.arm_sha256(a) for a in arms.load_arms(moved / "tlc.toml")] == first
    text = path.read_text(encoding="utf-8").replace("seeds = [0]\n", "seeds = [1]\n")
    (moved / "tlc.toml").write_text(text, encoding="utf-8")
    changed = [arms.arm_sha256(a) for a in arms.load_arms(moved / "tlc.toml")]
    assert changed[:2] == first[:2] and changed[2] != first[2]  # N2's seeds changed, nothing else
    _, other = bpe_writer(moved / "tok" / "tokenizer.json", [("a", "b")], specials=list(codecs.TLC_SPECIALS))
    (moved / "tlc.toml").write_text(REGISTRY.format(sha=other), encoding="utf-8")
    assert arms.arm_sha256(arms.load_arms(moved / "tlc.toml")[0]) != first[0]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda t, sha: t.replace(sha, "0" * 64), "the registry says"),
        (lambda t, sha: t.replace('id = "N2"', 'id = "N1"'), "duplicate arm ids"),
        (lambda t, sha: t.replace('id = "N2"', 'id = "N2"\ntable = "bilingual-B"'), "more than one table"),
        (lambda t, sha: t.replace('table = "tlc-N"', 'table = "tlc-T"'), "table must be one of"),
        (lambda t, sha: t.replace('format = "tokstudy-arms/1"', 'format = "tokstudy-arms/2"'), "known version"),
        (lambda t, sha: t.replace('kind = "hnet"', 'kind = "hnet"\nlr = 1'), "unknown keys"),
        (lambda t, sha: t.replace('kind = "hnet"', 'kind = "chunked"'), "kind must be one of"),
        (lambda t, sha: t.replace('kind = "subword"', 'kind = "bytes"'), "has no tokenizer"),
        (lambda t, sha: t.replace('kind = "bytes"', 'kind = "subword"'), "needs tokenizer"),
        (lambda t, sha: t.replace('main = "twin_a0"', 'main = "bilingual_trunk"'), "trains on"),
        (lambda t, sha: t.replace('specials = "tlc"', 'specials = "bilingual"'), "special ids"),
        (lambda t, sha: t.replace("seeds = [0]", "seeds = []"), "seeds"),
        (lambda t, sha: t.replace("seeds = [0]", "seeds = [0, 0]"), "seeds"),
    ],
)
def test_refusals(registry, edit, message):
    path, sha = registry
    path.write_text(edit(path.read_text(encoding="utf-8"), sha), encoding="utf-8")
    with pytest.raises(TokstudyError, match=message):
        arms.load_arms(path)


def test_a_tokenizer_with_other_special_ids_is_refused(tmp_path, bpe_writer):
    _, sha = bpe_writer(tmp_path / "tok" / "tokenizer.json", specials=list(codecs.BILINGUAL_SPECIALS))
    path = tmp_path / "tlc.toml"
    path.write_text(REGISTRY.format(sha=sha), encoding="utf-8")
    with pytest.raises(TokstudyError, match="special tokens"):
        arms.codec_for(arms.load_arms(path)[0])


def test_command_line(registry, capsys):
    from surogate_tokstudy.__main__ import main

    path, sha = registry
    assert main(["arms", str(path)]) == 0
    lines = capsys.readouterr().out.strip().split("\n")
    assert [line.split("\t")[:3] for line in lines] == [["N0", "tlc-N", "subword"], ["N1", "tlc-N", "hnet"],
                                                         ["N2", "tlc-N", "bytes"]]  # fmt: skip
    path.write_text(path.read_text(encoding="utf-8").replace(sha, "1" * 64), encoding="utf-8")
    assert main(["arms", str(path)]) == 1
    assert "the registry says" in capsys.readouterr().err
