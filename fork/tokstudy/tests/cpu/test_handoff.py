"""The tokenizer hand-off (surogate_tokstudy.handoff, the workflow's per-project wiring) and the committed arm
registries arms/tlc.toml and arms/bilingual.toml.

The registries are checked as they are committed (unpinned, so nothing loads them before the owner pins them) and
then pinned in a copy of the repository layout with tokenizers built by the projects' own code on SYNTHETIC FIXTURE
corpora: the sha256 is written into the one tokenizer line of each arm, the projects' records are checked against
the corpus the stream reads, and every way a tokenizer can come from another corpus or split is refused.
"""

import json
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from surogate_tokstudy import arms
from surogate_tokstudy.canonical import TokstudyError, sha256_file
from surogate_tokstudy.handoff import handoff
from surogate_tokstudy.models.build import positional_limit
from surogate_tokstudy.models.hnet import HNetConfig
from surogate_tokstudy.prereg import load_prereg

TOKSTUDY = Path(__file__).resolve().parents[2]
REPO = TOKSTUDY.parents[1]
REGISTRIES = {"tlc": TOKSTUDY / "arms" / "tlc.toml", "bilingual": TOKSTUDY / "arms" / "bilingual.toml"}
PREREGS = {"tlc": TOKSTUDY / "prereg" / "tlc.toml", "bilingual": TOKSTUDY / "prereg" / "bilingual.toml"}


# --- the committed registries ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("project", ["tlc", "bilingual"])
def test_the_committed_registry_agrees_with_its_prereg_and_waits_for_its_tokenizers(project):
    data = tomllib.loads(REGISTRIES[project].read_text(encoding="utf-8"))
    prereg = load_prereg(PREREGS[project])
    assert data["table"] == prereg.table
    entries = {a["id"]: a for a in data["arm"]}
    seeds = tomllib.loads(PREREGS[project].read_text(encoding="utf-8"))["seeds"]
    assert set(entries) == set(seeds)  # every preregistered arm, nothing else
    for arm_id, entry in entries.items():
        assert len(entry["seeds"]) >= seeds[arm_id] and set(entry["stages"]) <= set(prereg.stages)
        if entry["kind"] == "subword":
            assert entry["tokenizer"]["sha256"] == arms.PLACEHOLDER
            path = (REGISTRIES[project].parent / entry["tokenizer"]["path"]).resolve()
            assert path.is_relative_to(REPO / "data" / "tokstudy" / project), path
            ignored = subprocess.run(["git", "check-ignore", "-q", "--no-index", str(path.relative_to(REPO))], cwd=REPO)
            assert ignored.returncode == 0, f"{path} would be committed"
        if entry["kind"] == "hnet":
            HNetConfig(**entry["model"]["hnet"])
    with pytest.raises(TokstudyError, match="not pinned yet"):
        arms.load_arms(REGISTRIES[project])


def test_every_bilingual_arm_holds_a_row_of_k_train_positions():
    """The trunk's max_seq bounds every arm's rows (an H-Net's chunks, one per byte at worst), so the train setup
    would refuse an arm whose limit is below K_train; checked without the tokenizer files."""
    k_train = load_prereg(PREREGS["bilingual"]).k_train
    data = tomllib.loads(REGISTRIES["bilingual"].read_text(encoding="utf-8"))
    for entry in data["arm"]:
        spec = arms.ArmSpec(
            entry["id"],
            "bilingual-B",
            entry["kind"],
            None,
            {},
            entry["model"],
            "bilingual_trunk",
            tuple(entry["seeds"]),
            tuple(entry["stages"]),
        )
        assert positional_limit(spec) >= k_train, entry["id"]


# --- pinning in a copy of the repository layout ----------------------------------------------------------------------


def _layout(tmp_path, project):
    registry = tmp_path / "fork" / "tokstudy" / "arms" / f"{project}.toml"
    registry.parent.mkdir(parents=True)
    shutil.copy(REGISTRIES[project], registry)
    return registry


def _tlc_tokenizer(spec, out, vocab_size=300, pretokenizer="qwen2"):
    from surogate_tlc.data import corpus, tokenizer

    return tokenizer.train_tokenizer(corpus.load_corpus(spec), out, vocab_size=vocab_size, pretokenizer=pretokenizer)


def test_pin_writes_the_tlc_sha256_and_keeps_the_rest_of_the_file(tmp_path, tlc_corpus_writer, capsys):
    registry = _layout(tmp_path, "tlc")
    spec = tlc_corpus_writer(tmp_path / "corpus")
    leader = _tlc_tokenizer(spec, tmp_path / "data" / "tokstudy" / "tlc" / "tok" / "leader")
    sha = sha256_file(leader / "tokenizer.json")
    before = registry.read_text(encoding="utf-8")
    assert arms.main([str(registry), "--tlc", str(spec)]) == 1  # without --pin a placeholder is refused
    assert "is not pinned yet" in capsys.readouterr().err
    record = tmp_path / "handoff.json"
    assert arms.main([str(registry), "--tlc", str(spec), "--pin", "--record", str(record)]) == 0
    assert f"N0\t{sha}\tbpe qwen2 vocab 300\tpinned" in capsys.readouterr().out
    after = registry.read_text(encoding="utf-8")
    assert after == before.replace('sha256 = "unpinned"', f'sha256 = "{sha}"')  # comments and layout kept
    loaded = {a.id: a for a in arms.load_arms(registry)}
    assert loaded["N0"].tokenizer.sha256 == sha and loaded["N1"].kind == "hnet"
    doc = json.loads(record.read_text(encoding="utf-8"))
    assert doc["format"] == "tokstudy-handoff/1" and doc["table"] == "tlc-N" and doc["pinned"] == ["N0"]
    assert doc["registry_sha256"] == sha256_file(registry) and doc["other_arms"] == ["N1", "N2"]
    (n0,) = doc["arms"]
    assert n0["tokenizer"] == {
        "path": "../../../data/tokstudy/tlc/tok/leader/tokenizer.json",
        "sha256": sha,
        "previous": "unpinned",
    }
    assert n0["provenance"]["corpus"]["spec_sha256"] == sha256_file(spec) == doc["corpus"]["spec_sha256"]
    assert n0["provenance"]["kind"] == "bpe" and n0["provenance"]["vocab_size"] == 300
    # pinned: the check passes and writes nothing; a rebuilt tokenizer is refused until it is pinned again
    assert arms.main([str(registry), "--tlc", str(spec)]) == 0 and registry.read_text(encoding="utf-8") == after
    shutil.rmtree(leader)
    _tlc_tokenizer(spec, leader, pretokenizer="gpt2")
    capsys.readouterr()
    assert arms.main([str(registry), "--tlc", str(spec)]) == 1
    assert f"is pinned to {sha}" in capsys.readouterr().err


def test_a_tlc_tokenizer_of_another_corpus_or_record_is_refused(tmp_path, tlc_corpus_writer):
    registry = _layout(tmp_path, "tlc")
    spec = tlc_corpus_writer(tmp_path / "corpus")
    leader = _tlc_tokenizer(spec, tmp_path / "data" / "tokstudy" / "tlc" / "tok" / "leader")
    other = tmp_path / "other" / "spec.toml"
    shutil.copytree(spec.parent, other.parent)
    other.write_text(other.read_text(encoding="utf-8").replace("seed = 5", "seed = 6"), encoding="utf-8")
    with pytest.raises(TokstudyError, match="the stream reads another corpus"):
        handoff(registry, tlc=other, pin=True)
    with pytest.raises(TokstudyError, match="give --tlc SPEC"):
        handoff(registry, bilingual=spec, pin=True)
    manifest = leader / "tokenizer_manifest.json"
    record = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps({**record, "tokenizer_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(TokstudyError, match="records tokenizer sha256 0000"):
        handoff(registry, tlc=spec, pin=True)
    manifest.write_text(json.dumps({**record, "corpus": {**record["corpus"], "trained_on": "all"}}), encoding="utf-8")
    with pytest.raises(TokstudyError, match="not the train split"):
        handoff(registry, tlc=spec, pin=True)
    manifest.unlink()
    with pytest.raises(TokstudyError, match="cannot read a TLC tokenizer manifest"):
        handoff(registry, tlc=spec, pin=True)
    assert 'sha256 = "unpinned"' in registry.read_text(encoding="utf-8")  # nothing was written


def _bilingual_tokenizers(derived, root, *, split="seed_train"):
    from surogate_bilingual.data import tokenizer

    files = [derived.root / e["path"] for e in derived.files(split=split)]
    recipes = {
        "B0": (400, "bytelevel"),
        "B1": (300, "bytelevel"),
        "B2": (400, "qwen2-digits"),
        "B3": (400, "han-isolate"),
    }
    for arm_id, (vocab, pre) in recipes.items():
        out = root / arm_id / "tokenizer.json"
        tokenizer.train_tokenizer(files, vocab, out, derived, split, pretokenizer=pre)
    return recipes


def test_pin_checks_every_bilingual_tokenizer_against_the_derived_manifest(tmp_path, bilingual_built):
    derived_path, derived, _ = bilingual_built
    registry = _layout(tmp_path, "bilingual")
    tok = tmp_path / "data" / "tokstudy" / "bilingual" / "tok"
    recipes = _bilingual_tokenizers(derived, tok)
    doc = handoff(registry, bilingual=derived_path, pin=True)
    assert doc["pinned"] == ["B0", "B1", "B2", "B3"] and doc["other_arms"] == ["B4", "B5"]
    for arm in doc["arms"]:
        p = arm["provenance"]
        assert (p["vocab_size"], p["pretokenizer"]) == recipes[arm["id"]]
        assert p["corpus"] == {"manifest_sha256": derived.sha256, "split": "seed_train"}
        assert arm["tokenizer"]["sha256"] == sha256_file(tok / arm["id"] / "tokenizer.json")
    loaded = {a.id: a for a in arms.load_arms(registry)}
    assert [loaded[i].tokenizer.sha256 for i in ("B0", "B1", "B2", "B3")] == [
        a["tokenizer"]["sha256"] for a in doc["arms"]
    ]
    # a tokenizer that saw another split is refused, whatever the registry says
    shutil.rmtree(tok / "B1")
    from surogate_bilingual.data import tokenizer

    other = sorted({e["split"] for e in derived.doc["files"]} - {"seed_train", *tokenizer.HELDOUT})[0]
    files = [derived.root / e["path"] for e in derived.files(split=other)]
    tokenizer.train_tokenizer(files, 300, tok / "B1" / "tokenizer.json", derived, other)
    with pytest.raises(TokstudyError, match="not seed_train"):
        handoff(registry, bilingual=derived_path, pin=True)


def test_a_registry_whose_tokenizer_line_cannot_be_rewritten_is_refused(tmp_path, tlc_corpus_writer):
    """--pin rewrites one line per arm; a tokenizer written as a sub-table is refused, and the file is unchanged."""
    registry = _layout(tmp_path, "tlc")
    spec = tlc_corpus_writer(tmp_path / "corpus")
    _tlc_tokenizer(spec, tmp_path / "data" / "tokstudy" / "tlc" / "tok" / "leader")
    registry.write_text(
        'format = "tokstudy-arms/1"\ntable = "tlc-N"\n\n[[arm]]\nid = "N0"\nkind = "subword"\nseeds = [0]\n'
        'stages = ["decision"]\n\n[arm.tokenizer]\npath = "../../../data/tokstudy/tlc/tok/leader/tokenizer.json"\n'
        'sha256 = "unpinned"\n',
        encoding="utf-8",
    )
    before = registry.read_bytes()
    with pytest.raises(TokstudyError, match="on one line"):
        handoff(registry, tlc=spec, pin=True)
    assert registry.read_bytes() == before
