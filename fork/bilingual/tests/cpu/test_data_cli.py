"""The I4 commands end to end through `python -m surogate_bilingual` (manifest, split, decontam, tokenizer,
synth, shard), and their --dry-run, which writes nothing."""

import json
from pathlib import Path

from surogate_bilingual.__main__ import COMMANDS, is_implemented, main

TRACK_S_SPEC = Path(__file__).resolve().parents[2] / "examples" / "track_s.toml"


def test_i4_commands_are_implemented():
    for name in ("tokenizer", "manifest", "split", "decontam", "shard", "synth"):
        assert is_implemented(name) and COMMANDS[name].step == "I4", name


def _tree(path):
    return sorted(p.relative_to(path).as_posix() for p in path.rglob("*"))


def test_i4_pipeline_through_the_command_line(tmp_path, corpus_writer, capsys):
    corpus = tmp_path / "corpus"
    spec = corpus_writer(corpus, n=300)
    w = tmp_path / "work"
    w.mkdir()
    before = _tree(tmp_path)
    assert main(["manifest", "--spec", str(spec), "--root", str(corpus), "--dry-run"]) == 0
    assert main(["synth", "--spec", str(TRACK_S_SPEC), "--seed", "3", "--dry-run"]) == 0
    assert _tree(tmp_path) == before  # dry runs write nothing
    assert main(["manifest", "--spec", str(spec), "--root", str(corpus), "--out", str(w / "m.json")]) == 0
    assert "warning: the licence of source 'zh-nat' is not verified" in capsys.readouterr().out
    assert main(["split", "assign", "--manifest", str(w / "m.json"), "--seed", "1", "--out", str(w / "s.json")]) == 0
    assert (
        main(["decontam", "--manifest", str(w / "m.json"), "--split", str(w / "s.json"), "--out", str(w / "o.json")])
        == 0
    )
    assert json.loads((w / "o.json").read_text())["format"] == "bilingual-overlap/1"
    assert (
        main(
            [
                "split",
                "materialize",
                "--manifest",
                str(w / "m.json"),
                "--split",
                str(w / "s.json"),
                "--exclude",
                str(w / "o.json"),
                "--out",
                str(w / "splits"),
            ]
        )
        == 0
    )
    derived = w / "splits" / "manifest.json"
    before = _tree(tmp_path)
    assert main(["tokenizer", "train", "--manifest", str(derived), "--dry-run"]) == 0
    assert _tree(tmp_path) == before
    assert main(["tokenizer", "train", "--manifest", str(derived), "--vocab", "600", "--out", str(w / "tok.json")]) == 0
    assert main(["tokenizer", "check", str(w / "tok.json"), "--samples", "100"]) == 0
    assert main(["synth", "--spec", str(TRACK_S_SPEC), "--seed", "3", "--profile", "ci", "--out", str(w / "ts")]) == 0
    assert (w / "ts" / "datacard.json").is_file()
    assert (
        main(
            [
                "shard",
                "--manifest",
                str(derived),
                "--split",
                "zh_train",
                "--tokenizer",
                str(w / "tok.json"),
                "--route",
                "zh",
                "--add",
                str(w / "ts" / "zh" / "track_s.jsonl"),
                "--seq",
                "64",
                "--out",
                str(w / "zh_shard"),
            ]
        )
        == 0
    )
    header = json.loads((w / "zh_shard" / "shard.json").read_text())
    assert header["split"] == "zh_train" and header["n_docs"] > 0 and len(header["inputs"]) == 2
    before = _tree(tmp_path)
    assert (
        main(
            [
                "shard",
                "--manifest",
                str(derived),
                "--split",
                "seed_train",
                "--tokenizer",
                str(w / "tok.json"),
                "--route",
                "seed",
                "--seq",
                "64",
                "--dry-run",
            ]
        )
        == 0
    )
    assert _tree(tmp_path) == before
    # D_seed through the command line: the mixture is built first, its card and tag minimums go into the header
    lines = (w / "ts" / "seed" / "track_s.jsonl").read_text(encoding="utf-8").splitlines()[:5]
    (w / "ts_seed_part.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert (
        main(
            [
                "shard",
                "--manifest",
                str(derived),
                "--split",
                "seed_train",
                "--tokenizer",
                str(w / "tok.json"),
                "--route",
                "seed",
                "--seq",
                "64",
                "--seed-mix",
                "--add",
                str(w / "ts_seed_part.jsonl"),
                "--out",
                str(w / "seed_shard"),
            ]
        )
        == 0
    )
    header = json.loads((w / "seed_shard" / "shard.json").read_text())
    card = header["extra"]["seed_mix"]
    assert card["kind"] == "d_seed" and card["max_share_error"] <= 0.005
    assert header["min_tag_counts"] == card["min_tag_counts"] and header["mixed_window_fraction"]["floor"] == 0.01
    assert all(header["tag_counts"][t] >= n for t, n in card["min_tag_counts"].items())
    # refusals exit 1 with a message instead of a traceback
    assert (
        main(
            [
                "shard",
                "--manifest",
                str(derived),
                "--split",
                "seed_train",
                "--tokenizer",
                str(w / "tok.json"),
                "--seed-mix",
                "--add",
                str(w / "ts" / "seed" / "track_s.jsonl"),
                "--dry-run",
            ]
        )
        == 1
    )
    assert "usable tokens" in capsys.readouterr().err
    assert main(["tokenizer", "train", "--manifest", str(derived), "--split", "dev", "--out", str(w / "t2.json")]) == 1
    assert "never trains on a held-out split" in capsys.readouterr().err
