"""Binding probes (surogate_tlc.eval.probes, README: 绑定探针): the scoring core on toy encoders and scorers, the
floors of the summary, and evaluate_suite and `eval --suite` with a stand-in for the native scorer."""

import dataclasses
import json
import random

import numpy as np
import pytest
from surogate_tlc.__main__ import main
from surogate_tlc.data import corpus, suites, synth
from surogate_tlc.data.suites import Item
from surogate_tlc.eval import probes

FRAME = [1, 3]


def chars(text):
    return [ord(c) for c in text], [(i, i + 1) for i in range(len(text))]


def merging(text):
    """Characters, except that a space and the letter after it are one token (as in GPT-2-style pre-tokenizers)."""
    ids, offsets, i = [], [], 0
    while i < len(text):
        j = i + 2 if text[i] == " " and text[i + 1 : i + 2].isalpha() else i + 1
        ids.append(ord(text[i]) if j == i + 1 else 1000 + ord(text[i + 1]))
        offsets.append((i, j))
        i = j
    return ids, offsets


def item(prompt, candidates, gold, **fields):
    fields = {"id": "i", "task": "T1", "tail": "\n", "heuristics": {}, "strata": {}, **fields}
    return Item(prompt=prompt, candidates=candidates, gold=gold, **fields)


def test_rows_score_the_candidate_and_the_first_tail_character():
    it = item("q = ", ["'a'", "'bb'"], "'bb'", tail="\nz = 1\n", anchor=0)
    rows = probes.rows_of(it, chars, FRAME, 64)
    assert rows["rows"] == [FRAME + chars("q = 'a'\n")[0], FRAME + chars("q = 'bb'\n")[0]]
    assert rows["first"] == [2 + 4, 2 + 4]
    assert rows["distance"] == 4 and rows["pieces"] == 4  # four context tokens end after offset 0; 'bb' is 4
    # Not at the start of its document: no framing.
    assert probes.rows_of(item("q = ", ["'a'"], "'a'", start=7), chars, FRAME, 64)["first"] == [4]


def test_a_token_across_the_prompt_end_moves_the_shared_context():
    it = item("z = ", ["a", "bc"], "a", tail=" + 1\n", query=None)
    rows = probes.rows_of(it, merging, FRAME, 64)
    # " a" starts in the prompt: the context ends before it for every candidate; the space after is scored too.
    assert rows["first"] == [2 + 3, 2 + 3]
    assert rows["rows"][0][2:] == [ord("z"), ord(" "), ord("="), 1000 + ord("a"), ord(" ")]
    assert rows["rows"][1][2:] == [ord("z"), ord(" "), ord("="), 1000 + ord("b"), ord("c"), ord(" ")]


def test_items_that_cannot_be_scored():
    def unstable(text):  # the context's tokens depend on the candidate
        ids, offsets = chars(text)
        return ([0] if "_" in text else [1]) + ids[1:], offsets

    assert probes.rows_of(item("ab = ", ["_x", "y"], "y"), unstable, FRAME, 64) == {"skip": "context"}
    natural = item("x" * 40 + " = ", ["'a'", "'b'"], "'a'")  # 2 framing + 43 context + 4 scored tokens
    kept = probes.rows_of(natural, chars, FRAME, 40)  # natural code keeps its last tokens
    assert all(len(r) == 40 for r in kept["rows"]) and kept["first"] == [36, 36]
    assert probes.rows_of(natural, chars, FRAME, 4)["skip"] == "too_long"  # the candidate alone does not fit
    program = dataclasses.replace(natural, program=natural.prompt + "'a'\n")
    assert probes.rows_of(program, chars, FRAME, 48)["skip"] == "too_long"  # a whole program or nothing
    assert "rows" in probes.rows_of(program, chars, FRAME, 49)


def test_score_rows_sums_the_scored_log_probabilities():
    rows = [[1, 3, 10, 11, 12], [1, 3, 20, 21], [1, 3, 30, 31, 32, 33]]
    firsts = [3, 2, 4]
    calls = []

    def scorer(inputs, targets, positions):
        assert inputs.shape == targets.shape == positions.shape == (2, 8) and inputs.dtype == np.int32
        calls.append((inputs.copy(), targets.copy()))
        return np.where(targets == -100, 0.0, -targets / 100.0)

    scores = probes.score_rows(rows, firsts, scorer, batch=2, seq_len=8, pad_id=0)
    assert scores == pytest.approx([-(11 + 12) / 100, -(20 + 21) / 100, -(32 + 33) / 100])
    assert len(calls) == 2
    targets = calls[0][1]
    assert targets[0].tolist() == [-100, -100, 11, 12, -100, -100, -100, -100]
    assert (calls[1][1][1] == -100).all() and (calls[1][0][1] == 0).all()  # the filler row


def prefers(letter):
    """A stand-in model: every target costs 1 nat, `letter` 0.1."""
    return lambda inputs, targets, positions: np.where(targets == -100, 0, np.where(targets == ord(letter), -0.1, -1))


def test_probe_picks_the_best_scored_candidate():
    items = [
        item("q = ", ["'a'", "'b'"], "'b'", id="right"),
        item("q = ", ["'a'", "'b'"], "'a'", id="wrong"),
        item("q" * 30 + " = ", ["'a'", "'b'"], "'a'", id="long", program="q" * 30 + " = 'a'\n"),
    ]
    results = probes.probe(items, chars, prefers("b"), frame=FRAME, batch=3, seq_len=16, pad_id=0)
    assert [(r["id"], r["status"]) for r in results] == [("right", "scored"), ("wrong", "scored"), ("long", "too_long")]
    assert results[0]["correct"] and not results[1]["correct"] and results[1]["choice"] == "'b'"
    assert results[0]["scores"] == pytest.approx([-1 * 4, -1 * 3 - 0.1])


def _scored(name, cell, gold, choice, **heuristics):
    it = item(
        "x", ["'a'", "'b'", "'c'", "'d'"], gold, id=name, strata={"pattern": cell, **STRATA}, heuristics=heuristics
    )
    result = {"id": name, "status": "scored", "choice": choice, "correct": choice == gold}
    return it, {**result, "distance": 100, "pieces": 1}


STRATA = {"live": 4, "hops": 1, "overwrites": 0, "depth": 0, "family": "assign", "pool": "seen", "heldout": []}


def test_summary_floors():
    pairs = [
        # flat: same_name is exact; recent right on half; one choice is the stale value
        _scored("a1", "flat", "'a'", "'a'", same_name="'a'", recent="'a'", stale="'b'"),
        _scored("a2", "flat", "'a'", "'a'", same_name="'a'", recent="'a'", stale="'b'"),
        _scored("a3", "flat", "'a'", "'a'", same_name="'a'", recent="'c'", stale="'b'"),
        _scored("a4", "flat", "'a'", "'b'", same_name="'a'", recent="'c'", stale="'b'"),
        # shadow-out: no heuristic is exact
        _scored("b1", "shadow-out", "'a'", "'a'", same_name="'b'", recent="'a'", stale=None),
        _scored("b2", "shadow-out", "'a'", "'d'", same_name="'a'", recent="'c'", stale=None),
    ]
    items, results = zip(*pairs, strict=True)
    summary = probes.summarize(list(items), list(results))["T1"]
    cells = {c["pattern"]: c for c in summary["cells"]}
    assert cells["flat"]["exact"] == ["same_name"] and cells["flat"]["floor"] == 1
    assert cells["flat"]["shortcut_floor"] == 0.5 and cells["flat"]["accuracy"] == 0.75
    assert cells["shadow-out"]["exact"] == [] and cells["shadow-out"]["floor"] == 0.5
    assert summary["accuracy"] == pytest.approx(4 / 6) and summary["chance"] == 0.25
    assert summary["above_floor"] == pytest.approx((3 - 4 * 1 + 1 - 2 * 0.5) / 6)
    assert summary["above_shortcut_floor"] == pytest.approx((3 - 4 * 0.5 + 1 - 2 * 0.5) / 6)
    # disagreement: a3, a4 (recent and stale wrong), and no shadow-out item (one heuristic is right on each)
    assert summary["disagreement"] == {"items": 2, "accuracy": 0.5}
    assert summary["stale_rate"] == 0.25
    keys = "depth distance family heldout hops live overwrites pattern pieces pool"
    assert sorted(summary["by"]) == keys.split()
    assert list(summary["by"]["distance"]) == ["<=128"] and summary["by"]["heldout"]["none"]["items"] == 6
    assert summary["by"]["pattern"]["flat"]["above_floor"] == pytest.approx(-0.25)


@pytest.fixture
def suite_file(tmp_path):
    """T1 items of two small cells, and one T3 item that is not at its document's start."""
    rng = random.Random(0)
    items = [
        synth.make_item("T1", {**cell, "family": "assign"}, rng, "seen", "eval", f"T1-{i}")
        for i, cell in enumerate(c for c in synth.cells("T1", ["flat", "shadow-in"]) if c["live"] <= 4)
    ]
    heuristics = {"recent": "count", "frequent": "total"}
    t3 = item("    total = ", ["count", "total"], "total", tail=" + 1\n", start=40, task="T3", id="T3:x:1")
    items.append(dataclasses.replace(t3, heuristics=heuristics, strata={"candidates": "2"}))
    path = tmp_path / "suite.jsonl"
    suites.write_items(path, items)
    return path, items


def test_evaluate_suite_runs_the_core_with_the_model_tokenizer(small_model, shard_dirs, suite_file):
    path, items = suite_file
    seen = []

    def uniform(inputs, targets, positions):
        seen.append(inputs.copy())
        assert inputs.shape == (4, 512) and inputs.dtype == np.int32
        return np.where(targets == -100, 0.0, -1.0)

    result = probes.evaluate_suite(small_model, shard_dirs[512], path, batch=4, scorer=uniform)
    assert result["suite_sha256"] == corpus.file_sha256(path)
    assert result["seq_len"] == 512 and len(result["results"]) == len(items)
    assert all(r["status"] == "scored" for r in result["results"])
    assert set(result["summary"]) == {"T1", "T3"} and result["summary"]["T1"]["items"] == len(items) - 1
    rows = np.concatenate(seen)
    framed = rows[(rows[:, 0] == 1) & (rows[:, 1] == 3)]  # bos, code_start: the T1 rows only
    assert len(framed) == sum(len(i.candidates) for i in items if i.task == "T1")


def test_eval_command_with_a_suite(small_model, shard_dirs, suite_file, tmp_path, monkeypatch, capsys):
    path, items = suite_file
    monkeypatch.setattr(probes, "native_scorer", lambda *a, **k: prefers("a"))
    out = tmp_path / "probe.json"
    args = ["eval", str(small_model), str(shard_dirs[512]), "--suite", str(path), "--batch", "8", "--out", str(out)]
    assert main(args) == 0
    shown, written = json.loads(capsys.readouterr().out), json.loads(out.read_text())
    assert "results" not in shown and len(written["results"]) == len(items)
    assert shown["summary"] == written["summary"]
