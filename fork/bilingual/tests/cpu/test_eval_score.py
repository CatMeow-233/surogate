"""I6-E1 (candidate ranking) and I6-E2 (scoring rules, skip records, items_sha256); SCI-11, IMPL-15."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from surogate_bilingual.config import profile
from surogate_bilingual.data.tokenizer import load_tokenizer
from surogate_bilingual.eval import items as it
from surogate_bilingual.eval import score
from surogate_bilingual.model.lm import RoutedLM

V = 64


class BigramStub(torch.nn.Module):
    """Returns fixed logits: the row of the current token (a bigram table), recording every batch it sees."""

    def __init__(self, table: torch.Tensor, max_seq: int = 64):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.table = table
        self.cfg = SimpleNamespace(max_seq=max_seq)
        self.calls = []

    def forward(self, tokens, plan=None):
        self.calls.append(tokens.clone())
        return self.table[tokens]


def hand_logprob(table, prompt, cand):
    logp = table.log_softmax(-1)
    prev, total = prompt[-1], 0.0
    for c in cand:
        total += float(logp[prev, c])
        prev = c
    return total


def test_i6_e1_the_choice_is_the_argmax_of_hand_computed_sums_and_means():
    table = torch.randn(V, V, generator=torch.Generator().manual_seed(0), dtype=torch.float64).float()
    prompt = [5, 6, 7]
    cands = [[8], [9], [10, 11], [12, 13, 14], [15, 16]]
    sums = [hand_logprob(table, prompt, c) for c in cands]
    means = [s / len(c) for s, c in zip(sums, cands, strict=True)]
    for norm, expected in (("sum", sums), ("mean", means)):
        out = score.rank_candidates(BigramStub(table), prompt, None, cands, norm)
        assert out.norm == norm
        assert max(abs(a - b) for a, b in zip(out.scores, expected, strict=True)) <= 1e-5
        assert out.pred_index == max(range(len(expected)), key=lambda i: expected[i])
    # auto: mean when lengths differ, sum when every candidate has the same length
    assert score.rank_candidates(BigramStub(table), prompt, None, cands, "auto").norm == "mean"
    assert score.rank_candidates(BigramStub(table), prompt, None, [[8], [9], [10]], "auto").norm == "sum"


def _table_with(row_probs: dict[int, dict[int, float]]) -> torch.Tensor:
    """Logits whose softmax rows are the given probabilities (the rest spread evenly)."""
    table = torch.zeros(V, V)
    for row, probs in row_probs.items():
        rest = (1.0 - sum(probs.values())) / (V - len(probs))
        p = torch.full((V,), rest)
        for col, value in probs.items():
            p[col] = value
        table[row] = p.log()
    return table


def test_i6_e1_sum_and_mean_disagree_on_a_constructed_case():
    # A = [8] with log p = -2; B = [9, 10, 11] with log p = -0.9 per token: sum -2.7 (A wins), mean -0.9 (B wins)
    e = math.exp
    table = _table_with({7: {8: e(-2.0), 9: e(-0.9)}, 9: {10: e(-0.9)}, 10: {11: e(-0.9)}})
    cands = [[8], [9, 10, 11]]
    by_sum = score.rank_candidates(BigramStub(table), [7], None, cands, "sum")
    by_mean = score.rank_candidates(BigramStub(table), [7], None, cands, "mean")
    assert abs(by_sum.logprobs[0] + 2.0) <= 1e-5 and abs(by_sum.logprobs[1] + 2.7) <= 1e-5
    assert by_sum.pred_index == 0 and by_mean.pred_index == 1
    assert score.rank_candidates(BigramStub(table), [7], None, cands, "auto").pred_index == 1  # unequal: mean


def test_i6_e1_only_candidate_tokens_are_scored_and_the_prompt_is_a_shared_prefix():
    table = torch.randn(V, V, generator=torch.Generator().manual_seed(1))
    prompt, cands = [5, 6, 7], [[8, 9], [10], [11, 12, 13]]
    stub = BigramStub(table)
    base = score.rank_candidates(stub, prompt, None, cands, "sum")
    batch = stub.calls[0]
    assert all(row[: len(prompt)].tolist() == prompt for row in batch)  # one shared prefix
    for i, c in enumerate(cands):  # each row holds the prompt and exactly its own candidate (then padding)
        assert batch[i, len(prompt) : len(prompt) + len(c)].tolist() == c
    changed = table.clone()
    changed[5] = torch.randn(V)  # the predictions of the prompt's own tokens
    changed[6] = torch.randn(V)
    assert score.rank_candidates(BigramStub(changed), prompt, None, cands, "sum").logprobs == base.logprobs
    changed[7] = torch.randn(V)  # the row that predicts the first candidate token does count
    assert score.rank_candidates(BigramStub(changed), prompt, None, cands, "sum").logprobs != base.logprobs
    # windows: a prompt longer than max_seq is cut from the left; the bigram scores do not change
    long_prompt = list(range(20, 40)) + prompt
    windowed = score.rank_candidates(BigramStub(table, max_seq=6), long_prompt, None, cands, "sum")
    assert max(abs(a - b) for a, b in zip(windowed.logprobs, base.logprobs, strict=True)) <= 1e-6
    with pytest.raises(ValueError, match="too long"):
        score.rank_candidates(BigramStub(table, max_seq=3), prompt, None, [[1, 2, 3, 4]], "sum")


# --- I6-E2 ---------------------------------------------------------------------------------------------------------


def test_i6_e2_extract_number_uses_digit_boundaries():
    assert score.extract_number("143", "43") is False
    assert score.extract_number("430", "43") is False
    assert score.extract_number("2025", "25") is False
    assert score.extract_number("13 … 2024", ["2024", "3"]) is False
    assert score.extract_number("答案是43", "43") is True
    assert score.extract_number("The answer is 43.", "43") is True
    assert score.extract_number("2024年和3月", ["2024", "3"]) is True
    assert score.extract_number("全角４３", "43") is False
    # an empty anchor list is "not applicable", never a pass (v1.0: all([]) == True)
    assert score.extract_number("anything", []) is None
    with pytest.raises(ValueError):
        score.extract_number("x", "4.3")


def test_i6_e2_slot_scoring_is_exact_and_language_aware():
    zh = {"name": "Alda Brightford", "year": "1987年", "city": "北京", "hobby": "摄影"}
    text = "Alda Brightford 于1987年出生在北京，喜欢摄影。"
    assert score.score_slots(text, zh, "zh") == {"name": True, "year": True, "city": True, "hobby": True}
    bad = "Alda Brightfordson 于21987年出生在北京，是一名摄影师。"
    assert score.score_slots(bad, zh, "zh") == {"name": False, "year": False, "city": True, "hobby": False}
    en = {"name": "Alda Brightford", "year": "1987", "occupation": "photographer"}
    assert score.score_slots("Alda Brightford, born 1987, is a photographer.", en, "en") == dict.fromkeys(en, True)
    assert score.score_slots("Alda Brightford, born 19870, photographers", en, "en") == {
        "name": True, "year": False, "occupation": False}  # fmt: skip
    # the language check: names are Latin script in both languages and are left out
    assert score.lang_ok(text, "zh", ignore=["Alda Brightford"])
    assert not score.lang_ok("Alda Brightford was born in Beijing.", "zh", ignore=["Alda Brightford"])
    assert score.lang_ok("The answer is 43.", "en") and not score.lang_ok("答案是43", "en")
    assert not score.lang_ok("1987 2024", "en")  # no letters: not in any language


@pytest.fixture(scope="module")
def stand_in_tok(tiny_root):
    return load_tokenizer(tiny_root / "tok" / "tokenizer.json")


def _ctx(model, tok, routing, items, arm="B4"):
    return score.EvalContext(model, tok, model.expert_names, routing, arm, 0, it.items_sha256(items))


def test_i6_e2_a_missing_expert_produces_a_skip_record_with_its_reason(stand_in_tok, item_maker):
    model = RoutedLM(profile("tiny", "swiglu"), ["zh", "bridge"]).eval()
    items = [
        item_maker(0, "E1", [("en", "question", "e1 e2 e3")], ["e4", "e5"], 0, direction="en>en",
                   knowledge_source="en"),
        item_maker(1, "E1", [("zh", "question", "z1 z2 z3")], ["z4", "z5"], 1, direction="zh>zh",
                   knowledge_source="zh"),
        item_maker(2, "E3ctx", [("zh", "fact", "z1\n"), ("en", "rule", "e1\n"), ("zh", "question", "z2")],
                   ["z3", "e3"], 0, variant="cross", direction="zh>en>zh", distractor=1),
    ]  # fmt: skip
    preds = score.evaluate(_ctx(model, stand_in_tok, it.Routing("R_out"), items), items)
    assert "missing expert ['en']" in preds[0]["skip_reason"] and preds[0]["correct"] is None
    assert preds[1]["skip_reason"] is None and preds[1]["correct"] in (True, False)
    assert "missing expert ['en']" in preds[2]["skip_reason"]  # its rule segment is English
    # R_src: E3-ctx has no knowledge source, E3-par two of them; both are skipped with the reason
    par = item_maker(3, "E3par", [("zh", "question", "z1 z2")], ["z3", "z4"], 0, variant="same_city",
                     direction="zh+en>zh", knowledge_sources={"a": "zh", "b": "en"})  # fmt: skip
    model2 = RoutedLM(profile("tiny", "swiglu"), ["zh", "en"]).eval()
    src = score.evaluate(_ctx(model2, stand_in_tok, it.Routing("R_src"), items[1:] + [par]), items[1:] + [par])
    assert src[0]["skip_reason"] is None
    assert "no knowledge_source" in src[1]["skip_reason"]
    assert "two partitions" in src[2]["skip_reason"]
    from surogate_bilingual.eval.report import results_from_predictions

    rows = results_from_predictions(preds, n_boot=100)
    e1 = next(r for r in rows if r["metric"] == "E1/qa/acc")
    assert e1["n_items"] == 1 and e1["skipped"] == 1  # a skipped item is counted, never scored


def test_i6_e2_items_sha256_is_recorded(stand_in_tok, item_maker):
    model = RoutedLM(profile("tiny", "swiglu"), ["zh", "en"]).eval()
    items = [item_maker(i, "E1", [("zh", "question", f"z{i} z{i + 1}")], ["z1", "z2"], 0) for i in range(3)]
    sha = it.items_sha256(items)
    preds = score.evaluate(_ctx(model, stand_in_tok, it.Routing("R_out"), items), items)
    assert {p["items_sha256"] for p in preds} == {sha}
    changed = [dict(items[0], gold_index=1), *items[1:]]
    assert it.items_sha256(changed) != sha and it.items_sha256(items[::-1]) != sha


def _write_items(path, items):
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    return path


def test_i6_e2_eval_tasks_command_writes_predictions_and_results(tiny_assemblies, item_maker, tmp_path, capsys):
    items = [item_maker(i, "E3ctx", [("zh", "fact", f"z{i} z2\n"), ("en", "rule", "e1 e2\n"),
                                     ("zh", "question", f"z{i + 3}")], ["z5", "z6", "z7", "e5"], i % 3,
                        variant="cross", direction="zh>en>zh", distractor=3, question_template=f"q{i % 4}")
             for i in range(12)]  # fmt: skip
    test_item = item_maker(99, "E1", [("zh", "question", "z1")], ["z2", "z3"], 0, split="test")
    path = _write_items(tmp_path / "items.jsonl", [*items, test_item])
    out = tmp_path / "eval" / "B6"
    argv = ["--model", str(tiny_assemblies["zh_en"]), "--arm", "B6", "--items", str(path), "--out", str(out),
            "--n-boot", "200"]  # fmt: skip
    assert score.main([*argv, "--dry-run"]) == 0 and not out.exists()
    assert score.main(argv) == 0
    preds = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert len(preds) == 12 and {p["format"] for p in preds} == {"bilingual-pred/1"}
    assert {(p["route_policy"], p["gate_policy"]) for p in preds} == {("R_out", "dual_const(0.5,0.5)")}
    doc = json.loads((out / "results.json").read_text())
    assert doc["format"] == "bilingual-result/1" and doc["items_sha256"] == preds[0]["items_sha256"]
    assert doc["model"]["fusion"] == "output" and doc["split"] == "dev"
    acc = next(r for r in doc["results"] if r["metric"] == "E3ctx/cross/acc")
    assert acc["status"] == "done" and acc["n_items"] == 12 and acc["ci"]["n_clusters"] == 4
    # the continuous secondary endpoint: gold score minus the best other candidate (sum: equal lengths)
    p = preds[0]
    lp = p["candidate_logprobs"]
    assert p["norm"] == "sum" and p["margin"] == lp[p["gold_index"]] - max(
        x for i, x in enumerate(lp) if i != p["gold_index"]
    )
    assert any(r["metric"] == "E3ctx/cross/margin" and r["status"] == "done" for r in doc["results"])
    assert score.main(argv) == 1  # no silent overwrite
    capsys.readouterr()
    assert score.main([*argv, "--split", "test", "--overwrite"]) == 1  # the test split needs the MS §9.7 lock
    assert "preregistration lock" in capsys.readouterr().err
