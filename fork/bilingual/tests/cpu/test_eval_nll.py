"""I6-E4: BPB on constructed logits, the labelled 2x2 matrix, the switch decomposition (MA-08) and the
after-switch spike. Values from the tiny packs are implementation checks, not results."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from surogate_bilingual.data.docs import Doc, Segment
from surogate_bilingual.data.tokenizer import load_tokenizer
from surogate_bilingual.eval import nll, switch
from surogate_bilingual.eval.items import Seg, bos
from surogate_bilingual.eval.models import load_model
from surogate_bilingual.routing.plan import RoutePlan

V = 64


class BigramStub(torch.nn.Module):
    def __init__(self, table, max_seq=64):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.table = table
        self.cfg = SimpleNamespace(max_seq=max_seq)

    def forward(self, tokens, plan=None):
        return self.table[tokens]


def test_i6_e4_bpb_is_nats_over_ln2_bytes_on_constructed_logits(tiny_root):
    tok = load_tokenizer(tiny_root / "tok" / "tokenizer.json")
    table = torch.randn(V, V, generator=torch.Generator().manual_seed(2))
    doc = Doc("d/0", "en", (Segment(text="e1 e2 e3 e4", lang="en"),))
    ids, n_bytes, langs = nll.doc_tokens(tok, doc)
    assert ids[0] == tok.token_to_id("<eos>") and n_bytes == len(b"e1 e2 e3 e4") and set(langs) == {"en"}
    logp = table.log_softmax(-1)
    hand = -sum(float(logp[a, b]) for a, b in zip(ids, ids[1:], strict=False))
    got = nll.token_nll(BigramStub(table), ids)
    assert abs(float(got.sum()) - hand) <= 1e-5
    assert abs(nll.bpb(hand, n_bytes) - hand / (math.log(2) * n_bytes)) <= 1e-15
    # windows of max_seq advance by the stride; every target is scored once with the same value
    long_ids = [1] + list(range(5, 40))
    full = nll.token_nll(BigramStub(table), long_ids)
    windowed = nll.token_nll(BigramStub(table, max_seq=8), long_ids, stride=3)
    assert np.isfinite(windowed).all() and float(np.abs(full - windowed).max()) <= 1e-6
    with pytest.raises(ValueError):
        nll.bpb(1.0, 0)


def test_i6_e4_the_2x2_matrix_carries_labels(tiny_assemblies, tiny_root):
    ev = load_model(tiny_assemblies["zh_en"])
    docs = [Doc(f"zh/{i}", "zh", (Segment(text=f"z{i} z{i + 1} z{i + 2}", lang="zh"),)) for i in range(4)]
    docs += [Doc(f"en/{i}", "en", (Segment(text=f"e{i} e{i + 1} e{i + 2} e{i + 3}", lang="en"),)) for i in range(4)]
    docs.append(Doc("mix/0", "codeswitch", (Segment(text="z1", lang="zh"), Segment(text="e1", lang="en"))))
    rows = {name: (lambda n, name=name: RoutePlan.forced(name, ev.names, 1, n)) for name in ev.names}
    m = nll.nll_matrix(ev.model, ev.tokenizer, docs, rows)
    assert m["rows"] == ["zh", "en"] and m["cols"] == ["zh", "en"] and "forced route" in m["label"]
    assert m["docs"] == [[4, 4], [4, 4]] and m["skipped_mixed_docs"] == 1
    assert all(isinstance(x, float) and x > 0 for row in m["bpb"] for x in row)
    # the experts differ, so the matrix is not one number repeated
    assert m["bpb"][0] != m["bpb"][1]


def _switch_segments(tok, langs):
    words = {"zh": "z{} z{} z{}", "en": "e{} e{} e{}"}
    segs = []
    for k, (lang, role) in enumerate(langs):
        text = words[lang].format(k + 1, k + 2, k + 3)
        segs.append(Seg(lang, role, tuple(tok.encode(text, add_special_tokens=False).ids)))
    return [bos(tok, segs[0].lang), *segs]


def test_i6_e4_switch_decomposition(tiny_assemblies):
    ev = load_model(tiny_assemblies["zh_en"])
    tok = ev.tokenizer
    same = switch.decompose_segments(
        ev.model, _switch_segments(tok, [("en", "source"), ("en", "instruction"), ("en", "answer")])
    )
    assert not same["switched"] and same["M"] == same["R_new"] == same["R_old"]  # exactly, not approximately
    out = switch.decompose_segments(
        ev.model, _switch_segments(tok, [("zh", "source"), ("en", "instruction"), ("en", "answer")])
    )
    assert {"M", "R_new", "R_old"} <= set(out) and out["switched"]
    assert (out["new_route"], out["old_route"]) == ("en", "zh")
    assert abs(out["M"] - out["R_new"]) > 1e-6 and abs(out["mismatch_cost"] - (out["M"] - out["R_new"])) <= 1e-15
    assert out["L_ref"] is None
    # with a translated history, L_ref is the target read after a history already in the target language
    ref = switch.decompose_segments(
        ev.model,
        _switch_segments(tok, [("zh", "source"), ("en", "instruction"), ("en", "answer")]),
        reference_history=_switch_segments(tok, [("en", "source"), ("en", "instruction")]),
    )
    assert ref["L_ref"] is not None and abs(ref["reading_cost"] - (ref["R_new"] - ref["L_ref"])) <= 1e-15
    with pytest.raises(ValueError, match="two experts"):
        switch.decompose_segments(load_model(tiny_assemblies["zh"]).model, _switch_segments(tok, [("zh", "answer")]))


def test_i6_e4_spike_after_switch_averages_exactly_the_first_k_targets():
    routes = ["zh"] * 10 + ["en"] * 30 + ["zh"] * 5 + ["en"] * 20
    values = np.arange(len(routes), dtype=np.float64)
    out = nll.spike_after_switch(values, routes, k=16)
    assert out["n_switches"] == 3 and [s["at"] for s in out["per_switch"]] == [10, 40, 45]
    # switch at 10: positions 10..25; at 40: 40..44 only (the next switch at 45 stops it); at 45: 45..60
    expected = list(range(10, 26)) + list(range(40, 45)) + list(range(45, 61))
    assert out["n_tokens"] == len(expected) and abs(out["mean"] - float(np.mean(expected))) <= 1e-12
    # non-targets are skipped and do not count toward k
    mask = np.ones(len(routes), dtype=bool)
    mask[10:13] = False
    masked = nll.spike_after_switch(values, routes, mask, k=16)
    assert [p["n"] for p in masked["per_switch"]] == [16, 5, 16]
    assert masked["per_switch"][0]["mean"] == float(np.mean(range(13, 29)))
    assert nll.spike_after_switch(values[:10], routes[:10], k=16)["mean"] is None  # no switch


def test_eval_nll_and_switch_commands(tiny_assemblies, item_maker, tmp_path, capsys):
    docs = tmp_path / "dev.jsonl"
    docs.write_text(
        "".join(
            json.dumps({"key": f"d{i}", "kind": lang, "text": text}) + "\n"
            for i, (lang, text) in enumerate([("zh", "z1 z2 z3"), ("en", "e1 e2 e3")])
        )
    )
    out = tmp_path / "nll.json"
    argv = ["--model", str(tiny_assemblies["zh_en"]), "--docs", str(docs), "--out", str(out)]
    assert nll.main([*argv, "--dry-run"]) == 0 and not out.exists()
    assert nll.main(argv) == 0
    doc = json.loads(out.read_text())
    assert doc["format"] == "bilingual-nll/1" and doc["matrix"]["rows"] == ["zh", "en"]
    seed = ["--model", str(tiny_assemblies["seed"]), "--docs", str(docs), "--out", str(tmp_path / "b0.json")]
    assert nll.main(seed) == 0 and json.loads((tmp_path / "b0.json").read_text())["matrix"]["rows"] == ["seed"]
    e4 = item_maker(
        0,
        "E4",
        [("zh", "source", "z1 z2 z3\n"), ("en", "instruction", "e1 e2"), ("en", "answer", None)],
        variant="translate",
        direction="zh>en",
        stages=[{"answer_lang": "en", "slots": {"name": "Ab Cd"}, "reference": "e3 e4 e5"}],
        slots={"name": "Ab Cd"},
        reference="e3 e4 e5",
    )
    items = tmp_path / "E4.dev.jsonl"
    items.write_text(json.dumps(e4) + "\n")
    sw = tmp_path / "switch.json"
    assert switch.main(["--model", str(tiny_assemblies["zh_en"]), "--items", str(items), "--out", str(sw)]) == 0
    sdoc = json.loads(sw.read_text())
    assert sdoc["format"] == "bilingual-switch/1" and sdoc["items"][0]["switched"]
    assert set(sdoc["summary"]) == {"translate/zh>en"}
    capsys.readouterr()
    assert switch.main(["--model", str(tiny_assemblies["zh"]), "--items", str(items), "--dry-run"]) == 1


def test_eval_nll_reads_the_split_from_the_materialized_manifest_and_guards_test(tiny_assemblies, tmp_path, capsys):
    """eval-nll records each file's split (from `split materialize`'s derived manifest), refuses test documents before
    the MS §9.7 lock, refuses mixing splits, and records arm and replicate for `stats confirm --nll`."""
    from surogate_bilingual.canonical import atomic_write_json, sha256_file

    out = tmp_path / "splits"
    files = []
    for split, lang, text in (("dev", "zh", "z1 z2 z3"), ("test", "en", "e1 e2 e3")):
        path = out / split / "src.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"key": f"{split}0", "kind": lang, "text": text}) + "\n")
        files.append({"path": f"{split}/src.jsonl", "source": "src", "sha256": sha256_file(path), "split": split})
    atomic_write_json(out / "manifest.json", {"format": "bilingual-manifest/1", "files": files})
    model = ["--model", str(tiny_assemblies["zh_en"]), "--arm", "T0", "--replicate", "1"]
    dev_out = tmp_path / "dev.nll.json"
    assert nll.main([*model, "--docs", str(out / "dev" / "src.jsonl"), "--out", str(dev_out)]) == 0
    doc = json.loads(dev_out.read_text())
    assert (doc["split"], doc["arm"], doc["replicate"]) == ("dev", "T0", 1)
    capsys.readouterr()
    test_docs = ["--docs", str(out / "test" / "src.jsonl")]
    assert nll.main([*model, *test_docs, "--dry-run"]) == 1
    assert "preregistration lock" in capsys.readouterr().err
    assert nll.main([*model, *test_docs, "--allow-test", "--dry-run"]) == 0
    both = ["--docs", str(out / "dev" / "src.jsonl"), str(out / "test" / "src.jsonl"), "--allow-test", "--dry-run"]
    assert nll.main([*model, *both]) == 1 and "mix the splits" in capsys.readouterr().err
    (out / "dev" / "src.jsonl").write_text("{}\n")  # changed after materialize: refused
    assert nll.main([*model, "--docs", str(out / "dev" / "src.jsonl"), "--dry-run"]) == 1
    assert "differs from its entry" in capsys.readouterr().err


def test_verify_compares_the_forced_route_with_each_standalone_expert_on_real_text(tiny_assemblies, tmp_path):
    """MS §9.4: forced single route, assembled vs standalone, on real dev text at T = max_seq (fp32 CPU: 0.0 expected)
    as well as on the recorded random-token probe."""
    from surogate_bilingual.__main__ import main as cli
    from surogate_bilingual.assembly.assemble import verify_fidelity

    docs = tmp_path / "dev.jsonl"
    words = [f"z{i % 28}" for i in range(40)] + [f"e{i % 29}" for i in range(40)]
    docs.write_text(
        "".join(
            json.dumps({"key": f"d{i}", "kind": "zh", "text": " ".join(words[i:] + words[:i])}) + "\n" for i in range(8)
        )
    )
    report = verify_fidelity(tiny_assemblies["zh_en"], docs=[docs], doc_windows_n=3)
    assert report.passed and [s.name for s in report.docs] == ["zh", "en"]
    assert all(s.max_abs == 0.0 and s.argmax_agreement == 1.0 for s in report.docs)
    assert report.doc_source["windows"] == 3 and report.doc_source["width"] == 16
    out = tmp_path / "fidelity.json"
    assert cli(["verify", str(tiny_assemblies["zh_en"]), "--docs", str(docs), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["passed"] and len(doc["docs"]) == 2 and doc["doc_source"]["files"][str(docs)]
    assert cli(["verify", str(tiny_assemblies["zh_en"]), "--docs", str(tmp_path / "missing.jsonl"), "--dry-run"]) == 1


class TwoBackStub(torch.nn.Module):
    """Logits that depend on the current AND the previous token of the window (a sentinel before the window), so
    a target scored with less context than max_seq - stride gets another value (CR1 mutation N16)."""

    def __init__(self, max_seq: int, vocab: int = 48):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        g = torch.Generator().manual_seed(4)
        self.now = torch.randn(vocab, vocab, generator=g)
        self.before = torch.randn(vocab + 1, vocab, generator=g)  # row `vocab` = no previous token in the window
        self.cfg = SimpleNamespace(max_seq=max_seq)
        self.vocab = vocab

    def forward(self, tokens, plan=None):
        prev = torch.cat([torch.full_like(tokens[:, :1], self.vocab), tokens[:, :-1]], dim=1)
        return self.now[tokens] + self.before[prev]


def test_windowed_token_nll_scores_each_target_once_with_full_context():
    ids = [int(x) for x in torch.randint(0, 48, (61,), generator=torch.Generator().manual_seed(9))]
    full = nll.token_nll(TwoBackStub(max_seq=128), ids)  # one window: every target has all its context
    for width, stride in ((16, 8), (16, 14), (12, 5), (9, 1)):
        windowed = nll.token_nll(TwoBackStub(max_seq=width), ids, stride=stride)
        # targets after the first window have at least width - stride >= 2 tokens of context, which this stub
        # needs (its logits read one token back): the value must equal the full-context one, target by target
        assert np.isfinite(windowed).all() and float(np.abs(full - windowed).max()) <= 1e-5, (width, stride)
