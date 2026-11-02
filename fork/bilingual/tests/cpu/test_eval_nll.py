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
