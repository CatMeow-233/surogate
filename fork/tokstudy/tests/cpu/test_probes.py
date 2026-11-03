"""Probes (implementation step S15, the probe half) on SYNTHETIC FIXTURE items written by these tests in the real
schemas (tokstudy-probe-items/1, and bilingual-item/1 for the Track S adapter), in fp64 on one CPU thread:

- `scorable` is false exactly when a merge crosses the boundary between the stem and a candidate; byte arms are
  always scorable; coverage is reported per arm;
- under uniform logits a candidate's NLL is its token count times ln V, ranked by the sum when the byte lengths are
  equal and per byte otherwise;
- a candidate's NLL equals the document NLL of stem + candidate minus that of the stem (score_documents), for a
  subword, a byte and an H-Net arm of a real tiny model;
- Track S items become probe items with <eos>, the prompt segments, the question as stem and answer_sep +
  candidate as candidates; the probe command scores a trained run.
"""

import json
import math

import pytest
import torch
from surogate_tokstudy import evaluate as E
from surogate_tokstudy import probes as P
from surogate_tokstudy.arms import ArmSpec
from surogate_tokstudy.canonical import TokstudyError
from surogate_tokstudy.codecs import PROJECT_SPECIALS, ByteCodec, SubwordCodec
from surogate_tokstudy.docs import EvalDoc, Seg
from surogate_tokstudy.models.build import build_model

TLC = PROJECT_SPECIALS["tlc"]
TINY_HNET = {
    "d0": 16,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "ratio": 4.0,
    "head_dim": 8,
    "state": 4,
    "ssd_chunk": 8,
}


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(threads)


def item(stem, candidates, gold=0, key="k", domain="en", context=None):
    return P.ProbeItem(key, domain, tuple(context or (Seg("<|bos|>"),)), stem, tuple(candidates), gold)


@pytest.fixture(scope="module")
def merging(tmp_path_factory, bpe_writer):
    """A BPE with one merge, a + b (no pre-tokenizer: the merge acts anywhere)."""
    path, sha = bpe_writer(tmp_path_factory.mktemp("bpe") / "tokenizer.json", [("a", "b")], specials=tuple(TLC))
    return SubwordCodec.from_tokenizer_json(path, sha, TLC, check=True)


def test_scorable_is_false_exactly_when_a_merge_crosses_the_boundary(merging):
    assert P.scorable(merging, "xa", "b") == (
        False,
        "a token merges across the boundary between the stem and a candidate",
    )
    assert P.scorable(merging, "xa", "c")[0] and P.scorable(merging, "x", "ab")[0] and P.scorable(merging, "", "ab")[0]
    assert P.scorable(merging, "xab", "ab")[0]  # the merge stays on each side
    for stem, candidate in (("xa", "b"), ("a", "bb")):
        assert P.scorable(ByteCodec(TLC), stem, candidate) == (True, None)
    rows = P.score_items(uniform(merging.vocab_size), merging, [item("xa", ["b", "c"]), item("x", ["ab", "c"])])
    assert [r.scorable for r in rows] == [False, True] and rows[0].reason.startswith("a token merges")
    assert P.coverage(rows) == {"en": {"items": 2, "scorable": 1, "share": 0.5}}


class Uniform(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, ids):
        return torch.zeros(*ids.shape, self.vocab, dtype=torch.float64) + self.p


def uniform(vocab):
    return Uniform(vocab)


def test_uniform_logits_give_ln_v_per_candidate_token_and_the_byte_length_rule(merging):
    lnv = math.log(merging.vocab_size)
    rows = P.score_items(uniform(merging.vocab_size), merging, [item("x", ["ab", "c"], gold=1), item("x", ["c", "d"])])
    first, second = rows
    assert first.candidate_nll == pytest.approx((lnv, lnv)) and first.candidate_bytes == (2, 1)
    assert first.norm == "per_byte" and first.pred == 0 and first.correct is False  # 1 token over 2 bytes wins
    assert first.gold_logprob_per_byte == pytest.approx(-lnv)
    assert second.norm == "sum" and second.pred == 0 and second.correct  # a tie goes to the lowest index
    byte_rows = P.score_items(uniform(ByteCodec(TLC).vocab_size), ByteCodec(TLC), [item("x", ["ab", "c"])])
    assert byte_rows[0].candidate_nll == pytest.approx((2 * math.log(261), math.log(261)))


ARMS = {"subword": None, "bytes": None, "hnet": TINY_HNET}


@pytest.mark.parametrize("kind", ["subword", "bytes", "hnet"])
def test_a_candidates_nll_is_the_document_nll_difference(kind, tmp_path, bpe_writer):
    texts = ["def total(x):\n    return x + 1\n", "for i in range(3): print(i)\n"]
    path, sha = bpe_writer(tmp_path / "tok.json", specials=tuple(TLC), pre="gpt2", train_on=texts * 5, vocab_size=300)
    codec = SubwordCodec.from_tokenizer_json(path, sha, TLC, check=True) if kind == "subword" else ByteCodec(TLC)
    model_cfg = {"profile": "tiny"} | ({"hnet": TINY_HNET} if kind == "hnet" else {})
    arm = ArmSpec("x", "tlc-N", kind, None, TLC, model_cfg, "twin_a0", (0,), ("decision",))
    model = build_model(arm, vocab_size=codec.vocab_size if kind == "subword" else None, init_seed=1).double()
    context = (Seg("<|bos|>"), Seg("<|code_start|>"))
    items = [item("def total(x):\n    return", [" x + 1", " None", " total"], context=context)]
    if kind != "subword":
        items[0] = item("def total(x):\n    retur", ["n x", "n", "ns"], context=context)
    rows = P.score_items(model, codec, items, precision="fp64")
    assert rows[0].scorable, rows[0]
    stem = E.score_documents(model, codec, [EvalDoc("s", "py", items[0].document(None))], precision="fp64")[0]
    for c, candidate in enumerate(items[0].candidates):
        full = E.score_documents(model, codec, [EvalDoc("f", "py", items[0].document(candidate))], precision="fp64")
        assert rows[0].candidate_nll[c] == pytest.approx(full[0].nll_nats - stem.nll_nats, rel=1e-9)
        assert rows[0].candidate_bytes[c] == len(candidate.encode())


def test_track_s_items_become_probe_items(tmp_path):
    def track_s(i, family, segments, candidates, gold, sep=" "):
        return {
            "format": "bilingual-item/1",
            "id": f"{family}-{i}",
            "family": family,
            "variant": "SYNTHETIC FIXTURE",
            "direction": "zh>zh",
            "split": "dev",
            "template_ids": {"question": "q0"},
            "segments": [{"lang": lang, "role": role, "text": text} for lang, role, text in segments],
            "answer_lang": segments[-1][0],
            "answer_sep": sep,
            "candidates": candidates,
            "gold_index": gold,
            "distractor_index": None,
            "chance": 1 / max(len(candidates), 1),
            "knowledge_source": None,
            "structured": {},
            "difficulty": {},
            "generator": {"version": "test", "seed": 0},
        }

    rows = [
        track_s(0, "E1", [("zh", "context", "甲住在乙。\n"), ("zh", "question", "甲住在哪里？")], ["乙", "丙"], 0),
        track_s(1, "E4", [("en", "instruction", "Translate.\n"), ("en", "question", "A")], [], 0),
    ]
    path = tmp_path / "items.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    items = P.items_from_bilingual([path])
    assert len(items) == 1  # the generated family E4 has no candidates
    got = items[0]
    assert got.context == (Seg("<eos>"), Seg(text="甲住在乙。\n", lang="zh")) and got.stem == "甲住在哪里？"
    assert got.candidates == (" 乙", " 丙") and got.domain == "zh" and got.lang == "zh"
    assert got.document(" 乙")[-1] == Seg(text="甲住在哪里？ 乙", lang="zh")
    out = tmp_path / "probe-items.jsonl"
    P.write_items(out, items)
    assert P.read_items(out) == items and P.items_sha256(items) == P.items_sha256(P.read_items(out))
    with pytest.raises(TokstudyError, match="no ranked Track S item"):
        P.items_from_bilingual([path], split="test")
    with pytest.raises(TokstudyError, match="at least two candidates"):
        item("x", ["a"])


def test_the_probe_command_scores_a_trained_run(tmp_path, bpe_writer, capsys):
    from surogate_tokstudy import stream as S
    from surogate_tokstudy import train as T
    from surogate_tokstudy.arms import load_arms
    from surogate_tokstudy.prereg import load_prereg

    docs = [EvalDoc(f"d{i}", "py", (Seg("<|bos|>"), Seg(text=f"x = {i}\n" * 3), Seg("<|eos|>"))) for i in range(40)]
    stream = S.build_stream(docs, tmp_path / "stream", seed=0, k_train=32, source={"project": "tlc"})
    (tmp_path / "arms.toml").write_text(
        'format = "tokstudy-arms/1"\ntable = "tlc-N"\n'
        '[[arm]]\nid = "N2"\nkind = "bytes"\nmodel = {profile = "tiny"}\nseeds = [0]\nstages = ["decision"]\n',
        encoding="utf-8",
    )
    (tmp_path / "prereg.toml").write_text(
        'format = "tokstudy-prereg/1"\ntable = "tlc-N"\n[budget]\nF_target = 100_000_000\npoints = [1.0]\n'
        "[context]\nK_train = 32\nK_eval = 32\n",
        encoding="utf-8",
    )
    result = T.train(
        load_arms(tmp_path / "arms.toml")[0],
        tmp_path / "runs",
        stream=stream,
        prereg=load_prereg(tmp_path / "prereg.toml"),
        seed=0,
        config=T.TrainConfig(lr=1e-3, batch_rows=4, micro_rows=4, precision="fp64"),
        registry_path=tmp_path / "arms.toml",
    )
    items_path = tmp_path / "items.jsonl"
    P.write_items(
        items_path, [item("x = ", ["1", "22"], key="a", domain="py"), item("x =", [" 3", " 4"], key="b", domain="py")]
    )
    out = tmp_path / "N2.probe"
    argv = [result["dir"], "--items", str(items_path), "--out", str(out), "--device", "cpu"]
    assert P.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["coverage"] == {"py": {"items": 2, "scorable": 2, "share": 1.0}}
    header, rows = P.read_probe(out)
    assert header["arm"] == "N2" and header["arm_kind"] == "bytes" and header["length_norm"] == "bytes"
    assert [r.key for r in rows] == ["a", "b"] and all(r.scorable for r in rows)
    assert P.main([result["dir"], "--out", str(out)]) == 1 and "give --items" in capsys.readouterr().err
