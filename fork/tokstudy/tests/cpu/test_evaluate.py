"""Cross-arm bits per byte on a common byte prefix (implementation step S15, the BPB half; the probes are a later
step). In fp64 on one CPU thread, on SYNTHETIC documents:

- a model with uniform logits gives BPB = n_text_targets ln V / (ln 2 bytes) for every arm kind of both tables;
- a SubwordLM over a zero-merge BPE and a ByteLM with the same weights (rows permuted to the BPE's ids) give the
  same per-document NLLs;
- common_prefix_bytes is the largest character boundary <= K_eval that every arm holds in one pass;
- specials are left out of the numerator (the <eos> variant apart), text bytes make the denominator, batching does
  not change a document's NLL, and the evalset refuses an arm it was not built for.
"""

import json
import math
import random
from pathlib import Path

import pytest
import torch
from surogate_tokstudy import evaluate as E
from surogate_tokstudy import stream as S
from surogate_tokstudy import train as T
from surogate_tokstudy.arms import ArmSpec, load_arms
from surogate_tokstudy.canonical import TokstudyError
from surogate_tokstudy.codecs import BYTE_OFFSET, BYTE_VOCAB, PROJECT_SPECIALS, TLC_SPECIALS, ByteCodec, SubwordCodec
from surogate_tokstudy.docs import EvalDoc, Seg, encode_doc, frame_plain, truncate
from surogate_tokstudy.models.build import build_model
from surogate_tokstudy.models.hnet import HNetLM
from surogate_tokstudy.models.lm import ByteLM, SubwordLM
from surogate_tokstudy.models.twin_a0 import A0Twin, twin_config
from surogate_tokstudy.prereg import load_prereg

F64 = torch.float64
TINY_HNET = {
    "d0": 16,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "ratio": 4.0,
    "head_dim": 8,
    "state": 4,
    "ssd_chunk": 8,
}
TABLES = {
    "tlc-N": ("tlc", "twin_a0", {"profile": "tiny"}),
    "bilingual-B": ("bilingual", "bilingual_trunk", {"profile": "tiny", "overrides": {"max_seq": 256}}),
}


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(threads)


def arm(table, kind, tokenizer=None, *, model=None, arm_id=None):
    project, main, base = TABLES[table]
    model = dict(model or base)
    if kind == "hnet":
        model["hnet"] = dict(TINY_HNET)
    return ArmSpec(
        arm_id or f"{kind}-arm", table, kind, tokenizer, PROJECT_SPECIALS[project], model, main, (0,), ("decision",)
    )


def documents(project, n=6, seed=0):
    rng = random.Random(seed)
    alphabet = ("a", "b", " ", "\n", "é", "变", "🙂", "<|eos|>")
    docs = []
    for i in range(n):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(5, 40)))
        domain = ("py", "en")[i % 2] if project == "tlc" else ("zh", "en")[i % 2]
        docs.append(frame_plain(EvalDoc(f"d{i}", domain, (Seg(text=text),)), project))
    return docs


@pytest.fixture(scope="module")
def bpe(tmp_path_factory, bpe_writer):
    """A SYNTHETIC trained BPE per project (its specials at 0-4) and the arms' codecs."""
    out = {}
    for project, specials in PROJECT_SPECIALS.items():
        texts = [d.texts()[0] for d in documents(project, 40, seed=1)]
        path, sha = bpe_writer(
            tmp_path_factory.mktemp(project) / "tokenizer.json",
            specials=tuple(specials),
            pre="gpt2",
            train_on=texts,
            vocab_size=300,
        )
        out[project] = SubwordCodec.from_tokenizer_json(path, sha, specials, check=True)
    return out


def model_and_codec(table, kind, bpe):
    project = TABLES[table][0]
    a = arm(table, kind)
    codec = bpe[project] if kind == "subword" else ByteCodec(PROJECT_SPECIALS[project], check=True)
    model = build_model(a, vocab_size=codec.vocab_size if kind == "subword" else None, init_seed=3).to(F64)
    return model, codec


KINDS = [(t, k) for t in TABLES for k in ("subword", "bytes", "hnet")]


@pytest.mark.parametrize("table,kind", KINDS)
def test_uniform_logits_give_ln_v_per_text_target(table, kind, bpe):
    model, codec = model_and_codec(table, kind, bpe)
    if isinstance(model, HNetLM):
        weight = model.embedding.weight
    elif isinstance(model.main, A0Twin):
        weight = model.main.embedding.weight
    else:
        weight = model.main.lm.backbone.emb.weight
    with torch.no_grad():
        weight.zero_()  # tied heads: every logit is 0
    docs = documents(TABLES[table][0])
    rows = E.score_documents(model, codec, docs, batch=3, precision="fp64")
    V = weight.shape[0]
    for doc, row in zip(docs, rows, strict=True):
        enc = encode_doc(codec, doc)
        assert row.bytes == doc.text_bytes() == enc.text_bytes
        assert row.n_targets == int((~enc.special[1:]).sum()) and row.n_eos == 1
        assert row.nll_nats == pytest.approx(row.n_targets * math.log(V), rel=1e-12)
    for domain in {d.domain for d in docs}:
        picked = [r for r in rows if r.domain == domain]
        expected = sum(r.n_targets for r in picked) * math.log(V) / (math.log(2) * sum(r.bytes for r in picked))
        assert E.bpb(rows, domain) == pytest.approx(expected, rel=1e-12)
        with_eos = sum(r.n_targets + r.n_eos for r in picked) * math.log(V)
        assert E.bpb(rows, domain, with_eos=True) == pytest.approx(
            with_eos / (math.log(2) * sum(r.bytes for r in picked)), rel=1e-12
        )


def test_a_zero_merge_bpe_and_the_byte_codec_with_the_same_weights_score_alike(tmp_path, bpe_writer):
    from tokenizers import pre_tokenizers

    path, sha = bpe_writer(tmp_path / "zero" / "tokenizer.json", specials=tuple(TLC_SPECIALS))
    sub_codec = SubwordCodec.from_tokenizer_json(path, sha, TLC_SPECIALS, check=True)
    assert sub_codec.vocab_size == BYTE_VOCAB
    byte_model = ByteLM(A0Twin(twin_config("tiny"), vocab_size=BYTE_VOCAB, init_seed=5)).to(F64)
    sub_model = SubwordLM(A0Twin(twin_config("tiny"), vocab_size=BYTE_VOCAB, init_seed=5)).to(F64)
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    from surogate_tokstudy.codecs import BYTE_OF_CHAR

    with torch.no_grad():
        target = sub_model.main.embedding.weight
        source = byte_model.main.embedding.weight
        target[:BYTE_OFFSET] = source[:BYTE_OFFSET]
        for i, char in enumerate(alphabet):
            target[BYTE_OFFSET + i] = source[BYTE_OFFSET + BYTE_OF_CHAR[char]]
    docs = documents("tlc", 8)
    a = E.score_documents(sub_model, sub_codec, docs, precision="fp64")
    b = E.score_documents(byte_model, ByteCodec(TLC_SPECIALS), docs, precision="fp64")
    for x, y in zip(a, b, strict=True):
        assert (x.bytes, x.n_targets, x.n_eos) == (y.bytes, y.n_targets, y.n_eos)
        assert x.nll_nats == pytest.approx(y.nll_nats, rel=1e-12) and x.eos_nats == pytest.approx(y.eos_nats, rel=1e-12)
    for domain in ("py", "en"):
        assert E.bpb(a, domain) == pytest.approx(E.bpb(b, domain), rel=1e-12)


def test_batching_and_order_do_not_change_a_documents_nll(bpe):
    for table, kind in KINDS:
        model, codec = model_and_codec(table, kind, bpe)
        docs = documents(TABLES[table][0], 7, seed=4)
        one = E.score_documents(model, codec, docs, batch=1, precision="fp64")
        many = E.score_documents(model, codec, docs, batch=4, precision="fp64")
        assert [r.key for r in many] == [d.key for d in docs]
        for x, y in zip(one, many, strict=True):
            assert x.nll_nats == pytest.approx(y.nll_nats, rel=1e-10, abs=1e-12), (table, kind)


def test_the_common_prefix_is_the_largest_boundary_every_arm_holds(bpe):
    small = {"profile": "tiny", "overrides": {"max_seq": 24}}
    arms = [
        arm("bilingual-B", "bytes", model=small, arm_id="B5"),
        arm("bilingual-B", "hnet", model=small, arm_id="B4"),
        arm("bilingual-B", "subword", None, model=small, arm_id="B0"),
    ]
    codecs = {
        "B5": ByteCodec(PROJECT_SPECIALS["bilingual"]),
        "B4": ByteCodec(PROJECT_SPECIALS["bilingual"]),
        "B0": bpe["bilingual"],
    }
    limits = E.arm_limits(arms)
    assert limits == {"B5": 24, "B4": 24, "B0": 24}
    for doc in documents("bilingual", 10, seed=7):
        for k_eval in (5, 30, 1000):
            k = E.common_prefix_bytes(doc, arms, k_eval=k_eval, codecs=codecs, limits=limits)
            fits = lambda m, doc=doc: all(len(encode_doc(codecs[a], truncate(doc, m)).ids) <= 24 for a in codecs)  # noqa: E731
            boundaries = [int(b) for b in E._char_boundaries(doc) if b <= k_eval]
            assert k == max(b for b in boundaries if fits(b))
            assert k in boundaries
    # the Mamba twin has no positional limit: the prefix is K_eval at a character boundary
    tlc = [arm("tlc-N", "bytes", arm_id="N2")]
    doc = documents("tlc", 1)[0]
    assert E.common_prefix_bytes(doc, tlc, k_eval=10**6) == doc.text_bytes()


def test_scoring_refuses_documents_it_cannot_measure_exactly(bpe):
    model, codec = model_and_codec("tlc-N", "bytes", bpe)
    with pytest.raises(TokstudyError, match="special first position"):
        E.score_documents(model, codec, [EvalDoc("k", "py", (Seg(text="abc"),))])


# --- evalsets, NLL files and the command line ------------------------------------------------------------------------


def write_table(root: Path, bpe_writer):
    """A SYNTHETIC tlc-N table with a stream, a registry (N0 subword, N2 bytes) and a prereg; returns paths."""
    docs = [EvalDoc(f"train/{d.key}", d.domain, d.segs) for d in documents("tlc", 120, seed=2)]
    S.build_stream(docs, root / "stream", seed=0, k_train=32, source={"project": "tlc", "fixture": "SYNTHETIC"})
    path, sha = bpe_writer(
        root / "tok" / "tokenizer.json",
        specials=tuple(TLC_SPECIALS),
        pre="gpt2",
        train_on=[d.texts()[0] for d in docs],
        vocab_size=300,
    )
    (root / "arms.toml").write_text(
        'format = "tokstudy-arms/1"\ntable = "tlc-N"\n'
        f'[[arm]]\nid = "N0"\nkind = "subword"\ntokenizer = {{path = "tok/tokenizer.json", sha256 = "{sha}"}}\n'
        'model = {profile = "tiny"}\nseeds = [0]\nstages = ["decision"]\n'
        '[[arm]]\nid = "N2"\nkind = "bytes"\nmodel = {profile = "tiny"}\nseeds = [0]\nstages = ["decision"]\n',
        encoding="utf-8",
    )
    (root / "prereg.toml").write_text(
        'format = "tokstudy-prereg/1"\ntable = "tlc-N"\n[budget]\nF_target = 300_000_000\npoints = [0.5, 1.0]\n'
        "[context]\nK_train = 32\nK_eval = 24\n",
        encoding="utf-8",
    )
    samples = root / "val.jsonl"
    samples.write_text(
        "".join(json.dumps({"text": d.texts()[0]}, ensure_ascii=False) + "\n" for d in documents("tlc", 10, seed=9)),
        encoding="utf-8",
    )
    return root


def test_evalset_scoring_and_nll_files_end_to_end(tmp_path, bpe_writer, capsys):
    root = write_table(tmp_path, bpe_writer)
    assert (
        E.main(
            [
                "prefix",
                "--arms",
                str(root / "arms.toml"),
                "--prereg",
                str(root / "prereg.toml"),
                "--docs",
                f"py={root / 'val.jsonl'}",
                "--project",
                "tlc",
                "--out",
                str(root / "evalset"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest, sha, docs = E.open_evalset(root / "evalset")
    assert manifest["k_eval"] == 24 and all(d.text_bytes() <= 24 for d in docs) and len(docs) == 10
    assert {a["id"] for a in manifest["arms"]} == {"N0", "N2"}
    arms = {a.id: a for a in load_arms(root / "arms.toml")}
    config = T.TrainConfig(lr=3e-3, batch_rows=4, micro_rows=4, precision="fp64")
    stream = S.open_stream(root / "stream")
    for arm_id in ("N0", "N2"):
        result = T.train(
            arms[arm_id],
            root / "runs",
            stream=stream,
            prereg=load_prereg(root / "prereg.toml"),
            seed=0,
            config=config,
            registry_path=root / "arms.toml",
        )
        T.branch(result["dir"], endpoint=0.5)
        for point in (0.5, 1.0):
            out = root / f"{arm_id}-{point}.nll"
            assert (
                E.main(
                    [
                        "score",
                        result["dir"],
                        "--point",
                        str(point),
                        "--evalset",
                        str(root / "evalset"),
                        "--out",
                        str(out),
                        "--device",
                        "cpu",
                    ]
                )
                == 0
            )
            printed = json.loads(capsys.readouterr().out)
            header, rows = E.read_nll(out)
            assert header["format"] == E.NLL_FORMAT and header["evalset_sha256"] == sha
            assert header["arm"] == arm_id and header["point"] == point and header["backend"] == "tokstudy-torch"
            assert [r.key for r in rows] == [d.key for d in docs]
            assert printed["summary"]["py"]["bpb"] == pytest.approx(E.bpb(rows, "py"))
            assert sum(r.bytes for r in rows) == sum(d.text_bytes() for d in docs)
    # an evalset built for other arms refuses this run: the common prefix belongs to the table's arm set
    other = root / "other.toml"
    other.write_text((root / "arms.toml").read_text().replace('stages = ["decision"]\n', 'stages = ["final"]\n', 1))
    E.build_evalset(docs, load_arms(other), root / "evalset2", k_eval=24, source={"project": "tlc"})
    n0 = next(Path(r) for r in (root / "runs" / "tlc-N").iterdir() if r.name.startswith("N0"))
    assert E.main(["score", str(n0), "--evalset", str(root / "evalset2"), "--out", str(root / "x.nll")]) == 1
    assert "not an arm the evalset's common prefix" in capsys.readouterr().err
    with pytest.raises(TokstudyError, match="not empty"):
        E.build_evalset(docs, list(arms.values()), root / "evalset", k_eval=24, source={})
    with pytest.raises(TokstudyError, match="one table"):
        E.build_evalset(docs, [arms["N2"], arm("bilingual-B", "bytes")], root / "evalset3", k_eval=24, source={})


def test_evaluation_accepts_a_run_of_another_commit_and_continuing_it_does_not(tmp_path, bpe_writer, monkeypatch):
    from surogate_tokstudy import registry

    root = write_table(tmp_path, bpe_writer)
    arms = {a.id: a for a in load_arms(root / "arms.toml")}
    config = T.TrainConfig(lr=3e-3, batch_rows=4, micro_rows=4, precision="fp64")
    result = T.train(
        arms["N2"],
        root / "runs",
        stream=S.open_stream(root / "stream"),
        seed=0,
        config=config,
        prereg=load_prereg(root / "prereg.toml"),
        registry_path=root / "arms.toml",
    )
    docs = documents("tlc", 5, seed=9)
    E.build_evalset(docs, list(arms.values()), root / "evalset", k_eval=24, source={"project": "tlc"})
    monkeypatch.setattr(registry, "code_identity", lambda start=None: {"commit": "f" * 40, "dirty": False})
    with pytest.raises(TokstudyError, match="or code changed"):
        T.branch(result["dir"], endpoint=0.5)
    out = E.evaluate_point(result["dir"], 1.0, root / "evalset", root / "n2.nll")
    header, rows = E.read_nll(out["nll"])
    assert header["run_id"] == result["run_id"] and header["eval_code"]["commit"] == "f" * 40
    assert header["run_code"] == result["code"] and len(rows) == 5


def test_tlc_native_writer_of_tokstudy_nll_has_the_same_fields_and_domains(tmp_path):
    """TLC's `eval --dump-nll` (surogate_tlc.eval.loss, the native table's writer) and this package agree on the
    neutral format without importing each other: the TLC source is read as text, its DocNLL fields, format name and
    domains (those of docs.tlc_documents: code sources "py", text sources "en") compared with this package's; a line
    in TLC's layout reads back through read_nll."""
    import ast
    import dataclasses

    loss_py = Path(__file__).resolve().parents[3] / "tlc" / "src" / "surogate_tlc" / "eval" / "loss.py"
    tree = ast.parse(loss_py.read_text(encoding="utf-8"))
    constants = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("NLL_FORMAT", "BACKEND", "DOMAINS")
    }
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DocNLL")
    fields = [n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)]
    assert fields == [f.name for f in dataclasses.fields(E.DocNLL)]
    assert constants["NLL_FORMAT"] == E.NLL_FORMAT and constants["BACKEND"] == "native"
    assert constants["DOMAINS"] == {"code": "py", "text": "en"}
    line = {
        "key": "val-00000.bin:0",
        "domain": "py",
        "bytes": 10,
        "nll_nats": 1.5,
        "n_targets": 4,
        "eos_nats": 0.1,
        "n_eos": 1,
        "special_nats": 0.2,
        "n_special": 2,
    }
    header = {"format": E.NLL_FORMAT, "table": "tlc-T", "backend": "native", "arm": "T0"}
    path = tmp_path / "T0.nll"
    path.write_text(json.dumps(header) + "\n" + json.dumps(line, sort_keys=True) + "\n", encoding="utf-8")
    read_header, rows = E.read_nll(path)
    assert read_header["backend"] == "native" and rows == [E.DocNLL(**line)]
