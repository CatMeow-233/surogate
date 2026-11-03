"""The eval command's loss (docs/training.md: 评测), with a stand-in for the native scorer: every served row of the
split is scored once, the loss averages over valid targets (also per source and per kind), and the unigram baseline
counts the train split's targets."""

import math

import numpy as np
import pytest
from surogate_tlc.data import shards
from surogate_tlc.data.shards import manifest, shard_files
from surogate_tlc.eval import loss


def _rows(data, split):
    parts = [shards.served_rows(shards.read_shard(p), 64) for p in shard_files(data, split)]
    return [np.concatenate(arrays) for arrays in zip(*parts, strict=True)]


def test_unigram_loss_counts_the_train_targets(shard_dirs):
    data = shard_dirs[64]
    counts = np.ones(320)
    for target in _rows(data, "train")[1].ravel():
        if target != -100:
            counts[target] += 1
    targets = _rows(data, "val")[1]
    expected = -np.mean([math.log(counts[t] / counts.sum()) for t in targets.ravel() if t != -100])
    assert loss.unigram_loss(data, targets, 320) == pytest.approx(expected)


@pytest.mark.parametrize("batch, max_rows", [(3, None), (4, 2)])
def test_evaluate_scores_each_served_row_once(small_model, shard_dirs, batch, max_rows):
    data = shard_dirs[64]
    seen = []

    def uniform(inputs, targets, positions):
        assert inputs.shape == targets.shape == positions.shape == (batch, 64) and inputs.dtype == np.int32
        seen.append((inputs.copy(), targets.copy()))
        return np.where(targets == -100, 0.0, -math.log(320)).astype(np.float32)

    result = loss.evaluate(small_model, data, batch=batch, max_rows=max_rows, scorer=uniform)
    inputs, targets, _ = (a[:max_rows] for a in _rows(data, "val"))
    scored = np.concatenate([i for i, _ in seen])[: len(inputs)]
    assert np.array_equal(scored, inputs) and result["rows"] == len(inputs)
    assert result["valid_targets"] == int((targets != -100).sum())
    if max_rows is None:
        assert result["valid_targets"] == manifest(data)["splits"]["val"]["valid_targets"]
    # The filled rows of the last batch carry no targets.
    assert (np.concatenate([t for _, t in seen])[len(inputs) :] == -100).all()
    assert result["loss"] == pytest.approx(math.log(320)) and result["unigram_loss"] < result["loss"]
    assert result["below_unigram"] is False
    assert result["data_manifest_sha256"] == shards.manifest_sha256(data)
    # Per source: the row ranges of the manifest; a uniform scorer gives every source the same loss.
    by_source = result["by_source"]
    assert sum(v["valid_targets"] for v in by_source.values()) == result["valid_targets"]
    assert all(v["loss"] == pytest.approx(math.log(320)) for v in by_source.values())
    assert set(result["by_kind"]) <= {"code", "text"}
    if max_rows is None:
        val = {s["name"]: s["real_tokens"]["val"] - s["rows"]["val"] for s in manifest(data)["sources"]}
        assert {name: v["valid_targets"] for name, v in by_source.items()} == {n: c for n, c in val.items() if c}


def test_row_sources_follow_the_documents(shard_dirs, corpus_spec, trained_tokenizer):
    """The manifest's per-shard source ranges give each served row the source of the document it holds."""
    from surogate_tlc.data import corpus, tokenizer

    data, c = shard_dirs[64], corpus.load_corpus(corpus_spec)
    tok = tokenizer.load(trained_tokenizer, ordinary=True)
    special = tokenizer.special_ids(tok)
    expected = [
        doc.source.name
        for doc, ids in shards.encode(tok, corpus.documents(c), special)
        if corpus.split_of(c, doc) == "val"
        for _ in shards.windows(ids, 64)
    ]
    assert loss._row_sources(manifest(data), shard_files(data, "val")) == expected


def test_evaluate_refuses_another_tokenizer(tmp_path, tokenizer_dir, shard_dirs):
    from surogate_tlc import hf, init, modeldir

    cfg = hf.row_config("A0", num_hidden_layers=2, tlc_layer_pattern="MM", chunk_size=16, vocab_size=320)
    other = modeldir.write_model_dir(tmp_path / "other", cfg, tokenizer_dir, init_seed=0)
    init.init_model_dir(other)
    with pytest.raises(ValueError, match="another tokenizer"):
        loss.evaluate(other, shard_dirs[64], scorer=lambda *a: None)


def test_greedy_stops_at_a_stop_id_or_the_row_end():
    from surogate_tlc.eval.samples import greedy

    def next_position(inputs, last):  # the next token is the next position's index
        logits = np.zeros((inputs.shape[0], 16), np.float32)
        logits[np.arange(len(last)), (last + 1) % 16] = 1
        return logits

    kwargs = {"batch": 1, "pad_id": 0, "stop_ids": {3}, "max_new": 5}
    assert greedy([[1, 2], [1]], next_position, seq_len=8, **kwargs) == [[2, 3], [1, 2, 3]]
    assert greedy([[1, 2], [1]], next_position, seq_len=3, **kwargs) == [[2], [1, 2]]
    assert greedy([[1, 2]], next_position, seq_len=8, **{**kwargs, "max_new": 1}) == [[2]]


@pytest.mark.parametrize("stop", ["<|eos|>", "<|code_end|>"])
def test_samples_start_like_shard_documents_and_stop_at_either_end(small_model, shard_dirs, stop):
    """A prompt is the start of a document as the shards frame it (special-token strings are text), and both
    <|eos|> and <|code_end|> end a sample."""
    from surogate_tlc.data import corpus
    from surogate_tlc.data import tokenizer as tok
    from surogate_tlc.eval import samples

    tokenizer = tok.load(small_model, ordinary=True)
    special = tok.special_ids(tokenizer)
    seen = []

    def next_logits(inputs, last):
        seen.append((inputs.copy(), last.copy()))
        logits = np.zeros((inputs.shape[0], 320), np.float32)
        logits[:, special[stop]] = 1.0
        return logits

    prompts = [("code", "x = 1\n"), ("text", "a <|eos|> b <|code_end|> c")]
    result = samples.sample(small_model, shard_dirs[64], prompts=prompts, max_new=4, batch=2, next_logits=next_logits)
    assert [s["tokens"] for s in result["samples"]] == [[special[stop]]] * 2
    (inputs, last), *_ = seen
    for row, (kind, text) in enumerate(prompts):
        doc = corpus.Document(corpus.Source("s", kind, (), {}), "s/x", "s/x", text)
        [(_, framed)] = shards.encode(tokenizer, [doc], special)
        prompt = framed[:-2] if kind == "code" else framed[:-1]  # without <|code_end|> <|eos|>, or <|eos|>
        assert inputs[row, : len(prompt)].tolist() == prompt and last[row] == len(prompt) - 1, kind


# --- bits per byte and the per-document NLL (tokenizer study, spec S6) ----------------------------------------------


def _id_scorer(inputs, targets, positions):
    """A stand-in whose log-probability of a target depends only on its id: -0.001 * (id + 1)."""
    return np.where(targets == -100, 0.0, -0.001 * (targets.astype(np.float64) + 1)).astype(np.float32)


def _expected_docs(corpus_spec, tokenizer_dir, split):
    """Per document of `split`, in shard order: (key, domain, bytes, text nats, text targets, eos nats, n eos,
    other special nats, n other) computed from the framed ids, every target once."""
    from surogate_tlc.data import corpus, tokenizer

    c = corpus.load_corpus(corpus_spec)
    tok = tokenizer.load(tokenizer_dir, ordinary=True)
    special = tokenizer.special_ids(tok)
    pieces = tokenizer.token_bytes(tok)
    out = []
    for doc, ids in shards.encode(tok, corpus.documents(c), special):
        if corpus.split_of(c, doc) != split:
            continue
        nll = {t: float(np.float32(0.001 * (t + 1))) for t in ids}
        targets = ids[1:]
        text = [t for t in targets if pieces[t] is not None]
        eos = [t for t in targets if t == special["<|eos|>"]]
        other = [t for t in targets if pieces[t] is None and t != special["<|eos|>"]]
        out.append((doc.key, "py" if doc.source.kind == "code" else "en", sum(len(pieces[t]) for t in text),
                    math.fsum(nll[t] for t in text), len(text), math.fsum(nll[t] for t in eos), len(eos),
                    math.fsum(nll[t] for t in other), len(other)))  # fmt: skip
        assert out[-1][2] == loss.content_bytes(doc.text, doc.source.kind)
    return out


def test_token_bytes_cover_the_text_and_leave_out_the_specials(trained_tokenizer):
    from surogate_tlc.data import tokenizer

    tok = tokenizer.load(trained_tokenizer, ordinary=True)
    pieces = tokenizer.token_bytes(tok)
    assert len(pieces) == tok.get_vocab_size()
    assert [i for i, b in enumerate(pieces) if b is None] == [0, 1, 2, 3, 4]
    assert {b for b in pieces if b is not None and len(b) == 1} == {bytes([i]) for i in range(256)}
    for text in tokenizer.EDGE_CASES:
        ids = tok.encode(text, add_special_tokens=False).ids
        assert b"".join(pieces[i] for i in ids) == text.encode("utf-8"), text


@pytest.mark.parametrize("split", ["val", "train"])
def test_bpb_counts_every_text_byte_once_and_no_special(small_model, shard_dirs, corpus_spec, trained_tokenizer,
                                                         split):  # fmt: skip
    """Windows of 64 tokens share a token, so a long document spans rows; each of its targets counts once, the
    bytes are its text pieces' UTF-8 bytes (fence lines of text documents are not text), specials are apart."""
    result = loss.evaluate(small_model, shard_dirs[64], split=split, batch=3, scorer=_id_scorer, corpus=corpus_spec)
    expected = _expected_docs(corpus_spec, trained_tokenizer, split)
    assert result["documents"] == {"complete": len(expected), "partial": 0, "keys": "corpus"}
    assert any(n > 1 for n in _rows_per_document(shard_dirs[64], split)), "the fixture needs a windowed document"
    for domain in {e[1] for e in expected}:
        mine = [e for e in expected if e[1] == domain]
        n_bytes = sum(e[2] for e in mine)
        assert result["bpb"][domain]["bytes"] == n_bytes and result["bpb"][domain]["documents"] == len(mine)
        assert result["bpb"][domain]["bpb"] == pytest.approx(math.fsum(e[3] for e in mine) / (math.log(2) * n_bytes),
                                                              rel=1e-12)  # fmt: skip
        with_eos = math.fsum(e[3] + e[5] for e in mine) / (math.log(2) * n_bytes)
        assert result["bpb"][domain]["bpb_with_eos"] == pytest.approx(with_eos, rel=1e-12)
    assert sum(v["bytes"] for v in result["bpb_by_source"].values()) == sum(e[2] for e in expected)


def _rows_per_document(data, split):
    inputs = np.concatenate([shards.served_rows(shards.read_shard(p), 64)[0] for p in shard_files(data, split)])
    starts = np.flatnonzero(inputs[:, 0] == 1)
    return np.diff(np.append(starts, len(inputs)))


def test_dump_nll_writes_one_exact_line_per_document(tmp_path, small_model, shard_dirs, corpus_spec,
                                                      trained_tokenizer):  # fmt: skip
    import json

    out = tmp_path / "T0.nll"
    loss.evaluate(small_model, shard_dirs[64], batch=4, scorer=_id_scorer, dump=loss.NllDump(out, "T0"))
    header, *lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert header["format"] == "tokstudy-nll/1" and header["backend"] == "native" and header["table"] == "tlc-T"
    assert header["arm"] == "T0" and header["seed"] == 0 and header["keys"] == "shard"
    assert header["data_manifest_sha256"] == shards.manifest_sha256(shard_dirs[64])
    assert header["domains"] == {"code": "py", "text": "en"} and header["partial_documents"] == 0
    expected = _expected_docs(corpus_spec, trained_tokenizer, "val")
    assert [line["key"] for line in lines] == [f"val-00000.bin:{n}" for n in range(len(expected))]
    fields = ["key", "domain", "bytes", "nll_nats", "n_targets", "eos_nats", "n_eos", "special_nats", "n_special"]
    for line, e in zip(lines, expected, strict=True):
        assert list(line) == sorted(fields)
        assert (line["domain"], line["bytes"], line["n_targets"], line["n_eos"], line["n_special"]) == (
            e[1],
            e[2],
            e[4],
            e[6],
            e[8],
        )
        for field, value in (("nll_nats", e[3]), ("eos_nats", e[5]), ("special_nats", e[7])):
            assert line[field] == pytest.approx(value, rel=1e-12)


def test_bytes_per_document_do_not_depend_on_the_tokenizer(tmp_path, small_model, shard_dirs, corpus_spec,
                                                           tokenizer_dir):  # fmt: skip
    """The denominator of bits per byte is the document's text, so two tokenizers give every key the same bytes."""
    from surogate_tlc import hf, init, modeldir
    from surogate_tlc.data import corpus

    other_data = shards.build_shards(corpus.load_corpus(corpus_spec), tokenizer_dir, tmp_path / "gpt2", seq_len=64)
    assert other_data["tokenizer"]["sha256"] != shards.manifest(shard_dirs[64])["tokenizer"]["sha256"]
    cfg = hf.row_config("A0", num_hidden_layers=2, tlc_layer_pattern="MM", chunk_size=16, vocab_size=320)
    other = init.init_model_dir(modeldir.write_model_dir(tmp_path / "other", cfg, tokenizer_dir, init_seed=1)).parent
    a = loss.evaluate(small_model, shard_dirs[64], scorer=_id_scorer, corpus=corpus_spec, dump=loss.NllDump(
        tmp_path / "a.nll", "T0"))  # fmt: skip
    b = loss.evaluate(other, tmp_path / "gpt2", scorer=_id_scorer, corpus=corpus_spec, dump=loss.NllDump(
        tmp_path / "b.nll", "T1"))  # fmt: skip
    import json

    read = lambda p: {d["key"]: d["bytes"] for d in map(json.loads, p.read_text().splitlines()[1:])}  # noqa: E731
    assert read(tmp_path / "a.nll") == read(tmp_path / "b.nll")
    assert {d: v["bytes"] for d, v in a["bpb"].items()} == {d: v["bytes"] for d, v in b["bpb"].items()}


def test_a_document_cut_by_max_rows_is_partial_and_left_out(small_model, shard_dirs, corpus_spec, trained_tokenizer):
    per_doc = _rows_per_document(shard_dirs[64], "val")
    long = int(np.flatnonzero(per_doc > 1)[0])
    cut = int(per_doc[:long].sum()) + 1  # one row into the first windowed document
    result = loss.evaluate(small_model, shard_dirs[64], max_rows=cut, scorer=_id_scorer, corpus=corpus_spec)
    assert result["documents"] == {"complete": long, "partial": 1, "keys": "corpus"}
    expected = _expected_docs(corpus_spec, trained_tokenizer, "val")[:long]
    assert sum(v["bytes"] for v in result["bpb"].values()) == sum(e[2] for e in expected)
    whole = loss.evaluate(small_model, shard_dirs[64], max_rows=int(per_doc[: long + 1].sum()), scorer=_id_scorer)
    assert whole["documents"]["complete"] == long + 1 and whole["documents"]["partial"] == 0


def test_corpus_keys_must_describe_the_shards(tmp_path, small_model, shard_dirs, corpus_spec):
    changed = tmp_path / "corpus.toml"
    changed.write_text(corpus_spec.read_text(encoding="utf-8").replace("seed = 7", "seed = 8"), encoding="utf-8")
    for path in corpus_spec.parent.iterdir():
        if path.name != "corpus.toml":
            (tmp_path / path.name).symlink_to(path)
    with pytest.raises(ValueError, match="not the corpus spec the shards were built from"):
        loss.evaluate(small_model, shard_dirs[64], scorer=_id_scorer, corpus=changed)


def test_dump_nll_needs_an_arm_and_refuses_a_non_finite_nll(tmp_path, small_model, shard_dirs):
    from surogate_tlc.__main__ import main

    with pytest.raises(SystemExit, match="--dump-nll and --arm go together"):
        main(["eval", str(small_model), str(shard_dirs[64]), "--dump-nll", str(tmp_path / "x.nll")])
    nan = lambda i, t, p: np.where(t == -100, 0.0, np.nan).astype(np.float32)  # noqa: E731
    with pytest.raises(ValueError, match="not finite"):
        loss.evaluate(small_model, shard_dirs[64], scorer=nan, dump=loss.NllDump(tmp_path / "x.nll", "T0"))
    assert not (tmp_path / "x.nll").exists()
