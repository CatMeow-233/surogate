"""I4-D5: shards (bilingual-shard/1; IMPL-07, IMPL-03)."""

import json

import numpy as np
import pytest

from surogate_bilingual.data import shards as SH
from surogate_bilingual.data.docs import Doc, Segment
from surogate_bilingual.data.tokenizer import encode_user_text

ROUTES = ["zh", "en", "bridge", "seed"]


def _docs():
    return [
        Doc("a", "zh", (Segment(text="第一篇，中文。", lang="zh"),)),
        Doc("b", "en", (Segment(text="The second one.", lang="en"),), route="en"),
        Doc("c", "parallel", (Segment(text="源句。\n", lang="zh", route="zh", loss=False),
                              Segment(text="Target text.", lang="en", route="bridge"))),
        Doc("d", "codeswitch", (Segment(tag="<|zh|>"), Segment(text="我说", lang="zh"), Segment(tag="<|en|>"),
                                Segment(text=" hello <|zh|> literally", lang="en"))),
    ]  # fmt: skip


def _write(tmp_path, pipeline, docs=None, **kw):
    return SH.write_shard(tmp_path / "shard", docs or _docs(), pipeline["tok"], ROUTES, "seed_train", "m" * 64,
                          tokenizer_sha256=pipeline["tokenizer_sha256"], **kw)  # fmt: skip


def _reader(path, pipeline, **kw):
    return SH.ShardReader(path, tokenizer_sha256=pipeline["tokenizer_sha256"],
                          vocab_size=pipeline["tok"].get_vocab_size(), **kw)  # fmt: skip


def _expected(docs, tok, default=None):
    ids, routes, loss, langs = [], [], [], []
    for d in docs:
        last_r, last_l = None, 1
        for s in d.segments:
            piece = [tok.token_to_id(s.tag)] if s.tag else encode_user_text(tok, s.text)
            r = s.route or d.route or default
            r = 255 if r is None else ROUTES.index(r)
            ids += piece
            routes += [r] * len(piece)
            loss += [int(s.loss)] * len(piece)
            langs += [s.lang if s.text is not None else None] * len(piece)
            last_r, last_l = r, int(s.loss)
        ids.append(1)
        routes.append(last_r)
        loss.append(last_l)
        langs.append(None)
    return ids, routes, loss, langs


def test_i4_d5_round_trip(tmp_path, data_pipeline):
    tok = data_pipeline["tok"]
    path = _write(tmp_path, data_pipeline, default_route="seed", seq=8)
    r = _reader(path, data_pipeline)
    ids, routes, loss, _ = _expected(_docs(), tok, "seed")
    assert r.tokens.tolist() == ids and r.routes.tolist() == routes and r.loss_mask.tolist() == loss
    assert r.header["n_docs"] == 4 and r.header["eos_between_docs"] is True
    offsets = r.doc_offsets.tolist()
    assert offsets[0] == 0 and offsets[-1] == len(ids) and len(offsets) == 5
    for i in range(4):  # every document ends with <eos>
        assert r.tokens[offsets[i + 1] - 1] == 1
    # the literal "<|zh|>" inside user text stayed text; only the two tag segments are control ids
    assert r.header["tag_counts"] == {"<|zh|>": 1, "<|en|>": 1, "<|bridge|>": 0}
    assert r.header["route_names"] == ROUTES and r.header["split"] == "seed_train"
    assert set(r.header["files"]) == set(SH.FILES)


def test_i4_d5_tampered_file_and_wrong_tokenizer_are_refused(tmp_path, data_pipeline):
    path = _write(tmp_path, data_pipeline)
    with pytest.raises(SH.ShardError, match="tokenizer"):
        SH.ShardReader(path, tokenizer_sha256="0" * 64, vocab_size=None)
    data = bytearray((path / "routes.bin").read_bytes())
    data[0] ^= 1
    (path / "routes.bin").write_bytes(bytes(data))
    with pytest.raises(SH.ShardError, match="sha256 differs"):
        _reader(path, data_pipeline)
    header = json.loads((path / "shard.json").read_text())
    header["format"] = "bilingual-shard/2"
    (path / "shard.json").write_text(json.dumps(header))
    with pytest.raises(ValueError, match="not a known version"):
        _reader(path, data_pipeline)


class _BigVocab:
    """A tokenizer stand-in with 70,000 ids, for the uint32 path."""

    encode_special_tokens = False

    def get_vocab_size(self):
        return 70_000

    def token_to_id(self, text):
        return {"<unk>": 0, "<eos>": 1, "<|zh|>": 2, "<|en|>": 3, "<|bridge|>": 4}.get(text)

    def encode(self, text, add_special_tokens=False):
        class E:
            ids = [69_999 - (ord(c) % 1000) for c in text]

        return E()


def test_i4_d5_dtype_follows_the_vocabulary_and_is_read_from_the_header(tmp_path, data_pipeline):
    assert SH.token_dtype(65535) == "uint16" and SH.token_dtype(65536) == "uint32"
    small = _write(tmp_path, data_pipeline)
    assert json.loads((small / "shard.json").read_text())["dtype"] == "uint16"
    assert (small / "tokens.bin").stat().st_size == 2 * json.loads((small / "shard.json").read_text())["n_tokens"]
    big = SH.write_shard(
        tmp_path / "big", _docs()[:2], _BigVocab(), ROUTES, "seed_train", None, tokenizer_sha256="b" * 64
    )
    header = json.loads((big / "shard.json").read_text())
    assert header["dtype"] == "uint32" and (big / "tokens.bin").stat().st_size == 4 * header["n_tokens"]
    r = SH.ShardReader(big, tokenizer_sha256="b" * 64, vocab_size=70_000)
    assert r.tokens.dtype == np.dtype("<u4") and int(r.tokens.max()) > 65535
    with pytest.raises(SH.ShardError, match="does not fit"):
        SH.ShardReader(big, tokenizer_sha256="b" * 64, vocab_size=32_768)
    with pytest.raises(SH.ShardError, match="does not fit"):
        _reader(small, data_pipeline | {"tok": _BigVocab()})


def _brute_mixed(langs, seq):
    starts = range(len(langs) - seq + 1)
    return sum(1 for s in starts if {"zh", "en"} <= set(langs[s : s + seq])) / len(starts)


@pytest.mark.parametrize("seq", [3, 8, 64])
def test_i4_d5_tag_counts_and_mixed_windows_equal_brute_force(tmp_path, data_pipeline, seq):
    tok = data_pipeline["tok"]
    docs = list(data_pipeline["derived"].iter_docs(split="seed_train"))[:120] + _docs()
    path = _write(tmp_path, data_pipeline, docs=docs, seq=seq)
    header = json.loads((path / "shard.json").read_text())
    ids, _, _, langs = _expected(docs, tok)
    assert header["mixed_window_fraction"]["value"] == pytest.approx(_brute_mixed(langs, seq), abs=1e-15)
    assert header["mixed_window_fraction"]["seq"] == seq
    assert header["tag_counts"] == {t: ids.count(tok.token_to_id(t)) for t in ("<|zh|>", "<|en|>", "<|bridge|>")}
    # the chunked scan agrees with itself across chunk sizes (the carry between chunks)
    codes = np.array([SH.LANG_CODE[x] for x in langs], dtype=np.uint8)
    assert all(SH.mixed_window_fraction(codes, seq, chunk=c) == header["mixed_window_fraction"]["value"]
               for c in (1, 7, 100))  # fmt: skip


def test_i4_d5_seed_shard_holds_each_tag_at_least_the_declared_minimum(tmp_path, data_pipeline, seed_mixture):
    _, docs, card = seed_mixture  # tag share s = DEFAULTS seed_mix["tagged"] (0.03)
    tok = data_pipeline["tok"]
    assert card["declared_shares"]["tagged"] == 0.03
    declared = card["min_tag_counts"]
    assert declared["<|zh|>"] > 0 and declared["<|en|>"] > 0 and declared["<|bridge|>"] == 0
    path = _write(tmp_path, data_pipeline, docs=docs, min_tag_counts=declared, default_route="seed")
    header = json.loads((path / "shard.json").read_text())
    for tag, minimum in declared.items():
        assert header["tag_counts"][tag] >= minimum
    with pytest.raises(SH.ShardError, match="below the declared minimum"):
        SH.write_shard(tmp_path / "short", [d for d in docs if d.meta.get("mix") != "tagged"], tok, ROUTES,
                       "seed_train", None, tokenizer_sha256="x", min_tag_counts=declared)  # fmt: skip
    assert not (tmp_path / "short").exists()  # a refused shard leaves nothing behind
