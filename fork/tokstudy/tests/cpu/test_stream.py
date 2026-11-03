"""Documents, framing and the training stream (implementation step S9).

Framing parity: tokstudy's TLC documents, encoded with a TLC tokenizer, are exactly the token rows TLC's own shards
serve (TLC's code runs in an isolated interpreter, as the conftest explains), and its bilingual documents are
exactly the bilingual shard writer's tokens; the val (dev) keys are the projects' own split. Truncation never splits
a character; the stream is deterministic, independent of the input order, and holds its byte mixture within 0.5%;
the rows of two arms cover the same byte spans in the same order. Every corpus here is a SYNTHETIC FIXTURE written
by the tests.
"""

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from surogate_tokstudy import docs as D
from surogate_tokstudy import stream as S
from surogate_tokstudy.canonical import TokstudyError, sha256_file
from surogate_tokstudy.codecs import BILINGUAL_SPECIALS, TLC_SPECIALS, ByteCodec, SubwordCodec

REPO = Path(__file__).resolve().parents[4]

# --- a SYNTHETIC TLC corpus ------------------------------------------------------------------------------------------

TEXTS = (
    "Intro line.\n```python\nx = 1\r\ny = 'é'\n```\nAfter the block.\n",
    "Open fence never closed:\n```\nprint('<|eos|> stays text')\n",
    "No code at all, just prose about 变量 and 🙂.\r\n",
    "  ```js\nlet a = 2;\n```\n```\nsecond block\n```\ntail\n",
)


def write_tlc_corpus(root: Path, n_code: int = 60) -> Path:
    """A TLC corpus spec with a code source (groups of 3) and a text source with fences: SYNTHETIC FIXTURE."""
    root.mkdir(parents=True, exist_ok=True)
    code = [{"text": f"def f{i}(x):\n    return x * {i}  # <|code_end|>\n", "group": f"repo{i // 3}"}
            for i in range(n_code)]  # fmt: skip
    (root / "code.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in code) + "\n", "utf-8")
    text = [{"text": TEXTS[i % len(TEXTS)] + f"doc {i}\n"} for i in range(24)]
    (root / "text.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in text) + "\n", "utf-8")
    source = ('[[source]]\nname = "{0}"\nkind = "{1}"\nfiles = ["{0}.jsonl"]\nurl = "SYNTHETIC FIXTURE"\n'
              'license = "SYNTHETIC FIXTURE"\ndownloaded = "2026-11-03"\nfilters = []\n')  # fmt: skip
    (root / "spec.toml").write_text("seed = 5\nval_fraction = 0.25\n" + source.format("code", "code")
                                    + source.format("text", "text"), encoding="utf-8")  # fmt: skip
    return root / "spec.toml"


# argv: REPO, spec, out. Trains a TLC tokenizer (300 ids) on the spec's train split with TLC's own recipe, builds
# TLC's shards and prints, per split, the document keys and the token ids of every served row.
TLC_SHARDS_SCRIPT = """
import json, sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/fork/tlc/src"]
import numpy as np
from surogate_tlc.data import corpus, shards, tokenizer
spec, out = Path(sys.argv[2]), Path(sys.argv[3])
c = corpus.load_corpus(spec)
tok = tokenizer.train_tokenizer(c, out / "tok", vocab_size=300)
shards.build_shards(c, tok, out / "shards", seq_len=512)
result = {"tokenizer": str(tok / "tokenizer.json")}
for split in ("train", "val"):
    result[split + "_keys"] = [d.key for d in corpus.documents(c) if corpus.split_of(c, d) == split]
    rows = []
    for path in shards.shard_files(out / "shards", split):
        inputs, targets, _ = shards.served_rows(shards.read_shard(path), 512)
        for x, t in zip(inputs, targets):
            n = int((t != shards.IGNORE).sum()) + 1
            rows.append([int(i) for i in x[:n]])
    result[split] = rows
special = tokenizer.special_ids(tokenizer.load(tok))
result["segments"] = [shards.segments(d.text, d.source.kind, special) for d in corpus.documents(c)]
print(json.dumps(result))
"""


@pytest.fixture(scope="module")
def tlc_built(tmp_path_factory):
    root = tmp_path_factory.mktemp("tlc-corpus")
    spec = write_tlc_corpus(root)
    out = tmp_path_factory.mktemp("tlc-shards")
    command = [sys.executable, "-I", "-B", "-c", TLC_SHARDS_SCRIPT, str(REPO), str(spec), str(out)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    return spec, json.loads(result.stdout)


def test_tlc_documents_frame_exactly_as_tlc_shards_serve_them(tlc_built):
    spec, built = tlc_built
    path = Path(built["tokenizer"])
    codec = SubwordCodec.from_tokenizer_json(path, sha256_file(path), TLC_SPECIALS, check=True)
    for split in ("train", "val"):
        documents = list(D.tlc_documents(spec, split))
        assert [d.key for d in documents] == built[split + "_keys"]  # the project's own split
        assert [D.encode_doc(codec, d).ids.tolist() for d in documents] == built[split]
    assert {d.domain for d in D.tlc_documents(spec, "train")} == {"py", "en"}
    # the segments themselves, special by special, against surogate_tlc.data.shards.segments
    ids = {name: i for name, i in TLC_SPECIALS.items()}
    every = [*D.tlc_documents(spec, "train"), *D.tlc_documents(spec, "val")]
    order = {key: n for n, key in enumerate(built["train_keys"] + built["val_keys"])}
    framed = [None] * len(every)
    for doc in every:
        framed[order[doc.key]] = [ids[s.special] if s.special else s.text for s in doc.segs]
    from surogate_tlc.data.corpus import documents, load_corpus

    keys = [d.key for d in documents(load_corpus(spec))]
    by_key = dict(zip(keys, built["segments"], strict=True))
    assert all(f == by_key[k] for k, f in zip(built["train_keys"] + built["val_keys"], framed, strict=True))


def test_tlc_documents_refuse_an_unknown_split(tlc_built):
    with pytest.raises(TokstudyError, match="a TLC split"):
        list(D.tlc_documents(tlc_built[0], "dev"))


# --- the bilingual pipeline ------------------------------------------------------------------------------------------


def _han(rng, n):
    return "".join(chr(0x4E00 + rng.randrange(0, 2000)) for _ in range(n))


def _words(rng, n):
    return " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "x1", "2026"]) for _ in range(n))


@pytest.fixture(scope="module")
def bilingual_built(tmp_path_factory):
    """manifest -> split -> materialize -> tokenizer with the bilingual project's own code, on a SYNTHETIC corpus."""
    from surogate_bilingual.data import manifest, split, tokenizer

    rng = random.Random(7)
    base = tmp_path_factory.mktemp("bilingual")
    corpus = base / "corpus"
    corpus.mkdir()
    zh = [{"id": f"z{i}", "text": _han(rng, 20) + "，" + _han(rng, 8) + "。"} for i in range(60)]
    en = [{"id": f"e{i}", "text": _words(rng, 15) + ", <|en|> stays text."} for i in range(60)]
    cs = [{"id": f"c{i}", "segments": [{"lang": "zh", "text": _han(rng, 6)}, {"tag": "<|en|>"},
                                       {"lang": "en", "text": _words(rng, 4)}]} for i in range(10)]  # fmt: skip
    for name, rows in (("zh", zh), ("en", en), ("cs", cs)):
        (corpus / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                             encoding="utf-8")  # fmt: skip
    (corpus / "spec.toml").write_text(
        '[[source]]\nname = "zh-fix"\nkind = "zh"\nfiles = ["zh.jsonl"]\nlicence = "CC0-1.0"\n'
        '[[source]]\nname = "en-fix"\nkind = "en"\nfiles = ["en.jsonl"]\nlicence = "CC0-1.0"\n'
        '[[source]]\nname = "cs-fix"\nkind = "codeswitch"\nfiles = ["cs.jsonl"]\nlicence = "CC0-1.0"\n',
        encoding="utf-8",
    )
    m = manifest.load_manifest(manifest.build_manifest(corpus / "spec.toml", corpus, base / "manifest.json"))
    s = split.split_doc(m.iter_docs(), split.default_fractions(), seed=1, manifest_sha256=m.sha256)
    derived_path = split.write_split_corpora(m, s, base / "splits")
    derived = manifest.load_manifest(derived_path)
    files = [derived.root / e["path"] for e in derived.files(split="seed_train")]
    tok_path = tokenizer.train_tokenizer(files, 400, base / "tok" / "tokenizer.json", derived)
    return derived_path, derived, tok_path


def test_bilingual_documents_match_the_shard_writer_and_the_split(bilingual_built):
    from surogate_bilingual.data import shards, tokenizer

    derived_path, derived, tok_path = bilingual_built
    tok = tokenizer.load_tokenizer(tok_path)
    codec = SubwordCodec.from_tokenizer_json(tok_path, sha256_file(tok_path), BILINGUAL_SPECIALS, check=True)
    for split in ("seed_train", "dev"):
        ours = list(D.bilingual_documents(derived_path, split))
        theirs = list(derived.iter_docs(split=split))
        assert [d.key for d in ours] == [d.key for d in theirs] and ours
        for mine, doc in zip(ours, theirs, strict=True):
            ids, _, _, _ = shards.encode_doc(doc, tok, ("seed",), "seed")
            assert D.encode_doc(codec, mine).ids.tolist() == [BILINGUAL_SPECIALS["<eos>"], *ids]
    domains = {d.domain for d in D.bilingual_documents(derived_path, "seed_train")}
    assert domains == {"zh", "en", "mixed"}
    # a code-switched document forces a boundary where the language changes, and at its tag
    mixed = next(d for d in D.bilingual_documents(derived_path, "seed_train") if d.domain == "mixed")
    enc = D.encode_doc(ByteCodec(BILINGUAL_SPECIALS), mixed)
    tag = int(np.flatnonzero(enc.ids == BILINGUAL_SPECIALS["<|en|>"])[0])
    assert enc.forced()[tag] and enc.route_start[tag + 1] and enc.route_start.sum() == 1


def test_bilingual_documents_check_files_and_refuse_loss_free_segments(bilingual_built, tmp_path):
    derived_path = bilingual_built[0]
    with pytest.raises(TokstudyError, match="no files of split"):
        list(D.bilingual_documents(derived_path, "no-such-split"))
    from surogate_bilingual.data.docs import Doc, Segment

    doc = Doc("k", "en", (Segment(text="hidden", lang="en", loss=False),))
    with pytest.raises(TokstudyError, match="loss = false"):
        D.bilingual_doc(doc)
    raw = json.loads(Path(derived_path).read_text(encoding="utf-8"))
    raw["derived_from"] = None
    (tmp_path / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(TokstudyError, match="not a derived manifest"):
        list(D.bilingual_documents(tmp_path / "manifest.json", "dev"))


# --- truncation and windows ------------------------------------------------------------------------------------------

ALPHABET = ("a", " ", "\n", "é", "变", "🙂", "𠀀", " ", "\r")


def random_doc(rng, n_segs=6):
    segs = [D.Seg("<|bos|>")]
    for _ in range(n_segs):
        if rng.random() < 0.3:
            segs.append(D.Seg(rng.choice(["<|code_start|>", "<|code_end|>"])))
        else:
            segs.append(D.Seg(text="".join(rng.choice(ALPHABET) for _ in range(rng.randrange(1, 12)))))
    segs.append(D.Seg("<|eos|>"))
    return D.EvalDoc(f"doc{rng.random()}", "py", tuple(segs))


def test_truncate_keeps_a_utf8_prefix_and_drops_later_specials():
    rng = random.Random(0)
    for _ in range(60):
        doc = random_doc(rng)
        full = "".join(doc.texts()).encode()
        for k in range(len(full) + 2):
            cut = D.truncate(doc, k)
            text = "".join(cut.texts()).encode()
            assert full.startswith(text) and len(text) <= k
            assert len(text) == len(full[:k].decode("utf-8", "ignore").encode())  # the longest character prefix
            if k >= len(full):
                assert cut == doc
            else:  # the specials kept are exactly those at text offsets below the cut, in order
                offset, kept = 0, []
                for seg in doc.segs:
                    if seg.special is not None and offset < k:
                        kept.append(seg.special)
                    offset += len(seg.text.encode()) if seg.text is not None else 0
                assert [x.special for x in cut.segs if x.special is not None] == kept
    doc = D.EvalDoc("d", "en", (D.Seg("<|bos|>"), D.Seg(text="ab"), D.Seg("<|eos|>")))
    assert D.truncate(doc, 2) == doc and D.truncate(doc, 1).segs == (D.Seg("<|bos|>"), D.Seg(text="a"))
    assert D.truncate(doc, 0).segs == ()


def test_windows_hold_at_most_k_positions_at_character_boundaries():
    rng = random.Random(1)
    codec = ByteCodec(TLC_SPECIALS)
    for _ in range(40):
        doc = random_doc(rng, 10)
        for k in (8, 9, 13, 64):
            ws = S.windows(doc, k)
            assert all(len(D.encode_doc(codec, w.segs).ids) <= k for w in ws)
            joined = b"".join("".join(t.text for t in w.segs if t.text is not None).encode() for w in ws)
            assert joined == "".join(doc.texts()).encode()
            spans = [w.span for w in ws]
            assert spans[0][0] == 0 and spans[-1][1] == len(joined)
            assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))
            assert [w.last for w in ws] == [False] * (len(ws) - 1) + [True]
            assert sum(len(w.segs) for w in ws) >= len(doc.segs)
    with pytest.raises(TokstudyError, match="at least"):
        S.windows(doc, 4)


# --- the stream ------------------------------------------------------------------------------------------------------


def mixture_docs(n=1200, seed=3):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        domain = "py" if i % 3 else "en"
        text = "".join(rng.choice(ALPHABET[:5]) for _ in range(rng.randrange(20, 200)))
        out.append(D.EvalDoc(f"{domain}/{i}", domain, (D.Seg("<|bos|>"), D.Seg(text=text), D.Seg("<|eos|>"))))
    return out


SOURCE = {"project": "tlc", "fixture": "SYNTHETIC"}


def test_the_stream_is_deterministic_and_independent_of_input_order(tmp_path):
    docs = mixture_docs()
    a = S.build_stream(docs, tmp_path / "a", seed=0, k_train=64, source=SOURCE, mixture={"py": 0.7, "en": 0.3})
    shuffled = list(docs)
    random.Random(9).shuffle(shuffled)
    b = S.build_stream(shuffled, tmp_path / "b", seed=0, k_train=64, source=SOURCE, mixture={"py": 0.7, "en": 0.3})
    assert a.doc == b.doc and a.sha256 == b.sha256
    for part in a.doc["parts"]:
        assert (tmp_path / "a" / part["path"]).read_bytes() == (tmp_path / "b" / part["path"]).read_bytes()
    c = S.build_stream(docs, tmp_path / "c", seed=1, k_train=64, source=SOURCE, mixture={"py": 0.7, "en": 0.3})
    assert c.doc["order_sha256"] != a.doc["order_sha256"]
    keys_a = [d.key for _, d in S.iter_docs(S.open_stream(tmp_path / "a"))]
    keys_c = [d.key for _, d in S.iter_docs(S.open_stream(tmp_path / "c"))]
    assert keys_a != keys_c
    assert [d.key for _, d in S.iter_docs(a, 17)] == keys_a[17:]


def test_the_stream_holds_its_byte_mixture(tmp_path):
    docs = mixture_docs()
    st = S.build_stream(docs, tmp_path / "s", seed=0, k_train=64, source=SOURCE, mixture={"py": 0.6, "en": 0.4})
    by = st.doc["by_domain"]
    total = by["py"]["bytes"] + by["en"]["bytes"]
    assert abs(by["en"]["bytes"] / total - 0.4) <= 0.005 and abs(by["py"]["bytes"] / total - 0.6) <= 0.005
    assert st.doc["achieved_mixture"]["en"] == pytest.approx(by["en"]["bytes"] / total)
    # the scarce domain (en: a third of the documents, here 40 % of the bytes) is taken whole
    en_total = sum(d.text_bytes() for d in docs if d.domain == "en")
    assert by["en"]["bytes"] == en_total and st.doc["unused_docs"] > 0
    # every prefix of the stream stays near the mixture too (the interleave), after the first few documents
    seen = {"py": 0, "en": 0}
    for n, (_, doc) in enumerate(S.iter_docs(st), 1):
        seen[doc.domain] += doc.text_bytes()
        if n >= 100:
            assert abs(seen["en"] / (seen["en"] + seen["py"]) - 0.4) < 0.02
    capped = S.build_stream(docs, tmp_path / "cap", seed=0, k_train=64, source=SOURCE, mixture={"py": 0.6, "en": 0.4},
                            max_units=20000)  # fmt: skip
    assert 20000 <= capped.doc["units"] < 20000 + 2 * 200
    with pytest.raises(TokstudyError, match="too few units"):
        S.build_stream(docs, tmp_path / "big", seed=0, k_train=64, source=SOURCE, mixture={"py": 0.6, "en": 0.4},
                       max_units=10**9)  # fmt: skip


def test_the_stream_refuses_bad_inputs(tmp_path):
    docs = mixture_docs(30)
    with pytest.raises(TokstudyError, match="occurs twice"):
        S.build_stream(docs + docs[:1], tmp_path / "dup", seed=0, k_train=64, source=SOURCE)
    with pytest.raises(TokstudyError, match="positive shares"):
        S.build_stream(docs, tmp_path / "mix", seed=0, k_train=64, source=SOURCE, mixture={"py": 0.5})
    with pytest.raises(TokstudyError, match="no documents of domain"):
        S.build_stream(docs, tmp_path / "zh", seed=0, k_train=64, source=SOURCE, mixture={"py": 0.5, "zh": 0.5})
    st = S.build_stream(docs, tmp_path / "ok", seed=0, k_train=64, source=SOURCE, mixture={"py": 1.0})
    assert st.doc["dropped"] == {"domain en not in the mixture": 10}
    with pytest.raises(TokstudyError, match="not empty"):
        S.build_stream(docs, tmp_path / "ok", seed=0, k_train=64, source=SOURCE)
    part = tmp_path / "ok" / st.doc["parts"][0]["path"]
    part.write_bytes(part.read_bytes() + b"\n")
    with pytest.raises(TokstudyError, match="sha256"):
        S.open_stream(tmp_path / "ok")


def test_rows_of_two_arms_cover_the_same_byte_spans_in_the_same_order(tmp_path, bpe_writer):
    docs = mixture_docs(80)
    st = S.build_stream(docs, tmp_path / "s", seed=0, k_train=48, source=SOURCE, mixture={"py": 0.5, "en": 0.5})
    path, sha = bpe_writer(tmp_path / "tok" / "tokenizer.json", specials=tuple(TLC_SPECIALS), pre="gpt2",
                           train_on=[d.texts()[0] for d in docs], vocab_size=320)  # fmt: skip
    sub = SubwordCodec.from_tokenizer_json(path, sha, TLC_SPECIALS, check=True)
    byte = ByteCodec(TLC_SPECIALS, check=True)
    rows_s, rows_b = list(S.rows_for_arm(st, sub)), list(S.rows_for_arm(st, byte))
    assert [(r.doc, r.window, r.span) for r in rows_s] == [(r.doc, r.window, r.span) for r in rows_b]
    assert all(len(r.ids) <= 48 for r in rows_b) and max(len(r.ids) for r in rows_b) == 48
    assert sum(len(r.ids) for r in rows_s) < sum(len(r.ids) for r in rows_b)  # the subword arm compresses
    assert all(np.array_equal(r.forced, r.ids < 5) for r in rows_b)
    # resuming at a cursor gives exactly the tail of the rows
    k = 23
    tail = list(S.rows_for_arm(st, byte, start=rows_b[k].next_cursor))
    assert [(r.doc, r.window) for r in tail] == [(r.doc, r.window) for r in rows_b[k + 1 :]]
    c = S.bytes_per_token(st, sub)
    assert c > 1 and S.t_arm(48, c) % 8 == 0


def test_stream_command_line(tmp_path):
    spec = write_tlc_corpus(tmp_path / "corpus")
    prereg = tmp_path / "prereg.toml"
    prereg.write_text('format = "tokstudy-prereg/1"\ntable = "tlc-N"\n[budget]\nF_target = 1e9\n'
                      '[context]\nK_train = 64\nK_eval = 64\n[mixture]\npy = 0.7\nen = 0.3\n', encoding="utf-8")  # fmt: skip
    command = [sys.executable, "-I", "-B", "-c", "import sys; sys.path[:0] = sys.argv[1:4]; "
               "from surogate_tokstudy.__main__ import main; sys.exit(main(sys.argv[4:]))",
               *(str(REPO / "fork" / p / "src") for p in ("tokstudy", "tlc", "bilingual")), "stream"]  # fmt: skip
    ok = subprocess.run([*command, "--prereg", str(prereg), "--tlc", str(spec), "--out", str(tmp_path / "s")],
                        capture_output=True, text=True, timeout=120)  # fmt: skip
    assert ok.returncode == 0, ok.stderr
    summary = json.loads(ok.stdout)
    assert summary["docs"] > 0 and set(summary["achieved_mixture"]) == {"py", "en"}
    bad = subprocess.run([*command, "--prereg", str(prereg), "--out", str(tmp_path / "t")], capture_output=True,
                         text=True, timeout=120)  # fmt: skip
    assert bad.returncode == 1 and "give exactly one of" in bad.stderr
