"""`compose`: one project corpus spec from several export directories. Fixture parquet (SYNTHETIC FIXTURE content)
-> fetch (fake Hub) -> compose, read back with the projects' own loaders from this checkout: TLC's corpus spec (paths
relative to the spec, `..` allowed) and the bilingual manifest (paths relative to the corpus root)."""

import json

import pytest
from corpora_fixtures import MARK, export_fixture, stack3_rows, ufw_rows, unpc_rows
from surogate_corpora.compose import compose, main

INCLUDED = """# SYNTHETIC FIXTURE: an owner-written source next to the exports
[[source]]
name = "synthetic-binding"
kind = "code"
files = ["synth/train.jsonl"]
url = "https://example.invalid/synth"
license = "MIT"
downloaded = "2026-11-03"
generated = true
filters = []
"""


def _tlc_exports(hub, client_for, monkeypatch, root):
    py = export_fixture(
        hub,
        client_for,
        monkeypatch,
        root / "exports" / "py",
        "stack3-py",
        "tlc-python-permissive",
        [stack3_rows(60), stack3_rows(60, offset=60)],
    )
    en = export_fixture(
        hub, client_for, monkeypatch, root / "exports" / "en", "ufw-en", "tlc-english", [ufw_rows(30, zh=False)], rg=10
    )
    return py, en


def test_a_tlc_spec_composed_elsewhere_reads_every_exported_document(fake_hub, client_for, monkeypatch, tmp_path):
    from surogate_tlc.data import corpus as tlc

    (_, py_doc, _), (_, en_doc, _) = _tlc_exports(fake_hub, client_for, monkeypatch, tmp_path)
    (tmp_path / "corpus" / "synth").mkdir(parents=True)
    (tmp_path / "corpus" / "synth" / "train.jsonl").write_text(
        json.dumps({"text": f"# {MARK}\nx = 1\n"}) + "\n", encoding="utf-8"
    )
    include = tmp_path / "corpus" / "synth.toml"
    include.write_text(INCLUDED, encoding="utf-8")
    spec = tmp_path / "corpus" / "spec.toml"
    summary = compose(
        "tlc",
        [tmp_path / "exports" / "py", tmp_path / "exports" / "en"],
        spec,
        includes=[include],
        top={"seed": 5, "val_fraction": 0.2},
    )
    corpus = tlc.load_corpus(spec)
    assert [(s.name, s.kind) for s in corpus.sources] == [
        ("stack3-py", "code"),
        ("ufw-en", "text"),
        ("synthetic-binding", "code"),
    ]
    assert corpus.seed == 5 and corpus.val_fraction == 0.2
    assert '"../exports/py/parts/' in spec.read_text(encoding="utf-8")  # relative to the spec, outside its directory
    docs = list(tlc.documents(corpus))
    n_py, n_en = py_doc["totals"]["docs"], en_doc["totals"]["docs"]
    assert [d.source.name for d in docs].count("stack3-py") == n_py > 0
    assert [d.source.name for d in docs].count("ufw-en") == n_en > 0
    assert sum(len(d.text.encode("utf-8")) for d in docs if d.source.name != "synthetic-binding") == (
        py_doc["totals"]["text_bytes"] + en_doc["totals"]["text_bytes"]
    )
    # the byte mixture the plans produced, by kind of the exports
    mix = summary["mixture_by_bytes"]
    total = py_doc["totals"]["text_bytes"] + en_doc["totals"]["text_bytes"]
    assert mix["code"]["text_bytes"] == py_doc["totals"]["text_bytes"]
    assert mix["text"]["share"] == round(en_doc["totals"]["text_bytes"] / total, 6)
    text = spec.read_text(encoding="utf-8")
    assert py_doc["plan_sha256"] in text and py_doc["provenance_sha256"] in text and "included verbatim" in text
    assert summary["included"] == [{"file": "synth.toml", "sources": ["synthetic-binding"]}]


def test_a_bilingual_spec_under_the_corpus_root_builds_a_manifest(fake_hub, client_for, monkeypatch, tmp_path):
    from surogate_bilingual.data.manifest import build_manifest, load_manifest

    root = tmp_path / "corpus"
    _, zh, _ = export_fixture(
        fake_hub, client_for, monkeypatch, root / "ufw-zh", "ufw-zh", "bilingual-zh-web", [ufw_rows(48, zh=True)], rg=16
    )
    _, en, _ = export_fixture(
        fake_hub, client_for, monkeypatch, root / "ufw-en", "ufw-en", "bilingual-en", [ufw_rows(30, zh=False)], rg=10
    )
    _, par, _ = export_fixture(
        fake_hub,
        client_for,
        monkeypatch,
        root / "unpc",
        "unpc-en-zh",
        "bilingual-parallel-unpc",
        [unpc_rows(20)],
        rg=10,
    )
    spec = tmp_path / "spec.toml"
    summary = compose("bilingual", [root / "ufw-zh", root / "ufw-en", root / "unpc"], spec, root=root)
    manifest = load_manifest(build_manifest(spec, root, tmp_path / "manifest.json"))
    assert set(manifest.sources) == {
        "ufw-zh-cci3",
        "ufw-zh-chinesewebtext",
        "ufw-zh-industrycorpus",
        "ufw-en",
        "unpc-en-zh",
    }
    assert manifest.doc["unverified_licences"] == sorted(manifest.sources)
    docs = list(manifest.iter_docs())
    assert len(docs) == zh["totals"]["docs"] + en["totals"]["docs"] + par["totals"]["docs"]
    assert {d.kind for d in docs} == {"zh", "en", "parallel"}
    assert set(summary["mixture_by_bytes"]) == {"zh", "en", "parallel"}


def test_compose_refuses_what_would_make_a_wrong_spec(fake_hub, client_for, monkeypatch, tmp_path, capsys):
    _tlc_exports(fake_hub, client_for, monkeypatch, tmp_path)
    exports = [tmp_path / "exports" / "py", tmp_path / "exports" / "en"]
    spec = tmp_path / "spec.toml"

    def refused(argv, message):
        assert main(argv) == 1
        err = capsys.readouterr().err
        assert message in err, err
        assert not spec.exists()

    args = ["--out", str(spec), *(a for e in exports for a in ("--export", str(e)))]
    refused(["--project", "bilingual", *args], "writes the tlc format, not a bilingual corpus")
    refused(["--project", "tlc", *args, "--export", str(exports[0])], "source names occur twice: ['stack3-py']")
    refused(["--project", "tlc", *args, "--root", str(tmp_path)], "--root is bilingual's corpus root")
    refused(["--project", "bilingual", "--out", str(spec), "--seed", "1", "--export", str(exports[0])], "TLC spec keys")
    bad = tmp_path / "bad.toml"
    bad.write_text("seed = 3\n", encoding="utf-8")
    refused(["--project", "tlc", *args, "--include", str(bad)], "holds [[source]] tables only")
    # an export that does not verify: one byte of a part changed
    part = next((exports[1] / "parts").iterdir())
    data = bytearray(part.read_bytes())
    data[-2] ^= 1
    part.write_bytes(bytes(data))
    refused(["--project", "tlc", *args], "the export does not verify")
    part.write_bytes(bytes(data[:-2]) + bytes([data[-2] ^ 1]) + bytes(data[-1:]))
    assert main(["--project", "tlc", *args, "--seed", "2"]) == 0
    assert "seed = 2\nval_fraction = 0.01\n" in spec.read_text(encoding="utf-8")
    capsys.readouterr()
    assert main(["--project", "tlc", *args]) == 1  # never overwrites
    assert "never overwrites" in capsys.readouterr().err


def test_a_bilingual_export_outside_the_root_is_refused(fake_hub, client_for, monkeypatch, tmp_path):
    from surogate_corpora.canonical import CorporaError

    export_fixture(
        fake_hub,
        client_for,
        monkeypatch,
        tmp_path / "elsewhere",
        "ufw-en",
        "bilingual-en",
        [ufw_rows(10, zh=False)],
        rg=10,
    )
    (tmp_path / "root").mkdir()
    with pytest.raises(CorporaError, match="outside the corpus root"):
        compose("bilingual", [tmp_path / "elsewhere"], tmp_path / "root" / "spec.toml", root=tmp_path / "root")
