"""I4-D2: the corpus manifest (bilingual-manifest/1)."""

import json
import os

import pytest

from surogate_bilingual.data import manifest as M


def test_i4_d2_every_file_has_sha256_bytes_and_document_count(corpus_writer, tmp_path):
    spec = corpus_writer(tmp_path / "c", n=40)
    doc = json.loads(M.build_manifest(spec, tmp_path / "c", tmp_path / "m.json").read_text(encoding="utf-8"))
    assert doc["format"] == "bilingual-manifest/1"
    files = {f["path"]: f for f in doc["files"]}
    assert set(files) == {"zh.jsonl", "en.txt", "parallel.jsonl", "codeswitch.jsonl"}
    from surogate_bilingual.canonical import sha256_file

    for path, f in files.items():
        assert f["sha256"] == sha256_file(tmp_path / "c" / path)
        assert f["bytes"] == (tmp_path / "c" / path).stat().st_size
    assert files["zh.jsonl"]["n_docs"] == 40 and files["en.txt"]["n_docs"] == 40  # blank-line separated .txt
    assert files["parallel.jsonl"]["n_docs"] == 20 and files["codeswitch.jsonl"]["n_docs"] == 20
    for source in doc["sources"]:
        assert set(source) == set(M.SOURCE_FIELDS)


def test_i4_d2_missing_licence_is_an_error(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"text": "x"}\n', encoding="utf-8")
    spec = tmp_path / "spec.toml"
    spec.write_text('[[source]]\nname = "a"\nkind = "zh"\nfiles = ["a.jsonl"]\n', encoding="utf-8")
    with pytest.raises(M.ManifestError, match="`licence` is required"):
        M.build_manifest(spec, tmp_path, tmp_path / "m.json")
    assert not (tmp_path / "m.json").exists()
    spec.write_text('[[source]]\nname = "a"\nkind = "zh"\nfiles = ["a.jsonl"]\nlicence = "free for all"\n')
    with pytest.raises(M.ManifestError, match="SPDX-style"):
        M.build_manifest(spec, tmp_path, tmp_path / "m.json")


def test_i4_d2_licence_verified_defaults_to_false_and_is_surfaced(corpus_writer, tmp_path):
    spec = corpus_writer(tmp_path / "c", n=10)
    doc = json.loads(M.build_manifest(spec, tmp_path / "c", tmp_path / "m.json").read_text(encoding="utf-8"))
    verified = {s["name"]: s["licence_verified"] for s in doc["sources"]}
    assert verified == {"zh-nat": False, "en-nat": True, "par": False, "cs": False}
    assert doc["unverified_licences"] == ["cs", "par", "zh-nat"]  # the D_seed card copies it (test_seed_mix.py)


def test_i4_d2_manifest_sha256_is_stable_across_runs_and_listing_order(corpus_writer, tmp_path, monkeypatch):
    spec = corpus_writer(tmp_path / "c", n=20)
    first = M.manifest_doc(spec, tmp_path / "c")["manifest_sha256"]
    assert M.manifest_doc(spec, tmp_path / "c")["manifest_sha256"] == first
    real_glob = type(tmp_path).glob

    def reversed_glob(self, pattern):
        return reversed(sorted(real_glob(self, pattern)))

    monkeypatch.setattr(type(tmp_path), "glob", reversed_glob)
    assert M.manifest_doc(spec, tmp_path / "c")["manifest_sha256"] == first
    monkeypatch.undo()
    # a copy of the corpus elsewhere (another root, another manifest location) gives the same hash
    import shutil

    shutil.copytree(tmp_path / "c", tmp_path / "elsewhere" / "c")
    out = M.build_manifest(
        tmp_path / "elsewhere" / "c" / "spec.toml", tmp_path / "elsewhere" / "c", tmp_path / "x.json"
    )
    loaded = M.load_manifest(out)
    assert loaded.sha256 == first and loaded.root == (tmp_path / "elsewhere" / "c").resolve()
    assert os.path.isabs(json.loads(out.read_text())["root"]) is False  # stored relative to the manifest
    # changing one byte changes the hash
    (tmp_path / "c" / "zh.jsonl").write_bytes((tmp_path / "c" / "zh.jsonl").read_bytes() + '{"text": "新"}\n'.encode())
    assert M.manifest_doc(spec, tmp_path / "c")["manifest_sha256"] != first


def test_i4_d2_a_tampered_manifest_or_file_is_refused(corpus_writer, tmp_path):
    spec = corpus_writer(tmp_path / "c", n=10)
    path = M.build_manifest(spec, tmp_path / "c", tmp_path / "m.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["files"][0]["n_docs"] += 1
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(M.ManifestError, match="manifest_sha256 does not match"):
        M.load_manifest(path)
    path = M.build_manifest(spec, tmp_path / "c", tmp_path / "m2.json")
    (tmp_path / "c" / "en.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(M.ManifestError, match="sha256 differs"):
        list(M.load_manifest(path).iter_docs())
