"""Step S4: exporters and provenance. Fixture parquet -> fetch (fake Hub) -> export, read back with the projects' own
loaders (surogate_tlc.data.corpus, surogate_bilingual.data.manifest and data.parallel, from this checkout)."""

import json
import unicodedata

import pytest
from corpora_fixtures import MARK, UFW_ZH_SOURCES, serve_source, stack3_rows, ufw_rows, unpc_rows
from surogate_corpora import profiles
from surogate_corpora.plan import build_plan
from surogate_corpora.run import fetch
from surogate_corpora.verify import verify_export

LT, VENDOR, SPDX = (
    "source_meta.file_meta.license_type",
    "source_meta.file_meta.is_vendor",
    "source_meta.file_meta.detected_licenses",
)


def _export(hub, client_for, monkeypatch, out, source_id, profile, files_rows, rg=20, budget=10**9, **kwargs):
    source, _ = serve_source(hub, monkeypatch, source_id, files_rows, row_group_size=rg)
    client = client_for(hub)
    plan = build_plan(profiles.resolve(profile, kwargs.pop("params", None)), budget_bytes=budget, seed=3, client=client)
    doc = fetch(plan, out, client=client, downloaded="2026-11-03", **kwargs)
    return plan, doc, source


def _kept(plan, source, files_rows, rg, keep):
    """The fixture rows the plan's units keep, in export order (units in rank order, rows in file order)."""
    paths = source.expand_files()
    out = []
    for unit in plan.units:
        rows = files_rows[paths.index(unit["path"])][unit["rg"] * rg : (unit["rg"] + 1) * rg]
        out += [r for r in rows if keep(r)]
    return out


def _lines(out, path):
    return [json.loads(line) for line in (out / path).read_bytes().splitlines()]


def test_tlc_code_export_loads_with_the_tlc_corpus_reader(fake_hub, client_for, monkeypatch, tmp_path):
    from surogate_tlc.data import corpus as tlc

    files = [stack3_rows(60), stack3_rows(60, offset=60)]
    out = tmp_path / "export"
    plan, doc, source = _export(fake_hub, client_for, monkeypatch, out, "stack3-py", "tlc-python-permissive", files)
    kept = _kept(plan, source, files, 20, lambda r: r[LT] == "permissive" and r[VENDOR] is False)
    assert doc["totals"]["docs"] == len(kept) > 0
    spec = out / "spec.toml"
    spec.write_text("seed = 5\nval_fraction = 0.3\n\n" + (out / "spec_fragment.toml").read_text(), encoding="utf-8")
    corpus = tlc.load_corpus(spec)
    (entry,) = corpus.sources
    assert entry.kind == "code" and entry.provenance["revision"] == source.revision
    assert entry.provenance["license"] == "ODC-By-1.0 (UltraData-Stack-v3-L2) + per-file SPDX (permissive only)"
    assert entry.provenance["filters"][-1] == f"corpora export {plan.sha256}"
    docs = list(tlc.documents(corpus))
    assert [d.text for d in docs] == [r["content"] for r in kept]
    assert [d.group for d in docs] == [f"group:{r['repo_name']}" for r in kept]
    by_repo = {}
    for d in docs:
        by_repo.setdefault(d.group, set()).add(tlc.split_of(corpus, d))
    assert all(len(splits) == 1 for splits in by_repo.values())  # a repository never straddles the split
    attribution = [json.loads(line) for line in (out / "attribution.jsonl").read_text().splitlines()]
    assert [a["repo"] for a in attribution] == sorted({r["repo_name"] for r in kept})
    for a in attribution:
        rows = [r for r in kept if r["repo_name"] == a["repo"]]
        assert a["spdx"] == sorted({s for r in rows for s in r[SPDX]})
        assert a["commit_ids"] == sorted({r["source_meta.commit_id"] for r in rows})
        assert a["files"] == len(rows) and a["bytes"] == sum(len(r["content"].encode()) for r in rows)
    assert doc["attribution"]["sha256"] and verify_export(out) == []


def test_tlc_english_export_has_no_group(fake_hub, client_for, monkeypatch, tmp_path):
    from surogate_tlc.data import corpus as tlc

    files = [ufw_rows(30, zh=False)]
    out = tmp_path / "en"
    plan, doc, source = _export(fake_hub, client_for, monkeypatch, out, "ufw-en", "tlc-english", files, rg=10)
    lines = _lines(out, doc["parts"][0]["path"])
    assert all("group" not in line and set(line) == {"id", "meta", "text"} for line in lines)
    spec = out / "spec.toml"
    spec.write_text("seed = 0\nval_fraction = 0.1\n" + (out / "spec_fragment.toml").read_text(), encoding="utf-8")
    docs = list(tlc.documents(tlc.load_corpus(spec)))
    assert sorted(d.text for d in docs) == sorted(r["content"] for r in files[0])
    assert all(d.group == d.key for d in docs) and not (out / "attribution.jsonl").exists()


def test_bilingual_zh_export_builds_a_manifest_with_one_unverified_source_per_upstream(
    fake_hub, client_for, monkeypatch, tmp_path
):
    from surogate_bilingual.data.manifest import build_manifest, load_manifest

    files = [ufw_rows(48, zh=True), ufw_rows(48, zh=True, offset=48)]
    out = tmp_path / "zh"
    plan, doc, source = _export(fake_hub, client_for, monkeypatch, out, "ufw-zh", "bilingual-zh-all", files, rg=16)
    manifest = load_manifest(build_manifest(out / "spec_fragment.toml", out, tmp_path / "manifest.json"))
    names = {f"ufw-zh-{s.lower()}" for s in UFW_ZH_SOURCES}
    assert set(manifest.sources) == names
    assert all(not s["licence_verified"] for s in manifest.sources.values())
    assert manifest.doc["unverified_licences"] == sorted(names)
    assert manifest.sources["ufw-zh-tele"]["licence"] == "Apache-2.0"
    assert manifest.sources["ufw-zh-wudao"]["licence"] == "LicenseRef-BAAI-Data-Agreement"
    docs = list(manifest.iter_docs())
    kept = _kept(plan, source, files, 16, lambda r: True)
    assert sorted(d.text for d in docs) == sorted(r["content"] for r in kept)
    assert all(d.kind == "zh" and d.meta["script"]["hant"] == 0 for d in docs)
    for d in docs:  # each document sits in its own upstream source's part series
        assert d.source == f"ufw-zh-{d.meta['upstream_source'].lower()}"
    lines = _lines(out, doc["parts"][0]["path"])
    assert all("group" not in line for line in lines)


def test_the_web_pool_keeps_only_its_three_sources(fake_hub, client_for, monkeypatch, tmp_path):
    files = [ufw_rows(48, zh=True)]
    out = tmp_path / "web"
    plan, doc, source = _export(fake_hub, client_for, monkeypatch, out, "ufw-zh", "bilingual-zh-web", files, rg=16)
    kept = [r for r in files[0] if r["source"] in profiles.WEB_POOL]
    assert doc["totals"]["docs"] == len(kept) and doc["totals"]["drops"] == {"source_in": 48 - len(kept)}
    assert {p["path"].split("/")[1].rsplit("-", 1)[0] for p in doc["parts"]} == {
        "ufw-zh-cci3", "ufw-zh-chinesewebtext", "ufw-zh-industrycorpus"
    }  # fmt: skip


def test_parallel_export_is_parsed_by_the_bilingual_bridge_layout(fake_hub, client_for, monkeypatch, tmp_path):
    from surogate_bilingual.data import parallel
    from surogate_bilingual.data.manifest import build_manifest, load_manifest

    files = [unpc_rows(30), unpc_rows(30)]
    out = tmp_path / "par"
    _export(fake_hub, client_for, monkeypatch, out, "unpc-en-zh", "bilingual-parallel-unpc", files, rg=10)
    manifest = load_manifest(build_manifest(out / "spec_fragment.toml", out, tmp_path / "m.json"))
    assert manifest.sources["unpc-en-zh"]["licence"] == "LicenseRef-UN-Parallel-Corpus-Disclaimer"
    docs = list(manifest.iter_docs())
    assert len(docs) == 60 and len({d.key for d in docs}) == 60  # the ids name the file, so they never collide
    assert all(d.kind == "parallel" and d.pair_id and parallel.direction(d) == "en>zh" for d in docs)
    laid = parallel.bridge_layout(docs[0])
    assert [s.route for s in laid.segments] == ["en", "bridge"] and laid.segments[1].text.startswith(MARK)


def test_two_runs_give_byte_identical_parts_and_verify_catches_one_changed_byte(
    fake_hub, client_for, monkeypatch, tmp_path
):
    files = [stack3_rows(60), stack3_rows(60, offset=60)]
    a = tmp_path / "a"
    plan, doc, source = _export(fake_hub, client_for, monkeypatch, a, "stack3-py", "tlc-python-all", files,
                                part_bytes=600)  # fmt: skip
    b = tmp_path / "b"
    fetch(plan, b, client=client_for(fake_hub), downloaded="2026-11-03", part_bytes=600)
    parts = sorted(p.relative_to(a) for p in (a / "parts").iterdir())
    assert len(parts) > 2  # the small part size rolled over
    assert parts == sorted(p.relative_to(b) for p in (b / "parts").iterdir())
    assert all((a / p).read_bytes() == (b / p).read_bytes() for p in parts)
    assert all(p["bytes"] <= 600 for p in doc["parts"])
    assert verify_export(a) == [] and verify_export(b) == []
    target = a / parts[1]
    data = bytearray(target.read_bytes())
    data[5] ^= 0x01
    target.write_bytes(bytes(data))
    problems = verify_export(a)
    assert len(problems) == 1 and problems[0].startswith(f"{parts[1].as_posix()}: sha256 ")
    (a / "parts" / "stray-00000.jsonl").write_text("{}\n")
    assert "parts/stray-00000.jsonl: on disk but not listed in provenance.json" in verify_export(a)


TRICKY = [
    f"{MARK} crlf\r\nline two\r\n",
    f"{MARK} lone\rcarriage return",
    f"{MARK}\ttab\t and trailing spaces   ",
    f"{MARK} line separator and paragraph separator",
    f"{MARK} full-width ＡＢＣ１２３ and ligature ﬁ and ①",  # NFKC would change these
    f"{MARK} emoji \U0001f600 and CJK Extension B \U00020000\U0002a6d6",
    f"{MARK} combining é vs é, and a NUL \x00 inside",
    f'{MARK} json specials " \\ / \b \f',
]


def test_exported_text_bytes_equal_the_input_bytes(fake_hub, client_for, monkeypatch, tmp_path):
    files = [ufw_rows(len(TRICKY), zh=False, texts=TRICKY)]
    out = tmp_path / "tricky"
    _, doc, _ = _export(fake_hub, client_for, monkeypatch, out, "ufw-en", "bilingual-en", files, rg=4)
    texts = [line["text"] for p in doc["parts"] for line in _lines(out, p["path"])]
    assert sorted(t.encode() for t in texts) == sorted(t.encode() for t in TRICKY)
    assert sum(unicodedata.normalize("NFKC", t) != t for t in texts) >= 2  # the export normalised nothing
    assert doc["totals"]["text_bytes"] == sum(len(t.encode()) for t in TRICKY)


def test_the_provenance_records_units_parts_and_tools(fake_hub, client_for, monkeypatch, tmp_path):
    files = [stack3_rows(40)]
    out = tmp_path / "prov"
    plan, doc, source = _export(fake_hub, client_for, monkeypatch, out, "stack3-py", "tlc-python-permissive", files)
    assert doc["format"] == "corpora-export/1" and doc["plan_sha256"] == plan.sha256
    assert doc["profile"] == plan.doc["profile_def"] and set(doc["tools"]) == {"python", "pyarrow", "surogate_corpora"}
    assert set(doc["code"]) == {"commit", "dirty"}
    unit = doc["units"][0]
    assert unit["etag"] == plan.file(unit["path"])["lfs_sha256"] and unit["xet_hash"]
    assert unit["rows_read"] == 20 and unit["rows_read"] == unit["rows_kept"] + sum(unit["drops"].values())
    assert doc["totals"]["rows_kept"] == sum(u["rows_kept"] for u in doc["units"]) == doc["totals"]["docs"]
    assert doc["totals"]["drops"] == {
        "licence_permissive": sum(r[LT] != "permissive" for r in files[0]),
        "not_vendor": sum(r[LT] == "permissive" and r[VENDOR] for r in files[0]),
    }


@pytest.mark.parametrize("bad", ["provenance", "fragment"])
def test_verify_reports_a_damaged_manifest(fake_hub, client_for, monkeypatch, tmp_path, bad):
    files = [ufw_rows(20, zh=False)]
    out = tmp_path / "v"
    _export(fake_hub, client_for, monkeypatch, out, "ufw-en", "bilingual-en", files, rg=10)
    if bad == "provenance":
        doc = json.loads((out / "provenance.json").read_text())
        doc["totals"]["docs"] += 1
        (out / "provenance.json").write_text(json.dumps(doc))
        assert verify_export(out) == ["provenance.json: provenance_sha256 does not match the document's content "
                                      "(edited by hand?)"]  # fmt: skip
    else:
        (out / "spec_fragment.toml").write_text("# emptied\n")
        assert verify_export(out) == [
            "spec_fragment.toml: sha256 differs from provenance.json",
            "spec_fragment.toml: its files are not exactly the parts provenance.json lists",
        ]
