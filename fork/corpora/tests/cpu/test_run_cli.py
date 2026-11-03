"""Step S5: the resumable fetch loop, the command line (dry runs, refusals) and sample verification."""

import json
import subprocess
import sys

import pytest
from corpora_fixtures import serve_source, stack3_rows, ufw_rows, write_rows_json
from surogate_corpora import export, profiles, run
from surogate_corpora.__main__ import main
from surogate_corpora.canonical import CorporaError, json_text
from surogate_corpora.plan import build_plan
from surogate_corpora.sources import get
from surogate_corpora.verify import verify_export, verify_samples

FILES = [stack3_rows(60), stack3_rows(60, offset=60), stack3_rows(60, offset=120)]


@pytest.fixture
def planned(fake_hub, client_for, monkeypatch):
    serve_source(fake_hub, monkeypatch, "stack3-py", FILES, row_group_size=20)
    client = client_for(fake_hub)
    plan = build_plan(profiles.resolve("tlc-python-permissive"), budget_bytes=10**9, seed=1, client=client)
    return plan, client


def _parts(out):
    return {p.name: p.read_bytes() for p in sorted((out / "parts").iterdir())}


def _recording(monkeypatch):
    reads = []
    original = run._unit

    def recorded(plan, unit, *args):
        reads.append((unit["path"], unit["rg"]))
        return original(plan, unit, *args)

    monkeypatch.setattr(run, "_unit", recorded)
    return reads


@pytest.mark.parametrize("k", [0, 1, 4])
def test_an_interrupted_fetch_resumes_to_the_same_parts_without_rereading_done_units(planned, tmp_path, monkeypatch, k):
    plan, client = planned
    clean = tmp_path / "clean"
    run.fetch(plan, clean, client=client, downloaded="2026-11-03", part_bytes=700)
    reads = _recording(monkeypatch)
    out = tmp_path / "resumed"

    def stop(index):
        if index == k:
            raise KeyboardInterrupt  # what the owner's Ctrl-C does after unit k is recorded

    with pytest.raises(KeyboardInterrupt):
        run.fetch(plan, out, client=client, downloaded="2026-11-03", part_bytes=700, after_unit=stop)
    assert len(reads) == k + 1 and not (out / "provenance.json").exists()
    run.fetch(plan, out, client=client, downloaded="2026-11-03", part_bytes=700)
    assert sorted(reads) == sorted((u["path"], u["rg"]) for u in plan.units) and len(set(reads)) == len(reads)
    assert _parts(out) == _parts(clean)
    totals = [json.loads((d / "provenance.json").read_text())["totals"] for d in (out, clean)]
    for t in totals:  # a resumed run re-opens its files, so only the bytes fetched (footers) differ
        t.pop("bytes_fetched")
    assert totals[0] == totals[1]
    assert (out / "attribution.jsonl").read_bytes() == (clean / "attribution.jsonl").read_bytes()
    assert verify_export(out) == []


def test_a_unit_interrupted_mid_write_is_cut_back_and_redone(planned, tmp_path, monkeypatch):
    plan, client = planned
    clean = tmp_path / "clean"
    run.fetch(plan, clean, client=client, downloaded="2026-11-03", part_bytes=700)
    out = tmp_path / "out"
    original = export.Writer.write
    calls = {"n": 0}

    def failing(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 13:  # inside the second unit (the first keeps fewer rows)
            self.sync()  # what it wrote so far reaches the disk, as it may in a real crash
            raise OSError("disk full")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(export.Writer, "write", failing)
    with pytest.raises(OSError, match="disk full"):
        run.fetch(plan, out, client=client, downloaded="2026-11-03", part_bytes=700)
    done = (out / "done.jsonl").read_text().splitlines()
    assert 1 <= len(done) - 1 < len(plan.units)  # some units done, the interrupted one not
    recorded = sum(size for parts in json.loads(done[-1])["writers"]["stack3-py"].values() for _, size, _ in parts)
    on_disk = sum(p.stat().st_size for p in (out / "parts").iterdir())
    assert on_disk > recorded  # the interrupted unit had written lines that the resume must cut away
    monkeypatch.setattr(export.Writer, "write", original)
    run.fetch(plan, out, client=client, downloaded="2026-11-03", part_bytes=700)
    assert _parts(out) == _parts(clean) and verify_export(out) == []


def test_rerunning_a_finished_export_rewrites_only_its_manifests(planned, tmp_path, monkeypatch):
    plan, client = planned
    out = tmp_path / "out"
    run.fetch(plan, out, client=client, downloaded="2026-11-03")
    before = _parts(out)
    reads = _recording(monkeypatch)
    run.fetch(plan, out, client=client, downloaded="2026-11-04")
    assert reads == [] and _parts(out) == before
    assert json.loads((out / "provenance.json").read_text())["downloaded"] == "2026-11-04"


def test_fetch_refuses_other_directories_and_other_plans(planned, fake_hub, client_for, tmp_path):
    plan, client = planned
    out = tmp_path / "out"
    run.fetch(plan, out, client=client, downloaded="2026-11-03")
    with pytest.raises(CorporaError, match="--no-resume"):
        run.fetch(plan, out, client=client, resume=False)
    other = build_plan(profiles.resolve("tlc-python-all"), budget_bytes=10**9, seed=1, client=client)
    with pytest.raises(CorporaError, match="another plan"):
        run.fetch(other, out, client=client)
    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "notes.txt").write_text("x")
    with pytest.raises(CorporaError, match="holds no done list"):
        run.fetch(plan, stray, client=client)


# --- the command line -----------------------------------------------------------------------------------------------


def test_plan_and_fetch_dry_runs_write_nothing(planned, fake_hub, tmp_path, capsys):
    plan, _ = planned
    work = tmp_path / "work"
    work.mkdir()
    args = ["plan", "--profile", "tlc-python-permissive", "--budget-bytes", "3000", "--seed", "1"]
    assert main([*args, "--dry-run", "--endpoint", fake_hub.endpoint]) == 0
    printed = capsys.readouterr().out
    assert "estimated kept bytes" in printed and "estimated transfer" in printed and "plan_sha256" in printed
    assert list(work.iterdir()) == []
    plan_file = work / "plan.json"
    assert main([*args, "--out", str(plan_file), "--endpoint", fake_hub.endpoint]) == 0
    capsys.readouterr()
    assert main(["fetch", "--plan", str(plan_file), "--out", str(work / "export"), "--dry-run"]) == 0
    assert "estimated transfer" in capsys.readouterr().out
    assert sorted(p.name for p in work.iterdir()) == ["plan.json"]
    assert (
        main(["fetch", "--plan", str(plan_file), "--out", str(work / "export"), "--endpoint", fake_hub.endpoint]) == 0
    )
    assert main(["verify", str(work / "export")]) == 0
    assert "clean" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "reason"),
    [
        (["plan", "--profile", "tlc-python-unlicensed-ablation", "--budget-bytes", "10", "--dry-run"],
         "--allow-unlicensed"),
        (["plan", "--profile", "tlc-python", "--budget-bytes", "10", "--dry-run"], "unknown profile"),
        (["plan", "--profile", "bilingual-en", "--budget-bytes", "10", "--param", "score_min=0.5", "--dry-run"],
         "no parameters"),
        (["plan", "--profile", "bilingual-en", "--budget-bytes", "0", "--dry-run"], "budget must be positive"),
    ],
)  # fmt: skip
def test_plan_refusals_exit_1_with_a_reason(argv, reason, capsys, fake_hub):
    assert main([*argv, "--endpoint", fake_hub.endpoint]) == 1
    err = capsys.readouterr().err
    assert "refused:" in err and reason in err


def test_fetch_and_verify_refusals_exit_1_with_a_reason(planned, tmp_path, capsys):
    plan, _ = planned
    plan_file = tmp_path / "plan.json"
    doc = dict(plan.doc)
    doc["seed"] = 2
    plan_file.write_text(json_text(doc))
    assert main(["fetch", "--plan", str(plan_file), "--out", str(tmp_path / "x")]) == 1
    assert "plan_sha256 does not match" in capsys.readouterr().err
    (tmp_path / "broken").mkdir()
    assert main(["verify", str(tmp_path / "broken")]) == 1
    assert "provenance.json" in capsys.readouterr().out


def test_the_module_entry_point_exits_1_on_a_refusal(tmp_path):
    from conftest import SRC

    code = "import runpy, sys; sys.argv[1:] = sys.argv[2:]; runpy.run_module('surogate_corpora', run_name='__main__')"
    result = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(SRC)!r}); " + code, "x",
         "plan", "--profile", "tlc-python-unlicensed-ablation", "--budget-bytes", "1", "--dry-run"],
        capture_output=True, text=True, timeout=120, cwd=tmp_path,
    )  # fmt: skip
    assert result.returncode == 1 and "--allow-unlicensed" in result.stderr, result.stdout + result.stderr


# --- sample verification --------------------------------------------------------------------------------------------


def test_verify_samples_checks_the_schema_and_exports_the_rows_unchanged(tmp_path, capsys):
    source = get("ufw-zh")
    texts = ["SYNTHETIC FIXTURE 這是繁體的說明。", "SYNTHETIC FIXTURE 这是简体的说明，ＡＢＣ。", "SYNTHETIC FIXTURE x"]
    rows = [{**r, "score": float(r["score"])} for r in ufw_rows(9, zh=True, texts=texts)]  # server view: float32
    rows_json = write_rows_json(tmp_path / "rows.json", source, rows, truncated=[8])
    report = verify_samples(rows_json, "ufw-zh", "bilingual-zh-all", tmp_path / "out")
    assert report["schema"] == "ok" and report["rows"] == 9 and report["truncated_rows"] == 1
    assert report["kept"] == 8 and report["drops"] == {"truncated_cell": 1}
    assert report["bytes_equal"] and report["text_bytes_kept"] == report["text_bytes_out"]
    assert report["hant_dominant_docs"] == 3 and report["nfkc_would_change"] == 3
    assert report["parts"] == {"ufw-zh-tele": 3, "ufw-zh-cci3": 3, "ufw-zh-chinesewebtext": 2}
    assert json.loads((tmp_path / "out" / "sample_report.json").read_text()) == report
    assert "SYNTHETIC" not in json.dumps(report)  # the report keeps counts, never text
    web = verify_samples(rows_json, "ufw-zh", "bilingual-zh-web", tmp_path / "web")
    assert web["kept"] == 5 and web["drops"] == {"source_in": 3, "truncated_cell": 1}


def test_verify_samples_refuses_a_schema_mismatch_and_the_registry_refusals(tmp_path, capsys):
    source = get("ufw-en")
    rows = [{"content": "SYNTHETIC FIXTURE", "score": 0.7, "source": "FineWeb"}]
    bad = write_rows_json(tmp_path / "bad.json", source, rows)
    data = json.loads(bad.read_text())
    del data["rows"][0]["row"]["source"]
    bad.write_text(json.dumps(data))
    with pytest.raises(CorporaError, match=r"columns \['source'\] differ"):
        verify_samples(bad, "ufw-en", "bilingual-en", tmp_path / "o1")
    good = write_rows_json(tmp_path / "good.json", source, rows)
    data = json.loads(good.read_text())
    data["features"][1]["type"]["dtype"] = "string"
    good.write_text(json.dumps(data))
    with pytest.raises(CorporaError, match="features differ"):
        verify_samples(good, "ufw-en", "bilingual-en", tmp_path / "o2")
    code = get("code-l2-py")
    l2 = write_rows_json(tmp_path / "l2.json", code, [{"uuid": "u", "repo_name": "r/r", "relative_path": "a.py",
                         "content": "SYNTHETIC FIXTURE", "category": "TOOL", "algo_rel_score": 0.9,
                         "quality_score": 3.0}])  # fmt: skip
    argv = ["verify-samples", str(l2), "--source", "code-l2-py", "--out", str(tmp_path / "o3")]
    assert main([*argv, "--profile", "tlc-python-permissive"]) == 1
    assert "which source code-l2-py does not have" in capsys.readouterr().err
    assert main([*argv, "--profile", "tlc-python-unlicensed-ablation"]) == 1
    assert "--allow-unlicensed" in capsys.readouterr().err
    assert main([*argv, "--profile", "tlc-python-unlicensed-ablation", "--allow-unlicensed"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["kept"] == 1 and printed["every_doc_has_a_group"] and printed["bytes_equal"]
    argv[3] = "code-l3-py"
    assert main([*argv, "--profile", "tlc-python-all"]) == 1
    assert "model-generated" in capsys.readouterr().err
