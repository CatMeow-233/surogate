"""Run registry and environment fingerprint (docs/spec.md, I5-T14; IMPL-21)."""

import json

import pytest

from surogate_bilingual.__main__ import main as cli
from surogate_bilingual.registry import env
from surogate_bilingual.registry import run as registry
from surogate_bilingual.train.loop import train

IDENTITY = {
    "config_hash": "c" * 64,
    "data_manifest_sha256": "m" * 64,
    "tokenizer_sha256": "t" * 64,
    "seed_fingerprint": None,
    "recipe": "seed",
    "init_seed": 1,
    "data_seed": 2,
}


def test_i5_t14_run_id_is_deterministic_and_covers_every_identity_field():
    ident = {**IDENTITY, "code_commit": "abc"}
    assert registry.run_id(ident) == registry.run_id(dict(reversed(list(ident.items()))))
    assert len(registry.run_id(ident)) == 16 and int(registry.run_id(ident), 16) >= 0
    ids = {registry.run_id(ident)}
    for key in registry.IDENTITY_KEYS:
        ids.add(registry.run_id({**ident, key: "changed"}))
    assert len(ids) == 1 + len(registry.IDENTITY_KEYS)
    with pytest.raises(registry.RegistryError, match="lacks"):
        registry.run_id({k: v for k, v in ident.items() if k != "data_seed"})


def test_i5_t14_a_name_is_bound_to_one_run_id(tmp_path):
    clean = ("abc", False)
    run_dir, doc, resumed = registry.open_run(
        tmp_path, "r0", IDENTITY, {"matrix_row": "B0"}, allow_dirty=False, code=clean
    )
    assert not resumed and doc["status"] == "created" and doc["code_commit"] == "abc"
    assert json.loads((tmp_path / "index.json").read_text()) == {"r0": doc["run_id"]}
    assert registry.open_run(tmp_path, "r0", IDENTITY, {}, allow_dirty=False, code=clean)[2]  # same id: resume
    with pytest.raises(registry.RegistryError, match="bound to run_id"):
        registry.open_run(tmp_path, "r0", {**IDENTITY, "init_seed": 9}, {}, allow_dirty=False, code=clean)
    with pytest.raises(registry.RegistryError, match="bound to run_id"):  # a new commit is a new run
        registry.open_run(tmp_path, "r0", IDENTITY, {}, allow_dirty=False, code=("def", False))
    with pytest.raises(registry.RegistryError, match="run names"):
        registry.open_run(tmp_path, "../escape", IDENTITY, {}, allow_dirty=False, code=clean)


def test_a_run_name_with_the_temporary_file_mark_is_refused(tmp_path, tiny_stage):
    """CR1 review probe: a run named like a crash leftover ("*.tmp-*") was deleted by the next open_run."""
    clean = ("abc", False)
    for name in ("seed-I-r0.tmp-lr6e-4", "x.tmp-1"):
        with pytest.raises(registry.RegistryError, match="temporary files"):
            registry.open_run(tmp_path, name, IDENTITY, {}, allow_dirty=False, code=clean)
        with pytest.raises(ValueError, match="temporary files"):
            tiny_stage("tiny-seed", f'run.name="{name}"')
    kept, _, _ = registry.open_run(tmp_path, "seed-I-r0-lr6e-4", IDENTITY, {}, allow_dirty=False, code=clean)
    (kept / "state").mkdir()
    registry.open_run(tmp_path, "zh-I-r0", {**IDENTITY, "init_seed": 5}, {}, allow_dirty=False, code=clean)
    assert (kept / "state").is_dir() and sorted(p.name for p in tmp_path.iterdir()) == [
        "index.json",
        "seed-I-r0-lr6e-4",
        "zh-I-r0",
    ]


def test_i5_t14_a_dirty_tree_is_refused_unless_allowed_and_then_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "code_identity", lambda: ("abc", True))
    with pytest.raises(registry.RegistryError, match="code_dirty"):
        registry.open_run(tmp_path, "r0", IDENTITY, {}, allow_dirty=False)
    assert not (tmp_path / "r0").exists() and not (tmp_path / "index.json").exists()
    run_dir, doc, _ = registry.open_run(tmp_path, "r0", IDENTITY, {}, allow_dirty=True)
    stored = registry.read_run(run_dir)
    assert stored["code_dirty"] is True and stored["allow_dirty"] is True and stored == doc


def test_i5_t14_the_loop_refuses_a_dirty_tree_and_a_rebound_name(tiny_trained, tiny_stage, tmp_path, monkeypatch):
    root, runs = tiny_trained["root"], tmp_path / "runs"
    monkeypatch.setattr(env, "code_identity", lambda: ("abc", True))
    with pytest.raises(registry.RegistryError, match="code_dirty"):
        train(tiny_stage("tiny-zh", "registry.allow_dirty=false", "run.steps=2"), root=root, runs=runs)
    result = train(tiny_stage("tiny-zh", "run.steps=2"), root=root, runs=runs)
    doc = registry.read_run(result.run_dir)
    assert doc["code_dirty"] is True and doc["allow_dirty"] is True and doc["status"] == "done"
    with pytest.raises(registry.RegistryError, match="bound to run_id"):
        train(tiny_stage("tiny-zh", "run.steps=3"), root=root, runs=runs)  # same name, other config


def test_i5_t14_environment_json_has_every_required_key(tiny_trained):
    for result in tiny_trained["results"].values():
        doc = json.loads((result.run_dir / "environment.json").read_text("utf-8"))
        assert set(env.REQUIRED_KEYS) <= set(doc) and doc["format"] == "bilingual-env/1"
        assert doc["threads"]["torch"] >= 1 and doc["torch"]
    fresh = env.environment()
    assert set(env.REQUIRED_KEYS) <= set(fresh)
    if fresh["gpu"] is None:  # a CPU host: the CUDA fields are null, never guessed
        assert fresh["cuda"] is None and fresh["cudnn"] is None


def test_env_and_registry_commands(tiny_trained, tmp_path, capsys):
    assert cli(["env", "--out", str(tmp_path / "e.json"), "--dry-run"]) == 0
    assert not (tmp_path / "e.json").exists() and '"format": "bilingual-env/1"' in capsys.readouterr().out
    assert cli(["env", "--out", str(tmp_path / "e.json")]) == 0 and (tmp_path / "e.json").is_file()
    assert cli(["registry", "list", "--runs", str(tiny_trained["runs"])]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in ("tiny-seed", "tiny-zh", "tiny-en"))
    assert cli(["registry", "show", "tiny-zh", "--runs", str(tiny_trained["runs"])]) == 0
    assert json.loads(capsys.readouterr().out)["recipe"] == "expert"
    assert cli(["registry", "show", "nope", "--runs", str(tiny_trained["runs"])]) == 1


def test_registry_budget_checks_token_matching_and_totals_tuning(tmp_path, capsys):
    """CR1 review: matched-token claims (B2 = ZH + EN, B12a = B2, the T arms) are checked from the registry, and
    tuning trials (run.trial) are tabulated with their compute."""
    clean = ("abc", False)
    runs = [
        ("zh-I-r0", "B1", 0, None, 100, 0.5),
        ("en-I-r0", "B1", 0, None, 100, 0.5),
        ("b2-I-r0", "B2", 0, None, 200, 1.0),
        ("b12a-I-r0", "B12a", 0, None, 150, 1.5),
        ("seed-I-r0-lr3e-4", "B0", 0, "lr-3e-4", 300, 2.0),
    ]
    for i, (name, row, rep, trial, tokens, hours) in enumerate(runs):
        run_dir, _, _ = registry.open_run(
            tmp_path,
            name,
            {**IDENTITY, "init_seed": i},
            {"matrix_row": row, "replicate": rep, "trial": trial, "p_update": 7},
            allow_dirty=False,
            code=clean,
        )
        registry.update_run(
            run_dir, status="done", finished_tokens=tokens, compute={"device_seconds": hours * 3600, "gpu_hours": hours}
        )
    table = registry.budget_table(tmp_path)
    by = {(r["row"], r["replicate"]): r for r in table["rows"]}
    assert by[("B1", 0)]["tokens"] == 200 and by[("B1", 0)]["gpu_hours"] == 1.0 and len(by[("B1", 0)]["runs"]) == 2
    status = {(tuple(c["rows"]), c["replicate"]): c["status"] for c in table["token_matches"]}
    assert status[(("B2", "B1"), 0)] == "matched" and status[(("B12a", "B2"), 0)] == "mismatch"
    assert ("B0", 0) not in by and table["tuning"]["gpu_hours"] == 2.0
    assert table["tuning"]["trials"][0]["trial"] == "lr-3e-4"
    assert cli(["registry", "budget", "--runs", str(tmp_path)]) == 0
    assert '"mismatch"' in capsys.readouterr().out
