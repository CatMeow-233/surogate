"""Step S2: parquet footers and row groups over Range reads, and deterministic plans (local fake Hub, fixtures)."""

import json

import pytest
from corpora_fixtures import code_l2_rows, serve_source, stack3_rows, ufw_rows, unpc_rows
from surogate_corpora import profiles
from surogate_corpora.canonical import CorporaError, json_text
from surogate_corpora.parquet import RemoteParquet, footer_info, iter_rows
from surogate_corpora.plan import FORMAT, build_plan, rank_key
from surogate_corpora.sources import get

ROWS = {
    "stack3-py": lambda n: stack3_rows(n),
    "ufw-en": lambda n: ufw_rows(n, zh=False),
    "ufw-zh": lambda n: ufw_rows(n, zh=True),
    "code-l2-py": code_l2_rows,
    "unpc-en-zh": unpc_rows,
}


def _remote(hub, client, source, index=0):
    files = client.list_files(source.repo, source.revision, source.files)
    return RemoteParquet(client, files[index])


@pytest.mark.parametrize("source_id", sorted(ROWS))
def test_footers_give_exact_counts_and_the_registry_schema(fake_hub, client_for, monkeypatch, source_id):
    source, blobs = serve_source(fake_hub, monkeypatch, source_id, [ROWS[source_id](47)], row_group_size=10)
    opened = _remote(fake_hub, client_for(fake_hub), source)
    footer = opened.footer
    assert footer.num_rows == 47
    assert [rg.num_rows for rg in footer.row_groups] == [10, 10, 10, 10, 7]
    assert dict(footer.schema) == dict(source.schema)  # the fixture writer and the registry agree
    assert opened.bytes_fetched <= len(blobs[0])
    assert footer_info(opened.metadata) == footer


def test_a_nested_column_of_one_row_group_reads_the_right_values_and_less_than_the_file(
    fake_hub, client_for, monkeypatch
):
    rows = stack3_rows(400)
    source, blobs = serve_source(fake_hub, monkeypatch, "stack3-py", [rows], row_group_size=100)
    opened = _remote(fake_hub, client_for(fake_hub), source)
    before = opened.bytes_fetched
    table = opened.read(2, ["source_meta.file_meta.license_type", "source_meta.file_meta.detected_licenses", "uuid"])
    got = list(iter_rows(table, ()))
    want = rows[200:300]
    assert [r["uuid"] for r in got] == [r["uuid"] for r in want]
    assert [r["source_meta.file_meta.license_type"] for r in got] == [
        r["source_meta.file_meta.license_type"] for r in want
    ]
    assert [r["source_meta.file_meta.detected_licenses"] for r in got] == [
        r["source_meta.file_meta.detected_licenses"] for r in want
    ]
    assert set(got[0]) == {"uuid", "source_meta.file_meta.license_type", "source_meta.file_meta.detected_licenses"}
    assert opened.bytes_fetched - before < len(blobs[0]) // 4


def test_text_columns_come_out_as_bytes_and_struct_prefixes_project(fake_hub, client_for, monkeypatch):
    source, _ = serve_source(fake_hub, monkeypatch, "unpc-en-zh", [unpc_rows(30)], row_group_size=10)
    opened = _remote(fake_hub, client_for(fake_hub), source)
    rows = list(iter_rows(opened.read(1, ["translation"]), source.content))
    assert rows[0] == {
        "translation.en": b"SYNTHETIC FIXTURE 10: the committee met.",
        "translation.zh": "SYNTHETIC FIXTURE 10：委员会举行了会议。".encode(),
    }
    with pytest.raises(CorporaError, match="no row group 3"):
        opened.read(3, ["translation"])


# --- plans ----------------------------------------------------------------------------------------------------------


def _stack3_plan_inputs(fake_hub, monkeypatch, n_files=4, rows=120, rg=20):
    files = [stack3_rows(rows, offset=i * rows) for i in range(n_files)]
    source, _ = serve_source(fake_hub, monkeypatch, "stack3-py", files, row_group_size=rg)
    return source, files


def _plan(fake_hub, client_for, *, seed=0, budget=2000, margin=0.05, name="tlc-python-permissive"):
    return build_plan(
        profiles.resolve(name), budget_bytes=budget, seed=seed, client=client_for(fake_hub), margin=margin
    )


def test_the_same_inputs_give_a_byte_identical_plan_and_the_seed_changes_the_order(fake_hub, client_for, monkeypatch):
    _stack3_plan_inputs(fake_hub, monkeypatch)
    a, b = _plan(fake_hub, client_for), _plan(fake_hub, client_for)
    assert json_text(dict(a.doc)) == json_text(dict(b.doc)) and a.sha256 == b.sha256
    assert a.doc["format"] == FORMAT
    other = _plan(fake_hub, client_for, seed=1)
    assert [(u["path"], u["rg"]) for u in other.units] != [(u["path"], u["rg"]) for u in a.units]
    keys = [u["rank_key"] for u in a.units]
    assert keys == sorted(keys)
    source = get("stack3-py")
    assert all(u["rank_key"] == rank_key(0, source.repo, source.revision, u["path"], u["rg"]) for u in a.units)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_first_unit_is_never_the_first_rows_of_the_first_file(fake_hub, client_for, monkeypatch, seed):
    """Pinned for the fixture seed list: rank keys spread the plan over files and row groups."""
    source, _ = _stack3_plan_inputs(fake_hub, monkeypatch)
    plan = _plan(fake_hub, client_for, seed=seed)
    first = plan.units[0]
    assert (first["path"], first["rg"]) != (source.expand_files()[0], 0)
    assert len({u["path"] for u in plan.units}) > 1


def test_the_budget_stops_within_one_unit_and_estimates_use_the_metadata_filters(fake_hub, client_for, monkeypatch):
    _, files = _stack3_plan_inputs(fake_hub, monkeypatch)
    budget, margin = 2000, 0.05
    plan = _plan(fake_hub, client_for, budget=budget, margin=margin)
    estimates = [u["est_kept_bytes"] for u in plan.units]
    target = budget * (1 + margin)
    assert sum(estimates[:-1]) < target <= sum(estimates)
    assert plan.doc["estimate"]["kept_bytes"] == sum(estimates) and plan.doc["estimate"]["budget_reached"]
    # each estimate is the summed size_bytes of the permissive, non-vendored rows of that row group (hand count)
    paths = get("stack3-py").expand_files()
    for unit in plan.units:
        rows = files[paths.index(unit["path"])][unit["rg"] * 20 : (unit["rg"] + 1) * 20]
        hand = sum(
            r["source_meta.file_meta.size_bytes"]
            for r in rows
            if r["source_meta.file_meta.license_type"] == "permissive" and r["source_meta.file_meta.is_vendor"] is False
        )
        assert unit["est_kept_bytes"] == hand
    transfer = sum(
        next(f for f in plan.doc["files"] if f["path"] == u["path"])["row_groups"][u["rg"]]["pass_compressed"]
        for u in plan.units
    )
    assert plan.doc["estimate"]["transfer_bytes"] == transfer > 0


def test_a_plan_that_runs_out_of_units_says_so(fake_hub, client_for, monkeypatch):
    _stack3_plan_inputs(fake_hub, monkeypatch, n_files=1, rows=40)
    plan = _plan(fake_hub, client_for, budget=10**9)
    assert len(plan.units) == plan.doc["estimate"]["units_total"] == 2
    assert not plan.doc["estimate"]["budget_reached"]


def test_a_profile_without_metadata_filters_estimates_the_content_column(fake_hub, client_for, monkeypatch):
    source, _ = serve_source(fake_hub, monkeypatch, "unpc-en-zh", [unpc_rows(40), unpc_rows(40)], row_group_size=10)
    plan = _plan(fake_hub, client_for, budget=500, name="bilingual-parallel-unpc")
    for unit in plan.units:
        entry = next(f for f in plan.doc["files"] if f["path"] == unit["path"])["row_groups"][unit["rg"]]
        assert unit["est_kept_bytes"] == entry["content_uncompressed"] > 0
    # no metadata pass: only the footers were read, one HEAD per file
    assert len(fake_hub.requests("api", "HEAD")) == 2


def test_dry_run_footers_replace_the_footer_reads(fake_hub, client_for, monkeypatch):
    source, _ = serve_source(fake_hub, monkeypatch, "unpc-en-zh", [unpc_rows(40)], row_group_size=10)
    client = client_for(fake_hub)
    footers = {
        f.path: RemoteParquet(client, f).footer for f in client.list_files(source.repo, source.revision, source.files)
    }
    heads = len(fake_hub.requests("api", "HEAD"))
    plan = build_plan(profiles.resolve("bilingual-parallel-unpc"), budget_bytes=300, seed=0, client=client,
                      dry_run_footers=footers)  # fmt: skip
    assert len(fake_hub.requests("api", "HEAD")) == heads and plan.doc["estimate"]["metadata_bytes_fetched"] == 0


def test_a_schema_that_differs_from_the_registry_is_refused(fake_hub, client_for, monkeypatch):
    import dataclasses

    from surogate_corpora import sources

    serve_source(fake_hub, monkeypatch, "ufw-en", [ufw_rows(20, zh=False)], row_group_size=10)
    changed = dataclasses.replace(sources.SOURCES["ufw-en"], schema={**sources.UFW_SCHEMA, "score": "float"})
    monkeypatch.setitem(sources.SOURCES, "ufw-en", changed)
    with pytest.raises(CorporaError, match="schema differs"):
        _plan(fake_hub, client_for, name="bilingual-en")


def test_the_plan_records_the_profile_and_refuses_a_redefined_one(fake_hub, client_for, monkeypatch, tmp_path):
    from surogate_corpora.plan import Plan, load_plan

    _stack3_plan_inputs(fake_hub, monkeypatch, n_files=1, rows=40)
    plan = _plan(fake_hub, client_for)
    path = tmp_path / "plan.json"
    path.write_text(json_text(dict(plan.doc)), encoding="utf-8")
    assert load_plan(path).sha256 == plan.sha256 and load_plan(path).profile().name == "tlc-python-permissive"
    edited = json.loads(path.read_text())
    edited["seed"] = 9
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(CorporaError, match="plan_sha256 does not match"):
        load_plan(path)
    redefined = Plan({**plan.doc, "profile_def": {**plan.doc["profile_def"], "content_filters": []}})
    with pytest.raises(CorporaError, match="defined differently"):
        redefined.profile()
