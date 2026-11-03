"""Step S1: the source registry, the Hub client and the Range file, against a local http.server (no network)."""

import dataclasses
import random
import re

import pytest
from corpora_fixtures import code_l2_rows, parquet_bytes, small_source
from surogate_corpora import profiles, sources
from surogate_corpora.canonical import CorporaError
from surogate_corpora.hub import HubError, RemoteFile
from surogate_corpora.rangefile import RangeFile

# --- the registry ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "revision", ["main", "refs/convert/parquet", "113743ad", "113743ADAA92066EF6DB2A5B7035F59B4AD6EB30"]
)
def test_the_registry_refuses_a_revision_that_is_not_a_full_sha(revision):
    with pytest.raises(CorporaError, match="full 40-hex commit sha"):
        dataclasses.replace(sources.get("stack3-py"), revision=revision)


def test_every_registry_entry_is_pinned_and_expands_to_the_listed_file_counts():
    counts = {s.id: len(s.expand_files()) for s in sources.SOURCES.values()}
    # file counts read from the Hub API on 2026-11-03
    assert counts == {
        "stack3-py": 45,
        "ufw-en": 2048,
        "ufw-zh": 256,
        "code-l2-py": 119,
        "code-l3-py": 147,
        "unpc-en-zh": 10,
    }
    assert all(re.fullmatch(r"[0-9a-f]{40}", s.revision) for s in sources.SOURCES.values())
    assert sources.get("ufw-zh").expand_files()[0] == "data/ultrafineweb_zh/ultrafineweb-zh-part-001-of-256.parquet"
    assert sources.get("unpc-en-zh").expand_files()[-1] == "en-zh/train/0009.parquet"


def test_registry_patterns_and_columns_are_checked():
    with pytest.raises(CorporaError, match="exactly one"):
        dataclasses.replace(sources.get("ufw-en"), files="data/x.parquet")
    with pytest.raises(CorporaError, match="not in its schema"):
        dataclasses.replace(sources.get("ufw-en"), columns=("content", "url"))
    with pytest.raises(CorporaError, match="unknown source"):
        sources.get("ufw-ja")


@pytest.mark.parametrize("name", sorted(profiles.PROFILES))
def test_every_profile_refuses_the_synthetic_code_l3(name):
    profile = profiles.resolve(name, allow_unlicensed=True)
    with pytest.raises(CorporaError, match="model-generated"):
        profiles.check_source(profile, "code-l3-py")
    assert "code-l3-py" not in profile.sources


def test_licence_ids_are_accepted_by_the_bilingual_manifest():
    from surogate_bilingual.data.manifest import LICENCE_ID

    ids = {i for s in sources.SOURCES.values() for i in s.licence.spdx.values()}
    assert ids == {
        "ODC-By-1.0",
        "Apache-2.0",
        "LicenseRef-Skywork-Community-License",
        "LicenseRef-WanJuan-1.0",
        "LicenseRef-BAAI-Data-Agreement",
        "LicenseRef-MiChao-Usage-Agreement",
        "LicenseRef-UN-Parallel-Corpus-Disclaimer",
    }
    assert all(LICENCE_ID.fullmatch(i) for i in ids)
    assert not any(s.licence.verified for s in sources.SOURCES.values())
    assert set(sources.get("ufw-zh").licence.spdx) == set(sources.UFW_ZH_UPSTREAM)


# --- the Hub client against the fake Hub ---------------------------------------------------------------------------


def _serve(hub, n_files=3, rows=40, row_group_size=10):
    source = small_source("code-l2-py", n_files)
    data = [parquet_bytes(source.schema, code_l2_rows(rows), row_group_size) for _ in range(n_files)]
    for path, blob in zip(source.expand_files(), data):
        hub.files[(source.repo, source.revision, path)] = blob
    return source, data


def test_list_files_follows_pages_and_matches_the_registry(fake_hub, client_for):
    source, data = _serve(fake_hub, n_files=5)
    fake_hub.page_size = 2
    files = client_for(fake_hub).list_files(source.repo, source.revision, source.files)
    assert [f.path for f in files] == source.expand_files()
    assert [f.size for f in files] == [len(d) for d in data]
    assert len(fake_hub.requests("api", "GET")) == 3  # pages of 2, 2, 1


def test_list_files_refuses_a_missing_or_an_extra_file(fake_hub, client_for):
    source, _ = _serve(fake_hub, n_files=3)
    client = client_for(fake_hub)
    narrow = small_source("code-l2-py", 2)
    with pytest.raises(HubError, match="outside the registry's range"):
        client.list_files(narrow.repo, narrow.revision, narrow.files)
    fake_hub.hidden.add(source.expand_files()[1])
    with pytest.raises(HubError, match="1 of 3 files"):
        client.list_files(source.repo, source.revision, source.files)
    with pytest.raises(HubError, match="full 40-hex"):
        client.list_files(source.repo, "main", source.files)


def test_the_token_goes_to_the_api_host_only(fake_hub, client_for):
    source, data = _serve(fake_hub, n_files=1)
    fake_hub.token = "secret-token"
    with pytest.raises(HubError, match="HTTP 401"):
        client_for(fake_hub).list_files(source.repo, source.revision, source.files)
    client = client_for(fake_hub, token="secret-token")
    (remote,) = client.list_files(source.repo, source.revision, source.files)
    resolved = client.resolve(remote)
    assert resolved.url.startswith(f"http://{fake_hub.cdn_host}/") and dict(resolved.headers) == {}
    f = RangeFile(resolved, block=1024)
    assert f.read() == data[0]
    api = [r for r in fake_hub.log if r.host == "api"]
    cdn = [r for r in fake_hub.log if r.host == "cdn"]
    assert api and all(r.headers.get("authorization") == "Bearer secret-token" for r in api[1:])
    assert cdn and not any("authorization" in r.headers for r in cdn)


@pytest.mark.parametrize(
    ("header", "value", "message"),
    [
        ("X-Linked-Size", "12", "size 12 at resolve differs"),
        ("X-Linked-ETag", '"' + "0" * 64 + '"', "ETag 0+ at resolve differs"),
        ("X-Xet-Hash", "f" * 64, "xet hash f+ at resolve differs"),
        ("X-Repo-Commit", "1" * 40, "answered for commit 1+"),
    ],
)
def test_resolve_refuses_a_file_that_differs_from_the_listing(fake_hub, client_for, header, value, message):
    source, _ = _serve(fake_hub, n_files=1)
    client = client_for(fake_hub)
    (remote,) = client.list_files(source.repo, source.revision, source.files)
    fake_hub.header_overrides[remote.path] = {header: value}
    with pytest.raises(HubError, match=message):
        client.resolve(remote)


# --- the Range file -------------------------------------------------------------------------------------------------


def _opened(fake_hub, client_for, **kwargs):
    source, data = _serve(fake_hub, n_files=1, rows=200, row_group_size=50)
    client = client_for(fake_hub)
    (remote,) = client.list_files(source.repo, source.revision, source.files)
    return client, remote, data[0], RangeFile(client.resolve(remote), refresh=lambda: client.resolve(remote), **kwargs)


def test_arbitrary_slices_equal_the_file_bytes(fake_hub, client_for):
    _, _, data, f = _opened(fake_hub, client_for, block=1000, cache_blocks=3)
    rng = random.Random(7)
    for _ in range(200):
        start = rng.randrange(len(data))
        n = rng.choice([0, 1, 7, 999, 1000, 1001, 4096, rng.randrange(len(data))])
        f.seek(start)
        assert f.read(n) == data[start : start + n]
    f.seek(-10, 2)
    assert f.read() == data[-10:] and f.read(5) == b""
    f.seek(0)
    assert f.read() == data


def test_bytes_fetched_counts_whole_blocks_and_the_cache_saves_refetches(fake_hub, client_for):
    _, _, data, f = _opened(fake_hub, client_for, block=1024, cache_blocks=4)
    size = len(data)
    assert size > 8 * 1024
    f.seek(1500)
    f.read(100)  # block 1
    assert (f.bytes_fetched, f.requests) == (1024, 1)
    f.seek(1024)
    f.read(1024)  # block 1 again: cached
    assert (f.bytes_fetched, f.requests) == (1024, 1)
    f.seek(1000)
    f.read(3000)  # blocks 0..3: 0, 2, 3 missing; 0 alone, then 2-3 in one request
    assert (f.bytes_fetched, f.requests) == (4 * 1024, 3)
    f.seek(size - 5)
    f.read(100)  # the short last block
    last = size - (size // 1024) * 1024
    assert f.bytes_fetched == 4 * 1024 + last
    ranges = [r.headers["range"] for r in fake_hub.requests("cdn")]
    assert ranges == ["bytes=1024-2047", "bytes=0-1023", "bytes=2048-4095", f"bytes={size - last}-{size - 1}"]


def test_large_reads_are_split_into_bounded_requests(fake_hub, client_for):
    _, _, data, f = _opened(fake_hub, client_for, block=1024, max_request=4096)
    assert f.read(10 * 1024) == data[: 10 * 1024]
    assert f.requests == 3 and f.bytes_fetched == 10 * 1024  # 4 + 4 + 2 blocks


def test_a_403_on_the_cdn_re_resolves_once(fake_hub, client_for):
    client, remote, data, f = _opened(fake_hub, client_for, block=1024)
    fake_hub.forbid_signatures.add(int(f.url.url.rpartition("sig=")[2]))  # the URL f holds now
    resolves = len(fake_hub.requests("api", "HEAD"))
    assert f.read(2048) == data[:2048]
    assert f.refreshes == 1 and len(fake_hub.requests("api", "HEAD")) == resolves + 1
    fake_hub.forbid_all = True
    f.seek(5000)
    with pytest.raises(CorporaError, match="HTTP 403"):
        f.read(10)
    assert f.refreshes == 2  # one re-resolve for the failing read, then the refusal


def test_an_expiring_url_is_re_resolved_before_the_request(fake_hub, client_for):
    fake_hub.expires = 1000
    _, _, data, f = _opened(fake_hub, client_for, block=1024, clock=lambda: 990.0)
    assert f.url.expires == 1000.0
    assert f.read(10) == data[:10] and f.refreshes == 1


def test_a_server_that_ignores_range_is_refused_before_the_body_is_read(fake_hub, client_for):
    _, _, _, f = _opened(fake_hub, client_for, block=1024)
    fake_hub.ignore_range = True
    with pytest.raises(CorporaError, match="ignored the Range header"):
        f.read(10)
    assert f.bytes_fetched == 0


def test_remote_file_round_trips_through_json():
    f = RemoteFile("o/r", "a" * 40, "x/y.parquet", 10, "b" * 64, None)
    assert RemoteFile.from_json(f.to_json()) == f
