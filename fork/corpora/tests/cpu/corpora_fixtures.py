"""Test helpers: a local stand-in for the Hugging Face Hub, and SYNTHETIC FIXTURE parquet files in the registry's
schemas. Nothing here is real data; every text says SYNTHETIC FIXTURE.

FakeHub runs two http.server instances in daemon threads, on two ports of 127.0.0.1 (so they are two hosts to the
client): the API host serves the tree listing (paginated with a Link header) and the HEAD resolve, which redirects
(302) to a signed URL on the CDN host; the CDN host serves Range GETs. Knobs inject the failures the client must
handle, and every request is logged with its headers.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MARK = "SYNTHETIC FIXTURE"


# --- the fake Hub ---------------------------------------------------------------------------------------------------


@dataclass
class Logged:
    host: str  # "api" | "cdn"
    method: str
    path: str
    headers: dict[str, str]


@dataclass
class FakeHub:
    files: dict[tuple[str, str, str], bytes]  # (repo, revision, path) -> bytes
    token: str | None = None  # the API host answers 401 without it
    page_size: int = 1000
    expires: int = 4102444800  # 2100-01-01: far in the future unless a test sets it
    forbid_signatures: set = field(default_factory=set)  # CDN answers 403 for these signature numbers
    forbid_all: bool = False
    ignore_range: bool = False  # CDN answers 200 with the whole file
    header_overrides: dict[str, dict[str, str]] = field(default_factory=dict)  # path -> resolve headers to replace
    hidden: set = field(default_factory=set)  # paths left out of the listing
    log: list[Logged] = field(default_factory=list)
    signature: int = 0

    def start(self) -> FakeHub:
        self.api = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _handler(self, "api"))
        self.cdn = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _handler(self, "cdn"))
        for server in (self.api, self.cdn):
            threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        return self

    def stop(self) -> None:
        for server in (self.api, self.cdn):
            server.shutdown()
            server.server_close()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.api.server_address[1]}"

    @property
    def cdn_host(self) -> str:
        return f"127.0.0.1:{self.cdn.server_address[1]}"

    def blob(self, sha: str) -> bytes:
        return next(data for data in self.files.values() if hashlib.sha256(data).hexdigest() == sha)

    def requests(self, host: str, method: str | None = None) -> list[Logged]:
        return [r for r in self.log if r.host == host and (method is None or r.method == method)]


def _handler(hub: FakeHub, host: str):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, status: int, headers: Mapping[str, str] = (), body: bytes = b"") -> None:
            self.send_response(status)
            for k, v in dict(headers).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _route(self) -> None:
            hub.log.append(Logged(host, self.command, self.path, {k.lower(): v for k, v in self.headers.items()}))
            url = urllib.parse.urlsplit(self.path)
            if host == "api":
                if hub.token and self.headers.get("Authorization") != f"Bearer {hub.token}":
                    return self._send(401)
                parts = url.path.split("/")
                if url.path.startswith("/api/datasets/") and parts[5] == "tree" and self.command == "GET":
                    return self._tree(parts, url)
                if url.path.startswith("/datasets/") and parts[4] == "resolve" and self.command == "HEAD":
                    return self._resolve(parts)
                return self._send(404)
            return self._cdn(url)

        def _tree(self, parts, url) -> None:
            repo, revision = f"{parts[3]}/{parts[4]}", parts[6]
            directory = urllib.parse.unquote("/".join(parts[7:]))
            entries = []
            for (r, rev, path), data in sorted(hub.files.items()):
                if (r, rev) == (repo, revision) and path.rpartition("/")[0] == directory and path not in hub.hidden:
                    sha = hashlib.sha256(data).hexdigest()
                    entries.append(
                        {
                            "type": "file",
                            "path": path,
                            "size": len(data),
                            "oid": sha[:40],
                            "lfs": {"oid": sha, "size": len(data), "pointerSize": 130},
                            "xetHash": "x" + sha[1:],
                        }
                    )
            cursor = int(urllib.parse.parse_qs(url.query).get("cursor", ["0"])[0])
            page = entries[cursor : cursor + hub.page_size]
            headers = {"Content-Type": "application/json"}
            if cursor + hub.page_size < len(entries):
                headers["Link"] = f'<{url.path}?cursor={cursor + hub.page_size}>; rel="next"'
            self._send(200, headers, json.dumps(page).encode())

        def _resolve(self, parts) -> None:
            repo, revision = f"{parts[2]}/{parts[3]}", parts[5]
            path = urllib.parse.unquote("/".join(parts[6:]))
            data = hub.files.get((repo, revision, path))
            if data is None:
                return self._send(404)
            sha = hashlib.sha256(data).hexdigest()
            hub.signature += 1
            headers = {
                "Location": f"http://{hub.cdn_host}/cdn/{sha}?Expires={hub.expires}&sig={hub.signature}",
                "X-Linked-Size": str(len(data)),
                "X-Linked-ETag": f'"{sha}"',
                "X-Xet-Hash": "x" + sha[1:],
                "X-Repo-Commit": revision,
            }
            headers.update(hub.header_overrides.get(path, {}))
            self._send(302, headers)

        def _cdn(self, url) -> None:
            query = urllib.parse.parse_qs(url.query)
            if hub.forbid_all or int(query["sig"][0]) in hub.forbid_signatures:
                return self._send(403)
            data = hub.blob(url.path.rsplit("/", 1)[1])
            spec = self.headers.get("Range")
            if hub.ignore_range or not spec:
                return self._send(200, {}, data)
            start, end = (int(x) for x in spec.removeprefix("bytes=").split("-"))
            end = min(end, len(data) - 1)
            self._send(206, {"Content-Range": f"bytes {start}-{end}/{len(data)}"}, data[start : end + 1])

        do_GET = do_HEAD = _route

    return Handler


# --- parquet fixtures in the registry schemas ----------------------------------------------------------------------

ARROW = {"string": "string", "float": "float32", "double": "float64", "int64": "int64", "bool": "bool_"}


def arrow_schema(schema: Mapping[str, str]):
    import pyarrow as pa

    tree: dict[str, Any] = {}
    for leaf, kind in schema.items():
        node = tree
        *parents, name = leaf.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[name] = kind

    def to_type(node):
        if isinstance(node, str):
            return pa.list_(pa.string()) if node == "list<string>" else getattr(pa, ARROW[node])()
        return pa.struct([pa.field(k, to_type(v)) for k, v in node.items()])

    return pa.schema([pa.field(k, to_type(v)) for k, v in tree.items()])


def nest(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for leaf, value in row.items():
        node = out
        *parents, name = leaf.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[name] = value
    return out


def parquet_bytes(schema: Mapping[str, str], rows: Sequence[Mapping[str, Any]], row_group_size: int) -> bytes:
    """A parquet file of `rows` (flat dotted dicts) in `schema`, SNAPPY, `row_group_size` rows per row group."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([nest(r) for r in rows], schema=arrow_schema(schema))
    buffer = io.BytesIO()
    pq.write_table(table, buffer, row_group_size=row_group_size, compression="snappy")
    return buffer.getvalue()


UFW_ZH_SOURCES = ("Tele", "CCI3", "ChineseWebText", "IndustryCorpus", "SkyPile", "WanJuan", "WuDao", "MiChao")


def stack3_rows(n: int, offset: int = 0) -> list[dict[str, Any]]:
    """Stack-v3 rows: every third file no_license, every fifth vendored, seven repositories."""
    rows = []
    for i in range(offset, offset + n):
        content = f"# {MARK} {i}\ndef f{i}(x):\n    return x + {i}\n"
        rows.append(
            {
                "uuid": f"fixture-{i:06d}",
                "repo_name": f"fixture-org/repo{i % 7}",
                "relative_path": f"pkg/mod{i}.py",
                "content": content,
                "category": "ALGO",
                "algo_rel_score": 0.95,
                "quality_score": 4.0,
                "source_meta.github_metadata.branch": "main",
                "source_meta.github_metadata.commit_count": i,
                "source_meta.github_metadata.repo_created_at": "2020-01-01",
                "source_meta.github_metadata.is_fork": False,
                "source_meta.github_metadata.is_org_owned": True,
                "source_meta.github_metadata.forked_from": None,
                "source_meta.github_metadata.stars": 1,
                "source_meta.github_metadata.forks": 0,
                "source_meta.github_metadata.issues": 0,
                "source_meta.github_metadata.pull_requests": 0,
                "source_meta.file_meta.content_id": f"cid{i}",
                "source_meta.file_meta.size_bytes": len(content.encode()),
                "source_meta.file_meta.file_timestamp": 1700000000,
                "source_meta.file_meta.language": "Python",
                "source_meta.file_meta.is_vendor": i % 5 == 4,
                "source_meta.file_meta.license_type": "no_license" if i % 3 == 2 else "permissive",
                "source_meta.file_meta.detected_licenses": [] if i % 3 == 2 else (["MIT"] if i % 2 else ["Apache-2.0"]),
                "source_meta.commit_id": f"{i % 7:040x}",
            }
        )
    return rows


def ufw_rows(n: int, *, zh: bool, offset: int = 0, texts: Sequence[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for i in range(offset, offset + n):
        if texts is not None:
            content = texts[(i - offset) % len(texts)]
        elif zh:
            content = f"{MARK} {i}：这是测试用的合成文本，第{i}篇。"
        else:
            content = f"{MARK} {i}: a synthetic English test document number {i}."
        source = UFW_ZH_SOURCES[(i // 3) % len(UFW_ZH_SOURCES)] if zh else "FineWeb"
        rows.append({"content": content, "score": f"0.{500 + (i * 37) % 500:03d}", "source": source})
    return rows


def code_l2_rows(n: int) -> list[dict[str, Any]]:
    return [
        {
            "uuid": f"l2-{i}",
            "repo_name": f"fixture-org/l2repo{i % 3}",
            "relative_path": f"m{i}.py",
            "content": f"# {MARK} {i}\nprint({i})\n",
            "category": "TOOL",
            "algo_rel_score": 0.9,
            "quality_score": 3.5,
        }
        for i in range(n)
    ]


def unpc_rows(n: int) -> list[dict[str, Any]]:
    return [
        {"translation.en": f"{MARK} {i}: the committee met.", "translation.zh": f"{MARK} {i}：委员会举行了会议。"}
        for i in range(n)
    ]


def hub_files(source, files: Sequence[bytes]) -> dict[tuple[str, str, str], bytes]:
    """The fixture files under the first paths of `source`'s registry pattern. The registry expects every file of
    its range, so tests use a copy of the source whose range is just these files (see `small_source`)."""
    paths = source.expand_files()
    assert len(paths) == len(files), (len(paths), len(files))
    return {(source.repo, source.revision, path): data for path, data in zip(paths, files)}


def small_source(source_id: str, n_files: int):
    """The registry entry with its file range cut to the first `n_files` files (same names, same width)."""
    import dataclasses
    import re

    from surogate_corpora.sources import RANGE, get

    source = get(source_id)
    first = RANGE.search(source.files).group(1)
    last = f"{int(first) + n_files - 1:0{len(first)}d}"
    files = re.sub(r"\{(\d+)\.\.(\d+)\}", f"{{{first}..{last}}}", source.files)
    # the "of-NNNNN" part of the names stays: the files keep their real names
    return dataclasses.replace(source, files=files)


def write_rows_json(path: Path, source, rows: Sequence[Mapping[str, Any]], truncated: Sequence[int] = ()) -> Path:
    """A datasets-server style /rows answer for `rows` (flat dicts) in the source's server schema."""
    features = []
    tree = nest({leaf: kind for leaf, kind in source.server_schema().items()})

    def feature(node):
        if isinstance(node, str):
            return (
                [{"dtype": "string", "_type": "Value"}] if node == "list<string>" else {"dtype": node, "_type": "Value"}
            )
        return {k: feature(v) for k, v in node.items()}

    if set(tree) == {"translation"}:
        features = [
            {"feature_idx": 0, "name": "translation", "type": {"languages": ["en", "zh"], "_type": "Translation"}}
        ]
    else:
        features = [{"feature_idx": i, "name": k, "type": feature(v)} for i, (k, v) in enumerate(tree.items())]
    out = {
        "features": features,
        "rows": [
            {"row_idx": i, "row": nest(r), "truncated_cells": ["content"] if i in truncated else []}
            for i, r in enumerate(rows)
        ],
        "num_rows_total": len(rows),
        "num_rows_per_page": 100,
        "partial": False,
    }
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return path


def serve_source(
    hub: FakeHub, monkeypatch, source_id: str, files_rows: Sequence[Sequence[Mapping[str, Any]]], row_group_size: int
):
    """Put one fixture file per entry of `files_rows` on the fake Hub under the source's real names, and make the
    registry's entry list exactly those files for the test. Returns (source, file bytes)."""
    from surogate_corpora import sources

    source = small_source(source_id, len(files_rows))
    monkeypatch.setitem(sources.SOURCES, source_id, source)
    blobs = [parquet_bytes(source.schema, rows, row_group_size) for rows in files_rows]
    hub.files.update(hub_files(source, blobs))
    return source, blobs


def export_fixture(
    hub: FakeHub,
    client_for,
    monkeypatch,
    out: Path,
    source_id: str,
    profile: str,
    files_rows: Sequence[Sequence[Mapping[str, Any]]],
    rg: int = 20,
    budget: int = 10**9,
    **kwargs,
):
    """Serve `files_rows` as the source's fixture files, plan the profile over them (seed 3) and fetch into `out`;
    returns (plan, provenance, source)."""
    from surogate_corpora import profiles
    from surogate_corpora.plan import build_plan
    from surogate_corpora.run import fetch

    source, _ = serve_source(hub, monkeypatch, source_id, files_rows, row_group_size=rg)
    client = client_for(hub)
    plan = build_plan(profiles.resolve(profile, kwargs.pop("params", None)), budget_bytes=budget, seed=3, client=client)
    doc = fetch(plan, out, client=client, downloaded="2026-11-03", **kwargs)
    return plan, doc, source
