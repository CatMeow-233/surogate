"""The Hugging Face Hub client: file listings at a pinned revision, and the resolution of a file to the URL its bytes
are served from, checked against the listing.

- `list_files` reads `/api/datasets/<repo>/tree/<revision>/<dir>` page by page (the `Link: rel="next"` header) and
  returns the files a registry pattern expands to, with their size, LFS sha256 and xet hash. A file the pattern
  names but the listing lacks, or a listed file that matches the pattern's shape but is outside its range, is
  refused: the registry and the remote disagree.
- `resolve` sends HEAD to `/datasets/<repo>/resolve/<revision>/<path>` without following the redirect. It refuses a
  size, ETag (LFS sha256), xet hash or commit that differs from the listing, and returns the redirect target.
- The token is sent only to the endpoint's host. A resolved URL on another host (the CDN the redirect names) never
  gets it: `ResolvedURL.headers` is what range requests send, and it holds Authorization only for the endpoint host.
- Proxy and CA come from the environment (urllib's ProxyHandler reads HTTPS_PROXY and no_proxy; the default SSL
  context reads SSL_CERT_FILE). Transient failures (connection errors, timeouts, HTTP 429 and 5xx) are retried with
  backoff, honouring Retry-After.
"""

from __future__ import annotations

import http.client
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .canonical import CorporaError
from .sources import RANGE, REVISION, expand

USER_AGENT = "surogate-corpora/0.1"
REDIRECTS = (301, 302, 303, 307, 308)
TRANSIENT = (429, 500, 502, 503, 504)
NEXT_LINK = re.compile(r'<([^>]+)>\s*;\s*rel="?next"?')


class HubError(CorporaError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # a redirect surfaces as an HTTPError carrying the Location header


def build_opener() -> urllib.request.OpenerDirector:
    """urllib with proxies and CA from the environment, and no redirect following."""
    return urllib.request.build_opener(_NoRedirect)


@dataclass(frozen=True)
class Response:
    status: int
    headers: Mapping[str, str]  # lower-case names
    body: bytes


def open_url(
    opener: urllib.request.OpenerDirector, url: str, *, method: str = "GET", headers: Mapping[str, str], timeout: float
) -> tuple[int, Mapping[str, str], Any]:
    """(status, lower-case headers, open body stream) of one request; a non-2xx answer is returned, not raised."""
    request = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT, **headers})
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as err:
        return err.code, {k.lower(): v for k, v in err.headers.items()}, err
    return response.status, {k.lower(): v for k, v in response.headers.items()}, response


TRANSIENT_ERRORS = (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, http.client.HTTPException)


def backoff(attempt: int, headers: Mapping[str, str] | None = None) -> float:
    retry_after = (headers or {}).get("retry-after")
    if retry_after and retry_after.strip().isdigit():
        return min(float(retry_after), 300.0)
    return min(2.0**attempt, 60.0)


@dataclass(frozen=True)
class RemoteFile:
    repo: str
    revision: str
    path: str
    size: int
    lfs_sha256: str | None
    xet_hash: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> RemoteFile:
        return cls(**{k: data[k] for k in ("repo", "revision", "path", "size", "lfs_sha256", "xet_hash")})


@dataclass(frozen=True)
class ResolvedURL:
    file: RemoteFile
    url: str
    headers: tuple[tuple[str, str], ...]  # sent with every range request on `url`
    etag: str | None  # observed X-Linked-ETag (the LFS sha256), unquoted
    xet_hash: str | None  # observed X-Xet-Hash
    expires: float | None  # the signed URL's Expires, epoch seconds, when it has one


def _etag(value: str | None) -> str | None:
    if not value:
        return None
    return value.removeprefix("W/").strip('"')


class HubClient:
    def __init__(
        self,
        token: str | None = None,
        endpoint: str = "https://huggingface.co",
        timeout: float = 60,
        retries: int = 5,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.token = token
        self.endpoint = endpoint.rstrip("/")
        self.host = urllib.parse.urlsplit(self.endpoint).netloc
        self.timeout = timeout
        self.retries = retries
        self.sleep = sleep
        self.opener = build_opener()
        self.requests = 0

    # --- low level --------------------------------------------------------------------------------------------

    def auth_headers(self, url: str) -> dict[str, str]:
        """Authorization for the endpoint's host only."""
        if self.token and urllib.parse.urlsplit(url).netloc == self.host:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def request(self, url: str, *, method: str = "GET", accept: tuple[int, ...] = (200,)) -> Response:
        """One API request with retries; refuses any status outside `accept` (after retrying transient ones)."""
        for attempt in range(self.retries + 1):
            try:
                status, headers, body = open_url(
                    self.opener, url, method=method, headers=self.auth_headers(url), timeout=self.timeout
                )
                with body:
                    data = b"" if method == "HEAD" else body.read()
            except TRANSIENT_ERRORS as err:
                if attempt == self.retries:
                    raise HubError(f"{method} {url}: {err} (after {attempt + 1} attempts)") from None
                self.sleep(backoff(attempt))
                continue
            self.requests += 1
            if status in accept:
                return Response(status, headers, data)
            if status in TRANSIENT and attempt < self.retries:
                self.sleep(backoff(attempt, headers))
                continue
            raise HubError(f"{method} {url}: HTTP {status}")
        raise AssertionError("unreachable")

    def get_json(self, url: str) -> tuple[Any, Mapping[str, str]]:
        response = self.request(url)
        try:
            return json.loads(response.body), response.headers
        except ValueError as err:
            raise HubError(f"GET {url}: not JSON: {err}") from None

    # --- listing ----------------------------------------------------------------------------------------------

    def list_files(self, repo: str, revision: str, pattern: str) -> list[RemoteFile]:
        """The files `pattern` expands to, as the Hub lists them at `revision` (sorted by path)."""
        if not REVISION.fullmatch(revision):
            raise HubError(f"{repo}: revision must be a full 40-hex commit sha, not {revision!r}")
        expected = expand(pattern)
        directories = {path.rpartition("/")[0] for path in expected}
        if len(directories) != 1:
            raise HubError(f"{pattern!r}: the range must be inside one directory")
        directory = directories.pop()
        url = f"{self.endpoint}/api/datasets/{repo}/tree/{revision}/{urllib.parse.quote(directory)}"
        listed: dict[str, Mapping[str, Any]] = {}
        while url:
            page, headers = self.get_json(url)
            if not isinstance(page, list):
                raise HubError(f"GET {url}: expected a list of tree entries")
            listed.update((entry["path"], entry) for entry in page if entry.get("type") == "file")
            match = NEXT_LINK.search(headers.get("link", ""))
            url = urllib.parse.urljoin(url, match.group(1)) if match else ""
            if url and urllib.parse.urlsplit(url).netloc != self.host:
                raise HubError(f"the tree listing's next page is on another host: {url}")
        missing = [path for path in expected if path not in listed]
        if missing:
            raise HubError(
                f"{repo}@{revision[:8]}: {len(missing)} of {len(expected)} files of {pattern!r} are not listed "
                f"(first: {missing[0]})"
            )
        shape = re.compile(re.escape(pattern).replace(re.escape(RANGE.search(pattern).group(0)), r"\d+"))
        extra = sorted(p for p in listed if shape.fullmatch(p) and p not in set(expected))
        if extra:
            raise HubError(f"{repo}@{revision[:8]}: listed files outside the registry's range: {extra[:3]}")
        out = []
        for path in expected:
            entry = listed[path]
            lfs = entry.get("lfs") or {}
            out.append(RemoteFile(repo, revision, path, int(entry["size"]), lfs.get("oid"), entry.get("xetHash")))
        return out

    # --- resolution -------------------------------------------------------------------------------------------

    def resolve_url(self, f: RemoteFile) -> str:
        return f"{self.endpoint}/datasets/{f.repo}/resolve/{f.revision}/{urllib.parse.quote(f.path)}"

    def resolve(self, f: RemoteFile) -> ResolvedURL:
        """Where `f`'s bytes are served from; refuses a size, ETag, xet hash or commit that differs from the
        listing."""
        url = self.resolve_url(f)
        response = self.request(url, method="HEAD", accept=(200, *REDIRECTS))
        headers = response.headers
        if response.status in REDIRECTS:
            if "location" not in headers:
                raise HubError(f"HEAD {url}: redirect without a Location")
            target = urllib.parse.urljoin(url, headers["location"])
            size = headers.get("x-linked-size")
            etag = _etag(headers.get("x-linked-etag"))
        else:
            target = url
            size = headers.get("x-linked-size") or headers.get("content-length")
            etag = _etag(headers.get("x-linked-etag") or headers.get("etag"))
        xet = headers.get("x-xet-hash")
        commit = headers.get("x-repo-commit")
        if commit and commit != f.revision:
            raise HubError(f"{f.path}: the Hub answered for commit {commit}, not the pinned {f.revision}")
        if size is None or int(size) != f.size:
            raise HubError(f"{f.path}: size {size} at resolve differs from the listing's {f.size}")
        if etag and f.lfs_sha256 and etag != f.lfs_sha256:
            raise HubError(f"{f.path}: ETag {etag} at resolve differs from the listing's LFS sha256 {f.lfs_sha256}")
        if xet and f.xet_hash and xet != f.xet_hash:
            raise HubError(f"{f.path}: xet hash {xet} at resolve differs from the listing's {f.xet_hash}")
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(target).query)
        expires = query.get("Expires", [None])[0]
        return ResolvedURL(
            file=f,
            url=target,
            headers=tuple(self.auth_headers(target).items()),
            etag=etag,
            xet_hash=xet,
            expires=float(expires) if expires and expires.isdigit() else None,
        )
