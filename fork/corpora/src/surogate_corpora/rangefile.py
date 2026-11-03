"""A seekable, readable file over HTTP Range requests on a resolved URL, with a small block cache. It never writes to
disk: what pyarrow reads is fetched, handed over and forgotten, except the last `cache_blocks` blocks.

A read fetches the aligned blocks that cover it and are not cached, coalescing consecutive missing blocks into one
request of at most `max_request` bytes (so a retry costs at most that much). `bytes_fetched` counts the body bytes
received: whole blocks, the file's last block short. Every answer must be 206 with `Content-Range: bytes a-b/size`
for exactly the asked range and the listed size; a 200 (a server that ignores Range) is refused before its body is
read, since it would be the whole file. A 401/403 (an expired or revoked signed URL) re-resolves once through
`refresh`; so does a URL whose `Expires` is less than a minute away. Transient failures are retried with backoff.

The block is 64 KiB by default, not the 8 MiB the spec first proposed: requests are coalesced per read, so a larger
block only adds over-fetch at both ends of every read, and the UN PC row groups are about 160 KB [verified, footer]
(an 8 MiB block would fetch about 50 times the data).
"""

from __future__ import annotations

import io
import time
from collections import OrderedDict
from collections.abc import Callable

from .canonical import CorporaError
from .hub import TRANSIENT, TRANSIENT_ERRORS, USER_AGENT, ResolvedURL, backoff, build_opener, open_url

EXPIRY_MARGIN = 60.0  # seconds


class RangeFile(io.RawIOBase):
    def __init__(
        self,
        url: ResolvedURL,
        *,
        block: int = 64 << 10,
        cache_blocks: int = 16,
        refresh: Callable[[], ResolvedURL] | None = None,
        timeout: float = 60,
        retries: int = 5,
        max_request: int = 16 << 20,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ):
        super().__init__()
        if block <= 0 or cache_blocks <= 0 or max_request < block:
            raise ValueError("block and cache_blocks must be positive and max_request at least one block")
        self.url = url
        self.size = url.file.size
        self.block = block
        self.cache_blocks = cache_blocks
        self.refresh = refresh
        self.timeout = timeout
        self.retries = retries
        self.max_blocks = max_request // block
        self.sleep = sleep
        self.clock = clock
        self.opener = build_opener()
        self.pos = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.bytes_fetched = 0
        self.requests = 0
        self.refreshes = 0

    # --- io.RawIOBase -------------------------------------------------------------------------------------------

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        if base + offset < 0:
            raise ValueError(f"negative seek position {base + offset}")
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        out = memoryview(buffer).cast("B")
        first, last = self.pos // self.block, (self.pos + n - 1) // self.block
        blocks = {i: self.cache[i] for i in range(first, last + 1) if i in self.cache}  # before any eviction
        blocks.update(self._fetch([i for i in range(first, last + 1) if i not in blocks]))
        written = 0
        for index in range(first, last + 1):
            data = blocks.pop(index)
            start = self.pos + written - index * self.block
            piece = data[start : start + n - written]
            out[written : written + len(piece)] = piece
            written += len(piece)
            self._remember(index, data)
        self.pos += written
        return written

    # --- fetching -----------------------------------------------------------------------------------------------

    def _remember(self, index: int, data: bytes) -> None:
        self.cache[index] = data
        self.cache.move_to_end(index)
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)

    def _fetch(self, missing: list[int]) -> dict[int, bytes]:
        """The missing blocks, consecutive ones in shared requests of at most max_request bytes."""
        runs: list[list[int]] = []
        for index in missing:
            if runs and index == runs[-1][-1] + 1 and len(runs[-1]) < self.max_blocks:
                runs[-1].append(index)
            else:
                runs.append([index])
        out = {}
        for run in runs:
            start = run[0] * self.block
            end = min((run[-1] + 1) * self.block, self.size) - 1
            data = self._get(start, end)
            for k, index in enumerate(run):
                out[index] = data[k * self.block : (k + 1) * self.block]
        return out

    def _refresh(self) -> None:
        if self.refresh is None:
            raise CorporaError(f"{self.url.file.path}: the signed URL expired and there is no way to re-resolve it")
        url = self.refresh()
        if url.file != self.url.file:
            raise CorporaError(f"{self.url.file.path}: re-resolving gave another file: {url.file}")
        self.url = url
        self.refreshes += 1

    def _get(self, start: int, end: int) -> bytes:
        refreshed = False
        for attempt in range(self.retries + 1):
            if self.url.expires is not None and self.clock() > self.url.expires - EXPIRY_MARGIN and not refreshed:
                self._refresh()
                refreshed = True
            headers = {**dict(self.url.headers), "Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT}
            try:
                status, answer, body = open_url(self.opener, self.url.url, headers=headers, timeout=self.timeout)
                with body:
                    if status == 206:
                        data = body.read()
            except TRANSIENT_ERRORS as err:
                if attempt == self.retries:
                    raise CorporaError(f"{self.url.file.path} bytes {start}-{end}: {err}") from None
                self.sleep(backoff(attempt))
                continue
            self.requests += 1
            if status == 206:
                expected = f"bytes {start}-{end}/{self.size}"
                if answer.get("content-range") != expected:
                    raise CorporaError(
                        f"{self.url.file.path}: Content-Range {answer.get('content-range')!r}, expected {expected!r}"
                    )
                if len(data) != end - start + 1:
                    if attempt == self.retries:
                        raise CorporaError(f"{self.url.file.path}: short body {len(data)} for bytes {start}-{end}")
                    self.sleep(backoff(attempt))
                    continue
                self.bytes_fetched += len(data)
                return data
            if status == 200:
                raise CorporaError(
                    f"{self.url.file.path}: the server ignored the Range header (HTTP 200); refusing to read the "
                    "whole file"
                )
            if status in (401, 403) and not refreshed:
                self._refresh()
                refreshed = True
                continue
            if status in TRANSIENT and attempt < self.retries:
                self.sleep(backoff(attempt, answer))
                continue
            raise CorporaError(f"{self.url.file.path} bytes {start}-{end}: HTTP {status}")
        raise CorporaError(f"{self.url.file.path} bytes {start}-{end}: gave up after {self.retries + 1} attempts")
