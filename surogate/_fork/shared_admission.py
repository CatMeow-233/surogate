"""FIFO admission and client-disconnect abort for SharedModelServer (ledger topic T28).

Upstream defect (282f9901): ``SharedModelServer.admit`` answers 429 as soon as all
``max_concurrency`` lanes are busy. The colocated GRPO run keeps
``max_inflight_rollouts`` requests in flight (``batch_size`` by default), against
``min(16, batch_size)`` lanes (``native_colocate.py``), so every request past the
lanes is refused. The OpenAI SDK retries a 429 ten times with 0.5-8 s jittered
backoff (``utils/client.py`` sets ``max_retries=10``); after about 42-56 s the
rollout fails, verifiers stores the error in the rollout, and the scheduler counts
it toward ``MAX_ROLLOUT_ATTEMPTS_PER_GROUP`` and reschedules it. A freed lane goes
to whichever POST arrives first, which is usually a refill request issued
milliseconds after a completion, so backed-off originals lose to later groups,
GRPO groups finish piecemeal, and groups are dropped. ``capacity.py`` sizes
``max_pending_requests`` and ``pending_timeout_ms`` for exactly this load and
``native_colocate.py`` passes them in the settings, but this server never read
them. The native SharedServer is not affected: its pending queue is FIFO.

This module gives the server what the native engine has:

- A request that finds every lane busy waits in arrival order, bounded by
  ``settings["max_pending_requests"]`` waiters plus one per lane, and by
  ``settings["pending_timeout_ms"]`` of waiting. Past either bound, and while
  the server is training (``sleeping``) or closed, it is refused with 429 as
  before. ``begin_training()`` and ``close()`` already set ``sleeping`` and notify
  the server's condition, which fails every waiter. The extra slot per lane is
  for the handoff: ``do_POST`` writes the answer inside ``admit()``, so a lane
  stays taken until its handler leaves, while the client may already have sent
  the request that replaces the answered one. capacity.py sizes
  ``max_pending_requests`` to the run's in-flight count minus the lanes, with no
  slack, and without these slots the replacement found the queue full (4-9 of
  about 270 requests per batch in the CPU emulation).
- A waiter whose client has closed its connection is never admitted, and an
  admitted request whose client has gone stops at its next token check. The
  scheduler cancels its in-flight rollouts at every batch end
  (``Scheduler.cancel_inflight_rollouts``); httpx closes those connections, and
  without this check the queue would hand lanes to requests nobody waits for
  during the pack gap before ``begin_training()``, each decoding to
  ``max_tokens`` and lengthening its drain. The test is cpp-httplib's
  ``is_socket_alive``, which the native server uses for the same purpose
  (``DataSink::is_writable``): the socket is readable and a one-byte
  ``MSG_PEEK`` returns end of file or an error. A per-connection test, rather
  than a server-wide abort bumped by the scheduler, leaves validation rollouts
  and RULER judge calls that share this server running, because nobody
  cancelled them.

Switches (read when the server is built; a value outside the listed ones fails
construction, in either mode):

- ``SUROGATE_SHARED_SERVER_QUEUE=fifo`` (default) or ``reject``: ``reject`` is
  upstream's admission, unchanged, with no disconnect checks at all. It is the
  only way a run turns the queue off: the infer config refuses
  ``max_pending_requests: 0``.
- ``SUROGATE_SHARED_SERVER_ABORT=1`` (default) or ``0``: with ``0``, admitted
  requests run to completion after their client leaves (upstream's behaviour);
  queued waiters are still skipped when their client has left, because the FIFO
  would otherwise admit them. Under ``reject`` it has no effect, but is still
  checked, so a misspelled value in an A/B matrix is not taken silently.

Result contract: a request that is admitted runs exactly as before; only who
gets a lane, and when, changes. A rollout aborted because its client left would
have written its answer to a closed socket (``BrokenPipeError``, swallowed by
``do_POST``); the session is released without caching its prefix, as a failed
generation is. A client that half-closes its sending side after the request
(``shutdown(SHUT_WR)``) is indistinguishable from one that left and is treated
as gone, as in the native server; httpx, requests and aiohttp do not do this.
"""

from __future__ import annotations

import collections
import os
import select
import socket
import threading
import time

QUEUE_ENV = "SUROGATE_SHARED_SERVER_QUEUE"
ABORT_ENV = "SUROGATE_SHARED_SERVER_ABORT"

# Used when the settings carry no bound (the server built outside native_colocate,
# e.g. by tests). The engine's own default (serve_options.h; capacity.py
# ENGINE_DEFAULT_PENDING), and capacity.py's hold for the default client timeout.
DEFAULT_MAX_PENDING = 16
DEFAULT_PENDING_TIMEOUT_MS = 1_080_000

# A queued request re-checks its client and its deadline at least this often; a
# freed lane, begin_training() and close() wake it at once through the condition.
WAIT_POLL_S = 1.0
# An admitted request checks its client at most this often, from the token loop.
ABORT_CHECK_S = 0.05


class ClientDisconnected(ConnectionResetError):
    """The client closed its connection; nobody reads this request's answer.

    A ConnectionResetError, so ``do_POST`` drops it without writing a response,
    as it drops a client that leaves mid-stream.
    """


def peer_closed(connection) -> bool:
    """True when the peer has closed the connection (cpp-httplib's ``!is_socket_alive``)."""
    try:
        fd = connection.fileno()
    except (OSError, ValueError):
        return True
    if fd < 0:
        return True
    try:
        if hasattr(select, "poll"):
            poller = select.poll()
            poller.register(fd, select.POLLIN)
            readable = bool(poller.poll(0))
        else:  # pragma: no cover - Windows
            readable = bool(select.select([fd], [], [], 0)[0])
    except (OSError, ValueError):
        return True
    if not readable:
        return False
    try:
        # Readable: either data (the client would be pipelining, which HTTP/1.0
        # here never asks for) or end of file. The peek does not consume data.
        return connection.recv(1, socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)) == b""
    except (TimeoutError, BlockingIOError, InterruptedError):
        return False
    except OSError:
        return True


def _refuse(message):
    from surogate.grpo.shared_model import Busy  # imported late: shared_model imports this module

    raise Busy(message)


def _switch(environ, name, default, allowed):
    value = environ.get(name, default).strip().lower() or default
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}, not {value!r}")
    return value


class Admission:
    """Admission policy of one SharedModelServer. Every method runs under the server's condition."""

    def __init__(self, settings: dict, environ=None):
        environ = os.environ if environ is None else environ
        self.mode = _switch(environ, QUEUE_ENV, "fifo", ("fifo", "reject"))
        self.fifo = self.mode == "fifo"
        # Parsed in both modes so that an invalid value fails here under reject too.
        abort = _switch(environ, ABORT_ENV, "1", ("0", "1")) == "1"
        self.abort = self.fifo and abort
        pending = settings.get("max_pending_requests")
        timeout_ms = settings.get("pending_timeout_ms")
        self.max_pending = DEFAULT_MAX_PENDING if pending is None else int(pending)
        # Waiters past max_pending_requests: one per lane for an answered request that still holds it.
        # max_pending_requests=0 asks for no queue and gets none. Only settings passed
        # directly carry it: GRPOInferenceConfig refuses 0 and capacity.py derives at least 16.
        self.queue_limit = self.max_pending + int(settings["max_concurrency"]) if self.max_pending else 0
        self.pending_timeout_s = (DEFAULT_PENDING_TIMEOUT_MS if timeout_ms is None else int(timeout_ms)) / 1000.0
        if self.max_pending < 0 or self.pending_timeout_s <= 0:
            raise ValueError("max_pending_requests must be nonnegative and pending_timeout_ms positive")
        self.queue = collections.deque()
        self.local = threading.local()
        self.counts = collections.Counter()

    def wait(self, server, model, connection=None) -> None:
        """Wait, holding ``server.condition``, until this request may take a lane.

        Returns with a free lane when admitted, and otherwise returns for the
        caller's own checks to refuse (training or closed: 429; unknown model:
        400). Raises Busy (429) when the queue is full or the wait passed
        pending_timeout_ms, and ClientDisconnected when the client left.
        """
        self.local.connection = connection if self.abort else None
        self.local.checked = time.monotonic()
        if server.sleeping or server.closed:
            self.counts["refused_training"] += 1
            return
        busy = server.active >= server.capacity
        if not self.fifo or model not in (server.model, server.adapter):
            self.counts["refused_capacity"] += busy
            return
        if not busy and not self.queue:
            return
        if len(self.queue) >= self.queue_limit:
            # Refused here rather than by the caller: a lane can be free while
            # the queue's head has been notified but not yet run.
            self.counts["refused_capacity"] += 1
            _refuse("all lanes are busy and the request queue is full; retry")
        ticket = object()
        self.queue.append(ticket)
        self.counts["queued"] += 1
        self.counts["max_queued"] = max(self.counts["max_queued"], len(self.queue))
        deadline = time.monotonic() + self.pending_timeout_s
        try:
            while True:
                if server.sleeping or server.closed:
                    self.counts["refused_training"] += 1
                    return
                if connection is not None and peer_closed(connection):
                    self.counts["disconnected_waiting"] += 1
                    raise ClientDisconnected("client closed the connection while queued")
                if self.queue[0] is ticket and server.active < server.capacity:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.counts["refused_timeout"] += 1
                    _refuse("request waited longer than pending_timeout_ms for a lane; retry")
                server.condition.wait(min(remaining, WAIT_POLL_S))
        finally:
            self.queue.remove(ticket)
            # The next waiter may now be the head of a free lane.
            server.condition.notify_all()

    def check_client(self) -> None:
        """Raise ClientDisconnected when this thread's admitted request has lost its client."""
        connection = getattr(self.local, "connection", None)
        if connection is None:
            return
        now = time.monotonic()
        if now - self.local.checked < ABORT_CHECK_S:
            return
        self.local.checked = now
        if peer_closed(connection):
            self.local.connection = None
            self.counts["aborted_running"] += 1
            raise ClientDisconnected("client closed the connection during generation")

    def summary(self) -> dict:
        counts = self.counts
        return dict(
            admission=self.mode,
            admission_abort=self.abort,
            admission_max_pending=self.max_pending,
            admission_queue_limit=self.queue_limit,
            admission_waiting=len(self.queue),
            **{
                f"admission_{key}": counts[key]
                for key in (
                    "queued",
                    "max_queued",
                    "refused_capacity",
                    "refused_timeout",
                    "refused_training",
                    "disconnected_waiting",
                    "aborted_running",
                )
            },
        )
