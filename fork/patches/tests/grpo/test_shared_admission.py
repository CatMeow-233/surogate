"""FIFO admission and client-disconnect abort of SharedModelServer (ledger topic T28).

Upstream answered 429 to every request past `max_concurrency`; the OpenAI SDK
then backs off and, after its retries, the rollout is lost. These tests drive
the real HTTP handler, `admit()` and the token loop; only the model is a stub
whose decode steps can be held. The tests that fail on upstream's server are
marked in their docstrings.
"""

import asyncio
import collections
import concurrent.futures
import http.client
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import torch

from surogate._fork import shared_admission
from surogate.grpo.shared_model import SharedModelServer

REPO_ROOT = Path(__file__).resolve().parents[4]
VOCAB = 16
EOS = 1
# Token 2 has the largest logit and is never a stop token: a greedy request
# decodes to its max_tokens unless something stops it.
LOGITS = np.array([[0.0, -2.0, 2.0] + [-3.0] * (VOCAB - 3)], dtype=np.float32)


class Tokenizer:
    eos_token_id = EOS

    def decode(self, ids, **kwargs):
        return "".join(chr(65 + i) for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return [0, 2]


class Model:
    """Stands in for the trainer. A request is named by its second prompt token (its tag)."""

    batch_size = 1
    seq_length = 4096

    def __init__(self):
        self.lock = threading.Lock()
        self.steps = collections.Counter()  # tag -> decode steps run
        self.order = []  # tags in the order their first step ran
        self.started = collections.defaultdict(threading.Event)
        self.gates = {}  # tag -> Event each of its steps waits for
        self.delay = 0.0
        weight = torch.zeros(8, dtype=torch.bfloat16)
        self.get_shared_base_weights = lambda: dict(embedding=weight)

    def next_token_logits(self, inputs, positions):
        tag = int(inputs[0, 1])
        with self.lock:
            if not self.steps[tag]:
                self.order.append(tag)
            self.steps[tag] += 1
        self.started[tag].set()
        gate = self.gates.get(tag)
        if gate is not None:
            assert gate.wait(20), f"request {tag} was never released"
        if self.delay:
            time.sleep(self.delay)
        return LOGITS


@pytest.fixture
def serve(monkeypatch):
    servers = []

    def build(env=None, **settings):
        for name in (shared_admission.QUEUE_ENV, shared_admission.ABORT_ENV):
            monkeypatch.delenv(name, raising=False)
        for name, value in (env or {}).items():
            monkeypatch.setenv(name, value)
        model = Model()
        options = dict(host="127.0.0.1", port=0, model="base", max_context=model.seq_length, max_concurrency=1)
        server = SharedModelServer(model, Tokenizer(), dict(vocab_size=VOCAB), options | settings)
        servers.append(server)
        server.publish("policy", [], 0)
        admitted = []  # tags in the order they passed admit(): prepare() runs right after it
        prepare = server.prepare

        def recording(request_body):
            admitted.append(request_body["tokens"][1])
            return prepare(request_body)

        server.prepare = recording
        url = f"http://127.0.0.1:{server.http.server_port}/v1/chat/completions/tokens"
        return SimpleNamespace(server=server, model=model, url=url, port=server.http.server_port, admitted=admitted)

    yield build
    for server in servers:
        for gate in server.trainer.gates.values() if server.trainer else ():
            gate.set()
        server.close()


def body(tag, max_tokens=3):
    return dict(model="policy", tokens=[0, tag], temperature=0, max_tokens=max_tokens)


def until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.005)


def wait_queued(service, count):
    """Until `count` requests wait for a lane (on upstream's server, which has no queue, a pause)."""
    admission = getattr(service.server, "admission", None)
    if admission is None:
        time.sleep(0.3)
    else:
        until(lambda: len(admission.queue) == count)


def hold(service, tag, pool, max_tokens=3):
    """Start request `tag` and hold it inside its first decode step, occupying the lane."""
    service.model.gates[tag] = threading.Event()
    future = pool.submit(requests.post, service.url, json=body(tag, max_tokens), timeout=30)
    assert service.model.started[tag].wait(10)
    return future


def test_a_request_past_the_lanes_waits_instead_of_being_refused(serve):
    """Fails on upstream's server: the second request is answered 429 at once."""
    service = serve()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        first = hold(service, 3, pool)
        second = pool.submit(requests.post, service.url, json=body(4), timeout=30)
        wait_queued(service, 1)
        assert not second.done(), second.result().text
        service.model.gates[3].set()
        assert first.result().status_code == 200
        response = second.result()
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["token_ids"] == [2, 2, 2]
    assert service.admitted == service.model.order == [3, 4]
    summary = service.server.summary()
    assert summary["admission"] == "fifo" and summary["admission_queued"] == 1
    assert summary["admission_refused_capacity"] == 0 and summary["admission_waiting"] == 0


def test_waiting_requests_are_admitted_in_arrival_order(serve):
    """Fails on upstream's server: every request past the lane is answered 429."""
    service = serve(max_concurrency=1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        first = hold(service, 3, pool)
        waiting = []
        for count, tag in enumerate((9, 5, 12, 7), start=1):
            waiting.append(pool.submit(requests.post, service.url, json=body(tag), timeout=30))
            wait_queued(service, count)
        service.model.gates[3].set()
        responses = [first.result()] + [future.result() for future in waiting]
    assert [response.status_code for response in responses] == [200] * 5
    assert service.admitted == service.model.order == [3, 9, 5, 12, 7]
    assert service.server.summary()["admission_max_queued"] == 4


def test_a_later_arrival_does_not_take_a_lane_from_waiting_requests(serve):
    """A request arriving while others wait joins the end of the queue, even as a lane frees."""
    service = serve(max_concurrency=2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        held = [hold(service, 3, pool)]
        # Admitted to the second lane; it waits for the model, which 3 holds.
        held.append(pool.submit(requests.post, service.url, json=body(4), timeout=30))
        until(lambda: service.server.active == 2)
        waiting = []
        for count, tag in enumerate((5, 6), start=1):
            waiting.append(pool.submit(requests.post, service.url, json=body(tag), timeout=30))
            wait_queued(service, count)
        # Free a lane and send another request at once: 5 is first in line, then 6, then 8.
        service.model.gates[3].set()
        waiting.append(pool.submit(requests.post, service.url, json=body(8), timeout=30))
        assert [future.result().status_code for future in held + waiting] == [200] * 5
    assert service.admitted == [3, 4, 5, 6, 8]


def test_the_queue_holds_max_pending_requests_plus_one_per_lane(serve):
    service = serve(max_pending_requests=1, max_concurrency=1)
    assert service.server.summary()["admission_queue_limit"] == 2
    with concurrent.futures.ThreadPoolExecutor() as pool:
        first = hold(service, 3, pool)
        waiting = []
        for count, tag in enumerate((4, 5), start=1):
            waiting.append(pool.submit(requests.post, service.url, json=body(tag), timeout=30))
            wait_queued(service, count)
        refused = requests.post(service.url, json=body(6), timeout=10)
        assert refused.status_code == 429
        assert refused.json()["error"]["type"] == "capacity_error"
        service.model.gates[3].set()
        assert [future.result().status_code for future in [first] + waiting] == [200] * 3
    assert service.admitted == [3, 4, 5]
    assert service.server.summary()["admission_refused_capacity"] == 1


def test_no_queue_when_max_pending_requests_is_zero(serve):
    # Only settings passed directly carry 0: the infer config refuses it, so a run
    # turns the queue off with SUROGATE_SHARED_SERVER_QUEUE=reject.
    service = serve(max_pending_requests=0)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        first = hold(service, 3, pool)
        assert requests.post(service.url, json=body(4), timeout=10).status_code == 429
        service.model.gates[3].set()
        assert first.result().status_code == 200
    assert service.admitted == [3]


def test_a_request_waiting_past_pending_timeout_is_refused(serve):
    service = serve(pending_timeout_ms=200)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        first = hold(service, 3, pool)
        started = time.monotonic()
        refused = requests.post(service.url, json=body(4), timeout=10)
        waited = time.monotonic() - started
        assert refused.status_code == 429 and waited >= 0.2
        assert "pending_timeout_ms" in refused.json()["error"]["message"]
        service.model.gates[3].set()
        assert first.result().status_code == 200
    assert service.admitted == [3]
    assert service.server.summary()["admission_refused_timeout"] == 1


def test_begin_training_refuses_waiting_requests_and_drains_admitted_ones(serve):
    service = serve()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        first = hold(service, 3, pool)
        second = pool.submit(requests.post, service.url, json=body(4), timeout=30)
        wait_queued(service, 1)
        pause = pool.submit(service.server.begin_training)
        assert second.result(timeout=5).status_code == 429
        assert not pause.done()
        assert requests.post(service.url, json=body(5), timeout=10).status_code == 429
        service.model.gates[3].set()
        assert first.result().status_code == 200
        pause.result(timeout=10)
    assert service.admitted == [3]
    summary = service.server.summary()
    assert summary["admission_refused_training"] == 2 and summary["admission_waiting"] == 0


def test_a_waiting_request_whose_client_left_is_never_admitted(serve):
    """The batch-end cancel closes queued connections; they must not reach a lane."""
    service = serve()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        first = hold(service, 3, pool)
        connection = http.client.HTTPConnection("127.0.0.1", service.port, timeout=10)
        connection.request(
            "POST", "/v1/chat/completions/tokens", json.dumps(body(4)), {"Content-Type": "application/json"}
        )
        wait_queued(service, 1)
        third = pool.submit(requests.post, service.url, json=body(5), timeout=30)
        wait_queued(service, 2)
        connection.close()
        service.model.gates[3].set()
        assert first.result().status_code == third.result().status_code == 200
    assert service.admitted == service.model.order == [3, 5]
    assert service.server.summary()["admission_disconnected_waiting"] == 1
    assert service.server.active == 0


@pytest.mark.parametrize("abort", ["1", "0"])
def test_an_admitted_request_stops_when_its_client_leaves(serve, abort):
    """Fails on upstream's server (and with SUROGATE_SHARED_SERVER_ABORT=0): the request decodes to max_tokens."""
    service = serve(env={shared_admission.ABORT_ENV: abort})
    service.model.delay = 0.002
    connection = http.client.HTTPConnection("127.0.0.1", service.port, timeout=10)
    connection.request(
        "POST",
        "/v1/chat/completions/tokens",
        json.dumps(body(3, max_tokens=1500)),
        {"Content-Type": "application/json"},
    )
    until(lambda: service.model.steps[3] >= 20)
    connection.close()
    started = time.monotonic()
    service.server.begin_training()
    drained = time.monotonic() - started
    if abort == "1":
        assert drained < 1.5 and service.model.steps[3] < 1000
        assert service.server.summary()["admission_aborted_running"] == 1
    else:
        assert service.model.steps[3] == 1500
        assert service.server.summary()["admission_aborted_running"] == 0
    assert service.server.active == 0
    # The lane is free again after the next publication.
    service.server.publish("policy", [], 1)
    assert requests.post(service.url, json=body(4), timeout=10).status_code == 200


def test_reject_restores_upstream_admission(serve):
    service = serve(env={shared_admission.QUEUE_ENV: "reject"}, max_pending_requests=64)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        first = hold(service, 3, pool)
        refused = requests.post(service.url, json=body(4), timeout=10)
        assert refused.status_code == 429
        assert refused.json()["error"]["message"].startswith("shared model is training or at request capacity")
        service.model.gates[3].set()
        assert first.result().status_code == 200
    summary = service.server.summary()
    assert summary["admission"] == "reject" and summary["admission_abort"] is False
    assert summary["admission_refused_capacity"] == 1 and summary["admission_queued"] == 0


@pytest.mark.parametrize(
    "env",
    [
        {shared_admission.QUEUE_ENV: "lifo"},
        {shared_admission.ABORT_ENV: "yes"},
        # ABORT has no effect under reject, but a misspelled value is still refused.
        {shared_admission.ABORT_ENV: "yes", shared_admission.QUEUE_ENV: "reject"},
    ],
)
def test_unknown_switch_values_are_refused(serve, env):
    with pytest.raises(ValueError, match=next(iter(env))):
        serve(env=env)


def test_bounds_come_from_the_settings_native_colocate_passes():
    admission = shared_admission.Admission(
        dict(max_concurrency=16, max_pending_requests=112, pending_timeout_ms=1_080_000), environ={}
    )
    assert (admission.max_pending, admission.queue_limit, admission.pending_timeout_s) == (112, 128, 1080.0)
    default = shared_admission.Admission(dict(max_concurrency=4), environ={})
    assert default.max_pending == shared_admission.DEFAULT_MAX_PENDING
    with pytest.raises(ValueError):
        shared_admission.Admission(dict(max_concurrency=4, pending_timeout_ms=0), environ={})
    # capacity.py sizes both for this run's load and native_colocate.py hands them to the server.
    source = (REPO_ROOT / "surogate/grpo/native_colocate.py").read_text()
    assert 'settings["max_pending_requests"] = infer_config.max_pending_requests' in source
    assert 'settings["pending_timeout_ms"] = infer_config.pending_timeout_ms' in source
    assert "size_pending_capacity(infer_config, orch_config)" in source


openai = pytest.importorskip("openai")


def test_the_rollout_client_gets_every_answer_without_retrying(serve):
    """Fails on upstream's server: with the SDK's retries off, 4 of the 6 requests raise RateLimitError.

    The colocated run keeps more rollouts in flight than there are lanes; each
    one must come back as an answer rather than as a refusal the SDK retries.
    """
    service = serve(max_concurrency=2)
    service.model.delay = 0.005

    async def run():
        client = openai.AsyncOpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{service.port}/v1", max_retries=0)
        try:
            calls = [
                client.chat.completions.create(
                    model="policy",
                    messages=[dict(role="user", content="q")],
                    max_tokens=20,
                    temperature=0,
                    extra_body=dict(tokens=[0, tag]),
                )
                for tag in range(3, 9)
            ]
            return await asyncio.gather(*calls, return_exceptions=True)
        finally:
            await client.close()

    results = asyncio.run(run())
    assert [type(result).__name__ for result in results] == ["ChatCompletion"] * 6
    assert sorted(service.model.steps.values()) == [20] * 6


def test_cancelled_rollouts_release_their_queue_place_and_lane(serve):
    """Cancelling SDK calls, as Scheduler.cancel_inflight_rollouts does, frees the lane and the queue.

    The running call stops decoding, the queued call never starts, and the
    request that was queued behind them gets the lane.
    """
    service = serve()
    service.model.delay = 0.002

    async def run():
        client = openai.AsyncOpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{service.port}/v1", max_retries=0)

        def call(tag, max_tokens):
            return asyncio.create_task(
                client.chat.completions.create(
                    model="policy",
                    messages=[dict(role="user", content="q")],
                    max_tokens=max_tokens,
                    temperature=0,
                    extra_body=dict(tokens=[0, tag]),
                )
            )

        try:
            running = call(3, 2000)
            await asyncio.to_thread(until, lambda: service.model.steps[3] >= 10)
            queued = call(4, 2000)
            await asyncio.to_thread(wait_queued, service, 1)
            kept = call(5, 5)
            await asyncio.to_thread(wait_queued, service, 2)
            for task in (running, queued):
                task.cancel()
            await asyncio.gather(running, queued, return_exceptions=True)
            return await asyncio.wait_for(kept, 10)
        finally:
            await client.close()

    kept = asyncio.run(run())
    assert kept.choices[0].finish_reason == "length"
    assert 4 not in service.model.steps and service.model.steps[3] < 1000 and service.model.steps[5] == 5
    summary = service.server.summary()
    assert summary["admission_aborted_running"] == 1
    # The queued call's connection closes with the running one's, so it is skipped
    # in the queue; were its lane free first, it would stop before its first step.
    assert summary["admission_disconnected_waiting"] + summary["admission_aborted_running"] == 2
