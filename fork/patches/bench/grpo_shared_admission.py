"""CPU emulation of one GRPO batch on the training-path SharedModelServer (ledger topic T28).

    python fork/patches/bench/grpo_shared_admission.py --mode reject|fifo|fifo-noabort [--seed N] [--time-scale S]

Real code: SharedModelServer's HTTP handler, admit() and the T28 admission
(surogate/_fork/shared_admission.py), the OpenAI SDK's retry loop with
max_retries=10 as surogate/grpo/utils/client.py sets it, and
Scheduler.generate_batch with its refill, failure accounting and batch-end
cancel. Faked: decoding (each admitted request sleeps per_token seconds per
token and calls the token-loop disconnect check, with one lane per admitted
request), the environment (one turn) and the buffer (pass-through, no filters).
`--time-scale` scales the SDK's backoff constants and per_token together.
`--mode` sets SUROGATE_SHARED_SERVER_QUEUE / SUROGATE_SHARED_SERVER_ABORT, and
the queue bound is what native_colocate.py passes after capacity.py sizes it.
Prints one JSON report on stdout (the scheduler's warnings go to stderr).
Derived from the R22 verification harness of the python-rl efficiency roadmap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import httpx  # noqa: E402
import openai._base_client as obc  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from surogate.grpo import shared_model as sm  # noqa: E402
from surogate.grpo.orchestrator.scheduler import InflightRolloutInfo, Scheduler  # noqa: E402
from surogate.grpo.utils.logger import get_logger  # noqa: E402


class FakeTrainer:
    seq_length = 8192

    def get_shared_base_weights(self):
        return {}


class TimedServer(sm.SharedModelServer):
    """The worktree's SharedModelServer (T28 admission) with decode replaced by per-token sleeps."""

    def __init__(self, *args, **kwargs):
        self.lock = threading.Lock()
        self.attempts = defaultdict(int)  # rid -> number of POSTs seen
        self.admitted = {}  # rid -> (t_admit, attempts_before)
        self.finished = {}
        self.decoded = []  # (t, rid) per decoded token
        super().__init__(*args, **kwargs)

    def prepare(self, body):
        rid = body["rid"]
        with self.lock:
            self.admitted[rid] = (time.monotonic(), self.attempts[rid])
        return dict(rid=rid, tokens=int(body["tokens_n"]), per_token=float(body["per_token"]))

    def generate(self, request, callback=None):
        for _ in range(request["tokens"]):
            self.admission.check_client()  # the hook upstream's _generate now runs per token
            time.sleep(request["per_token"])
            with self.lock:
                self.decoded.append((time.monotonic(), request["rid"]))
        with self.lock:
            self.finished[request["rid"]] = time.monotonic()
        return dict(
            id="chatcmpl-x",
            created=int(time.time()),
            object="chat.completion",
            model=self.adapter,
            choices=[dict(index=0, message=dict(role="assistant", content="ok"), finish_reason="stop")],
            usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class PassThroughBuffer:
    def __init__(self):
        self.rollout_buffer = []

    def update(self, rollouts):
        self.rollout_buffer.extend(rollouts)

    def sample_examples(self, n):
        return [{} for _ in range(n)]

    def sample_rollouts(self, n):
        n = min(n, len(self.rollout_buffer))
        sampled, self.rollout_buffer = self.rollout_buffer[-n:], self.rollout_buffer[:-n]
        return sampled


class HttpScheduler(Scheduler):
    """Real generate_batch / refill / failure accounting; schedule_rollout issues a real SDK request."""

    TASK = "math"

    def __init__(self, client, server, args):
        self.logger = get_logger()
        self.step = 0
        self.batch_size = args.batch
        self.token_batch_size = None
        self.rollouts_per_example = args.group
        self.max_inflight_rollouts = args.inflight
        self.json_logging = True
        self.update_policy_task = None
        self.inflight_requests = {}
        self.scoring_tasks = {}
        self.groups = {}
        self.next_group_id = 0
        self.deferred_group_scoring_tasks = set()
        self.checkpoint_ready = asyncio.Event()
        self.checkpoint_ready.set()
        self.dropped_groups_by_task = defaultdict(int)
        self.total_dropped_groups = 0
        self.empty_rollouts_by_task = defaultdict(int)
        self.errored_rollouts_by_task = defaultdict(int)
        self.total_rollouts_by_task = defaultdict(int)
        self.last_batch_generation_time = 0.0
        self.cancelled_rollouts_count = 0
        self.inflight_policy_update_task = None
        self.buffer = PassThroughBuffer()
        self.prefetch_batches = False  # native colocate passes prefetch_batches=False
        self.config = SimpleNamespace(batch_stall_timeout=600)
        self.client, self.server, self.args = client, server, args
        self.rng = random.Random(args.seed)
        self.next_rid = 0
        self.issued = {}  # rid -> (group_id, t_issue)
        self.lost = []  # (rid, group_id, t_lost)
        self.completed = {}  # rid -> t_complete
        self.dropped = []
        self.original_groups = args.batch // args.group

    async def maybe_update_policy(self):
        return None

    async def update_policy_loop(self):
        await asyncio.Event().wait()

    async def drop_group(self, group_id):
        self.dropped.append(group_id)
        return await super().drop_group(group_id)

    def _tokens(self):
        # Completion length: 15% reach max_tokens, the rest log-uniform between 64 and max_tokens.
        if self.rng.random() < self.args.cap_fraction:
            return self.args.max_tokens
        lo, hi = 64, self.args.max_tokens
        return int(lo * (hi / lo) ** self.rng.random())

    async def _produce(self, group_id, rid, tokens):
        try:
            await self.client.chat.completions.create(
                model="policy",
                messages=[{"role": "user", "content": "q"}],
                max_tokens=1,
                extra_body=dict(rid=rid, tokens_n=tokens, per_token=self.args.per_token),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # verifiers wraps it as vf.ModelError into state["error"]
            self.lost.append((rid, group_id, time.monotonic()))
            return {"trajectory": [], "error": {"error_chain_repr": f"ModelError() -> {type(exc).__name__}"}}
        if self.args.score_delay:
            await asyncio.sleep(self.args.score_delay)
        self.completed[rid] = time.monotonic()
        return {"trajectory": [{"tokens": None}], "error": None, "example_id": group_id, "reward": 1.0}

    async def schedule_rollout(self, group_id):
        group = self.groups.get(group_id)
        if group is None or group.rollouts_to_schedule <= 0:
            return
        group.rollouts_to_schedule -= 1
        rid = self.next_rid
        self.next_rid += 1
        tokens = self._tokens()
        self.issued[rid] = (group_id, time.monotonic())
        task = asyncio.create_task(self._produce(group_id, rid, tokens))
        self.inflight_requests[task] = InflightRolloutInfo(
            off_policy_steps=0,
            client_config=SimpleNamespace(api_base_url="x", extra_headers={}),
            task=self.TASK,
            group_id=group_id,
        )


def main():
    # The GRPO logger writes to sys.stdout; keep stdout for the report alone.
    report_stream, sys.stdout = sys.stdout, sys.stderr
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["reject", "fifo", "fifo-noabort"], default="reject")
    p.add_argument("--lanes", type=int, default=16)
    p.add_argument("--inflight", type=int, default=128)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--cap-fraction", type=float, default=0.15)
    p.add_argument("--per-token", type=float, default=0.012)
    p.add_argument("--score-delay", type=float, default=0.0)
    p.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="scale SDK backoff constants and per_token together (1.0 = faithful)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=18431)
    args = p.parse_args()
    args.per_token *= args.time_scale
    args.score_delay *= args.time_scale
    obc.INITIAL_RETRY_DELAY = 0.5 * args.time_scale
    obc.MAX_RETRY_DELAY = 8.0 * args.time_scale

    import os

    os.environ["SUROGATE_SHARED_SERVER_QUEUE"] = "reject" if args.mode == "reject" else "fifo"
    os.environ["SUROGATE_SHARED_SERVER_ABORT"] = "0" if args.mode == "fifo-noabort" else "1"
    # What native_colocate passes after capacity.size_pending_capacity: demand - lanes, at least 16.
    settings = dict(
        host="127.0.0.1",
        port=args.port,
        model="base",
        max_context=4096,
        max_concurrency=args.lanes,
        max_pending_requests=max(16, args.inflight - args.lanes),
        pending_timeout_ms=1_080_000,
    )
    config = {"vocab_size": 1000, "model_type": "qwen3_moe", "eos_token_id": 0}
    tok = SimpleNamespace(eos_token_id=0)
    server = TimedServer(FakeTrainer(), tok, config, settings)

    # Count every POST (attempt) per rid, including the ones that 429.
    original_do_post = server.http.RequestHandlerClass.do_POST

    def counting_do_post(handler):
        try:
            length = int(handler.headers.get("Content-Length", 0))
            raw = handler.rfile.read(length)
            body = json.loads(raw)
            with server.lock:
                server.attempts[body.get("rid")] += 1
        except Exception:
            raw = b""
        import io

        handler.rfile = io.BytesIO(raw)
        return original_do_post(handler)

    server.http.RequestHandlerClass.do_POST = counting_do_post
    server.publish("policy", [], 0)

    async def run():
        http = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=8192, max_keepalive_connections=8192),
            timeout=httpx.Timeout(1200, connect=1200),
        )
        client = AsyncOpenAI(
            api_key="EMPTY", base_url=f"http://127.0.0.1:{args.port}/v1", max_retries=10, http_client=http
        )
        sched = HttpScheduler(client, server, args)
        t0 = time.monotonic()
        batch = await sched.generate_batch(step=0)
        t_batch = time.monotonic() - t0
        sched.t_cancelled = time.monotonic()  # generate_batch cancelled in-flight rollouts before returning
        await sched.stop()
        t1 = time.monotonic()
        await asyncio.to_thread(server.begin_training)
        t_drain = time.monotonic() - t1
        await http.aclose()
        return sched, batch, t_batch, t_drain

    sched, batch, t_batch, t_drain = asyncio.run(run())
    server.close()

    orig = sched.original_groups
    lost_orig = [r for r, g, _ in sched.lost if g < orig]
    groups_in_batch = sorted({r["example_id"] for r in batch})
    orig_in_batch = sum(1 for g in groups_in_batch if g < orig)
    # Admission order: for each admission, was it a request that had already been refused (retry) or fresh?
    adm = sorted(server.admitted.items(), key=lambda kv: kv[1][0])
    fresh = sum(1 for rid, (t, a) in adm if a <= 1)
    retried = len(adm) - fresh
    waits_orig = [server.admitted[r][0] - sched.issued[r][1] for r in server.admitted if sched.issued[r][0] < orig]
    waits_surplus = [server.admitted[r][0] - sched.issued[r][1] for r in server.admitted if sched.issued[r][0] >= orig]
    # Originals that were refused at their first POST: how many were ever admitted?
    first_refused_orig = [r for r, (g, _) in sched.issued.items() if g < orig and r < args.inflight]
    admitted_first_wave = sum(1 for r in first_refused_orig if r in server.admitted)
    surplus_admitted_before_last_orig = 0
    if waits_orig:
        last_orig_adm = max(server.admitted[r][0] for r in server.admitted if sched.issued[r][0] < orig)
        surplus_admitted_before_last_orig = sum(
            1 for r in server.admitted if sched.issued[r][0] >= orig and server.admitted[r][0] < last_orig_adm
        )

    def stats(xs):
        if not xs:
            return "n=0"
        xs = sorted(xs)
        return f"n={len(xs)} mean={sum(xs) / len(xs):.1f}s p50={xs[len(xs) // 2]:.1f}s max={xs[-1]:.1f}s"

    report = dict(
        mode=args.mode,
        lanes=args.lanes,
        inflight=args.inflight,
        per_token=args.per_token,
        time_scale=args.time_scale,
        score_delay=args.score_delay,
        seed=args.seed,
        batch_seconds=round(t_batch, 1),
        begin_training_wait_s=round(t_drain, 2),
        server_429=server.summary()["admission_refused_capacity"],
        admission={k: v for k, v in server.summary().items() if k.startswith("admission")},
        tokens_after_cancel=sum(1 for t, _ in server.decoded if t > sched.t_cancelled),
        requests_issued=len(sched.issued),
        admitted=len(server.admitted),
        lost_rollouts=len(sched.lost),
        lost_originals=len(lost_orig),
        dropped_groups=len(sched.dropped),
        dropped_original_groups=sum(1 for g in sched.dropped if g < orig),
        groups_in_batch=len(groups_in_batch),
        original_groups_in_batch=orig_in_batch,
        admissions_fresh=fresh,
        admissions_after_retry=retried,
        surplus_admitted_before_last_original=surplus_admitted_before_last_orig,
        first_wave_admitted=f"{admitted_first_wave}/{len(first_refused_orig)}",
        wait_to_admit_original=stats(waits_orig),
        wait_to_admit_surplus=stats(waits_surplus),
        cancelled_at_batch_end=sched.cancelled_rollouts_count,
    )
    print(json.dumps(report, indent=1), file=report_stream)


if __name__ == "__main__":
    main()
