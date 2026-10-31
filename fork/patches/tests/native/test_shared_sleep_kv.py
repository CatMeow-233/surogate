"""Source pins of ledger topic T29 (the native GRPO SharedServer evicts retained KV before sleeping).

`GenerationService::begin_shared_training` drains the rollouts and puts the engine to sleep. Upstream slept with
the finished lanes' prefixes still retained, so the elastic KV region copied every used granule into fresh pinned
host memory, and `publish_shared_adapter` mapped them back and copied them in, then evicted them with
`shrink_kv()` before reopening admission. T29 calls `shrink_kv()` before `sleep()`, which drops exactly what
the publication would drop, before it is copied. These pins read one function body each, so moved lines do not
matter; they fail when a sync drops or reshapes the hunk, or when upstream changes code the argument relies on:
no request can be admitted between the sleep and the publication's eviction, the publication still evicts after
its wake, `shrink_kv` evicts idle retained lanes and archived prefixes, and the region backs up only granules
that still hold pages. Behaviour is checked only on a GPU (see fork/docs/performance.md).
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
SERVICE = REPO / "csrc/src/serve/serve/generation_service.cpp"
EXECUTOR = REPO / "csrc/src/serve/runtime/engine/concurrent_executor.h"
REGION = REPO / "csrc/src/serve/core/elastic_kv_region.cu"
SHARED_BINDING = REPO / "csrc/src/binding/py_serve_shared.cpp"


def body(path: Path, signature: str) -> str:
    """The brace-balanced body of the first definition matching `signature` (a regex), whitespace collapsed."""
    text = path.read_text(encoding="utf-8")
    match = re.search(signature + r"[^;{]*\{", text)
    assert match, f"{path.relative_to(REPO)}: no definition matching {signature!r}"
    depth, start = 1, match.end()
    for end in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            return " ".join(text[start:end].split())
    raise AssertionError(f"{path.relative_to(REPO)}: unbalanced braces after {signature!r}")


def in_order(code: str, *parts: str) -> None:
    at = 0
    for part in parts:
        found = code.find(part, at)
        assert found >= 0, f"{part!r} missing or out of order"
        at = found + len(part)


@pytest.fixture(scope="module")
def begin_training() -> str:
    return body(SERVICE, r"void GenerationService::begin_shared_training\(\)")


def test_begin_shared_training_evicts_retained_kv_before_sleeping(begin_training):
    in_order(
        begin_training,
        "while (active_requests() != 0) {",
        'std::getenv("SUROGATE_SHARED_SLEEP_KEEP_KV")',
        "if (!keep_kv) { engine_->shrink_kv(); }",
        "engine_->sleep();",
        "} catch (...) {",
        "shared_training_ = false;",
    )
    # The eviction runs after the drain and inside the try, so a failure reopens rollouts as a failed sleep does.
    assert begin_training.count("engine_->shrink_kv();") == 1
    assert begin_training.count("engine_->sleep();") == 1


def test_the_switch_keeps_the_old_order_only_when_set_to_a_nonzero_value(begin_training):
    assert (
        'return value != nullptr && std::string_view(value) != "" && std::string_view(value) != "0";' in begin_training
    )


def test_no_request_is_admitted_while_the_shared_model_trains():
    # New requests are refused for the whole phase, and the engine cannot be woken by anyone but the publication.
    in_order(
        body(SERVICE, r"std::shared_ptr<RequestLifetime> GenerationService::acquire_request_lifetime\(\) const"),
        "if (shared_training_) {",
        "rollouts are paused for a training step",
    )
    in_order(
        body(SERVICE, r"void GenerationService::wake_up\(\)"),
        "if (shared_training_) { throw std::logic_error(",
        "engine_->wake();",
    )


def test_the_publication_still_evicts_everything_it_restores_before_reopening():
    publish = body(SERVICE, r"void GenerationService::publish_shared_adapter\(")
    in_order(
        publish,
        "if (!shared_training_ || request_capacity_->active != 0) {",
        "engine_->wake();",
        "engine_->shrink_kv();",
        "shared_training_ = false;",
    )


def test_shrink_kv_evicts_idle_retained_lanes_and_archived_prefixes():
    shrink = body(EXECUTOR, r"void shrink_kv\(\)")
    in_order(
        shrink,
        "auto paused = pause_execution();",
        "instance_.program->evict_archived_prefixes();",
        "if (slots_[lane] == nullptr && instance_.program->has_retained_lane(lane)) {",
        "instance_.program->evict_retained_lane(lane);",
        "instance_.program->kv_settle();",
    )


def test_the_region_backs_up_only_granules_that_still_hold_pages():
    sleep = body(REGION, r"std::size_t ElasticKvRegion::sleep\(\)")
    in_order(
        sleep,
        "wait_idle();",
        "if (granule.used_pages != 0) {",
        "cudaMallocHost(",
        "cudaMemcpyDeviceToHost",
        "impl.unmap_granule(g);",
    )
    # A released page leaves its granule's count at once, under the region's mutex, so an evicted lane's granules
    # are empty by the time sleep() reads them.
    release = body(REGION, r"void ElasticKvRegion::release_page\(std::int32_t page\) noexcept try")
    in_order(
        release, "const std::lock_guard<std::mutex> lock(impl.mutex);", "if (--granule.used_pages != 0) { return; }"
    )
    wake = body(REGION, r"std::size_t ElasticKvRegion::wake\(\)")
    in_order(
        wake,
        "if (granule.used_pages == 0) {",
        "continue;",
        "impl.map_granule(g);",
        "cudaMemcpyHostToDevice",
        "cudaFreeHost(granule.backup[p]);",
    )


def test_the_binding_enters_training_through_begin_shared_training():
    # The constructor's first call (no KV yet, so the eviction is a no-op) and SharedServer.begin_training.
    assert SHARED_BINDING.read_text(encoding="utf-8").count("service_->begin_shared_training();") == 2
