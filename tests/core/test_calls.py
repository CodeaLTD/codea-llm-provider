"""Call tracking: ``start_call``, ``track_call`` and ``PendingCalls`` (Task 4 Step 1; AC-05.1 to AC-05.3)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request

import pytest
from conftest import BuildRig, Rig

from llm_provider import (
    CallCompleted,
    CallFailed,
    CallStarted,
    ResolvedLLM,
    Secret,
    StoredCredential,
    Usage,
)
from llm_provider.telemetry.calls import PendingCalls
from llm_provider.testing import StubLLMServer

LIVE_KEY = "live-key-for-the-test-0042"  # none of scrub's own patterns match it


def resolve(rig: Rig, provider: str = "openai", *, key: str = LIVE_KEY, base_url: str | None = None) -> ResolvedLLM:
    selection = rig.service.select(provider, base_url=base_url)
    return rig.service.resolve(selection, user_credential=StoredCredential(Secret(key)))


def ask(stub: StubLLMServer) -> Usage:
    """One non-streaming OpenAI call to the stub; the usage it reports."""
    request = urllib.request.Request(
        f"{stub.url}/chat/completions",
        data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        usage = json.load(response)["usage"]
    return Usage(usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"])


def events_of(rig: Rig, kind: type) -> list:
    return [e for e in rig.sink.events if isinstance(e, kind)]


# -- AC-05.1 -------------------------------------------------------------------------------------


def test_track_call_emits_started_then_completed_with_latency_and_usage(build: BuildRig) -> None:
    rig = build()
    with StubLLMServer(delay_seconds=0.05) as stub:
        resolved = resolve(rig, base_url=stub.url)
        with rig.telemetry.track_call(resolved, stage="draft") as call:
            call.usage = ask(stub)
            call.finish_reason = "stop"

    started, completed = rig.sink.events
    assert isinstance(started, CallStarted)
    assert isinstance(completed, CallCompleted)
    assert started.call_id == completed.call_id == call.call_id
    assert (completed.provider, completed.model, completed.origin) == ("openai", "gpt-4o", "user")
    assert completed.latency_ms >= 50
    assert completed.usage == Usage(11, 7, 18)
    assert completed.finish_reason == "stop"
    assert started.attrs == completed.attrs == {"stage": "draft"}
    assert call.done


def test_track_call_reports_an_exception_as_call_failed_and_reraises(build: BuildRig) -> None:
    rig = build()
    resolved = resolve(rig)

    with pytest.raises(ValueError, match="boom"), rig.telemetry.track_call(resolved):
        raise ValueError("boom")

    started, failed = rig.sink.events
    assert isinstance(started, CallStarted)
    assert isinstance(failed, CallFailed)
    assert failed.error_type == "ValueError"
    assert failed.error_message == "boom"
    assert failed.call_id == started.call_id
    assert (failed.error_kind, failed.status_code, failed.error_code, failed.usage) == (None, None, None, None)


async def test_track_call_reports_timeout_and_cancellation(build: BuildRig) -> None:
    rig = build()
    resolved = resolve(rig)

    with pytest.raises(TimeoutError), rig.telemetry.track_call(resolved):
        await asyncio.wait_for(asyncio.sleep(10), timeout=0.05)

    entered = asyncio.Event()

    async def cancelled_call() -> None:
        with rig.telemetry.track_call(resolved):
            entered.set()
            await asyncio.sleep(10)

    task = asyncio.create_task(cancelled_call())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [e.error_type for e in events_of(rig, CallFailed)] == ["TimeoutError", "CancelledError"]


def test_a_started_call_is_not_done_until_it_ends(build: BuildRig) -> None:
    rig = build()
    call = rig.telemetry.start_call(resolve(rig), call_id="my-call")

    assert (call.call_id, call.done) == ("my-call", False)
    assert [e.call_id for e in events_of(rig, CallStarted)] == ["my-call"]  # type: ignore[attr-defined]
    call.complete()
    assert call.done


# -- AC-05.2 -------------------------------------------------------------------------------------


def test_complete_from_another_thread_keeps_the_attrs_and_collector_captured_at_start(build: BuildRig) -> None:
    rig = build()
    resolved = resolve(rig)

    with rig.telemetry.collect() as collector:
        with rig.telemetry.scope(stage="a"):
            call = rig.telemetry.start_call(resolved)
        thread = threading.Thread(target=call.complete, args=(Usage(3, 4, 7), "stop"))  # a plain thread: no context
        thread.start()
        thread.join()

    (completed,) = events_of(rig, CallCompleted)
    assert completed.attrs["stage"] == "a"
    assert completed.usage == Usage(3, 4, 7)
    assert collector.summary()["call_count"] == 1
    assert collector.summary()["total_tokens"] == 7


def test_second_complete_or_fail_is_a_no_op(build: BuildRig) -> None:
    rig = build()
    resolved = resolve(rig)

    first = rig.telemetry.start_call(resolved)
    first.complete(Usage(1, 1, 2))
    first.complete(Usage(9, 9, 18))
    first.fail(RuntimeError("late"))
    second = rig.telemetry.start_call(resolved)
    second.fail(RuntimeError("first"))
    second.fail(RuntimeError("second"))
    second.complete()

    assert [e.usage for e in events_of(rig, CallCompleted)] == [Usage(1, 1, 2)]
    assert [e.error_message for e in events_of(rig, CallFailed)] == ["first"]


def test_a_second_end_racing_from_many_threads_emits_once(build: BuildRig) -> None:
    rig = build()
    call = rig.telemetry.start_call(resolve(rig))
    barrier = threading.Barrier(8)

    def end() -> None:
        barrier.wait()
        call.complete(Usage(1, 1, 2))

    threads = [threading.Thread(target=end) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(events_of(rig, CallCompleted)) == 1


def test_failed_call_keeps_the_usage_already_set(build: BuildRig) -> None:
    rig = build()
    resolved = resolve(rig)

    with (
        rig.telemetry.collect() as collector,
        pytest.raises(ValueError),
        rig.telemetry.track_call(resolved) as call,
    ):
        call.usage = Usage(3, 4, 7)
        raise ValueError("reply rejected")

    (failed,) = events_of(rig, CallFailed)
    assert failed.usage == Usage(3, 4, 7)
    assert collector.summary()["total_tokens"] == 7
    assert collector.summary()["call_count"] == 1  # counted once, not once here and once by the handle


def test_the_usage_set_on_the_handle_wins_over_none_but_an_argument_wins_over_the_handle(build: BuildRig) -> None:
    rig = build()
    resolved = resolve(rig)

    first = rig.telemetry.start_call(resolved)
    first.usage = Usage(1, 1, 2)
    first.finish_reason = "length"
    first.complete()
    second = rig.telemetry.start_call(resolved)
    second.usage = Usage(1, 1, 2)
    second.complete(Usage(5, 5, 10), "stop")

    one, two = events_of(rig, CallCompleted)
    assert (one.usage, one.finish_reason) == (Usage(1, 1, 2), "length")
    assert (two.usage, two.finish_reason) == (Usage(5, 5, 10), "stop")


# -- AC-05.3 -------------------------------------------------------------------------------------


def test_pending_calls_pair_by_key_and_count_unknown_ends(build: BuildRig) -> None:
    rig = build()
    resolved = resolve(rig)
    pending = PendingCalls(rig.telemetry)

    pending.start("a", resolved, stage="one")
    pending.start("b", resolved)
    assert pending.complete("a", Usage(1, 2, 3), "stop") is True
    assert pending.fail("b", RuntimeError("nope")) is True
    assert pending.unmatched == 0
    assert pending.complete("a") is False  # already ended
    assert pending.fail("never-started", RuntimeError("x")) is False

    assert pending.unmatched == 2
    (completed,) = events_of(rig, CallCompleted)
    assert (completed.usage, completed.finish_reason, completed.attrs["stage"]) == (Usage(1, 2, 3), "stop", "one")
    assert [e.error_message for e in events_of(rig, CallFailed)] == ["nope"]


def test_pending_calls_report_an_abandoned_call_once(build: BuildRig) -> None:
    rig = build()
    resolved = resolve(rig)
    pending = PendingCalls(rig.telemetry, ttl_seconds=0.01)

    pending.start("lost", resolved)
    time.sleep(0.05)
    pending.start("next", resolved)  # this start sweeps: frameworks never call sweep() themselves
    assert [e.error_type for e in events_of(rig, CallFailed)] == ["Abandoned"]
    pending.complete("next")  # ended at once, so only "lost" can be reported from here on
    time.sleep(0.05)
    pending.sweep()
    pending.sweep()

    abandoned = [e for e in events_of(rig, CallFailed) if e.error_type == "Abandoned"]
    assert len(abandoned) == 1
    assert abandoned[0].call_id == events_of(rig, CallStarted)[0].call_id  # type: ignore[attr-defined]
    assert pending.complete("lost") is False  # a late end for it is counted as unmatched
    assert pending.unmatched == 1


def test_pending_calls_do_not_report_a_call_that_ended_in_time(build: BuildRig) -> None:
    rig = build()
    resolved = resolve(rig)
    pending = PendingCalls(rig.telemetry, ttl_seconds=0.01)

    pending.start("a", resolved)
    pending.complete("a")
    time.sleep(0.05)
    pending.sweep()

    assert events_of(rig, CallFailed) == []


def test_starting_a_key_that_is_still_open_abandons_the_earlier_call(build: BuildRig) -> None:
    rig = build()
    resolved = resolve(rig)
    pending = PendingCalls(rig.telemetry)

    pending.start("k", resolved)
    pending.start("k", resolved)
    pending.complete("k")

    assert [e.error_type for e in events_of(rig, CallFailed)] == ["Abandoned"]
    assert len(events_of(rig, CallCompleted)) == 1


def test_failed_call_message_is_scrubbed_of_the_live_key(build: BuildRig) -> None:
    rig = build()
    resolved = resolve(rig)
    pending = PendingCalls(rig.telemetry)

    with pytest.raises(RuntimeError), rig.telemetry.track_call(resolved):
        raise RuntimeError(f"401: bad key {LIVE_KEY}")
    pending.start("p", resolved)
    pending.fail("p", RuntimeError(f"401: bad key {LIVE_KEY}"))
    handle = rig.telemetry.start_call(resolved)
    handle.fail(RuntimeError(f"401: bad key {LIVE_KEY}"))

    messages = [e.error_message for e in events_of(rig, CallFailed)]
    assert len(messages) == 3
    assert all(LIVE_KEY not in m and "401: bad key" in m for m in messages)
    assert LIVE_KEY not in repr(handle)


def test_a_handle_keeps_the_live_key_wrapped_while_the_call_is_open(build: BuildRig) -> None:
    rig = build()
    handle = rig.telemetry.start_call(resolve(rig))

    # A debugger or an error reporter that walks the handle's attributes must see no key.
    assert LIVE_KEY not in repr(vars(handle))
    handle.fail(RuntimeError(f"401: bad key {LIVE_KEY}"))
    assert LIVE_KEY not in repr(vars(handle))
    (failed,) = events_of(rig, CallFailed)
    assert LIVE_KEY not in failed.error_message


def test_a_call_with_no_key_has_nothing_to_scrub(build: BuildRig) -> None:
    rig = build()
    resolved = rig.service.resolve(rig.service.select("ollama"))  # no key at all: origin "none"
    assert resolved.api_key is None

    with pytest.raises(ValueError), rig.telemetry.track_call(resolved):
        raise ValueError("plain text")

    assert [e.error_message for e in events_of(rig, CallFailed)] == ["plain text"]
