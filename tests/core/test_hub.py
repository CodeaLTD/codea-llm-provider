"""The telemetry hub: sinks, scopes and usage collection (Task 3 Step 1; §Telemetry)."""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from llm_provider import Usage
from llm_provider.telemetry import (
    CallCompleted,
    CallFailed,
    CallStarted,
    LLMEvent,
    Telemetry,
)
from llm_provider.testing import RecordingSink

LOGGER = "llm_provider.telemetry"


def started(call_id: str = "c1") -> CallStarted:
    return CallStarted(provider="p", model="m", call_id=call_id, origin="server")


def completed(usage: Usage | None) -> CallCompleted:
    return CallCompleted(
        provider="p", model="m", call_id="c", origin="server", latency_ms=1.0, usage=usage, finish_reason="stop"
    )


def failed(usage: Usage | None = None) -> CallFailed:
    return CallFailed(
        provider="p",
        model="m",
        call_id="c",
        origin="server",
        latency_ms=1.0,
        error_type="InvalidReply",
        error_message="rejected",
        usage=usage,
    )


class RaisingSink:
    def emit(self, event: LLMEvent) -> None:
        raise RuntimeError("sink is broken")


class OrderSink:
    """Notes its name in a shared list, then records the event."""

    def __init__(self, name: str, order: list[str]) -> None:
        self.name, self.order = name, order
        self.events: list[LLMEvent] = []

    def emit(self, event: LLMEvent) -> None:
        self.order.append(self.name)
        self.events.append(event)


class RemovingSink:
    """On its first event, removes another sink from the hub."""

    def __init__(self, hub: Telemetry, victim: RecordingSink) -> None:
        self.hub, self.victim = hub, victim

    def emit(self, event: LLMEvent) -> None:
        self.hub.remove_sink(self.victim)


def test_emit_reaches_every_sink_in_order() -> None:
    order: list[str] = []
    first, second = OrderSink("first", order), OrderSink("second", order)
    t = Telemetry([first, second])
    event = started()

    t.emit(event)

    assert order == ["first", "second"]
    assert first.events == [event]
    assert second.events == [event]


def test_a_sink_with_only_a_call_method_is_not_a_sink() -> None:
    """The contract is ``emit(event)`` (decision.md, Protocols); a bare callable receives nothing."""
    seen: list[LLMEvent] = []
    t = Telemetry([seen.append])  # type: ignore[list-item]

    t.emit(started())

    assert seen == []
    assert t.dropped_sink_errors == 1


def test_a_raising_sink_is_swallowed_counted_logged_once_and_the_next_sink_still_receives(
    caplog: pytest.LogCaptureFixture,
) -> None:
    after = RecordingSink()
    t = Telemetry([RaisingSink(), after])

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        for i in range(3):
            t.emit(started(f"c{i}"))

    assert t.dropped_sink_errors == 3
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == LOGGER]
    assert len(warnings) == 1
    assert "sink is broken" not in warnings[0].getMessage()  # exception text may hold a secret
    assert [e.call_id for e in after.events] == ["c0", "c1", "c2"]  # type: ignore[union-attr]


def test_a_raising_sink_is_logged_again_on_every_100th_failure(caplog: pytest.LogCaptureFixture) -> None:
    t = Telemetry([RaisingSink()], logger_name="custom.telemetry")

    with caplog.at_level(logging.DEBUG, logger="custom.telemetry"):
        for _ in range(250):
            t.emit(started())

    assert t.dropped_sink_errors == 250
    assert len([r for r in caplog.records if r.name == "custom.telemetry"]) == 3  # failures 1, 100 and 200


def test_a_sink_removed_during_an_emit_still_receives_that_event() -> None:
    """Copy-on-write: ``emit`` works on the sink tuple it started with."""
    victim = RecordingSink()
    t = Telemetry()
    t.add_sink(RemovingSink(t, victim))
    t.add_sink(victim)

    t.emit(started("c1"))
    t.emit(started("c2"))

    assert [e.call_id for e in victim.events] == ["c1"]  # type: ignore[union-attr]


def test_a_sink_added_during_an_emit_waits_for_the_next_event() -> None:
    late = RecordingSink()

    class AddingSink:
        def emit(self, event: LLMEvent) -> None:
            t.add_sink(late)

    t = Telemetry([AddingSink()])

    t.emit(started("c1"))
    assert late.events == []
    t.emit(started("c2"))

    assert [e.call_id for e in late.events] == ["c2"]  # type: ignore[union-attr]


def test_add_and_remove_sink_while_another_thread_emits() -> None:
    steady = RecordingSink()
    t = Telemetry([steady])
    errors: list[BaseException] = []

    def emitter() -> None:
        try:
            for _ in range(1000):
                t.emit(started())
        except BaseException as exc:  # noqa: BLE001 - the test reports whatever escapes
            errors.append(exc)

    thread = threading.Thread(target=emitter)
    thread.start()
    while thread.is_alive():
        extra = RecordingSink()
        t.add_sink(extra)
        t.remove_sink(extra)
    thread.join()

    assert errors == []
    assert len(steady.events) == 1000  # the sink that stayed attached missed nothing


def test_remove_sink_stops_delivery_and_ignores_an_unknown_sink() -> None:
    sink = RecordingSink()
    t = Telemetry([sink])

    t.remove_sink(sink)
    t.remove_sink(sink)
    t.emit(started())

    assert sink.events == []


async def test_scopes_are_isolated_under_asyncio_gather() -> None:
    t = Telemetry()

    async def worker(name: str, delay: float) -> dict[str, object]:
        with t.scope(worker=name):
            await asyncio.sleep(delay)
            return t.current_attrs()

    a, b = await asyncio.gather(worker("a", 0.02), worker("b", 0.01))

    assert a == {"worker": "a"}
    assert b == {"worker": "b"}
    assert t.current_attrs() == {}


async def test_scope_is_visible_inside_asyncio_to_thread() -> None:
    t = Telemetry()

    with t.scope(request="r1"):
        seen = await asyncio.to_thread(t.current_attrs)

    assert seen == {"request": "r1"}


def test_nested_scopes_merge_inner_wins() -> None:
    t = Telemetry()

    with t.scope(a=1, k="outer"):
        with t.scope(b=2, k="inner"):
            assert t.current_attrs() == {"a": 1, "b": 2, "k": "inner"}
        assert t.current_attrs() == {"a": 1, "k": "outer"}
    assert t.current_attrs() == {}


def test_scope_resets_after_an_exception() -> None:
    t = Telemetry()

    with pytest.raises(ValueError), t.scope(a=1):
        raise ValueError("boom")

    assert t.current_attrs() == {}


def test_current_attrs_merges_static_then_scope_then_extra() -> None:
    t = Telemetry(static_attrs={"service": "svc", "k": "static"})

    with t.scope(k="scope", s=1):
        assert t.current_attrs(k="extra") == {"service": "svc", "k": "extra", "s": 1}
        assert t.current_attrs() == {"service": "svc", "k": "scope", "s": 1}


def test_collect_summary_has_the_exact_shape() -> None:
    t = Telemetry()

    with t.collect() as collector:
        t.emit(completed(Usage(3, 4, 7)))
        t.emit(completed(Usage()))
        t.emit(completed(Usage(1, 1, 2, estimated=True)))

    assert collector.summary() == {
        "prompt_tokens": 4,
        "completion_tokens": 5,
        "total_tokens": 9,
        "call_count": 3,
        "estimated_call_count": 1,
    }


def test_collect_counts_a_failed_call_only_when_it_carries_usage() -> None:
    t = Telemetry()

    with t.collect() as collector:
        t.emit(failed(Usage(5, 0, 5)))
        t.emit(failed())
        t.emit(completed(None))
        t.emit(started())

    assert collector.calls == [Usage(5, 0, 5), Usage()]  # a completed call without usage still counts, as zero
    assert collector.summary()["call_count"] == 2


def test_collect_ignores_events_outside_the_block_and_in_other_hubs() -> None:
    t, other = Telemetry(), Telemetry()

    t.emit(completed(Usage(1, 1, 2)))
    with t.collect() as collector:
        other.emit(completed(Usage(1, 1, 2)))
    t.emit(completed(Usage(1, 1, 2)))

    assert collector.calls == []
    assert collector.summary() == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "call_count": 0,
        "estimated_call_count": 0,
    }
