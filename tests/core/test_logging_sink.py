"""``LoggingSink`` writes one flat log record per event (Task 3 Step 1; §Telemetry)."""

from __future__ import annotations

import logging

import pytest

from llm_provider import Usage
from llm_provider.telemetry import (
    CallCompleted,
    CallFailed,
    LoggingSink,
    Telemetry,
    event_name,
    scrub,
)

LOGGER = "llm_provider.telemetry"
KEY = "sk-abc12345-live-key"


def completed(**overrides: object) -> CallCompleted:
    fields: dict[str, object] = {
        "provider": "openai",
        "model": "gpt-x",
        "call_id": "c1",
        "origin": "user",
        "latency_ms": 42.5,
        "usage": Usage(10, 5, 15, estimated=True),
        "finish_reason": "stop",
    }
    return CallCompleted(**(fields | overrides))  # type: ignore[arg-type]


def test_logging_sink_emits_flat_fields(caplog: pytest.LogCaptureFixture) -> None:
    t = Telemetry([LoggingSink()])

    with caplog.at_level(logging.INFO, logger=LOGGER):
        t.emit(completed(attrs={"job": "j7"}))

    (record,) = [r for r in caplog.records if r.name == LOGGER]
    assert record.getMessage() == "llm_call_completed"
    assert record.levelno == logging.INFO
    assert record.provider == "openai"  # type: ignore[attr-defined]
    assert record.model == "gpt-x"  # type: ignore[attr-defined]
    assert record.latency_ms == 42.5  # type: ignore[attr-defined]
    assert record.prompt_tokens == 10  # type: ignore[attr-defined]
    assert record.completion_tokens == 5  # type: ignore[attr-defined]
    assert record.total_tokens == 15  # type: ignore[attr-defined]
    assert record.usage_estimated is True  # type: ignore[attr-defined]
    assert record.job == "j7"  # type: ignore[attr-defined]


def test_a_key_in_an_error_reaches_the_log_only_in_scrubbed_form(caplog: pytest.LogCaptureFixture) -> None:
    """The builder of a ``CallFailed`` scrubs its message; the record then holds no copy of the key."""
    event = CallFailed(
        provider="openai",
        model="gpt-x",
        call_id="c1",
        origin="user",
        latency_ms=3.0,
        error_type="AuthError",
        error_message=scrub(f"401: invalid key {KEY}", [KEY]),
    )
    t = Telemetry([LoggingSink()])

    with caplog.at_level(logging.INFO, logger=LOGGER):
        t.emit(event)

    (record,) = [r for r in caplog.records if r.name == LOGGER]
    assert record.error_message == "401: invalid key ***"  # type: ignore[attr-defined]
    assert KEY not in repr(record.__dict__)
    assert "abc12345" not in repr(record.__dict__)


def test_logging_sink_leaves_out_usage_fields_when_there_is_no_usage(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER):
        LoggingSink().emit(completed(usage=None))

    (record,) = [r for r in caplog.records if r.name == LOGGER]
    assert not hasattr(record, "total_tokens")
    assert not hasattr(record, "usage_estimated")


def test_logging_sink_logs_a_failure_with_its_scrubbed_message(caplog: pytest.LogCaptureFixture) -> None:
    event = CallFailed(
        provider="openai",
        model="gpt-x",
        call_id="c1",
        origin="user",
        latency_ms=3.0,
        error_type="AuthError",
        error_message="rejected ***",
        usage=Usage(2, 0, 2),
    )

    with caplog.at_level(logging.INFO, logger=LOGGER):
        LoggingSink().emit(event)

    (record,) = [r for r in caplog.records if r.name == LOGGER]
    assert record.getMessage() == event_name(event)
    assert record.error_type == "AuthError"  # type: ignore[attr-defined]
    assert record.error_message == "rejected ***"  # type: ignore[attr-defined]
    assert record.total_tokens == 2  # type: ignore[attr-defined]


def test_logging_sink_renames_attrs_that_collide_with_logrecord(caplog: pytest.LogCaptureFixture) -> None:
    t = Telemetry([LoggingSink()])

    with caplog.at_level(logging.INFO, logger=LOGGER), t.scope(name="x", message="m", provider="mine", free="ok"):
        t.emit(completed(attrs=t.current_attrs()))  # a LogRecord field would raise KeyError if passed through

    (record,) = [r for r in caplog.records if r.name == LOGGER]
    assert record.attr_name == "x"  # type: ignore[attr-defined]
    assert record.attr_message == "m"  # type: ignore[attr-defined]
    assert record.attr_provider == "mine"  # type: ignore[attr-defined]
    assert record.provider == "openai"  # type: ignore[attr-defined]  # the event's own field is kept
    assert record.name == LOGGER  # the record's own name is untouched
    assert record.free == "ok"  # type: ignore[attr-defined]


def test_logging_sink_uses_its_own_logger_and_level(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        LoggingSink("my.app.llm", level=logging.DEBUG).emit(completed())

    assert [(r.name, r.levelno) for r in caplog.records] == [("my.app.llm", logging.DEBUG)]
