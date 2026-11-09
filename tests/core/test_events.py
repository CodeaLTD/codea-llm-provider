"""Events are frozen values with a pinned set of fields (Task 3 Step 1; GC Telemetry)."""

from __future__ import annotations

import dataclasses
from datetime import UTC

import pytest

from llm_provider import Usage
from llm_provider.telemetry.events import (
    CallCompleted,
    CallFailed,
    CallStarted,
    CredentialFallback,
    event_name,
)

BASE = {"at", "provider", "model", "attrs"}
EXPECTED_FIELDS = {
    CallStarted: BASE | {"call_id", "origin"},
    CallCompleted: BASE | {"call_id", "origin", "latency_ms", "usage", "finish_reason", "applied"},
    CallFailed: BASE
    | {
        "call_id",
        "origin",
        "latency_ms",
        "error_type",
        "error_message",
        "error_kind",
        "status_code",
        "error_code",
        "usage",
    },
    CredentialFallback: BASE | {"reason"},
}
TEXT_CARRYING_NAMES = {"messages", "prompt", "response", "content", "api_key"}


def test_event_fields_are_a_whitelist() -> None:
    for cls, expected in EXPECTED_FIELDS.items():
        names = {f.name for f in dataclasses.fields(cls)}
        assert names == expected, cls.__name__
        assert not names & TEXT_CARRYING_NAMES, cls.__name__


def test_events_are_frozen_and_stamped_in_utc() -> None:
    event = CallStarted(provider="p", model="m", call_id="c1", origin="user")

    assert event.at.tzinfo is UTC
    assert dict(event.attrs) == {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.model = "other"  # type: ignore[misc]


def test_call_completed_defaults_applied_to_empty_and_failed_to_unclassified() -> None:
    done = CallCompleted(
        provider="p",
        model="m",
        call_id="c1",
        origin="server",
        latency_ms=12.5,
        usage=Usage(1, 2, 3),
        finish_reason="stop",
    )
    failed = CallFailed(
        provider="p",
        model="m",
        call_id="c1",
        origin="server",
        latency_ms=1.0,
        error_type="TimeoutError",
        error_message="timed out",
    )

    assert done.applied == ()
    assert (failed.error_kind, failed.status_code, failed.error_code, failed.usage) == (None, None, None, None)


def test_event_name_for_each_event() -> None:
    started = CallStarted(provider="p", model="m", call_id="c", origin="none")
    done = CallCompleted(
        provider="p", model="m", call_id="c", origin="none", latency_ms=0.0, usage=None, finish_reason=None
    )
    failed = CallFailed(
        provider="p", model="m", call_id="c", origin="none", latency_ms=0.0, error_type="E", error_message=""
    )
    fallback = CredentialFallback(provider="p", model="m", reason="user_key_unavailable")

    assert [event_name(e) for e in (started, done, failed, fallback)] == [
        "llm_call_started",
        "llm_call_completed",
        "llm_call_failed",
        "llm_credential_fallback",
    ]
