"""``parse_retry_after``: every vendor's wait hint in one number of seconds (Task 5a Step 2; AC-07.2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from llm_provider.resilience.retry_after import parse_retry_after

NOW = datetime(2026, 11, 12, 10, 0, 0, tzinfo=UTC)


def http_date(moment: datetime) -> str:
    return format_datetime(moment, usegmt=True)


def test_retry_after_delta_seconds() -> None:
    assert parse_retry_after({"Retry-After": "120"}, now=NOW) == 120.0


def test_retry_after_http_date_in_the_future_and_past() -> None:
    ahead = http_date(NOW + timedelta(seconds=90))
    behind = http_date(NOW - timedelta(seconds=90))

    assert parse_retry_after({"Retry-After": ahead}, now=NOW) == 90.0
    assert parse_retry_after({"Retry-After": behind}, now=NOW) == 0.0


def test_retry_after_ms_wins_over_retry_after() -> None:
    headers = {"Retry-After": "120", "retry-after-ms": "250"}

    assert parse_retry_after(headers, now=NOW) == 0.25


def test_gemini_retry_delay_from_the_error_body() -> None:
    body = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": []},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "30s"},
            ],
        }
    }

    assert parse_retry_after({}, body, now=NOW) == 30.0
    assert parse_retry_after(None, body, now=NOW) == 30.0


def test_openai_reset_durations_take_the_larger() -> None:
    headers = {"x-ratelimit-reset-requests": "250ms", "x-ratelimit-reset-tokens": "6m0s"}

    assert parse_retry_after(headers, now=NOW) == 360.0
    assert parse_retry_after({"x-ratelimit-reset-requests": "250ms"}, now=NOW) == 0.25
    assert parse_retry_after({"x-ratelimit-reset-tokens": "1h2m3.5s"}, now=NOW) == 3723.5


@pytest.mark.parametrize("value", ["soon", "-5", "", "  ", "1e3", "nan"])
def test_unparseable_values_give_none(value: str) -> None:
    assert parse_retry_after({"Retry-After": value}, now=NOW) is None
    assert parse_retry_after({"retry-after-ms": value}, now=NOW) is None
    assert parse_retry_after({"x-ratelimit-reset-tokens": value}, now=NOW) is None
    assert parse_retry_after({}, {"error": {"details": [{"retryDelay": value}]}}, now=NOW) is None


def test_an_unusable_value_falls_through_to_the_next_source() -> None:
    headers = {"retry-after-ms": "soon", "Retry-After": "bad", "x-ratelimit-reset-tokens": "2s"}

    assert parse_retry_after(headers, now=NOW) == 2.0


@pytest.mark.parametrize("name", ["Retry-After", "retry-after", "RETRY-AFTER"])
def test_header_names_are_case_insensitive(name: str) -> None:
    assert parse_retry_after({name: "7"}, now=NOW) == 7.0
    assert parse_retry_after({f"{name}-ms": "500"}, now=NOW) == 0.5
    assert parse_retry_after({"X-RateLimit-Reset-Tokens": "9s"}, now=NOW) == 9.0


def test_no_hint_gives_none() -> None:
    assert parse_retry_after(None, now=NOW) is None
    assert parse_retry_after({}, now=NOW) is None
    assert parse_retry_after({}, {"error": {"message": "nothing here"}}, now=NOW) is None
    assert parse_retry_after({}, "not a dict", now=NOW) is None
