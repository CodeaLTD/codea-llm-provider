"""``parse_retry_after``: how long a provider asks the caller to wait (§Resilience, ``Retry-After``).

The sources are read in this order, and the first that gives a number wins:

1. ``retry-after-ms`` (milliseconds);
2. ``Retry-After``, as delta-seconds or as an HTTP-date (a date in the past means 0);
3. Gemini's ``RetryInfo.retryDelay`` in the error body (``"30s"``);
4. OpenAI's ``x-ratelimit-reset-requests`` and ``x-ratelimit-reset-tokens`` durations (``"6m0s"``,
   ``"250ms"``): the larger.

Anything that can't be read gives ``None``, and the policy backs off instead.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_DURATION = re.compile(r"(?:\d+(?:\.\d+)?(?:ms|h|m|s))+")
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
_OPENAI_RESETS = ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")
_MAX_DEPTH = 6


def parse_retry_after(headers: Mapping[str, str] | None, body: Any = None, *, now: datetime) -> float | None:
    """Seconds to wait before retrying, or ``None`` when no source says. Header names match in any case."""
    lowered = {str(name).lower(): str(value) for name, value in (headers or {}).items()}

    seconds = _milliseconds(lowered.get("retry-after-ms"))
    if seconds is None:
        seconds = _retry_after_header(lowered.get("retry-after"), now)
    if seconds is None:
        seconds = _duration(_find_retry_delay(body, 0))
    if seconds is None:
        resets = [_duration(lowered.get(name)) for name in _OPENAI_RESETS]
        known = [value for value in resets if value is not None]
        seconds = max(known) if known else None
    return seconds


def _milliseconds(value: str | None) -> float | None:
    text = (value or "").strip()
    return float(text) / 1000.0 if _NUMBER.fullmatch(text) else None


def _retry_after_header(value: str | None, now: datetime) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    if _NUMBER.fullmatch(text):
        return float(text)
    try:
        moment = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return max(0.0, (moment - now).total_seconds())


def _duration(value: str | None) -> float | None:
    """``"30s"``, ``"250ms"``, ``"6m0s"``, ``"1h2m3.5s"`` in seconds."""
    text = (value or "").strip()
    if not _DURATION.fullmatch(text):
        return None
    return sum(float(amount) * _UNIT_SECONDS[unit] for amount, unit in _DURATION_PART.findall(text))


def _find_retry_delay(node: Any, depth: int) -> str | None:
    """The first ``retryDelay`` string in an error body, wherever Google nests it (``error.details[]``)."""
    if depth > _MAX_DEPTH:
        return None
    if isinstance(node, Mapping):
        delay = node.get("retryDelay")
        if isinstance(delay, str):
            return delay
        children: Any = node.values()
    elif isinstance(node, list | tuple):
        children = node
    else:
        return None
    for child in children:
        found = _find_retry_delay(child, depth + 1)
        if found is not None:
            return found
    return None
