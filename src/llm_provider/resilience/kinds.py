"""``ErrorKind``: what went wrong with a call, in the words the retry and failover rules use (§Resilience)."""

from __future__ import annotations

from enum import StrEnum


class ErrorKind(StrEnum):
    """The twelve kinds a provider failure is sorted into. ``invalid_reply`` joins them in plan 10."""

    rate_limited = "rate_limited"
    quota_exhausted = "quota_exhausted"
    overloaded = "overloaded"
    server_error = "server_error"
    timeout = "timeout"
    connection = "connection"
    context_overflow = "context_overflow"
    content_blocked = "content_blocked"
    model_unavailable = "model_unavailable"
    auth = "auth"
    bad_request = "bad_request"
    unknown = "unknown"
