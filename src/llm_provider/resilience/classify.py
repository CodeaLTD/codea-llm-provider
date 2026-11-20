"""``classify_error``: sort any exception from a model call into an ``ErrorKind`` (§Resilience).

It works by duck typing, because the core imports no SDK: the status comes from ``status_code``,
``status``, ``response.status_code`` or a numeric ``code``; the headers from ``response.headers`` or
``headers``; the codes and text from the error's body (OpenAI's and Anthropic's ``body``, Google's
``details``, or ``response.json()``) and from its own attributes. A classifier a host or an
integration passes in runs first, for the errors its framework wraps.

What each vendor documents, as read on 2026-11-12 (Task 5a Step 1; the table in *Design →
Resilience* follows it):

- OpenAI: https://developers.openai.com/api/docs/guides/error-codes and
  https://developers.openai.com/api/docs/guides/rate-limits. A 429 is a rate limit, or billing
  (``insufficient_quota``; the page also lists ``credit_balance_exhausted`` and the spend-limit
  codes); 503 is overload (``server_is_overloaded``); ``Retry-After`` comes in seconds and
  ``x-ratelimit-reset-requests`` / ``x-ratelimit-reset-tokens`` in durations (``"6m0s"``).
- Anthropic: https://platform.claude.com/docs/en/api/errors and
  https://platform.claude.com/docs/en/api/rate-limits. ``invalid_request_error`` 400,
  ``authentication_error`` 401, ``billing_error`` 402, ``permission_error`` 403, ``not_found_error``
  404, ``request_too_large`` 413, ``rate_limit_error`` 429, ``api_error`` 500, ``timeout_error`` 504,
  ``overloaded_error`` 529; ``retry-after`` is whole seconds and the ``anthropic-ratelimit-*-reset``
  headers are RFC 3339 (not read here: ``Retry-After`` is the hint).
- Gemini: https://aistudio.google.com/docs/troubleshooting, https://aistudio.google.com/docs/api-errors
  and https://aistudio.google.com/api-reference/generate-content (the ``finishReason`` and
  ``promptFeedback.blockReason`` lists). 429 ``RESOURCE_EXHAUSTED`` is a per-minute limit or a daily
  quota; the daily one shows as a ``QuotaFailure`` violation whose ``quotaId`` ends in ``PerDay``.

Not found on those pages, and so kept from the source and the stub until a person checks them: the
wording of each vendor's overflow message, OpenAI's ``context_length_exceeded`` and
``content_policy_violation`` codes, Anthropic's ``refusal`` stop reason, and Gemini's ``retryDelay``
format. See the report on plan 07 (AC-07.1).

Phrase matching is a case-insensitive substring match against a short list per kind. It reads the
full text in memory only: the text of ``bad_request``, ``unknown`` and ``context_overflow`` errors
never leaves this module (Decision 20).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..telemetry.scrub import scrub
from .kinds import ErrorKind
from .retry_after import parse_retry_after

_CODE = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_WITHHELD = frozenset({ErrorKind.bad_request, ErrorKind.unknown, ErrorKind.context_overflow})
_MAX_DEPTH = 6

_QUOTA_WORDS = frozenset(
    {
        "insufficient_quota",
        "quota_exceeded",
        "credit_balance_exhausted",
        "billing_error",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
    }
)
_QUOTA_PHRASES = ("credit balance is too low", "reached your specified api usage limits")
_DAILY_PHRASES = ("per day", "daily quota")
_RATE_WORDS = frozenset({"rate_limit_error", "rate_limit_exceeded", "too_many_requests"})
_OVERLOAD_WORDS = frozenset({"overloaded_error", "server_is_overloaded"})
_OVERFLOW_WORDS = frozenset({"context_length_exceeded", "exceed_context_size_error", "request_too_large"})
_OVERFLOW_PHRASES = (
    "prompt is too long",
    "maximum context length",
    "maximum number of tokens allowed",
    "exceeds the available context size",
)
_BLOCKED_WORDS = frozenset({"content_filter", "content_policy_violation"})
_BLOCKED_PHRASES = ("content filtering policy",)
_BLOCKED_FINISH_REASONS = frozenset(
    {
        "content_filter",
        "refusal",
        "safety",
        "blocklist",
        "prohibited_content",
        "spii",
        "image_safety",
        "image_prohibited_content",
    }
)


@dataclass(frozen=True)
class ErrorInfo:
    """One failure, classified. ``message`` is scrubbed, and withheld for the kinds Decision 20 names."""

    kind: ErrorKind
    status_code: int | None
    error_code: str | None
    retry_after_seconds: float | None
    message: str


Classifier = Callable[[BaseException], ErrorInfo | None]


def classify_error(
    error: BaseException,
    *,
    secrets: Iterable[str] = (),
    classifiers: Sequence[Classifier] = (),
    now: datetime | None = None,
) -> ErrorInfo:
    """The kind, status, code, wait hint and safe message of ``error``.

    ``classifiers`` run first, in order, and the first that returns an ``ErrorInfo`` wins; its message
    then gets the same treatment as ours. ``now`` fixes the clock for an HTTP-date ``Retry-After``.
    """
    live = tuple(secrets)
    for classifier in classifiers:
        info = classifier(error)
        if info is not None:
            return _safe(info, error, live)
    return _classify(error, live, now if now is not None else datetime.now(UTC))


def kind_for_finish_reason(reason: str | None) -> ErrorKind | None:
    """``content_blocked`` for a finish reason that says the output was filtered or refused, else ``None``."""
    if reason and reason.lower() in _BLOCKED_FINISH_REASONS:
        return ErrorKind.content_blocked
    return None


# -- classification -----------------------------------------------------------------------------


def _classify(error: BaseException, secrets: tuple[str, ...], now: datetime) -> ErrorInfo:
    response = _attr(error, "response")
    body = _body_of(error, response)
    sources = _sources(error, body)
    status = _status_of(error, response, sources)
    code = _clean_code(_first_text(sources, ("code", "type", "status")))
    words = _words(sources)
    text = " ".join([str(error), *_messages(sources)]).lower()
    kind = _kind(error, status, words, text, body)
    headers = _headers_of(error, response)
    return _build(error, kind, status, code, parse_retry_after(headers, body, now=now), secrets)


def _safe(info: ErrorInfo, error: BaseException, secrets: tuple[str, ...]) -> ErrorInfo:
    """A classifier's result with a safe message and a valid code."""
    return _build(
        error,
        info.kind,
        info.status_code,
        _clean_code(info.error_code),
        info.retry_after_seconds,
        secrets,
        info.message,
    )


def _build(
    error: BaseException,
    kind: ErrorKind,
    status: int | None,
    code: str | None,
    retry_after: float | None,
    secrets: tuple[str, ...],
    text: str | None = None,
) -> ErrorInfo:
    if kind in _WITHHELD:
        message = _neutral(kind, status, code)
    else:
        message = scrub(text if text is not None else (str(error) or type(error).__name__), secrets)
    return ErrorInfo(kind, status, code, retry_after, message)


def _neutral(kind: ErrorKind, status: int | None, code: str | None) -> str:
    parts = [f"HTTP {status}"] if status is not None else []
    if code:
        parts.append(f"code {code}")
    head = f"{kind.value}: {', '.join(parts)}" if parts else kind.value
    return f"{head} (provider text withheld; see __cause__)"


def _kind(error: BaseException, status: int | None, words: frozenset[str], text: str, body: Any) -> ErrorKind:
    if status is None:
        by_type = _kind_by_type(error)
        if by_type is not None:
            return by_type
    if status in (401, 403):
        return ErrorKind.auth
    if status == 408:
        return ErrorKind.timeout
    if status == 402 or _is_quota(status, words, text, body):
        return ErrorKind.quota_exhausted
    if status == 429 or (status is None and words & _RATE_WORDS):
        return ErrorKind.rate_limited
    if status in (503, 529) or words & _OVERLOAD_WORDS:
        return ErrorKind.overloaded
    if status is not None and status >= 500:
        return ErrorKind.server_error
    if status in (None, 400, 413):
        if status == 413 or words & _OVERFLOW_WORDS or _has_any(text, _OVERFLOW_PHRASES):
            return ErrorKind.context_overflow
        if words & _BLOCKED_WORDS or _has_any(text, _BLOCKED_PHRASES):
            return ErrorKind.content_blocked
    if status == 404 or "model_not_found" in words or _model_retired(text):
        return ErrorKind.model_unavailable
    if status is not None and 400 <= status < 500:
        return ErrorKind.bad_request
    return ErrorKind.unknown


def _kind_by_type(error: BaseException) -> ErrorKind | None:
    """Timeouts and connection failures carry no status: tell them by their class names."""
    names = [cls.__name__.lower() for cls in type(error).__mro__]
    if any("timeout" in name for name in names):  # before "connect": APITimeoutError is an APIConnectionError
        return ErrorKind.timeout
    if any("connect" in name for name in names):
        return ErrorKind.connection
    return None


def _is_quota(status: int | None, words: frozenset[str], text: str, body: Any) -> bool:
    if words & _QUOTA_WORDS or _has_any(text, _QUOTA_PHRASES):
        return True
    gemini_limit = status == 429 or "resource_exhausted" in words
    if not gemini_limit:
        return False
    return _has_any(text, _DAILY_PHRASES) or any("perday" in leaf for leaf in _leaves(body, 0))


def _model_retired(text: str) -> bool:
    return "model" in text and ("decommissioned" in text or "deprecated" in text)


def _has_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


# -- reading the error ---------------------------------------------------------------------------


def _attr(source: Any, name: str) -> Any:
    try:
        return getattr(source, name, None)
    except Exception:  # noqa: BLE001  a property that raises is as good as no attribute
        return None


def _body_of(error: BaseException, response: Any) -> Any:
    for candidate in (_attr(error, "body"), _attr(error, "details")):
        if isinstance(candidate, Mapping):
            return candidate
    reader = _attr(response, "json")
    if callable(reader):
        try:
            candidate = reader()
        except Exception:  # noqa: BLE001  a body that won't parse gives no body
            return None
        if isinstance(candidate, Mapping):
            return candidate
    return None


def _sources(error: BaseException, body: Any) -> list[Any]:
    """Where codes and text live, most specific first: ``body["error"]``, the body, the error itself."""
    found: list[Any] = []
    if isinstance(body, Mapping):
        inner = body.get("error")
        if isinstance(inner, Mapping):
            found.append(inner)
        found.append(body)
    found.append(error)
    return found


def _get(source: Any, name: str) -> Any:
    return source.get(name) if isinstance(source, Mapping) else _attr(source, name)


def _first_text(sources: Sequence[Any], names: Sequence[str]) -> str | None:
    for name in names:
        for source in sources:
            value = _get(source, name)
            if isinstance(value, str) and value:
                return value
    return None


def _words(sources: Sequence[Any]) -> frozenset[str]:
    found = (_get(source, name) for source in sources for name in ("code", "type", "status"))
    return frozenset(value.lower() for value in found if isinstance(value, str))


def _messages(sources: Sequence[Any]) -> list[str]:
    found = (_get(source, "message") for source in sources)
    return [value for value in found if isinstance(value, str)]


def _is_http_status(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599


def _status_of(error: BaseException, response: Any, sources: Sequence[Any]) -> int | None:
    candidates = (
        _attr(error, "status_code"),
        _attr(error, "status"),
        _attr(response, "status_code"),
        _attr(error, "code"),
        *(_get(source, "code") for source in sources),
    )
    return next((value for value in candidates if _is_http_status(value)), None)


def _headers_of(error: BaseException, response: Any) -> Mapping[str, str] | None:
    for holder in (response, error):
        headers = _attr(holder, "headers")
        if headers is not None and hasattr(headers, "items") and len(headers):
            return headers  # type: ignore[no-any-return]
    return None


def _clean_code(code: str | None) -> str | None:
    """Keep a code only if it is a short identifier: some servers put free text there."""
    return code if code is not None and _CODE.fullmatch(code) else None


def _leaves(node: Any, depth: int) -> Iterable[str]:
    """Every string in a nested body, lower-cased."""
    if depth > _MAX_DEPTH:
        return
    if isinstance(node, str):
        yield node.lower()
    elif isinstance(node, Mapping):
        for value in node.values():
            yield from _leaves(value, depth + 1)
    elif isinstance(node, list | tuple):
        for value in node:
            yield from _leaves(value, depth + 1)
