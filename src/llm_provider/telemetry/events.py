"""Events a call emits (§Telemetry).

Every field is on a pinned whitelist: no prompt, reply or tool text, and no key. Error text is
scrubbed (``scrub``) by whoever builds the event.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Literal

from ..types import Usage

Origin = Literal["user", "server", "placeholder", "none"]


def _now() -> datetime:
    return datetime.now(UTC)


def _no_attrs() -> Mapping[str, Any]:
    return MappingProxyType({})


@dataclass(frozen=True, kw_only=True)
class _Event:
    provider: str
    model: str
    at: datetime = field(default_factory=_now)
    attrs: Mapping[str, Any] = field(default_factory=_no_attrs)


@dataclass(frozen=True, kw_only=True)
class CallStarted(_Event):
    call_id: str
    origin: Origin


@dataclass(frozen=True, kw_only=True)
class CallCompleted(_Event):
    call_id: str
    origin: Origin
    latency_ms: float
    usage: Usage | None
    finish_reason: str | None
    applied: tuple[str, ...] = ()  # what the compatibility layer did; filled from plan 10


@dataclass(frozen=True, kw_only=True)
class CallFailed(_Event):
    call_id: str
    origin: Origin
    latency_ms: float
    error_type: str
    error_message: str  # scrubbed
    error_kind: str | None = None  # filled when failures are classified (plan 07)
    status_code: int | None = None
    error_code: str | None = None
    usage: Usage | None = None  # set when the provider answered but the reply was rejected


@dataclass(frozen=True, kw_only=True)
class CredentialFallback(_Event):
    reason: str


LLMEvent = CallStarted | CallCompleted | CallFailed | CredentialFallback

_NAMES: dict[type, str] = {
    CallStarted: "llm_call_started",
    CallCompleted: "llm_call_completed",
    CallFailed: "llm_call_failed",
    CredentialFallback: "llm_credential_fallback",
}


def event_name(event: LLMEvent) -> str:
    return _NAMES[type(event)]
