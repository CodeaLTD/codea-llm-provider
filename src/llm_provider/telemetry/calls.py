"""``CallHandle`` and ``PendingCalls``: one model call from its start to its end (§Telemetry).

A handle captures everything it needs at the start of the call (the attrs, the active usage
collector, the live key), so the call can end on another thread or context and still be reported
correctly.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..types import ResolvedLLM, Usage
from .events import CallCompleted, CallFailed, CallStarted
from .scrub import scrub

if TYPE_CHECKING:
    from ..secret import Secret
    from .hub import Telemetry, UsageCollector


class Abandoned(Exception):
    """A call whose end never arrived. Its name is the ``error_type`` of the ``CallFailed`` that reports it."""


class CallHandle:
    """One call in flight. ``complete`` or ``fail`` ends it: the first one wins, later ones do nothing.

    Set ``usage`` and ``finish_reason`` as the reply arrives. A ``fail`` keeps any ``usage`` already
    set, because the provider billed those tokens. Safe to end from any thread.
    """

    def __init__(
        self,
        telemetry: Telemetry,
        resolved: ResolvedLLM,
        *,
        call_id: str,
        attrs: Mapping[str, Any],
        collector: UsageCollector | None,
    ) -> None:
        self._telemetry = telemetry
        self._provider = resolved.provider
        self._model = resolved.model
        self._origin = resolved.origin
        self._attrs = attrs
        self._collector = collector
        self._key: Secret | None = resolved.api_key  # kept wrapped; revealed only to scrub a failure
        self._started = time.perf_counter()
        self._lock = threading.Lock()
        self._done = False
        self.call_id = call_id
        self.usage: Usage | None = None
        self.finish_reason: str | None = None

    @property
    def done(self) -> bool:
        return self._done

    def complete(self, usage: Usage | None = None, finish_reason: str | None = None) -> None:
        """Report the call as finished. An argument left out falls back to the value set on the handle."""
        if not self._finish():
            return
        self._telemetry._emit_to(
            CallCompleted(
                provider=self._provider,
                model=self._model,
                attrs=self._attrs,
                call_id=self.call_id,
                origin=self._origin,
                latency_ms=self._elapsed_seconds() * 1000.0,
                usage=usage if usage is not None else self.usage,
                finish_reason=finish_reason if finish_reason is not None else self.finish_reason,
            ),
            self._collector,
        )

    def fail(self, error: BaseException) -> None:
        """Report the call as failed, with the error's type and its scrubbed message."""
        if not self._finish():
            return
        secrets = (self._key.reveal(),) if self._key is not None else ()
        self._telemetry._emit_to(
            CallFailed(
                provider=self._provider,
                model=self._model,
                attrs=self._attrs,
                call_id=self.call_id,
                origin=self._origin,
                latency_ms=self._elapsed_seconds() * 1000.0,
                error_type=type(error).__name__,
                error_message=scrub(str(error), secrets),
                usage=self.usage,
            ),
            self._collector,
        )

    def _emit_started(self) -> None:
        self._telemetry._emit_to(
            CallStarted(
                provider=self._provider,
                model=self._model,
                attrs=self._attrs,
                call_id=self.call_id,
                origin=self._origin,
            ),
            self._collector,
        )

    def _finish(self) -> bool:
        """True for the one caller that ends the call."""
        with self._lock:
            if self._done:
                return False
            self._done = True
            return True

    def _elapsed_seconds(self) -> float:
        return time.perf_counter() - self._started


def new_call_id() -> str:
    return uuid.uuid4().hex


class PendingCalls:
    """Pairs a framework's separate start and end callbacks by key.

    ``start`` opens a call under a key; ``complete`` and ``fail`` end it and return ``False`` when
    no call is open under that key (counted in ``unmatched``). A call with no end within
    ``ttl_seconds`` is reported once as ``CallFailed(error_type="Abandoned")``; ``start`` checks
    for these on every call, and ``sweep`` can be called on its own.
    """

    def __init__(self, telemetry: Telemetry, *, ttl_seconds: float = 600.0) -> None:
        self._telemetry = telemetry
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._open: dict[Any, CallHandle] = {}
        self._unmatched = 0

    @property
    def unmatched(self) -> int:
        """How many ends arrived for a key with no open call."""
        return self._unmatched

    def start(self, key: Any, resolved: ResolvedLLM, **attrs: Any) -> None:
        self.sweep()
        handle = self._telemetry.start_call(resolved, **attrs)
        with self._lock:
            replaced = self._open.pop(key, None)  # a key started twice: the first call can no longer end
            self._open[key] = handle
        if replaced is not None:
            replaced.fail(Abandoned("a new call started under the same key"))

    def complete(self, key: Any, usage: Usage | None = None, finish_reason: str | None = None) -> bool:
        handle = self._take(key)
        if handle is None:
            return False
        handle.complete(usage, finish_reason)
        return True

    def fail(self, key: Any, error: BaseException) -> bool:
        handle = self._take(key)
        if handle is None:
            return False
        handle.fail(error)
        return True

    def sweep(self) -> None:
        """Report each call older than the TTL, once, and stop tracking it."""
        with self._lock:
            expired = [k for k, h in self._open.items() if h._elapsed_seconds() > self._ttl]
            handles = [self._open.pop(k) for k in expired]
        for handle in handles:
            handle.fail(Abandoned(f"no end arrived within {self._ttl} seconds"))

    def _take(self, key: Any) -> CallHandle | None:
        with self._lock:
            handle = self._open.pop(key, None)
            if handle is None:
                self._unmatched += 1
            return handle
