"""``Telemetry``: delivers events to the host's sinks and totals usage (§Telemetry)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any

from ..interfaces import TelemetrySink
from ..types import Usage
from .events import CallCompleted, CallFailed, LLMEvent, event_name

_WARN_EVERY = 100


class UsageCollector:
    """The usage of each call that ended inside a ``Telemetry.collect()`` block."""

    def __init__(self) -> None:
        self.calls: list[Usage] = []
        self._lock = threading.Lock()

    def record(self, usage: Usage) -> None:
        with self._lock:
            self.calls.append(usage)

    def summary(self) -> dict[str, int]:
        """Token totals over the calls; a count a provider left out counts as 0."""
        with self._lock:
            calls = list(self.calls)
        return {
            "prompt_tokens": sum(u.prompt_tokens or 0 for u in calls),
            "completion_tokens": sum(u.completion_tokens or 0 for u in calls),
            "total_tokens": sum(u.total_tokens or 0 for u in calls),
            "call_count": len(calls),
            "estimated_call_count": sum(1 for u in calls if u.estimated),
        }


class Telemetry:
    """The hub a host attaches its sinks to; every call's events go through it.

    - ``emit(event)`` gives an event to each sink (an object with ``emit(event)``) in order.
      A sink that raises is counted in ``dropped_sink_errors`` and skipped.
    - ``scope(**attrs)`` tags the calls made inside the block; ``static_attrs`` tags them all.
    - ``collect()`` totals the usage of the calls that end inside the block.
    """

    def __init__(
        self,
        sinks: Iterable[TelemetrySink] = (),
        *,
        static_attrs: Mapping[str, Any] = MappingProxyType({}),
        logger_name: str = "llm_provider.telemetry",
    ) -> None:
        self._sinks: tuple[TelemetrySink, ...] = tuple(sinks)
        self._static_attrs: Mapping[str, Any] = MappingProxyType(dict(static_attrs))
        self._logger = logging.getLogger(logger_name)
        self._lock = threading.Lock()
        self._failures: dict[int, int] = {}  # sink id -> failures so far
        self._dropped = 0
        self._scope: ContextVar[Mapping[str, Any]] = ContextVar("llm_provider_scope", default=MappingProxyType({}))
        self._collector: ContextVar[UsageCollector | None] = ContextVar("llm_provider_collector", default=None)

    @property
    def dropped_sink_errors(self) -> int:
        return self._dropped

    def add_sink(self, sink: TelemetrySink) -> None:
        """Add ``sink`` after the others. It replaces the sink tuple, so an emit in progress is unaffected."""
        with self._lock:
            self._sinks = (*self._sinks, sink)

    def remove_sink(self, sink: TelemetrySink) -> None:
        """Remove ``sink``; one that is not attached is ignored. An emit in progress still reaches it."""
        with self._lock:
            self._sinks = tuple(s for s in self._sinks if s != sink)
            self._failures.pop(id(sink), None)

    def emit(self, event: LLMEvent) -> None:
        """Give ``event`` to each sink in order. A sink that raises is counted and skipped."""
        collector = self._collector.get()
        if collector is not None:
            if isinstance(event, CallCompleted):
                collector.record(event.usage or Usage())
            elif isinstance(event, CallFailed) and event.usage is not None:
                collector.record(event.usage)  # the provider billed these tokens
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception as exc:  # noqa: BLE001 - a sink must never break a model call
                self._sink_failed(sink, event, exc)

    @contextmanager
    def scope(self, **attrs: Any) -> Iterator[None]:
        """Add ``attrs`` to the events of every call made inside the block, on this context."""
        token = self._scope.set(MappingProxyType({**self._scope.get(), **attrs}))
        try:
            yield
        finally:
            self._scope.reset(token)

    def current_attrs(self, **extra: Any) -> dict[str, Any]:
        """Static attrs, then the active scope's, then ``extra``; a later one wins."""
        return {**self._static_attrs, **self._scope.get(), **extra}

    @contextmanager
    def collect(self) -> Iterator[UsageCollector]:
        """Total the usage of the calls that end inside the block, on this context."""
        collector = UsageCollector()
        token = self._collector.set(collector)
        try:
            yield collector
        finally:
            self._collector.reset(token)

    def _sink_failed(self, sink: TelemetrySink, event: LLMEvent, exc: Exception) -> None:
        with self._lock:
            self._dropped += 1
            count = self._failures.get(id(sink), 0) + 1
            self._failures[id(sink)] = count
        if count == 1 or count % _WARN_EVERY == 0:
            # The exception's text could hold a secret, so only its type is logged.
            self._logger.warning(
                "telemetry sink %s failed on %s with %s (%d failure(s) so far)",
                type(sink).__name__,
                event_name(event),
                type(exc).__name__,
                count,
            )
