"""Helpers for tests: ``RecordingSink`` here; ``StubLLMServer`` arrives in plan 05."""

from __future__ import annotations

import threading

from .telemetry.events import LLMEvent


class RecordingSink:
    """A telemetry sink that keeps every event it receives. Safe to share between threads."""

    def __init__(self) -> None:
        self.events: list[LLMEvent] = []
        self._lock = threading.Lock()

    def emit(self, event: LLMEvent) -> None:
        with self._lock:
            self.events.append(event)
