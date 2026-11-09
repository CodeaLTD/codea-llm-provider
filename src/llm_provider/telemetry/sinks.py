"""Sinks the library ships (§Telemetry)."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from .events import LLMEvent, event_name

# Names a ``LogRecord`` owns: passing one of them in ``extra`` raises ``KeyError``.
_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class LoggingSink:
    """Logs each event as one record named ``event_name``, with the event's fields as flat attributes.

    A call's usage becomes ``prompt_tokens``, ``completion_tokens``, ``total_tokens`` and
    ``usage_estimated``. An attr whose name is taken, by the log record or by an event field, is
    logged as ``attr_<name>``.
    """

    def __init__(self, logger_name: str = "llm_provider.telemetry", level: int = logging.INFO) -> None:
        self._logger = logging.getLogger(logger_name)
        self._level = level

    def emit(self, event: LLMEvent) -> None:
        fields = {
            f.name: getattr(event, f.name) for f in dataclasses.fields(event) if f.name not in ("attrs", "usage", "at")
        }
        fields["at"] = event.at.isoformat()
        usage = getattr(event, "usage", None)
        if usage is not None:
            fields.update(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                usage_estimated=usage.estimated,
            )
        taken = _RECORD_FIELDS | fields.keys()
        extra: dict[str, Any] = dict(fields)
        for name, value in event.attrs.items():
            extra[f"attr_{name}" if name in taken else name] = value
        self._logger.log(self._level, event_name(event), extra=extra)
