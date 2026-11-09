"""Protocols the host or an integration implements (§Core types)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypeVar

from .secret import Secret
from .types import ProviderSpec, ResolvedLLM

if TYPE_CHECKING:  # telemetry imports this module, so the import cannot run
    from .telemetry.events import LLMEvent

ClientT_co = TypeVar("ClientT_co", covariant=True)


class ProviderRegistry(Protocol):
    def get(self, id: str) -> ProviderSpec:
        """Raises ``UnknownProviderError`` when ``id`` is not declared."""
        ...

    def find(self, id: str | None) -> ProviderSpec | None: ...

    def all(self) -> Sequence[ProviderSpec]: ...


class ServerCredentials(Protocol):
    def get(self, credential_key: str) -> Secret | None: ...


class TelemetrySink(Protocol):
    """Receives each event a ``Telemetry`` hub emits: any object with an ``emit(event)`` method.

    A sink must not block. One that raises is swallowed and counted; it never breaks a model call.
    """

    def emit(self, event: LLMEvent) -> None: ...


class ClientAdapter(Protocol[ClientT_co]):
    def build(self, resolved: ResolvedLLM) -> ClientT_co: ...
