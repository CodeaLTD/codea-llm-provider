"""Protocols the host or an integration implements (§Core types)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

from .secret import Secret
from .types import ProviderSpec, ResolvedLLM

ClientT_co = TypeVar("ClientT_co", covariant=True)


class ProviderRegistry(Protocol):
    def get(self, id: str) -> ProviderSpec:
        """Raises ``UnknownProviderError`` when ``id`` is not declared."""
        ...

    def find(self, id: str | None) -> ProviderSpec | None: ...

    def all(self) -> Sequence[ProviderSpec]: ...


class ServerCredentials(Protocol):
    def get(self, credential_key: str) -> Secret | None: ...


class ClientAdapter(Protocol[ClientT_co]):
    def build(self, resolved: ResolvedLLM) -> ClientT_co: ...
