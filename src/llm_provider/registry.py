"""``InMemoryProviderRegistry``: the host's providers, in the host's order."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .errors import UnknownProviderError
from .types import ProviderSpec


class InMemoryProviderRegistry:
    def __init__(self, specs: Iterable[ProviderSpec]) -> None:
        by_id: dict[str, ProviderSpec] = {}
        for spec in specs:
            if spec.id in by_id:
                raise ValueError(f"Duplicate provider id: {spec.id!r}")
            by_id[spec.id] = spec
        self._by_id = by_id

    def get(self, id: str) -> ProviderSpec:
        try:
            return self._by_id[id]
        except KeyError:
            raise UnknownProviderError(id) from None

    def find(self, id: str | None) -> ProviderSpec | None:
        return None if id is None else self._by_id.get(id)

    def all(self) -> Sequence[ProviderSpec]:
        return tuple(self._by_id.values())
