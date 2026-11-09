"""``MappingServerCredentials``: the server's keys, from a mapping the host passes in.

The library never reads the environment (GC); the host builds the mapping.
"""

from __future__ import annotations

from collections.abc import Mapping

from .secret import Secret


class MappingServerCredentials:
    """Reads keys from the mapping the host passes in.

    The mapping is held by reference, not copied, so a host that updates it (for example to
    rotate a key) is read live on the next ``get``.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, str | Secret | None]) -> None:
        self._values = values

    def get(self, credential_key: str) -> Secret | None:
        """The key as a ``Secret``; an empty or missing value counts as unset."""
        value = self._values.get(credential_key)
        if value is None:
            return None
        secret = value if isinstance(value, Secret) else Secret(value)
        return secret if secret.reveal() else None
