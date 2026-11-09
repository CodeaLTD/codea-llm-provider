"""``Secret``: a value that never shows itself (GC Secrets)."""

from __future__ import annotations

_HIDDEN = "Secret(***)"


class Secret:
    """Wraps an API key so ``repr``, ``str`` and f-strings never show it.

    Only ``reveal()`` returns the value. Call it inside an integration's ``build()`` or in
    the host's own code, nowhere else.

    The guarantee covers ``repr``, ``str`` and formatting. ``pickle`` and ``copy`` carry the
    value, so a host must not pickle a ``Secret``, or an object that holds one, into storage
    it does not trust.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return _HIDDEN

    def __str__(self) -> str:
        return _HIDDEN

    def __format__(self, format_spec: str) -> str:
        return format(_HIDDEN, format_spec)
