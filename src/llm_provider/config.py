"""``LLMProviderConfig``: the host's settings for the resolution service (§Resolution rules).

The host builds it from its own settings; the library reads no environment (GC).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from .errors import ConfigurationError
from .types import RoleSettings, _freeze, _set

UnavailablePolicy = Literal["raise", "use_server"]
"""What to do with a user key that cannot be used: ``"raise"`` (the default) or fall back to the server key."""


@dataclass(frozen=True)
class LLMProviderConfig:
    default_provider: str
    default_model: str | None = None  # None: the default provider's own default model
    default_base_url: str | None = None  # attaches to the default provider only
    max_tokens: int | None = None
    timeout_seconds: float = 60.0
    default_params: Mapping[str, Any] = field(default_factory=dict)
    roles: Mapping[str, RoleSettings] = field(default_factory=dict)
    on_unavailable_user_credential: UnavailablePolicy = "raise"

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ConfigurationError(f"timeout_seconds must be more than 0, got {self.timeout_seconds!r}")
        if self.on_unavailable_user_credential not in get_args(UnavailablePolicy):
            raise ConfigurationError(
                f"on_unavailable_user_credential must be one of {get_args(UnavailablePolicy)}, "
                f"got {self.on_unavailable_user_credential!r}"
            )
        _set(self, "default_params", _freeze(self.default_params))
        _set(self, "roles", _freeze(self.roles))
