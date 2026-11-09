"""Frozen value types (§Core types).

``Mapping`` fields are stored read-only (``MappingProxyType``), copied from what the host passes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Self

from .secret import Secret


def _freeze(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


def _set(obj: object, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)  # the one way to normalise a field of a frozen dataclass


class Omit:
    """The type of ``OMIT``."""

    _instance: ClassVar[Any] = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return "OMIT"


OMIT = Omit()
"""As a role or call value: send nothing for this setting."""


@dataclass(frozen=True)
class ModelProfile:
    """What one model needs or rejects."""

    params: Mapping[str, Any] = field(default_factory=dict)
    drop_params: frozenset[str] = frozenset()
    max_tokens_param: str = "max_tokens"
    timeout_seconds: float | None = None
    context_window: int | None = None

    def __post_init__(self) -> None:
        _set(self, "params", _freeze(self.params))
        _set(self, "drop_params", frozenset(self.drop_params))


@dataclass(frozen=True)
class ProviderSpec:
    """One provider as the host declares it. ``requires_key`` is true exactly when a key is named."""

    id: str
    display_name: str
    default_model: str
    api: str
    credential_key: str | None
    default_base_url: str | None = None
    placeholder_api_key: str | None = None
    auth_header: str | None = None
    default_headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None
    default_params: Mapping[str, Any] = field(default_factory=dict)
    model_profiles: Mapping[str, ModelProfile] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.auth_header is not None:
            if self.credential_key is None:
                raise ValueError("auth_header needs a credential_key: there is no key to send in it")
            if self.auth_header.lower() in {name.lower() for name in self.default_headers}:
                raise ValueError(f"default_headers may not hold the auth header {self.auth_header!r}")
        for name in ("default_headers", "default_params", "model_profiles", "attributes"):
            _set(self, name, _freeze(getattr(self, name)))

    @property
    def requires_key(self) -> bool:
        return self.credential_key is not None

    def profile_for(self, model: str) -> ModelProfile:
        """The exact id, else the first matching pattern in declaration order, else an empty profile."""
        if model in self.model_profiles:
            return self.model_profiles[model]
        for pattern, profile in self.model_profiles.items():
            if fnmatchcase(model, pattern):
                return profile
        return ModelProfile()


@dataclass(frozen=True)
class LLMSelection:
    """What a host stores for a run."""

    provider: str
    model: str
    base_url: str | None = None


@dataclass(frozen=True)
class StoredCredential:
    """A user's key, already decrypted by the host. A plain ``str`` is wrapped into ``Secret``."""

    api_key: Secret
    base_url: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.api_key, str):
            _set(self, "api_key", Secret(self.api_key))


@dataclass(frozen=True)
class CredentialUnavailable:
    """A user key exists but cannot be used, for example it no longer decrypts."""

    reason: str


@dataclass(frozen=True)
class Usage:
    """Token counts. ``estimated`` is true when the library estimated any of them."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated: bool = False


@dataclass(frozen=True)
class ResolvedLLM:
    """Everything an integration needs to build a client for one call."""

    spec: ProviderSpec
    selection: LLMSelection
    api_key: Secret | None
    base_url: str | None
    max_tokens: int | None  # None: do not send
    max_tokens_param: str
    timeout_seconds: float
    params: Mapping[str, Any]
    dropped_params: frozenset[str]
    headers: Mapping[str, str]  # non-secret
    auth_header: str | None
    context_window: int | None
    origin: Literal["user", "server", "placeholder", "none"]
    degraded_reason: str | None = None

    def __post_init__(self) -> None:
        _set(self, "params", _freeze(self.params))
        _set(self, "headers", _freeze(self.headers))
        _set(self, "dropped_params", frozenset(self.dropped_params))

    @property
    def provider(self) -> str:
        return self.selection.provider

    @property
    def model(self) -> str:
        return self.selection.model

    @property
    def api(self) -> str:
        return self.spec.api


@dataclass(frozen=True)
class ProviderOption:
    """One provider as a host shows it to a user."""

    id: str
    display_name: str
    default_model: str
    api: str
    requires_key: bool
    available: bool
    configured_by_user: bool
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _set(self, "attributes", _freeze(self.attributes))


@dataclass(frozen=True)
class RoleSettings:
    """Per-role overrides. ``None`` means inherit."""

    max_tokens: int | Omit | None = None
    timeout_seconds: float | None = None
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _set(self, "params", _freeze(self.params))
