"""``LLMProviderService``: turns the host's settings and a user's key into a ``ResolvedLLM`` (§Resolution rules).

``resolve`` is pure: apart from ``server_credentials.get`` it calls nothing, so it opens no
connection and reads no environment. Every method is sync.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .config import LLMProviderConfig
from .errors import (
    ConfigurationError,
    CredentialNotApplicableError,
    MissingCredentialError,
    UserCredentialUnavailableError,
)
from .interfaces import ClientAdapter, ProviderRegistry, ServerCredentials
from .secret import Secret
from .telemetry import CredentialFallback, Telemetry, scrub
from .types import (
    OMIT,
    CredentialUnavailable,
    LLMSelection,
    Omit,
    ProviderOption,
    ProviderSpec,
    ResolvedLLM,
    RoleSettings,
    StoredCredential,
)

UserCredential = StoredCredential | CredentialUnavailable


class LLMProviderService:
    def __init__(
        self,
        *,
        config: LLMProviderConfig,
        registry: ProviderRegistry,
        server_credentials: ServerCredentials,
        telemetry: Telemetry | None = None,
    ) -> None:
        registry.get(config.default_provider)  # raises UnknownProviderError
        self.config = config
        self.registry = registry
        self._server_credentials = server_credentials
        self._telemetry = telemetry

    # -- selection ---------------------------------------------------------------------------

    def default_selection(self) -> LLMSelection:
        return self._default_selection(None, None)

    def select(
        self,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        *,
        user_credential: UserCredential | None = None,
    ) -> LLMSelection:
        """The provider, model and base URL for a run.

        With no ``provider`` this is the default provider and no credential is checked. With one, a
        default model or base URL attaches only when it is the default provider: a default meant
        for one provider never reaches another.
        """
        if provider is None:
            return self._default_selection(model, base_url)
        spec = self.registry.get(provider)
        is_default = provider == self.config.default_provider
        selection = LLMSelection(
            provider=provider,
            model=model or (self.config.default_model if is_default else None) or spec.default_model,
            base_url=base_url or (self.config.default_base_url if is_default else None),
        )
        self._check_credential(spec, user_credential)
        return selection

    def _default_selection(self, model: str | None, base_url: str | None) -> LLMSelection:
        spec = self.registry.get(self.config.default_provider)
        return LLMSelection(
            provider=spec.id,
            model=model or self.config.default_model or spec.default_model,
            base_url=base_url or self.config.default_base_url,
        )

    def _check_credential(self, spec: ProviderSpec, user_credential: UserCredential | None) -> None:
        """A provider that needs a key passes with a user key or a server key; nothing else does."""
        if not spec.requires_key or isinstance(user_credential, StoredCredential):
            return
        if self._server_key(spec) is not None:
            return  # an unavailable user key is left to ``resolve`` and the policy
        if isinstance(user_credential, CredentialUnavailable):
            raise UserCredentialUnavailableError(spec.id, spec.display_name, scrub(user_credential.reason))
        assert spec.credential_key is not None
        raise MissingCredentialError(spec.id, spec.display_name, spec.credential_key)

    # -- resolution --------------------------------------------------------------------------

    def resolve(
        self,
        selection: LLMSelection,
        *,
        user_credential: UserCredential | None = None,
        role: str | None = None,
        max_tokens: int | Omit | None = None,
        params: dict[str, Any] | None = None,
    ) -> ResolvedLLM:
        spec = self.registry.get(selection.provider)
        role_settings = self._role(role)
        profile = spec.profile_for(selection.model)

        api_key, origin, degraded_reason, credential_base_url = self._key_and_origin(spec, selection, user_credential)
        base_url = credential_base_url or selection.base_url or spec.default_base_url

        merged = self._merge_params(spec, profile.params, role_settings, params)
        for forbidden in {"max_tokens", profile.max_tokens_param}:
            if forbidden in merged:
                raise ConfigurationError(f"params may not hold {forbidden!r}: the token cap has its own setting")
        dropped = frozenset(key for key in profile.drop_params if key in merged)
        merged = {key: value for key, value in merged.items() if key not in dropped}

        return ResolvedLLM(
            spec=spec,
            selection=selection,
            api_key=api_key,
            base_url=base_url,
            max_tokens=self._max_tokens(max_tokens, role_settings),
            max_tokens_param=profile.max_tokens_param,
            timeout_seconds=self._first_set(
                role_settings.timeout_seconds,
                profile.timeout_seconds,
                spec.timeout_seconds,
                self.config.timeout_seconds,
            ),
            params=merged,
            dropped_params=dropped,
            headers=spec.default_headers,
            auth_header=spec.auth_header,
            context_window=profile.context_window,
            origin=origin,
            degraded_reason=degraded_reason,
        )

    def _role(self, role: str | None) -> RoleSettings:
        if role is None:
            return RoleSettings()
        try:
            return self.config.roles[role]
        except KeyError:
            raise ConfigurationError(f"Unknown role: {role!r}") from None

    def _key_and_origin(
        self, spec: ProviderSpec, selection: LLMSelection, user_credential: UserCredential | None
    ) -> tuple[Secret | None, Any, str | None, str | None]:
        """The key, its origin, the reason it fell back (if it did) and the credential's own base URL."""
        degraded_reason: str | None = None
        if isinstance(user_credential, StoredCredential):  # step 1
            return user_credential.api_key, "user", None, user_credential.base_url
        server_key = self._server_key(spec)  # asked once, and only when no user key is used
        if isinstance(user_credential, CredentialUnavailable):  # step 2
            reason = scrub(user_credential.reason)  # the host's text may quote a key
            if self.config.on_unavailable_user_credential == "raise" or (spec.requires_key and server_key is None):
                raise UserCredentialUnavailableError(spec.id, spec.display_name, reason)
            degraded_reason = reason
            self._emit_fallback(selection, reason)
        if server_key is not None:  # step 3
            return server_key, "server", degraded_reason, None
        if spec.placeholder_api_key is not None:  # step 4
            return Secret(spec.placeholder_api_key), "placeholder", degraded_reason, None
        if spec.credential_key is None:  # step 5
            return None, "none", degraded_reason, None
        raise MissingCredentialError(spec.id, spec.display_name, spec.credential_key)  # step 6

    def _server_key(self, spec: ProviderSpec) -> Secret | None:
        if spec.credential_key is None:
            return None
        return self._server_credentials.get(spec.credential_key)

    def _emit_fallback(self, selection: LLMSelection, reason: str) -> None:
        """``reason`` is already scrubbed."""
        if self._telemetry is None:
            return
        self._telemetry.emit(
            CredentialFallback(
                provider=selection.provider,
                model=selection.model,
                reason=reason,
                attrs=self._telemetry.current_attrs(),
            )
        )

    def _max_tokens(self, call_value: int | Omit | None, role: RoleSettings) -> int | None:
        value = self._first_set(call_value, role.max_tokens, self.config.max_tokens)
        return None if value is None or value is OMIT else value

    def _merge_params(
        self,
        spec: ProviderSpec,
        profile_params: Any,
        role: RoleSettings,
        call_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Later wins: config, spec, profile, role, call. A key set to ``OMIT`` is removed."""
        merged: dict[str, Any] = {}
        for layer in (self.config.default_params, spec.default_params, profile_params, role.params, call_params or {}):
            merged.update(layer)
        return {key: value for key, value in merged.items() if value is not OMIT}

    @staticmethod
    def _first_set(*values: Any) -> Any:
        """The first value that is not ``None`` (``OMIT`` counts as set), else ``None``."""
        return next((value for value in values if value is not None), None)

    # -- the rest ----------------------------------------------------------------------------

    def list_options(self, *, user_providers: Iterable[str] = ()) -> Sequence[ProviderOption]:
        """Every provider as a host shows it: available with a server key or a user key, or if it needs none."""
        user_ids = frozenset(user_providers)
        options = []
        for spec in self.registry.all():
            configured_by_user = spec.requires_key and spec.id in user_ids
            options.append(
                ProviderOption(
                    id=spec.id,
                    display_name=spec.display_name,
                    default_model=spec.default_model,
                    api=spec.api,
                    requires_key=spec.requires_key,
                    available=not spec.requires_key or configured_by_user or self._server_key(spec) is not None,
                    configured_by_user=configured_by_user,
                    attributes=spec.attributes,
                )
            )
        return tuple(options)

    def check_credential_target(self, provider: str) -> ProviderSpec:
        """The spec of a provider that takes a user key. Raises for an unknown id or a keyless provider."""
        spec = self.registry.get(provider)
        if not spec.requires_key:
            raise CredentialNotApplicableError(f"Provider {provider!r} takes no credential")
        return spec

    def client(self, adapter: ClientAdapter[Any], selection: LLMSelection, **resolve_kwargs: Any) -> Any:
        """Build a framework client: ``adapter.build(self.resolve(selection, **resolve_kwargs))``."""
        return adapter.build(self.resolve(selection, **resolve_kwargs))
