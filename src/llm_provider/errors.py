"""Errors. Each keeps its fields as attributes and has a neutral message (GC No host vocabulary)."""

from __future__ import annotations


class LLMProviderError(Exception):
    """Base class for every error this library raises."""


class ConfigurationError(LLMProviderError):
    """The host's configuration cannot serve the call."""


class UnknownProviderError(ConfigurationError):
    def __init__(self, provider_id: str) -> None:
        super().__init__(f"Unknown LLM provider: {provider_id!r}")
        self.provider_id = provider_id


class MissingCredentialError(ConfigurationError):
    def __init__(self, provider: str, display_name: str, credential_key: str) -> None:
        super().__init__(f"No credential for provider {provider!r} ({display_name}): {credential_key!r} is not set")
        self.provider = provider
        self.display_name = display_name
        self.credential_key = credential_key


class UserCredentialUnavailableError(ConfigurationError):
    def __init__(self, provider: str, display_name: str, reason: str) -> None:
        super().__init__(f"The stored credential for provider {provider!r} ({display_name}) cannot be used: {reason}")
        self.provider = provider
        self.display_name = display_name
        self.reason = reason


class CredentialNotApplicableError(LLMProviderError, ValueError):
    """A credential was given for a provider that takes none."""
