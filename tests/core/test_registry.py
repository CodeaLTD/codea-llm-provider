"""InMemoryProviderRegistry and the errors (Task 2 Step 1; §Core types)."""

from __future__ import annotations

import pytest

from llm_provider import (
    ConfigurationError,
    CredentialNotApplicableError,
    InMemoryProviderRegistry,
    LLMProviderError,
    MissingCredentialError,
    ProviderSpec,
    UnknownProviderError,
    UserCredentialUnavailableError,
)


def spec(provider_id: str) -> ProviderSpec:
    return ProviderSpec(
        id=provider_id,
        display_name=provider_id.title(),
        default_model="m",
        api="openai_compatible",
        credential_key=None,
    )


def test_registry_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError):
        InMemoryProviderRegistry([spec("a"), spec("b"), spec("a")])


def test_get_unknown_raises_unknown_provider_error_with_the_id() -> None:
    registry = InMemoryProviderRegistry([spec("a")])

    with pytest.raises(UnknownProviderError) as caught:
        registry.get("nope")

    err = caught.value
    assert err.provider_id == "nope"
    assert str(err) == "Unknown LLM provider: 'nope'"
    assert isinstance(err, ConfigurationError)
    assert registry.get("a").id == "a"


def test_find_none_returns_none() -> None:
    registry = InMemoryProviderRegistry([spec("a")])

    assert registry.find(None) is None
    assert registry.find("nope") is None
    found = registry.find("a")
    assert found is not None and found.id == "a"


def test_all_keeps_host_order() -> None:
    registry = InMemoryProviderRegistry(spec(i) for i in ["c", "a", "b"])

    assert [s.id for s in registry.all()] == ["c", "a", "b"]


def test_errors_carry_fields_and_neutral_text() -> None:
    missing = MissingCredentialError("anthropic", "Anthropic", "ANTHROPIC_API_KEY")
    unavailable = UserCredentialUnavailableError("anthropic", "Anthropic", "no longer decrypts")

    assert (missing.provider, missing.display_name, missing.credential_key) == (
        "anthropic",
        "Anthropic",
        "ANTHROPIC_API_KEY",
    )
    assert "ANTHROPIC_API_KEY" in str(missing)
    assert "CrewAI" not in str(missing) and "agent" not in str(missing)
    assert (unavailable.provider, unavailable.display_name, unavailable.reason) == (
        "anthropic",
        "Anthropic",
        "no longer decrypts",
    )
    assert "CrewAI" not in str(unavailable) and "agent" not in str(unavailable)
    assert isinstance(missing, ConfigurationError) and isinstance(unavailable, ConfigurationError)
    assert issubclass(ConfigurationError, LLMProviderError)
    assert issubclass(CredentialNotApplicableError, LLMProviderError)
    assert issubclass(CredentialNotApplicableError, ValueError)


def test_all_returns_a_read_only_snapshot() -> None:
    registry = InMemoryProviderRegistry([spec("a"), spec("b")])

    listed = registry.all()

    with pytest.raises(TypeError):
        listed[0] = spec("z")  # type: ignore[index]
    assert [s.id for s in registry.all()] == ["a", "b"]


def test_credential_not_applicable_error_is_neutral_and_catchable_as_value_error() -> None:
    with pytest.raises(ValueError) as caught:
        raise CredentialNotApplicableError("Provider 'ollama' takes no credential")

    assert isinstance(caught.value, LLMProviderError)
    assert "CrewAI" not in str(caught.value) and "agent" not in str(caught.value)
