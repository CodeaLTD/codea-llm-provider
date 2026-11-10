"""LLMProviderService.select (Task 5 Step 1; AC-04.1, AC-04.2)."""

from __future__ import annotations

import pytest
from conftest import BuildRig

from llm_provider import (
    ConfigurationError,
    CredentialUnavailable,
    LLMSelection,
    MissingCredentialError,
    Secret,
    StoredCredential,
    UnknownProviderError,
    UserCredentialUnavailableError,
)


def test_select_without_provider_returns_the_default_without_a_credential_check(build: BuildRig) -> None:
    rig = build(server_keys={}, default_provider="anthropic", default_model="claude-opus-4-8")

    assert rig.service.select() == LLMSelection("anthropic", "claude-opus-4-8", None)
    assert rig.credentials.asked == []


def test_select_without_provider_takes_the_call_values_then_the_defaults(build: BuildRig) -> None:
    rig = build(default_provider="ollama", default_base_url="http://gpu:11434/v1")

    assert rig.service.select() == LLMSelection("ollama", "llama3.1", "http://gpu:11434/v1")
    assert rig.service.select(model="phi4", base_url="http://other/v1") == LLMSelection(
        "ollama", "phi4", "http://other/v1"
    )


def test_select_unknown_provider_raises_unknown_provider_error(build: BuildRig) -> None:
    with pytest.raises(UnknownProviderError) as caught:
        build().service.select("nope")

    assert caught.value.provider_id == "nope"


def test_explicit_provider_without_model_gets_its_own_default_model(build: BuildRig) -> None:
    rig = build(default_provider="anthropic", default_model="claude-opus-4-8")

    assert rig.service.select("openai").model == "gpt-4o"


def test_explicit_provider_does_not_inherit_the_default_base_url(build: BuildRig) -> None:
    rig = build(default_provider="ollama", default_base_url="http://localhost:11434")

    assert rig.service.select("anthropic").base_url is None


def test_selecting_the_default_provider_explicitly_keeps_the_default_model_and_base_url(build: BuildRig) -> None:
    rig = build(default_provider="ollama", default_model="llama3.1:70b", default_base_url="http://gpu:11434/v1")

    assert rig.service.select("ollama") == LLMSelection("ollama", "llama3.1:70b", "http://gpu:11434/v1")


def test_explicit_model_and_base_url_win_over_the_defaults(build: BuildRig) -> None:
    rig = build(default_provider="ollama", default_model="llama3.1:70b", default_base_url="http://gpu:11434/v1")

    assert rig.service.select("ollama", "phi4", "http://x/v1") == LLMSelection("ollama", "phi4", "http://x/v1")


def test_hosted_provider_without_any_credential_raises_missing_credential_error(build: BuildRig) -> None:
    with pytest.raises(MissingCredentialError) as caught:
        build(server_keys={}).service.select("openai")

    err = caught.value
    assert (err.provider, err.display_name, err.credential_key) == ("openai", "OpenAI", "OPENAI_API_KEY")
    assert isinstance(err, ConfigurationError)


def test_hosted_provider_with_only_a_user_credential_is_accepted(build: BuildRig) -> None:
    rig = build(server_keys={})

    selection = rig.service.select("openai", user_credential=StoredCredential(Secret("sk-user")))

    assert selection.provider == "openai"


def test_keyless_provider_needs_no_credential_to_be_selected(build: BuildRig) -> None:
    assert build(server_keys={}).service.select("lmstudio").provider == "lmstudio"


def test_unavailable_user_credential_without_a_server_key_raises_at_select(build: BuildRig) -> None:
    for policy in ("raise", "use_server"):
        rig = build(server_keys={}, on_unavailable_user_credential=policy)

        with pytest.raises(UserCredentialUnavailableError) as caught:
            rig.service.select("openai", user_credential=CredentialUnavailable("cannot decrypt"))

        assert (caught.value.provider, caught.value.reason) == ("openai", "cannot decrypt")


def test_unavailable_user_credential_with_a_server_key_passes_select_and_resolve_applies_the_policy(
    build: BuildRig,
) -> None:
    rig = build()

    selection = rig.service.select("openai", user_credential=CredentialUnavailable("cannot decrypt"))

    assert selection.provider == "openai"
    with pytest.raises(UserCredentialUnavailableError):
        rig.service.resolve(selection, user_credential=CredentialUnavailable("cannot decrypt"))
