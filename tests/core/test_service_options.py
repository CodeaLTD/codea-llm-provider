"""list_options, check_credential_target, client and construction (Task 5 Step 1; AC-04.7)."""

from __future__ import annotations

import pytest
from conftest import BuildRig, CountingCredentials, fixture_specs

from llm_provider import (
    ConfigurationError,
    CredentialNotApplicableError,
    InMemoryProviderRegistry,
    LLMSelection,
    ResolvedLLM,
    UnknownProviderError,
)
from llm_provider.config import LLMProviderConfig
from llm_provider.service import LLMProviderService


def options_by_id(rig_options: object) -> dict[str, object]:
    return {o.id: o for o in rig_options}  # type: ignore[attr-defined]


def test_list_options_flags(build: BuildRig) -> None:
    rig = build(server_keys={"OPENAI_API_KEY": "sk-server"})

    options = options_by_id(rig.service.list_options(user_providers=("anthropic",)))

    openai = options["openai"]
    assert (openai.available, openai.configured_by_user) == (True, False)  # type: ignore[attr-defined]
    anthropic = options["anthropic"]
    assert (anthropic.available, anthropic.configured_by_user) == (True, True)  # type: ignore[attr-defined]
    vllm = options["my-vllm"]
    assert (vllm.available, vllm.configured_by_user) == (False, False)  # type: ignore[attr-defined]
    for keyless in ("ollama", "lmstudio"):
        option = options[keyless]
        assert (option.available, option.configured_by_user, option.requires_key) == (True, False, False)  # type: ignore[attr-defined]


def test_list_options_keeps_the_registry_order_and_copies_the_spec_fields(build: BuildRig) -> None:
    rig = build()

    options = rig.service.list_options()

    assert [o.id for o in options] == [s.id for s in fixture_specs()]
    first = options[0]
    assert (first.display_name, first.default_model, first.api, first.requires_key) == (
        "Anthropic",
        "claude-sonnet-4-5",
        "anthropic",
        True,
    )


def test_a_user_key_on_top_of_a_server_key_is_reported_as_user_configured(build: BuildRig) -> None:
    options = options_by_id(build().service.list_options(user_providers=["openai"]))

    assert options["openai"].configured_by_user is True  # type: ignore[attr-defined]


def test_a_user_key_for_a_keyless_provider_is_not_user_configured(build: BuildRig) -> None:
    options = options_by_id(build().service.list_options(user_providers=["ollama"]))

    assert options["ollama"].configured_by_user is False  # type: ignore[attr-defined]


def test_check_credential_target_rejects_keyless_and_unknown(build: BuildRig) -> None:
    service = build().service

    assert service.check_credential_target("openai").id == "openai"
    with pytest.raises(CredentialNotApplicableError):
        service.check_credential_target("ollama")
    with pytest.raises(UnknownProviderError):
        service.check_credential_target("nope")


def test_service_rejects_a_default_provider_missing_from_the_registry(build: BuildRig) -> None:
    with pytest.raises(UnknownProviderError):
        build(default_provider="nope")


def test_default_selection_has_the_default_model_and_base_url(build: BuildRig) -> None:
    rig = build(default_provider="ollama", default_model="phi4", default_base_url="http://gpu/v1")

    assert rig.service.default_selection() == LLMSelection("ollama", "phi4", "http://gpu/v1")
    assert build().service.default_selection() == LLMSelection("anthropic", "claude-sonnet-4-5", None)


def test_client_builds_through_the_adapter(build: BuildRig) -> None:
    rig = build()
    seen: list[ResolvedLLM] = []

    class Adapter:
        def build(self, resolved: ResolvedLLM) -> str:
            seen.append(resolved)
            return "the client"

    client = rig.service.client(Adapter(), LLMSelection("openai", "gpt-4o"), max_tokens=7)

    assert client == "the client"
    assert seen[0].provider == "openai" and seen[0].max_tokens == 7


def test_service_exposes_its_config_and_registry() -> None:
    config = LLMProviderConfig(default_provider="ollama")
    registry = InMemoryProviderRegistry(fixture_specs())
    service = LLMProviderService(config=config, registry=registry, server_credentials=CountingCredentials({}))

    assert service.config is config and service.registry is registry


def test_service_arguments_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        LLMProviderService(LLMProviderConfig(default_provider="ollama"))  # type: ignore[misc]


def test_service_construction_error_is_a_configuration_error(build: BuildRig) -> None:
    with pytest.raises(ConfigurationError):
        build(default_provider="nope")
