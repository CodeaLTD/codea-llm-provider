"""LLMProviderService.resolve (Task 5 Step 1; AC-04.3 to AC-04.6)."""

from __future__ import annotations

import socket
from dataclasses import replace

import pytest
from conftest import BuildRig, fixture_specs

from llm_provider import (
    OMIT,
    ConfigurationError,
    CredentialFallback,
    CredentialUnavailable,
    LLMSelection,
    MissingCredentialError,
    ModelProfile,
    ProviderSpec,
    RoleSettings,
    Secret,
    StoredCredential,
    UnknownProviderError,
    UserCredentialUnavailableError,
)
from llm_provider.service import LLMProviderService


def user(key: str = "sk-user-key", base_url: str | None = None) -> StoredCredential:
    return StoredCredential(Secret(key), base_url)


def spec_with(provider_id: str = "p", **overrides: object) -> ProviderSpec:
    fields: dict[str, object] = {
        "id": provider_id,
        "display_name": "P",
        "default_model": "m",
        "api": "openai",
        "credential_key": "P_KEY",
    }
    return ProviderSpec(**{**fields, **overrides})  # type: ignore[arg-type]


P_KEYS = {"P_KEY": "sk-server-p"}


# -- key and origin ---------------------------------------------------------------------------


def test_user_credential_beats_server_key(build: BuildRig) -> None:
    rig = build()

    resolved = rig.service.resolve(LLMSelection("openai", "gpt-4o"), user_credential=user("sk-mine"))

    assert resolved.origin == "user"
    assert resolved.api_key is not None and resolved.api_key.reveal() == "sk-mine"


def test_server_key_when_no_user_credential(build: BuildRig) -> None:
    resolved = build().service.resolve(LLMSelection("openai", "gpt-4o"))

    assert resolved.origin == "server"
    assert resolved.api_key is not None and resolved.api_key.reveal() == "sk-server-openai"


def test_placeholder_for_a_keyless_provider_that_needs_one(build: BuildRig) -> None:
    resolved = build().service.resolve(LLMSelection("lmstudio", "local-model"))

    assert resolved.origin == "placeholder"
    assert resolved.api_key is not None and resolved.api_key.reveal() == "lm-studio"


def test_keyless_provider_needs_no_key(build: BuildRig) -> None:
    resolved = build().service.resolve(LLMSelection("ollama", "llama3.1"))

    assert resolved.origin == "none"
    assert resolved.api_key is None


def test_missing_server_key_raises_missing_credential_error_naming_the_key(build: BuildRig) -> None:
    with pytest.raises(MissingCredentialError) as caught:
        build(server_keys={}).service.resolve(LLMSelection("openai", "gpt-4o"))

    assert caught.value.credential_key == "OPENAI_API_KEY"


# -- base URL ---------------------------------------------------------------------------------


def test_user_base_url_overrides_selection_base_url(build: BuildRig) -> None:
    resolved = build().service.resolve(
        LLMSelection("openai", "gpt-4o", "http://selected/v1"), user_credential=user(base_url="http://mine/v1")
    )

    assert resolved.base_url == "http://mine/v1"


def test_user_credential_without_base_url_keeps_selection_base_url(build: BuildRig) -> None:
    resolved = build().service.resolve(LLMSelection("openai", "gpt-4o", "http://selected/v1"), user_credential=user())

    assert resolved.base_url == "http://selected/v1"


def test_spec_base_url_is_the_last_fallback(build: BuildRig) -> None:
    rig = build()

    assert rig.service.resolve(LLMSelection("ollama", "llama3.1")).base_url == "http://localhost:11434/v1"
    assert rig.service.resolve(LLMSelection("ollama", "llama3.1", "http://sel/v1")).base_url == "http://sel/v1"
    assert rig.service.resolve(LLMSelection("openai", "gpt-4o")).base_url is None
    assert rig.service.resolve(LLMSelection("openai", "gpt-4o"), user_credential=user()).base_url is None


# -- an unavailable user credential -----------------------------------------------------------


def test_unavailable_user_credential_raises_by_default(build: BuildRig) -> None:
    rig = build()

    with pytest.raises(UserCredentialUnavailableError) as caught:
        rig.service.resolve(LLMSelection("openai", "gpt-4o"), user_credential=CredentialUnavailable("cannot decrypt"))

    assert caught.value.reason == "cannot decrypt"
    assert rig.sink.events == []


def test_unavailable_user_credential_uses_server_key_when_policy_allows(build: BuildRig) -> None:
    rig = build(on_unavailable_user_credential="use_server")

    resolved = rig.service.resolve(
        LLMSelection("openai", "gpt-4o"), user_credential=CredentialUnavailable("cannot decrypt")
    )

    assert resolved.origin == "server"
    assert resolved.degraded_reason == "cannot decrypt"
    (event,) = rig.sink.events
    assert isinstance(event, CredentialFallback)
    assert (event.provider, event.model, event.reason) == ("openai", "gpt-4o", "cannot decrypt")


def test_unavailable_user_credential_with_no_server_key_raises_even_when_policy_allows(build: BuildRig) -> None:
    rig = build(server_keys={}, on_unavailable_user_credential="use_server")

    with pytest.raises(UserCredentialUnavailableError):
        rig.service.resolve(LLMSelection("openai", "gpt-4o"), user_credential=CredentialUnavailable("cannot decrypt"))

    assert rig.sink.events == []


def test_fallback_reason_in_the_event_is_scrubbed(build: BuildRig) -> None:
    rig = build(on_unavailable_user_credential="use_server")

    rig.service.resolve(
        LLMSelection("openai", "gpt-4o"), user_credential=CredentialUnavailable("bad key sk-abcdefghijkl")
    )

    (event,) = rig.sink.events
    assert isinstance(event, CredentialFallback)
    assert "abcdefghijkl" not in event.reason


def test_fallback_reason_is_scrubbed_in_degraded_reason_and_in_the_error(build: BuildRig) -> None:
    unavailable = CredentialUnavailable("bad key sk-abcdefghijkl")
    selection = LLMSelection("openai", "gpt-4o")

    resolved = build(on_unavailable_user_credential="use_server").service.resolve(
        selection, user_credential=unavailable
    )
    assert resolved.degraded_reason is not None and "abcdefghijkl" not in resolved.degraded_reason

    with pytest.raises(UserCredentialUnavailableError) as raised:  # the default policy
        build().service.resolve(selection, user_credential=unavailable)
    assert "abcdefghijkl" not in str(raised.value) and "abcdefghijkl" not in raised.value.reason

    with pytest.raises(UserCredentialUnavailableError) as at_select:  # no server key
        build(server_keys={}).service.select("openai", user_credential=unavailable)
    assert "abcdefghijkl" not in str(at_select.value) and "abcdefghijkl" not in at_select.value.reason


def test_fallback_asks_the_server_for_its_key_once(build: BuildRig) -> None:
    rig = build(on_unavailable_user_credential="use_server")

    rig.service.resolve(LLMSelection("openai", "gpt-4o"), user_credential=CredentialUnavailable("gone"))

    assert rig.credentials.asked == ["OPENAI_API_KEY"]


def test_fallback_event_carries_the_scope_attrs(build: BuildRig) -> None:
    rig = build(on_unavailable_user_credential="use_server")
    with rig.telemetry.scope(tenant="acme"):
        rig.service.resolve(LLMSelection("openai", "gpt-4o"), user_credential=CredentialUnavailable("gone"))

    (event,) = rig.sink.events
    assert event.attrs == {"tenant": "acme"}


def test_a_service_without_telemetry_still_falls_back(build: BuildRig) -> None:
    rig = build(on_unavailable_user_credential="use_server")

    service = LLMProviderService(
        config=rig.service.config, registry=rig.service.registry, server_credentials=rig.credentials
    )

    resolved = service.resolve(LLMSelection("openai", "gpt-4o"), user_credential=CredentialUnavailable("gone"))

    assert resolved.origin == "server"


# -- max_tokens -------------------------------------------------------------------------------


def test_max_tokens_precedence_call_then_role_then_config(build: BuildRig) -> None:
    rig = build(max_tokens=4000, roles={"short": RoleSettings(max_tokens=2000)})
    selection = LLMSelection("openai", "gpt-4o")

    assert rig.service.resolve(selection).max_tokens == 4000
    assert rig.service.resolve(selection, role="short").max_tokens == 2000
    assert rig.service.resolve(selection, role="short", max_tokens=1000).max_tokens == 1000
    assert rig.service.resolve(selection, max_tokens=1000).max_tokens == 1000


def test_max_tokens_is_none_when_nothing_sets_it(build: BuildRig) -> None:
    assert build().service.resolve(LLMSelection("openai", "gpt-4o")).max_tokens is None


def test_omit_removes_max_tokens(build: BuildRig) -> None:
    rig = build(max_tokens=4000, roles={"judge": RoleSettings(max_tokens=OMIT)})
    selection = LLMSelection("openai", "gpt-4o")

    assert rig.service.resolve(selection, role="judge").max_tokens is None
    assert rig.service.resolve(selection, max_tokens=OMIT).max_tokens is None  # type: ignore[arg-type]


def test_profile_sets_the_token_cap_parameter_name(build: BuildRig) -> None:
    spec = spec_with(model_profiles={"o*": ModelProfile(max_tokens_param="max_completion_tokens")})
    rig = build(server_keys=P_KEYS, specs=[spec], default_provider="p", max_tokens=500)

    resolved = rig.service.resolve(LLMSelection("p", "o3"))

    assert (resolved.max_tokens, resolved.max_tokens_param) == (500, "max_completion_tokens")
    assert rig.service.resolve(LLMSelection("p", "m")).max_tokens_param == "max_tokens"


# -- params -----------------------------------------------------------------------------------


def test_params_merge_call_over_role_over_default_and_omit_drops_a_key(build: BuildRig) -> None:
    rig = build(
        default_params={"temperature": 0.7},
        roles={"judge": RoleSettings(params={"temperature": 0.0})},
    )
    selection = LLMSelection("openai", "gpt-4o")

    assert rig.service.resolve(selection, role="judge").params == {"temperature": 0.0}
    assert rig.service.resolve(selection, role="judge", params={"top_p": 0.9, "temperature": OMIT}).params == {
        "top_p": 0.9
    }


def test_param_merge_order_config_spec_profile_role_call(build: BuildRig) -> None:
    spec = spec_with(
        default_params={"k2": "spec", "k3": "spec", "k4": "spec", "k5": "spec"},
        model_profiles={"m": ModelProfile(params={"k3": "profile", "k4": "profile", "k5": "profile"})},
    )
    rig = build(
        server_keys=P_KEYS,
        specs=[spec],
        default_provider="p",
        default_params={f"k{i}": "config" for i in range(1, 6)},
        roles={"r": RoleSettings(params={"k4": "role", "k5": "role"})},
    )

    resolved = rig.service.resolve(LLMSelection("p", "m"), role="r", params={"k5": "call"})

    assert dict(resolved.params) == {"k1": "config", "k2": "spec", "k3": "profile", "k4": "role", "k5": "call"}


def test_profile_drop_params_win_over_role_and_call_and_are_reported(build: BuildRig) -> None:
    spec = spec_with(model_profiles={"m": ModelProfile(drop_params=frozenset({"temperature", "seed"}))})
    rig = build(
        server_keys=P_KEYS,
        specs=[spec],
        default_provider="p",
        roles={"r": RoleSettings(params={"temperature": 0.0})},
    )

    resolved = rig.service.resolve(LLMSelection("p", "m"), role="r", params={"temperature": 0.2, "top_p": 0.9})

    assert dict(resolved.params) == {"top_p": 0.9}
    assert resolved.dropped_params == {"temperature"}  # "seed" was never set, so nothing was dropped


def test_params_may_not_carry_the_token_cap(build: BuildRig) -> None:
    spec = spec_with(model_profiles={"o*": ModelProfile(max_tokens_param="max_completion_tokens")})
    rig = build(server_keys=P_KEYS, specs=[spec], default_provider="p")

    with pytest.raises(ConfigurationError):
        rig.service.resolve(LLMSelection("p", "m"), params={"max_tokens": 5})
    with pytest.raises(ConfigurationError):
        rig.service.resolve(LLMSelection("p", "o3"), params={"max_completion_tokens": 5})
    with pytest.raises(ConfigurationError):  # either name is refused, whichever the profile sends
        rig.service.resolve(LLMSelection("p", "o3"), params={"max_tokens": 5})


def test_a_default_param_that_carries_the_token_cap_is_refused_too(build: BuildRig) -> None:
    rig = build(default_params={"max_tokens": 5})

    with pytest.raises(ConfigurationError):
        rig.service.resolve(LLMSelection("openai", "gpt-4o"))


# -- timeout ----------------------------------------------------------------------------------


def test_timeout_precedence_role_profile_spec_config(build: BuildRig) -> None:
    spec = spec_with(timeout_seconds=300, model_profiles={"slow": ModelProfile(timeout_seconds=600)})
    bare = replace(spec_with("q"), credential_key="Q_KEY")
    rig = build(
        server_keys={"P_KEY": "a", "Q_KEY": "b"},
        specs=[spec, bare],
        default_provider="p",
        timeout_seconds=45,
        roles={"r": RoleSettings(timeout_seconds=5)},
    )

    assert rig.service.resolve(LLMSelection("q", "m")).timeout_seconds == 45  # config
    assert rig.service.resolve(LLMSelection("p", "m")).timeout_seconds == 300  # spec over config
    assert rig.service.resolve(LLMSelection("p", "slow")).timeout_seconds == 600  # profile over spec
    assert rig.service.resolve(LLMSelection("p", "slow"), role="r").timeout_seconds == 5  # role over profile


# -- the rest of ResolvedLLM ------------------------------------------------------------------


def test_self_hosted_provider_resolves_like_any_keyed_provider(build: BuildRig) -> None:
    resolved = build().service.resolve(LLMSelection("my-vllm", "Qwen/Qwen3-8B"))

    assert resolved.origin == "server"
    assert resolved.base_url == "http://vllm.internal:8000/v1"
    assert resolved.timeout_seconds == 300
    assert resolved.auth_header == "X-Api-Key"
    assert resolved.context_window == 32768  # from the "Qwen/*" profile


def test_headers_and_auth_header_come_from_the_spec(build: BuildRig) -> None:
    rig = build()
    vllm = rig.service.resolve(LLMSelection("my-vllm", "Qwen/Qwen3-8B"))
    openai = rig.service.resolve(LLMSelection("openai", "gpt-4o"))

    assert dict(vllm.headers) == {"X-Team": "ai"}
    assert (openai.auth_header, dict(openai.headers)) == (None, {})
    assert vllm.spec.id == "my-vllm" and vllm.selection == LLMSelection("my-vllm", "Qwen/Qwen3-8B")


def test_unknown_role_raises_configuration_error(build: BuildRig) -> None:
    rig = build(roles={"judge": RoleSettings()})

    with pytest.raises(ConfigurationError):
        rig.service.resolve(LLMSelection("openai", "gpt-4o"), role="nope")


def test_resolve_touches_nothing_but_server_credentials(build: BuildRig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = build()

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("resolve opened a socket")

    monkeypatch.setattr(socket, "socket", no_network)
    resolved = rig.service.resolve(LLMSelection("openai", "gpt-4o"))

    assert resolved.origin == "server"
    assert rig.credentials.asked == ["OPENAI_API_KEY"]  # the one key the provider names, once
    assert rig.sink.events == []  # no event for an ordinary resolve


def test_resolve_asks_the_server_for_no_key_when_the_user_supplies_one(build: BuildRig) -> None:
    rig = build()

    rig.service.resolve(LLMSelection("openai", "gpt-4o"), user_credential=user())

    assert rig.credentials.asked == []


def test_resolve_of_an_unknown_provider_raises(build: BuildRig) -> None:
    with pytest.raises(UnknownProviderError):
        build().service.resolve(LLMSelection("nope", "m"))


def test_fixture_has_the_five_providers() -> None:
    assert [s.id for s in fixture_specs()] == ["anthropic", "openai", "ollama", "lmstudio", "my-vllm"]
