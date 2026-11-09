"""Value types (Task 2 Step 1; §Core types)."""

from __future__ import annotations

import dataclasses

import pytest

from llm_provider import (
    OMIT,
    CredentialUnavailable,
    LLMSelection,
    ModelProfile,
    ProviderOption,
    ProviderSpec,
    ResolvedLLM,
    RoleSettings,
    Secret,
    StoredCredential,
    Usage,
)

KEY = "sk-abc12345"


def make_spec(**overrides: object) -> ProviderSpec:
    values: dict[str, object] = {
        "id": "anthropic",
        "display_name": "Anthropic",
        "default_model": "claude-x",
        "api": "anthropic",
        "credential_key": "ANTHROPIC_API_KEY",
    }
    values.update(overrides)
    return ProviderSpec(**values)  # type: ignore[arg-type]


def test_resolved_llm_repr_hides_the_key() -> None:
    spec = make_spec()
    resolved = ResolvedLLM(
        spec=spec,
        selection=LLMSelection(provider="anthropic", model="claude-x"),
        api_key=Secret(KEY),
        base_url=None,
        max_tokens=1024,
        max_tokens_param="max_tokens",
        timeout_seconds=60.0,
        params={"temperature": 0.2},
        dropped_params=frozenset(),
        headers={"X-Trace": "on"},
        auth_header=None,
        context_window=None,
        origin="server",
    )

    assert KEY not in repr(resolved)
    assert KEY not in str(resolved)
    assert resolved.api_key is not None and resolved.api_key.reveal() == KEY
    assert (resolved.provider, resolved.model, resolved.api) == ("anthropic", "claude-x", "anthropic")
    assert resolved.degraded_reason is None


def test_provider_spec_attributes_are_read_only() -> None:
    spec = make_spec(
        default_headers={"X-A": "1"},
        default_params={"temperature": 0},
        model_profiles={"m": ModelProfile()},
        attributes={"where": "here"},
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.display_name = "other"  # type: ignore[misc]
    for field in ("default_headers", "default_params", "model_profiles", "attributes"):
        with pytest.raises(TypeError):
            getattr(spec, field)["new"] = "value"


def test_mapping_fields_do_not_follow_the_callers_dict() -> None:
    headers = {"X-A": "1"}
    spec = make_spec(default_headers=headers)

    headers["X-B"] = "2"

    assert dict(spec.default_headers) == {"X-A": "1"}


def test_requires_key_iff_credential_key() -> None:
    assert make_spec(credential_key="ANTHROPIC_API_KEY").requires_key is True
    assert make_spec(id="ollama", api="ollama", credential_key=None).requires_key is False


def test_auth_header_without_a_credential_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_spec(credential_key=None, auth_header="X-Api-Key")

    assert make_spec(auth_header="X-Api-Key").auth_header == "X-Api-Key"


def test_default_headers_may_not_hold_the_auth_header() -> None:
    with pytest.raises(ValueError):
        make_spec(auth_header="X-Api-Key", default_headers={"X-Api-Key": "nope"})
    with pytest.raises(ValueError):  # header names are case-insensitive
        make_spec(auth_header="X-Api-Key", default_headers={"x-api-key": "nope"})

    spec = make_spec(auth_header="X-Api-Key", default_headers={"X-Other": "ok"})
    assert spec.default_headers["X-Other"] == "ok"


def test_profile_for_prefers_an_exact_match_then_the_first_matching_pattern() -> None:
    a, b, c = ModelProfile(context_window=1), ModelProfile(context_window=2), ModelProfile(context_window=3)
    spec = make_spec(model_profiles={"o3*": a, "o3-mini": b, "o*": c})

    assert spec.profile_for("o3-mini") is b
    assert spec.profile_for("o3-pro") is a
    assert spec.profile_for("o1") is c
    assert spec.profile_for("gpt-4o") == ModelProfile()


def test_usage_is_not_estimated_by_default() -> None:
    usage = Usage()

    assert usage.estimated is False
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (None, None, None)


def test_omit_is_a_singleton_that_reprs_as_omit() -> None:
    assert repr(OMIT) == "OMIT"
    assert type(OMIT)() is OMIT
    assert RoleSettings(max_tokens=OMIT).max_tokens is OMIT
    assert RoleSettings().max_tokens is None


def test_model_profile_defaults_and_read_only_params() -> None:
    profile = ModelProfile(params={"top_p": 1}, drop_params={"temperature"})  # type: ignore[arg-type]

    assert profile.max_tokens_param == "max_tokens"
    assert profile.drop_params == frozenset({"temperature"})
    assert profile.timeout_seconds is None and profile.context_window is None
    with pytest.raises(TypeError):
        profile.params["x"] = 1  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.context_window = 5  # type: ignore[misc]


def test_stored_credential_wraps_a_plain_key_and_hides_it() -> None:
    wrapped = StoredCredential(KEY)  # type: ignore[arg-type]
    already = StoredCredential(Secret(KEY), base_url="http://my-host:1234/v1")

    assert isinstance(wrapped.api_key, Secret) and wrapped.api_key.reveal() == KEY
    assert already.base_url == "http://my-host:1234/v1"
    assert KEY not in repr(wrapped)


def test_small_values_are_frozen_and_keep_their_fields() -> None:
    selection = LLMSelection(provider="p", model="m")
    unavailable = CredentialUnavailable(reason="no longer decrypts")
    option = ProviderOption(
        id="p",
        display_name="P",
        default_model="m",
        api="openai",
        requires_key=True,
        available=True,
        configured_by_user=False,
        attributes={"k": "v"},
    )

    assert selection.base_url is None
    assert unavailable.reason == "no longer decrypts"
    with pytest.raises(dataclasses.FrozenInstanceError):
        selection.model = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        option.attributes["x"] = 1  # type: ignore[index]


def test_resolved_llm_and_role_settings_copy_their_mappings_read_only() -> None:
    params = {"temperature": 0.2}
    headers = {"X-Trace": "on"}
    dropped = {"top_p"}
    resolved = ResolvedLLM(
        spec=make_spec(),
        selection=LLMSelection(provider="anthropic", model="claude-x"),
        api_key=None,
        base_url=None,
        max_tokens=None,
        max_tokens_param="max_tokens",
        timeout_seconds=60.0,
        params=params,
        dropped_params=dropped,  # type: ignore[arg-type]
        headers=headers,
        auth_header=None,
        context_window=None,
        origin="none",
    )
    role = RoleSettings(params=params)

    params["top_k"] = 1
    headers["X-Other"] = "x"
    dropped.add("top_k")

    assert dict(resolved.params) == {"temperature": 0.2}
    assert dict(resolved.headers) == {"X-Trace": "on"}
    assert resolved.dropped_params == frozenset({"top_p"})
    assert dict(role.params) == {"temperature": 0.2}
    for mapping in (resolved.params, resolved.headers, role.params):
        with pytest.raises(TypeError):
            mapping["new"] = "value"  # type: ignore[index]
