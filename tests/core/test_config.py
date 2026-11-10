"""LLMProviderConfig (Task 5 Step 1; AC-04.7)."""

from __future__ import annotations

import dataclasses

import pytest

from llm_provider import ConfigurationError, RoleSettings
from llm_provider.config import LLMProviderConfig


def test_config_defaults() -> None:
    config = LLMProviderConfig(default_provider="anthropic")

    assert config.default_model is None
    assert config.default_base_url is None
    assert config.max_tokens is None
    assert config.timeout_seconds == 60.0
    assert config.default_params == {}
    assert config.roles == {}
    assert config.on_unavailable_user_credential == "raise"


@pytest.mark.parametrize("timeout", [0, -1, -0.5])
def test_config_rejects_non_positive_timeout_and_unknown_policy(timeout: float) -> None:
    with pytest.raises(ConfigurationError):
        LLMProviderConfig(default_provider="anthropic", timeout_seconds=timeout)

    with pytest.raises(ConfigurationError):
        LLMProviderConfig(default_provider="anthropic", on_unavailable_user_credential="ignore")  # type: ignore[arg-type]


def test_config_accepts_both_policies() -> None:
    raising = LLMProviderConfig(default_provider="a", on_unavailable_user_credential="raise")
    falling_back = LLMProviderConfig(default_provider="a", on_unavailable_user_credential="use_server")

    assert (raising.on_unavailable_user_credential, falling_back.on_unavailable_user_credential) == (
        "raise",
        "use_server",
    )


def test_config_is_frozen_and_holds_its_mappings_read_only() -> None:
    source = {"temperature": 0.7}
    config = LLMProviderConfig(default_provider="a", default_params=source, roles={"judge": RoleSettings()})
    source["temperature"] = 1.0

    assert config.default_params["temperature"] == 0.7
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_tokens = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.default_params["x"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        config.roles["x"] = RoleSettings()  # type: ignore[index]
