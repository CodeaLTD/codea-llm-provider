"""Shared fixtures for the configuration and resolution tests (Task 5 Step 1)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from llm_provider import (
    InMemoryProviderRegistry,
    MappingServerCredentials,
    ModelProfile,
    ProviderSpec,
    Telemetry,
)
from llm_provider.config import LLMProviderConfig
from llm_provider.service import LLMProviderService
from llm_provider.testing import RecordingSink

SERVER_KEYS: Mapping[str, str] = {
    "ANTHROPIC_API_KEY": "sk-server-anthropic",
    "OPENAI_API_KEY": "sk-server-openai",
    "MY_VLLM_API_KEY": "sk-server-vllm",
}


def fixture_specs() -> list[ProviderSpec]:
    return [
        ProviderSpec(
            id="anthropic",
            display_name="Anthropic",
            default_model="claude-sonnet-4-5",
            api="anthropic",
            credential_key="ANTHROPIC_API_KEY",
        ),
        ProviderSpec(
            id="openai",
            display_name="OpenAI",
            default_model="gpt-4o",
            api="openai",
            credential_key="OPENAI_API_KEY",
        ),
        ProviderSpec(
            id="ollama",
            display_name="Ollama",
            default_model="llama3.1",
            api="openai_compatible",
            credential_key=None,
            default_base_url="http://localhost:11434/v1",
        ),
        ProviderSpec(
            id="lmstudio",
            display_name="LM Studio",
            default_model="local-model",
            api="openai_compatible",
            credential_key=None,
            default_base_url="http://localhost:1234/v1",
            placeholder_api_key="lm-studio",
        ),
        ProviderSpec(
            id="my-vllm",
            display_name="My vLLM",
            default_model="Qwen/Qwen3-8B",
            api="openai_compatible",
            credential_key="MY_VLLM_API_KEY",
            default_base_url="http://vllm.internal:8000/v1",
            auth_header="X-Api-Key",
            default_headers={"X-Team": "ai"},
            timeout_seconds=300,
            model_profiles={"Qwen/*": ModelProfile(context_window=32768)},
        ),
    ]


class CountingCredentials:
    """Server credentials that record every key asked for."""

    def __init__(self, values: Mapping[str, str]) -> None:
        self._inner = MappingServerCredentials(values)
        self.asked: list[str] = []

    def get(self, credential_key: str) -> Any:
        self.asked.append(credential_key)
        return self._inner.get(credential_key)


class Rig:
    """A service with the sink its telemetry feeds."""

    def __init__(
        self,
        service: LLMProviderService,
        sink: RecordingSink,
        credentials: CountingCredentials,
        telemetry: Telemetry,
    ) -> None:
        self.service = service
        self.sink = sink
        self.credentials = credentials
        self.telemetry = telemetry


BuildRig = Callable[..., Rig]


@pytest.fixture
def build() -> BuildRig:
    """``build(server_keys=..., specs=..., **config)``: the default provider is ``anthropic``."""

    def _build(
        *,
        server_keys: Mapping[str, str] | None = None,
        specs: Sequence[ProviderSpec] | None = None,
        **config: Any,
    ) -> Rig:
        sink = RecordingSink()
        credentials = CountingCredentials(SERVER_KEYS if server_keys is None else server_keys)
        config.setdefault("default_provider", "anthropic")
        telemetry = Telemetry([sink])
        service = LLMProviderService(
            config=LLMProviderConfig(**config),
            registry=InMemoryProviderRegistry(fixture_specs() if specs is None else specs),
            server_credentials=credentials,
            telemetry=telemetry,
        )
        return Rig(service, sink, credentials, telemetry)

    return _build
