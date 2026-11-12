"""A host with no framework: resolve a call, send it with any HTTP client, report it.

The library picks the provider, model and key (``LLMProviderService``) and reports the call
(``Telemetry.track_call``). Sending is yours: here it is ``urllib``, but any HTTP client works.

Run it (it starts its own stub server, so it needs no network and no real key)::

    python examples/no_framework.py
"""

from __future__ import annotations

import json
import urllib.request

from llm_provider import (
    InMemoryProviderRegistry,
    LLMProviderConfig,
    LLMProviderService,
    MappingServerCredentials,
    ProviderSpec,
    Secret,
    StoredCredential,
    Telemetry,
    Usage,
)
from llm_provider.testing import RecordingSink, StubLLMServer

DEMO_KEY = "demo-key-0001"


def ask(service: LLMProviderService, telemetry: Telemetry, prompt: str, user_key: str) -> str:
    """One call: resolve it, send it inside ``track_call``, and set ``call.usage`` from the reply."""
    resolved = service.resolve(
        service.select(),
        user_credential=StoredCredential(Secret(user_key)),  # the user's key, already decrypted by the host
        max_tokens=64,
    )
    payload = {
        "model": resolved.model,
        "messages": [{"role": "user", "content": prompt}],
        resolved.max_tokens_param: resolved.max_tokens,
        **resolved.params,
    }
    headers = {"Content-Type": "application/json", **resolved.headers}
    if resolved.api_key is not None:  # the host reveals the key only here, where it builds the request
        headers[resolved.auth_header or "Authorization"] = (
            resolved.api_key.reveal() if resolved.auth_header else f"Bearer {resolved.api_key.reveal()}"
        )
    request = urllib.request.Request(
        f"{resolved.base_url}/chat/completions", data=json.dumps(payload).encode(), headers=headers
    )

    with telemetry.track_call(resolved, stage="example") as call:
        with urllib.request.urlopen(request, timeout=resolved.timeout_seconds) as response:
            reply = json.load(response)
        usage = reply.get("usage") or {}
        call.usage = Usage(usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens"))
        call.finish_reason = reply["choices"][0]["finish_reason"]
    return str(reply["choices"][0]["message"]["content"])


def main() -> RecordingSink:
    """Run one call against a stub server and return the sink that recorded its events."""
    sink = RecordingSink()
    telemetry = Telemetry([sink])
    with StubLLMServer(require_header=("Authorization", f"Bearer {DEMO_KEY}")) as stub:
        service = LLMProviderService(
            config=LLMProviderConfig(default_provider="demo"),
            registry=InMemoryProviderRegistry(
                [
                    ProviderSpec(
                        id="demo",
                        display_name="Demo",
                        default_model="demo-model",
                        api="openai_compatible",
                        credential_key="DEMO_API_KEY",
                        default_base_url=stub.url,
                    )
                ]
            ),
            server_credentials=MappingServerCredentials({}),  # no server key: the user's key is used
            telemetry=telemetry,
        )
        answer = ask(service, telemetry, "Say hello.", DEMO_KEY)

    print("answer:", answer)
    for event in sink.events:
        print(type(event).__name__, event)
    return sink


if __name__ == "__main__":
    main()
