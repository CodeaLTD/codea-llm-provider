# Changelog

All notable changes to this project are recorded here.

## Unreleased

- Call tracking: `Telemetry.start_call` and `Telemetry.track_call` report one model call as `CallStarted`, then `CallCompleted` with latency and usage, or `CallFailed` with a scrubbed message (any `BaseException`, cancellation and timeouts included); `CallHandle` can end on any thread and keeps the attrs and `collect()` block of its start; `PendingCalls` pairs a framework's start and end callbacks by key, counts unmatched ends and reports a call with no end within its TTL once as `Abandoned`. `testing.StubLLMServer` plays the OpenAI (also streamed), Anthropic Messages and Gemini formats, with `usage=`, `require_header`, and `script=` of `StubReply` replies in each vendor's error shape. `examples/no_framework.py` shows a host with no framework.
- Resolution: `LLMProviderConfig` (defaults, roles, timeout and the `on_unavailable_user_credential` policy) and `LLMProviderService` with `default_selection`, `select`, `resolve`, `list_options`, `check_credential_target` and `client`; `resolve` turns the settings and a user's key into a `ResolvedLLM` by fixed precedence rules and does no I/O.
- Observability: `Telemetry` hub (sinks with an `emit(event)` method, `scope`, `collect`), the `CallStarted`, `CallCompleted`, `CallFailed` and `CredentialFallback` events, `scrub`, `LoggingSink`, the `TelemetrySink` protocol and `RecordingSink` for tests.
- Core types and registry: `Secret`, `OMIT`, `ProviderSpec`, `ModelProfile`, `LLMSelection`, `StoredCredential`, `CredentialUnavailable`, `Usage`, `ResolvedLLM`, `ProviderOption` and `RoleSettings` as frozen values; the `LLMProviderError` family; the `ProviderRegistry`, `ServerCredentials` and `ClientAdapter` protocols; `InMemoryProviderRegistry` and `MappingServerCredentials`.
- Package scaffold: `llm_provider` imports with no third-party module, and CI checks the boundary rules.
