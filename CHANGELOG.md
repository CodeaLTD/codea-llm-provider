# Changelog

All notable changes to this project are recorded here.

## Unreleased

- Resolution: `LLMProviderConfig` (defaults, roles, timeout and the `on_unavailable_user_credential` policy) and `LLMProviderService` with `default_selection`, `select`, `resolve`, `list_options`, `check_credential_target` and `client`; `resolve` turns the settings and a user's key into a `ResolvedLLM` by fixed precedence rules and does no I/O.
- Observability: `Telemetry` hub (sinks with an `emit(event)` method, `scope`, `collect`), the `CallStarted`, `CallCompleted`, `CallFailed` and `CredentialFallback` events, `scrub`, `LoggingSink`, the `TelemetrySink` protocol and `RecordingSink` for tests.
- Core types and registry: `Secret`, `OMIT`, `ProviderSpec`, `ModelProfile`, `LLMSelection`, `StoredCredential`, `CredentialUnavailable`, `Usage`, `ResolvedLLM`, `ProviderOption` and `RoleSettings` as frozen values; the `LLMProviderError` family; the `ProviderRegistry`, `ServerCredentials` and `ClientAdapter` protocols; `InMemoryProviderRegistry` and `MappingServerCredentials`.
- Package scaffold: `llm_provider` imports with no third-party module, and CI checks the boundary rules.
