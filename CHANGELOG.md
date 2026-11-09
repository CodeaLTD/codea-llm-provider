# Changelog

All notable changes to this project are recorded here.

## Unreleased

- Core types and registry: `Secret`, `OMIT`, `ProviderSpec`, `ModelProfile`, `LLMSelection`, `StoredCredential`, `CredentialUnavailable`, `Usage`, `ResolvedLLM`, `ProviderOption` and `RoleSettings` as frozen values; the `LLMProviderError` family; the `ProviderRegistry`, `ServerCredentials` and `ClientAdapter` protocols; `InMemoryProviderRegistry` and `MappingServerCredentials`.
- Package scaffold: `llm_provider` imports with no third-party module, and CI checks the boundary rules.
