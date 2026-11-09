# codea-llm-provider

Framework-agnostic Python LLM library: picks the provider, model and key, makes one request work
the same on frontier and local models, retries with `Retry-After`, fails over between providers,
and reports every call. Adapters for CrewAI and LangChain/LangGraph.

Status: under construction, not released yet. The full documentation arrives with `v0.1.0`.

## Install

```
pip install "git+https://github.com/CodeaLTD/codea-llm-provider.git"
```

The core has no dependencies. Extras: `crewai`, `langchain`.

## License

MIT. See [LICENSE](LICENSE).
