"""codea-llm-provider: a framework-agnostic LLM library.

The core is standard library only. Framework adapters live in ``llm_provider.integrations``
and are installed through extras.
"""

__version__ = "0.1.0.dev0"

from .credentials import MappingServerCredentials
from .errors import (
    ConfigurationError,
    CredentialNotApplicableError,
    LLMProviderError,
    MissingCredentialError,
    UnknownProviderError,
    UserCredentialUnavailableError,
)
from .interfaces import ClientAdapter, ProviderRegistry, ServerCredentials
from .registry import InMemoryProviderRegistry
from .secret import Secret
from .types import (
    OMIT,
    CredentialUnavailable,
    LLMSelection,
    ModelProfile,
    Omit,
    ProviderOption,
    ProviderSpec,
    ResolvedLLM,
    RoleSettings,
    StoredCredential,
    Usage,
)

__all__ = [
    "OMIT",
    "ClientAdapter",
    "ConfigurationError",
    "CredentialNotApplicableError",
    "CredentialUnavailable",
    "InMemoryProviderRegistry",
    "LLMProviderError",
    "LLMSelection",
    "MappingServerCredentials",
    "MissingCredentialError",
    "ModelProfile",
    "Omit",
    "ProviderOption",
    "ProviderRegistry",
    "ProviderSpec",
    "ResolvedLLM",
    "RoleSettings",
    "Secret",
    "ServerCredentials",
    "StoredCredential",
    "UnknownProviderError",
    "Usage",
    "UserCredentialUnavailableError",
    "__version__",
]
