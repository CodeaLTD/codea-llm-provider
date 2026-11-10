"""codea-llm-provider: a framework-agnostic LLM library.

The core is standard library only. Framework adapters live in ``llm_provider.integrations``
and are installed through extras.
"""

__version__ = "0.1.0.dev0"

from .config import LLMProviderConfig
from .credentials import MappingServerCredentials
from .errors import (
    ConfigurationError,
    CredentialNotApplicableError,
    LLMProviderError,
    MissingCredentialError,
    UnknownProviderError,
    UserCredentialUnavailableError,
)
from .interfaces import ClientAdapter, ProviderRegistry, ServerCredentials, TelemetrySink
from .registry import InMemoryProviderRegistry
from .secret import Secret
from .service import LLMProviderService
from .telemetry import (
    CallCompleted,
    CallFailed,
    CallStarted,
    CredentialFallback,
    LLMEvent,
    LoggingSink,
    Telemetry,
    UsageCollector,
    event_name,
    scrub,
)
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
    "CallCompleted",
    "CallFailed",
    "CallStarted",
    "ClientAdapter",
    "ConfigurationError",
    "CredentialFallback",
    "CredentialNotApplicableError",
    "CredentialUnavailable",
    "InMemoryProviderRegistry",
    "LLMEvent",
    "LLMProviderConfig",
    "LLMProviderError",
    "LLMProviderService",
    "LLMSelection",
    "LoggingSink",
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
    "Telemetry",
    "TelemetrySink",
    "UnknownProviderError",
    "Usage",
    "UsageCollector",
    "UserCredentialUnavailableError",
    "__version__",
    "event_name",
    "scrub",
]
