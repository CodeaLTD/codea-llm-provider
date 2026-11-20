"""Resilience: classify a failed call and read its ``Retry-After`` (§Resilience).

Plan 07 builds the first half, the error kinds. Plans 08 and 10 add the policy, the caller and the
checked replies.
"""

from .classify import Classifier, ErrorInfo, classify_error, kind_for_finish_reason
from .kinds import ErrorKind
from .retry_after import parse_retry_after

__all__ = [
    "Classifier",
    "ErrorInfo",
    "ErrorKind",
    "classify_error",
    "kind_for_finish_reason",
    "parse_retry_after",
]
