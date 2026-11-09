"""``scrub``: take secrets out of text before it reaches an event or a log line (GC Secrets)."""

from __future__ import annotations

import re
from collections.abc import Iterable

_MASK = "***"
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_SK_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{5,}")  # "sk-" and at least 5 more: 8 characters or more
_API_KEY_VALUE = re.compile(r"""(?i)((?:x-api-key|api[_-]key)["']?\s*[:=]\s*["']?)[^\s"',;&})\]]+""")


def scrub(text: str, secrets: Iterable[str] = (), *, limit: int = 300) -> str:
    """Replace live secrets, bearer tokens, ``sk-`` tokens and ``api_key`` values, then cut to ``limit``.

    ``secrets`` are plain strings: the caller holds the live key. Longer secrets go first, so one
    that contains another leaves no tail. The cut comes last, so no piece of a secret survives it.
    """
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        text = text.replace(secret, _MASK)
    text = _BEARER.sub(f"Bearer {_MASK}", text)
    text = _SK_TOKEN.sub(f"sk-{_MASK}", text)
    text = _API_KEY_VALUE.sub(rf"\g<1>{_MASK}", text)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…" if limit > 0 else ""
