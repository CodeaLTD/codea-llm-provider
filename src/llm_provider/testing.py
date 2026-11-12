"""Helpers for tests: ``RecordingSink`` and ``StubLLMServer`` (standard library only).

``StubLLMServer`` is a small local HTTP server that speaks the OpenAI Chat Completions, Anthropic
Messages and Gemini ``generateContent`` formats, so a test can play any provider without a network.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType, TracebackType
from typing import Any, Self

from .telemetry.events import LLMEvent

_VENDORS: tuple[str, ...] = ("openai", "anthropic", "gemini")


class RecordingSink:
    """A telemetry sink that keeps every event it receives. Safe to share between threads."""

    def __init__(self) -> None:
        self.events: list[LLMEvent] = []
        self._lock = threading.Lock()

    def emit(self, event: LLMEvent) -> None:
        with self._lock:
            self.events.append(event)


# -- what the stub says -------------------------------------------------------------------------

DEFAULT_USAGE: Mapping[str, int] = MappingProxyType({"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18})
"""What the stub reports by default. Usage is always written with the OpenAI names; each format maps them."""

_REPLY_TEXT = ("stub", " reply")
_ANTHROPIC_USAGE_NAMES = {"prompt_tokens": "input_tokens", "completion_tokens": "output_tokens"}
_GEMINI_USAGE_NAMES = {
    "prompt_tokens": "promptTokenCount",
    "completion_tokens": "candidatesTokenCount",
    "total_tokens": "totalTokenCount",
}

# kind -> vendor -> (HTTP status, error type (for Gemini: its status word), OpenAI error code, message)
_Failure = tuple[int, str, str | None, str]
_FAILURES: dict[str, dict[str, _Failure]] = {
    "quota_exhausted": {
        "openai": (
            429,
            "insufficient_quota",
            "insufficient_quota",
            "You exceeded your current quota, please check your plan and billing details.",
        ),
        "anthropic": (
            400,
            "invalid_request_error",
            None,
            (
                "Your credit balance is too low to access the Anthropic API. "
                "Please go to Plans & Billing to upgrade or purchase credits."
            ),
        ),
        "gemini": (
            429,
            "RESOURCE_EXHAUSTED",
            None,
            "You exceeded your current quota, please check your plan and billing details.",
        ),
    },
    "overloaded": {
        "openai": (503, "server_error", None, "The engine is currently overloaded, please try again later."),
        "anthropic": (529, "overloaded_error", None, "Overloaded"),
        "gemini": (503, "UNAVAILABLE", None, "The model is overloaded. Please try again later."),
    },
    "context_overflow": {
        "openai": (
            400,
            "invalid_request_error",
            "context_length_exceeded",
            (
                "This model's maximum context length is 128000 tokens. However, your messages resulted in "
                "130000 tokens. Please reduce the length of the messages."
            ),
        ),
        "anthropic": (400, "invalid_request_error", None, "prompt is too long: 210000 tokens > 200000 maximum"),
        "gemini": (
            400,
            "INVALID_ARGUMENT",
            None,
            "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).",
        ),
    },
    "model_not_found": {
        "openai": (
            404,
            "invalid_request_error",
            "model_not_found",
            "The model does not exist or you do not have access to it.",
        ),
        "anthropic": (404, "not_found_error", None, "model: unknown-model"),
        "gemini": (
            404,
            "NOT_FOUND",
            None,
            "models/unknown-model is not found for API version v1beta, or is not supported for generateContent.",
        ),
    },
    "content_blocked": {  # Gemini blocks a prompt with a 200 and ``promptFeedback``, so it has no entry here
        "openai": (
            400,
            "invalid_request_error",
            "content_policy_violation",
            "Your request was rejected as a result of our safety system.",
        ),
        "anthropic": (400, "invalid_request_error", None, "Output blocked by content filtering policy"),
    },
}

_ANTHROPIC_TYPES = {
    401: "authentication_error",
    404: "not_found_error",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}
_GEMINI_STATUSES = {
    401: "UNAUTHENTICATED",
    404: "NOT_FOUND",
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
    503: "UNAVAILABLE",
}
_AUTH_HEADERS = ("authorization", "x-api-key", "api-key", "x-goog-api-key")


def _error_body(vendor: str, status: int, kind: str, code: str | None, message: str) -> dict[str, Any]:
    """An error body in ``vendor``'s own shape. ``kind`` is the error type (for Gemini: its status word)."""
    if vendor == "anthropic":
        return {"type": "error", "error": {"type": kind, "message": message}}
    if vendor == "gemini":
        return {"error": {"code": status, "message": message, "status": kind}}
    return {"error": {"message": message, "type": kind, "param": None, "code": code}}


def _vendor_failure(kind: str, vendor: str) -> StubReply:
    if vendor not in _VENDORS:
        raise ValueError(f"vendor must be one of {_VENDORS}, got {vendor!r}")
    status, error_kind, code, message = _FAILURES[kind][vendor]
    return StubReply(status=status, body=_error_body(vendor, status, error_kind, code, message))


def _openai_chunk(model: str, **fields: Any) -> dict[str, Any]:
    return {"id": "chatcmpl-stub", "object": "chat.completion.chunk", "created": 0, "model": model, **fields}


def _delta_chunk(text: str, model: str = "stub") -> str:
    chunk = _openai_chunk(model, choices=[{"index": 0, "delta": {"content": text}, "finish_reason": None}])
    return f"data: {json.dumps(chunk)}\n\n"


@dataclass(frozen=True)
class StubReply:
    """One scripted answer. With no ``body`` and no ``chunks``, a 200 is a normal reply and an error status a generic one.

    - ``body``: a ``dict`` or ``list`` is sent as JSON; ``str`` or ``bytes`` as they are.
    - ``chunks``: sent as a server-sent-events stream, one piece at a time.
    - ``drop_after_chunks``: close the connection after that many chunks, with no end marker.
    """

    status: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)
    body: Any = None
    chunks: Sequence[str | bytes] | None = None
    drop_after_chunks: int | None = None

    @classmethod
    def rate_limited(cls, *, retry_after: float | str | None = None, retry_after_ms: float | None = None) -> StubReply:
        """429, with ``Retry-After`` (seconds, or any text such as an HTTP date) and ``retry-after-ms`` if given."""
        headers: dict[str, str] = {}
        if retry_after is not None:
            headers["Retry-After"] = f"{retry_after:g}" if isinstance(retry_after, float) else str(retry_after)
        if retry_after_ms is not None:
            headers["retry-after-ms"] = f"{retry_after_ms:g}"
        body = _error_body("openai", 429, "requests", "rate_limit_exceeded", "Rate limit reached for requests")
        return cls(status=429, headers=headers, body=body)

    @classmethod
    def quota_exhausted(cls, vendor: str) -> StubReply:
        return _vendor_failure("quota_exhausted", vendor)

    @classmethod
    def overloaded(cls, vendor: str) -> StubReply:
        return _vendor_failure("overloaded", vendor)

    @classmethod
    def context_overflow(cls, vendor: str) -> StubReply:
        return _vendor_failure("context_overflow", vendor)

    @classmethod
    def model_not_found(cls, vendor: str) -> StubReply:
        return _vendor_failure("model_not_found", vendor)

    @classmethod
    def content_blocked(cls, vendor: str) -> StubReply:
        if vendor == "gemini":
            return cls(body={"promptFeedback": {"blockReason": "SAFETY"}})
        return _vendor_failure("content_blocked", vendor)

    @classmethod
    def stream_then_drop(cls, n: int) -> StubReply:
        """A stream that sends ``n`` chunks and then breaks the connection."""
        chunks = [_delta_chunk(f"{i} ") for i in range(n + 1)] + ["data: [DONE]\n\n"]
        return cls(chunks=chunks, drop_after_chunks=n)


@dataclass(frozen=True)
class RecordedRequest:
    """What the stub received: the path, the headers (names in lower case) and the JSON body."""

    path: str
    headers: Mapping[str, str]
    body: Any


def _last_user_text(vendor: str, body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    if vendor == "gemini":
        turns, parts_key = body.get("contents"), "parts"
        if not isinstance(turns, list):
            return ""
        for turn in reversed(turns):
            if isinstance(turn, dict) and turn.get("role", "user") == "user":
                return " ".join(p.get("text", "") for p in turn.get(parts_key, []) if isinstance(p, dict))
        return ""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return str(content)
    return ""


def _mapped_usage(usage: Mapping[str, int], names: Mapping[str, str]) -> dict[str, int]:
    return {names[key]: value for key, value in usage.items() if key in names}


class StubLLMServer:
    """A local server that plays an LLM provider, for tests. Use it as a context manager.

    - ``url`` ends in ``/v1`` (``POST {url}/chat/completions``, OpenAI format, also streamed);
      ``anthropic_url`` serves ``POST {anthropic_url}/v1/messages``; ``gemini_url`` serves
      ``POST {gemini_url}/v1beta/models/{model}:generateContent``.
    - ``usage`` is written with the OpenAI names (``prompt_tokens``, ``completion_tokens``,
      ``total_tokens``) and sent in each format's own names, as a dict is given: zeros and partial
      dicts work. ``None`` sends no usage anywhere. A stream carries a usage chunk only when the
      request has ``stream_options.include_usage``.
    - ``require_header=(name, value)`` answers 401 unless the request carries that header.
    - ``fail_with_status`` answers every request with that status.
    - ``script`` answers requests in order; after the last scripted reply the stub answers normally.
    - A failure body echoes the auth header received and the text of the last user message, as some
      servers' validation errors do.
    """

    def __init__(
        self,
        *,
        delay_seconds: float = 0.0,
        fail_with_status: int | None = None,
        require_header: tuple[str, str] | None = None,
        usage: Mapping[str, int] | None = DEFAULT_USAGE,
        script: Sequence[StubReply] = (),
    ) -> None:
        self._delay = delay_seconds
        self._fail_with_status = fail_with_status
        self._require_header = require_header
        self._usage = None if usage is None else dict(usage)
        self._script = list(script)
        self._lock = threading.Lock()
        self._requests: list[RecordedRequest] = []
        self._scripted_so_far = 0
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------------------------

    def __enter__(self) -> Self:
        server = _Server(("127.0.0.1", 0), _Handler, self)
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        server, thread = self._server, self._thread
        if server is None or thread is None:
            return
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        self._server = self._thread = None

    @property
    def _root(self) -> str:
        if self._server is None:
            raise RuntimeError("StubLLMServer is not running: use it in a `with` block")
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    @property
    def url(self) -> str:
        return f"{self._root}/v1"

    @property
    def anthropic_url(self) -> str:
        return self._root

    @property
    def gemini_url(self) -> str:
        return self._root

    # -- what it saw ---------------------------------------------------------------------------

    @property
    def requests(self) -> list[RecordedRequest]:
        with self._lock:
            return list(self._requests)

    @property
    def request_count(self) -> int:
        with self._lock:
            return len(self._requests)

    # -- answering -----------------------------------------------------------------------------

    def _handle(self, path: str, headers: Mapping[str, str], body: Any) -> StubReply:
        """Record one request and decide the reply: the auth check, then the script, then a forced failure."""
        with self._lock:
            self._requests.append(RecordedRequest(path, MappingProxyType(dict(headers)), body))
        vendor = _vendor_of(path)
        if vendor is None:
            return StubReply(status=404, body={"error": f"the stub has no route {path}"})
        if self._delay:
            time.sleep(self._delay)
        if self._require_header is not None:
            name, value = self._require_header
            if headers.get(name.lower()) != value:
                return self._failure(vendor, 401, headers, body)
        with self._lock:
            scripted = self._script[self._scripted_so_far] if self._scripted_so_far < len(self._script) else None
            if scripted is not None:
                self._scripted_so_far += 1
        if scripted is not None:
            return self._scripted_reply(scripted, vendor, headers, body)
        if self._fail_with_status is not None:
            return self._failure(vendor, self._fail_with_status, headers, body)
        return self._normal(vendor, body)

    def _scripted_reply(self, scripted: StubReply, vendor: str, headers: Mapping[str, str], body: Any) -> StubReply:
        """A reply with no body and no chunks stands for the stub's own: a normal reply, or a generic failure."""
        if scripted.body is not None or scripted.chunks is not None:
            return scripted
        if scripted.status >= 400:
            return self._failure(vendor, scripted.status, headers, body, scripted.headers)
        plain = self._normal(vendor, body)
        return replace(plain, status=scripted.status, headers={**plain.headers, **scripted.headers})

    def _failure(
        self,
        vendor: str,
        status: int,
        headers: Mapping[str, str],
        body: Any,
        extra_headers: Mapping[str, str] | None = None,
    ) -> StubReply:
        names = [self._require_header[0].lower()] if self._require_header else []
        auth = next(
            (f"{name}: {headers[name]}" for name in (*names, *_AUTH_HEADERS) if name in headers), "(no auth header)"
        )
        message = f"Invalid request. auth header: {auth}. last user message: {_last_user_text(vendor, body)}"
        if vendor == "anthropic":
            kind = _ANTHROPIC_TYPES.get(status, "invalid_request_error")
        elif vendor == "gemini":
            kind = _GEMINI_STATUSES.get(status, "INVALID_ARGUMENT")
        else:
            kind = "server_error" if status >= 500 else "invalid_request_error"
        return StubReply(
            status=status, headers=dict(extra_headers or {}), body=_error_body(vendor, status, kind, None, message)
        )

    def _normal(self, vendor: str, body: Any) -> StubReply:
        request = body if isinstance(body, dict) else {}
        model = str(request.get("model", "stub"))
        usage = self._usage
        text = "".join(_REPLY_TEXT)
        if vendor == "anthropic":
            reply: dict[str, Any] = {
                "id": "msg_stub",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
            }
            if usage is not None:
                reply["usage"] = _mapped_usage(usage, _ANTHROPIC_USAGE_NAMES)
        elif vendor == "gemini":
            reply = {
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP", "index": 0}
                ],
                "modelVersion": model,
            }
            if usage is not None:
                reply["usageMetadata"] = _mapped_usage(usage, _GEMINI_USAGE_NAMES)
        elif request.get("stream") is True:
            return StubReply(chunks=self._openai_stream(model, request))
        else:
            reply = {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            }
            if usage is not None:
                reply["usage"] = dict(usage)
        return StubReply(body=reply)

    def _openai_stream(self, model: str, request: Mapping[str, Any]) -> list[str]:
        def sse(chunk: dict[str, Any]) -> str:
            return f"data: {json.dumps(chunk)}\n\n"

        first = _openai_chunk(
            model, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
        )
        last = _openai_chunk(model, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])
        chunks = [sse(first), *(_delta_chunk(piece, model) for piece in _REPLY_TEXT), sse(last)]
        options = request.get("stream_options")
        if self._usage is not None and isinstance(options, dict) and options.get("include_usage"):
            chunks.append(sse(_openai_chunk(model, choices=[], usage=self._usage)))
        chunks.append("data: [DONE]\n\n")
        return chunks


def _vendor_of(path: str) -> str | None:
    if path.endswith("/chat/completions"):
        return "openai"
    if path.endswith("/messages"):
        return "anthropic"
    if ":generateContent" in path:
        return "gemini"
    return None


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], stub: StubLLMServer) -> None:
        super().__init__(address, handler)
        self.stub = stub


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def do_POST(self) -> None:
        stub = self.server.stub
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            body: Any = json.loads(raw) if raw else None
        except ValueError:
            body = None
        headers = {name.lower(): value for name, value in self.headers.items()}
        self._send(stub._handle(self.path.split("?", 1)[0], headers, body))

    def _send(self, reply: StubReply) -> None:
        self.close_connection = True
        self.send_response(reply.status)
        headers = {name.lower(): (name, value) for name, value in reply.headers.items()}
        if reply.chunks is not None:
            headers.setdefault("content-type", ("Content-Type", "text/event-stream"))
            self._send_headers(headers, ("Transfer-Encoding", "chunked"))
            limit = len(reply.chunks) if reply.drop_after_chunks is None else reply.drop_after_chunks
            for chunk in reply.chunks[:limit]:
                data = chunk if isinstance(chunk, bytes) else chunk.encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
            if reply.drop_after_chunks is None:
                self.wfile.write(b"0\r\n\r\n")  # the end of the chunked body; a dropped stream never sends it
            return
        if isinstance(reply.body, bytes):
            payload = reply.body
        elif isinstance(reply.body, str):
            payload = reply.body.encode()
        elif reply.body is None:
            payload = b""
        else:
            payload = json.dumps(reply.body).encode()
            headers.setdefault("content-type", ("Content-Type", "application/json"))
        self._send_headers(headers, ("Content-Length", str(len(payload))))
        self.wfile.write(payload)

    def _send_headers(self, headers: Mapping[str, tuple[str, str]], framing: tuple[str, str]) -> None:
        for name, value in headers.values():
            self.send_header(name, value)
        self.send_header(*framing)
        self.send_header("Connection", "close")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the test output quiet."""
