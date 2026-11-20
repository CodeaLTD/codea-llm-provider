"""``classify_error`` and ``kind_for_finish_reason`` (Task 5a Step 2; AC-07.3 to AC-07.6).

The exceptions are small fakes with the attributes the SDKs' errors carry: ``status_code``, ``code``,
``type``, ``body`` and ``response`` for OpenAI and Anthropic; ``code`` (a number), ``status`` (a word)
and ``details`` for Google's. Plans 06, 15 and 16 confirm the real shapes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any

import pytest
from conftest import BuildRig

from llm_provider import CallFailed, Secret, StoredCredential
from llm_provider.resilience import (
    ErrorInfo,
    ErrorKind,
    classify_error,
    kind_for_finish_reason,
)

NOW = datetime(2026, 11, 12, 10, 0, 0, tzinfo=UTC)
KEY = "live-key-for-the-test-0042"  # none of scrub's own patterns match it
PROMPT = "ZEBRA-prompt-fragment-9f3a"  # a distinctive piece of a user's prompt that a provider's error quotes


class FakeResponse:
    def __init__(self, status_code: int, headers: dict[str, str] | None = None, body: Any = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no body")
        return self._body


class FakeAPIError(Exception):
    """What ``openai.APIStatusError`` and ``anthropic.APIStatusError`` look like from outside."""

    def __init__(
        self,
        message: str = "provider said no",
        *,
        status_code: int | None = None,
        code: Any = None,
        type: str | None = None,
        body: Any = None,
        response: Any = None,
        headers: Any = None,
    ) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code
        if type is not None:
            self.type = type
        if body is not None:
            self.body = body
        if response is not None:
            self.response = response
        if headers is not None:
            self.headers = headers


class FakeGoogleError(Exception):
    """What ``google.genai.errors.APIError`` looks like: a numeric ``code``, a ``status`` word, ``details``."""

    def __init__(self, code: int, status: str, message: str, details: Any = None) -> None:
        super().__init__(f"{code} {status}. {message}")
        self.code = code
        self.status = status
        self.message = message
        self.details = (
            details if details is not None else {"error": {"code": code, "status": status, "message": message}}
        )


def anthropic_error(status: int, kind: str, message: str) -> FakeAPIError:
    body = {"type": "error", "error": {"type": kind, "message": message}}
    return FakeAPIError(f"Error code: {status} - {body}", status_code=status, body=body)


def google_quota_error(quota_id: str) -> FakeGoogleError:
    details = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "message": "You exceeded your current quota.",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaMetric": "generate_content_requests", "quotaId": quota_id}],
                }
            ],
        }
    }
    return FakeGoogleError(429, "RESOURCE_EXHAUSTED", "You exceeded your current quota.", details)


# -- AC-07.3 -------------------------------------------------------------------------------------

CASES: list[tuple[str, BaseException, ErrorKind]] = [
    ("429", FakeAPIError(status_code=429, code="rate_limit_exceeded"), ErrorKind.rate_limited),
    (
        "429-anthropic",
        anthropic_error(429, "rate_limit_error", "Number of request tokens exceeded"),
        ErrorKind.rate_limited,
    ),
    ("429-insufficient-quota", FakeAPIError(status_code=429, code="insufficient_quota"), ErrorKind.quota_exhausted),
    (
        "429-insufficient-quota-in-body",
        FakeAPIError(status_code=429, body={"message": "x", "type": "insufficient_quota", "code": None}),
        ErrorKind.quota_exhausted,
    ),
    (
        "429-gemini-per-day",
        google_quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
        ErrorKind.quota_exhausted,
    ),
    (
        "429-gemini-per-minute",
        google_quota_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"),
        ErrorKind.rate_limited,
    ),
    ("402-anthropic-billing", anthropic_error(402, "billing_error", "billing problem"), ErrorKind.quota_exhausted),
    (
        "400-anthropic-credit-balance",
        anthropic_error(400, "invalid_request_error", "Your credit balance is too low to access the Anthropic API."),
        ErrorKind.quota_exhausted,
    ),
    ("529", anthropic_error(529, "overloaded_error", "Overloaded"), ErrorKind.overloaded),
    ("503", FakeAPIError(status_code=503), ErrorKind.overloaded),
    ("503-gemini", FakeGoogleError(503, "UNAVAILABLE", "The model is overloaded."), ErrorKind.overloaded),
    ("502", FakeAPIError(status_code=502), ErrorKind.server_error),
    ("500", anthropic_error(500, "api_error", "Internal server error"), ErrorKind.server_error),
    ("504", FakeAPIError(status_code=504), ErrorKind.server_error),
    (
        "400-context-length-code",
        FakeAPIError("x", status_code=400, code="context_length_exceeded", type="invalid_request_error"),
        ErrorKind.context_overflow,
    ),
    (
        "400-prompt-is-too-long",
        anthropic_error(400, "invalid_request_error", "prompt is too long: 210000 tokens > 200000 maximum"),
        ErrorKind.context_overflow,
    ),
    (
        "400-vllm",  # vLLM's `code` is the number 400, and its text counts tokens
        FakeAPIError(
            "This model's maximum context length is 8192 tokens. However, your prompt contains 9000 input tokens.",
            status_code=400,
            code=400,
            type="BadRequestError",
        ),
        ErrorKind.context_overflow,
    ),
    (
        "400-sglang-flat-body",  # SGLang's body isn't nested under "error"
        FakeAPIError(
            status_code=400,
            body={
                "object": "error",
                "message": "Requested token count exceeds the model's maximum context length of 4096 tokens.",
                "type": "BadRequestError",
                "code": 400,
            },
        ),
        ErrorKind.context_overflow,
    ),
    (
        "400-gemini-input-token-count",
        FakeGoogleError(
            400,
            "INVALID_ARGUMENT",
            "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).",
        ),
        ErrorKind.context_overflow,
    ),
    (
        "413",
        FakeAPIError("Request exceeds the maximum allowed number of bytes.", status_code=413),
        ErrorKind.context_overflow,
    ),
    ("400-content-filter", FakeAPIError(status_code=400, code="content_filter"), ErrorKind.content_blocked),
    ("400-content-policy", FakeAPIError(status_code=400, code="content_policy_violation"), ErrorKind.content_blocked),
    (
        "400-anthropic-content-filtering",
        anthropic_error(400, "invalid_request_error", "Output blocked by content filtering policy"),
        ErrorKind.content_blocked,
    ),
    ("404-model-not-found", FakeAPIError(status_code=404, code="model_not_found"), ErrorKind.model_unavailable),
    ("404-anthropic", anthropic_error(404, "not_found_error", "model: unknown-model"), ErrorKind.model_unavailable),
    (
        "404-gemini",
        FakeGoogleError(404, "NOT_FOUND", "models/x is not found for API version v1beta"),
        ErrorKind.model_unavailable,
    ),
    (
        "400-decommissioned",
        FakeAPIError("The model `old-model` has been decommissioned.", status_code=400),
        ErrorKind.model_unavailable,
    ),
    ("401", FakeAPIError(status_code=401), ErrorKind.auth),
    ("403", anthropic_error(403, "permission_error", "no permission"), ErrorKind.auth),
    ("422", FakeAPIError(status_code=422), ErrorKind.bad_request),
    (
        "400",
        FakeAPIError("Unsupported parameter: tools", status_code=400, code="invalid_request_error"),
        ErrorKind.bad_request,
    ),
    ("status-in-response", FakeAPIError(response=FakeResponse(429)), ErrorKind.rate_limited),
    ("no-status-at-all", RuntimeError("something odd"), ErrorKind.unknown),
]


@pytest.mark.parametrize(("error", "kind"), [pytest.param(e, k, id=name) for name, e, k in CASES])
def test_classify_maps_status_and_codes(error: BaseException, kind: ErrorKind) -> None:
    assert classify_error(error, now=NOW).kind is kind


def test_status_and_code_are_read_from_wherever_the_error_keeps_them() -> None:
    plain = classify_error(FakeAPIError(status_code=429, code="rate_limit_exceeded"), now=NOW)
    assert (plain.status_code, plain.error_code) == (429, "rate_limit_exceeded")

    in_response = FakeAPIError(response=FakeResponse(503, body={"error": {"type": "server_error", "message": "m"}}))
    assert (classify_error(in_response, now=NOW).status_code, classify_error(in_response, now=NOW).error_code) == (
        503,
        "server_error",
    )

    google = classify_error(FakeGoogleError(404, "NOT_FOUND", "gone"), now=NOW)
    assert (google.status_code, google.error_code) == (404, "NOT_FOUND")

    vllm = classify_error(FakeAPIError("m", status_code=400, code=400, type="BadRequestError"), now=NOW)
    assert (vllm.status_code, vllm.error_code) == (400, "BadRequestError")  # the number 400 is not an error code

    anthropic = classify_error(anthropic_error(529, "overloaded_error", "Overloaded"), now=NOW)
    assert (anthropic.status_code, anthropic.error_code) == (529, "overloaded_error")

    nothing = classify_error(RuntimeError("x"), now=NOW)
    assert (nothing.status_code, nothing.error_code) == (None, None)


class APITimeoutError(Exception):
    pass


class ReadTimeout(Exception):
    pass


class APIConnectionError(Exception):
    pass


class ConnectError(Exception):
    pass


class _ConnectionDropped(ConnectionError):
    pass


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        pytest.param(TimeoutError("timed out"), ErrorKind.timeout, id="TimeoutError"),
        pytest.param(APITimeoutError("Request timed out."), ErrorKind.timeout, id="APITimeoutError"),
        pytest.param(ReadTimeout("read timed out"), ErrorKind.timeout, id="ReadTimeout"),
        pytest.param(ConnectionError("refused"), ErrorKind.connection, id="ConnectionError"),
        pytest.param(_ConnectionDropped("reset"), ErrorKind.connection, id="subclass-of-ConnectionError"),
        pytest.param(APIConnectionError("Connection error."), ErrorKind.connection, id="APIConnectionError"),
        pytest.param(ConnectError("all connection attempts failed"), ErrorKind.connection, id="ConnectError"),
    ],
)
def test_classify_timeouts_and_connection_errors_by_type(error: BaseException, kind: ErrorKind) -> None:
    info = classify_error(error, now=NOW)

    assert info.kind is kind
    assert info.status_code is None
    assert info.message  # the text of these kinds describes the network, so it is kept


def test_a_timeout_that_is_also_a_connection_error_is_a_timeout() -> None:
    class APITimeoutErrorOverConnection(APIConnectionError):  # the real APITimeoutError subclasses APIConnectionError
        pass

    assert classify_error(APITimeoutErrorOverConnection("Request timed out."), now=NOW).kind is ErrorKind.timeout


def test_extra_classifiers_run_first() -> None:
    chosen = ErrorInfo(ErrorKind.auth, 401, "my_code", None, "custom")
    calls: list[str] = []

    def declines(error: BaseException) -> ErrorInfo | None:
        calls.append("declines")
        return None

    def claims(error: BaseException) -> ErrorInfo | None:
        calls.append("claims")
        return chosen

    def never_reached(error: BaseException) -> ErrorInfo | None:
        calls.append("never_reached")
        return ErrorInfo(ErrorKind.unknown, None, None, None, "")

    error = FakeAPIError(status_code=429, code="rate_limit_exceeded")
    info = classify_error(error, classifiers=(declines, claims, never_reached), now=NOW)

    assert (info.kind, info.status_code, info.error_code) == (ErrorKind.auth, 401, "my_code")
    assert calls == ["declines", "claims"]
    assert classify_error(error, classifiers=(declines,), now=NOW).kind is ErrorKind.rate_limited


def test_a_classifiers_message_gets_the_same_treatment_as_ours() -> None:
    leaky = ErrorInfo(ErrorKind.bad_request, 400, "invalid_request_error", None, f"bad input: {PROMPT}")
    chatty = ErrorInfo(ErrorKind.rate_limited, 429, None, None, f"slow down {KEY}")

    assert PROMPT not in classify_error(RuntimeError(), classifiers=(lambda e: leaky,), now=NOW).message
    kept = classify_error(RuntimeError(), classifiers=(lambda e: chatty,), secrets=(KEY,), now=NOW)
    assert "slow down" in kept.message
    assert KEY not in kept.message


def test_classify_reads_retry_after_from_the_exception_response() -> None:
    from_response = FakeAPIError(status_code=429, response=FakeResponse(429, {"Retry-After": "7"}))
    from_attribute = FakeAPIError(status_code=429, headers={"retry-after-ms": "1500"})
    from_date = FakeAPIError(
        status_code=429,
        response=FakeResponse(429, {"retry-after": format_datetime(NOW + timedelta(seconds=90), usegmt=True)}),
    )
    from_body = FakeGoogleError(
        429,
        "RESOURCE_EXHAUSTED",
        "slow down",
        {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "30s"}]}},
    )

    assert classify_error(from_response, now=NOW).retry_after_seconds == 7.0
    assert classify_error(from_attribute, now=NOW).retry_after_seconds == 1.5
    assert classify_error(from_date, now=NOW).retry_after_seconds == 90.0
    assert classify_error(from_body, now=NOW).retry_after_seconds == 30.0
    assert classify_error(FakeAPIError(status_code=429), now=NOW).retry_after_seconds is None


# -- AC-07.4 -------------------------------------------------------------------------------------


def test_classified_message_is_scrubbed() -> None:
    error = FakeAPIError(f"429: rate limited, key {KEY} and Bearer abc.def-ghi", status_code=429)

    info = classify_error(error, secrets=(KEY,), now=NOW)

    assert info.kind is ErrorKind.rate_limited
    assert KEY not in info.message
    assert "abc.def-ghi" not in info.message
    assert "rate limited" in info.message


def test_a_long_message_is_cut() -> None:
    info = classify_error(FakeAPIError("x" * 5000, status_code=503), now=NOW)

    assert 0 < len(info.message) <= 300


def test_bad_request_unknown_and_overflow_messages_hold_no_provider_text(build: BuildRig) -> None:
    bad_request = FakeAPIError(
        f"Invalid input: {PROMPT}", status_code=400, code="invalid_request_error", body={"message": PROMPT}
    )
    overflow = FakeAPIError(
        f"prompt is too long: {PROMPT}", status_code=400, code="context_length_exceeded", type="invalid_request_error"
    )
    unknown = RuntimeError(f"surprise: {PROMPT}")

    rig = build()
    resolved = rig.service.resolve(rig.service.select("openai"), user_credential=StoredCredential(Secret(KEY)))
    for error in (bad_request, overflow, unknown):
        handle = rig.telemetry.start_call(resolved)
        handle.fail(error)
    events = [e for e in rig.sink.events if isinstance(e, CallFailed)]

    infos = [classify_error(e, now=NOW) for e in (bad_request, overflow, unknown)]
    assert [i.kind for i in infos] == [ErrorKind.bad_request, ErrorKind.context_overflow, ErrorKind.unknown]
    assert infos[0].message == (
        "bad_request: HTTP 400, code invalid_request_error (provider text withheld; see __cause__)"
    )
    assert infos[1].message == (
        "context_overflow: HTTP 400, code context_length_exceeded (provider text withheld; see __cause__)"
    )
    assert infos[2].message == "unknown (provider text withheld; see __cause__)"
    assert [e.error_message for e in events] == [i.message for i in infos]
    assert all(PROMPT not in i.message for i in infos)
    assert all(PROMPT not in e.error_message for e in events)
    assert all(PROMPT in str(e) for e in (bad_request, overflow, unknown))  # the host still has it on the exception

    kept = classify_error(FakeAPIError(f"Rate limit reached: {PROMPT}", status_code=429), now=NOW)
    assert PROMPT in kept.message  # a 429's text describes server state: kept, scrubbed (D20)


# -- AC-07.5 -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("invalid_request_error", "invalid_request_error"),
        ("model.not-found_2", "model.not-found_2"),
        ("a" * 64, "a" * 64),
        ("a" * 65, None),
        ("has a space", None),
        (f"free text {PROMPT}", None),
        ("semi;colon", None),
        ("new\nline", None),
        ("", None),
    ],
)
def test_free_text_error_codes_are_dropped(code: str, expected: str | None) -> None:
    from_attribute = classify_error(FakeAPIError(status_code=400, code=code), now=NOW)
    from_body = classify_error(FakeAPIError(status_code=400, body={"error": {"code": code}}), now=NOW)

    assert from_attribute.error_code == expected
    assert from_body.error_code == expected
    assert PROMPT not in from_attribute.message


# -- AC-07.6 -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason",
    ["content_filter", "SAFETY", "refusal", "safety", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"],
)
def test_finish_reasons_that_mean_content_blocked(reason: str) -> None:
    assert kind_for_finish_reason(reason) is ErrorKind.content_blocked


@pytest.mark.parametrize("reason", ["stop", "length", "STOP", "MAX_TOKENS", "end_turn", "tool_calls", "", None])
def test_other_finish_reasons_mean_nothing_went_wrong(reason: str | None) -> None:
    assert kind_for_finish_reason(reason) is None


def test_call_failed_carries_the_error_kind_status_and_code(build: BuildRig) -> None:
    rig = build()
    resolved = rig.service.resolve(rig.service.select("openai"), user_credential=StoredCredential(Secret(KEY)))
    error = FakeAPIError(f"slow down {KEY}", status_code=429, code="rate_limit_exceeded")

    with pytest.raises(FakeAPIError), rig.telemetry.track_call(resolved):
        raise error

    (failed,) = [e for e in rig.sink.events if isinstance(e, CallFailed)]
    assert (failed.error_kind, failed.status_code, failed.error_code) == ("rate_limited", 429, "rate_limit_exceeded")
    assert failed.error_type == "FakeAPIError"
    assert "slow down" in failed.error_message
    assert KEY not in failed.error_message


def test_error_kinds_are_the_twelve_in_the_design() -> None:
    assert [k.value for k in ErrorKind] == [
        "rate_limited",
        "quota_exhausted",
        "overloaded",
        "server_error",
        "timeout",
        "connection",
        "context_overflow",
        "content_blocked",
        "model_unavailable",
        "auth",
        "bad_request",
        "unknown",
    ]
    assert ErrorKind.rate_limited == "rate_limited"  # a StrEnum: it compares and logs as its name


# -- the stub's helpers (AC-07.1) ----------------------------------------------------------------


def error_from_stub(vendor: str, reply: Any) -> BaseException:
    """The exception a vendor's SDK would raise for one of the stub's failures."""
    if vendor == "gemini":
        error = reply.body["error"]
        return FakeGoogleError(error["code"], error["status"], error["message"], reply.body)
    body = reply.body["error"] if vendor == "openai" else reply.body  # the OpenAI SDK keeps only the inner dict
    return FakeAPIError(
        str(reply.body), status_code=reply.status, body=body, response=FakeResponse(reply.status, dict(reply.headers))
    )


@pytest.mark.parametrize("vendor", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize(
    ("helper", "kind"),
    [
        ("quota_exhausted", ErrorKind.quota_exhausted),
        ("overloaded", ErrorKind.overloaded),
        ("context_overflow", ErrorKind.context_overflow),
        ("model_not_found", ErrorKind.model_unavailable),
    ],
)
def test_the_stubs_failures_classify_as_their_names_say(vendor: str, helper: str, kind: ErrorKind) -> None:
    from llm_provider.testing import StubReply

    reply = getattr(StubReply, helper)(vendor)

    assert classify_error(error_from_stub(vendor, reply), now=NOW).kind is kind


@pytest.mark.parametrize("vendor", ["openai", "anthropic"])
def test_the_stubs_content_block_and_rate_limit_classify_as_their_names_say(vendor: str) -> None:
    from llm_provider.testing import StubReply

    blocked = classify_error(error_from_stub(vendor, StubReply.content_blocked(vendor)), now=NOW)
    limited = classify_error(error_from_stub("openai", StubReply.rate_limited(retry_after=4.0)), now=NOW)

    assert blocked.kind is ErrorKind.content_blocked
    assert (limited.kind, limited.retry_after_seconds) == (ErrorKind.rate_limited, 4.0)
