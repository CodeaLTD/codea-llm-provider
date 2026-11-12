"""``StubLLMServer``: three wire formats, usage shapes, auth and scripted replies (Task 4 Step 1; AC-05.4)."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

import pytest

from llm_provider.testing import StubLLMServer, StubReply

Response = tuple[int, Mapping[str, str], bytes]

HI = [{"role": "user", "content": "hi"}]


def post(url: str, body: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> Response:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, {k.lower(): v for k, v in response.headers.items()}, response.read()
    except urllib.error.HTTPError as error:
        with error:
            return error.code, {k.lower(): v for k, v in error.headers.items()}, error.read()


def post_json(url: str, body: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> tuple[int, Any]:
    status, _, raw = post(url, body, headers)
    return status, json.loads(raw)


def openai(stub: StubLLMServer, **extra: Any) -> tuple[int, Any]:
    return post_json(f"{stub.url}/chat/completions", {"model": "m", "messages": HI, **extra})


def anthropic(stub: StubLLMServer, **extra: Any) -> tuple[int, Any]:
    body = {"model": "m", "max_tokens": 16, "messages": HI, **extra}
    return post_json(f"{stub.anthropic_url}/v1/messages", body)


def gemini(stub: StubLLMServer, **extra: Any) -> tuple[int, Any]:
    body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], **extra}
    return post_json(f"{stub.gemini_url}/v1beta/models/gemini-x:generateContent", body)


def sse_data(raw: bytes) -> list[str]:
    """The ``data:`` payloads of an SSE body, in order."""
    return [line.removeprefix("data:").strip() for line in raw.decode().splitlines() if line.startswith("data:")]


def stream(stub: StubLLMServer, **extra: Any) -> list[str]:
    status, _, raw = post(f"{stub.url}/chat/completions", {"model": "m", "messages": HI, "stream": True, **extra})
    assert status == 200
    return sse_data(raw)


def test_stub_serves_openai_anthropic_and_gemini_formats_with_their_usage_shapes() -> None:
    with StubLLMServer() as stub:
        assert stub.url.endswith("/v1")
        status, body = openai(stub)
        assert status == 200
        assert isinstance(body["choices"][0]["message"]["content"], str)
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}

        status, body = anthropic(stub)
        assert status == 200
        assert body["type"] == "message"
        assert body["content"][0]["type"] == "text"
        assert body["usage"] == {"input_tokens": 11, "output_tokens": 7}

        status, body = gemini(stub)
        assert status == 200
        assert isinstance(body["candidates"][0]["content"]["parts"][0]["text"], str)
        assert body["usageMetadata"] == {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18}

        assert [r.path for r in stub.requests] == [
            "/v1/chat/completions",
            "/v1/messages",
            "/v1beta/models/gemini-x:generateContent",
        ]
        assert stub.requests[0].body["messages"] == HI
        assert stub.request_count == 3


def test_stub_streams_openai_and_sends_a_usage_chunk_only_when_asked() -> None:
    with StubLLMServer() as stub:
        asked = stream(stub, stream_options={"include_usage": True})
        plain = stream(stub)

    assert asked[-1] == "[DONE]"
    assert plain[-1] == "[DONE]"
    chunks = [json.loads(d) for d in asked[:-1]]
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks[:-1] if c["choices"])
    assert not any("usage" in json.loads(d) for d in plain[:-1])


def test_stub_can_send_no_zero_or_partial_usage() -> None:
    cases: list[tuple[dict[str, int] | None, dict[str, Any], dict[str, Any], dict[str, Any]]] = [
        (None, {}, {}, {}),
        (
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            {"input_tokens": 0, "output_tokens": 0},
            {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0},
        ),
        (
            {"completion_tokens": 7},
            {"completion_tokens": 7},
            {"output_tokens": 7},
            {"candidatesTokenCount": 7},
        ),
    ]
    for usage, expected_openai, expected_anthropic, expected_gemini in cases:
        with StubLLMServer(usage=usage) as stub:
            _, body = openai(stub)
            assert body.get("usage", {}) == expected_openai
            assert ("usage" in body) == (usage is not None)

            _, body = anthropic(stub)
            assert body.get("usage", {}) == expected_anthropic
            assert ("usage" in body) == (usage is not None)

            _, body = gemini(stub)
            assert body.get("usageMetadata", {}) == expected_gemini
            assert ("usageMetadata" in body) == (usage is not None)

            chunks = [json.loads(d) for d in stream(stub, stream_options={"include_usage": True})[:-1]]
            usage_chunks = [c for c in chunks if "usage" in c]
            if usage is None:
                assert usage_chunks == []  # no usage chunk, even when it was asked for
            else:
                assert [c["usage"] for c in usage_chunks] == [expected_openai]


def test_stub_rejects_a_missing_or_wrong_required_header() -> None:
    body = {"model": "m", "messages": HI}
    with StubLLMServer(require_header=("X-Api-Key", "right-key")) as stub:
        url = f"{stub.url}/chat/completions"
        assert post(url, body)[0] == 401
        assert post(url, body, {"X-Api-Key": "wrong-key"})[0] == 401
        assert post(url, body, {"x-api-key": "right-key"})[0] == 200  # the name is not case-sensitive
        assert post(f"{stub.anthropic_url}/v1/messages", body)[0] == 401
        assert post(f"{stub.gemini_url}/v1beta/models/g:generateContent", {"contents": []})[0] == 401


def test_stub_echoes_the_auth_header_and_the_last_user_message_on_failure() -> None:
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "the first question"},
        {"role": "assistant", "content": "an answer"},
        {"role": "user", "content": [{"type": "text", "text": "the last question"}]},
    ]
    auth = {"Authorization": "Bearer abc-token-123"}
    with StubLLMServer(fail_with_status=400) as stub:
        status, _, raw = post(f"{stub.url}/chat/completions", {"model": "m", "messages": messages}, auth)
        assert status == 400
        assert "Bearer abc-token-123" in raw.decode()
        assert "the last question" in raw.decode()
        assert "the first question" not in raw.decode()

        status, _, raw = post(f"{stub.anthropic_url}/v1/messages", {"model": "m", "messages": messages}, auth)
        assert status == 400
        assert "Bearer abc-token-123" in raw.decode() and "the last question" in raw.decode()

        contents = [
            {"role": "user", "parts": [{"text": "the first question"}]},
            {"role": "model", "parts": [{"text": "an answer"}]},
            {"role": "user", "parts": [{"text": "the last question"}]},
        ]
        status, _, raw = post(
            f"{stub.gemini_url}/v1beta/models/g:generateContent", {"contents": contents}, {"x-goog-api-key": "g-key-9"}
        )
        assert status == 400
        assert "g-key-9" in raw.decode() and "the last question" in raw.decode()
        assert "the first question" not in raw.decode()


def test_stub_echoes_the_auth_header_when_it_rejects_a_required_header() -> None:
    with StubLLMServer(require_header=("X-Api-Key", "right-key")) as stub:
        status, _, raw = post(
            f"{stub.url}/chat/completions", {"model": "m", "messages": HI}, {"X-Api-Key": "wrong-key-77"}
        )

    assert status == 401
    assert "wrong-key-77" in raw.decode()


@pytest.mark.parametrize(
    ("make", "status", "retry_after_header", "retry_after_ms_header"),
    [
        (StubReply.rate_limited(), 429, None, None),
        (StubReply.rate_limited(retry_after=2), 429, "2", None),
        (StubReply.rate_limited(retry_after_ms=1500), 429, None, "1500"),
    ],
)
def test_stub_rate_limited_helper_sets_its_headers(
    make: StubReply, status: int, retry_after_header: str | None, retry_after_ms_header: str | None
) -> None:
    with StubLLMServer(script=[make]) as stub:
        got_status, headers, raw = post(f"{stub.url}/chat/completions", {"model": "m", "messages": HI})

    assert got_status == status
    assert headers.get("retry-after") == retry_after_header
    assert headers.get("retry-after-ms") == retry_after_ms_header
    assert json.loads(raw)["error"]["code"] == "rate_limit_exceeded"


def test_stub_script_plays_replies_in_order_with_vendor_error_shapes() -> None:
    script = [
        StubReply.rate_limited(retry_after=2),
        StubReply.overloaded("anthropic"),
        StubReply.quota_exhausted("openai"),
        StubReply.context_overflow("gemini"),
        StubReply.model_not_found("openai"),
        StubReply.content_blocked("openai"),
        StubReply(status=418, headers={"X-Custom": "yes"}, body={"hello": "teapot"}),
        StubReply(),  # a plain reply
    ]
    with StubLLMServer(script=script) as stub:
        seen = [post(f"{stub.url}/chat/completions", {"model": "m", "messages": HI}) for _ in range(len(script) + 1)]
        count = stub.request_count

    statuses = [status for status, _, _ in seen]
    assert statuses == [429, 529, 429, 400, 404, 400, 418, 200, 200]  # the ninth runs past the script: a plain reply
    assert count == 9
    bodies = [json.loads(raw) for _, _, raw in seen]
    assert seen[0][1]["retry-after"] == "2"
    assert bodies[0]["error"]["code"] == "rate_limit_exceeded"
    assert bodies[1] == {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    assert bodies[2]["error"]["code"] == "insufficient_quota"
    assert bodies[3]["error"]["status"] == "INVALID_ARGUMENT" and bodies[3]["error"]["code"] == 400
    assert bodies[4]["error"]["code"] == "model_not_found"
    assert bodies[5]["error"]["code"] == "content_policy_violation"
    assert seen[6][1]["x-custom"] == "yes" and bodies[6] == {"hello": "teapot"}
    assert bodies[7]["usage"]["total_tokens"] == 18


@pytest.mark.parametrize("vendor", ["openai", "anthropic", "gemini"])
def test_stub_helpers_answer_every_vendor_in_its_own_error_shape(vendor: str) -> None:
    helpers = [
        StubReply.quota_exhausted,
        StubReply.overloaded,
        StubReply.context_overflow,
        StubReply.model_not_found,
        StubReply.content_blocked,
    ]
    with StubLLMServer(script=[h(vendor) for h in helpers]) as stub:
        url = {
            "openai": f"{stub.url}/chat/completions",
            "anthropic": f"{stub.anthropic_url}/v1/messages",
            "gemini": f"{stub.gemini_url}/v1beta/models/g:generateContent",
        }[vendor]
        body = {"model": "m", "max_tokens": 8, "messages": HI, "contents": []}
        replies = [post_json(url, body) for _ in helpers]

    for status, payload in replies:
        if vendor == "gemini" and status == 200:  # a blocked prompt: Gemini answers 200 with promptFeedback
            assert payload["promptFeedback"]["blockReason"] == "SAFETY"
            continue
        assert status >= 400
        assert "error" in payload
        if vendor == "anthropic":
            assert payload["type"] == "error" and "type" in payload["error"] and "message" in payload["error"]
        elif vendor == "gemini":
            assert {"code", "message", "status"} <= payload["error"].keys()
        else:
            assert {"message", "type", "code"} <= payload["error"].keys()


def test_stub_stream_then_drop_cuts_the_stream_after_n_chunks_without_a_done_marker() -> None:
    with StubLLMServer(script=[StubReply.stream_then_drop(2)]) as stub:
        request = urllib.request.Request(
            f"{stub.url}/chat/completions",
            data=json.dumps({"model": "m", "messages": HI, "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
            with pytest.raises(http.client.IncompleteRead) as cut:
                response.read()

    partial = cut.value.partial.decode()
    assert len(sse_data(partial.encode())) == 2
    assert "[DONE]" not in partial


def test_stub_replays_scripted_chunks_as_a_stream() -> None:
    chunks = ['data: {"choices": [{"delta": {"content": "a"}}]}\n\n', "data: [DONE]\n\n"]
    with StubLLMServer(script=[StubReply(chunks=chunks)]) as stub:
        status, headers, raw = post(f"{stub.url}/chat/completions", {"model": "m", "messages": HI, "stream": True})

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")
    assert raw.decode() == "".join(chunks)


def test_stub_delay_applies_to_every_request_and_requests_record_headers_and_body() -> None:
    with StubLLMServer(delay_seconds=0.05) as stub:
        post(f"{stub.url}/chat/completions", {"model": "m", "messages": HI}, {"X-Trace": "t1"})
        (recorded,) = stub.requests

    assert recorded.path == "/v1/chat/completions"
    assert recorded.headers["x-trace"] == "t1"
    assert recorded.body == {"model": "m", "messages": HI}


def test_stub_stops_listening_when_the_block_ends() -> None:
    with StubLLMServer() as stub:
        url = f"{stub.url}/chat/completions"
    with pytest.raises(urllib.error.URLError):
        post(url, {"model": "m", "messages": HI})
