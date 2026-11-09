"""``scrub`` removes secrets from text and caps its length (Task 3 Step 1; RF4)."""

from __future__ import annotations

from llm_provider.telemetry.scrub import scrub

KEY = "live-key-9f8e7d6c5b4a"
BEARER = "Bearer eyJhbGciOi.abc-DEF_123"
SK = "sk-proj-abcdef123456"


def test_scrub_removes_live_key_bearer_and_sk_tokens_and_caps_length() -> None:
    text = f"401 for {KEY}; header Authorization: {BEARER}; token {SK}; " + "x" * 600

    out = scrub(text, [KEY])

    assert KEY not in out
    assert "eyJhbGciOi" not in out
    assert "abcdef123456" not in out
    assert len(out) <= 300


def test_scrub_removes_api_key_values_in_both_spellings() -> None:
    out = scrub('{"api_key": "abc123xyz"} x-api-key: def456uvw api_key=ghi789rst')

    for leaked in ("abc123xyz", "def456uvw", "ghi789rst"):
        assert leaked not in out
    assert "api_key" in out
    assert "x-api-key" in out


def test_scrub_removes_every_occurrence_of_each_secret_and_ignores_empty_ones() -> None:
    out = scrub(f"a {KEY} b {KEY} c other-secret", [KEY, "", "other-secret"])

    assert KEY not in out
    assert "other-secret" not in out
    assert out.startswith("a ") and out.endswith(" c ***")


def test_scrub_leaves_ordinary_text_and_short_texts_alone() -> None:
    assert scrub("the task is done; disk-usage ok") == "the task is done; disk-usage ok"
    assert scrub("x" * 300) == "x" * 300


def test_scrub_truncates_to_the_given_limit() -> None:
    out = scrub("y" * 50, limit=10)

    assert len(out) == 10
    assert out.endswith("…")


def test_scrub_removes_a_secret_before_cutting_so_no_piece_survives() -> None:
    out = scrub("z" * 295 + KEY, [KEY])

    assert KEY[:6] not in out
    assert len(out) <= 300
