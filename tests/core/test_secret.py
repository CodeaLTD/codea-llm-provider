"""Secret hides its value everywhere except ``reveal()`` (Task 2 Step 1; GC Secrets)."""

from __future__ import annotations

from llm_provider import Secret

KEY = "sk-abc12345"


def test_secret_never_shows_its_value() -> None:
    s = Secret(KEY)

    shown = [
        repr(s),
        str(s),
        f"{s}",
        f"{s!r}",
        f"{s:>40}",
        "%s" % s,  # noqa: UP031
        repr({"k": s}),
        repr([s]),
        repr((s,)),
    ]

    for text in shown:
        assert KEY not in text
        assert "abc12345" not in text
    assert s.reveal() == KEY
