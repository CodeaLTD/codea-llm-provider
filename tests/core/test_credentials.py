"""MappingServerCredentials (Task 2 Step 1)."""

from __future__ import annotations

from llm_provider import MappingServerCredentials, Secret


def test_mapping_server_credentials_returns_secret_and_treats_empty_as_unset() -> None:
    credentials = MappingServerCredentials(
        {"FROM_STR": "sk-one", "FROM_SECRET": Secret("sk-two"), "EMPTY": "", "NONE": None}
    )

    from_str = credentials.get("FROM_STR")
    from_secret = credentials.get("FROM_SECRET")
    assert isinstance(from_str, Secret) and from_str.reveal() == "sk-one"
    assert isinstance(from_secret, Secret) and from_secret.reveal() == "sk-two"
    assert credentials.get("EMPTY") is None
    assert credentials.get("NONE") is None
    assert credentials.get("MISSING") is None


def test_an_empty_secret_counts_as_unset_and_the_keys_stay_out_of_the_repr() -> None:
    credentials = MappingServerCredentials({"A": Secret(""), "B": "sk-hidden-value"})

    assert credentials.get("A") is None
    assert "sk-hidden-value" not in repr(credentials)
