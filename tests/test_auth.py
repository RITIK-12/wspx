import pytest

from wspx.auth import TokenError, check_bearer, mint, verify


def test_roundtrip() -> None:
    token = mint("k", "s_1")
    assert verify("k", token).session == "s_1"


@pytest.mark.parametrize(
    "tamper",
    [
        lambda t: t[:-2] + ("AA" if not t.endswith("AA") else "BB"),
        lambda t: t.replace("wspx_s_", "wspx_x_"),
        lambda t: "garbage",
    ],
)
def test_forgery_rejected(tamper) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(TokenError):
        verify("k", tamper(mint("k", "s_1")))


def test_wrong_secret_expiry_and_scope() -> None:
    with pytest.raises(TokenError, match="signature"):
        verify("other", mint("k", "s_1"))
    with pytest.raises(TokenError, match="expired"):
        verify("k", mint("k", "s_1", ttl=-1))
    with pytest.raises(TokenError, match="scope"):
        verify("k", mint("k", "s_1", scope="mcp"))


def test_check_bearer() -> None:
    assert check_bearer("abc", "abc")
    assert not check_bearer("abc", "abd")
    assert not check_bearer("abc", None)
