"""Signed tokens shared by the control plane and the gateway.

The control plane mints a short-lived *session token* for each driver runner.
The runner's SDK clients present it to the gateway in place of an API key;
the gateway verifies it with the shared secret and swaps in the real key.
Neither the runner nor the lab ever sees a provider key.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass

from wspx.errors import WspxError

PREFIX = "wspx_s_"


class TokenError(WspxError):
    """A token is malformed, forged or expired."""


@dataclass(frozen=True)
class SessionClaims:
    session: str
    expires: float
    scope: str = "llm"


def new_secret() -> str:
    return secrets.token_urlsafe(32)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def mint(secret: str, session: str, *, ttl: float = 24 * 3600, scope: str = "llm") -> str:
    body = _b64(
        json.dumps(
            {"sid": session, "exp": time.time() + ttl, "scope": scope}, sort_keys=True
        ).encode()
    )
    sig = _b64(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())
    return f"{PREFIX}{body}.{sig}"


def verify(secret: str, token: str, *, scope: str = "llm") -> SessionClaims:
    if not token.startswith(PREFIX) or "." not in token:
        raise TokenError("not a wspx session token")
    body, _, sig = token[len(PREFIX) :].partition(".")
    expected = _b64(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        raise TokenError("bad token signature")
    try:
        data = json.loads(_unb64(body))
    except (ValueError, json.JSONDecodeError):
        raise TokenError("malformed token") from None
    if data.get("scope") != scope:
        raise TokenError(f"token scope {data.get('scope')!r} is not {scope!r}")
    if data["exp"] < time.time():
        raise TokenError("token expired")
    return SessionClaims(data["sid"], float(data["exp"]), data["scope"])


def check_bearer(expected: str, presented: str | None) -> bool:
    """Constant-time comparison for static admin/API tokens."""
    return bool(presented) and hmac.compare_digest(expected.encode(), (presented or "").encode())
