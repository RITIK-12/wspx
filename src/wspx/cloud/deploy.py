"""``wspx deploy``: put the app into the user's Modal account."""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
from dataclasses import dataclass

from wspx.auth import new_secret
from wspx.cloud import APP_NAME, CONTROL_SECRET, ENVIRONMENT, KEY_SECRETS


@dataclass
class Deployment:
    control_url: str
    gateway_url: str
    api_token: str


def _modal(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "modal", *args], capture_output=True, text=True, check=check
    )


def ensure_environment() -> None:
    proc = _modal("environment", "list", "--json", check=False)
    if ENVIRONMENT not in proc.stdout:
        _modal("environment", "create", ENVIRONMENT, check=False)


def ensure_secrets(api_token: str | None = None) -> str:
    """Create the control secret once (API token, shared secret, admin token) and an empty keys secret."""
    proc = _modal("secret", "list", "--env", ENVIRONMENT, "--json", check=False)
    api_token = api_token or "wspx_" + secrets.token_urlsafe(24)
    if CONTROL_SECRET not in proc.stdout:
        _modal(
            "secret",
            "create",
            CONTROL_SECRET,
            f"WSPX_API_TOKEN={api_token}",
            f"WSPX_GATEWAY_SECRET={new_secret()}",
            f"WSPX_ADMIN_TOKEN={new_secret()}",
            "--env",
            ENVIRONMENT,
        )
    else:
        api_token = ""  # already created; the CLI keeps its own copy
    for name, _ in KEY_SECRETS.values():
        if name not in proc.stdout:  # every key secret must exist for the app to deploy
            _modal("secret", "create", name, "WSPX_UNSET=1", "--env", ENVIRONMENT)
    return api_token


def set_key(provider: str, value: str) -> None:
    """Store a provider key in the keys secret (never on local disk)."""
    if provider not in KEY_SECRETS:
        raise ValueError(f"unknown provider {provider!r}; use one of {', '.join(KEY_SECRETS)}")
    name, var = KEY_SECRETS[provider]
    _modal("secret", "create", name, f"{var}={value}", "--env", ENVIRONMENT, "--force")


def deploy() -> Deployment:  # pragma: no cover - needs a Modal account
    ensure_environment()
    token = ensure_secrets()
    os.environ.setdefault("MODAL_ENVIRONMENT", ENVIRONMENT)
    from wspx.cloud.app import app, control, gateway

    with __import__("modal").enable_output():
        app.deploy(name=APP_NAME, environment_name=ENVIRONMENT)
    return Deployment(control.get_web_url() or "", gateway.get_web_url() or "", token)
