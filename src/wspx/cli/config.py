"""Where the CLI finds the deployed control plane: ``~/.config/wspx/config.toml``.

Environment variables ``WSPX_URL`` and ``WSPX_TOKEN`` override the file.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from wspx.errors import ConfigError


def path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "wspx" / "config.toml"


@dataclass
class CliConfig:
    url: str = ""
    token: str = ""
    session: str = ""
    """The default session for tool commands."""


def load() -> CliConfig:
    cfg = CliConfig()
    p = path()
    if p.exists():
        data = tomllib.loads(p.read_text())
        cloud = data.get("cloud", {})
        cfg = CliConfig(cloud.get("url", ""), cloud.get("token", ""), data.get("session", ""))
    cfg.url = os.environ.get("WSPX_URL", cfg.url)
    cfg.token = os.environ.get("WSPX_TOKEN", cfg.token)
    cfg.session = os.environ.get("WSPX_SESSION", cfg.session)
    return cfg


def save(cfg: CliConfig) -> Path:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)

    def q(text: str) -> str:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = [
        f"session = {q(cfg.session)}",
        "",
        "[cloud]",
        f"url = {q(cfg.url)}",
        f"token = {q(cfg.token)}",
        "",
    ]
    p.write_text("\n".join(lines))
    p.chmod(0o600)
    return p


def require(cfg: CliConfig) -> CliConfig:
    if not cfg.url or not cfg.token:
        raise ConfigError(
            "not connected to a wspx deployment; run `wspx deploy` (or set WSPX_URL and WSPX_TOKEN)"
        )
    return cfg
