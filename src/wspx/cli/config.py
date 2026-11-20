"""The CLI's own state: ``~/.config/wspx/config.toml``.

It holds where the deployed control plane is and which session each project
(and the shell outside any project) is working on. ``WSPX_URL``, ``WSPX_TOKEN``
and ``WSPX_SESSION`` override it.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
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
    """The current session outside any project."""
    projects: dict[str, str] = field(default_factory=dict)
    """Project root → that project's current session."""

    def current(self, root: str = "") -> str:
        """The session tool commands use in project ``root`` ("" outside any project)."""
        return os.environ.get("WSPX_SESSION") or (
            self.projects.get(root, "") if root else self.session
        )

    def set_current(self, root: str, session: str) -> None:
        if root:
            self.projects[root] = session
        else:
            self.session = session


def load() -> CliConfig:
    cfg = CliConfig()
    p = path()
    if p.exists():
        data = tomllib.loads(p.read_text())
        cloud = data.get("cloud", {})
        cfg = CliConfig(
            cloud.get("url", ""),
            cloud.get("token", ""),
            data.get("session", ""),
            {str(k): str(v) for k, v in data.get("projects", {}).items()},
        )
    cfg.url = os.environ.get("WSPX_URL", cfg.url)
    cfg.token = os.environ.get("WSPX_TOKEN", cfg.token)
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
    if cfg.projects:
        lines += ["[projects]", *(f"{q(k)} = {q(v)}" for k, v in sorted(cfg.projects.items())), ""]
    p.write_text("\n".join(lines))
    p.chmod(0o600)
    return p


def require(cfg: CliConfig) -> CliConfig:
    if not cfg.url or not cfg.token:
        raise ConfigError(
            "not connected to a wspx deployment; run `wspx deploy` (or set WSPX_URL and WSPX_TOKEN)"
        )
    return cfg
