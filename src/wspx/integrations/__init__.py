"""Files shipped to external agents: the wspx skill, the Claude Code plugin and Codex templates."""

from __future__ import annotations

from importlib.resources import files


def skill_text() -> str:
    """``SKILL.md`` without its front matter: the built-in driver's system prompt."""
    text = files("wspx.integrations").joinpath("SKILL.md").read_text(encoding="utf-8")
    if text.startswith("---"):
        text = text.split("---", 2)[2]
    return text.strip()


def read(name: str) -> str:
    return files("wspx.integrations").joinpath(name).read_text(encoding="utf-8")
