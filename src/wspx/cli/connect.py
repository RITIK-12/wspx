"""``wspx connect claude|codex``: point a local agent at the hosted MCP endpoint (§5.10)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from wspx.errors import WspxError
from wspx.integrations import read


def claude(url: str, token: str, session: str, *, project: Path) -> list[str]:
    """Register the MCP server, add HTTP hooks for the trace, and install the skill."""
    done = []
    if shutil.which("claude") is None:
        raise WspxError("the `claude` CLI is not on PATH; install Claude Code first")
    cmd = [
        "claude",
        "mcp",
        "add",
        "--transport",
        "http",
        "wspx",
        f"{url.rstrip('/')}/mcp/",
        "--header",
        f"Authorization: Bearer {token}",
        "--header",
        f"X-Wspx-Session: {session}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 and "already exists" not in proc.stderr:
        raise WspxError(f"claude mcp add failed: {proc.stderr.strip()}")
    done.append("registered the wspx MCP server")
    settings = project / ".claude" / "settings.json"
    data = json.loads(settings.read_text()) if settings.exists() else {}
    hook: dict[str, Any] = {
        "type": "http",
        "url": f"{url.rstrip('/')}/api/ingest/claude-code?session={session}",
        "headers": {"Authorization": f"Bearer {token}"},
    }
    hooks = data.setdefault("hooks", {})
    for event in ("PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop"):
        entries = hooks.setdefault(event, [])
        if not any(
            h.get("hooks", [{}])[0].get("url", "").startswith(hook["url"].split("?")[0])
            for h in entries
        ):
            entries.append({"matcher": "*", "hooks": [hook]})
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps(data, indent=2) + "\n")
    done.append(f"added trace hooks to {settings}")
    skill = project / ".claude" / "skills" / "wspx" / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(read("SKILL.md"))
    done.append(f"installed the skill at {skill}")
    return done


def codex(url: str, token: str, session: str, *, project: Path) -> list[str]:
    """Add the local stdio MCP proxy and command hooks to ``.codex/config.toml``; install the skill."""
    done = []
    config = project / ".codex" / "config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    text = config.read_text() if config.exists() else ""
    if "[mcp_servers.wspx]" not in text:
        text += (
            "\n[mcp_servers.wspx]\n"
            'command = "wspx"\n'
            'args = ["mcp"]\n'
            f'env = {{ WSPX_URL = "{url}", WSPX_TOKEN = "{token}", WSPX_SESSION = "{session}" }}\n'
        )
    if "wspx hook codex" not in text:
        text += read("codex-hooks.toml")
    config.write_text(text)
    done.append(f"configured {config}")
    skill = project / ".codex" / "skills" / "wspx" / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(read("SKILL.md"))
    agents = project / "AGENTS.md"
    snippet = read("AGENTS.snippet.md")
    if not agents.exists() or "wspx" not in agents.read_text():
        agents.write_text((agents.read_text() + "\n" if agents.exists() else "") + snippet)
        done.append("added the wspx section to AGENTS.md")
    done.append(f"installed the skill at {skill}")
    return done
