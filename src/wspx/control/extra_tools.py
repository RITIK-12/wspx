"""Tools beyond wspx's own for a session's driver: the project's, and remote MCP servers'.

A project tool (``[[driver.tools]]``) is a Python function in the project, run in the lab
like a ``run`` script. A remote MCP server (``[[driver.mcp]]``) is called by the control
plane, which holds its token; only the tools its ``allow`` list names are offered, under
``<server>__<tool>``. Both are logged, budgeted and loop-checked like wspx's own tools.
"""

from __future__ import annotations

import base64
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from wspx.errors import WspxError
from wspx.tools import TOOLS

SEPARATOR = "__"
_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")


@dataclass
class Extra:
    name: str
    """As the model sees it."""
    description: str
    schema: dict[str, Any]
    kind: str
    """``project`` or ``mcp``."""
    call: str = ""
    """Project tools: ``path.py:function``."""
    server: str = ""
    remote: str = ""
    """MCP tools: the tool's name on its server."""

    def offered(self) -> dict[str, Any]:
        """What a runner is given: name, description and argument schema."""
        return {"name": self.name, "description": self.description, "schema": self.schema}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def project_tools(specs: list[dict[str, Any]]) -> list[Extra]:
    out = []
    for s in specs:
        name = str(s.get("name", ""))
        if not _NAME.match(name) or SEPARATOR in name:
            raise WspxError(f"project tool {name!r}: letters, digits, - and _ (no '__')")
        if name in TOOLS:
            raise WspxError(f"project tool {name!r} has the name of a wspx tool; rename it")
        out.append(
            Extra(
                name,
                str(s.get("description") or f"The project's {name} tool."),
                dict(s.get("parameters") or {"type": "object"}),
                "project",
                call=str(s["call"]),
            )
        )
    return out


Connect = Callable[[dict[str, Any]], Any]
"""An MCP server spec → what :class:`mcp.client.Client` connects to."""


def token_for(server: str) -> str | None:
    """A remote MCP server's token: ``WSPX_MCP_TOKEN_<NAME>`` here, else the deployment's
    store of them (``wspx keys set mcp:<name>``), which labs and runners cannot reach."""
    env = os.environ.get(f"WSPX_MCP_TOKEN_{server.upper().replace('-', '_')}")
    if env:
        return env
    try:  # pragma: no cover - needs a Modal deployment
        import modal

        from wspx.cloud import ENVIRONMENT, MCP_TOKENS

        value = modal.Dict.from_name(MCP_TOKENS, environment_name=ENVIRONMENT).get(server)
        return str(value) if value else None
    except Exception:
        return None


def http_connect(spec: dict[str, Any]) -> Any:
    """Streamable HTTP to the server's URL, with its token when it takes one."""
    import httpx2
    from mcp.client.streamable_http import streamable_http_client

    headers = {}
    if spec.get("token"):
        token = token_for(spec["name"])
        if not token:
            raise WspxError(
                f"MCP server {spec['name']} takes a token and none is stored; "
                f"`wspx keys set mcp:{spec['name']}`"
            )
        headers["authorization"] = f"Bearer {token}"
    client = httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(120.0, read=600.0))
    return streamable_http_client(spec["url"], http_client=client)


@dataclass
class Remote:
    """One session's remote MCP servers, as its ``[[driver.mcp]]`` lists them."""

    servers: list[dict[str, Any]]
    connect: Connect = http_connect
    tools: dict[str, Extra] = field(default_factory=dict)

    def discover(self) -> list[Extra]:
        """Each server's allowed tools (a server that cannot be reached offers none, and says
        why in the returned problems)."""
        import anyio

        found: list[Extra] = []
        for spec in self.servers:
            allow = set(spec.get("allow") or [])

            async def listing(spec: dict[str, Any] = spec) -> Any:
                from mcp.client import Client

                async with Client(self.connect(spec)) as client:
                    return await client.list_tools()

            listed = anyio.run(listing)
            for tool in listed.tools:
                if tool.name not in allow:
                    continue
                name = f"{spec['name']}{SEPARATOR}{tool.name}"
                found.append(
                    Extra(
                        name[:64],
                        f"[{spec['name']}] {tool.description or tool.name}",
                        dict(tool.input_schema or {"type": "object"}),
                        "mcp",
                        server=spec["name"],
                        remote=tool.name,
                    )
                )
        self.tools = {t.name: t for t in found}
        return found

    def call(self, tool: Extra, args: dict[str, Any]) -> tuple[str, list[tuple[bytes, str]], bool]:
        """Text, images (bytes and media type) and whether the server reported an error."""
        import anyio
        from mcp import types

        spec = next(s for s in self.servers if s["name"] == tool.server)

        async def calling() -> Any:
            from mcp.client import Client

            async with Client(self.connect(spec)) as client:
                return await client.call_tool(tool.remote, args)

        result = anyio.run(calling)
        texts: list[str] = []
        images: list[tuple[bytes, str]] = []
        for part in result.content:
            if isinstance(part, types.TextContent):
                texts.append(part.text)
            elif isinstance(part, types.ImageContent):
                images.append((base64.b64decode(part.data), part.mime_type))
            elif isinstance(part, types.EmbeddedResource):
                res = part.resource
                text = getattr(res, "text", None)
                texts.append(text if text is not None else f"[resource {res.uri}]")
            else:
                texts.append(f"[{getattr(part, 'type', 'content')} not shown]")
        if result.structured_content and not texts:
            import json

            texts.append(json.dumps(result.structured_content)[:20_000])
        return "\n".join(texts), images, bool(result.is_error)
