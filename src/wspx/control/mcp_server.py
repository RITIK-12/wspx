"""The hosted MCP endpoint (streamable HTTP) and the local stdio proxy (§5.10).

Both expose the same tools as the built-in driver. A client binds to a session
with the ``X-Wspx-Session`` header (or ``?session=``); every call is logged
in the trace DB under that session with a ``call_id`` the client's hooks can
join on.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import anyio
from mcp import types
from mcp.server.lowlevel import Server

from wspx import __version__
from wspx.integrations import skill_text
from wspx.tools import mcp_tools

CallFn = Callable[[str | None, str, dict[str, Any]], dict[str, Any]]
"""(session or None, tool, arguments) → tool result dict."""
FetchFn = Callable[[str, str], tuple[bytes, str]]
"""(session, attachment object id) → bytes and media type."""


def _tool_list() -> list[types.Tool]:
    return [
        types.Tool(name=t["name"], description=t["description"], input_schema=t["inputSchema"])
        for t in mcp_tools()
    ]


def _content(result: dict[str, Any], text: str, fetch: FetchFn | None) -> list[Any]:
    """Text, then each attachment: images as MCP image content, other files as resources."""
    import base64

    out: list[Any] = [types.TextContent(type="text", text=text)]
    session = result.get("session")
    for a in result.get("attachments") or []:
        label = a.get("name") or a.get("oid", "")[:16]
        if fetch is None or not session:
            out.append(types.TextContent(type="text", text=f"[attachment {label}]"))
            continue
        try:
            data, kind = fetch(session, a["oid"])
        except Exception as exc:
            out.append(types.TextContent(type="text", text=f"[attachment {label}: {exc}]"))
            continue
        encoded = base64.b64encode(data).decode()
        if kind.startswith("image/"):
            out.append(types.ImageContent(type="image", data=encoded, mime_type=kind))
        else:
            resource = types.BlobResourceContents(
                uri=f"wspx://{session}/attachments/{a['oid']}", mime_type=kind, blob=encoded
            )
            out.append(types.EmbeddedResource(type="resource", resource=resource))
    return out


def build_server(call: CallFn, *, fetch: FetchFn | None = None) -> Server[Any]:
    async def list_tools(_ctx: Any, _params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=_tool_list())

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        request = getattr(ctx, "request", None)
        session = None
        if request is not None:
            session = request.headers.get("x-wspx-session") or request.query_params.get("session")
        result = await anyio.to_thread.run_sync(
            call, session, params.name, dict(params.arguments or {})
        )
        text = result.get("summary", "")
        if result.get("call_id"):
            text += f"\n[call {result['call_id']}]"
        return types.CallToolResult(
            content=_content(result, text, fetch), is_error=not result.get("ok", True)
        )

    return Server(
        "wspx",
        version=__version__,
        instructions=skill_text(),
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def run_stdio(call: CallFn, *, fetch: FetchFn | None = None) -> None:
    """``wspx mcp``: a local stdio server forwarding to the hosted control plane."""
    from mcp.server.stdio import stdio_server

    server = build_server(call, fetch=fetch)

    async def main() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(main)
