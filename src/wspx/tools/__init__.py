"""The tool surface (§4.2): one schema per tool, shared by every driver and the CLI.

Schemas are Pydantic models; MCP and provider tool definitions are generated
from them, so the built-in driver, external agents over MCP and humans at the
CLI all see the same tools with the same arguments.
"""

from wspx.tools.schemas import (
    TOOLS,
    ToolSpec,
    anthropic_tools,
    gemini_tools,
    mcp_tools,
    openai_tools,
)

__all__ = ["TOOLS", "ToolSpec", "anthropic_tools", "gemini_tools", "mcp_tools", "openai_tools"]
