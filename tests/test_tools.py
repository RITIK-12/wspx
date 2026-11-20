import json

import pytest

from wspx.tools import TOOLS, anthropic_tools, gemini_tools, mcp_tools, openai_tools
from wspx.tools.schemas import GrepArgs


def test_every_tool_from_the_design_is_present() -> None:
    assert set(TOOLS) == {
        "info",
        "ls",
        "grep",
        "show",
        "contrast",
        "generate",
        "trace",
        "edit",
        "check",
        "test",
        "jobs",
        "vcs",
        "export",
        "run",
    }


@pytest.mark.parametrize("export", [mcp_tools, anthropic_tools, openai_tools, gemini_tools])
def test_schemas_are_self_contained(export) -> None:  # type: ignore[no-untyped-def]
    tools = export()
    assert len(tools) == len(TOOLS)
    text = json.dumps(tools)
    assert "$ref" not in text
    assert "$defs" not in text


def test_grep_alias_and_limits() -> None:
    args = GrepArgs.model_validate({"expression": "norm", "in": "L*.mlp.down", "top": 5})
    assert args.scope == "L*.mlp.down"
    with pytest.raises(ValueError):  # noqa: PT011
        GrepArgs.model_validate({"expression": "norm", "top": 500})
    with pytest.raises(ValueError):  # noqa: PT011
        GrepArgs.model_validate({"expression": "norm", "unknown": 1})


def test_provider_shapes() -> None:
    assert anthropic_tools()[0].keys() == {"name", "description", "input_schema"}
    assert openai_tools()[0]["type"] == "function"
    assert "parameters_json_schema" in gemini_tools()[0]
    assert mcp_tools()[0]["inputSchema"]["type"] == "object"
