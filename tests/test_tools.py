import json

import pytest

from wspx.tools import TOOLS, anthropic_tools, gemini_tools, mcp_tools, openai_tools
from wspx.tools.schemas import GrepArgs


def test_tools_come_in_groups_map_first() -> None:
    groups = [t.description.split("]", 1)[0].lstrip("[") for t in TOOLS.values()]
    assert list(TOOLS) == [
        "arch",
        "show",
        "grep",
        "generate",
        "trace",
        "activations",
        "contrast",
        "edit",
        "check",
        "test",
        "vcs",
        "export",
        "run",
    ]
    assert groups == ["map"] + ["inspect"] * 2 + ["probe"] * 4 + ["edit"] * 2 + ["keep"] * 4


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


def test_no_text_names_a_tool_that_was_removed() -> None:
    """Seen live: after `info` and `ls` became `arch`, the workspace notes, the restart
    message and the image-prompt schema still told the driver to call `info`."""
    import re
    from pathlib import Path

    import wspx

    root = Path(wspx.__file__).parent
    pattern = re.compile(r"`{1,2}(info|ls)\b(?!\.)")
    hits = [
        f"{path.relative_to(root)}:{n}"
        for path in [*root.rglob("*.py"), *root.rglob("*.md")]
        if "control/ui" not in path.as_posix()
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert hits == []
    assert "info" not in json.dumps(mcp_tools()).replace("information", "")
