"""Images reach every driver provider: each is shown a red square through a tool result and
asked its colour. Real API calls through your deployment's gateway (pennies each).

``WSPX_RELEASE_DRIVERS``: comma-separated driver names, e.g.
``anthropic:claude-haiku-5-5,openai:gpt-5-mini,google:gemini-3-flash``; ``WSPX_GATEWAY_URL``
and ``WSPX_LLM_TOKEN``: a session's gateway URL and model-call token.
"""

from __future__ import annotations

import io
import os

import pytest

pytestmark = pytest.mark.release

DRIVERS = [d for d in os.environ.get("WSPX_RELEASE_DRIVERS", "").split(",") if d]


@pytest.mark.parametrize("driver", DRIVERS or ["(none configured)"])
def test_a_tool_image_reaches_the_driver(driver: str) -> None:
    if not DRIVERS:
        pytest.skip("set WSPX_RELEASE_DRIVERS")
    from PIL import Image

    from wspx.driver.media import Attachment, Media
    from wspx.driver.messages import Message, ToolCall, ToolOutput
    from wspx.driver.providers import make_provider

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 20, 20)).save(buf, format="PNG")
    png = buf.getvalue()
    provider = make_provider(
        driver, gateway=os.environ["WSPX_GATEWAY_URL"], token=os.environ["WSPX_LLM_TOKEN"]
    )
    kind = driver.split(":", maxsplit=1)[0]
    provider.media = Media(
        {"sha256:red": png}.__getitem__,
        provider=kind if kind in ("anthropic", "openai", "google") else "compat",
    )
    history = [
        Message(
            "user",
            text="Call run on plot.py, then tell me the colour of the square it drew, in one word.",
        ),
        Message(
            "assistant",
            tool_calls=[ToolCall("c1", "run", {"script": "plot.py"})],
            provider="replayed",
        ),
        Message(
            "tool",
            outputs=[
                ToolOutput(
                    "c1",
                    "run",
                    "drew a square",
                    attachments=[Attachment("sha256:red", "image/png", "square.png")],
                )
            ],
        ),
    ]
    turn = provider.step("Answer briefly.", history, [], effort="low", max_tokens=2000)
    assert "red" in turn.text.lower(), turn.text
