"""Images and files from tools (and with the task) reach every provider in its own form."""

from __future__ import annotations

import base64
import io
from typing import Any

import pytest

from wspx.driver.media import KEEP_MESSAGES, Attachment, Media, fit_image
from wspx.driver.messages import Message, ToolCall, ToolOutput, dump_message, load_message

PIL = pytest.importorskip("PIL.Image")


def png(w: int = 4, h: int = 3) -> bytes:
    buf = io.BytesIO()
    PIL.new("RGB", (w, h), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


BLOBS = {"sha256:img": png(), "sha256:pdf": b"%PDF-1.4 tiny", "sha256:csv": b"a,b\n1,2\n"}
IMG = Attachment("sha256:img", "image/png", "attn.png", width=4, height=3)
PDF = Attachment("sha256:pdf", "application/pdf", "paper.pdf")
CSV = Attachment("sha256:csv", "text/csv", "scores.csv")


def history(*extra: Message) -> list[Message]:
    return [
        Message("user", text="study the map", attachments=[PDF]),
        Message("assistant", tool_calls=[ToolCall("c1", "run", {"script": "plot.py"})]),
        Message("tool", outputs=[ToolOutput("c1", "run", "ran plot.py", attachments=[IMG, CSV])]),
        *extra,
    ]


def media(provider: str, **kw: Any) -> Media:
    return Media(BLOBS.__getitem__, provider=provider, **kw)


def test_anthropic_puts_images_and_documents_in_the_tool_result() -> None:
    from wspx.driver.providers.anthropic import AnthropicProvider

    p = AnthropicProvider("claude-opus-5-5", client=object())
    p.media = media("anthropic")
    msgs = p.messages(history())
    task = msgs[0]["content"]
    assert task[-1]["type"] == "document"
    assert task[-1]["source"]["media_type"] == "application/pdf"
    (result,) = msgs[2]["content"]
    kinds = [b["type"] for b in result["content"]]
    assert kinds == ["text", "image", "text"]
    assert base64.b64decode(result["content"][1]["source"]["data"]) == BLOBS["sha256:img"]
    assert result["content"][2]["text"].startswith("[file scores.csv]\na,b")


def test_openai_puts_them_in_the_function_call_output() -> None:
    from wspx.driver.providers.openai import OpenAIProvider

    p = OpenAIProvider("gpt-5", client=object())
    p.media = media("openai")
    items = p.items(history())
    out = next(i for i in items if i.get("type") == "function_call_output")
    assert [x["type"] for x in out["output"]] == ["input_text", "input_image", "input_text"]
    assert out["output"][1]["image_url"].startswith("data:image/png;base64,")
    task = items[0]
    assert task["content"][1] == {
        "type": "input_file",
        "filename": "paper.pdf",
        "file_data": "data:application/pdf;base64,"
        + base64.b64encode(BLOBS["sha256:pdf"]).decode(),
    }


def test_gemini_carries_the_bytes_in_the_function_response() -> None:
    pytest.importorskip("google.genai")
    from wspx.driver.providers.google import GoogleProvider

    p = GoogleProvider("gemini-3-pro", client=object())
    p.media = media("google")
    contents = p.contents(history())
    response = contents[2].parts[0].function_response
    assert response.parts[0].inline_data.mime_type == "image/png"
    assert response.parts[0].inline_data.data == BLOBS["sha256:img"]
    assert "[file scores.csv]" in response.response["result"]


def test_openai_compatible_servers_get_a_user_message_after_the_tool_message() -> None:
    from wspx.driver.providers.compat import ChatProvider

    p = ChatProvider("qwen", name="openai-compatible:local#qwen", client=object())
    p.media = media("compat")
    msgs = p.messages("system", history())
    tool = next(m for m in msgs if m["role"] == "tool")
    assert tool["content"] == "ran plot.py"
    follow = msgs[msgs.index(tool) + 1]
    assert follow["role"] == "user"
    assert [c["type"] for c in follow["content"]] == ["text", "image_url", "text"]


def test_old_attachments_become_placeholders_and_no_vision_degrades() -> None:
    later = [
        Message(
            "tool",
            outputs=[
                ToolOutput(
                    f"c{i}", "run", "x", attachments=[Attachment(f"sha256:img{i}", "image/png")]
                )
            ],
        )
        for i in range(KEEP_MESSAGES)
    ]
    m = Media(lambda oid: png(), provider="anthropic")
    visible = m.visible(history(*later))
    assert "sha256:img" not in visible  # pushed out by newer messages
    assert m.part(IMG, visible="sha256:img" in visible).text.startswith("[attachment attn.png")
    blind = Media(BLOBS.__getitem__, provider="anthropic", vision=False)
    assert "does not take images" in blind.part(IMG, True).text


def test_large_images_are_fitted_to_the_provider() -> None:
    _kind, data = fit_image(png(4000, 1000), "image/png", 1568, 5 * 2**20)
    with PIL.open(io.BytesIO(data)) as im:
        assert max(im.size) == 1568


def test_the_transcript_keeps_ids_not_bytes() -> None:
    msg = history()[2]
    dumped = dump_message(msg)
    saved = dumped["outputs"][0]["attachments"][0]
    assert saved["oid"] == "sha256:img"
    assert set(saved) <= {"oid", "media_type", "name", "size", "width", "height", "caption"}
    assert load_message(dumped).outputs[0].attachments[0] == IMG
