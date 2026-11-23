"""The open model's trace: what the open-weight model was given and what it said back."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from tests.lab.test_session import tiny_llama
from wspx.lab import server as lab_server
from wspx.lab.model_io import clip, collapse, split_output
from wspx.lab.sdk import open_lab


class TestParsing:
    def test_think_blocks_are_reasoning(self) -> None:
        out = split_output("<think>The capital is Paris.</think>\n\nParis<|im_end|>")
        assert out == {"reasoning": "The capital is Paris.", "tool_calls": [], "answer": "Paris"}

    def test_a_think_the_template_opened_in_the_prompt(self) -> None:
        out = split_output("weighing it up</think>Paris")
        assert (out["reasoning"], out["answer"]) == ("weighing it up", "Paris")

    def test_analysis_and_final_channels(self) -> None:
        raw = (
            "<|channel|>analysis<|message|>check facts<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Paris<|return|>"
        )
        out = split_output(raw)
        assert (out["reasoning"], out["answer"]) == ("check facts", "Paris")

    def test_tool_calls(self) -> None:
        raw = (
            'Let me look.<tool_call>\n{"name": "search", "arguments": {"q": "x"}}\n</tool_call>'
            "<tool_call>oops</tool_call>"
        )
        out = split_output(raw)
        assert out["tool_calls"] == [{"name": "search", "arguments": {"q": "x"}}, {"raw": "oops"}]
        assert out["answer"] == "Let me look."

    def test_only_end_of_turn_markers_are_trimmed(self) -> None:
        assert split_output("<div>hi</div><end_of_turn>")["answer"] == "<div>hi</div>"
        assert split_output("Paris</s><pad><pad>")["answer"] == "Paris"

    def test_plain_text_is_the_answer(self) -> None:
        assert split_output("just text") == {
            "reasoning": None,
            "tool_calls": [],
            "answer": "just text",
        }

    def test_image_tokens_collapse_and_long_text_is_clipped(self) -> None:
        assert collapse("a" + "<|image_pad|>" * 64 + "b") == "a<|image_pad|>×64b"
        assert collapse("<x><x>") == "<x><x>"  # short runs stay as they are
        long = "x" * 20_000
        clipped = clip(long, 1000)
        assert len(clipped) < 1100
        assert "characters omitted" in clipped


SUITES = textwrap.dedent("""
    from wspx.testing import suite, expect

    @suite(split="dev", seed=0)
    def visible(model):
        yield expect.true(len(model.generate([1, 2, 3], max_tokens=3)) >= 0, "runs")

    @suite(split="heldout", seed=1)
    def secret(model):
        yield expect.true(len(model.generate([4, 5, 6], max_tokens=3)) >= 0, "runs")
""")


@pytest.fixture
def client(tmp_path: Path) -> Any:
    (tmp_path / "wtests").mkdir()
    (tmp_path / "wtests" / "suites.py").write_text(SUITES)
    lab = open_lab(tiny_llama(), store=tmp_path / "s", workdir=tmp_path)
    with TestClient(lab_server.LabServer(lab, exit_on_lapse=False).app()) as c:
        yield c


def tool(client: Any, name: str, args: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = client.post(f"/tool/{name}", json={"args": args, "call_id": "c"}).json()
    return out


def test_each_tool_call_returns_what_the_open_model_saw_and_wrote(client: Any) -> None:
    out = tool(client, "generate", {"prompt": [[1, 2, 3], [7, 8]], "max_tokens": 4})
    records = out["model_io"]["records"]
    assert [r["kind"] for r in records] == ["generate", "generate"]
    first = records[0]
    assert first["prompt"] == "token ids [1, 2, 3]"  # no tokenizer: the ids themselves
    assert (first["purpose"], first["weights"], first["prompt_tokens"]) == ("generate", "base", 3)
    assert first["output_tokens"] == 4
    assert first["stop"] == "max_tokens"
    assert first["settings"]["max_tokens"] == 4
    assert first["batch"] == 2  # generated together, so they share the batch's time
    assert first["seconds"] > 0
    assert "model_io" not in str(out["summary"])
    nothing = tool(client, "arch", {})
    assert nothing["model_io"] == {"records": [], "omitted": 0}


def test_records_name_the_weights_and_the_purpose(client: Any) -> None:
    tool(client, "edit", {"op": "scale", "args": {"targets": "L0.mlp.down[:, 0]", "factor": 0.5}})
    edited = tool(client, "generate", {"prompt": [1, 2, 3], "max_tokens": 2})["model_io"]
    assert edited["records"][0]["weights"].startswith("edited tree ")
    assert edited["records"][0]["weights"].endswith("(1 staged)")
    checked = tool(client, "check", {})["model_io"]["records"]
    assert {r["purpose"] for r in checked} == {"check canary"}
    assert {r["kind"] for r in checked} == {"forward"}  # no tokenizer: no canary generation


def test_held_out_items_are_recorded_and_marked(client: Any) -> None:
    out = tool(client, "test", {"suite": "secret"})
    (record,) = out["model_io"]["records"]
    assert (record["purpose"], record["split"]) == ("suite secret", "heldout")
    assert record["prompt"] == "token ids [4, 5, 6]"
    visible = tool(client, "test", {"suite": "visible"})["model_io"]["records"]
    assert "split" not in visible[0]
    cached = tool(client, "test", {"suite": "visible"})["model_io"]["records"]
    assert cached == []  # a cached run does not prompt the model


def test_a_chat_prompt_is_recorded_as_the_model_saw_it(tmp_path: Path) -> None:
    from tests.lab.test_media import ask, picture, tiny_vlm

    model, processor = tiny_vlm()
    lab = open_lab(
        model,
        store=tmp_path / "s",
        workdir=tmp_path,
        tokenizer=processor.tokenizer,
        processor=processor,
    )
    picture(tmp_path / "a.png", 0)
    with TestClient(lab_server.LabServer(lab, exit_on_lapse=False).app()) as c:
        out = tool(c, "generate", {"prompt": ask("a.png", "is it red?"), "max_tokens": 3})
    (record,) = out["model_io"]["records"]
    assert record["messages"][0] == {"role": "system", "content": "Answer yes or no."}
    assert record["messages"][1]["content"][0] == {"type": "image", "image": "a.png"}
    assert "Answer yes or no." in record["prompt"]  # the instruction, through the template
    assert "is it red?" in record["prompt"]
    assert "×" in record["prompt"]  # the image's tokens, collapsed
    assert record["kind"] == "generate"
    assert record["output_tokens"] >= 1


def test_stored_prompts_and_frame_videos_keep_their_media_paths(tmp_path: Path) -> None:
    """The UI shows a prompt's images from the paths it named: a stored prompt is recorded as
    the messages it names, and a video given as frames keeps its frame paths (never pixels)."""
    import json

    from tests.lab.test_media import ask, picture, tiny_vlm

    model, processor = tiny_vlm()
    lab = open_lab(
        model,
        store=tmp_path / "s",
        workdir=tmp_path,
        tokenizer=processor.tokenizer,
        processor=processor,
    )
    picture(tmp_path / "prompts" / "a.png", 0)
    (tmp_path / "prompts" / "scenes.jsonl").write_text(
        json.dumps(ask("prompts/a.png", "red?")) + "\n"
    )
    frames = [f"prompts/clip/{i}.png" for i in range(20)]
    for i, f in enumerate(frames):
        picture(tmp_path / f, i)
    video = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "video", "video": frames}, {"type": "text", "text": "moved?"}],
            }
        ]
    }
    with TestClient(lab_server.LabServer(lab, exit_on_lapse=False).app()) as c:
        stored = tool(c, "generate", {"prompt": {"item": "scenes/0"}, "max_tokens": 2})
        clip = tool(c, "generate", {"prompt": video, "max_tokens": 2})
    (record,) = stored["model_io"]["records"]
    assert record["item"] == "scenes/0"
    assert record["messages"][1]["content"][0] == {"type": "image", "image": "prompts/a.png"}
    (record,) = clip["model_io"]["records"]
    part = record["messages"][0]["content"][0]
    assert (part["type"], part["frames"], part["video"]) == ("video", 20, frames[:16])
    assert len(json.dumps(record)) < 20_000  # paths, never pixels
