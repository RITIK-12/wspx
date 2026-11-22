"""Image and video prompts on a vision-language model, through a real processor.

A tiny Qwen3-VL with a character-level tokenizer and the real Qwen3-VL processor, so images
and video go through the same code paths a full checkpoint uses: chat template, pixel
patches, grids and placeholder tokens.
"""

from __future__ import annotations

import json
import math
import string
from pathlib import Path
from typing import Any

import pytest
import torch

T = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL.Image")

from wspx.errors import WspxError  # noqa: E402
from wspx.lab.sdk import open_lab  # noqa: E402
from wspx.testing import api, runner  # noqa: E402
from wspx.tools.dispatch import call  # noqa: E402

SPECIAL = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
]
VOCAB = {t: i for i, t in enumerate([*SPECIAL, *string.printable[:95], "\n", "<unk>"])}
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{% if m['content'] is string %}"
    "{{ m['content'] }}{% else %}{% for c in m['content'] %}{% if c['type'] == 'image' %}"
    "<|vision_start|><|image_pad|><|vision_end|>{% elif c['type'] == 'video' %}"
    "<|vision_start|><|video_pad|><|vision_end|>{% else %}{{ c['text'] }}{% endif %}{% endfor %}"
    "{% endif %}{% if not (loop.last and continue_final_message) %}<|im_end|>\n{% endif %}"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
PIXELS = {"shortest_edge": 16 * 16, "longest_edge": 32 * 32}


def tiny_vlm() -> tuple[Any, Any]:
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    tk = Tokenizer(models.WordLevel(VOCAB, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tk.decoder = decoders.Fuse()
    tok = T.PreTrainedTokenizerFast(
        tokenizer_object=tk,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<unk>",
        additional_special_tokens=SPECIAL,
    )
    processor = T.Qwen3VLProcessor(
        image_processor=T.Qwen2VLImageProcessor(
            patch_size=4, temporal_patch_size=1, merge_size=1, size=PIXELS
        ),
        tokenizer=tok,
        video_processor=T.Qwen3VLVideoProcessor(
            patch_size=4, temporal_patch_size=1, merge_size=1, size=PIXELS, min_frames=1
        ),
        chat_template=TEMPLATE,
    )
    text = {
        "vocab_size": 128,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "intermediate_size": 96,
        "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
    }
    vision = {
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "out_hidden_size": 64,
        "patch_size": 4,
        "temporal_patch_size": 1,
        "spatial_merge_size": 1,
        "deepstack_visual_indexes": [],
    }
    config = T.Qwen3VLConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=VOCAB["<|image_pad|>"],
        video_token_id=VOCAB["<|video_pad|>"],
        vision_start_token_id=VOCAB["<|vision_start|>"],
        vision_end_token_id=VOCAB["<|vision_end|>"],
    )
    torch.manual_seed(0)
    model = T.AutoModelForImageTextToText.from_config(config).eval()
    model.generation_config.eos_token_id = VOCAB["<|im_end|>"]
    return model, processor


def picture(path: Path, seed: int) -> str:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL.fromarray((rng.random((24, 24, 3)) * 255).astype("uint8")).save(path)
    return str(path)


def ask(image: str, question: str = "done?") -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "Answer yes or no."},
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
            },
        ]
    }


@pytest.fixture
def lab(tmp_path: Path) -> Any:
    model, processor = tiny_vlm()
    for i in range(4):
        picture(tmp_path / "prompts" / "frames" / f"{i}.png", i)
    rows = [ask(f"prompts/frames/{i}.png") for i in range(4)]
    (tmp_path / "prompts" / "scenes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return open_lab(
        model,
        store=tmp_path / "store",
        workdir=tmp_path,
        tokenizer=processor.tokenizer,
        processor=processor,
    )


def test_every_reading_tool_takes_image_prompts(lab: Any) -> None:
    info = call(lab, "arch").summary
    assert "images or video" in info
    assert "scenes (4)" in info
    first, second = {"item": "scenes/0"}, {"item": "scenes/1"}
    out = call(lab, "generate", {"prompt": first, "max_tokens": 3})
    assert out.ok, out.summary
    batch = call(lab, "generate", {"prompt": [first, ask("prompts/frames/2.png")], "max_tokens": 2})
    assert batch.ok
    assert "[1]" in batch.summary
    trace = call(lab, "trace", {"prompt": first})
    assert trace.ok, trace.summary
    assert "logit lens" in trace.summary
    attribution = call(lab, "trace", {"prompt": first, "reference": second, "target": "y"})
    assert attribution.ok, attribution.summary
    assert "attribution" in attribution.summary
    diff = call(
        lab,
        "contrast",
        {"positive": [first, second], "negative": "prompts/scenes.jsonl", "name": "d"},
    )
    assert diff.ok, diff.summary
    fires = call(lab, "activations", {"prompts": [first, {"item": "scenes/3"}], "top": 3})
    assert fires.ok, fires.summary
    assert "fires-on" in fires.summary


def test_the_image_reaches_the_model(lab: Any) -> None:
    """Two different pictures behind the same text give different logits."""
    a = lab.text.batch([{"item": "scenes/0"}])
    b = lab.text.batch([{"item": "scenes/1"}])
    assert a.media
    assert "pixel_values" in a.extra
    assert "image_grid_thw" in a.extra
    assert torch.equal(a.ids, b.ids)
    from wspx.lab import analysis

    with torch.no_grad():
        la = analysis.forward(lab.model, lab.ir, a.ids, a.mask, a.extra)
        lb = analysis.forward(lab.model, lab.ir, b.ids, b.mask, b.extra)
    assert not torch.allclose(la[0, -1], lb[0, -1])


def test_video_from_frames_and_from_a_file(lab: Any, tmp_path: Path) -> None:
    frames = [picture(tmp_path / "prompts" / "clip" / f"{i}.png", 10 + i) for i in range(4)]
    rel = [str(Path(f).relative_to(tmp_path)) for f in frames]
    as_frames = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": rel, "fps": 2},
                    {"type": "text", "text": "what moved?"},
                ],
            }
        ]
    }
    out = call(lab, "generate", {"prompt": as_frames, "max_tokens": 2})
    assert out.ok, out.summary
    assert "pixel_values_videos" in lab.text.batch([as_frames]).extra

    av = pytest.importorskip("av")
    path = tmp_path / "prompts" / "clip.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 24, 24, "yuv420p"
        for i in range(8):
            image = (np.full((24, 24, 3), i * 30)).astype("uint8")
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    from_file = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "prompts/clip.mp4"},
                    {"type": "text", "text": "?"},
                ],
            }
        ]
    }
    out = call(lab, "trace", {"prompt": from_file})
    assert out.ok, out.summary


def test_tools_cannot_read_test_data_or_leave_the_project(lab: Any, tmp_path: Path) -> None:
    (tmp_path / "wtests" / "data").mkdir(parents=True)
    picture(tmp_path / "wtests" / "data" / "x.png", 99)
    (tmp_path / "wtests" / "data" / "answers.jsonl").write_text('{"prompt": "a", "label": true}\n')
    for prompt, words in [
        (ask("wtests/data/x.png"), "test data"),
        (ask("../outside.png"), "inside the project"),
        (ask("https://example.com/a.png"), "no network"),
        ({"item": "scenes/9"}, "has 4 prompts"),
        ({"item": "nope"}, "<set>/<n>"),
    ]:
        out = call(lab, "generate", {"prompt": prompt, "max_tokens": 1})
        assert not out.ok
        assert words in out.summary, out.summary
    out = call(
        lab,
        "contrast",
        {
            "positive": "wtests/data/answers.jsonl",
            "negative": "wtests/data/answers.jsonl",
            "name": "x",
        },
    )
    assert not out.ok
    assert "test data" in out.summary
    out = call(lab, "contrast", {"positive": "/etc/x.jsonl", "negative": "../y.jsonl", "name": "x"})
    assert not out.ok
    assert "inside the project" in out.summary


def test_text_only_models_refuse_image_prompts(tmp_path: Path) -> None:
    model, processor = tiny_vlm()
    lab = open_lab(model, store=tmp_path / "s", workdir=tmp_path, tokenizer=processor.tokenizer)
    out = call(lab, "generate", {"prompt": {"item": "scenes/0"}, "max_tokens": 1})
    assert not out.ok
    assert "processor" in out.summary


def test_suites_score_image_prompts_and_blind_suites_show_aggregates_only(
    lab: Any, tmp_path: Path
) -> None:
    from wspx.lab.evaluate import SuiteModel

    model = SuiteModel(lab.model, lab.ir, lab.text.unrestricted(), lab.at_base)
    prompts = [{"item": f"scenes/{i}"} for i in range(4)]
    outs = model.generate_many(prompts, max_tokens=4, batch_size=3, watch=["y", "n"])
    assert len(outs) == 4
    for g in outs:
        assert len(g.watched) == len(g.ids) == len(g.ends)
        if g.ids:
            assert all(v <= 0 for v in g.watched[0])
            assert g.step_at(0) == 0
            assert g.logp(0, "y") == g.watched[0][0]
    alone = model.generate(prompts[0], max_tokens=4)
    assert alone == outs[0].text
    lp = model.logprob(prompts[0], "ye")
    assert math.isfinite(lp)
    assert lp < 0
    # suites may read their own data folders (tools may not)
    picture(tmp_path / "wtests" / "data" / "x.png", 5)
    assert model.logit_diff(ask("wtests/data/x.png"), "y", "n") == pytest.approx(
        model.logit_diff(ask("wtests/data/x.png"), "y", "n")
    )

    labels = [True, False, True, False]

    def verify(m: Any) -> Any:
        for i, g in enumerate(m.generate_many(prompts, max_tokens=2, watch=["y", "n"])):
            said = g.text.startswith("y")
            margin = g.watched[0][0] - g.watched[0][1] if g.ids else math.nan
            yield api.expect.margin(
                said == labels[i],
                margin if labels[i] else -margin,
                group="yes" if labels[i] else "no",
            )

    suite = api.Suite("grouped", verify, blind=True, source="v1")
    result = runner.run_suite(suite, model, "t")
    text = runner.summary(result, vs=result)
    assert "blind" in text
    assert "by group: yes" in text
    assert "balanced" in text
    assert "✗" not in text
    assert "item 0" not in text
    view = runner.driver_view(result)
    assert "checks" not in view
    assert "failures" not in view
    assert set(view["groups"]) == {"yes", "no"}
    with pytest.raises(WspxError):
        runner.heldout_items(result, caller="driver")
    assert runner.heldout_items(result, caller="ui") == result.checks


def test_prompt_lists_that_arrive_as_strings_are_understood(lab: Any) -> None:
    """Models sometimes JSON-encode an array into the string branch of the schema."""
    texts = json.dumps(["The lamp is on.", "The door is open."])
    items = json.dumps([{"item": "scenes/0"}, {"item": "scenes/1"}])
    for positive, negative in [
        (texts, json.dumps(["The lamp is off.", "The door is shut."])),
        (items, json.dumps([{"item": "scenes/2"}, {"item": "scenes/3"}])),
        ([json.dumps({"item": "scenes/0"}), {"item": "scenes/1"}], [{"item": "scenes/2"}]),
        ("The lamp is on.", "The lamp is off."),
        ("prompts/scenes.jsonl", [{"item": "scenes/2"}]),
    ]:
        out = call(lab, "contrast", {"positive": positive, "negative": negative, "name": "d"})
        assert out.ok, out.summary
    fires = call(lab, "activations", {"prompts": items, "top": 2})
    assert fires.ok, fires.summary
    gen = call(lab, "generate", {"prompt": json.dumps({"item": "scenes/0"}), "max_tokens": 2})
    assert gen.ok, gen.summary
    long = call(lab, "contrast", {"positive": "x" * 5000 + ".jsonl", "negative": "y", "name": "d"})
    assert not long.ok
    assert "failed" not in long.summary  # a clean error, not a crash


def test_a_stored_prompt_takes_a_reply(lab: Any) -> None:
    started = {"item": "scenes/0", "reply": "Let me look.\n\nAnswer: "}
    plain = lab.text.batch([{"item": "scenes/0"}])
    replied = lab.text.batch([started])
    assert replied.ids.shape[1] > plain.ids.shape[1]
    assert lab.text.decode(replied.ids[0].tolist()).endswith("Answer: ")
    out = call(lab, "trace", {"prompt": started})
    assert out.ok, out.summary


def test_info_counts_the_vision_tower_on_image_prompts(lab: Any) -> None:
    """Seen live: info told the driver the whole vision tower was unexercised, with no role."""
    assert not lab.ir.unexercised
    vision = lab.ir.vision_only
    assert vision
    assert all(name.startswith("model.visual.") for name in vision)
    info = call(lab, "arch", {}).summary
    assert "UNEXERCISED" not in info
    assert "phases prefill, decode, vision" in info
    assert f"model.visual ({len(vision)} tensors; read on image prompts only" in info
    hits = call(
        lab, "grep", {"expression": "rownorm", "in": "model.visual.blocks.*.mlp.linear_fc2.weight"}
    )
    assert hits.ok, hits.summary
    # Seen live on Inkling-2-Mini: the image pass never reached the per-tensor facts, so every
    # vision matrix was kind `param` with no in/out axes and grep guessed its axis.
    fc2 = lab.ir.primary_role("model.visual.blocks.0.mlp.linear_fc2.weight")
    assert fc2 is not None
    assert (fc2.in_dim, fc2.out_dim) == (1, 0)
    assert fc2.kind != "param"


def test_check_sees_a_vision_tower_edit(lab: Any) -> None:
    """Seen live on Inkling-2-Mini: check's canary was text only, so a vision-tower edit
    showed KL 0.000 and check could not tell it from no edit."""
    assert lab._smoke().image_kl == 0.0  # at base nothing moves
    lab.edit.scale("model.visual.blocks.0.mlp.linear_fc2.weight", factor=0.0)
    report = lab.check()
    assert report.stats["canary_kl"] == pytest.approx(0.0, abs=1e-6)  # text never sees it
    assert report.stats["canary_image_kl"] > 1e-6
    assert "image canary KL" in report.summary()


def test_an_ir_from_an_older_capture_is_rebuilt(tmp_path: Path) -> None:
    import json

    from wspx.lab.ir.model_ir import IR_VERSION
    from wspx.lab.sdk import _cached_ir, _ir_path
    from wspx.store import Store

    model, processor = tiny_vlm()
    lab = open_lab(model, store=tmp_path / "s", workdir=tmp_path, processor=processor)
    store = Store(tmp_path / "s")
    fp = lab.ir.fingerprint
    assert _cached_ir(store, fp) is not None
    old = json.loads(store.read_file(_ir_path(fp)) or b"{}")
    assert old["version"] == IR_VERSION
    store.write_file(_ir_path(fp), json.dumps({**old, "version": 1}).encode())
    assert _cached_ir(store, fp) is None
