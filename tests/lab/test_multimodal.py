"""Vision-language checkpoints (the 8-10B multimodal reasoning models): load, capture, every tool.

The language side is what the tools work on, driven by text prompts; the vision tower
loads with the checkpoint and is addressable by its parameter names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

T = pytest.importorskip("transformers")

from wspx.lab.sdk import open_lab  # noqa: E402
from wspx.tools.dispatch import call  # noqa: E402

TEXT = {
    "vocab_size": 128,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 96,
}
VISION = {
    "depth": 1,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 2,
    "out_hidden_size": 64,
    "patch_size": 4,
    "temporal_patch_size": 1,
    "spatial_merge_size": 1,
}
MROPE = {"rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]}}


def config(kind: str) -> Any:
    if kind == "qwen3_vl":
        return T.Qwen3VLConfig(
            text_config={**TEXT, **MROPE}, vision_config={**VISION, "deepstack_visual_indexes": []}
        )
    if kind == "glm4v":
        return T.Glm4vConfig(text_config={**TEXT, **MROPE}, vision_config=VISION)
    if kind == "qwen3_5":  # hybrid: a linear-attention (SSM) block and a full-attention one
        layers = {"layer_types": ["linear_attention", "full_attention"]}
        return T.Qwen3_5Config(text_config={**TEXT, **layers}, vision_config=VISION)
    vision = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "image_size": 16,
        "patch_size": 4,
    }
    return T.Gemma3Config(
        text_config={**TEXT, "sliding_window": 8}, vision_config=vision, mm_tokens_per_image=4
    )


@pytest.mark.parametrize("kind", ["qwen3_vl", "glm4v", "qwen3_5", "gemma3"])
def test_every_tool_works_on_a_vision_language_model(kind: str, tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = T.AutoModelForImageTextToText.from_config(config(kind)).eval()
    lab = open_lab(model, store=tmp_path / "s", workdir=tmp_path)
    assert "2 blocks on a residual stream" in call(lab, "info").summary
    steps: list[tuple[str, dict[str, Any]]] = [
        ("ls", {"limit": 3}),
        ("grep", {"expression": "colnorm > 1σ", "top": 2}),  # the default scope, readers included
        ("generate", {"prompt": [1, 2, 3], "max_tokens": 4}),
        ("trace", {"prompt": [1, 2, 3, 4]}),
        ("contrast", {"positive": [[1, 2, 3]], "negative": [[4, 5, 6]], "name": "d"}),
        ("edit", {"op": "scale", "args": {"targets": "L0.mlp.down", "factor": 0.9}}),
        ("check", {}),
        ("vcs", {"action": "commit", "message": "scale L0 down"}),
    ]
    for name, args in steps:
        result = call(lab, name, args)
        assert result.ok, f"{name}: {result.summary}"


def test_a_vision_language_checkpoint_loads_with_its_vision_weights(tmp_path: Path) -> None:
    from wspx.lab.loader import load_hf

    torch.manual_seed(0)
    T.AutoModelForImageTextToText.from_config(config("qwen3_vl")).save_pretrained(tmp_path / "m")
    loaded = load_hf(str(tmp_path / "m"), dtype="float32", local_only=True)
    names = [n for n, _ in loaded.model.named_parameters()]
    assert type(loaded.model).__name__ == "Qwen3VLForConditionalGeneration"
    assert any(".visual." in n for n in names)  # the vision tower came with it
