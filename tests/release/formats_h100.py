"""Every quantized format vLLM serves, edited live on one H100: each checkpoint is served
by vLLM (the only copy), the lab edits a weight in the checkpoint's own format and pushes
it, both engines are compared before and after, and a discard must give the base's answers
back exactly.

    modal run --env wspx tests/release/formats_h100.py
"""

from __future__ import annotations

import json

import modal

from wspx.cloud.app import models_volume, vllm_lab_image

app = modal.App("wspx-formats-h100")
image = vllm_lab_image.add_local_file(
    "tests/release/live_phases.py", "/root/live_phases.py", copy=True
)


def _scale(label: str, target: str, factor: float = 1.1) -> dict:  # type: ignore[type-arg]
    return {"label": label, "op": "scale", "args": {"targets": target, "factor": factor}}


CASES = [
    ("Qwen--Qwen3-0.6B-FP8", [_scale("fp8 tiles: L14 down_proj", "L14.mlp.down_proj")]),
    ("RedHatAI--Qwen3-0.6B-FP8-dynamic", [_scale("fp8 rows: L14 down_proj", "L14.mlp.down_proj")]),
    (
        "RedHatAI--Qwen2.5-0.5B-Instruct-quantized.w8a8",
        [_scale("int8 rows: L12 down_proj", "L12.mlp.down_proj")],
    ),
    (
        "RedHatAI--Qwen3-0.6B-quantized.w4a16",
        [_scale("int4 + zero points: L14 down_proj", "L14.mlp.down_proj")],
    ),
    (
        "Qwen--Qwen2.5-0.5B-Instruct-GPTQ-Int4",
        [_scale("gptq int4 (act-order): L12 down_proj", "L12.mlp.down_proj")],
    ),
    ("Qwen--Qwen2.5-0.5B-Instruct-AWQ", [_scale("awq int4: L12 down_proj", "L12.mlp.down_proj")]),
    ("nvidia--Qwen3-8B-FP4", [_scale("modelopt nvfp4: L18 down_proj", "L18.mlp.down_proj")]),
    (
        "openai--gpt-oss-20b",
        [_scale("mxfp4 experts 0-3 of L12", "L12.mlp.experts.down_proj[0:4]")],
    ),
]


@app.function(
    image=image,
    gpu="H100",
    cpu=8.0,
    memory=160 * 1024,
    timeout=4 * 3600,
    volumes={"/models": models_volume.read_only()},
)
def formats(only: str = "") -> list[dict]:  # type: ignore[type-arg]
    import subprocess
    import sys
    from pathlib import Path

    out = []
    for i, (name, edits) in enumerate(CASES):
        if only and not any(o and o in name for o in only.split(",")):
            continue
        model, path = f"/models/{name}", f"/tmp/formats-{i}.json"
        if not Path(model, "config.json").exists():
            out.append({"model": name, "error": "not in the models volume"})
            continue
        child = (
            "import json, sys; sys.path.insert(0, '/root'); import live_phases as m\n"
            f"m.run({model!r}, json.loads({json.dumps(edits)!r}), {path!r})\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", child], capture_output=True, text=True, check=False
        )
        try:
            out.append(json.loads(Path(path).read_text()))
        except (OSError, ValueError):
            out.append({"model": name, "error": done.stderr[-6000:]})
    return out


@app.local_entrypoint()
def main(only: str = "") -> None:
    print(json.dumps(formats.remote(only), indent=2, default=str))
