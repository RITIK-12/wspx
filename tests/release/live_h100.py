"""Edits reaching a running vLLM on one H100, without a reload: Qwen3-0.6B (both engines
on the GPU) and Inkling-2-Mini (NVFP4 and MXFP8 experts; transformers' copy streamed from
host memory). Each push is timed, both engines are compared after it, and going back to
the base must give the base's answers exactly.

    modal run --env wspx tests/release/live_h100.py
"""

from __future__ import annotations

import json

import modal
from live_phases import INKLING_EDITS, QWEN_EDITS

from wspx.cloud.app import models_volume, vllm_lab_image

app = modal.App("wspx-live-h100")
image = vllm_lab_image.add_local_file(
    "tests/release/live_phases.py", "/root/live_phases.py", copy=True
)
CASES = [
    ("/models/Qwen--Qwen3-0.6B", QWEN_EDITS),
    ("/models/thinkingmachines--Inkling-2-Mini", INKLING_EDITS),
]


@app.function(
    image=image,
    gpu="H100",
    cpu=8.0,
    memory=160 * 1024,
    timeout=3 * 3600,
    volumes={"/models": models_volume.read_only()},
)
def live() -> list[dict]:  # type: ignore[type-arg]
    import subprocess
    import sys
    from pathlib import Path

    out = []
    for i, (model, edits) in enumerate(CASES):
        path = f"/tmp/live-{i}.json"
        child = (
            "import json, sys; sys.path.insert(0, '/root'); import live_phases as m\n"
            f"m.run({model!r}, json.loads({json.dumps(edits)!r}), {path!r})\n"
        )
        done = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
        try:
            out.append(json.loads(Path(path).read_text()))
        except (OSError, ValueError):
            out.append({"model": model, "error": done.stderr[-6000:]})
    return out


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(live.remote(), indent=2, default=str))
