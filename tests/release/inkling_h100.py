"""Inkling-2-Mini on one H100: edits to its 4-bit (NVFP4, layers 0-19) and 8-bit (MXFP8,
layers 20-23) experts in the lab, timed text and image prompts, then the edited export
served by vLLM next to the base checkpoint.

    modal run --env wspx tests/release/inkling_h100.py

One H100 cannot hold transformers' bf16 copy (~53 GB) and vLLM's (~20 GB plus its KV cache)
at once, so the lab runs first (in a process of its own), exports and exits, and vLLM serves
the export after. The GPU is released when the function returns. Expect ~30 minutes.
"""

from __future__ import annotations

import json

import modal
from inkling_phases import MODEL, _image, _vllm

from wspx.cloud.app import models_volume, vllm_lab_image

app = modal.App("wspx-inkling-h100")
image = vllm_lab_image.add_local_file(
    "tests/release/inkling_phases.py", "/root/inkling_phases.py", copy=True
)


@app.function(
    image=image,
    gpu="H100",
    cpu=8.0,
    memory=160 * 1024,
    timeout=3 * 3600,
    volumes={"/models": models_volume.read_only()},
)
def inkling() -> dict:  # type: ignore[type-arg]
    import subprocess
    import sys
    from pathlib import Path

    Path("/tmp/work/prompts").mkdir(parents=True, exist_ok=True)
    _image("/tmp/work/prompts/circle.png")  # both phases ask about it
    # the lab runs in a process of its own, so all of its GPU memory is gone before vLLM
    # starts (seen: clearing the lab in-process left ~56 GB held, and vLLM ran out)
    child = (
        "import json, sys, traceback; sys.path.insert(0, '/root'); import inkling_phases as m\n"
        "r = {}\n"
        "try:\n    m._lab_phase(r)\n"
        "except Exception:\n    r['lab_error'] = traceback.format_exc()[-4000:]\n"
        "json.dump(r, open('/tmp/lab.json', 'w'), default=str)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, check=False
    )
    try:
        report: dict = json.loads(Path("/tmp/lab.json").read_text())  # type: ignore[type-arg]
    except (OSError, ValueError):
        report = {"lab_error": done.stderr[-4000:]}
    if "export" in report:
        report["vllm_edited"] = _vllm("/tmp/export")
    report["vllm_base"] = _vllm(MODEL)
    return report


@app.local_entrypoint()
def main() -> None:
    report = inkling.remote()
    print(json.dumps(report, indent=2, default=str))
