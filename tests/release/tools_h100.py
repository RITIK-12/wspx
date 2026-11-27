"""Every tool on Inkling-2-Mini served by vLLM, on one H100, as a deployed lab runs it: the
lab server starts as the sandbox starts it (the vLLM lab image, vLLM holding the only copy
of the model) and each tool is called over its HTTP API, background jobs included.

    modal run --detach --env wspx tests/release/tools_h100.py
"""

from __future__ import annotations

import json

import modal

from wspx.cloud.app import models_volume, vllm_lab_image

app = modal.App("wspx-tools-h100")
image = vllm_lab_image.add_local_file(
    "tests/release/tools_phases.py", "/root/tools_phases.py", copy=True
)


@app.function(
    image=image,
    gpu="H100",
    cpu=8.0,
    memory=160 * 1024,
    timeout=3 * 3600,
    volumes={"/models": models_volume.read_only()},
)
def tools() -> dict:  # type: ignore[type-arg]
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")
    import tools_phases

    out = "/tmp/tools.json"
    tools_phases.run(out)
    report: dict = json.loads(Path(out).read_text())  # type: ignore[type-arg]
    # in the app's logs too: a detached run (--detach) outlives a dropped connection
    print("WSPX-REPORT " + json.dumps(report, default=str), flush=True)
    return report


@app.local_entrypoint()
def main() -> None:
    report = tools.remote()
    print(json.dumps(report, indent=2, default=str))
    print(
        f"{len(report['calls']) - len(report['failed'])}/{len(report['calls'])} calls ok; "
        f"failed: {report['failed'] or 'none'}; checks failed: {report['checks_failed'] or 'none'}"
    )
