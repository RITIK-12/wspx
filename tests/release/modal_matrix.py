"""Run the release matrix on Modal: ``modal run tests/release/modal_matrix.py``.

It uses wspx's vLLM lab image on one H100 (Inkling's vLLM kernels need sm90) and the models
volume, and writes ``release-matrix.jsonl`` to the store volume's ``release/`` folder. It
spends GPU time in your Modal account: about an hour of H100 for the full matrix.
"""

from __future__ import annotations

import modal

from wspx.cloud.app import hf_secret, models_volume, store_volume, vllm_lab_image

app = modal.App("wspx-release-matrix")
image = vllm_lab_image.add_local_dir("tests", "/root/tests", copy=True).pip_install("pytest")


@app.function(
    image=image,
    gpu="H100",
    timeout=4 * 3600,
    volumes={"/root/.cache/huggingface": models_volume, "/store": store_volume},
    secrets=[hf_secret],
)
def matrix(selection: str = "") -> str:
    import os
    import subprocess

    env = {
        **os.environ,
        "WSPX_RELEASE": "1",
        "WSPX_RELEASE_REPORT": "/store/release/release-matrix.jsonl",
    }
    os.makedirs("/store/release", exist_ok=True)
    args = [
        "python",
        "-m",
        "pytest",
        "/root/tests/release/test_matrix.py",
        "-x",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    if selection:
        args += ["-k", selection]
    done = subprocess.run(args, env=env, cwd="/root", capture_output=True, text=True, check=False)
    store_volume.commit()
    return done.stdout[-6000:] + done.stderr[-3000:]


@app.local_entrypoint()
def main(selection: str = "") -> None:
    print(matrix.remote(selection))
