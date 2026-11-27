"""Inkling-2-Mini on one H100: edits to its 4-bit (NVFP4, layers 0-19) and 8-bit (MXFP8,
layers 20-23) experts in the lab, timed text and image prompts, then the edited export
served by vLLM next to the base checkpoint.

    modal run --env wspx tests/release/inkling_h100.py

One H100 cannot hold transformers' bf16 copy (~53 GB) and vLLM's (~20 GB plus its KV cache)
at once, so the lab runs first, exports, frees the GPU, and vLLM serves the export after.
The GPU is released when the function returns. Expect ~40 minutes of H100 time.
"""

from __future__ import annotations

import json

import modal

from wspx.cloud.app import models_volume, vllm_lab_image

app = modal.App("wspx-inkling-h100")
MODEL = "/models/thinkingmachines--Inkling-2-Mini"
TEXTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a haiku about the ocean.",
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than 50.",
]
IMAGE_QUESTION = "What shape and colour is in this image? Answer in one sentence."
HF_TOKENS = 64
VLLM_TOKENS = 128


def _image(path: str) -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((96, 96, 352, 352), fill=(220, 30, 30))
    img.save(path)


def _lab_phase(report: dict) -> None:  # type: ignore[type-arg]
    import gc
    import time
    from pathlib import Path

    import torch
    from safetensors import safe_open

    from wspx.lab import analysis, checkpoint
    from wspx.lab.sdk import open_lab

    work = Path("/tmp/work")
    (work / "prompts").mkdir(parents=True, exist_ok=True)
    _image(str(work / "prompts" / "circle.png"))
    started = time.time()
    lab = open_lab(MODEL, store="/tmp/store", workdir=work)
    report["lab_load_s"] = round(time.time() - started, 1)
    served = lab.tree.served
    report["stored_formats"] = {
        "tensors": len(served.formats) if served else 0,
        "nvfp4": sum(f.kind == "nvfp4" for f in served.formats.values()) if served else 0,
        "mxfp8": sum(f.kind == "mxfp8" for f in served.formats.values()) if served else 0,
    }
    device = next(lab.model.parameters()).device
    chat = lambda t: {"messages": [{"role": "user", "content": t}]}  # noqa: E731
    image = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "prompts/circle.png"},
                    {"type": "text", "text": IMAGE_QUESTION},
                ],
            }
        ]
    }

    def forward_ms(prompt: object, reps: int = 3) -> dict:  # type: ignore[type-arg]
        batch = lab.text.batch([prompt], device)
        with torch.no_grad():
            analysis.forward(lab.model, lab.ir, batch.ids, batch.mask, batch.extra)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                analysis.forward(lab.model, lab.ir, batch.ids, batch.mask, batch.extra)
            torch.cuda.synchronize()
        return {
            "tokens": int(batch.ids.shape[1]),
            "ms": round((time.perf_counter() - t) / reps * 1000, 1),
        }

    def generate(prompt: object) -> dict:  # type: ignore[type-arg]
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = lab.generate(prompt, max_tokens=HF_TOKENS)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        n = len(out.data["ids"][0])
        return {
            "text": out.data["outputs"][0],
            "tokens": n,
            "s": round(seconds, 2),
            "tok_s": round(n / seconds, 2),
        }

    def measure(label: str) -> None:
        report[label] = {
            "forward_text": forward_ms(chat(TEXTS[0])),
            "forward_image": forward_ms(image),
            "generate_text": generate(chat(TEXTS[0])),
            "generate_image": generate(image),
        }

    measure("hf_base")

    # ---- edits: 4-bit (NVFP4) and 8-bit (MXFP8) experts, and one the 4-bit grid erases
    torch.manual_seed(0)
    width = int(lab.model.config.get_text_config().hidden_size)
    direction = torch.randn(width)
    edits = [
        (
            "nvfp4 project_out, every expert of L10",
            "project_out",
            {"direction": direction, "targets": "L10.mlp.experts.down_proj"},
        ),
        (
            "nvfp4 scale, experts 0-15 of L5",
            "scale",
            {"targets": "L5.mlp.experts.gate_up_proj[0:16]", "factor": 1.3},
        ),
        (
            "mxfp8 project_out, every expert of L21",
            "project_out",
            {"direction": direction, "targets": "L21.mlp.experts.down_proj"},
        ),
        (
            "mxfp8 scale, part of expert 3 of L22",
            "scale",
            {"targets": "L22.mlp.experts.gate_up_proj[3, 0:256, :]", "factor": 0.8},
        ),
    ]
    rows = []
    for label, op, args in edits:
        t = time.perf_counter()
        try:
            ref = lab.edit(op, **args)
        except Exception as exc:  # recorded, not fatal: the report says what failed
            rows.append({"edit": label, "error": f"{type(exc).__name__}: {exc}"})
            continue
        row = {"edit": label, "s": round(time.perf_counter() - t, 2)}
        row["survival"] = [d.message for d in ref.report.diagnostics if d.code in ("W205", "E110")]
        exact = []
        for name in sorted(set(ref.record.tensors)):
            stored = served.stored(name) if served else None
            if stored is None:
                continue
            loaded = checkpoint.from_checkpoint(lab.model, stored)[name]
            exact.append(
                bool(torch.equal(loaded, dict(lab.model.named_parameters())[name].detach()))
            )
        row["live_equals_stored"] = all(exact) if exact else None
        rows.append(row)
    try:
        lab.edit("scale", targets="L3.mlp.experts.gate_up_proj[2, :, 1:5]", factor=1.03)
        rows.append({"edit": "tiny nvfp4 nudge", "refused": False})
    except Exception as exc:
        rows.append(
            {"edit": "tiny nvfp4 nudge", "refused": "E110" in str(exc), "why": str(exc)[:300]}
        )
    report["edits"] = rows
    measure("hf_edited")

    # ---- export in the stored format, and check unedited experts kept their bytes
    t = time.perf_counter()
    info = lab.export("merged", "../export").data
    report["export"] = {
        "s": round(time.perf_counter() - t, 1),
        "files": info["files"],
        "warnings": info["warnings"],
    }
    key = "model.llm.layers.15.mlp.experts.w13_weight"
    out_files = sorted(Path("/tmp/export").glob("model*.safetensors"))
    src_files = sorted(Path(MODEL).glob("model*.safetensors"))

    def read(files: list, name: str) -> torch.Tensor:  # type: ignore[type-arg]
        for f in files:
            with safe_open(str(f), framework="pt") as fh:
                if name in fh.keys():  # noqa: SIM118
                    return fh.get_tensor(name)
        raise KeyError(name)

    report["export"]["unedited_layer_bytes_identical"] = bool(
        torch.equal(read(out_files, key), read(src_files, key))
    )
    edited = "model.llm.layers.10.mlp.experts.w2_weight"
    report["export"]["edited_layer_bytes_changed"] = not bool(
        torch.equal(read(out_files, edited), read(src_files, edited))
    )
    lab.__dict__.clear()
    gc.collect()


VLLM_SCRIPT = r"""
import base64, json, sys, time
from vllm import LLM, SamplingParams
path, texts, question, image, tokens = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
t = time.perf_counter()
llm = LLM(model=path, max_model_len=16384, gpu_memory_utilization=0.88, seed=0)
load = time.perf_counter() - t
data = "data:image/png;base64," + base64.b64encode(open(image, "rb").read()).decode()
sp = SamplingParams(temperature=0, max_tokens=tokens)
def run(messages):
    t = time.perf_counter()
    outs = llm.chat(messages, sp, use_tqdm=False)
    s = time.perf_counter() - t
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    return outs, s, n
llm.chat([{"role": "user", "content": texts[0]}], SamplingParams(max_tokens=4), use_tqdm=False)  # warm up
res = {"load_s": round(load, 1)}
outs, s, n = run([[{"role": "user", "content": texts[0]}]])
res["text_single"] = {
    "text": outs[0].outputs[0].text, "tokens": n, "s": round(s, 2), "tok_s": round(n / s, 1)
}
img = [[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": data}}, {"type": "text", "text": question}
]}]]
outs, s, n = run(img)
res["image_single"] = {
    "text": outs[0].outputs[0].text, "tokens": n, "s": round(s, 2), "tok_s": round(n / s, 1),
    "prompt_tokens": len(outs[0].prompt_token_ids),
}
outs, s, n = run([[{"role": "user", "content": t}] for t in texts] * 16)
res["batch_64"] = {"tokens": n, "s": round(s, 2), "tok_s": round(n / s, 1)}
res["answers"] = [o.outputs[0].text for o in outs[: len(texts)]]
print("WSPX" + json.dumps(res))
"""


def _vllm(path: str) -> dict:  # type: ignore[type-arg]
    import subprocess

    done = subprocess.run(
        [
            "/opt/vllm/bin/python",
            "-c",
            VLLM_SCRIPT,
            path,
            json.dumps(TEXTS),
            IMAGE_QUESTION,
            "/tmp/work/prompts/circle.png",
            str(VLLM_TOKENS),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    line = next((x for x in done.stdout.splitlines() if x.startswith("WSPX")), None)
    if line is None:
        return {"error": done.stderr[-4000:]}
    return json.loads(line[4:])  # type: ignore[no-any-return]


@app.function(
    image=vllm_lab_image,
    gpu="H100",
    cpu=8.0,
    memory=160 * 1024,
    timeout=3 * 3600,
    volumes={"/models": models_volume.read_only()},
)
def inkling() -> dict:  # type: ignore[type-arg]
    import subprocess
    import sys
    import traceback
    from pathlib import Path

    # the lab runs in a process of its own, so all of its GPU memory is gone before vLLM
    # starts (seen: clearing the lab in-process left ~56 GB held, and vLLM ran out)
    child = (
        "import json, sys, traceback; sys.path.insert(0, '/root'); import inkling_h100 as m\n"
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
    del traceback
    if "export" in report:
        report["vllm_edited"] = _vllm("/tmp/export")
    report["vllm_base"] = _vllm(MODEL)
    return report


@app.local_entrypoint()
def main() -> None:
    report = inkling.remote()
    print(json.dumps(report, indent=2, default=str))
