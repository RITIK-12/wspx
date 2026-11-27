"""Edits reaching a running vLLM, no reload, from a lab that holds no copy of the model: the
phases ``live_h100.py`` runs, one model per process. Imports no Modal (a plain process
cannot import the Modal app)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

PROMPTS = [
    "The capital of France is",
    "Write one sentence about the ocean.",
    "List three prime numbers greater than 50.",
    "Explain in one sentence why the sky is blue.",
]
BATCH = 64
TOKENS = 64


def _lab(model: str, work: Path) -> tuple[Any, Any, Any]:
    """A lab with no copy of the model beside vLLM, planned the way the lab server plans it."""
    import resource

    import torch

    from wspx.config import VllmSection
    from wspx.lab.engines.vllm import VLLMEngine, can_serve, plan
    from wspx.lab.loader import float_sizes, resolve_source
    from wspx.lab.sdk import open_lab

    source = resolve_source(model, local_only=True)
    ok, why = can_serve(source)
    assert ok, why
    on_disk = sum(f.stat().st_size for f in source.glob("*.safetensors"))
    _, _, largest = float_sizes(model, local_only=True)
    settings = VllmSection(agreement_prompts=len(PROMPTS), max_model_len=4096)
    (work / "proj").mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    lab = open_lab(model, store=work / "store", workdir=work / "proj", weights="checkpoint")
    lab_s = time.perf_counter() - t
    held = torch.cuda.memory_allocated(0)
    p = plan(settings, checkpoint_bytes=on_disk, resident_bytes=held + 2 * largest, weightless=True)
    t = time.perf_counter()
    engine = VLLMEngine(
        lab.model,
        tree=lab.tree,
        ir=lab.ir,
        source=source,
        settings=settings,
        plan=p,
        state_key=lab.tree_key,
        log=work / "vllm.log",
    )
    lab.engine = engine
    setup = {
        "plan": p.describe(),
        "lab_open_s": round(lab_s, 1),
        "vllm_start_s": round(time.perf_counter() - t, 1),
        "lab_gpu_gb": round(held / 2**30, 2),
        "lab_host_peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
        "checkpoint_gb": round(on_disk / 2**30, 1),
        "info": engine.info,
    }
    return lab, engine, setup


def _answers(engine: Any, lab: Any) -> tuple[list[list[int]], float]:
    t = time.perf_counter()
    rows = engine.generate_scored(lab.text, PROMPTS, max_tokens=TOKENS)
    return [ids for ids, _ in rows], time.perf_counter() - t


def _throughput(engine: Any, lab: Any) -> dict[str, Any]:
    prompts = (PROMPTS * (BATCH // len(PROMPTS)))[:BATCH]
    t = time.perf_counter()
    out = engine.complete(lab.text, prompts, max_tokens=TOKENS)
    s = time.perf_counter() - t
    n = sum(len(ids) for ids, _, _ in out)
    return {"prompts": BATCH, "tokens": n, "s": round(s, 2), "tok_s": round(n / s, 1)}


def run(model: str, edits: list[dict[str, Any]], out: str) -> None:
    import traceback

    report: dict[str, Any] = {"model": model}
    work = Path("/tmp/live") / model.replace("/", "--")
    try:
        t = time.perf_counter()
        lab, engine, report["setup"] = _lab(model, work)
        report["setup"]["s"] = round(time.perf_counter() - t, 1)
        report["base_agreement"] = engine.check_agreement(lab.text, PROMPTS)
        base, s = _answers(engine, lab)
        report["base_answers_s"] = round(s, 2)
        report["base_throughput"] = _throughput(engine, lab)
        rows = []
        for e in edits:
            t = time.perf_counter()
            try:
                ref = lab.edit(e["op"], **e["args"])
            except Exception as exc:
                rows.append({"edit": e["label"], "error": f"{type(exc).__name__}: {exc}"[:800]})
                continue
            row: dict[str, Any] = {"edit": e["label"], "edit_s": round(time.perf_counter() - t, 2)}
            row["survival"] = [
                d.message for d in ref.report.diagnostics if d.code in ("W205", "E110")
            ]
            row["sync"] = engine.sync()  # the push alone, timed
            row["agreement"] = engine.check_agreement(lab.text, PROMPTS)
            answers, s = _answers(engine, lab)
            row["answers_changed"] = sum(a != b for a, b in zip(answers, base, strict=True))
            row["answers_s"] = round(s, 2)
            rows.append(row)
        report["edits"] = rows
        report["edited_throughput"] = _throughput(engine, lab)
        # back to the base: every layer a push sent must come back exactly
        lab.discard()
        report["discard_sync"] = engine.sync()
        again, _ = _answers(engine, lab)
        report["base_again_exact"] = sum(a == b for a, b in zip(again, base, strict=True))
        report["syncs"] = engine.syncs
        engine.close()
    except Exception:
        report["error"] = traceback.format_exc()[-6000:]
    log = work / "vllm.log"
    if log.exists():
        lines = log.read_text(errors="replace").splitlines()
        report["vllm_log"] = [
            x for x in lines if "Injected" in x or "Failed to load" in x or "rror" in x
        ][-20:]
    Path(out).write_text(json.dumps(report, default=str))


QWEN_EDITS = [
    {"label": "scale an MLP writer (L14 down_proj) by 0.9", "op": "scale",
     "args": {"targets": "L14.mlp.down_proj", "factor": 0.9}},
    {"label": "scale attention output (L3 o_proj) by 1.2", "op": "scale",
     "args": {"targets": "L3.self_attn.o_proj", "factor": 1.2}},
]  # fmt: skip
INKLING_EDITS = [
    {"label": "nvfp4 scale, experts 0-15 of L5", "op": "scale",
     "args": {"targets": "L5.mlp.experts.gate_up_proj[0:16]", "factor": 1.3}},
    {"label": "mxfp8 scale, part of expert 3 of L22", "op": "scale",
     "args": {"targets": "L22.mlp.experts.gate_up_proj[3, 0:256, :]", "factor": 0.8}},
]  # fmt: skip
