"""One release, every kind of model: edit, check transformers and vLLM agree, export, load
the export in stock vLLM, check again. Quantized checkpoints report survival.

Each case is a real checkpoint, small enough for one 80 GB GPU. A model vLLM cannot serve
(Inkling-2-Mini with vLLM 0.34) runs the transformers half and records why.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.release

CASES: list[dict[str, Any]] = [
    {"id": "Qwen/Qwen3-0.6B", "kind": "dense"},
    {"id": "allenai/OLMoE-1B-7B-0125-Instruct", "kind": "moe"},
    {"id": "Qwen/Qwen3-VL-2B-Instruct", "kind": "vlm"},
    {"id": "Qwen/Qwen3-0.6B-FP8", "kind": "fp8-blocks"},
    {"id": "Qwen/Qwen3-4B-AWQ", "kind": "int4"},
    {"id": "thinkingmachines/Inkling-2-Mini", "kind": "looped-nvfp4", "vllm": False},
]
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "User: Name three primary colours.\nAssistant:",
    "2 + 2 =",
]
VLLM_PYTHON = os.environ.get("WSPX_VLLM_PYTHON", "/opt/vllm/bin/python")
REPORT = Path(os.environ.get("WSPX_RELEASE_REPORT", "release-matrix.jsonl"))


def _writer(lab: Any) -> str:
    """A residual writer in the middle layer: what every model has and edits touch."""
    keys = [
        k for k in lab.ir.params if (r := lab.ir.primary_role(k)) is not None and r.kind == "writer"
    ]
    if not keys:
        pytest.skip("no residual writer found")
    return lab.ir.key_of(keys[len(keys) // 2])


def _stock_vllm(path: Path, prompts: list[str], ids: list[list[int]]) -> list[list[int]]:
    """Greedy answers from stock vLLM loading ``path`` as any user would."""
    script = textwrap.dedent(
        f"""
        import json, sys
        from vllm import LLM, SamplingParams, TokensPrompt
        llm = LLM(model={str(path)!r}, max_model_len=2048, gpu_memory_utilization=0.35, seed=0)
        outs = llm.generate([TokensPrompt(prompt_token_ids=i) for i in {ids!r}],
                            SamplingParams(temperature=0, max_tokens=16))
        print("WSPX" + json.dumps([list(o.outputs[0].token_ids) for o in outs]))
        """
    )
    done = subprocess.run(
        [VLLM_PYTHON, "-c", script], capture_output=True, text=True, timeout=1800, check=False
    )
    line = next((x for x in done.stdout.splitlines() if x.startswith("WSPX")), None)
    if line is None:
        raise AssertionError(f"stock vLLM could not load the export:\n{done.stderr[-3000:]}")
    return json.loads(line[4:])


@pytest.mark.parametrize("case", CASES, ids=[c["kind"] for c in CASES])
def test_edit_agree_export_reload(case: dict[str, Any], tmp_path: Path) -> None:
    import torch

    from wspx.config import VllmSection
    from wspx.lab.engines.vllm import VLLMEngine, plan
    from wspx.lab.loader import resolve_source
    from wspx.lab.sdk import open_lab

    assert torch.cuda.is_available(), "the release matrix needs a GPU"
    row: dict[str, Any] = {"model": case["id"], "kind": case["kind"]}
    lab = open_lab(case["id"], store=tmp_path / "store", workdir=tmp_path)
    text = lab.text
    serve = case.get("vllm", True)
    engine = None
    if serve:
        source = resolve_source(case["id"], local_only=True)
        on_disk = sum(f.stat().st_size for f in source.glob("*.safetensors"))
        settings = VllmSection(agreement_prompts=len(PROMPTS), max_model_len=2048)
        engine = VLLMEngine(
            lab.model,
            tree=lab.tree,
            ir=lab.ir,
            source=source,
            settings=settings,
            plan=plan(settings, checkpoint_bytes=on_disk, hf_bytes=torch.cuda.memory_allocated(0)),
            state_key=lab.tree_key,
        )
        lab.engine = engine
        row["base"] = engine.check_agreement(text, PROMPTS)

    target = _writer(lab)
    ref = lab.edit.scale(target, factor=0.9)
    row["edit"] = target
    row["survival"] = [d.message for d in ref.report.diagnostics if d.code in ("W205", "E110")]
    if lab.tree.served is not None:
        assert row["survival"], "a quantized checkpoint's edit reports how much survived"
    if engine is not None:
        row["edited"] = engine.check_agreement(text, PROMPTS)
        assert row["edited"]["agree"] >= 0.75, row
        assert row["edited"]["kl_first"] is None or row["edited"]["kl_first"] < 0.05, row

    out = tmp_path / "export"
    info = lab.export("merged", "export").data
    row["export_warnings"] = info.get("warnings", [])
    assert not [w for w in row["export_warnings"] if "differs" in w or "not written" in w], row
    if engine is not None:
        ids = [text.ids(p) for p in PROMPTS]
        served = [r for r, _ in engine.generate_scored(text, PROMPTS, max_tokens=16)]
        reloaded = _stock_vllm(out, PROMPTS, ids)
        row["reload_matches"] = sum(a == b for a, b in zip(served, reloaded, strict=True))
        assert row["reload_matches"] >= len(PROMPTS) - 1, row
        engine.close()
    else:
        row["vllm"] = "not served: " + (
            "vLLM 0.34 has no implementation of this architecture" if not serve else ""
        )
    with REPORT.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
