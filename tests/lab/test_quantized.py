"""Edits to a checkpoint stored quantized (NVFP4 and MXFP8, as Inkling's experts are).

The codecs are checked against bytes from the real Inkling-2-Mini checkpoint; the rest on a
toy Inkling stored the same way: transformers dequantizes it on load, the lab writes each
edit back in the stored format, and what it tests is what loading that export gives.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import torch

transformers = pytest.importorskip("transformers")

from safetensors.torch import load_file, save_file  # noqa: E402

from tests.lab.test_conformance import _tiny_inkling  # noqa: E402
from wspx.errors import CheckFailed  # noqa: E402
from wspx.lab import checkpoint, quant  # noqa: E402
from wspx.lab.sdk import open_lab  # noqa: E402

DATA = Path(__file__).parent / "data" / "inkling2_mini_experts.safetensors"
EXPERTS = re.compile(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight$")


def test_codecs_reproduce_the_real_checkpoints_bytes() -> None:
    """Decoding Inkling's stored blocks and encoding them again gives the same bytes, and
    the values are those transformers' own dequantizer gives."""
    t = load_file(str(DATA))
    nv = quant.Format(
        "nvfp4",
        "nvfp4.w2_weight",
        "nvfp4.w2_weight_scale",
        "nvfp4.w2_weight_amax",
        "nvfp4.w2_weight_outer_scale_denominator",
    )
    x = quant.decode(nv, t)
    factor = t["nvfp4.w2_weight_amax"] / t["nvfp4.w2_weight_outer_scale_denominator"]
    codes, scale = quant.nvfp4_encode(x[0], factor[0])
    assert torch.equal(codes, t["nvfp4.w2_weight"][0])
    assert torch.equal(scale.view(torch.uint8), t["nvfp4.w2_weight_scale"][0].view(torch.uint8))
    mx = quant.Format("mxfp8", "mxfp8.w2_weight", "mxfp8.w2_weight_scale")
    values, exps = quant.mxfp8_encode(quant.decode(mx, t)[0])
    assert torch.equal(values.view(torch.uint8), t["mxfp8.w2_weight"][0].view(torch.uint8))
    assert torch.equal(exps, t["mxfp8.w2_weight_scale"][0])

    from transformers.core_model_loading import Inkling2LoopDequantizeExperts

    op = Inkling2LoopDequantizeExperts()
    parts = {"w": t[nv.weight], "s": t[nv.scale], "a": t[nv.amax], "d": t[nv.denominator]}
    ref = op.convert(parts, ["w", "s", "a", "d"], ["o"], model=None, full_layer_name=None)["o"]
    assert torch.equal(ref, x.to(torch.bfloat16))


def test_requantize_changes_only_the_blocks_an_edit_touches() -> None:
    t = load_file(str(DATA))
    nv = quant.Format(
        "nvfp4",
        "nvfp4.w2_weight",
        "nvfp4.w2_weight_scale",
        "nvfp4.w2_weight_amax",
        "nvfp4.w2_weight_outer_scale_denominator",
    )
    delta = torch.zeros(1, 16, 1152)
    delta[0, 3, 40:44] = 0.01  # inside block 2 of row 3
    done = quant.requantize(nv, {k: t[k] for k in nv.parts}, delta)
    assert done.blocks == 1
    codes = done.parts[nv.weight][0].view(16, 72, 8)
    before = t[nv.weight][0].view(16, 72, 8)
    changed = (codes != before).any(-1).nonzero().tolist()
    assert changed == [[3, 2]]
    kept = quant.requantize(nv, {k: t[k] for k in nv.parts}, delta, keep_scales=True)
    assert torch.equal(
        kept.parts[nv.scale].view(torch.uint8), t[nv.scale].view(torch.uint8)
    )  # only codes move


def _stored_quantized(path: Path) -> tuple[Any, Path]:
    """A toy Inkling saved as the Hub's is: NVFP4 experts in layers 0-19, MXFP8 in 20-23."""
    model = _tiny_inkling().to(torch.bfloat16)
    src = path / "src"
    model.save_pretrained(str(src))
    tensors: dict[str, torch.Tensor] = {}
    for f in sorted(src.glob("*.safetensors")):
        tensors.update(load_file(str(f)))
        f.unlink()
    (src / "model.safetensors.index.json").unlink(missing_ok=True)
    nvfp4: list[str] = []
    for name in list(tensors):
        m = EXPERTS.match(name)
        if not m:
            continue
        x = tensors[name].float()
        if int(m.group(1)) < 20:
            amax = x.abs().flatten(1).amax(1)
            den = torch.tensor([2688.0])
            codes, scale = quant.nvfp4_encode(x, amax / den)
            tensors[name] = codes
            tensors[f"{name}_scale"] = scale
            tensors[f"{name}_amax"] = amax
            tensors[f"{name}_outer_scale_denominator"] = den
            nvfp4.append(name)
        else:
            values, exps = quant.mxfp8_encode(x)
            tensors[name], tensors[f"{name}_scale"] = values, exps
    save_file(tensors, str(src / "model.safetensors"))
    config = json.loads((src / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    (src / "config.json").write_text(json.dumps(config))
    rule = {"scale_selection_rule": "fouroversix_mae"}
    groups = [
        {"compute_dtype": "nvfp4", "required_activation_quantization": rule, "tensors": nvfp4},
        {
            "compute_dtype": "bfloat16",
            "weight_block_shape": [],
            "tensors": ["model.llm.norm.weight"],
        },
    ]
    (src / "hf_quant_config.json").write_text(json.dumps({"model": {"parameter_groups": groups}}))
    loaded = transformers.AutoModelForImageTextToText.from_pretrained(
        str(src), dtype=torch.bfloat16
    ).eval()
    return loaded, src


GATE_UP = "model.language_model.layers.3.mlp.experts.gate_up_proj"


@pytest.fixture
def lab(tmp_path: Path) -> Any:
    model, src = _stored_quantized(tmp_path)
    (tmp_path / "work").mkdir()
    return open_lab(model, store=tmp_path / "store", workdir=tmp_path / "work", source_dir=src)


def test_an_edit_is_written_in_the_stored_format_and_tested_as_served(
    lab: Any, tmp_path: Path
) -> None:
    served = lab.tree.served
    assert served is not None
    assert len(served.formats) == 48
    base = lab.tree.base(GATE_UP).clone()
    ref = lab.edit.scale("L3.mlp.experts.gate_up_proj[2]", factor=1.5)
    live = dict(lab.model.named_parameters())[GATE_UP].detach()
    assert torch.equal(live[[0, 1, 3]], base[[0, 1, 3]])  # other experts untouched, bit for bit
    assert not torch.equal(live[2], base[2])
    # what the lab holds is what loading the requantized tensors gives
    loaded = checkpoint.from_checkpoint(lab.model, served.stored(GATE_UP))[GATE_UP]
    assert torch.equal(loaded, live)
    survival = served.last[GATE_UP]
    assert survival.format == "NVFP4"
    assert 0.9 < survival.ratio < 1.1
    assert survival.cosine > 0.95
    assert (survival.clipped, survival.rescaled) == (0, 1)  # its amax grew with it
    codes = [d for d in ref.report.diagnostics if d.code == "W205"]
    assert codes
    assert "survives NVFP4" in codes[0].message

    out = tmp_path / "out"
    info = lab.export("merged", "../out").data
    assert not any("dequantized" in w for w in info["warnings"]), info["warnings"]
    written = {}
    for f in out.glob("model*.safetensors"):
        written.update(load_file(str(f)))
    source = load_file(str(tmp_path / "src" / "model.safetensors"))
    assert written["model.llm.layers.3.mlp.experts.w13_weight"].dtype == torch.uint8
    assert torch.equal(
        written["model.llm.layers.4.mlp.experts.w13_weight"],
        source["model.llm.layers.4.mlp.experts.w13_weight"],
    )
    assert "quantization_config" in json.loads((out / "config.json").read_text())
    again = transformers.AutoModelForImageTextToText.from_pretrained(str(out), dtype=torch.bfloat16)
    params = dict(lab.model.named_parameters())
    for name, p in again.named_parameters():
        assert torch.equal(p, params[name]), name


def test_an_edit_the_grid_erases_is_refused(lab: Any) -> None:
    state = lab.tree.state
    with pytest.raises(CheckFailed, match="E110"):
        lab.edit.scale("L3.mlp.experts.gate_up_proj[2, :, 1:5]", factor=1.03)
    assert lab.tree.state == state
    assert not lab.staged


def test_mxfp8_layers_and_promote_mode(lab: Any, tmp_path: Path) -> None:
    down = "model.language_model.layers.21.mlp.experts.down_proj"
    lab.edit.scale("L21.mlp.experts.down_proj[5]", factor=0.5)
    assert lab.tree.served.last[down].format == "MXFP8"
    lab.edit.scale("L3.mlp.experts.gate_up_proj[1]", factor=1.001, precision="promote")
    assert GATE_UP not in lab.tree.served.parts  # exact: not requantized
    info = lab.export("merged", "../out").data
    assert any("written in bf16 (promote mode)" in w for w in info["warnings"])
    config = json.loads((tmp_path / "out" / "hf_quant_config.json").read_text())
    groups = config["model"]["parameter_groups"]
    nvfp4 = next(g for g in groups if g["compute_dtype"] == "nvfp4")["tensors"]
    assert "model.llm.layers.3.mlp.experts.w13_weight" not in nvfp4
    written = load_file(str(next((tmp_path / "out").glob("model*.safetensors"))))
    assert written["model.llm.layers.3.mlp.experts.w13_weight"].dtype == torch.bfloat16
    assert "model.llm.layers.3.mlp.experts.w13_weight_scale" not in written


def test_vllm_gets_the_edited_layer_in_its_stored_format(lab: Any) -> None:
    import httpx

    from wspx.config import VllmSection
    from wspx.lab.engines.vllm import Transport, VLLMEngine

    pushed: dict[str, torch.Tensor] = {}

    class Record(Transport):
        def send(self, http: Any, chunks: Any, version: str) -> int:
            for chunk in chunks:
                pushed.update(chunk)
            return 0

    ready = httpx.MockTransport(lambda r: httpx.Response(200, json={"vllm": "0.34.0"}))
    engine = VLLMEngine(
        lab.model,
        tree=lab.tree,
        ir=lab.ir,
        source=lab.source_dir,
        settings=VllmSection(agreement_prompts=0),
        http=httpx.Client(base_url="http://w", transport=ready),
        transport=Record(),
        state_key=lab.tree_key,
    )
    lab.edit.scale("L3.mlp.experts.gate_up_proj[2]", factor=1.5)
    engine.sync()
    w13 = "model.llm.layers.3.mlp.experts.w13_weight"
    assert pushed[w13].dtype == torch.uint8
    assert {f"{w13}_scale", f"{w13}_amax", f"{w13}_outer_scale_denominator"} <= set(pushed)
    assert torch.equal(pushed[w13], lab.tree.served.stored(GATE_UP)[w13])
    # that layer in full (its attention-residual queries are stored outside layers.3)
    assert not [k for k in pushed if re.search(r"\.layers\.(?!3\.)\d+\.", k)]
    assert "model.llm.layers.3.mlp.experts.w2_weight" in pushed  # its unedited part as stored
