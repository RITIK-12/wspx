"""A lab without a copy of the model (vLLM holds it): the same lab, read from the checkpoint.

The working tree's values are the checkpoint's plus the edits, read when asked for, and a
forward pass runs a layer at a time. Everything a lab does must come out the same as with
the model in memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from tests.lab.test_session import POS, tiny_llama
from wspx.lab import analysis
from wspx.lab.sdk import open_lab
from wspx.lab.session import Lab

PROMPTS = [[1, 9, 2, 11], [1, 9, 2, 30, 31]]


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Lab, Lab]:
    model = tiny_llama().to(torch.bfloat16)
    ckpt = tmp_path / "ckpt"
    model.save_pretrained(ckpt)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    held = open_lab(str(ckpt), store=tmp_path / "sa", workdir=tmp_path / "a")
    light = open_lab(str(ckpt), store=tmp_path / "sb", workdir=tmp_path / "b", weights="checkpoint")
    return held, light


def logits(lab: Lab) -> torch.Tensor:
    batch = lab.text.batch(PROMPTS, lab.device)
    with torch.no_grad():
        return analysis.forward(lab.model, lab.ir, batch.ids, batch.mask, batch.extra).float()


def same_values(a: Lab, b: Lab) -> bool:
    return all(torch.equal(a.params[n].detach(), b.params[n].detach().cpu()) for n in a.params)


def test_the_skeleton_holds_no_layer_weights(pair: tuple[Lab, Lab]) -> None:
    from wspx.lab.weights import is_placeholder

    _, light = pair
    in_layers = [p for n, p in light.model.named_parameters() if ".layers." in n]
    assert in_layers
    assert all(is_placeholder(p) for p in in_layers)
    storages = {p.untyped_storage().data_ptr(): p.untyped_storage().nbytes() for p in in_layers}
    assert sum(storages.values()) <= 8  # one shared element per dtype, not the layers
    assert light.tree.virtual
    assert not light.tree.weights.edited


def test_values_and_logits_match_at_the_base(pair: tuple[Lab, Lab]) -> None:
    held, light = pair
    assert held.ir.fingerprint == light.ir.fingerprint
    assert same_values(held, light)
    assert torch.equal(logits(held), logits(light).cpu())


def test_edits_commits_and_discards_match(pair: tuple[Lab, Lab]) -> None:
    held, light = pair
    for lab in pair:
        lab.edit.scale("L1.mlp.down_proj", factor=1.5)
        lab.edit.scale("L0.self_attn.o_proj", factor=0.5)
    assert same_values(held, light)
    assert torch.equal(logits(held), logits(light).cpu())
    assert set(light.tree.weights.edited) == {
        "model.layers.1.mlp.down_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    }
    for lab in pair:
        lab.commit("two edits")
        lab.edit.scale("L2.mlp.down_proj", factor=2.0)
        lab.discard()
    assert same_values(held, light)
    assert "model.layers.2.mlp.down_proj.weight" not in light.tree.weights.edited


def test_probes_and_export_match(pair: tuple[Lab, Lab], tmp_path: Path) -> None:
    held, light = pair
    for lab in pair:
        lab.edit.scale("L1.mlp.down_proj", factor=1.25)
    a = held.generate(POS[:3], max_tokens=5)
    b = light.generate(POS[:3], max_tokens=5)
    assert a.data["ids"] == b.data["ids"]  # decoding with a KV cache, a layer at a time
    held.export("merged", "../out-held")
    light.export("merged", "../out-light")
    from safetensors.torch import load_file

    def tensors(d: Path) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for f in sorted(d.glob("*.safetensors")):
            out.update(load_file(str(f)))
        return out

    x, y = tensors(tmp_path / "out-held"), tensors(tmp_path / "out-light")
    assert x.keys() == y.keys()
    assert all(torch.equal(x[k], y[k]) for k in x)


def test_edits_reach_vllm_from_a_lab_without_weights(tmp_path: Path) -> None:
    import httpx
    import transformers

    from tests.lab.test_engines import FakeTransport, FakeWorker
    from wspx.config import VllmSection
    from wspx.lab.engines.vllm import VLLMEngine

    ckpt = tmp_path / "ckpt"
    tiny_llama().to(torch.bfloat16).save_pretrained(ckpt)
    (tmp_path / "w").mkdir()
    light = open_lab(str(ckpt), store=tmp_path / "s", workdir=tmp_path / "w", weights="checkpoint")
    served = transformers.AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16)
    worker = FakeWorker(served)  # what vLLM holds: the only full copy
    transport = FakeTransport(worker)
    light.engine = VLLMEngine(
        light.model,
        tree=light.tree,
        ir=light.ir,
        source=ckpt,
        settings=VllmSection(agreement_prompts=0),
        http=httpx.Client(base_url="http://worker", transport=httpx.MockTransport(worker.handle)),
        transport=transport,
        state_key=light.tree_key,
    )
    light.edit.scale("L1.mlp.down_proj", factor=1.5)
    light.generate(PROMPTS, max_tokens=3)
    assert transport.pushes[-1]
    assert all(".layers.1." in n for n in transport.pushes[-1])
    for n, p in worker.model.named_parameters():
        assert torch.equal(p.detach(), light.params[n].detach().cpu()), n
    light.discard()
    light.generate(PROMPTS, max_tokens=3)
    for n, p in worker.model.named_parameters():
        assert torch.equal(p.detach(), light.params[n].detach().cpu()), n


def _quantized_checkpoint(path: Path, kind: str) -> dict[str, torch.Tensor]:
    """tiny_llama written as a GPTQ or FP8-tiled checkpoint, as those quantizers write it."""
    import json

    from safetensors.torch import save_file

    from tests.lab.test_codecs import _fp8_block, _gptq

    model = tiny_llama()
    path.mkdir()
    model.config.architectures = ["LlamaForCausalLM"]
    model.config.save_pretrained(path)
    out: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        w = p.detach().float()
        stem = name.removesuffix(".weight")
        if ".layers." in name and w.dim() == 2 and "norm" not in name:
            if kind == "gptq":
                t = _gptq(w, 4, 16, torch.randperm(w.shape[1]))
                t.pop("_q")
                out.update({f"{stem}.{k.removeprefix('m.')}": v for k, v in t.items()})
            else:
                codes, s = _fp8_block(w, 16)
                out[name], out[f"{stem}.weight_scale_inv"] = codes, s
                out[f"{stem}.input_scale"] = torch.ones(1)
        else:
            out[name] = p.detach().to(torch.bfloat16)
    save_file(out, str(path / "model.safetensors"))
    cfg = json.loads((path / "config.json").read_text())
    cfg["quantization_config"] = (
        {"quant_method": "gptq", "bits": 4, "group_size": 16, "sym": True, "desc_act": True}
        if kind == "gptq"
        else {"quant_method": "fp8", "weight_block_size": [16, 16], "activation_scheme": "dynamic"}
    )
    (path / "config.json").write_text(json.dumps(cfg))
    return out


@pytest.mark.parametrize("kind", ["gptq", "fp8"])
def test_a_quantized_checkpoint_is_read_edited_and_written_in_its_format(
    tmp_path: Path, kind: str
) -> None:
    from safetensors.torch import load_file

    from wspx.lab import quant

    ckpt = tmp_path / "ckpt"
    stored = _quantized_checkpoint(ckpt, kind)
    (tmp_path / "w").mkdir()
    lab = open_lab(str(ckpt), store=tmp_path / "s", workdir=tmp_path / "w", weights="checkpoint")
    served = lab.tree.served
    assert served is not None
    name = "model.layers.1.mlp.down_proj.weight"
    fmt = served.formats[name]
    assert fmt.kind == kind
    expected = quant.decode(fmt, {k: stored[k] for k in fmt.parts}).to(torch.bfloat16)
    assert torch.equal(lab.params[name].cpu(), expected)
    assert torch.isfinite(logits(lab)).all()
    ref = lab.edit.scale("L1.mlp.down_proj", factor=1.1)
    (survival,) = [d for d in ref.report.diagnostics if d.code == "W205"]
    assert f"survives {kind.upper()}" in survival.message
    parts = served.stored(name)
    assert parts is not None
    assert set(parts) == set(fmt.parts)
    assert torch.equal(lab.params[name].cpu(), quant.decode(fmt, parts).to(torch.bfloat16))
    lab.export("merged", "../out")
    written = load_file(str(tmp_path / "out" / "model.safetensors"))

    def raw(t: torch.Tensor) -> torch.Tensor:
        return t.contiguous().view(torch.uint8)

    edited = [k for k in fmt.parts if not torch.equal(raw(written[k]), raw(stored[k]))]
    # GPTQ moves codes, and groups pushed past their range get larger scales (zeros and the
    # group map stay); FP8 grows each tile's scale by 1.1, leaving every code as it was
    stem = "model.layers.1.mlp.down_proj"
    expect = (
        [f"{stem}.qweight", f"{stem}.scales"] if kind == "gptq" else [f"{stem}.weight_scale_inv"]
    )
    assert edited == expect, edited
    for k, v in stored.items():
        if ".layers.0." in k or ".layers.2." in k:
            assert torch.equal(raw(written[k]), raw(v)), k
    if kind == "fp8":
        assert "model.layers.0.mlp.down_proj.input_scale" in written


def test_a_tied_head_the_checkpoint_does_not_store_is_the_embedding(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    model = tiny_llama().to(torch.bfloat16)
    model.config.tie_word_embeddings = True
    model.config.architectures = ["LlamaForCausalLM"]
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    model.config.save_pretrained(ckpt)
    state = {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file({k: v.contiguous() for k, v in state.items()}, str(ckpt / "model.safetensors"))
    (tmp_path / "w").mkdir()
    light = open_lab(str(ckpt), store=tmp_path / "s", workdir=tmp_path / "w", weights="checkpoint")
    assert "lm_head.weight" not in dict(light.model.named_parameters())
    assert torch.isfinite(logits(light)).all()
