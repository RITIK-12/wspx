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
