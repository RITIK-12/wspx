"""Every probe a tool runs gives the same answer on a lab without weights (vLLM holds the
model; the lab reads each layer from the checkpoint for its forward pass) as with the model
in memory. Tiny Inkling: looped layers, experts, a vision tower.

Seen live on Inkling-2-Mini: trace read its writers' weights after the forward pass, when
the layers held placeholders again, and failed (cuda vs cpu); expert probes matched expert
slices by where the weights were before the pass; the map was built without seeing a
single layer weight; attribution's backward ran on placeholders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("transformers")

from tests.lab.test_conformance import _tiny_inkling
from wspx.lab import analysis
from wspx.lab.probe import expert_writes
from wspx.lab.sdk import open_lab
from wspx.lab.session import Lab

PROMPT = [1, 9, 2, 11, 40, 41, 42]
REFERENCE = [1, 9, 2, 12, 50, 51, 52]
EXPERTS = "L7.mlp.experts.down_proj"


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Lab, Lab]:
    tmp = tmp_path_factory.mktemp("inkling")
    _tiny_inkling().to(torch.bfloat16).save_pretrained(tmp / "ckpt")
    (tmp / "a").mkdir()
    (tmp / "b").mkdir()
    held = open_lab(str(tmp / "ckpt"), store=tmp / "sa", workdir=tmp / "a")
    light = open_lab(str(tmp / "ckpt"), store=tmp / "sb", workdir=tmp / "b", weights="checkpoint")
    assert light.tree.virtual
    return held, light


def close(a: list[Any], b: list[Any]) -> bool:
    """The same units in the same order, with scores equal to float noise."""
    return [u for u, _ in a] == [u for u, _ in b] and all(
        abs(x - y) <= 1e-3 * max(1.0, abs(x)) for (_, x), (_, y) in zip(a, b, strict=True)
    )


def test_the_map_is_built_from_the_streamed_layers(pair: tuple[Lab, Lab]) -> None:
    held, light = pair

    def roles(lab: Lab) -> list[tuple[Any, ...]]:
        return sorted((r.param, r.kind, r.in_dim, r.out_dim, r.address) for r in lab.ir.roles)

    assert roles(light) == roles(held)
    assert any(r[1] == "writer" for r in roles(light))
    experts = light.resolve(EXPERTS)[0].param
    assert light.ir.passes_of(experts) == held.ir.passes_of(experts) == [1, 2, 3]


def test_trace_ranks_the_same_writers_experts_included(pair: tuple[Lab, Lab]) -> None:
    held, light = pair
    a, b = (lab.trace(PROMPT, top=8).data for lab in pair)
    assert close(a["writers"], b["writers"])
    assert all(s != 0 for _, s in b["writers"])
    with torch.no_grad(), expert_writes(light.model, light.ir) as routed:
        analysis.forward(light.model, light.ir, torch.tensor([PROMPT]), torch.ones(1, 7))
    assert routed.calls  # every expert routed at the last token
    assert routed.norms  # its weights read while the pass had them
    ours = analysis._top_expert_writers(light.model, light.ir, routed, top=8)
    with torch.no_grad(), expert_writes(held.model, held.ir) as theirs:
        analysis.forward(held.model, held.ir, torch.tensor([PROMPT]), torch.ones(1, 7))
    assert close(ours, analysis._top_expert_writers(held.model, held.ir, theirs, top=8))
    assert ours
    assert all(s != 0 for _, s in ours)


def test_a_diff_trace_names_the_edited_expert(pair: tuple[Lab, Lab]) -> None:
    for lab in pair:
        lab.edit.scale(f"{EXPERTS}[0:8]", factor=3.0)
    try:
        a, b = (lab.trace(PROMPT, vs="base", top=8).data for lab in pair)
    finally:
        for lab in pair:
            lab.discard()
    assert a["first_divergence"] == b["first_divergence"] is not None
    assert close(a["moved"], b["moved"])
    assert close(a["edited"], b["edited"])
    assert any(u.startswith(EXPERTS) for u, _ in b["edited"])


def test_attribution_reads_the_weights_again_for_backward(pair: tuple[Lab, Lab]) -> None:
    _, light = pair
    a, b = (lab.trace(PROMPT, reference=REFERENCE, top=8).data for lab in pair)
    assert a["target"] == b["target"]
    assert [u for u, _ in a["ranking"][:8]] == [u for u, _ in b["ranking"][:8]]
    assert any(s != 0 for _, s in b["ranking"])
    from wspx.lab.weights import is_placeholder

    streamed = light.model._wspx_streamed  # type: ignore[union-attr]
    assert all(is_placeholder(p) for b in streamed.blocks for p in b.parameters())
    assert not streamed.live  # nothing left behind on the GPU


def test_activations_see_each_experts_input(pair: tuple[Lab, Lab]) -> None:
    prompts = [PROMPT, REFERENCE]
    a, b = (lab.activations(prompts, scope=EXPERTS, top=10) for lab in pair)
    assert "kernel is fused" not in b.summary
    assert [h.address for h in a] == [h.address for h in b]
    assert b.hits
    assert all(h.expert is not None for h in b)
    dense_a, dense_b = (lab.activations(prompts, top=10) for lab in pair)
    assert [h.address for h in dense_a] == [h.address for h in dense_b]


def test_grep_reads_only_what_it_ranks(pair: tuple[Lab, Lab]) -> None:
    _, light = pair
    light.tree.weights._cache.clear()
    a, b = (lab.grep("colnorm > 2σ", scope="L3.self_attn.o_proj", top=5) for lab in pair)
    assert [h.address for h in a] == [h.address for h in b]
    read = set(light.tree.weights._cache)
    assert read
    assert all(".layers.3." in n for n in read)  # not the whole model first


def test_a_streamed_backward_leaves_no_weights_behind(tmp_path: Path) -> None:
    """The saved-tensor hooks hand back exactly what the forward pass used."""
    from wspx.lab.offload import refetch_weights

    ckpt = tmp_path / "ckpt"
    from tests.lab.test_session import tiny_llama

    tiny_llama().to(torch.bfloat16).save_pretrained(ckpt)
    (tmp_path / "w").mkdir()
    light = open_lab(str(ckpt), store=tmp_path / "s", workdir=tmp_path / "w", weights="checkpoint")
    held = open_lab(str(ckpt), store=tmp_path / "s2", workdir=tmp_path / "w")
    grads = []
    for lab in (held, light):
        embed = lab.model.get_input_embeddings()
        x = embed(torch.tensor([[1, 9, 2, 11]])).detach().float().requires_grad_(True)
        with refetch_weights(lab.model):
            out = lab.model(inputs_embeds=x.to(torch.bfloat16), use_cache=False).logits
            out[0, -1, 5].float().backward()
        assert x.grad is not None
        grads.append(x.grad)
    assert torch.allclose(grads[0], grads[1], atol=1e-3)
    assert grads[1].abs().sum() > 0
