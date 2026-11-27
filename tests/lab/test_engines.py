"""The engine seam: transformers by default, vLLM beside the lab with ``[lab] engine = "vllm"``.

There is no GPU or vLLM here, so a fake worker stands in for vLLM: a second copy of the
model behind the worker's HTTP protocol, loaded with whatever tensors the lab pushes.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch

pytest.importorskip("transformers")

from tests.lab.test_session import tiny_llama
from wspx.config import VllmSection
from wspx.errors import WspxError
from wspx.lab import analysis
from wspx.lab.engines import split_reasoning
from wspx.lab.engines.vllm import Transport, VLLMEngine, plan
from wspx.lab.sdk import open_lab
from wspx.lab.session import Lab
from wspx.lab.text import Text


class FakeWorker:
    """The worker's HTTP protocol over a copy of the model, as vLLM would hold it."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = copy.deepcopy(model)
        self.text = Text()
        self.requests: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/ready":
            return httpx.Response(200, json={"vllm": "0.34.0", "gpu": {"name": "Fake GPU"}})
        if path == "/generate":
            body = json.loads(request.content)
            self.requests.append(body)
            prompts = [r["ids"] for r in body["requests"]]
            rows = analysis.generate_scored(
                self.model,
                self.text,
                prompts,
                max_tokens=body["max_tokens"],
                temperature=body["temperature"],
                watch=body["watch"],
            )
            lines = [
                json.dumps(
                    {
                        "i": i,
                        "ids": ids,
                        "watched": watched,
                        "finish": "length",
                        "prompt_tokens": len(prompts[i]),
                    }
                )
                for i, (ids, watched) in reversed(list(enumerate(rows)))  # any order
            ]
            lines.append(json.dumps({"done": True, "seconds": 0.01}))
            return httpx.Response(200, content="\n".join(lines).encode())
        return httpx.Response(404, json={"detail": path})


class FakeTransport(Transport):
    """Loads pushed tensors into the fake worker's model by checkpoint name. ``with_every``:
    tensors the fake vLLM maps into every layer, so a push without them leaves modules
    incomplete (as vLLM's Inkling2Loop does with its message queries)."""

    def __init__(self, worker: FakeWorker, with_every: tuple[str, ...] = ()) -> None:
        self.worker = worker
        self.pushes: list[list[str]] = []
        self.with_every = with_every

    def send(
        self,
        http: httpx.Client,
        chunks: Iterable[dict[str, torch.Tensor]],
        version: str,
        more: Any = (),
    ) -> tuple[int, int]:
        state = self.worker.model.state_dict()
        names: list[str] = []
        sent = used = 0

        def load(cs: Iterable[dict[str, torch.Tensor]]) -> None:
            nonlocal sent
            for chunk in cs:
                for name, t in chunk.items():
                    with torch.no_grad():
                        state[name].copy_(t)
                    names.append(name)
                    sent += t.numel() * t.element_size()

        load(chunks)
        while self.with_every and not set(self.with_every) <= set(names):
            if used == len(more):
                raise WspxError("vLLM has only part of the tensors of a module")
            load(more[used]())
            used += 1
        self.pushes.append(names)
        return sent, used


def vllm_lab(tmp_path: Path, **settings: Any) -> tuple[Lab, FakeWorker, FakeTransport]:
    (tmp_path / "work").mkdir(exist_ok=True)
    lab = open_lab(tiny_llama(), store=tmp_path / "store", workdir=tmp_path / "work")
    worker = FakeWorker(lab.model)
    transport = FakeTransport(worker)
    lab.engine = VLLMEngine(
        lab.model,
        tree=lab.tree,
        ir=lab.ir,
        source=tmp_path / "none",
        settings=VllmSection(**settings),
        http=httpx.Client(base_url="http://worker", transport=httpx.MockTransport(worker.handle)),
        transport=transport,
        state_key=lab.tree_key,
    )
    return lab, worker, transport


def same_weights(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    sb = b.state_dict()
    return all(torch.equal(t, sb[k]) for k, t in a.state_dict().items())


PROMPTS = [[1, 9, 2, 11], [1, 9, 2, 30, 31]]


def test_vllm_generates_what_transformers_does_and_follows_every_edit(tmp_path: Path) -> None:
    lab, worker, transport = vllm_lab(tmp_path, agreement_prompts=2)
    hf = analysis.generate(lab.model, lab.text, PROMPTS, max_tokens=6)
    out = lab.generate(PROMPTS, max_tokens=6)
    assert out.data["ids"] == hf
    assert "generated by vLLM 0.34.0" in out.summary
    assert transport.pushes == []  # vLLM loaded the base: nothing to push
    assert lab.engine.agreement["exact"] == 2  # type: ignore[attr-defined]

    lab.edit.scale("L1.mlp.down", factor=1.5)
    edited = analysis.generate(lab.model, lab.text, PROMPTS, max_tokens=6)
    assert lab.generate(PROMPTS, max_tokens=6).data["ids"] == edited
    (pushed,) = transport.pushes
    assert all(".layers.1." in n for n in pushed)  # the whole edited layer, and only it
    assert len(pushed) == sum(1 for n, _ in lab.model.named_parameters() if ".layers.1." in n)
    assert same_weights(lab.model, worker.model)

    lab.discard()
    lab.generate(PROMPTS, max_tokens=6)
    assert len(transport.pushes) == 2
    assert same_weights(lab.model, worker.model)
    lab.generate(PROMPTS, max_tokens=6)
    assert len(transport.pushes) == 2  # nothing changed since


def test_adapter_edits_reach_vllm_merged(tmp_path: Path) -> None:
    """transformers runs adapter mode as a side path; vLLM gets base + delta in the weight."""
    lab, worker, _ = vllm_lab(tmp_path, agreement_prompts=0)
    lab.edit.project_out(torch.ones(48), "L0.mlp.down", precision="adapter")
    lab.generate(PROMPTS, max_tokens=2)
    name = next(n for n in lab.tree.state.adapter)
    served = worker.model.state_dict()[name]
    deltas = [lab.tree.delta(o) for o in lab.tree.state.adapter[name]]
    assert torch.equal(served, lab.tree.preview(name, deltas))
    assert not torch.equal(served, lab.tree.base(name))
    assert torch.equal(dict(lab.model.named_parameters())[name], lab.tree.base(name))


def test_suite_runs_are_cached_per_engine_and_say_which(tmp_path: Path) -> None:
    lab, _, _ = vllm_lab(tmp_path, agreement_prompts=0)
    out = lab.test("canary")
    run = out.data["runs"][0]
    assert run["engine"]["engine"] == "vllm"
    assert "vLLM 0.34.0" in out.summary
    stored = lab.store.glob(f"runs/{lab.tree_key()}", "*/*.json")
    assert any(p.endswith("-vllm.json") for p in stored)


def test_status_shows_the_engine(tmp_path: Path) -> None:
    lab, _, _ = vllm_lab(tmp_path, agreement_prompts=2)
    lab.generate(PROMPTS, max_tokens=2)
    status = lab.status().summary
    assert "engine: vLLM 0.34.0 on Fake GPU, weights base" in status
    assert "agrees with transformers on 2/2" in status


def test_complete_reports_finish_and_lengths_on_both_engines(tmp_path: Path) -> None:
    (tmp_path / "hf").mkdir()
    hf_lab = open_lab(tiny_llama(), store=tmp_path / "s1", workdir=tmp_path / "hf")
    rows = hf_lab.complete(PROMPTS, max_tokens=4)
    assert [r["prompt_tokens"] for r in rows] == [4, 5]
    assert all(r["finish"] in ("stop", "length") and r["output_tokens"] <= 4 for r in rows)
    lab, worker, _ = vllm_lab(tmp_path, agreement_prompts=0)
    again = lab.complete(PROMPTS, max_tokens=4)
    assert [r["ids"] for r in again] == [r["ids"] for r in rows]
    assert len(worker.requests[-1]["requests"]) == 2  # one batch: vLLM schedules it
    assert lab.progress is None


def test_split_reasoning() -> None:
    assert split_reasoning("<think>a b</think> 42") == ("a b", "42")
    assert split_reasoning("plan</think>\n\nanswer") == ("plan", "answer")
    assert split_reasoning("just this") == (None, "just this")


GB = 2**30


def test_plan_puts_each_engine_on_its_own_gpu_when_there_are_two() -> None:
    p = plan(VllmSection(), checkpoint_bytes=20 * GB, gpus=[("H100", 80 * GB), ("H100", 80 * GB)])
    assert (p.hf_device, p.vllm_gpu, p.memory) == ("cuda:0", 1, 0.90)
    assert "vLLM on GPU 1" in p.describe()


def test_plan_shares_one_gpu_or_says_why_not() -> None:
    p = plan(VllmSection(), checkpoint_bytes=16 * GB, gpus=[("A100", 80 * GB)], hf_bytes=17 * GB)
    assert p.hf_device is None
    assert 0.5 < p.memory < 0.75
    with pytest.raises(WspxError, match="do not fit on one 80 GB A100"):
        plan(VllmSection(), checkpoint_bytes=20 * GB, gpus=[("A100", 80 * GB)], hf_bytes=53 * GB)
    with pytest.raises(WspxError, match=r"raise \[lab\.vllm\] memory"):
        plan(
            VllmSection(memory=0.1),
            checkpoint_bytes=20 * GB,
            gpus=[("H100", 80 * GB), ("H100", 80 * GB)],
        )


def test_a_push_sends_only_changed_layers_and_a_failed_one_is_mended(tmp_path: Path) -> None:
    """vLLM reloads layer by layer and keeps what a push leaves out; after a push fails
    partway, what it holds is unknown, and the next push sends every layer."""
    lab, worker, transport = vllm_lab(tmp_path, agreement_prompts=0)
    lab.edit.scale("L1.mlp.down", factor=1.5)
    lab.generate(PROMPTS, max_tokens=2)
    assert transport.pushes[-1]
    assert all(".layers.1." in n for n in transport.pushes[-1])

    class Flaky(FakeTransport):
        def send(self, http: Any, chunks: Any, version: str, more: Any = ()) -> tuple[int, int]:
            next(iter(chunks))
            raise WspxError("vLLM worker /weights/update: 500 out of memory")

    lab.engine.transport = Flaky(worker)  # type: ignore[attr-defined]
    lab.edit.scale("L0.mlp.down", factor=0.5)
    with pytest.raises(WspxError, match="out of memory"):
        lab.generate(PROMPTS, max_tokens=2)
    assert lab.engine.stale  # type: ignore[attr-defined]
    lab.engine.transport = transport  # type: ignore[attr-defined]
    lab.generate(PROMPTS, max_tokens=2)
    every = {n for n, _ in lab.model.named_parameters()}
    assert set(transport.pushes[-1]) >= every - {"lm_head.weight"}
    assert not lab.engine.stale  # type: ignore[attr-defined]
    assert same_weights(lab.model, worker.model)


def test_a_model_that_maps_outside_tensors_into_its_layers_gets_them_each_push(
    tmp_path: Path,
) -> None:
    """vLLM's Inkling2Loop gives each layer a slice of two tensors kept outside the layer
    stack; a push of one layer alone leaves that layer's module incomplete. vLLM says so,
    the engine sends what lies outside the layers, and remembers to from then on."""
    lab, worker, _ = vllm_lab(tmp_path, agreement_prompts=0)
    transport = FakeTransport(worker, with_every=("model.norm.weight",))
    lab.engine.transport = transport  # type: ignore[attr-defined]
    lab.edit.scale("L1.mlp.down", factor=1.5)
    lab.generate(PROMPTS, max_tokens=2)
    assert "model.norm.weight" in transport.pushes[-1]
    assert lab.engine.with_outside  # type: ignore[attr-defined]
    assert not any(".layers.0." in n for n in transport.pushes[-1])  # not everything
    lab.edit.scale("L0.mlp.down", factor=0.5)
    lab.generate(PROMPTS, max_tokens=2)
    assert "model.norm.weight" in transport.pushes[-1]
    assert same_weights(lab.model, worker.model)


def test_the_worker_runs_from_its_own_folder_with_the_extension() -> None:
    """Next to wspx's engine module, ``import vllm`` would find wspx's ``vllm.py``."""
    from wspx.lab.engines import vllm as engine

    assert engine.WORKER.parent == engine.EXTENSION.parent
    assert (engine.WORKER.parent / "vllm.py").exists()  # why it cannot run from there
    source = engine.EXTENSION.read_text()
    assert not re.search(r"^\s*(import|from) wspx", source, re.M)
    assert engine.EXTENSION_CLS == "wspx_vllm_ext.WspxWorkerExtension"


def test_the_extension_lets_strict_loaders_take_part_of_a_checkpoint() -> None:
    import sys
    import types

    from wspx.lab.engines import vllm_ext

    active = {"on": False}
    reload = types.ModuleType("vllm.model_executor.model_loader.reload")
    reload.weight_update_active = lambda: active["on"]  # type: ignore[attr-defined]
    saved = {k: sys.modules.get(k) for k in ("vllm.model_executor.model_loader.reload",)}
    sys.modules["vllm.model_executor.model_loader.reload"] = reload

    class Strict:
        def load_weights(self, weights: Any) -> set[str]:
            names = {n for n, _ in weights}
            if names != {"a", "b"}:
                raise ValueError(f"Missing checkpoint tensors: {sorted({'a', 'b'} - names)}")
            return names

    try:
        vllm_ext._partial_ok(Strict)
        with pytest.raises(ValueError, match="Missing"):
            Strict().load_weights([("a", 1)])  # cold start: still strict
        active["on"] = True
        assert Strict().load_weights([("a", 1)]) == {"a"}  # an update: part is fine
        assert Strict().load_weights([("a", 1), ("b", 2)]) == {"a", "b"}
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_experts_that_repack_and_empty_their_weights_reload_in_place() -> None:
    """Inkling2's experts repack their checkpoint tensors into kernel tensors and empty
    them; vLLM's layerwise reload cannot redo that. The extension gives an emptied tensor
    its shape back when an update loads it, repacks after the update, and copies the result
    into the kernel tensors in use (captured CUDA graphs hold their addresses)."""
    import sys
    import types

    from wspx.lab.engines import vllm_ext

    active = {"on": False}
    reload = types.ModuleType("vllm.model_executor.model_loader.reload")
    reload.weight_update_active = lambda: active["on"]  # type: ignore[attr-defined]
    saved = sys.modules.get("vllm.model_executor.model_loader.reload")
    sys.modules["vllm.model_executor.model_loader.reload"] = reload

    class Humming:
        def __init__(self, m: torch.nn.Module) -> None:
            self.tensors = {
                p: {
                    "weight": getattr(m, f"{p}_weight") * 2,
                    "scale": getattr(m, f"{p}_weight_scale") + 1,
                }
                for p in ("w13", "w2")
            }

    class Experts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.nvfp4, self.num_experts = False, 2
            self.shapes = ((8, 64), (64, 32))
            for p, (n, k) in zip(("w13", "w2"), self.shapes, strict=True):
                self.register_parameter(
                    f"{p}_weight", torch.nn.Parameter(torch.zeros(2, n, k), requires_grad=False)
                )
                self.register_parameter(
                    f"{p}_weight_scale",
                    torch.nn.Parameter(torch.zeros(2, n, k // 32), requires_grad=False),
                )
            self.humming: Humming | None = None

        def _load_weight(self, param: torch.Tensor, loaded: torch.Tensor) -> None:
            param.data.copy_(loaded)

        def _load_block_scales(self, param: torch.Tensor, loaded: torch.Tensor) -> None:
            param.data.copy_(loaded)

        def process_weights_after_loading(self) -> None:
            self.humming = Humming(self)
            for p in ("w13", "w2"):
                for s in ("weight", "weight_scale"):
                    t = getattr(self, f"{p}_{s}")
                    t.data = t.data.new_empty((0,))

    try:
        vllm_ext._patch_experts(Experts)
        assert Experts._skip_layerwise_reload  # type: ignore[attr-defined]
        m = Experts()
        m.process_weights_after_loading()  # cold start
        assert m.humming is not None
        in_use = m.humming
        address = in_use.tensors["w2"]["weight"].data_ptr()
        active["on"] = True
        new: dict[str, torch.Tensor] = {}
        for p, (n, k) in zip(("w13", "w2"), m.shapes, strict=True):
            new[f"{p}_weight"] = torch.full((2, n, k), 3.0)
            new[f"{p}_weight_scale"] = torch.full((2, n, k // 32), 3.0)
        for name, value in new.items():
            loader = m._load_block_scales if "scale" in name else m._load_weight
            loader(getattr(m, name), value)
        assert m in vllm_ext._TOUCHED
        active["on"] = False
        vllm_ext._TOUCHED.clear()
        vllm_ext._rebuild_experts(m)
        assert m.humming is in_use  # the object the graphs read
        assert in_use.tensors["w2"]["weight"].data_ptr() == address
        assert torch.equal(in_use.tensors["w2"]["weight"], torch.full((2, 64, 32), 6.0))
        assert m.w13_weight.numel() == 0  # emptied again
    finally:
        if saved is None:
            sys.modules.pop("vllm.model_executor.model_loader.reload", None)
        else:
            sys.modules["vllm.model_executor.model_loader.reload"] = saved
