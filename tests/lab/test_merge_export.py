from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tests.lab.test_edit import make
from wspx.lab import checkpoint, merge, patch
from wspx.lab.export import export_bundle, export_merged, export_peft
from wspx.lab.ops import get
from wspx.lab.tensors import tensor_hash
from wspx.lab.tree import TreeState

transformers = pytest.importorskip("transformers")


def rank1(
    shape: tuple[int, int], u: torch.Tensor, v: torch.Tensor, index=None
) -> patch.TensorDelta:  # type: ignore[no-untyped-def]
    return patch.lowrank("w", shape, "b", u, v, index=index)


class TestOverlap:
    def test_disjoint_slices_same_direction_still_conflict(self) -> None:
        d = torch.randn(16)
        a = rank1((16, 8), d, torch.ones(4), index=(slice(None), slice(0, 4)))
        b = rank1((16, 8), d, torch.ones(4), index=(slice(None), slice(4, 8)))
        ov = merge.overlap("w", [a], [b])
        assert ov.slice == 0.0
        assert ov.out_space == pytest.approx(1.0, abs=1e-5)
        assert merge.choose(ov, None) == "needs-strategy"

    def test_orthogonal_edits_sum(self) -> None:
        e = torch.eye(16)
        a = rank1((16, 8), e[0], torch.ones(8))
        b = rank1((16, 8), e[1], torch.ones(8))
        ov = merge.overlap("w", [a], [b], acts=None)
        assert ov.out_space == pytest.approx(0.0, abs=1e-6)
        # same input direction, so they read the same inputs
        assert ov.in_space == pytest.approx(1.0, abs=1e-5)

    def test_activation_overlap(self) -> None:
        x = torch.randn(64, 8)
        a = torch.outer(torch.randn(16), torch.eye(8)[0])
        b = torch.outer(torch.randn(16), torch.eye(8)[0])
        assert merge.activation_overlap(a, b, x) == pytest.approx(1.0, abs=1e-5)

    def test_ties(self) -> None:
        a = patch.dense("w", (4,), "b", torch.tensor([1.0, -1.0, 0.1, 2.0]))
        b = patch.dense("w", (4,), "b", torch.tensor([3.0, 1.0, 0.0, -0.5]))
        merged = merge.ties([a], [b], keep=0.5).dense()
        assert merged[0] == pytest.approx(2.0)  # both kept and agree: disjoint mean of 1 and 3
        assert merged[1] == pytest.approx(0.0)  # the signs cancel, so no value is elected
        assert merged[3] == pytest.approx(2.0)  # b's -0.5 was trimmed


class TestExport:
    def test_merged_is_bit_identical(self, tmp_path: Path) -> None:
        model, ir, tree, ctx = make(torch.bfloat16)
        tree.stage(get("project_out")(ctx, torch.randn(32), "L*.mlp.down"))
        info = export_merged(model, ir, tmp_path / "out")
        from safetensors.torch import load_file

        loaded = load_file(str(tmp_path / "out" / "model.safetensors"))
        for name, param in model.named_parameters():
            assert tensor_hash(loaded[name]) == tensor_hash(param), name
        assert info["files"] == ["model.safetensors"]

    def test_sharding(self, tmp_path: Path) -> None:
        model, ir, _, _ = make()
        info = export_merged(model, ir, tmp_path / "out", shard_bytes=4096)
        assert len(info["files"]) > 1
        index = json.loads((tmp_path / "out" / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"]) >= {n for n, _ in model.named_parameters()}

    def test_peft_adapter_reproduces_the_edit(self, tmp_path: Path) -> None:
        peft = pytest.importorskip("peft")
        model, ir, tree, ctx = make(torch.float32)
        base = make(torch.float32)[0]
        tree.stage(get("project_out")(ctx, torch.randn(32), "L*.mlp.down"))
        tree.stage(
            get("rank_one_write")(ctx, "L0.attn.o", key=torch.randn(32), value=torch.randn(32))
        )
        info = export_peft(model, ir, tree, tmp_path / "adapter")
        assert info["svd_error"] == {}
        adapted = peft.PeftModel.from_pretrained(base, str(tmp_path / "adapter")).eval()
        ids = torch.randint(0, 40, (2, 7))
        with torch.no_grad():
            assert torch.allclose(adapted(ids), model(ids), atol=1e-5)

    def test_peft_conv1d_layout(self, tmp_path: Path) -> None:
        peft = pytest.importorskip("peft")
        from wspx.lab.ir.capture import build_ir
        from wspx.lab.ops import OpContext
        from wspx.lab.tree import WorkingTree

        cfg = transformers.GPT2Config(vocab_size=64, n_embd=32, n_layer=2, n_head=4, n_positions=32)
        torch.manual_seed(0)
        model = transformers.GPT2LMHeadModel(cfg).eval()
        torch.manual_seed(0)
        base = transformers.GPT2LMHeadModel(cfg).eval()
        ir = build_ir(model)
        store: dict[str, patch.TensorDelta] = {}
        tree = WorkingTree(model, ir, load_delta=store.__getitem__)
        ctx = OpContext(ir=ir, tree=tree, resolve=ir.resolve, direction=torch.as_tensor)
        tree.stage(get("project_out")(ctx, torch.randn(32), "L*.mlp.down"))
        info = export_peft(model, ir, tree, tmp_path / "adapter")
        cfg_json = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
        assert cfg_json["fan_in_fan_out"] is True
        assert info["modules"] == ["transformer.h.0.mlp.c_proj", "transformer.h.1.mlp.c_proj"]
        adapted = peft.PeftModel.from_pretrained(base, str(tmp_path / "adapter")).eval()
        ids = torch.randint(0, 64, (1, 6))
        with torch.no_grad():
            assert torch.allclose(adapted(ids).logits, model(ids).logits, atol=1e-5)

    def test_peft_svd_fallback_reports_error(self, tmp_path: Path) -> None:
        model, ir, tree, ctx = make()
        tree.stage(get("scale")(ctx, "L0.mlp.down", factor=1.5))
        info = export_peft(model, ir, tree, tmp_path / "a", max_rank=4)
        assert info["svd_error"]["blocks.0.w_out.weight"] > 0

    def test_bundle(self, tmp_path: Path) -> None:
        out = export_bundle({"sha256:" + "a" * 64: b"x"}, {"base": "fp:1"}, tmp_path / "b.tar.gz")
        assert out["objects"] == 1
        assert (tmp_path / "b.tar.gz").stat().st_size > 0

    def test_restore_after_export(self) -> None:
        model, _, tree, ctx = make()
        before = tensor_hash(model.blocks[0].w_out.weight)
        tree.stage(get("ablate")(ctx, "L0.mlp.down[:, 0]"))
        tree.checkout(TreeState())
        assert tensor_hash(model.blocks[0].w_out.weight) == before


def _collapse(name: str) -> str:
    return re.sub(r"\.\d+$", ".N", re.sub(r"\.\d+\.", ".N.", name))


def _written(out: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    tensors: dict[str, torch.Tensor] = {}
    for f in sorted(out.glob("model*.safetensors")):
        tensors.update(load_file(str(f)))
    return tensors


class TestCheckpointNames:
    """Seen in a review: merged exports kept transformers' in-memory names, so a model it
    renames or fuses on load (Inkling loads its interleaved ``w13_weight`` as ``gate_up_proj``)
    exported files that nothing else could load."""

    IR = SimpleNamespace(params={}, fingerprint="test")

    @staticmethod
    def _hub_like(model: torch.nn.Module, path: Path) -> None:
        """A source checkpoint shaped like Inkling's on the Hub: an index, a quantization
        config, draft heads outside the index, a tokenizer folder and a chat template."""
        model.save_pretrained(str(path), max_shard_size="100KB")  # the checkpoint's own names
        config = json.loads((path / "config.json").read_text())
        config["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
        (path / "config.json").write_text(json.dumps(config))
        (path / "hf_quant_config.json").write_text("{}")
        save_file({"fuse_weight": torch.ones(2)}, str(path / "mtp.safetensors"))
        (path / "tiktoken").mkdir()
        (path / "tiktoken" / "tokenizer.model").write_text("tokens")
        (path / "chat_template.jinja").write_text("{{ messages }}")

    def test_inkling_round_trips_under_its_checkpoint_names(self, tmp_path: Path) -> None:
        from tests.lab.test_conformance import _tiny_inkling

        model = _tiny_inkling()
        self._hub_like(model, tmp_path / "src")
        with torch.no_grad():  # edits to a fused tensor and to a merged one
            model.model.language_model.layers[7].mlp.experts.gate_up_proj[3, :5] += 1.0
            model.model.language_model.message_input.query[2, 1] -= 0.5
        info = export_merged(model, self.IR, tmp_path / "out", source=tmp_path / "src")  # type: ignore[arg-type]
        out = tmp_path / "out"
        index = tmp_path / "src" / "model.safetensors.index.json"
        source = set(json.loads(index.read_text())["weight_map"])
        assert set(_written(out)) == source
        assert info["names"] == "checkpoint"
        config = json.loads((out / "config.json").read_text())
        assert "quantization_config" not in config
        assert not (out / "hf_quant_config.json").exists()
        assert any("dequantized on load" in w for w in info["warnings"])
        assert (out / "mtp.safetensors").is_file()
        index = json.loads((out / "model.safetensors.index.json").read_text())
        assert "fuse_weight" not in index["weight_map"]
        assert (out / "tiktoken" / "tokenizer.model").read_text() == "tokens"
        assert (out / "chat_template.jinja").is_file()
        assert not any("differs" in w or "not written" in w for w in info["warnings"])

        again = transformers.AutoModelForImageTextToText.from_pretrained(
            str(out), dtype=torch.float32
        )
        live = dict(model.named_parameters())
        for name, param in again.named_parameters():
            assert torch.equal(param, live[name]), name

    def test_names_match_the_real_inkling_checkpoint(self, tmp_path: Path) -> None:
        """The real checkpoint's names (from its safetensors header) minus the NVFP4/MXFP8
        scales, which a dequantized export has no use for."""
        from tests.lab.test_conformance import _tiny_inkling

        export_merged(_tiny_inkling(), self.IR, tmp_path / "out")  # type: ignore[arg-type]
        real = {
            line
            for line in (Path(__file__).parent / "data" / "inkling2_mini_names.txt")
            .read_text()
            .splitlines()
            if line and not line.startswith("#")
        }
        real = {n for n in real if not checkpoint.QUANT_AUX.search(n)}
        assert {_collapse(n) for n in _written(tmp_path / "out")} == real
