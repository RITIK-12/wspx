from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tests.lab.test_edit import make
from wspx.lab import merge, patch
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
