from __future__ import annotations

import copy
import random
from pathlib import Path

import pytest
import torch

from tests.lab.conftest import TinyLM
from wspx.config import CheckSection
from wspx.errors import PatchError
from wspx.lab import patch
from wspx.lab.check import (
    EditPlan,
    SmokeMetrics,
    degenerate_reasons,
    numeric_check,
    perplexity,
    smoke_report,
    static_check,
    token_kl,
)
from wspx.lab.ir.capture import build_ir
from wspx.lab.ir.model_ir import ModelIR
from wspx.lab.ops import OpContext, get, registry
from wspx.lab.patch import TensorDelta, materialize
from wspx.lab.tensors import tensor_hash
from wspx.lab.tree import TreeState, WorkingTree


def make(
    dtype: torch.dtype = torch.float32, seed: int = 0
) -> tuple[TinyLM, ModelIR, WorkingTree, OpContext]:
    torch.manual_seed(seed)
    model = TinyLM().eval().to(dtype)
    ir = build_ir(model, example_inputs=(torch.randint(0, 40, (2, 10)),))
    store: dict[str, TensorDelta] = {}

    def load(oid: str) -> TensorDelta:
        return store[oid]

    tree = WorkingTree(model, ir, load_delta=load)
    ctx = OpContext(
        ir=ir,
        tree=tree,
        resolve=ir.resolve,
        direction=lambda d: torch.as_tensor(d, dtype=torch.float32),
    )
    return model, ir, tree, ctx


class TestDeltaCodec:
    def test_roundtrip_and_content_address(self) -> None:
        d = patch.lowrank(
            "w", (4, 3), "sha256:base", torch.ones(4), torch.arange(3.0), index=None, scale=0.5
        )
        again = TensorDelta.from_bytes(d.to_bytes())
        assert again.oid == d.oid
        assert torch.equal(again.dense(), d.dense())
        assert torch.equal(d.dense(), 0.5 * torch.outer(torch.ones(4), torch.arange(3.0)))

    def test_index_roundtrip(self) -> None:
        d = patch.dense("w", (3, 5), "b", torch.ones(2, 2), index=(slice(0, 2), [1, 4]), kind="set")
        again = TensorDelta.from_bytes(d.to_bytes())
        assert again.index == (slice(0, 2, None), [1, 4])
        assert again.dense()[:, [1, 4]].sum() == 4

    def test_sparse_requires_unique(self) -> None:
        with pytest.raises(PatchError, match="unique"):
            patch.sparse("w", (4,), "b", torch.tensor([1, 1]), torch.tensor([1.0, 2.0]))

    def test_compact_prefers_sparse(self) -> None:
        block = torch.zeros(10, 10)
        block[3, 4] = 2.0
        small = patch.compact(patch.dense("w", (10, 10), "b", block))
        assert small.kind == "sparse"
        assert torch.equal(small.dense(), block)


class TestMaterialize:
    def test_order_independent_and_single_rounding(self) -> None:
        torch.manual_seed(0)
        base = torch.randn(32, 16).to(torch.bfloat16)
        deltas = [
            patch.lowrank("w", (32, 16), "b", torch.randn(32, 3) * 1e-3, torch.randn(16, 3)),
            patch.dense(
                "w", (32, 16), "b", torch.randn(4, 16) * 1e-2, index=(slice(4, 8), slice(None))
            ),
            patch.sparse(
                "w", (32, 16), "b", torch.tensor([0, 17, 300]), torch.tensor([1e-3, -2e-3, 5e-4])
            ),
        ]
        expected = tensor_hash(materialize(base, deltas))
        for _ in range(5):
            random.shuffle(deltas)
            assert tensor_hash(materialize(base, deltas)) == expected

    def test_matches_fp32_reference(self) -> None:
        base = torch.randn(8, 8)
        d = patch.lowrank("w", (8, 8), "b", torch.randn(8), torch.randn(8))
        out = materialize(base, [d])
        assert torch.allclose(out, base + d.dense(), atol=0, rtol=0)

    def test_chunked_lowrank_adds_are_bit_identical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Chunking (for embedding-sized tensors) must not change a bit: commits store hashes."""
        base = torch.randn(37, 16)
        d = patch.lowrank("w", (37, 16), "b", torch.randn(37, 3), torch.randn(16, 3), scale=0.7)
        whole = materialize(base, [d])
        monkeypatch.setattr(patch, "LOWRANK_CHUNK", 16 * 5)  # five rows at a time
        assert torch.equal(materialize(base, [d]), whole)


class TestWorkingTree:
    def test_stage_and_restore_bit_exact(self) -> None:
        model, _ir, tree, ctx = make(torch.bfloat16)
        name = "blocks.0.w_out.weight"
        before = tensor_hash(model.blocks[0].w_out.weight)
        deltas = get("ablate")(ctx, "L0.mlp.down[:, 3]")
        tree.stage(deltas)
        assert model.blocks[0].w_out.weight[:, 3].abs().sum() == 0
        assert tree.hash(name) != before
        tree.checkout(TreeState())
        assert tensor_hash(model.blocks[0].w_out.weight) == before
        assert tree.hash(name) == before

    def test_eviction_rereads_base(self) -> None:
        torch.manual_seed(0)
        model = TinyLM().eval()
        pristine = copy.deepcopy(model.state_dict())
        ir = build_ir(model, example_inputs=(torch.randint(0, 40, (2, 10)),))
        reads: list[str] = []

        def reader(name: str) -> torch.Tensor:
            reads.append(name)
            return pristine[name].clone()

        store: dict[str, TensorDelta] = {}
        tree = WorkingTree(
            model, ir, load_delta=store.__getitem__, base_reader=reader, cache_bytes=0
        )
        ctx = OpContext(ir=ir, tree=tree, resolve=ir.resolve, direction=torch.as_tensor)
        for d in get("scale")(ctx, "L*.mlp.down", factor=2.0):
            store[d.oid] = d
        tree.stage(get("scale")(ctx, "L*.mlp.down", factor=2.0))
        tree.checkout(TreeState())
        assert reads  # bases were evicted and re-read
        assert torch.equal(model.blocks[1].w_out.weight, pristine["blocks.1.w_out.weight"])

    def test_promote_keeps_fp32_and_runs(self) -> None:
        model, _, tree, ctx = make(torch.bfloat16)
        tree.stage(get("scale")(ctx, "L1.mlp.down", factor=1.001), precision="promote")
        assert model.blocks[1].w_out.weight.dtype == torch.float32
        out = model(torch.randint(0, 40, (1, 5)))
        assert out.dtype == torch.bfloat16
        tree.checkout(TreeState())
        assert model.blocks[1].w_out.weight.dtype == torch.bfloat16

    def test_adapter_matches_merged(self) -> None:
        model, _, tree, ctx = make(torch.float32)
        ids = torch.randint(0, 40, (1, 6))
        d = torch.randn(32)
        deltas = get("project_out")(ctx, d, "L0.mlp.down")
        tree.stage(deltas)
        merged = model(ids)
        tree.checkout(TreeState())
        tree.stage(deltas, precision="adapter")
        adapted = model(ids)
        assert torch.allclose(merged, adapted, atol=1e-5)
        assert tree.state.adapter
        assert not tree.state.merged


class TestOps:
    def test_registry_lists_builtins(self) -> None:
        assert {
            "ablate",
            "project_out",
            "bake_steer",
            "rank_one_write",
            "transplant",
            "orthogonalize",
        } <= set(registry())

    def test_project_out_writer_columns(self) -> None:
        model, _, tree, ctx = make()
        d = torch.randn(32)
        tree.stage(get("project_out")(ctx, d, "L0.mlp.down", cols=[1, 5]))
        w = model.blocks[0].w_out.weight.detach()
        unit = d / d.norm()
        assert torch.allclose(unit @ w[:, [1, 5]], torch.zeros(2), atol=1e-5)
        assert (unit @ w[:, 0]).abs() > 1e-3  # untouched column keeps its component

    def test_project_out_reader(self) -> None:
        model, _, tree, ctx = make()
        d = torch.randn(32)
        tree.stage(get("project_out")(ctx, d, "L0.attn.q"))
        w = model.blocks[0].qkv.weight.detach()
        unit = d / d.norm()
        assert torch.allclose(w[:32] @ unit, torch.zeros(32), atol=1e-5)  # q rows no longer read d
        assert (w[32:64] @ unit).abs().max() > 1e-3  # k untouched

    def test_rank_one_write(self) -> None:
        model, _, tree, ctx = make()
        k, v = torch.randn(48), torch.randn(32)
        tree.stage(get("rank_one_write")(ctx, "L1.mlp.down", key=k, value=v))
        assert torch.allclose(model.blocks[1].w_out.weight.detach() @ k, v, atol=1e-4)

    def test_bake_steer_uses_a_writer_bias(self) -> None:
        model, _, tree, ctx = make()
        before = model.blocks[0].proj.bias.detach().clone()
        d = torch.randn(32)
        tree.stage(get("bake_steer")(ctx, d, scale=2.0, at=0))
        assert torch.allclose(
            model.blocks[0].proj.bias.detach() - before, 2.0 * d / d.norm(), atol=1e-5
        )

    def test_bake_steer_via_embedding(self) -> None:
        model, _, tree, ctx = make()
        before = model.tok.weight.detach().clone()
        d = torch.randn(32)
        tree.stage(get("bake_steer")(ctx, d, scale=2.0, at=-1))
        shift = model.tok.weight.detach() - before
        assert torch.allclose(shift, (2.0 * d / d.norm()).expand_as(shift), atol=1e-5)

    def test_bake_steer_without_bias_fails_helpfully(self) -> None:
        _, _, _, ctx = make()
        with pytest.raises(PatchError, match="no writer bias"):
            get("bake_steer")(ctx, torch.randn(32), scale=1.0, at=7)

    def test_transplant(self) -> None:
        model, ir, tree, ctx = make(seed=0)
        torch.manual_seed(1)
        donor = TinyLM().eval()
        ctx.sources["donor"] = (ir, donor)
        tree.stage(get("transplant")(ctx, "L1.mlp.down", source="donor"))
        # a set is an additive fp32 delta: exact up to fp32 rounding of (src - cur)
        assert torch.allclose(
            model.blocks[1].w_out.weight, donor.blocks[1].w_out.weight, atol=1e-7, rtol=0
        )

    def test_orthogonalize(self) -> None:
        _, _, _, ctx = make()
        a = get("project_out")(ctx, torch.randn(32), "L0.mlp.down")
        b = get("project_out")(ctx, torch.randn(32), "L0.mlp.down")
        (b2,) = get("orthogonalize")(ctx, b, against=a)
        ua = a[0].u[:, 0] / a[0].u[:, 0].norm()
        assert torch.allclose(ua @ b2.dense(), torch.zeros(48), atol=1e-5)


class TestChecks:
    cfg = CheckSection()

    def test_clean_edit_passes(self) -> None:
        _, ir, tree, ctx = make()
        plan = EditPlan(
            get("project_out")(ctx, torch.randn(32), "L0.mlp.down", strength=0.01), op="project_out"
        )
        report = static_check(ir, plan)
        report.extend(numeric_check(ir, tree, plan, self.cfg))
        assert report.ok, report.summary()

    def test_noop_below_resolution_is_blocked(self) -> None:
        _, ir, tree, _ctx = make(torch.bfloat16)
        deltas = [
            patch.dense(
                t.param,
                ir.params[t.param].shape,
                ir.params[t.param].hash or "",
                torch.full(ir.params[t.param].shape, 1e-9),
            )
            for t in ir.resolve("L1.mlp.down")
        ]
        report = numeric_check(ir, tree, EditPlan(deltas), self.cfg)
        codes = {d.code for d in report.diagnostics}
        assert "E105" in codes
        assert "W203" in codes
        sub = next(d for d in report.diagnostics if d.code == "W203")
        assert sub.severity == "error"
        assert any(f.action.get("precision") == "promote" for f in sub.fixes)

    def test_partial_subulp_warns(self) -> None:
        _, ir, tree, _ = make(torch.bfloat16)
        name = "blocks.1.w_out.weight"
        delta = torch.zeros(ir.params[name].shape)
        delta[:, :20] = 1e-9  # vanishes
        delta[:, 20:] = 0.5  # survives, carries most of the mass
        report = numeric_check(
            ir, tree, EditPlan([patch.dense(name, delta.shape, "", delta)]), self.cfg
        )
        sub = [d for d in report.diagnostics if d.code == "W203"]
        assert not sub or sub[0].severity == "warning"

    def test_tied_and_reader_and_invalid_op(self) -> None:
        _, ir, _, ctx = make()
        tied = static_check(ir, EditPlan(get("scale")(ctx, "embed", factor=1.1), op="scale"))
        assert "W101" in {d.code for d in tied.diagnostics}
        steer = get("bake_steer")(ctx, torch.randn(32), scale=0.1, at=-1)  # via the embedding
        (w101,) = [
            d
            for d in static_check(ir, EditPlan(steer, op="bake_steer")).diagnostics
            if d.code == "W101"
        ]
        assert any("softmax ignores" in line for line in w101.detail)
        norm = [patch.dense("blocks.0.ln1.weight", (32,), "x", torch.ones(32))]
        bad = static_check(ir, EditPlan(norm, op="project_out"))
        assert {"E103", "E106"} <= {d.code for d in bad.diagnostics}
        reader = static_check(
            ir, EditPlan(get("project_out")(ctx, torch.randn(32), "L0.attn.k"), op="project_out")
        )
        assert "W102" in {d.code for d in reader.diagnostics}
        assert reader.ok

    def test_fused_boundary_and_scope(self) -> None:
        _, ir, _, ctx = make()
        plan = EditPlan(
            get("scale")(ctx, "L0.attn.qkv[30:40]", factor=2.0), op="scale", scope=["nothing"]
        )
        codes = {d.code for d in static_check(ir, plan).diagnostics}
        assert {"W104", "E104"} <= codes

    def test_nan_and_big_change(self) -> None:
        _, ir, tree, _ = make()
        name = "blocks.0.w_out.weight"
        shape = ir.params[name].shape
        nan = numeric_check(
            ir,
            tree,
            EditPlan([patch.dense(name, shape, "", torch.full(shape, float("nan")))]),
            self.cfg,
        )
        assert "E107" in {d.code for d in nan.diagnostics}
        big = numeric_check(
            ir, tree, EditPlan([patch.dense(name, shape, "", torch.ones(shape))]), self.cfg
        )
        assert {"W201", "W202"} <= {d.code for d in big.diagnostics}

    def test_smoke_metrics(self) -> None:
        logits = torch.randn(2, 5, 10)
        assert token_kl(logits, logits) == pytest.approx(0.0, abs=1e-6)
        assert token_kl(logits, torch.randn(2, 5, 10)) > 0
        ids = torch.randint(0, 10, (2, 5))
        assert perplexity(logits, ids) > 1
        assert degenerate_reasons([]) == ["empty output"]
        assert degenerate_reasons([4] * 10)
        assert not degenerate_reasons(list(range(12)))
        report = smoke_report(
            SmokeMetrics(kl=0.5, ppl_base=10, ppl_new=20, degenerate=["empty output"]), self.cfg
        )
        assert {d.code for d in report.diagnostics} == {"W301", "W302", "W303"}


def test_transplant_from_base_restores_exactly(tmp_path: Path) -> None:
    from tests.lab.test_session import tiny_llama
    from wspx.lab.sdk import open_lab
    from wspx.lab.tensors import tensor_hash

    lab = open_lab(tiny_llama(), store=tmp_path / "s", workdir=tmp_path)
    base = tensor_hash(lab.model.get_parameter("model.layers.1.mlp.down_proj.weight"))
    lab.edit.scale("L1.mlp.down", factor=0.5)
    lab.commit("halve L1 down")
    lab.edit.transplant("L1.mlp.down")  # source defaults to the base model
    assert tensor_hash(lab.model.get_parameter("model.layers.1.mlp.down_proj.weight")) == base


def test_bake_steer_takes_a_block_name(tmp_path: Path) -> None:
    from tests.lab.test_session import tiny_llama
    from wspx.lab.sdk import open_lab

    lab = open_lab(tiny_llama(), store=tmp_path / "s", workdir=tmp_path)
    d = lab.contrast([[1, 2, 3]], [[4, 5, 6]], name="d")
    assert lab.edit.bake_steer(d, scale=0.5, at="L0").record.deltas
