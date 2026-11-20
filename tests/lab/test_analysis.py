from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

from wspx.errors import GrepError  # noqa: E402
from wspx.lab import analysis, results  # noqa: E402
from wspx.lab.grep import Environment, grep, parse, ranking_key  # noqa: E402
from wspx.lab.index import ModelIndex, build_index  # noqa: E402
from wspx.lab.ir.capture import build_ir  # noqa: E402
from wspx.lab.ir.model_ir import ModelIR  # noqa: E402
from wspx.lab.probe import Probe  # noqa: E402
from wspx.lab.text import Text  # noqa: E402


@pytest.fixture(scope="module")
def llama() -> tuple[torch.nn.Module, ModelIR]:
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=96,
        num_key_value_heads=2,
        head_dim=16,
    )
    model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    return model, build_ir(model)


def env_for(model: torch.nn.Module, ir: ModelIR, **kw: object) -> Environment:
    return Environment(ir, dict(model.named_parameters()), **kw)  # type: ignore[arg-type]


class TestParser:
    def test_precedence(self) -> None:
        tree = parse('writes(" Wait") > 0.3 and not nan or absmax > 1e3')
        assert type(tree).__name__ == "Binary"
        assert tree.op == "or"  # type: ignore[union-attr]

    def test_ranking_key(self) -> None:
        key = ranking_key(parse("colnorm > 4σ"))
        assert getattr(key, "name", None) == "colnorm"

    @pytest.mark.parametrize("text", ["", "norm >", "foo > 1", "writes()", "norm > > 1", "(norm"])
    def test_errors(self, text: str) -> None:
        with pytest.raises(GrepError):
            parse(text)


class TestGrep:
    def test_writes_with_token_ids_matches_direct_computation(self, llama: tuple) -> None:  # type: ignore[type-arg]
        model, ir = llama
        res = grep(env_for(model, ir), "writes(5)", ir.resolve("L1.mlp.down"))
        w = model.model.layers[1].mlp.down_proj.weight.detach().float()
        g = model.model.norm.weight.detach().float()
        u = model.lm_head.weight.detach().float()[5]
        cos = ((w * g[:, None]).T @ u) / ((w * g[:, None]).norm(dim=0) * u.norm())
        best = int(cos.argmax())
        assert res.hits[0].address == f"L1.mlp.down[:, {best}]"
        assert res.hits[0].score == pytest.approx(float(cos.max()), rel=1e-5)
        assert res.candidates == 96
        assert res.axis == "col"

    def test_sigma_threshold(self, llama: tuple) -> None:  # type: ignore[type-arg]
        model, ir = llama
        res = grep(env_for(model, ir), "colnorm > 2σ", ir.resolve("L*.mlp.down"))
        norms = torch.cat(
            [model.model.layers[i].mlp.down_proj.weight.detach().norm(dim=0) for i in range(3)]
        )
        assert len(res.hits) == int((norms > norms.mean() + 2 * norms.std(correction=0)).sum())

    def test_reads_and_axis_and_tensor_scope(self, llama: tuple) -> None:  # type: ignore[type-arg]
        model, ir = llama
        env = env_for(model, ir)
        assert grep(env, "reads(3)", ir.resolve("L0.attn.k")).axis == "row"
        health = grep(env, "nan or absmax > 1e3", ir.resolve("L*.attn.o"), axis="tensor")
        assert (health.candidates, len(health.hits)) == (3, 0)

    def test_unit_lengths_must_match(self, llama: tuple) -> None:  # type: ignore[type-arg]
        model, ir = llama
        with pytest.raises(GrepError, match="residual"):
            grep(env_for(model, ir), "writes(5)", ir.resolve("L0.mlp.down"), axis="row")

    def test_result_set_roundtrip(self, llama: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
        model, ir = llama
        res = grep(env_for(model, ir), "colnorm", ir.resolve("L*.mlp.down"))
        results.save(res, tmp_path / "r1.parquet", top=5)
        hits = results.load_hits(tmp_path / "r1.parquet")
        assert [h.address for h in hits] == [h.address for h in res.hits[:5]]
        assert len(results.load_hits(tmp_path / "r1.parquet", only_hits=False)) == 288
        view = results.views(res, {p.name: None for p in ir.params.values()})
        assert sum(view["hist"]["counts"]) == 288
        param = hits[0].param
        units = results.indices(hits, param)
        resolved = ir.resolve(
            f"{ir.primary_role(param).address}[:, @r1]",  # type: ignore[union-attr]
            handle_indices=lambda _h, _p: units,
        )
        assert resolved[0].index[1] == units


class TestIndexAndShow:
    def test_index_roundtrip(self, llama: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
        model, ir = llama
        index = build_index(ir, dict(model.named_parameters()))
        index.save(tmp_path)
        again = ModelIndex.load(tmp_path)
        name = "model.layers.0.mlp.down_proj.weight"
        assert again.tensors[name].shape == [64, 96]
        assert len(again.tensors[name].spectrum) == 16
        assert len(again.unit_tokens(name, 7)) == 10

    def test_show(self, llama: tuple) -> None:  # type: ignore[type-arg]
        model, ir = llama
        info = analysis.show(
            ir,
            dict(model.named_parameters()),
            ir.resolve("L1.mlp.down[:, 3]")[0],
            text=Text(None),
            values=4000,
        )
        assert len(info["values"]) == 64  # the column has 64 values; the cap is 2048
        assert info["vocab"]
        assert "norm" in analysis.render_show(info)


class TestProbesAndTraces:
    def test_points_match_block_outputs(self, llama: tuple) -> None:  # type: ignore[type-arg]
        model, ir = llama
        probe = Probe(model, ir)
        ids = torch.tensor([[1, 2, 3, 4]])
        seen: dict[int, torch.Tensor] = {}
        hooks = [
            layer.register_forward_hook(
                lambda _m, _a, out, i=i: seen.__setitem__(
                    i, out[0] if isinstance(out, tuple) else out
                )
            )
            for i, layer in enumerate(model.model.layers)
        ]
        with torch.no_grad(), probe.capture() as cap:
            model(input_ids=ids, use_cache=False)
        for h in hooks:
            h.remove()
        for i in range(3):
            assert torch.equal(cap.points[f"resid.L{i}"], seen[i])

    def test_contrast_finds_planted_direction(self, llama: tuple) -> None:  # type: ignore[type-arg]
        model, ir = llama
        probe, text = Probe(model, ir), Text(None)
        pos = [[1, 9, 2, i] for i in range(10, 20)]
        neg = [[1, 9, 2, i] for i in range(60, 70)]
        d = analysis.contrast(model, ir, probe, text, pos, neg, name="x", layer=1)
        assert d.provenance == "contrast.L1-resid.n20"
        assert d.separation > 1.0

    def test_traces(self, llama: tuple) -> None:  # type: ignore[type-arg]
        model, ir = llama
        probe, text = Probe(model, ir), Text(None)
        fwd = analysis.forward_trace(model, ir, probe, text, [1, 2, 3, 4, 5])
        assert len(fwd.arrays["resid_norm"]) == 3
        assert "logit lens" in fwd.summary

        @contextlib.contextmanager
        def zero_block_1() -> Iterator[None]:
            w = model.model.layers[1].mlp.down_proj.weight
            saved = w.detach().clone()
            with torch.no_grad():
                w.zero_()
            try:
                yield
            finally:
                with torch.no_grad():
                    w.copy_(saved)

        diff = analysis.diff_trace(model, ir, probe, text, [1, 2, 3, 4, 5], switch=zero_block_1())
        assert diff.data["first_divergence"] == 1
        attr = analysis.attribution_trace(
            model, ir, probe, text, [1, 2, 3, 4, 5], [1, 2, 3, 4, 6], top=4
        )
        assert len(attr.data["ranking"]) >= 4
        assert all(p.requires_grad for p in model.parameters())
