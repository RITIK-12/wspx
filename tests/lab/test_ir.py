from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from wspx.errors import AddressError, ModelLoadError
from wspx.lab.ir.capture import build_ir
from wspx.lab.ir.graph import compress_indices, expand_runs
from wspx.lab.ir.model_ir import ModelIR


def roles(ir: ModelIR) -> dict[str, str]:
    return {r.address: r.param for r in ir.roles}


class TestRunEncoding:
    @pytest.mark.parametrize(
        "values",
        [[], [3], list(range(64)), list(range(0, 63, 2)), [0, 1, 2, 10, 11, 12], [1, 5, 6, 7, 30]],
    )
    def test_roundtrip(self, values: list[int]) -> None:
        assert expand_runs(compress_indices(values)) == values

    def test_compact(self) -> None:
        assert compress_indices(list(range(8, 24))) == [[8, 24, 1]]
        assert compress_indices(list(range(1, 64, 2))) == [[1, 64, 2]]


class TestTinyTransformer:
    def test_blocks_and_stream(self, tiny_ir: ModelIR) -> None:
        assert tiny_ir.num_blocks == 2
        assert tiny_ir.layer_types() == [("L0-1", "TinyBlock")]
        stack = tiny_ir.structure["stack"]
        assert (stack["path"], stack["size"], stack["runs"]) == ("blocks", 2, [1, 1])
        assert tiny_ir.semantics.embed == "tok.weight"
        assert tiny_ir.semantics.unembed == "tok.weight"
        assert tiny_ir.semantics.tied
        assert tiny_ir.semantics.final_norm == ["norm.weight", "norm.bias"]

    def test_fused_qkv_split_from_data_flow(self, tiny_ir: ModelIR) -> None:
        by_addr = {r.address: r for r in tiny_ir.roles}
        assert by_addr["L0.attn.qkv"].param == "blocks.0.qkv.weight"
        assert by_addr["L0.attn.q"].split == [[0, 32, 1]]
        assert by_addr["L0.attn.k"].split == [[32, 64, 1]]
        assert by_addr["L0.attn.v"].split == [[64, 96, 1]]
        assert by_addr["L0.attn.q.bias"].split == [[0, 32, 1]]
        assert by_addr["L0.attn.o"].param == "blocks.0.proj.weight"

    def test_gate_is_what_passes_through_the_activation(self, tiny_ir: ModelIR) -> None:
        # chunk(2) puts "up" first and "gate" second here: names don't matter, flow does.
        by_addr = {r.address: r for r in tiny_ir.roles}
        assert by_addr["L1.mlp.gate_up"].param == "blocks.1.w_in.weight"
        assert by_addr["L1.mlp.gate"].split == [[48, 96, 1]]
        assert by_addr["L1.mlp.up"].split == [[0, 48, 1]]
        assert by_addr["L1.mlp.down"].param == "blocks.1.w_out.weight"

    def test_norms(self, tiny_ir: ModelIR) -> None:
        r = roles(tiny_ir)
        assert r["L0.attn.norm"] == "blocks.0.ln1.weight"
        assert r["L0.attn.norm.bias"] == "blocks.0.ln1.bias"
        assert r["L0.mlp.norm"] == "blocks.0.ln2.weight"

    def test_layout_of_linear_weights(self, tiny_ir: ModelIR) -> None:
        down = next(r for r in tiny_ir.roles if r.address == "L0.mlp.down")
        assert (down.in_dim, down.out_dim) == (1, 0)

    def test_unused_parameter_is_unexercised_not_guessed(self, tiny_ir: ModelIR) -> None:
        # It exists, so it has a key; nothing read it, so it has no axes and no layer.
        assert tiny_ir.unexercised == ["unused"]
        role = tiny_ir.primary_role("unused")
        assert role is not None
        assert (role.address, role.kind, role.block, role.in_dim) == (
            "unused",
            "vector",
            None,
            None,
        )
        assert "UNEXERCISED" in tiny_ir.render_info()

    def test_tied_alias(self, tiny_ir: ModelIR) -> None:
        assert tiny_ir.params["tok.weight"].aliases == ["head.weight"]
        assert tiny_ir.canonical("head.weight") == "tok.weight"


class TestResolution:
    def test_short_and_slice(self, tiny_ir: ModelIR) -> None:
        (t,) = tiny_ir.resolve("L1.mlp.down[:, 7]")
        assert t.param == "blocks.1.w_out.weight"
        assert t.index == (slice(None), 7)

    def test_split_role_slices_within_the_split(self, tiny_ir: ModelIR) -> None:
        (t,) = tiny_ir.resolve("L0.attn.k[2:4]")
        assert t.index == (slice(34, 36), slice(None))

    def test_heads(self, tiny_ir: ModelIR) -> None:
        (t,) = tiny_ir.resolve("L0.attn.v.h1")
        assert t.index[0] == list(range(72, 80))
        with pytest.raises(AddressError, match="out of range"):
            tiny_ir.resolve("L0.attn.v.h9")

    def test_globs(self, tiny_ir: ModelIR) -> None:
        params = [t.param for t in tiny_ir.resolve("L*.mlp.down")]
        assert params == ["blocks.0.w_out.weight", "blocks.1.w_out.weight"]
        # a glob that hits a fused weight and its sub-roles selects the weight once
        attn = [t.param for t in tiny_ir.resolve("L0.attn.*") if "bias" not in t.address]
        assert attn.count("blocks.0.qkv.weight") == 1

    def test_raw_and_alias(self, tiny_ir: ModelIR) -> None:
        (t,) = tiny_ir.resolve("head.weight[3]")
        assert (t.param, t.index) == ("tok.weight", (3, slice(None)))
        assert len(tiny_ir.resolve("blocks.*.qkv.weight")) == 2

    def test_tokens_need_a_tokenizer(self, tiny_ir: ModelIR) -> None:
        with pytest.raises(AddressError, match="tokenizer"):
            tiny_ir.resolve('embed[" Wait"]')
        (t,) = tiny_ir.resolve('embed[" Wait"]', token_id=lambda text: 11)
        assert t.index == (11, slice(None))

    def test_unknown_address_suggests(self, tiny_ir: ModelIR) -> None:
        with pytest.raises(AddressError, match=r"E101.*did you mean"):
            tiny_ir.resolve("L0.mlp.dwn")

    def test_too_many_indices(self, tiny_ir: ModelIR) -> None:
        with pytest.raises(AddressError, match="too many"):
            tiny_ir.resolve("L0.mlp.down[1, 2, 3]")

    def test_roundtrip_through_json(self, tiny_ir: ModelIR) -> None:
        data = json.loads(json.dumps(tiny_ir.to_dict()))
        restored = ModelIR.from_dict(data)
        assert restored.resolve("L0.attn.v.h1")[0].index == tiny_ir.resolve("L0.attn.v.h1")[0].index
        assert restored.graphs["forward"].nodes[0].op == tiny_ir.graphs["forward"].nodes[0].op
        assert restored.fingerprint == tiny_ir.fingerprint


class TestOtherClasses:
    def test_mlp(self) -> None:
        model = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.ReLU(), nn.Linear(16, 4))
        ir = build_ir(model, example_inputs=(torch.randn(2, 8),))
        r = roles(ir)
        # The sequential is the layer stack: its child index is the layer number.
        assert r["L0"] == "0.weight"
        assert r["L0.bias"] == "0.bias"
        assert r["L1"] == "1.weight"
        assert r["L3"] == "3.weight"
        assert r["unembed"] == "3.weight"
        assert not ir.unexercised

    def test_cnn_out_channel_selector(self) -> None:
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 4, 3))
        ir = build_ir(model, example_inputs=(torch.randn(2, 3, 12, 12),))
        (t,) = ir.resolve("L0[oc=5]")
        assert t.param == "0.weight"
        assert t.index[0] == 5

    @pytest.mark.parametrize(("kind", "gates"), [("LSTM", 4), ("GRU", 3)])
    def test_recurrent_gates(self, kind: str, gates: int) -> None:
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = getattr(nn, kind)(6, 10, num_layers=2, batch_first=True)
                self.out = nn.Linear(10, 3)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.out(self.rnn(x)[0])

        ir = build_ir(Net(), example_inputs=(torch.randn(2, 5, 6),))
        r = roles(ir)
        assert r[f"L1.{kind.lower()}.hh"] == "rnn.weight_hh_l1"
        gate = "forget" if kind == "LSTM" else "update"
        (t,) = ir.resolve(f"L0.{kind.lower()}.ih.{gate}")
        assert t.index[0] == list(range(10, 20))
        assert ir.params["rnn.weight_ih_l0"].shape[0] == gates * 10
        assert not ir.semantics.unassigned

    def test_generic_module_needs_inputs(self) -> None:
        with pytest.raises(ModelLoadError, match="example inputs"):
            build_ir(nn.Linear(2, 2))


class _LoopBlock(nn.Module):
    """A block whose first residual add in the trace is an inner one (``k + conv(k)``)."""

    def __init__(self, d: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.k = nn.Linear(d, 8)
        self.k_conv = nn.Conv1d(8, 8, 3, padding=2, groups=8)  # a depthwise short convolution
        self.proj = nn.Linear(8, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.k(self.norm(x))
        conv = self.k_conv(k.transpose(1, 2))[..., : k.shape[1]].transpose(1, 2)
        k = k + conv
        return x + self.proj(torch.tanh(k))


class _LoopNet(nn.Module):
    """Layers 2-3 run three times with the same weights, like a looped transformer."""

    def __init__(self, vocab: int = 30, d: int = 16) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(_LoopBlock(d) for _ in range(6))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.tok(ids)
        for layer in self.layers[:2]:
            x = layer(x)
        for _ in range(3):
            for layer in self.layers[2:4]:
                x = layer(x)
        for layer in self.layers[4:]:
            x = layer(x)
        return self.head(self.norm(x))


@pytest.fixture(scope="module")
def looped():  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    model = _LoopNet().eval()
    ids = torch.randint(0, 30, (2, 7))
    return model, ids, build_ir(model, example_inputs=(ids,))


class TestLoopedModel:
    """Seen live on Inkling-2-Mini: the traced graph took an inner short-convolution residual
    for the residual stream and numbered each loop pass as new layers (96 blocks for 24
    layers). The structure comes from the module tree and the call log instead."""

    def test_layers_and_passes(self, looped) -> None:  # type: ignore[no-untyped-def]
        _, _, ir = looped
        stack = ir.structure["stack"]
        assert (stack["path"], stack["size"], stack["calls"]) == ("layers", 6, 10)
        assert stack["runs"] == [1, 1, 3, 3, 1, 1]
        assert stack["segments"] == [[0, 1, 1], [2, 3, 3], [4, 5, 1]]
        assert stack["order"] == [[0, 1], [2, 3], [2, 3], [2, 3], [4, 5]]
        assert ir.num_blocks == 6
        info = ir.render_info()
        assert "run order: L0-1, L2-3, L2-3, L2-3, L4-5" in info
        assert "L2-3 run 3 times with the same weights" in info

    def test_keys_are_the_module_tree(self, looped) -> None:  # type: ignore[no-untyped-def]
        _, _, ir = looped
        assert ir.key_of("layers.3.proj.weight") == "L3.proj"
        assert ir.key_of("layers.3.proj.bias") == "L3.proj.bias"
        (t,) = ir.resolve("L3.proj")
        assert t.param == "layers.3.proj.weight"
        assert ir.runs_of("layers.3.proj.weight") == 3
        assert ir.runs_of("layers.4.proj.weight") == 1
        assert {t.param for t in ir.resolve("L3")} == {
            n for n in ir.params if n.startswith("layers.3.")
        }
        assert (ir.semantics.embed, ir.semantics.unembed) == ("tok.weight", "head.weight")
        assert ir.semantics.final_norm == ["norm.weight", "norm.bias"]
        for role in ir.roles:  # any short name kept lands in its own layer
            if role.block is not None:
                assert role.param.startswith(f"layers.{role.block}.")

    def test_each_pass_is_its_own_probe_point(self, looped) -> None:  # type: ignore[no-untyped-def]
        from wspx.lab.probe import Probe

        model, ids, ir = looped
        names = ["resid.embed", "resid.L2", "resid.L2@2", "resid.L2@3", "resid.L5"]
        with torch.no_grad(), Probe(model, ir).capture(names) as cap:
            model(ids)
        assert set(cap.points) == set(names)
        passes = [cap.points[n] for n in ("resid.L2", "resid.L2@2", "resid.L2@3")]
        assert all(p.shape == (2, 7, 16) for p in passes)
        assert not torch.equal(passes[0], passes[1])
        assert not torch.equal(passes[1], passes[2])
