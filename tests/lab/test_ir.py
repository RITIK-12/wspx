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
        assert tiny_ir.layer_types() == [("0-1", "attn+mlp")]
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
        assert tiny_ir.unexercised == ["unused"]
        assert "unused" in tiny_ir.semantics.unassigned
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
        assert r["L0.linear"] == "0.weight"
        assert r["L0.norm"] == "1.weight"
        assert r["L1.linear"] == "3.weight"
        assert r["unembed"] == "3.weight"
        assert not ir.unexercised

    def test_cnn_out_channel_selector(self) -> None:
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 4, 3))
        ir = build_ir(model, example_inputs=(torch.randn(2, 3, 12, 12),))
        (t,) = ir.resolve("L0.conv[oc=5]")
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
