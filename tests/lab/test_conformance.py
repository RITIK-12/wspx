"""The compile step is exact (§8, S1), checked on tiny random-init models built from real configs.

These architectures are regression fixtures, not a list of supported models: any
PyTorch model compiles the same way. For each one, every tensor must be exercised and
get exactly one structural key; ``L{i}`` must be the layer stack's i-th child; the
layer count must match the config; and every short name kept from the traced graph
must land inside the layer it names. No downloads are needed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

transformers = pytest.importorskip("transformers")

from wspx.lab.ir.capture import build_ir  # noqa: E402

pytestmark = pytest.mark.slow

C = {"vocab_size": 128, "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4}


def lm(config: Callable[[], Any]) -> Callable[[], Any]:
    return lambda: transformers.AutoModelForCausalLM.from_config(config())


T = transformers
LLMS: dict[str, tuple[Callable[[], Any], list[str]]] = {
    "llama": (
        lm(lambda: T.LlamaConfig(**C, intermediate_size=96, num_key_value_heads=2, head_dim=16)),
        [
            "L0.attn.q",
            "L0.attn.k",
            "L0.attn.v",
            "L0.attn.o",
            "L1.mlp.gate",
            "L1.mlp.up",
            "L1.mlp.down",
        ],
    ),
    "qwen3": (
        lm(lambda: T.Qwen3Config(**C, intermediate_size=96, num_key_value_heads=2, head_dim=16)),
        ["L0.attn.q_norm", "L0.attn.k_norm", "L1.mlp.down"],
    ),
    "qwen3_moe": (
        lm(
            lambda: T.Qwen3MoeConfig(
                **C,
                intermediate_size=96,
                moe_intermediate_size=32,
                num_experts=64,
                num_experts_per_tok=2,
                num_key_value_heads=2,
                head_dim=16,
            )
        ),
        ["L0.moe.router", "L0.moe.experts.gate", "L0.moe.experts.up", "L1.moe.experts.down"],
    ),
    "qwen3_5": (
        lm(
            lambda: T.Qwen3_5TextConfig(
                **C,
                intermediate_size=96,
                num_key_value_heads=2,
                head_dim=16,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                layer_types=["linear_attention", "full_attention"],
            )
        ),
        ["L0.ssm.out", "L1.attn.q", "L1.mlp.down"],
    ),
    "gemma3": (
        lm(
            lambda: T.Gemma3TextConfig(
                **C, intermediate_size=96, num_key_value_heads=2, head_dim=16, sliding_window=8
            )
        ),
        ["L0.attn.post_norm", "L0.mlp.post_norm", "L1.attn.q_norm"],
    ),
    "gemma4": (
        lm(
            lambda: T.Gemma4TextConfig(
                **C, intermediate_size=96, num_key_value_heads=2, head_dim=16
            )
        ),
        ["L0.attn.q", "L0.attn.o", "L0.attn.post_norm", "L1.mlp.gate", "L1.mlp.down"],
    ),
    "mistral": (
        lm(lambda: T.MistralConfig(**C, intermediate_size=96, num_key_value_heads=2, head_dim=16)),
        ["L1.mlp.gate"],
    ),
    "mixtral": (
        lm(
            lambda: T.MixtralConfig(
                **C,
                intermediate_size=96,
                num_local_experts=4,
                num_experts_per_tok=2,
                num_key_value_heads=2,
                head_dim=16,
            )
        ),
        ["L0.moe.router", "L1.moe.experts.down"],
    ),
    "phi3": (
        lm(lambda: T.Phi3Config(**C, intermediate_size=96, num_key_value_heads=4, pad_token_id=0)),
        ["L0.attn.qkv", "L0.attn.q", "L0.mlp.gate_up", "L0.mlp.gate", "L0.mlp.up"],
    ),
    "olmo2": (
        lm(lambda: T.Olmo2Config(**C, intermediate_size=96, num_key_value_heads=2)),
        ["L0.attn.q_norm", "L1.mlp.down"],
    ),
    "gpt_oss": (
        lm(
            lambda: T.GptOssConfig(
                **C,
                intermediate_size=32,
                num_local_experts=4,
                num_experts_per_tok=2,
                num_key_value_heads=2,
                head_dim=16,
                sliding_window=8,
            )
        ),
        ["L0.attn.sinks", "L0.moe.router.bias", "L0.moe.experts.gate", "L0.moe.experts.down.bias"],
    ),
    "deepseek_v3": (
        lm(
            lambda: T.DeepseekV3Config(
                **C,
                intermediate_size=96,
                moe_intermediate_size=32,
                n_routed_experts=8,
                n_shared_experts=1,
                num_experts_per_tok=2,
                n_group=2,
                topk_group=1,
                first_k_dense_replace=1,
                num_key_value_heads=4,
                q_lora_rank=32,
                kv_lora_rank=16,
                qk_rope_head_dim=8,
                qk_nope_head_dim=8,
                v_head_dim=16,
            )
        ),
        ["L0.attn.q", "L0.attn.kv", "L0.mlp.down", "L1.moe.router", "L1.moe.shared.down"],
    ),
    "gpt2": (
        lm(lambda: T.GPT2Config(vocab_size=128, n_embd=64, n_layer=2, n_head=4, n_positions=64)),
        ["pos_embed", "L0.attn.qkv", "L0.attn.q.bias", "L1.mlp.up", "L1.mlp.down"],
    ),
    "falcon": (
        lm(
            lambda: T.FalconConfig(
                vocab_size=128,
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                new_decoder_architecture=False,
                multi_query=True,
                parallel_attn=True,
            )
        ),
        ["L0.attn.qkv", "L0.attn.o", "L0.mlp.up", "L0.mlp.down"],
    ),
    "jamba": (
        lm(
            lambda: T.JambaConfig(
                **C,
                intermediate_size=96,
                num_key_value_heads=2,
                num_experts=4,
                num_experts_per_tok=2,
                expert_layer_period=2,
                expert_layer_offset=1,
                attn_layer_period=2,
                attn_layer_offset=1,
                mamba_d_state=8,
                mamba_d_conv=4,
                mamba_expand=2,
                use_mamba_kernels=False,
            )
        ),
        ["L0.ssm.out", "L1.attn.o", "L1.moe.router"],
    ),
    "bamba": (
        lm(
            lambda: T.BambaConfig(
                **C,
                intermediate_size=96,
                num_key_value_heads=2,
                attn_layer_indices=[1],
                mamba_n_heads=4,
                mamba_d_head=32,
                mamba_d_state=8,
                mamba_n_groups=1,
                mamba_chunk_size=8,
            )
        ),
        ["L0.ssm.out", "L1.attn.o"],
    ),
    "mamba": (
        lm(
            lambda: T.MambaConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2, state_size=4)
        ),
        ["L0.ssm.in", "L0.ssm.out", "L1.ssm.out"],
    ),
}

VISION: dict[str, tuple[Callable[[], Any], list[str]]] = {
    "vit": (
        lambda: T.ViTForImageClassification(
            T.ViTConfig(
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
                image_size=16,
                patch_size=8,
                num_labels=5,
            )
        ),
        ["L0.attn.q", "L1.mlp.down", "unembed"],
    ),
    "resnet": (
        lambda: T.ResNetForImageClassification(
            T.ResNetConfig(
                embedding_size=8,
                hidden_sizes=[8, 16],
                depths=[1, 1],
                layer_type="basic",
                num_labels=5,
            )
        ),
        ["L0.conv.conv1", "L0.conv.conv2", "L1.shortcut.proj", "unembed"],
    ),
}


def _check(model: Any, expected: list[str], *, lm: bool) -> None:
    torch.manual_seed(0)
    ir = build_ir(model.eval())
    assert ir.unexercised == [], ir.render_info()
    stack = ir.structure["stack"]
    assert stack is not None, ir.render_info()

    # One structural key per tensor, all distinct, each pointing where it says.
    keyed = [r for r in ir.roles if r.address == ir.key_of(r.param)]
    assert sorted(r.param for r in keyed) == sorted(ir.params)
    assert len({r.address for r in keyed}) == len(keyed)
    prefix = f"{stack['path']}." if stack["path"] else ""
    for role in keyed:
        if role.block is not None:
            child = stack["children"][role.block]
            assert role.param.startswith(f"{prefix}{child}."), (role.address, role.param)
            assert role.address == f"L{role.block}" or role.address.startswith(f"L{role.block}.")
    # Short names from the traced graph survive only inside their own layer.
    layer_of = ir.structure["layer_of"]
    for role in ir.roles:
        if role.block is not None and role.address != ir.key_of(role.param):
            assert layer_of.get(role.param) == role.block, (role.address, role.param)

    addresses = {r.address for r in ir.roles}
    missing = [a for a in expected if a not in addresses]
    assert not missing, (missing, sorted(addresses))
    for address in expected:
        assert ir.resolve(address)
    if lm:
        text = model.config.get_text_config()
        assert stack["size"] == text.num_hidden_layers
        assert stack["runs"] == [1] * stack["size"]
        assert set(ir.graphs) == {"prefill", "decode"}
        assert ir.num_blocks == stack["size"]
        assert ir.semantics.embed is not None
        assert ir.semantics.unembed is not None
        assert ir.resolve("embed")
        assert ir.resolve("unembed")


@pytest.mark.parametrize("arch", sorted(LLMS))
def test_language_models(arch: str) -> None:
    factory, expected = LLMS[arch]
    _check(factory(), expected, lm=True)


@pytest.mark.parametrize("arch", sorted(VISION))
def test_vision_models(arch: str) -> None:
    factory, expected = VISION[arch]
    _check(factory(), expected, lm=False)


def test_all_experts_exercised_with_more_experts_than_slots() -> None:
    """128 experts, 32 tokens × top-2 = 64 slots per pass: forced routing must still cover all."""
    model = T.AutoModelForCausalLM.from_config(
        T.Qwen3MoeConfig(
            **{**C, "num_hidden_layers": 1},
            intermediate_size=96,
            moe_intermediate_size=16,
            num_experts=128,
            num_experts_per_tok=2,
            num_key_value_heads=2,
            head_dim=16,
        )
    )
    ir = build_ir(model.eval())
    experts = ir.coverage["model.layers.0.mlp.experts.gate_up_proj"]
    assert (experts.rows_seen, experts.rows_total) == (128, 128)
    assert ir.capture["passes"] >= 3


def _tiny_inkling() -> Any:
    """Inkling-2-Mini's layout at toy width: 24 layers, 6-17 looped three times, 8 experts."""
    local = [i for i in range(24) if i % 6 != 5]
    text = {
        "model_type": "inkling2_loop_text",
        "vocab_size": 256,
        "unpadded_vocab_size": 250,
        "hidden_size": 64,
        "num_hidden_layers": 24,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 24,  # 4 heads × 24 ≠ width 64, as in the real model (24 × 128 vs 2304)
        "swa_head_dim": 24,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "d_rel": 4,
        "local_layer_ids": local,
        "n_layers_per_attn_res_block": 6,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "intermediate_size": 32,
        "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 8},
    }
    vision = {
        "model_type": "inkling2_vision",
        "text_hidden_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "projection_hidden_dim": 64,
        "posemb_grid_size": 4,
    }
    config = T.AutoConfig.for_model("inkling2_loop", text_config=text, vision_config=vision)
    torch.manual_seed(0)
    return T.AutoModelForImageTextToText.from_config(config).eval()


@pytest.fixture(scope="module")
def inkling() -> Any:
    return _tiny_inkling()


def test_a_looped_transformer_compiles_to_its_physical_layers(inkling: Any) -> None:
    """Inkling-2-Mini (seen live): 24 layers whose middle 12 run three times, attention
    residuals every 6 layers and a short convolution inside each attention. The traced
    graph read 96 blocks; the module tree and the call log give 24 layers and the loop."""
    model = inkling
    ir = build_ir(model)
    stack = ir.structure["stack"]
    assert (stack["path"], stack["size"]) == ("model.language_model.layers", 24)
    assert stack["runs"] == [1] * 6 + [3] * 12 + [1] * 6
    assert stack["segments"] == [[0, 5, 1], [6, 17, 3], [18, 23, 1]]
    assert stack["calls"] == 48
    # layer 0, sliding, full attention, looped sliding, looped full
    assert [v["layers"][0] for v in stack["variants"]] == [0, 1, 5, 6, 11]
    assert ir.key_of("model.language_model.layers.9.self_attn.o_proj.linear.weight") == (
        "L9.self_attn.o_proj.linear"
    )
    experts = ir.primary_role("model.language_model.layers.9.mlp.experts.down_proj")
    assert experts is not None
    assert (experts.kind, experts.experts) == ("expert", 8)
    router = ir.primary_role("model.language_model.layers.9.mlp.gate.weight")
    assert router is not None
    assert router.kind == "router"
    assert ir.structure["final_norm"] == "model.language_model.norm"
    assert ir.semantics.unembed == "lm_head.weight"
    # Seen live: the loop gates' projections, model-width but fed to tanh, were "writers"
    # and topped every writer ranking. The op graph shows where their output goes.
    layer = "model.language_model.layers.9"
    kind = {n: r.kind for n in ir.params if (r := ir.primary_role(n)) is not None}
    assert kind[f"{layer}.q_message_gate.forget_gate_proj.weight"] == "gate"
    assert kind[f"{layer}.mlp_message_gate.message_gate_proj.weight"] == "gate"
    assert kind[f"{layer}.self_attn.o_proj.linear.weight"] == "writer"
    assert kind[f"{layer}.self_attn.o_proj.gate_proj.weight"] == "reader"
    assert "gate = model-width output that only passes through sigmoid" in ir.render_info()
    info = ir.render_info()
    assert "L6-17 run 3 times with the same weights" in info
    assert "24 layers, L0-L23" in info
    # The traced graph's short names named the wrong layers here, so none survive.
    assert all(r.address == ir.key_of(r.param) or r.block is None for r in ir.roles)
