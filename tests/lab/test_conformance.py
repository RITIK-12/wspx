"""Conformance (§8, S1): tiny random-init models built from real configs.

Each must bind and exercise 100% of its parameters, derive a role for every
one of them, and expose the expected addresses. No downloads are needed.
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
    assert ir.semantics.unassigned == [], ir.semantics.unassigned
    addresses = {r.address for r in ir.roles}
    missing = [a for a in expected if a not in addresses]
    assert not missing, (missing, sorted(addresses))
    for address in expected:
        assert ir.resolve(address)
    if lm:
        assert set(ir.graphs) == {"prefill", "decode"}
        assert ir.num_blocks == 2
        assert ir.semantics.embed is not None
        assert ir.semantics.unembed is not None


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
