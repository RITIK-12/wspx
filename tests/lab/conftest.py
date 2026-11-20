from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402


class TinyBlock(nn.Module):
    """A pre-norm transformer block with fused qkv and a gated MLP, written by hand."""

    def __init__(self, d: int = 32, heads: int = 4, ff: int = 48) -> None:
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.w_in = nn.Linear(d, 2 * ff, bias=False)
        self.w_out = nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q, k, v = (z.view(b, t, self.heads, -1).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        up, gate = self.w_in(self.ln2(x)).chunk(2, dim=-1)
        return x + self.w_out(F.silu(gate) * up)


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 40, d: int = 32, layers: int = 2) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(TinyBlock(d) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.unused = nn.Parameter(torch.zeros(3))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.tok(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


@pytest.fixture(scope="session")
def tiny_lm() -> TinyLM:
    torch.manual_seed(0)
    return TinyLM().eval()


@pytest.fixture(scope="session")
def tiny_ir(tiny_lm: TinyLM):  # type: ignore[no-untyped-def]
    from wspx.lab.ir.capture import build_ir

    return build_ir(tiny_lm, example_inputs=(torch.randint(0, 40, (2, 10)),))
