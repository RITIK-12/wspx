"""The lab's transformers copy in host memory, run a layer at a time on the GPU.

With vLLM serving, the lab's own copy is for edits and probes; generation and evals go to
vLLM. When one GPU cannot hold both (Inkling-2-Mini: ~53 GB in bf16 beside vLLM's ~20 GB
and its KV cache), the copy stays in pinned host memory: what lies outside the layer stacks
(embeddings, norms, the head) lives on the GPU, and each layer's weights are copied in for
its forward pass and dropped after. Edits write the host tensors, which stay the lab's
weights; a forward pass costs one trip of the stacks over PCIe.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

STACK_SHARE = 0.05
"""A ``ModuleList`` is a layer stack when it holds at least this share of the weights."""


def stacks(model: nn.Module) -> list[tuple[str, nn.ModuleList]]:
    """The model's layer stacks (decoder layers, vision blocks), outermost only."""
    total = sum(p.numel() for p in model.parameters()) or 1
    out: list[tuple[str, nn.ModuleList]] = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.ModuleList) or len(m) < 2:
            continue
        if any(name.startswith(f"{o}.") for o, _ in out):
            continue
        if sum(p.numel() for p in m.parameters()) >= STACK_SHARE * total:
            out.append((name, m))
    return out


def sizes(model: nn.Module, dtype: torch.dtype | None = None) -> tuple[int, int, int]:
    """Bytes of the whole model, of what lies outside its stacks, and of its largest block
    (``dtype``: as it would be held; works on the meta device)."""

    def nbytes(params: Iterator[torch.Tensor]) -> int:
        return sum(p.numel() * (dtype.itemsize if dtype else p.element_size()) for p in params)

    total = nbytes(model.parameters())
    blocks = [b for _, s in stacks(model) for b in s]
    inside = sum(nbytes(b.parameters()) for b in blocks)
    largest = max((nbytes(b.parameters()) for b in blocks), default=0)
    return total, total - inside, largest


@dataclass
class Streamed:
    """A model whose layer blocks live on the host and run on ``device``."""

    model: nn.Module
    device: torch.device
    blocks: list[nn.Module]
    handles: list[Any] = field(default_factory=list)

    @property
    def resident_bytes(self) -> int:
        """What stays on the GPU between forward passes."""
        return sum(
            p.numel() * p.element_size()
            for p in self.model.parameters()
            if p.device.type == self.device.type
        )

    def describe(self) -> str:
        host = sum(p.numel() * p.element_size() for b in self.blocks for p in b.parameters())
        return (
            f"transformers streams {len(self.blocks)} layers ({host / 2**30:.1f} GB) from host "
            f"memory; {self.resident_bytes / 2**30:.1f} GB stays on {self.device}"
        )


def _params(block: nn.Module) -> list[nn.Parameter]:
    return list(block.parameters())


def stream(model: nn.Module, device: torch.device | str, *, pin: bool = True) -> Streamed:
    """Keep ``model``'s layer blocks in host memory and run each on ``device``."""
    device = torch.device(device)
    blocks = [b for _, s in stacks(model) for b in s]
    in_blocks = {id(p) for b in blocks for p in b.parameters()}
    with torch.no_grad():
        for module in model.modules():
            for p in module._parameters.values():
                if p is not None and id(p) not in in_blocks and p.device != device:
                    p.data = p.data.to(device)
            for key, b in module._buffers.items():  # buffers are small: all on the GPU
                if b is not None and b.device != device:
                    module._buffers[key] = b.to(device)
        for block in blocks:
            for p in _params(block):
                host = p.data if p.device.type == "cpu" else p.data.cpu()
                p.data = host.pin_memory() if pin and torch.cuda.is_available() else host
    out = Streamed(model, device, blocks)

    def bring(block: nn.Module, *_: Any, **__: Any) -> None:
        for p in _params(block):
            if p.device.type == "cpu":
                p._wspx_host = p.data  # type: ignore[attr-defined]
                p.data = p.data.to(device, non_blocking=True)

    def drop(block: nn.Module, *_: Any, **__: Any) -> None:
        for p in _params(block):
            host = p.__dict__.pop("_wspx_host", None)
            if host is not None:
                p.data = host

    for block in blocks:
        out.handles.append(block.register_forward_pre_hook(bring, with_kwargs=True))
        out.handles.append(block.register_forward_hook(drop, always_call=True))
    model._wspx_streamed = out  # type: ignore[assignment]
    return out


def unstream(model: nn.Module) -> bool:
    """Move a streamed model's layers onto its GPU, when they fit there; returns whether."""
    streamed: Streamed | None = getattr(model, "_wspx_streamed", None)
    if streamed is None:
        return True
    host = sum(p.numel() * p.element_size() for b in streamed.blocks for p in b.parameters())
    free, _ = torch.cuda.mem_get_info(streamed.device)
    if host * 1.1 + 2 * 2**30 > free:
        return False
    for handle in streamed.handles:
        handle.remove()
    with torch.no_grad():
        for block in streamed.blocks:
            for p in _params(block):
                p.data = p.data.to(streamed.device)
    del model._wspx_streamed
    return True


def device_of(model: nn.Module) -> torch.device:
    """Where a model's inputs go: its GPU when it streams, else its first weight's device."""
    streamed = getattr(model, "_wspx_streamed", None)
    if streamed is not None:
        return torch.device(streamed.device)
    return next(model.parameters()).device
