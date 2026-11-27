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
        from wspx.lab.weights import is_placeholder

        size = sum(p.numel() * p.element_size() for b in self.blocks for p in b.parameters())
        where = (
            "from the checkpoint (vLLM holds the model)"
            if any(is_placeholder(p) for b in self.blocks for p in b.parameters())
            else "from host memory"
        )
        return (
            f"transformers runs {len(self.blocks)} layers ({size / 2**30:.1f} GB) a layer at a "
            f"time, {where}; {self.resident_bytes / 2**30:.1f} GB stays on {self.device}"
        )


def _params(block: nn.Module) -> list[nn.Parameter]:
    return list(block.parameters())


def stream(
    model: nn.Module,
    device: torch.device | str,
    *,
    pin: bool = True,
    weights: Any = None,
) -> Streamed:
    """Keep ``model``'s layer blocks off the GPU and run each on ``device``.

    Without ``weights`` the blocks' tensors stay in (pinned) host memory. With ``weights``
    (:class:`wspx.lab.weights.LiveWeights`, a skeleton whose parameters are on the meta
    device) each block's values are read from it, the checkpoint plus the session's edits,
    for its forward pass; what lies outside the blocks is read onto ``device`` once."""
    device = torch.device(device)
    blocks = [b for _, s in stacks(model) for b in s]
    in_blocks = {id(p) for b in blocks for p in b.parameters()}
    names = {id(p): n for n, p in model.named_parameters()}
    with torch.no_grad():
        outside = [n for n, p in model.named_parameters() if id(p) not in in_blocks]
        if weights is not None:
            weights.load_resident([n for n in outside if not weights.resident(n)], device)
        for module in model.modules():
            for p in module._parameters.values():
                if p is not None and id(p) not in in_blocks and p.device != device:
                    p.data = p.data.to(device)
            for key, b in module._buffers.items():  # buffers are small: all on the GPU
                if b is not None and b.device != device:
                    module._buffers[key] = b.to(device)
        if weights is None:
            for block in blocks:
                for p in _params(block):
                    host = p.data if p.device.type == "cpu" else p.data.cpu()
                    p.data = host.pin_memory() if pin and torch.cuda.is_available() else host
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # what moved off is free for vLLM
    out = Streamed(model, device, blocks)

    held: dict[int, torch.Tensor] = {}  # a block's parameter → what it held before its turn

    def bring(block: nn.Module, *_: Any, **__: Any) -> None:
        from wspx.lab.weights import fill, is_placeholder

        params = _params(block)
        if weights is not None:
            wanted = [names[id(p)] for p in params if is_placeholder(p)]
            values = weights.layer(wanted, device)
            for p in params:
                v = values.get(names[id(p)])
                if v is not None:
                    held[id(p)] = fill(p, v if v.dtype == p.dtype else v.to(p.dtype))
            return
        for p in params:
            if p.device.type == "cpu":
                held[id(p)] = fill(p, p.data.to(device, non_blocking=True))

    def drop(block: nn.Module, *_: Any, **__: Any) -> None:
        from wspx.lab.weights import fill

        for p in _params(block):
            before = held.pop(id(p), None)
            if before is not None:
                fill(p, before)

    for block in blocks:
        out.handles.append(block.register_forward_pre_hook(bring, with_kwargs=True))
        out.handles.append(block.register_forward_hook(drop, always_call=True))
    model._wspx_streamed = out  # type: ignore[assignment]
    return out


def unstream(model: nn.Module) -> bool:
    """Move a streamed model's layers onto its GPU, when they fit there; returns whether
    (a skeleton streamed from the checkpoint stays as it is)."""
    streamed: Streamed | None = getattr(model, "_wspx_streamed", None)
    if streamed is None:
        return True
    from wspx.lab.weights import is_placeholder

    if any(is_placeholder(p) for b in streamed.blocks for p in b.parameters()):
        return False
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
