"""Edit ops: high-level edits that compile to the patch IR at staging time (§5.3).

An op runs once against the current working tree and returns concrete fp32
deltas. Its name and arguments are kept for readability (and so a merge can
re-apply the intent), but replay never re-runs it.

Third-party ops register under the ``wspx.ops`` entry-point group with the
signature ``op(ctx: OpContext, **args) -> list[TensorDelta]``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from importlib.metadata import entry_points
from typing import Any

import torch

from wspx.errors import PatchError
from wspx.lab.ir.model_ir import ModelIR, Target
from wspx.lab.patch import TensorDelta
from wspx.lab.tree import WorkingTree

Op = Callable[..., list[TensorDelta]]


@dataclass
class OpContext:
    """What an op may see: the IR, the live weights and ways to resolve names."""

    ir: ModelIR
    tree: WorkingTree
    resolve: Callable[[str], list[Target]]
    direction: Callable[[Any], torch.Tensor]
    """``dir:name`` handle, list or tensor → fp32 vector on the model's device."""
    sources: dict[str, tuple[ModelIR, torch.nn.Module]] = field(default_factory=dict)
    """Other loaded models, for ``transplant``."""

    def weight(self, name: str) -> torch.Tensor:
        """The live tensor (current working tree), as fp32."""
        return self.tree.params[name].detach().float()

    def base_hash(self, name: str) -> str:
        info = self.ir.params[name]
        if info.hash is None:
            raise PatchError(f"{name}: base hash unknown (IR built without hashes)")
        return info.hash

    def targets(self, spec: str | list[str] | Target | list[Target]) -> list[Target]:
        items = spec if isinstance(spec, list) else [spec]
        out: list[Target] = []
        for item in items:
            out.extend(self.resolve(item) if isinstance(item, str) else [item])
        if not out:
            raise PatchError(f"{spec!r} resolves to no weights")
        return out


@cache
def registry() -> dict[str, Op]:
    from wspx.lab.ops import builtin

    ops: dict[str, Op] = dict(builtin.BUILTIN)
    for ep in entry_points(group="wspx.ops"):
        if ep.name not in ops:
            ops[ep.name] = ep.load()
    return ops


def get(name: str) -> Op:
    ops = registry()
    if name not in ops:
        raise PatchError(f"unknown op {name!r}; available: {', '.join(sorted(ops))}")
    return ops[name]
