"""Loaded by vLLM in the process that holds the model (``--worker-extension-cls``).

The lab pushes only the layers an edit changed, one layer per update call. vLLM reloads
layer by layer (``VLLM_RELOAD_MODE=layerwise``) and puts back every layer an update did not
deliver, and its ``weight_update_active()`` tells a model's ``load_weights`` that a call may
hold part of the checkpoint. Some models' loaders do not consult it and refuse a call that
does not hold every tensor (Inkling2's: "Missing checkpoint tensors"); this module lets
those calls through while an update is running, and only then.

Like the worker, it runs with vLLM's own Python and imports nothing from wspx.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import Any

MISSING = ("Missing checkpoint tensors",)
"""How strict loaders say a call lacked tensors (the message's start)."""
STRICT_LOADERS = ("vllm.models.inkling2.nvidia.model:Inkling2ForConditionalGeneration",)


def _partial_ok(cls: type) -> None:
    original = cls.__dict__.get("load_weights")
    if original is None or getattr(original, "_wspx_partial", False):
        return

    @functools.wraps(original)
    def load_weights(self: Any, weights: Iterable[tuple[str, Any]]) -> Any:
        from vllm.model_executor.model_loader.reload import weight_update_active

        if not weight_update_active():
            return original(self, weights)
        weights = list(weights)
        try:
            return original(self, weights)
        except ValueError as exc:
            if not str(exc).startswith(MISSING):
                raise
            # every tensor in the call was loaded before the check ran
            return {name for name, _ in weights}

    load_weights._wspx_partial = True  # type: ignore[attr-defined]
    cls.load_weights = load_weights  # type: ignore[attr-defined]


def _subclasses(cls: type) -> list[type]:
    out = [cls]
    sub: type
    for sub in cls.__subclasses__():
        out += _subclasses(sub)
    return out


def patch() -> list[str]:
    """Let the strict loaders vLLM has take part of a checkpoint during an update."""
    import importlib

    done = []
    for spec in STRICT_LOADERS:
        module, _, name = spec.partition(":")
        try:
            cls = getattr(importlib.import_module(module), name)
        except Exception:  # this vLLM build has no such model
            continue
        for c in _subclasses(cls):
            _partial_ok(c)
            done.append(f"{c.__module__}.{c.__qualname__}")
    return done


PATCHED = patch()


class WspxWorkerExtension:
    """Injected into vLLM's worker class; importing this module applied the patches."""

    def wspx_patched_loaders(self) -> list[str]:
        return list(PATCHED)

    def wspx_incomplete(self) -> list[str]:
        """Modules an update running now has given only part of their tensors: finishing
        would process them with the rest missing. A model can map a checkpoint tensor kept
        outside the layer stack into every layer (Inkling2Loop's message queries)."""
        from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO

        runner = getattr(self, "model_runner", None)
        model = getattr(runner, "model", None)
        if model is None:
            return []
        out = []
        for name, module in model.named_modules():
            info = LAYERWISE_INFO.get(module)
            if info is None or not info.can_load():
                continue
            loaded, total = info.load_numel, info.load_numel_total
            if total and 0 < loaded < total:
                out.append(f"{name} ({type(module).__name__}: {loaded}/{total})")
        return out
