"""Loaded by vLLM in the process that holds the model (``--worker-extension-cls``).

The lab pushes only the layers an edit changed, one layer per update call. vLLM reloads
layer by layer (``VLLM_RELOAD_MODE=layerwise``) and puts back every layer an update did not
deliver, and its ``weight_update_active()`` tells a model's ``load_weights`` that a call may
hold part of the checkpoint. Some models' loaders do not consult it and refuse a call that
does not hold every tensor (Inkling2's: "Missing checkpoint tensors"); this module lets
those calls through while an update is running, and only then.

Some modules process their weights themselves after loading (a model-level
``process_weights_after_loading``, not a quantization method's), which vLLM's layerwise reload
does not rerun. For Inkling2 this module does it, in place so that captured CUDA graphs keep
their tensors:

- the experts (``Inkling2Experts``) repack their checkpoint tensors into a kernel layout and
  empty them. They are kept out of vLLM's layerwise reload; an update gives an emptied
  tensor its checkpoint shape again before it loads, and after the update the experts are
  repacked and the new kernel tensors copied into the old ones;
- the attention's shifted relative projection and the loop's message queries are derived
  from their weights again after an update, copied into the tensors already there.

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


_TOUCHED: list[Any] = []
"""Inkling2Experts an update running now has loaded into."""


def _experts_shape(module: Any, param: Any) -> tuple[int, ...] | None:
    pack, group = (2, 16) if module.nvfp4 else (1, 32)
    for name, (n, k) in zip(("w13", "w2"), module.shapes, strict=True):
        if param is getattr(module, f"{name}_weight"):
            return (module.num_experts, n, k // pack)
        if param is getattr(module, f"{name}_weight_scale"):
            return (module.num_experts, n, k // group)
    return None


def _patch_experts(cls: type) -> list[str]:
    """Inkling2Experts: out of the layerwise reload, reloaded and repacked in place."""
    import torch

    if getattr(cls, "_wspx_reload", False):
        return []
    cls._skip_layerwise_reload = True  # type: ignore[attr-defined]
    original_pwal = cls.process_weights_after_loading  # type: ignore[attr-defined]

    def with_room(loader: Any) -> Any:
        @functools.wraps(loader)
        def load(self: Any, param: Any, loaded: Any) -> Any:
            from vllm.model_executor.model_loader.reload import weight_update_active

            if weight_update_active():
                if self not in _TOUCHED:
                    _TOUCHED.append(self)
                if param.numel() == 0:  # emptied after repacking: its checkpoint shape again
                    shape = _experts_shape(self, param)
                    if shape is not None:
                        param.data = torch.empty(shape, dtype=param.dtype, device=param.device)
            return loader(self, param, loaded)

        return load

    for name in ("_load_weight", "_load_block_scales", "_load_expert_shard"):
        if name in cls.__dict__:
            setattr(cls, name, with_room(cls.__dict__[name]))
    cls._wspx_pwal = original_pwal  # type: ignore[attr-defined]
    cls._wspx_reload = True  # type: ignore[attr-defined]
    return [f"{cls.__module__}.{cls.__qualname__} (repacked in place)"]


def _rebuild_experts(module: Any) -> None:
    """Repack an update's experts and copy the result into the kernel tensors in use."""
    old = getattr(module, "humming", None)
    names = [f"{p}_{s}" for p in ("w13", "w2") for s in ("weight", "weight_scale")]
    if old is None:  # kernels that read the checkpoint layout: loaded in place already
        return
    empty = [n for n in names if getattr(module, n).numel() == 0]
    if empty:
        raise RuntimeError(f"an update gave these experts only part of their tensors: {empty}")
    type(module)._wspx_pwal(module)
    new = module.humming
    for projection, tensors in new.tensors.items():
        for key, tensor in tensors.items():
            if tensor is None:
                continue
            target = old.tensors[projection][key]
            if target.shape != tensor.shape or target.dtype != tensor.dtype:
                raise RuntimeError(
                    f"repacked {projection}.{key} is {tuple(tensor.shape)} {tensor.dtype}, the "
                    f"one in use {tuple(target.shape)} {target.dtype}"
                )
            target.copy_(tensor)
    module.humming = old  # what the captured graphs read


def _rederive(module: Any) -> None:
    """Run a module's own weight processing again, copying derived tensors that it makes
    anew into the ones already there (so captured graphs see them)."""
    import torch

    before = {k: v for k, v in vars(module).items() if isinstance(v, torch.Tensor)}
    module.process_weights_after_loading()
    for key, old in before.items():
        new = getattr(module, key, None)
        if not isinstance(new, torch.Tensor) or new is old:
            continue
        if new.shape == old.shape and new.dtype == old.dtype:
            old.copy_(new)
            setattr(module, key, old)


def _patch_finish() -> bool:
    """After vLLM finishes an update: repack the experts it touched, and derive Inkling2's
    other processed tensors again."""
    from vllm.model_executor.model_loader import reload

    original = reload.finish_reload
    if getattr(original, "_wspx", False):
        return True

    @functools.wraps(original)
    def finish_reload(model: Any, model_config: Any, device: Any) -> Any:
        out = original(model, model_config, device)
        touched, _TOUCHED[:] = list(_TOUCHED), []
        for module in touched:
            _rebuild_experts(module)
        for module in model.modules():
            cls = type(module)
            if (
                cls.__module__.startswith("vllm.models.inkling2")
                and "process_weights_after_loading" in cls.__dict__
                and not getattr(cls, "_wspx_reload", False)
                and getattr(module, "quant_method", None) is None
            ):
                _rederive(module)
        return out

    finish_reload._wspx = True  # type: ignore[attr-defined]
    reload.finish_reload = finish_reload
    return True


def patch_reload() -> list[str]:
    try:
        from vllm.models.inkling2.nvidia.moe import Inkling2Experts
    except Exception:
        return []
    done = _patch_experts(Inkling2Experts)
    _patch_finish()
    return done


PATCHED = patch() + patch_reload()


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
            if module in _TOUCHED:  # experts reloaded here: every tensor must have come
                parts = [f"{p}_{s}" for p in ("w13", "w2") for s in ("weight", "weight_scale")]
                if getattr(module, "humming", None) is not None and any(
                    getattr(module, n).numel() == 0 for n in parts
                ):
                    out.append(f"{name} ({type(module).__name__}: part of its experts)")
                continue
            info = LAYERWISE_INFO.get(module)
            if info is None or not info.can_load():
                continue
            loaded, total = info.load_numel, info.load_numel_total
            if total and 0 < loaded < total:
                out.append(f"{name} ({type(module).__name__}: {loaded}/{total})")
        return out
