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

import contextlib
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


_EXPERT_TENSORS = tuple(f"{p}_{s}" for p in ("w13", "w2") for s in ("weight", "weight_scale"))


def _patch_experts(cls: type) -> list[str]:
    """Inkling2Experts: out of the layerwise reload; an update gives its emptied tensors
    their checkpoint shape again, and its processing repacks them into the kernel tensors
    already in use."""
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

    @functools.wraps(original_pwal)
    def process_weights_after_loading(self: Any) -> None:
        old = getattr(self, "humming", None)
        empty = [n for n in _EXPERT_TENSORS if getattr(self, n).numel() == 0]
        if old is not None and len(empty) == len(_EXPERT_TENSORS):
            return  # repacked already, and no update loaded them again
        if old is not None and empty:
            raise RuntimeError(f"an update gave these experts only part of their tensors: {empty}")
        original_pwal(self)
        new = self.humming
        if old is None or new is old:
            return
        for projection, tensors in new.tensors.items():
            for key, tensor in tensors.items():
                if tensor is None:
                    continue
                target = old.tensors[projection][key]
                if target.shape != tensor.shape or target.dtype != tensor.dtype:
                    raise RuntimeError(
                        f"repacked {projection}.{key} is {tuple(tensor.shape)} {tensor.dtype}, "
                        f"the one in use {tuple(target.shape)} {target.dtype}"
                    )
                target.copy_(tensor)
        self.humming = old  # what the captured graphs read

    cls.process_weights_after_loading = process_weights_after_loading  # type: ignore[attr-defined]
    cls._wspx_reload = True  # type: ignore[attr-defined]
    return [f"{cls.__module__}.{cls.__qualname__} (repacked in place)"]


def _in_place(cls: type) -> bool:
    """A module's own weight processing, made to keep the tensors it derives: what it makes
    anew is copied into the tensor already there (captured graphs hold its address)."""
    import torch

    original = cls.__dict__.get("process_weights_after_loading")
    if original is None or getattr(original, "_wspx", False):
        return False

    @functools.wraps(original)
    def process_weights_after_loading(self: Any) -> Any:
        before = {k: v for k, v in vars(self).items() if isinstance(v, torch.Tensor)}
        out = original(self)
        for key, old in before.items():
            new = getattr(self, key, None)
            if not isinstance(new, torch.Tensor) or new is old:
                continue
            if new.shape == old.shape and new.dtype == old.dtype:
                old.copy_(new)
                setattr(self, key, old)
        return out

    process_weights_after_loading._wspx = True  # type: ignore[attr-defined]
    cls.process_weights_after_loading = process_weights_after_loading  # type: ignore[attr-defined]
    return True


def _patch_finish() -> bool:
    """Clear what an update recorded when vLLM starts and finishes one."""
    from vllm.model_executor.model_loader import reload

    original = reload.finish_reload
    if getattr(original, "_wspx", False):
        return True
    begin = reload.start_reload

    @functools.wraps(begin)
    def start_reload(model: Any) -> Any:
        SENT.clear()
        _TOUCHED.clear()
        return begin(model)

    @functools.wraps(original)
    def finish_reload(model: Any, model_config: Any, device: Any) -> Any:
        out = original(model, model_config, device)  # reruns modules' own processing
        SENT.clear()
        _TOUCHED.clear()
        return out

    finish_reload._wspx = True  # type: ignore[attr-defined]
    reload.start_reload = start_reload
    reload.finish_reload = finish_reload
    return True


def patch_reload() -> list[str]:
    import importlib
    import inspect
    import pkgutil

    try:
        from vllm.models.inkling2.nvidia.moe import Inkling2Experts
    except Exception:
        return []
    done = _patch_experts(Inkling2Experts)
    package = importlib.import_module("vllm.models.inkling2.nvidia")
    for info in pkgutil.iter_modules(package.__path__):
        try:
            module = importlib.import_module(f"{package.__name__}.{info.name}")
        except Exception:
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__ and cls is not Inkling2Experts and _in_place(cls):
                done.append(f"{cls.__module__}.{cls.__qualname__} (derived in place)")
    _patch_finish()
    return done


SOURCES: dict[str, set[str]] = {}
"""vLLM parameter → the checkpoint tensors the cold start loaded into it."""
SENT: set[str] = set()
"""Checkpoint tensors the update running now has delivered."""


def _set_loader(p: Any, loader: Any) -> bool:
    prop = getattr(type(p), "weight_loader", None)
    try:
        if isinstance(prop, property):  # vLLM's own parameter classes keep it here
            p._weight_loader = loader
        else:
            p.weight_loader = loader
    except Exception:
        return False
    return True


def _patch_cold_start() -> bool:
    """At cold start, record which parameters the load filled (a model's ``load_weights``
    names them) and which checkpoint tensor each loader call carried; after it, record the
    names each update delivers. An update is then checked against what a module had at
    cold start, and what completes it is named by checkpoint tensor."""
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    except Exception:
        return False
    original = DefaultModelLoader.load_weights
    if getattr(original, "_wspx", False):
        return True

    @functools.wraps(original)
    def load_weights(self: Any, model: Any, model_config: Any) -> Any:
        inner = model.load_weights
        seen: set[str] = set()
        by_storage: dict[int, str] = {}
        current: list[str | None] = [None]
        restore: list[tuple[Any, Any]] = []
        for pname, param in model.named_parameters():
            loader = getattr(param, "weight_loader", None)

            def recorded(*args: Any, _l: Any = loader, _n: str = pname, **kwargs: Any) -> Any:
                loaded: Any = args[1] if len(args) > 1 else kwargs.get("loaded_weight")
                source = current[0]
                with contextlib.suppress(Exception):
                    source = by_storage.get(loaded.untyped_storage().data_ptr(), source)
                if source is not None:
                    SOURCES.setdefault(_n, set()).add(source)
                return (_l or default_weight_loader)(*args, **kwargs)

            if _set_loader(param, recorded):
                restore.append((param, loader))

        def tracked(weights: Any) -> Any:
            for name, tensor in weights:
                current[0] = name
                with contextlib.suppress(Exception):
                    by_storage[tensor.untyped_storage().data_ptr()] = name
                yield name, tensor
            current[0] = None

        def recording(weights: Any) -> Any:
            out = inner(tracked(weights))
            if out:
                seen.update(out)
            return out

        model.load_weights = recording
        try:
            return original(self, model, model_config)
        finally:
            del model.load_weights
            for param, loader in restore:
                if loader is None:
                    with contextlib.suppress(Exception):
                        del param.weight_loader
                else:
                    _set_loader(param, loader)
            by_storage.clear()
            model._wspx_cold_loaded = seen
            updating = model.load_weights

            def delivered(weights: Any) -> Any:
                from vllm.model_executor.model_loader.reload import weight_update_active

                weights = list(weights)
                if weight_update_active():
                    SENT.update(name for name, _ in weights)
                return updating(weights)

            model.load_weights = delivered

    load_weights._wspx = True  # type: ignore[attr-defined]
    DefaultModelLoader.load_weights = load_weights
    return True


_patch_cold_start()
PATCHED = patch() + patch_reload()


class WspxWorkerExtension:
    """Injected into vLLM's worker class; importing this module applied the patches."""

    def wspx_patched_loaders(self) -> list[str]:
        return list(PATCHED)

    def wspx_incomplete(self) -> dict[str, list[str]]:
        """Modules an update running now has given only part of their tensors (finishing
        would process them with the rest missing), and the checkpoint tensors that complete
        them (a model can gather one module from tensors the checkpoint keeps apart:
        Inkling2Loop's per-layer queries are one list)."""
        from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO

        runner = getattr(self, "model_runner", None)
        model = getattr(runner, "model", None)
        if model is None:
            return {"modules": [], "need": []}
        cold: set[str] = getattr(model, "_wspx_cold_loaded", set())
        out: list[str] = []
        need: set[str] = set()
        for name, module in model.named_modules():
            prefix = f"{name}." if name else ""
            if module in _TOUCHED:  # experts reloaded here: every tensor must have come
                parts = [f"{p}_{s}" for p in ("w13", "w2") for s in ("weight", "weight_scale")]
                if getattr(module, "humming", None) is not None:
                    empty = [n for n in parts if getattr(module, n).numel() == 0]
                    if empty:
                        out.append(f"{name} ({type(module).__name__}: no {', '.join(empty)})")
                        for n in empty:
                            need |= SOURCES.get(prefix + n, set())
                continue
            info = LAYERWISE_INFO.get(module)
            if info is None or not info.can_load() or not info.loaded_weights:
                continue  # done (processed when its last tensor came), or not in this update
            got = {n for n, _ in info.loaded_weights}
            if cold:  # what the cold start loaded into this module, and not this update
                had = {n for n, _ in module.named_parameters(recurse=False) if prefix + n in cold}
                missing = sorted(had - got)
                if missing:
                    out.append(f"{name} ({type(module).__name__}: no {', '.join(missing[:4])})")
                    for n in missing:
                        need |= SOURCES.get(prefix + n, set())
                continue
            loaded, total = info.load_numel, info.load_numel_total
            if total and 0 < loaded < total:
                out.append(f"{name} ({type(module).__name__}: {loaded}/{total})")
        return {"modules": out, "need": sorted(need - SENT)}
