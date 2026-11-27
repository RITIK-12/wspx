"""Export (§5.5.3): merged checkpoints, PEFT adapters, GGUF and portable bundles.

``merged`` writes the live tensors, so the checkpoint is bit-identical to
what was tested. ``peft`` is exact for low-rank deltas and reports the error
of a truncated SVD otherwise.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from wspx.errors import WspxError
from wspx.lab import checkpoint
from wspx.lab.ir.model_ir import ModelIR
from wspx.lab.patch import TensorDelta
from wspx.lab.tree import WorkingTree

SHARD_BYTES = 5 * 1024**3


def export_merged(
    model: torch.nn.Module,
    ir: ModelIR,
    out: Path,
    *,
    source: Path | None = None,
    tokenizer: Any = None,
    processor: Any = None,
    shard_bytes: int = SHARD_BYTES,
) -> dict[str, Any]:
    """Sharded safetensors of the live weights under the checkpoint's own names and layouts
    (what transformers renamed or fused on load is turned back), plus every non-weight file
    of the source checkpoint: config, tokenizer, chat template, processor."""
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.safetensors"):
        stale.unlink()
    persistent = _persistent_buffers(model)
    live: dict[str, torch.Tensor] = dict(model.named_parameters())
    live.update({n: b for n, b in model.named_buffers() if n in persistent})
    promoted = [
        n
        for n, t in live.items()
        if t.dtype == torch.float32 and ir.params.get(n) and ir.params[n].dtype != "float32"
    ]
    src = checkpoint.read_source(source)
    dequantized = bool(src and src.quantized and getattr(model, "hf_quantizer", None) is None)
    written: dict[str, tuple[str, tuple[int, ...]]] = {}
    shards: list[list[str]] = []
    weight_map: dict[str, str] = {}
    pending: dict[str, torch.Tensor] = {}
    size = total = 0

    def flush() -> None:
        nonlocal pending, size
        if pending:
            tmp = f"part-{len(shards):05d}.safetensors"
            save_file(pending, str(out / tmp), metadata={"format": "pt", "wspx": ir.fingerprint})
            shards.append(sorted(pending))
            pending, size = {}, 0

    for batch in checkpoint.stream(model, live):
        for name, t in batch.items():
            nbytes = t.numel() * t.element_size()
            if pending and size + nbytes > shard_bytes:
                flush()
            pending[name] = t
            size += nbytes
            total += nbytes
            written[name] = (checkpoint.st_dtype(t), tuple(t.shape))
    flush()
    files = []
    for i, names in enumerate(shards):
        fname = (
            "model.safetensors"
            if len(shards) == 1
            else f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        )
        (out / f"part-{i:05d}.safetensors").rename(out / fname)
        files.append(fname)
        weight_map.update(dict.fromkeys(names, fname))
    warnings = []
    extras = sorted(src.extra_files) if src else []
    for name in extras:  # draft heads and the like: not loaded, so never edited
        shutil.copy2(src.path / name, out / name)  # type: ignore[union-attr]
    if len(shards) > 1 or extras:
        (out / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
        )
    _copy_configs(
        model, out, src, dequantized=dequantized, tokenizer=tokenizer, processor=processor
    )
    if extras:
        warnings.append(f"copied unedited (not part of the loaded model): {', '.join(extras)}")
    if promoted:
        warnings.append(
            f"{len(promoted)} tensor(s) stored in fp32 (promote mode); runtimes that cast the "
            f"whole model to one dtype will round them: {', '.join(promoted[:4])}"
        )
    if src and dequantized:
        lowered = sorted(
            k
            for k, (dtype, _) in written.items()
            if k in src.tensors and src.tensors[k][0] != dtype
        )
        method = (src.quantization or {}).get("quant_method", "hf_quant_config.json")
        warnings.append(
            f"the source checkpoint is quantized ({method}) and was dequantized on load: "
            f"{len(lowered)} tensor(s) are written in their loaded dtype instead "
            f"({', '.join(lowered[:2])}{', …' if len(lowered) > 2 else ''}) and the "
            "quantization config is left out, so loaders read the export unquantized"
        )
    if src:
        warnings += checkpoint.compare(written, src, dequantized=dequantized)
    return {
        "path": str(out),
        "files": files,
        "tensors": len(written),
        "warnings": warnings,
        "names": "checkpoint" if src and src.tensors else "loaded",
    }


def _persistent_buffers(model: torch.nn.Module) -> set[str]:
    out = set()
    for mod_name, module in model.named_modules():
        for name, _ in module.named_buffers(recurse=False):
            if name not in getattr(module, "_non_persistent_buffers_set", set()):
                out.add(f"{mod_name}.{name}" if mod_name else name)
    return out


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt", ".h5", ".msgpack")


def _copy_configs(
    model: torch.nn.Module,
    out: Path,
    src: checkpoint.Source | None,
    *,
    dequantized: bool = False,
    tokenizer: Any = None,
    processor: Any = None,
) -> None:
    """Every non-weight file of the source (config, tokenizer, templates, processor), or,
    without one, what the loaded objects can save."""
    if src is not None:
        for path in sorted(src.path.rglob("*")):
            rel = path.relative_to(src.path)
            if (
                not path.is_file()
                or rel.parts[0].startswith(".")
                or path.name.endswith(WEIGHT_SUFFIXES)
                or path.name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
                or (dequantized and path.name == "hf_quant_config.json")
            ):
                continue
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out / rel)
        if dequantized and (out / "config.json").is_file():
            config = json.loads((out / "config.json").read_text())
            config.pop("quantization_config", None)
            for sub in config.values():  # a VLM's text config can carry its own
                if isinstance(sub, dict):
                    sub.pop("quantization_config", None)
            (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    config = getattr(model, "config", None)
    if config is not None and not (out / "config.json").exists():
        config.save_pretrained(str(out))
        if getattr(model, "hf_quantizer", None) is None:  # the weights are not quantized
            saved = json.loads((out / "config.json").read_text())
            if saved.pop("quantization_config", None) is not None:
                (out / "config.json").write_text(json.dumps(saved, indent=2) + "\n")
    generation = getattr(model, "generation_config", None)
    if generation is not None and not (out / "generation_config.json").exists():
        # a generation config that does not validate is not worth failing the export over
        with contextlib.suppress(Exception):
            generation.save_pretrained(str(out))
    saved_by = (
        (tokenizer, ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")),
        (processor, ("preprocessor_config.json", "processor_config.json")),
    )
    for obj, files in saved_by:
        if obj is None or not hasattr(obj, "save_pretrained"):
            continue
        if not any((out / f).exists() for f in files):
            # one that cannot save leaves the rest of the export usable
            with contextlib.suppress(Exception):
                obj.save_pretrained(str(out))


def export_peft(
    model: torch.nn.Module,
    ir: ModelIR,
    tree: WorkingTree,
    out: Path,
    *,
    base_model: str | None = None,
    max_rank: int = 64,
) -> dict[str, Any]:
    """A PEFT LoRA adapter equal to the working tree's edits (exact for low-rank deltas)."""
    tensors: dict[str, torch.Tensor] = {}
    ranks: dict[str, int] = {}
    errors: dict[str, float] = {}
    fan_in_fan_out: set[bool] = set()
    state = tree.state
    names = sorted(set(state.merged) | set(state.adapter))
    if not names:
        raise WspxError("nothing to export: the working tree has no edits")
    for name in names:
        role = ir.primary_role(name)
        info = ir.params[name]
        deltas = [
            tree.delta(o) for o in (*state.merged.get(name, ()), *state.adapter.get(name, ()))
        ]
        if _restored(tree, name):
            continue  # patches that cancel out (say, a transplant back to base) change nothing
        module = model.get_submodule(info.module)
        if (
            len(info.shape) != 2
            or role is None
            or role.in_dim is None
            or not hasattr(module, "weight")
            or name.rpartition(".")[2] != "weight"
        ):
            kind = f"{type(module).__name__} {role.kind}" if role else type(module).__name__
            raise WspxError(
                f"{name}: PEFT LoRA export covers linear-layer weights, and this is a {kind}; "
                "export merged (the full weights) or bundle instead"
            )
        b, a, err = _lora_factors(deltas, info.shape, role.in_dim, max_rank)
        flip = role.in_dim == 0
        fan_in_fan_out.add(flip)
        key = f"base_model.model.{info.module}"
        tensors[f"{key}.lora_A.weight"] = a.contiguous()
        tensors[f"{key}.lora_B.weight"] = b.contiguous()
        ranks[info.module] = a.shape[0]
        if err > 0:
            errors[name] = err
    if not tensors:
        raise WspxError("nothing to export: the working tree's patches cancel out")
    if len(fan_in_fan_out) > 1:
        raise WspxError(
            "edits mix nn.Linear and Conv1D layouts; PEFT needs one fan_in_fan_out setting"
        )
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    r = max(ranks.values())
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model,
        "r": r,
        "lora_alpha": r,
        "rank_pattern": ranks,
        "alpha_pattern": ranks,
        "target_modules": sorted(ranks),
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": fan_in_fan_out.pop(),
        "inference_mode": True,
        "task_type": "CAUSAL_LM" if ir.is_lm else None,
        "init_lora_weights": True,
    }
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
    return {"path": str(out), "modules": sorted(ranks), "ranks": ranks, "svd_error": errors}


def _restored(tree: WorkingTree, name: str) -> bool:
    """Whether ``name``'s patches cancel out (and add no adapter).

    Cancelled patches can leave float32 rounding residue (a transplant back to base is
    computed against the rounded weight), so "cancel" means below float32 resolution.
    """
    state = tree.state
    if state.adapter.get(name):
        return False
    weight = tree.materialize_state(name, state, dtype=torch.float32)
    base = tree.base(name).float()
    eps = torch.finfo(torch.float32).eps
    return float((weight - base).abs().max()) <= 4 * eps * float(base.abs().max())


def _lora_factors(
    deltas: list[TensorDelta], shape: tuple[int, ...], in_dim: int, max_rank: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """``ΔW = B A`` in PEFT's convention (B [out, r], A [r, in]); exact when every delta is low-rank."""
    if all(d.kind == "lowrank" and d.index is None for d in deltas):
        us, vs = [], []
        for d in deltas:
            assert d.u is not None
            assert d.v is not None
            us.append(d.u * d.scale)
            vs.append(d.v)
        u, v = torch.cat(us, dim=1), torch.cat(vs, dim=1)
        # Stored blocks are u (dim 0) ⊗ v (dim 1). PEFT wants B [out, r] and A [r, in] with
        # ΔW = B A for nn.Linear ([out, in]) and ΔW = (B A)ᵀ for Conv1D ([in, out]).
        b, a = (u, v.T) if in_dim == 1 else (v, u.T)
        return b.float(), a.float(), 0.0
    full = torch.zeros(shape)
    for d in deltas:
        d.add_into(full)
    mat = full if in_dim == 1 else full.T  # always factor the [out, in] matrix
    u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    r = int(min(max_rank, int((s > 1e-7 * s.max()).sum()) or 1))
    b = u[:, :r] * s[:r]
    a = vh[:r]
    err = float((mat - b @ a).norm() / mat.norm().clamp_min(1e-30))
    return b, a, err


def export_gguf(merged_dir: Path, out_file: Path, *, outtype: str = "auto") -> dict[str, Any]:
    """Convert a merged export with llama.cpp's ``convert_hf_to_gguf.py``."""
    script = _find_converter()
    cmd = [
        sys.executable,
        str(script),
        str(merged_dir),
        "--outfile",
        str(out_file),
        "--outtype",
        outtype,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise WspxError(f"GGUF conversion failed:\n{proc.stderr[-2000:]}")
    return {"path": str(out_file)}


def _find_converter() -> Path:
    root = os.environ.get("LLAMA_CPP_DIR")
    candidates = [Path(root) / "convert_hf_to_gguf.py"] if root else []
    found = shutil.which("convert_hf_to_gguf.py")
    if found:
        candidates.append(Path(found))
    for path in candidates:
        if path.is_file():
            return path
    raise WspxError("GGUF export needs llama.cpp: set LLAMA_CPP_DIR to a llama.cpp checkout")


def export_bundle(
    objects: dict[str, bytes], manifest: dict[str, Any], out_file: Path
) -> dict[str, Any]:
    """A portable tarball: manifest plus every object needed to rebuild the commit on its base."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    import io

    with tarfile.open(out_file, "w:gz") as tar:
        data = json.dumps(manifest, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        for oid, blob in sorted(objects.items()):
            info = tarfile.TarInfo(f"objects/{oid.removeprefix('sha256:')}")
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
    return {"path": str(out_file), "objects": len(objects)}
