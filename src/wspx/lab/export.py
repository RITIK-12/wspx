"""Export (§5.5.3): merged checkpoints, PEFT adapters, GGUF and portable bundles.

``merged`` writes the live tensors, so the checkpoint is bit-identical to
what was tested. ``peft`` is exact for low-rank deltas and reports the error
of a truncated SVD otherwise.
"""

from __future__ import annotations

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
from wspx.lab.ir.model_ir import ModelIR
from wspx.lab.patch import TensorDelta
from wspx.lab.tensors import tensor_hash
from wspx.lab.tree import WorkingTree

SHARD_BYTES = 5 * 1024**3
CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def export_merged(
    model: torch.nn.Module,
    ir: ModelIR,
    out: Path,
    *,
    source: Path | None = None,
    shard_bytes: int = SHARD_BYTES,
) -> dict[str, Any]:
    """Sharded safetensors of the live weights, plus config and tokenizer files."""
    out.mkdir(parents=True, exist_ok=True)
    tensors = {name: p.detach().contiguous().cpu() for name, p in model.named_parameters()}
    tensors.update(
        {
            name: b.detach().contiguous().cpu()
            for name, b in model.named_buffers()
            if name in _persistent_buffers(model)
        }
    )
    shards: list[dict[str, torch.Tensor]] = [{}]
    size = 0
    for name in sorted(tensors):
        t = tensors[name]
        nbytes = t.numel() * t.element_size()
        if shards[-1] and size + nbytes > shard_bytes:
            shards.append({})
            size = 0
        shards[-1][name] = t
        size += nbytes
    weight_map: dict[str, str] = {}
    files = []
    for i, shard in enumerate(shards):
        fname = (
            "model.safetensors"
            if len(shards) == 1
            else f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        )
        save_file(shard, str(out / fname), metadata={"format": "pt", "wspx": ir.fingerprint})
        files.append(fname)
        weight_map.update(dict.fromkeys(shard, fname))
    if len(shards) > 1:
        total = sum(t.numel() * t.element_size() for t in tensors.values())
        (out / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
        )
    _copy_configs(model, out, source)
    promoted = [
        n
        for n, t in tensors.items()
        if t.dtype == torch.float32 and ir.params.get(n) and ir.params[n].dtype != "float32"
    ]
    warnings = []
    if promoted:
        warnings.append(
            f"{len(promoted)} tensor(s) stored in fp32 (promote mode); runtimes that cast the "
            f"whole model to one dtype will round them: {', '.join(promoted[:4])}"
        )
    hashes = {name: tensor_hash(t) for name, t in tensors.items() if name in ir.params}
    return {
        "path": str(out),
        "files": files,
        "tensors": len(tensors),
        "warnings": warnings,
        "hashes": hashes,
    }


def _persistent_buffers(model: torch.nn.Module) -> set[str]:
    out = set()
    for mod_name, module in model.named_modules():
        for name, _ in module.named_buffers(recurse=False):
            if name not in getattr(module, "_non_persistent_buffers_set", set()):
                out.add(f"{mod_name}.{name}" if mod_name else name)
    return out


def _copy_configs(model: torch.nn.Module, out: Path, source: Path | None) -> None:
    copied = False
    if source is not None:
        for name in CONFIG_FILES:
            if (source / name).is_file():
                shutil.copy2(source / name, out / name)
                copied = True
    config = getattr(model, "config", None)
    if config is not None and not (out / "config.json").exists():
        config.save_pretrained(str(out))
        copied = True
    del copied


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
        module = model.get_submodule(info.module)
        if (
            len(info.shape) != 2
            or role is None
            or role.in_dim is None
            or not hasattr(module, "weight")
            or name.rpartition(".")[2] != "weight"
        ):
            raise WspxError(
                f"{name}: PEFT LoRA covers 2-D linear weights only; export --merged instead"
            )
        deltas = [
            tree.delta(o) for o in (*state.merged.get(name, ()), *state.adapter.get(name, ()))
        ]
        b, a, err = _lora_factors(deltas, info.shape, role.in_dim, max_rank)
        flip = role.in_dim == 0
        fan_in_fan_out.add(flip)
        key = f"base_model.model.{info.module}"
        tensors[f"{key}.lora_A.weight"] = a.contiguous()
        tensors[f"{key}.lora_B.weight"] = b.contiguous()
        ranks[info.module] = a.shape[0]
        if err > 0:
            errors[name] = err
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
