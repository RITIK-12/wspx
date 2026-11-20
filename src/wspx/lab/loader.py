"""Loading models into the lab (§5.1.1).

1. **Resolve the source**: a local directory or a Hugging Face repo (already
   fetched into the models volume by the downloader; fetched here when run
   locally), or a custom ``nn.Module`` given as ``pkg.module:Class``.
2. **Instantiate on the meta device** to get the true parameter table at no cost.
3. **Bind the weights**; missing or unexpected keys are an error.
4. **Fingerprint**: repo, revision, modeling-code hash and checkpoint file hashes.

Base tensors that map one-to-one onto a checkpoint tensor can be re-read from
the read-only files, so the working tree's base cache may evict them.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from wspx.errors import ModelLoadError

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class Loaded:
    model: torch.nn.Module
    tokenizer: Any
    fingerprint: str
    source: str
    path: Path | None
    readable: dict[str, tuple[Path, str]] = field(default_factory=dict)
    """Parameter → (safetensors file, key) for tensors stored verbatim."""

    def base_reader(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        if name not in self.readable:
            raise ModelLoadError(f"{name} is not stored verbatim in the checkpoint")
        path, key = self.readable[name]
        with safe_open(str(path), framework="pt") as fh:
            tensor: torch.Tensor = fh.get_tensor(key)
        dtype = dict(self.model.named_parameters())[name].dtype
        return tensor.to(dtype)


def resolve_source(source: str, *, revision: str | None = None, local_only: bool = False) -> Path:
    path = Path(source).expanduser()
    if path.is_dir():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - lab dependency
        raise ModelLoadError("huggingface_hub is needed to fetch models") from exc
    try:
        return Path(
            snapshot_download(
                source,
                revision=revision,
                local_files_only=local_only,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "*.model",
                    "*.txt",
                    "*.jinja",
                    "*.tiktoken",
                ],
            )
        )
    except Exception as exc:
        raise ModelLoadError(f"cannot fetch {source}: {exc}") from exc


def load_hf(
    source: str,
    *,
    revision: str | None = None,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    device: str | None = None,
    local_only: bool = False,
) -> Loaded:
    """Load a causal LM (or any AutoModel) and fingerprint it."""
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    path = resolve_source(source, revision=revision, local_only=local_only)
    config = AutoConfig.from_pretrained(str(path), trust_remote_code=trust_remote_code)
    torch_dtype = DTYPES.get(dtype, "auto")
    is_lm = bool(getattr(config, "architectures", None)) and any(
        a.endswith(("ForCausalLM", "LMHeadModel", "ForConditionalGeneration"))
        for a in config.architectures or []
    )
    auto = AutoModelForCausalLM if is_lm else AutoModel
    with torch.device("meta"):
        skeleton = auto.from_config(config, trust_remote_code=trust_remote_code)  # type: ignore[no-untyped-call]
    expected = {n for n, _ in skeleton.named_parameters()}
    try:
        loaded: Any = auto.from_pretrained(
            str(path),
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map=device or ("auto" if torch.cuda.is_available() else None),
            output_loading_info=True,
        )
        model, info = loaded
    except Exception as exc:
        raise ModelLoadError(f"cannot load {source}: {exc}") from exc
    missing = [k for k in info.get("missing_keys", []) if k in expected]
    unexpected = list(info.get("unexpected_keys", []))
    if missing or unexpected:
        raise ModelLoadError(
            f"{source}: checkpoint does not match the architecture "
            f"(missing {missing[:5]}, unexpected {unexpected[:5]})"
        )
    model.eval()
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=trust_remote_code)
    except Exception:
        tokenizer = None
    readable = _verbatim_tensors(path, model)
    return Loaded(model, tokenizer, fingerprint(path, source, revision), source, path, readable)


def load_module(spec: str, weights: str | None = None, **kwargs: Any) -> Loaded:
    """``pkg.module:Class`` plus optional safetensors weights."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ModelLoadError(f"{spec!r}: expected 'package.module:Class'")
    try:
        cls = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ModelLoadError(f"cannot import {spec}: {exc}") from exc
    model = cls(**kwargs)
    path: Path | None = None
    if weights:
        from safetensors.torch import load_file

        path = Path(weights)
        state = load_file(str(path))
        result = model.load_state_dict(state, strict=True)
        del result
    digest = hashlib.sha256(spec.encode())
    if path is not None:
        digest.update(_file_hash(path).encode())
    model.eval()
    readable = {n: (path, n) for n, _ in model.named_parameters()} if path is not None else {}
    return Loaded(model, None, "fp:" + digest.hexdigest(), spec, path, readable)


def fingerprint(path: Path, source: str, revision: str | None) -> str:
    """repo + revision + code hash + checkpoint file hashes."""
    digest = hashlib.sha256()
    digest.update(f"{source}@{revision or ''}\n".encode())
    for file in sorted(path.glob("*.py")):
        digest.update(f"code {file.name} {_file_hash(file)}\n".encode())
    for file in sorted(path.glob("*.safetensors")):
        digest.update(f"weights {file.name} {_file_hash(file)}\n".encode())
    config = path / "config.json"
    if config.exists():
        digest.update(f"config {_file_hash(config)}\n".encode())
    return "fp:" + digest.hexdigest()


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _verbatim_tensors(path: Path, model: torch.nn.Module) -> dict[str, tuple[Path, str]]:
    """Parameters whose checkpoint tensor has the same name and shape (re-readable bases)."""
    from safetensors import safe_open

    index = path / "model.safetensors.index.json"
    files: dict[str, Path] = {}
    if index.exists():
        files = {k: path / v for k, v in json.loads(index.read_text())["weight_map"].items()}
    else:
        for file in path.glob("*.safetensors"):
            with safe_open(str(file), framework="pt") as fh:
                files.update(dict.fromkeys(fh.keys(), file))
    out: dict[str, tuple[Path, str]] = {}
    shapes: dict[Path, Any] = {}
    for name, param in model.named_parameters():
        found = files.get(name)
        if found is None:
            continue
        file = found
        if file not in shapes:
            with safe_open(str(file), framework="pt") as fh:
                shapes[file] = {k: fh.get_slice(k).get_shape() for k in fh.keys()}  # noqa: SIM118
        if list(shapes[file].get(name, [])) == list(param.shape):
            out[name] = (file, name)
    return out
