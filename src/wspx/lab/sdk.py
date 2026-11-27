"""The Python SDK: ``wspx.session()`` (§4.5).

Inside the lab (and inside scripts that ``run`` executes) ``wspx.session()``
returns the live session. On a machine with the ``lab`` extra installed it can
also open a local session directly::

    import wspx
    m = wspx.session("Qwen/Qwen3.5-0.8B", store="./wspx-store")
    hits = m.grep('writes(" Wait") > 0.3', scope="L*.mlp.down", top=50)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from wspx.config import WspxConfig
from wspx.errors import WspxError
from wspx.lab.ir.capture import build_ir
from wspx.lab.ir.model_ir import ModelIR
from wspx.lab.media import probe_inputs
from wspx.lab.session import Lab
from wspx.store import Store

__all__ = ["Lab", "activate", "open_lab", "session"]

_ACTIVE: Lab | None = None


def activate(lab: Lab | None) -> Lab | None:
    """Make ``lab`` the session ``wspx.session()`` returns; returns the previous one."""
    global _ACTIVE
    previous, _ACTIVE = _ACTIVE, lab
    return previous


def session(model: Any = None, **kwargs: Any) -> Lab:
    """The live session, or a new local one when a model is given."""
    if model is None and not kwargs:
        if _ACTIVE is None:
            raise WspxError("no active wspx session; pass a model to open one locally")
        return _ACTIVE
    lab = open_lab(model, **kwargs)
    activate(lab)
    return lab


def open_lab(
    model: str | torch.nn.Module,
    *,
    store: str | Path = "wspx-store",
    session: str = "local",
    branch: str | None = None,
    tokenizer: Any = None,
    example_inputs: Any = None,
    config: WspxConfig | None = None,
    revision: str | None = None,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    workdir: str | Path | None = None,
    processor: Any = None,
    source_dir: str | Path | None = None,
) -> Lab:
    """Load (or take) a model, build or reuse its IR from the store, and open a session.

    ``processor`` (image and video inputs) is loaded with a vision-language checkpoint; pass
    one with a model object.
    """
    st = Store(store)
    base_reader = None
    hashes = None
    fingerprint = None
    source_dir = Path(source_dir) if source_dir is not None else None
    if isinstance(model, str):
        from wspx.lab.loader import load_hf

        loaded = load_hf(model, revision=revision, dtype=dtype, trust_remote_code=trust_remote_code)
        module, tokenizer = loaded.model, tokenizer or loaded.tokenizer
        processor = processor or loaded.processor
        fingerprint = loaded.fingerprint
        source_dir = loaded.path
        readable = loaded.readable
        if readable:
            base_reader = loaded.base_reader
        hashes = False
    else:
        module = model
    ir = _cached_ir(st, fingerprint)
    if ir is None:
        ir = build_ir(
            module,
            example_inputs=example_inputs,
            fingerprint=fingerprint,
            hash_params=hashes is None,
            vision_inputs=probe_inputs(processor, next(module.parameters()).device),
        )
        _save_ir(st, ir)
    return Lab(
        module,
        ir,
        st,
        session=session,
        branch=branch,
        tokenizer=tokenizer,
        config=config or WspxConfig.discover(workdir),
        base_reader=base_reader,
        workdir=Path(workdir) if workdir else None,
        processor=processor,
        source_dir=source_dir,
    )


def _ir_path(fingerprint: str) -> str:
    return f"models/{fingerprint.replace(':', '_')}/ir.json"


def _cached_ir(store: Store, fingerprint: str | None) -> ModelIR | None:
    if fingerprint is None:
        return None
    raw = store.read_file(_ir_path(fingerprint))
    if raw is None:
        return None
    import json

    from wspx.lab.ir.model_ir import IR_VERSION

    data = json.loads(raw)
    if data.get("version") != IR_VERSION:
        return None  # captured by an older wspx: rebuild it
    return ModelIR.from_dict(data)


def _save_ir(store: Store, ir: ModelIR) -> None:
    import json

    store.write_file(_ir_path(ir.fingerprint), json.dumps(ir.to_dict()).encode())
