"""Build a Model IR from a live module: profiling passes, coverage and semantics (§5.1.1).

Language models get a prefill pass (batch 2, 16 tokens, one row left-padded)
and a decode step against the KV cache, so ops that differ between the two
phases are both recorded. MoE routers are forced to round-robin assignments,
with extra passes until every expert row has been read. A vision-language
model also gets one pass on an image prompt (``vision_inputs``), so its vision
tower counts as exercised; that graph is used for coverage only. Any parameter
that never appears in a graph is reported as unexercised and gets no guessed role.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch

from wspx.errors import ModelLoadError
from wspx.lab.ir.graph import OpGraph
from wspx.lab.ir.model_ir import Coverage, ModelIR, ParamInfo
from wspx.lab.ir.semantics import RecurrentLayout, Semantics, derive
from wspx.lab.ir.structure import build_structure, structural_semantics
from wspx.lab.ir.tape import Tape, _iter_tensors
from wspx.lab.tensors import dtype_name, tensor_hash

PREFILL_TOKENS = 16
BATCH = 2
LEFT_PAD = 3
MAX_ROUTING_PASSES = 64


def build_ir(
    model: torch.nn.Module,
    *,
    example_inputs: Any = None,
    fingerprint: str | None = None,
    param_hashes: Mapping[str, str] | None = None,
    force_routing: bool = True,
    seed: int = 0,
    hash_params: bool = True,
    vision_inputs: Mapping[str, Any] | None = None,
) -> ModelIR:
    """Capture ``model`` and derive its IR.

    ``example_inputs`` (a tuple of positional args or a dict of keyword args) is
    required for models that are neither causal LMs nor Hugging Face vision models.
    ``vision_inputs`` are a vision-language model's processed image prompt (see
    ``media.probe_inputs``).
    """
    model.eval()
    params, canonical = _param_table(model)
    recurrent = _recurrent_layouts(model, canonical)
    is_lm = _is_causal_lm(model)
    graphs: dict[str, OpGraph] = {}
    rows: dict[str, set[int]] = {}
    full: set[str] = set()
    vision_read: set[str] = set()
    vision_graph: OpGraph | None = None
    passes = 0
    gen = torch.Generator().manual_seed(seed)
    device = _device_of(model)

    def absorb(tape: Tape) -> None:
        for name, seen in tape.rows_read.items():
            rows.setdefault(name, set()).update(seen)
        full.update(tape.full_read)

    with torch.no_grad():
        if is_lm:
            vocab = _vocab_size(model)
            assert vocab is not None
            ids = torch.randint(0, vocab, (BATCH, PREFILL_TOKENS), generator=gen).to(device)
            mask = torch.ones(BATCH, PREFILL_TOKENS, dtype=torch.long, device=device)
            mask[1, :LEFT_PAD] = 0
            tape = Tape(model, "prefill", force_routing=force_routing)
            _generate_two(model, tape, ids, mask)
            graphs.update(tape.graphs)
            absorb(tape)
            routers = dict(tape.routers)
            tape.release()
            passes += len(tape.graphs)

            # Extra prefill passes rotate the forced assignment until every expert is hit.
            if force_routing and routers:
                step = min(r["slots"] for r in routers.values())
                for i in range(1, MAX_ROUTING_PASSES):
                    if not _row_deficit(rows, full, params):
                        break
                    tape = Tape(
                        model,
                        "routing",
                        force_routing=True,
                        route_offset=i * step,
                        track_features=False,
                    )
                    with tape.recording():
                        model(input_ids=ids, attention_mask=mask, use_cache=False)
                    absorb(tape)
                    tape.release()
                    passes += 1
        else:
            inputs = (
                example_inputs
                if example_inputs is not None
                else _default_inputs(model, gen, device)
            )
            tape = Tape(model, "forward", force_routing=force_routing)
            with tape.recording():
                out = _call(model, inputs)
                tape.mark_outputs(_primary_output(out))
            graphs["forward"] = tape.graph
            absorb(tape)
            tape.release()
            passes += 1
        if vision_inputs is not None:
            tape = Tape(model, "vision", force_routing=False, track_features=False)
            with tape.recording():
                model(**vision_inputs, use_cache=False)
            vision_read = set(tape.graph.params_read())
            vision_graph = tape.graph
            absorb(tape)
            tape.release()
            passes += 1

    main = graphs.get("prefill") or graphs["forward"]
    shapes = {name: info.shape for name, info in params.items()}
    module_of = {name: info.module for name, info in params.items()}
    try:
        traced = derive(main, shapes, module_of=module_of, recurrent=recurrent)
    except Exception as exc:  # the structure below does not depend on it
        traced = Semantics(notes=[f"the traced graph gave no short names ({type(exc).__name__})"])
    structure = build_structure(
        model,
        params,
        canonical,
        graphs,
        fallback_embed=traced.embed,
        fallback_unembed=traced.unembed,
        fact_graphs={"vision": vision_graph} if vision_graph is not None else None,
    )
    semantics = structural_semantics(traced, structure, params)

    coverage: dict[str, Coverage] = {}
    for name, info in params.items():
        phases = [phase for phase, g in graphs.items() if name in g.params_read()]
        if name in vision_read:
            phases.append("vision")
        read = bool(phases) or name in full or name in rows
        cov = Coverage(exercised=read, phases=phases)
        if name in rows and name not in full and info.shape:
            cov.rows_seen = len(rows[name])
            cov.rows_total = info.shape[0]
            cov.exercised = read and cov.rows_seen == cov.rows_total
        coverage[name] = cov

    if param_hashes is not None:
        hashes = dict(param_hashes)
    elif hash_params:
        hashes = {name: tensor_hash(p) for name, p in model.named_parameters()}
    else:
        hashes = {}  # computed lazily, before a tensor's first edit
        if fingerprint is None:
            raise ModelLoadError("a fingerprint is required when parameters are not hashed")
    for name, info in params.items():
        info.hash = hashes.get(name)
    if fingerprint is None:
        digest = hashlib.sha256()
        digest.update(type(model).__qualname__.encode())
        for name in sorted(hashes):
            digest.update(f"{name}={hashes[name]}\n".encode())
        fingerprint = "fp:" + digest.hexdigest()

    config = getattr(model, "config", None)
    labels: dict[int, str] = {}
    id2label = getattr(config, "id2label", None) if config is not None and not is_lm else None
    if isinstance(id2label, dict) and not all(
        str(v).startswith("LABEL_") for v in id2label.values()
    ):
        labels = {int(k): str(v) for k, v in id2label.items()}

    return ModelIR(
        fingerprint=fingerprint,
        arch=type(model).__name__,
        params=params,
        semantics=semantics,
        graphs=graphs,
        coverage=coverage,
        modules={path: type(m).__name__ for path, m in model.named_modules()},
        is_lm=is_lm,
        vocab_size=_vocab_size(model) if is_lm else None,
        labels=labels,
        capture={"method": "tape", "passes": passes, "batch": BATCH, "tokens": PREFILL_TOKENS},
        structure=structure,
    )


def _generate_two(
    model: torch.nn.Module, tape: Tape, ids: torch.Tensor, mask: torch.Tensor
) -> None:
    """Prefill plus one decode step, through the model's own generation code.

    ``generate`` knows every cache type (KV, sliding window, SSM state), so this
    exercises the real decode path. Glue between forward calls is not recorded.
    """
    calls = 0

    def pre(_module: torch.nn.Module, _args: Any, _kwargs: Any) -> None:
        nonlocal calls
        if calls == 1:
            tape.start_phase("decode")
        tape.active = calls < 2
        calls += 1

    def post(_module: torch.nn.Module, _args: Any, _kwargs: Any, out: Any) -> None:
        if tape.active:
            tape.mark_outputs(_logits(out))
        tape.active = False

    h_pre = model.register_forward_pre_hook(pre, with_kwargs=True)
    h_post = model.register_forward_hook(post, with_kwargs=True)
    config = getattr(model, "generation_config", None)
    pad = getattr(config, "pad_token_id", None)
    if pad is None:
        pad = 0
    tape.active = False  # only forward calls are recorded
    try:
        with tape.recording():
            model.generate(  # type: ignore[operator]
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=2,
                min_new_tokens=2,
                do_sample=False,
                pad_token_id=pad,
                use_cache=True,
            )
    finally:
        h_pre.remove()
        h_post.remove()
        tape.active = True


def _param_table(model: torch.nn.Module) -> tuple[dict[str, ParamInfo], dict[int, str]]:
    owners = {}
    for path, module in model.named_modules(remove_duplicate=False):
        for leaf, _ in module.named_parameters(recurse=False, remove_duplicate=False):
            owners[f"{path}.{leaf}" if path else leaf] = (path, type(module).__name__)
    params: dict[str, ParamInfo] = {}
    canonical: dict[int, str] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        key = id(param)
        if key in canonical:
            params[canonical[key]].aliases.append(name)
            continue
        canonical[key] = name
        module, cls = owners.get(name, (name.rpartition(".")[0], ""))
        params[name] = ParamInfo(name, tuple(param.shape), dtype_name(param.dtype), module, cls)
    return params, canonical


def _recurrent_layouts(
    model: torch.nn.Module, canonical: dict[int, str]
) -> dict[str, RecurrentLayout]:
    out: dict[str, RecurrentLayout] = {}
    for path, module in model.named_modules():
        if isinstance(module, torch.nn.RNNBase):
            flat = [canonical[id(getattr(module, n))] for n in module._flat_weights_names]
            out[path] = RecurrentLayout(module.mode, module.num_layers, module.bidirectional, flat)
    return out


def _row_deficit(rows: dict[str, set[int]], full: set[str], params: dict[str, ParamInfo]) -> bool:
    return any(
        name not in full and len(seen) < params[name].shape[0]
        for name, seen in rows.items()
        if name in params and params[name].shape
    )


def _is_causal_lm(model: torch.nn.Module) -> bool:
    try:
        from transformers import GenerationMixin
    except ImportError:  # pragma: no cover - transformers is a lab dependency
        return False
    config = getattr(model, "config", None)
    return (
        isinstance(model, GenerationMixin)
        and config is not None
        and not getattr(config, "is_encoder_decoder", False)
        and _vocab_size(model) is not None
    )


def _vocab_size(model: torch.nn.Module) -> int | None:
    """The vocabulary size; multimodal configs keep it in their text config."""
    config = getattr(model, "config", None)
    if config is None:
        return None
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    size = getattr(text, "vocab_size", None) or getattr(config, "vocab_size", None)
    return int(size) if size else None


def _default_inputs(model: torch.nn.Module, gen: torch.Generator, device: torch.device) -> Any:
    config = getattr(model, "config", None)
    channels = getattr(config, "num_channels", None)
    if channels:
        size = getattr(config, "image_size", None) or 32
        size = size if isinstance(size, int) else size[0]
        return {"pixel_values": torch.randn(BATCH, channels, size, size, generator=gen).to(device)}
    raise ModelLoadError(
        f"{type(model).__name__}: pass example inputs for profiling (a tuple of args or a dict)"
    )


def _call(model: torch.nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, tuple | list):
        return model(*inputs)
    return model(inputs)


def _logits(out: Any) -> Any:
    return getattr(out, "logits", out)


def _primary_output(out: Any) -> Any:
    """The model's main output: ``logits`` if present, else the first tensor."""
    logits = getattr(out, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    for tensor in _iter_tensors(out):
        return tensor
    return out


def _device_of(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")
