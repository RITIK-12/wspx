"""Edits in a quantized checkpoint's own format: what a runtime serves.

transformers dequantizes some checkpoints when it loads them (Inkling's NVFP4 and MXFP8
experts), so the lab holds bf16 values while vLLM serves the stored codes. For each edited
tensor stored quantized, :class:`Served` writes the edit into the stored format
(``quantize(dequantize(W) + Δ)`` on the blocks the edit touches; :mod:`wspx.lab.quant`)
and gives the lab what loading that checkpoint gives, so tests see the edit as it will be
served. Each edit gets a survival report: how much of it the format kept.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from wspx.lab import checkpoint, quant


@dataclass
class Survival:
    """How much of an edit a quantized format kept."""

    tensor: str
    format: str
    ratio: float
    """‖served change‖ / ‖intended change‖."""
    cosine: float
    """Cosine between the served change and the intended one."""
    blocks: int
    values: int
    clipped: int
    rescaled: int = 0
    kept: float = 0.0
    """The share of the intended change the served one keeps along it (``ratio × cosine``):
    rounding noise from re-encoding a block adds size, not this."""

    def describe(self) -> str:
        line = (
            f"{self.kept:.0%} of the edit survives {self.format} (the served change is "
            f"{self.ratio:.0%} of its size, cosine {self.cosine:.3f}); {self.values:,} stored "
            f"values in {self.blocks:,} blocks changed"
        )
        if self.rescaled:
            line += f"; {self.rescaled} expert(s) outgrew their scale and were re-encoded whole"
        if self.clipped:
            line += f", {self.clipped:,} clipped at the block's largest value"
        return line

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Served:
    """The source checkpoint's quantized tensors, and the edited ones in their format."""

    model: torch.nn.Module
    src: checkpoint.Source
    keep_scales: bool = False
    formats: dict[str, quant.Format] = field(default_factory=dict)
    parts: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    """In-memory name → its checkpoint tensors in the stored format, for edited tensors."""
    last: dict[str, Survival] = field(default_factory=dict)
    _names: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def open(
        cls, model: torch.nn.Module, source: Path | None, *, keep_scales: bool = False
    ) -> Served | None:
        """A :class:`Served` when ``source`` stores tensors quantized that the model holds
        dequantized; ``None`` otherwise."""
        src = checkpoint.read_source(source)
        if src is None or getattr(model, "hf_quantizer", None) is not None:
            return None
        rules: dict[str, str] = {}
        qc = src.path / "hf_quant_config.json"
        if qc.is_file():
            rules = quant.rules_from(json.loads(qc.read_text()))
        formats = quant.formats(src.tensors, rules)
        if not formats:
            return None
        return cls(model, src, keep_scales, formats)

    def checkpoint_names(self, name: str, shape: tuple[int, ...]) -> list[str]:
        if name not in self._names:
            self._names[name] = checkpoint.names_in_checkpoint(self.model, name, shape)
        return self._names[name]

    def quantized(self, name: str, shape: tuple[int, ...]) -> bool:
        return any(n in self.formats for n in self.checkpoint_names(name, shape))

    def _read(self, names: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        from safetensors import safe_open

        out: dict[str, torch.Tensor] = {}
        for fname in sorted(self.src.weight_files):
            with safe_open(str(self.src.path / fname), framework="pt") as fh:
                for key in names:
                    if key not in out and key in fh.keys():  # noqa: SIM118
                        out[key] = fh.get_tensor(key).to(device)
        return out

    def apply(self, name: str, base: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """What loading the checkpoint with this edit written in its format gives ``name``;
        the stored tensors and the survival report are kept for export and vLLM."""
        encoded = self.encode(name, base, value)
        if encoded is None:
            self.parts.pop(name, None)
            self.last.pop(name, None)
            return base
        parts, survival, live = encoded
        self.parts[name] = parts
        self.last[name] = survival
        return live

    def encode(
        self, name: str, base: torch.Tensor, value: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], Survival, torch.Tensor] | None:
        """``value`` written in the stored format (changing nothing here): the checkpoint
        tensors, how much of ``value - base`` survived, and what loading them gives.
        ``None`` when ``value`` equals ``base``."""
        delta = value.float() - base.float()
        if not bool(delta.ne(0).any()):
            return None
        ck_delta = checkpoint.to_checkpoint(self.model, {name: delta})
        plain = checkpoint.to_checkpoint(self.model, {name: value})
        parts: dict[str, torch.Tensor] = {}
        blocks = values = clipped = rescaled = 0
        kinds = set()
        for ck, d in ck_delta.items():
            fmt = self.formats.get(ck)
            if fmt is None:
                parts[ck] = plain[ck]
                continue
            stored = self._read(fmt.parts, d.device)
            done = quant.requantize(fmt, stored, d, keep_scales=self.keep_scales)
            parts.update(done.parts)
            blocks, values, clipped = (
                blocks + done.blocks,
                values + done.values,
                clipped + done.clipped,
            )
            rescaled += done.rescaled
            kinds.add(fmt.kind.upper())
        live = checkpoint.from_checkpoint(self.model, parts)[name].to(base.dtype)
        served = live.float() - base.float()
        norm = float(delta.norm())
        ratio = float(served.norm()) / norm if norm else 0.0
        cos = float(
            (served.flatten() @ delta.flatten()) / (served.norm() * delta.norm()).clamp_min(1e-30)
        )
        cos = 0.0 if math.isnan(cos) else cos
        survival = Survival(
            name,
            "/".join(sorted(kinds)),
            ratio,
            cos,
            blocks,
            values,
            clipped,
            rescaled,
            ratio * cos,
        )
        return parts, survival, live

    def stored(self, name: str) -> dict[str, torch.Tensor] | None:
        """The checkpoint tensors ``name`` is stored as with its edit, or ``None`` when it is
        unedited (the source's bytes are current)."""
        return self.parts.get(name)

    def source_parts(
        self, name: str, shape: tuple[int, ...], device: torch.device
    ) -> dict[str, torch.Tensor]:
        """The source's own stored tensors for ``name`` (codes, scales and factors)."""
        wanted: list[str] = []
        for ck in self.checkpoint_names(name, shape):
            fmt = self.formats.get(ck)
            wanted += fmt.parts if fmt else [ck]
        return self._read(wanted, device)


QUANTIZED_DTYPES = ("U8", "I8", "I32", "F8_E4M3", "F8_E5M2", "F4", "U4", "I4")


def unwritable(src: checkpoint.Source | None) -> str:
    """Why wspx cannot write edits in this checkpoint's stored format ("" when it can, or
    when the checkpoint is not quantized)."""
    if src is None or not src.quantized or not src.tensors:
        return ""
    rules: dict[str, str] = {}
    qc = src.path / "hf_quant_config.json"
    if qc.is_file():
        rules = quant.rules_from(json.loads(qc.read_text()))
    formats = quant.formats(src.tensors, rules)
    covered = {part for fmt in formats.values() for part in fmt.parts}
    stray = sorted(
        name
        for name, (dtype, _) in src.tensors.items()
        if name not in covered
        and (dtype in QUANTIZED_DTYPES or checkpoint.QUANT_AUX.search(name) is not None)
    )
    if not stray:
        return ""
    method = (src.quantization or {}).get("quant_method") or "a quantized format"
    return (
        f"its weights are stored {method} in a layout wspx cannot write edits in yet "
        f"({len(stray)} tensor(s), e.g. {stray[0]})"
    )
