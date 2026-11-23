"""The tier-0 index (§5.2.3): computed from weights alone, once per fingerprint.

* per-tensor statistics and a 10-bin histogram;
* per-unit norms and absmax along each weight's unit axis;
* top singular values of every matrix (and of every expert matrix);
* vocabulary projections: the tokens each writer unit writes most and each
  reader unit reads most (final-norm and pre-norm scales folded in).

It answers ``arch``, ``show --stats/--vocab`` and stat greps while the
lab is cold, and annotates grep hits. At commit time the lab refreshes the
entries of the tensors it touched.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from wspx.lab.grep import Environment, unit_batches
from wspx.lab.ir.model_ir import ModelIR

SPECTRUM_K = 16
VOCAB_K = 10
MAX_SVD_NUMEL = 64 * 1024 * 1024
LOGIT_BUDGET = 1 << 26  # elements per vocab-projection chunk


@dataclass
class TensorStats:
    shape: list[int]
    dtype: str
    norm: float
    mean: float
    std: float
    absmax: float
    kurtosis: float
    zeros: float
    nan: int
    inf: int
    hist_edges: list[float]
    hist_counts: list[int]
    spectrum: list[float] = field(default_factory=list)


def _histogram(values: torch.Tensor, lo: float, hi: float, bins: int) -> torch.Tensor:
    """Exact integer counts. ``torch.histc`` counts in the input's float dtype, which loses
    counts once a bin passes 2^24 (an embedding matrix has hundreds of millions of values)."""
    counts = torch.zeros(bins, dtype=torch.int64, device=values.device)
    width = (hi - lo) / bins
    for chunk in torch.split(values, 1 << 24):
        index = ((chunk - lo) / width).floor_().long().clamp_(0, bins - 1)
        counts += torch.bincount(index, minlength=bins)
    return counts


def tensor_stats(t: torch.Tensor, *, spectrum: bool = True) -> TensorStats:
    x = t.detach().float()
    flat = x.reshape(-1)
    finite = flat[torch.isfinite(flat)]
    mean = float(finite.mean()) if finite.numel() else float("nan")
    std = float(finite.std(correction=0)) if finite.numel() > 1 else 0.0
    kurt = (
        float(((finite - mean) ** 4).mean() / max(std, 1e-30) ** 4 - 3)
        if finite.numel() > 1
        else 0.0
    )
    lo, hi = (float(finite.min()), float(finite.max())) if finite.numel() else (0.0, 0.0)
    if hi == lo:
        hi = lo + 1e-12
    counts = _histogram(finite, lo, hi, 10)
    edges = [lo + (hi - lo) * i / 10 for i in range(11)]
    spec: list[float] = []
    if spectrum and x.dim() >= 2:
        mat = x[0] if x.dim() >= 3 else x  # expert weights: the first expert's spectrum
        mat = mat.reshape(mat.shape[0], -1)
        if mat.numel() <= MAX_SVD_NUMEL and torch.isfinite(mat).all():
            spec = [float(v) for v in torch.linalg.svdvals(mat)[:SPECTRUM_K]]
    return TensorStats(
        shape=list(t.shape),
        dtype=str(t.dtype).removeprefix("torch."),
        norm=float(finite.norm()),
        mean=mean,
        std=std,
        absmax=float(finite.abs().max()) if finite.numel() else 0.0,
        kurtosis=kurt,
        zeros=float((flat == 0).float().mean()),
        nan=int(torch.isnan(flat).sum()),
        inf=int(torch.isinf(flat).sum()),
        hist_edges=edges,
        hist_counts=[int(c) for c in counts.tolist()],
        spectrum=spec,
    )


@dataclass
class ModelIndex:
    fingerprint: str
    tensors: dict[str, TensorStats] = field(default_factory=dict)
    units: dict[str, dict[str, Any]] = field(default_factory=dict)
    """param → {"axis", "address", "norm", "absmax", "top_ids", "top_scores"} (lists per unit)."""

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tensors.json").write_text(
            json.dumps(
                {
                    "fingerprint": self.fingerprint,
                    "tensors": {k: asdict(v) for k, v in self.tensors.items()},
                }
            )
        )
        rows: dict[str, list[Any]] = {
            k: []
            for k in (
                "param",
                "expert",
                "unit",
                "address",
                "norm",
                "absmax",
                "top_ids",
                "top_scores",
            )
        }
        for param, data in self.units.items():
            n = len(data["norm"])
            rows["param"].extend([param] * n)
            rows["expert"].extend(data["expert"])
            rows["unit"].extend(data["unit"])
            rows["address"].extend(data["address"])
            rows["norm"].extend(data["norm"])
            rows["absmax"].extend(data["absmax"])
            rows["top_ids"].extend(data.get("top_ids") or [[] for _ in range(n)])
            rows["top_scores"].extend(data.get("top_scores") or [[] for _ in range(n)])
        pq.write_table(pa.table(rows), directory / "units.parquet", compression="zstd")

    @classmethod
    def load(cls, directory: Path) -> ModelIndex:
        meta = json.loads((directory / "tensors.json").read_text())
        index = cls(meta["fingerprint"], {k: TensorStats(**v) for k, v in meta["tensors"].items()})
        path = directory / "units.parquet"
        if path.exists():
            for row in pq.read_table(path).to_pylist():
                entry = index.units.setdefault(
                    row["param"],
                    {
                        k: []
                        for k in (
                            "expert",
                            "unit",
                            "address",
                            "norm",
                            "absmax",
                            "top_ids",
                            "top_scores",
                        )
                    },
                )
                for key in entry:
                    entry[key].append(row[key])
        return index

    def unit_tokens(
        self, param: str, unit: int, expert: int | None = None
    ) -> list[tuple[int, float]]:
        data = self.units.get(param)
        if not data:
            return []
        for i, (e, u) in enumerate(zip(data["expert"], data["unit"], strict=True)):
            if u == unit and (expert is None or e == expert):
                return list(
                    zip(data["top_ids"][i] or [], data["top_scores"][i] or [], strict=False)
                )
        return []


def build_index(
    ir: ModelIR,
    params: dict[str, torch.Tensor],
    *,
    names: list[str] | None = None,
    vocab: bool = True,
    existing: ModelIndex | None = None,
) -> ModelIndex:
    """Compute (or refresh, for ``names``) the tier-0 index."""
    index = existing or ModelIndex(ir.fingerprint)
    names = names if names is not None else list(ir.params)
    env = Environment(ir, params)
    with torch.no_grad():
        for name in names:
            index.tensors[name] = tensor_stats(params[name])
            role = ir.primary_role(name)
            if role is None or len(ir.params[name].shape) < 2:
                index.units.pop(name, None)
                continue
            targets = list(ir.resolve(name))
            entry: dict[str, list[Any]] = {
                k: []
                for k in ("expert", "unit", "address", "norm", "absmax", "top_ids", "top_scores")
            }
            for batch in unit_batches(ir, params, targets, None):
                v = batch.vectors
                entry["expert"].extend([-1 if batch.expert is None else batch.expert] * v.shape[0])
                entry["unit"].extend(batch.globals)
                entry["address"].extend(batch.address(i) for i in range(v.shape[0]))
                entry["norm"].extend(v.norm(dim=1).tolist())
                entry["absmax"].extend(v.abs().amax(dim=1).tolist())
                ids, scores = _vocab_top(env, role.kind, v) if vocab else (None, None)
                entry["top_ids"].extend(ids or [[] for _ in range(v.shape[0])])
                entry["top_scores"].extend(scores or [[] for _ in range(v.shape[0])])
            index.units[name] = entry
    return index


def _vocab_top(
    env: Environment, kind: str, vectors: torch.Tensor
) -> tuple[list[list[int]] | None, list[list[float]] | None]:
    sem = env.ir.semantics
    writes = kind in ("writer", "embed", "pos_embed", "stem", "expert")
    name = sem.unembed if writes else sem.embed
    if name is None:
        return None, None
    table = env.params[name].detach().float()
    role = next(
        (r for r in env.ir.roles_of(name) if r.kind == ("unembed" if writes else "embed")), None
    )
    vocab_dim = 0 if role is None or role.kind == "embed" else (role.out_dim or 0)
    table = table if vocab_dim == 0 else table.T
    if vectors.shape[1] != table.shape[1]:
        return None, None
    if writes:
        scale = env.final_scale()
        if scale is not None and scale.numel() == vectors.shape[1]:
            vectors = vectors * scale.to(vectors.device)
    chunk = max(1, LOGIT_BUDGET // max(table.shape[0], 1))
    ids: list[list[int]] = []
    scores: list[list[float]] = []
    table = table.to(vectors.device)
    for start in range(0, vectors.shape[0], chunk):
        logits = vectors[start : start + chunk] @ table.T
        top = torch.topk(logits, k=min(VOCAB_K, logits.shape[1]), dim=1)
        ids.extend(top.indices.tolist())
        scores.extend(top.values.tolist())
    return ids, scores
