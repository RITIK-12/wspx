"""Result sets: the full ranked list of a grep, persisted as Parquet, plus small views.

The driver sees the top hits; the store keeps every score. The UI gets
server-side aggregates (a histogram, per-block maxima, a downsampled heatmap)
as a few KB of JSON. Parquet never goes to the browser.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from wspx.lab.grep import GrepResult, Hit

SCHEMA = pa.schema(
    [
        ("rank", pa.int64()),
        ("address", pa.string()),
        ("param", pa.string()),
        ("expert", pa.int64()),
        ("unit", pa.int64()),
        ("score", pa.float64()),
        ("hit", pa.bool_()),
    ]
)


def save(result: GrepResult, path: Path, *, top: int | None = None) -> None:
    """Write every scored unit; ``hit`` marks the matches that were returned (top-k)."""
    returned = {
        (h.param, h.expert, h.unit)
        for h in result.hits[: top if top is not None else len(result.hits)]
    }
    rows = result.scores
    table = pa.table(
        {
            "rank": list(range(len(rows))),
            "address": [h.address for h in rows],
            "param": [h.param for h in rows],
            "expert": [-1 if h.expert is None else h.expert for h in rows],
            "unit": [h.unit for h in rows],
            "score": [h.score for h in rows],
            "hit": [(h.param, h.expert, h.unit) in returned for h in rows],
        },
        schema=SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def load_hits(path: Path, *, only_hits: bool = True) -> list[Hit]:
    table = pq.read_table(path).to_pylist()
    return [
        Hit(
            r["address"],
            r["param"],
            None if r["expert"] < 0 else r["expert"],
            r["unit"],
            r["score"],
        )
        for r in table
        if r["hit"] or not only_hits
    ]


def indices(hits: list[Hit], param: str, expert: int | None = None) -> list[int]:
    """Unit indices of ``param`` (and expert) among ``hits``, in rank order."""
    return [h.unit for h in hits if h.param == param and (expert is None or h.expert == expert)]


def views(
    result: GrepResult, block_of: dict[str, int | None], *, bins: int = 50, width: int = 64
) -> dict[str, Any]:
    scores = [h.score for h in result.scores if not math.isnan(h.score)]
    if not scores:
        return {"hist": [], "per_block_max": {}, "heatmap": []}
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    counts = [0] * bins
    for s in scores:
        counts[min(bins - 1, int((s - lo) / span * bins))] += 1
    per_block: dict[str, float] = {}
    rows: dict[str, list[float]] = defaultdict(list)
    for h in result.scores:
        block = block_of.get(h.param)
        key = str(block) if block is not None else h.param
        if not math.isnan(h.score):
            per_block[key] = max(per_block.get(key, -math.inf), h.score)
            rows[key].append(h.score)
    heatmap = []
    for key in sorted(rows, key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else 0, k)):
        values = rows[key]
        step = max(1, math.ceil(len(values) / width))
        heatmap.append(
            {"row": key, "cells": [max(values[i : i + step]) for i in range(0, len(values), step)]}
        )
    return {
        "hist": {"lo": lo, "hi": hi, "counts": counts},
        "per_block_max": per_block,
        "heatmap": heatmap,
    }
