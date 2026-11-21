"""The op graph: a framework-neutral record of one profiling pass.

Each node is one executed ATen op, tagged with the innermost module that ran
it. Activation tensors become numbered values; parameters never do. A node
lists the parameters it read (directly, through views, or through tensors
computed only from parameters) together with which parameter dims it
contracted, so every weight's wiring into every op is explicit.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any

INPUT = -1
"""Producer id of a value that entered the graph from outside (input ids, masks)."""


@dataclass(slots=True)
class ParamUse:
    name: str
    arg: int
    """Position among the node's flattened tensor arguments."""
    rows: list[int] | None = None
    """Leading-dim indices read (expert ids for 3-D expert weights); ``None`` = whole tensor."""
    in_dim: int | None = None
    """Parameter dim contracted against the activation (matmul-like ops only)."""
    out_dim: int | None = None
    """Parameter dim that becomes the op's output features."""


@dataclass(slots=True)
class Node:
    idx: int
    op: str
    module: str
    args: list[int]
    """Value id per flattened tensor argument; ``-1`` for parameters, buffers and constants."""
    params: list[ParamUse]
    outs: list[int]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def short_op(self) -> str:
        """``aten.mm.default`` → ``mm``."""
        parts = self.op.split(".")
        return parts[1] if len(parts) > 1 else parts[0]

    @property
    def activation_args(self) -> list[tuple[int, int]]:
        """``(position, value id)`` for each activation argument."""
        return [(pos, vid) for pos, vid in enumerate(self.args) if vid >= 0]


@dataclass(slots=True)
class Value:
    id: int
    shape: tuple[int, ...]
    dtype: str
    producer: int


@dataclass
class OpGraph:
    phase: str
    nodes: list[Node] = field(default_factory=list)
    values: dict[int, Value] = field(default_factory=dict)
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    """Module calls in order: modules that own parameters, and the children of module lists
    (a layer stack). A layer that runs three times appears three times."""

    @cached_property
    def consumers(self) -> dict[int, list[tuple[int, int]]]:
        """value id → ``[(node idx, arg position), …]`` in execution order."""
        out: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for node in self.nodes:
            for pos, vid in node.activation_args:
                out[vid].append((node.idx, pos))
        return dict(out)

    def producer(self, vid: int) -> Node | None:
        value = self.values.get(vid)
        if value is None or value.producer < 0:
            return None
        return self.nodes[value.producer]

    def params_read(self) -> set[str]:
        return {use.name for node in self.nodes for use in node.params}

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "nodes": [asdict(n) for n in self.nodes],
            "values": [asdict(v) for v in self.values.values()],
            "inputs": self.inputs,
            "outputs": self.outputs,
            "calls": self.calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpGraph:
        nodes = [
            Node(
                idx=n["idx"],
                op=n["op"],
                module=n["module"],
                args=list(n["args"]),
                params=[ParamUse(**p) for p in n["params"]],
                outs=list(n["outs"]),
                meta=n.get("meta", {}),
            )
            for n in data["nodes"]
        ]
        values = {
            v["id"]: Value(v["id"], tuple(v["shape"]), v["dtype"], v["producer"])
            for v in data["values"]
        }
        return cls(
            data["phase"],
            nodes,
            values,
            list(data["inputs"]),
            list(data["outputs"]),
            list(data.get("calls", [])),
        )


def compress_indices(indices: list[int]) -> list[list[int]]:
    """Encode sorted unique ints as arithmetic runs ``[start, stop, step]``.

    ``[0..63]`` → ``[[0, 64, 1]]``; ``[0, 2, 4, …]`` → ``[[0, 2n, 2]]``.
    """
    runs: list[list[int]] = []
    i, n = 0, len(indices)
    while i < n:
        start = indices[i]
        if i + 1 == n:
            runs.append([start, start + 1, 1])
            break
        step = indices[i + 1] - start
        j = i + 1
        while j + 1 < n and indices[j + 1] - indices[j] == step:
            j += 1
        if j == i + 1 and i + 2 < n and step != 1:
            # A lone pair with an odd gap is better as two singletons than a run.
            runs.append([start, start + 1, 1])
            i += 1
            continue
        runs.append([start, indices[j] + 1, step])
        i = j + 1
    return runs


def expand_runs(runs: list[list[int]]) -> list[int]:
    return [i for start, stop, step in runs for i in range(start, stop, step)]
