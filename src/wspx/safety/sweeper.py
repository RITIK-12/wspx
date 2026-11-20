"""The sweeper: terminates wspx sandboxes that hold no valid lease."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from wspx.safety import TAG_KEY, TAG_VALUE


class SandboxLike(Protocol):
    object_id: str

    def get_tags(self) -> dict[str, str]: ...

    def terminate(self) -> Any: ...


@dataclass
class SweepReport:
    checked: int
    terminated: list[str]
    kept: list[str]


def sweep(
    sandboxes: Iterable[SandboxLike],
    lease_of: Callable[[str], float | None],
    *,
    now: float | None = None,
    grace: float = 30.0,
) -> SweepReport:
    """Terminate every wspx sandbox whose lease is missing or expired (plus a little grace)."""
    now = now or time.time()
    terminated, kept = [], []
    checked = 0
    for sandbox in sandboxes:
        tags = sandbox.get_tags() or {}
        if tags.get(TAG_KEY) != TAG_VALUE:
            continue
        checked += 1
        expires = lease_of(sandbox.object_id)
        if expires is None or expires + grace < now:
            sandbox.terminate()
            terminated.append(sandbox.object_id)
        else:
            kept.append(sandbox.object_id)
    return SweepReport(checked, terminated, kept)


def modal_sweep(
    *, app_name: str = "wspx", environment: str = "wspx"
) -> SweepReport:  # pragma: no cover - needs Modal
    import modal

    from wspx.safety import LEASE_DICT

    app = modal.App.lookup(app_name, environment_name=environment)
    leases = modal.Dict.from_name(LEASE_DICT, environment_name=environment, create_if_missing=True)

    def lease_of(sandbox_id: str) -> float | None:
        value = leases.get(sandbox_id)
        return float(value) if value is not None else None

    boxes: list[Any] = list(modal.Sandbox.list(app_id=app.app_id, tags={TAG_KEY: TAG_VALUE}))
    return sweep(boxes, lease_of)
