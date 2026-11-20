"""Where labs and runners run: in-process (local) or on Modal (cloud)."""

from __future__ import annotations

import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from wspx.errors import LabUnavailable


@dataclass
class SessionRecord:
    id: str
    model: str
    driver: str | None = None
    task: str = ""
    gpu: str | None = None
    branch: str | None = None
    revision: str | None = None
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    status: str = "idle"
    """idle · running · paused · stopped · done · error"""
    tier: str = "cold"
    """hot · grace · cold"""
    grace_s: float = 900.0
    keep_warm: str = "auto"
    project: str = ""
    snapshot: str | None = None
    """Object id of the project snapshot the lab unpacks as its working directory."""
    created: float = 0.0
    fingerprint: str | None = None
    reason: str = ""
    load_seconds: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class LabHandle(Protocol):
    def call(
        self, tool: str, args: dict[str, Any], call_id: str, context: dict[str, Any]
    ) -> dict[str, Any]: ...

    def heartbeat(self, lease_s: float) -> dict[str, Any]: ...

    def stop(self) -> None: ...


class RunnerHandle(Protocol):
    def alive(self) -> bool: ...

    def stop(self) -> None: ...


class Backend(Protocol):
    def start_lab(self, record: SessionRecord) -> LabHandle: ...

    def start_runner(self, record: SessionRecord, env: dict[str, str]) -> RunnerHandle: ...

    def stop_all(self) -> int: ...


class _Poster(Protocol):
    def post(self, url: str, *, json: Any) -> Any: ...


def _post_tool(
    client: _Poster, tool: str, args: dict[str, Any], call_id: str, context: dict[str, Any]
) -> dict[str, Any]:
    """One tool call to a lab server. A failure inside the lab comes back as a failed result."""
    try:
        r = client.post(
            f"/tool/{tool}", json={"args": args, "call_id": call_id, "context": context}
        )
    except httpx.HTTPError as exc:
        raise LabUnavailable(f"the lab did not answer: {exc}") from exc
    if r.status_code >= 400:
        return {
            "summary": f"{tool}: the lab failed (HTTP {r.status_code}): {r.text[:300]}",
            "ok": False,
            "handles": [],
            "data": {},
        }
    data: dict[str, Any] = r.json()
    return data


class HttpLab:
    """A lab reached over HTTP (a sandbox's encrypted port and connect token)."""

    def __init__(
        self,
        url: str,
        token: str | None,
        *,
        lab_token: str | None = None,
        on_stop: Callable[[], None] | None = None,
        timeout: float = 3600.0,
    ) -> None:
        headers = {"authorization": f"Bearer {token}"} if token else {}
        if lab_token:
            headers["x-wspx-lab-token"] = lab_token
        self.client = httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=timeout)
        self.on_stop = on_stop

    def call(
        self, tool: str, args: dict[str, Any], call_id: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        return _post_tool(self.client, tool, args, call_id, context)

    def heartbeat(self, lease_s: float) -> dict[str, Any]:
        r = self.client.post("/heartbeat", json={"lease_s": lease_s}, timeout=30)
        r.raise_for_status()
        data: dict[str, Any] = r.json()
        return data

    def stop(self) -> None:
        if self.on_stop is not None:
            self.on_stop()
        self.client.close()


class InProcessLab:
    """A lab in this process, driven through the same server code and journal as the cloud."""

    def __init__(self, server: Any) -> None:
        from starlette.testclient import TestClient

        self.server = server
        self.client = TestClient(server.app())

    def call(
        self, tool: str, args: dict[str, Any], call_id: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        return _post_tool(self.client, tool, args, call_id, context)

    def heartbeat(self, lease_s: float) -> dict[str, Any]:
        data: dict[str, Any] = self.client.post("/heartbeat", json={"lease_s": lease_s}).json()
        return data

    def stop(self) -> None:
        self.client.close()


class ThreadRunner:
    def __init__(self, target: Callable[[], Any]) -> None:
        self._stop = threading.Event()
        self.thread = threading.Thread(target=target, name="wspx-runner", daemon=True)
        self.thread.start()

    def alive(self) -> bool:
        return self.thread.is_alive()

    def stop(self) -> None:
        self._stop.set()


class LocalBackend:
    """Labs and runners in this process: for development, tests and single-box self-hosting.

    ``lab_factory`` builds the Lab for a record (default: load the model with
    the HF loader). The lab still sees a read-only view of the store through an
    overlay, so the journal path is exercised exactly as in the cloud.
    """

    def __init__(
        self,
        store_root: str | Path,
        *,
        lab_factory: Callable[[SessionRecord, Any], Any] | None = None,
        runner_factory: Callable[[SessionRecord, dict[str, str]], Callable[[], Any]] | None = None,
    ) -> None:
        self.store_root = Path(store_root)
        self.lab_factory = lab_factory
        self.runner_factory = runner_factory
        self.labs: list[InProcessLab] = []
        self.runners: list[ThreadRunner] = []

    def start_lab(self, record: SessionRecord) -> LabHandle:
        from wspx.lab.server import LabServer
        from wspx.store.overlay import OverlayStore

        overlay = OverlayStore(self.store_root, tempfile.mkdtemp(prefix=f"wspx-{record.id}-"))
        lab = (
            self.lab_factory(record, overlay)
            if self.lab_factory is not None
            else _load_lab(record, overlay)
        )
        handle = InProcessLab(LabServer(lab, exit_on_lapse=False))
        self.labs.append(handle)
        return handle

    def start_runner(self, record: SessionRecord, env: dict[str, str]) -> RunnerHandle:
        if self.runner_factory is None:
            from wspx.driver.run import run_from_env

            def target() -> Any:
                return run_from_env(env)
        else:
            target = self.runner_factory(record, env)
        runner = ThreadRunner(target)
        self.runners.append(runner)
        return runner

    def stop_all(self) -> int:
        for lab in self.labs:
            lab.stop()
        n = len(self.labs)
        self.labs.clear()
        return n


def _load_lab(record: SessionRecord, store: Any) -> Any:
    from wspx.config import WspxConfig
    from wspx.lab.ir.capture import build_ir
    from wspx.lab.loader import load_hf
    from wspx.lab.sdk import _cached_ir, _save_ir
    from wspx.lab.session import Lab

    workdir = project_workdir(record, store)
    loaded = load_hf(
        record.model,
        revision=record.revision,
        dtype=record.dtype,
        trust_remote_code=record.trust_remote_code,
    )
    ir = _cached_ir(store, loaded.fingerprint)
    if ir is None:
        ir = build_ir(loaded.model, fingerprint=loaded.fingerprint, hash_params=False)
        _save_ir(store, ir)
    record.fingerprint = loaded.fingerprint
    return Lab(
        loaded.model,
        ir,
        store,
        session=record.id,
        branch=record.branch,
        tokenizer=loaded.tokenizer,
        base_reader=loaded.base_reader if loaded.readable else None,
        workdir=workdir,
        config=WspxConfig.discover(workdir) if workdir else None,
    )


def project_workdir(record: SessionRecord, store: Any) -> Path | None:
    """Unpack the session's project snapshot into a fresh working directory, if it has one."""
    if not record.snapshot:
        return None
    from wspx.project import extract

    return extract(
        store.objects.get(record.snapshot), tempfile.mkdtemp(prefix=f"wspx-{record.id}-work-")
    )
