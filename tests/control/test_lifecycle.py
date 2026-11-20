"""Session lifecycle races: Start while the lab loads, stop while it starts, halt across restarts."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from wspx.control.backend import SessionRecord
from wspx.control.gateway_client import GatewayAdmin
from wspx.control.sessions import SessionManager
from wspx.control.tracedb import TraceDB
from wspx.errors import LabCancelled, WspxError
from wspx.store import Store


class FakeRunner:
    def __init__(self) -> None:
        self.running = True

    def alive(self) -> bool:
        return self.running

    def stop(self) -> None:
        self.running = False


class FakeLab:
    def call(
        self, tool: str, args: dict[str, Any], call_id: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        return {"summary": f"{tool} ok", "ok": True, "handles": [], "data": {}}

    def heartbeat(self, lease_s: float) -> dict[str, Any]:
        return {}

    def stop(self) -> None:
        pass


class SlowBackend:
    """Labs start only once ``ready`` is set, like a sandbox pulling its image."""

    def __init__(self) -> None:
        self.ready = threading.Event()
        self.starting = threading.Event()
        self.labs_started = 0
        self.runners: list[FakeRunner] = []

    def start_lab(self, record: SessionRecord, cancelled: Any) -> FakeLab:
        self.labs_started += 1
        self.starting.set()
        while not self.ready.wait(0.01):
            if cancelled():
                raise LabCancelled(f"{record.id}'s lab was stopped while it started")
        return FakeLab()

    def start_runner(self, record: SessionRecord, env: dict[str, str]) -> FakeRunner:
        runner = FakeRunner()
        self.runners.append(runner)
        return runner

    def stop_all(self) -> int:
        return 0


def manager(tmp_path: Path, gateway: Any = None) -> tuple[SessionManager, SlowBackend]:
    backend = SlowBackend()
    m = SessionManager(
        Store(tmp_path / "store"), TraceDB(tmp_path / "trace.db"), backend, gateway=gateway
    )
    return m, backend


def wait(predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline, "timed out"
        time.sleep(0.01)


def test_start_while_the_lab_loads_runs_one_driver(tmp_path: Path) -> None:
    m, backend = manager(tmp_path)
    sid = m.create(model="m", driver="anthropic:x").id
    assert backend.starting.wait(5)
    m.start_driver(sid)  # the UI's Start, pressed while the lab is still loading
    backend.ready.set()
    wait(lambda: sid in m.labs)
    m.start_driver(sid)
    time.sleep(0.05)  # the prewarm thread's own start_driver has run by now
    assert len(backend.runners) == 1


def test_resume_replaces_a_driver_that_is_still_winding_down(tmp_path: Path) -> None:
    m, backend = manager(tmp_path)
    backend.ready.set()
    sid = m.create(model="m", driver="anthropic:x").id
    wait(lambda: len(backend.runners) == 1)
    m.pause(sid, "paused by user")
    m.resume(sid)
    assert [r.running for r in backend.runners] == [False, True]


def test_stop_cancels_a_lab_that_is_starting(tmp_path: Path) -> None:
    m, backend = manager(tmp_path)
    sid = m.create(model="m", driver="anthropic:x").id
    assert backend.starting.wait(5)

    stopper = threading.Thread(target=m.kill, args=(sid,))
    stopper.start()
    stopper.join(timeout=5)
    assert not stopper.is_alive(), "stop waited for the lab to finish starting"
    time.sleep(0.05)
    record = m.get(sid)
    assert (record.status, record.tier) == ("stopped", "cold")  # not an error: it was asked for
    assert sid not in m.labs
    assert not backend.runners  # the prewarm did not go on to start the driver

    late = m.tool(sid, "info", {}, source="driver")  # a driver call still in flight
    assert not late["ok"]
    assert "stopped" in late["summary"]
    assert backend.labs_started == 1

    backend.ready.set()
    assert m.tool(sid, "info", {}, source="human")["ok"]  # a person can still wake it
    assert backend.labs_started == 2


def test_calls_queued_behind_a_cancelled_start_do_not_restart_the_lab(tmp_path: Path) -> None:
    m, backend = manager(tmp_path)
    sid = m.create(model="m").id
    assert backend.starting.wait(5)
    results: list[Any] = []
    queued = threading.Thread(target=lambda: results.append(m.tool(sid, "info", {})))
    queued.start()
    time.sleep(0.05)  # the call waits on the lab start
    m.stop_lab(sid, "stopped")
    queued.join(timeout=5)
    assert not results[0]["ok"]
    assert "stopped" in results[0]["summary"]
    assert backend.labs_started == 1


class FakeGateway:
    def __init__(self, halted: bool | Exception, during: Any = None) -> None:
        self._halted = halted
        self.during = during

    def halted(self) -> bool:
        if self.during is not None:
            self.during()
        if isinstance(self._halted, Exception):
            raise self._halted
        return self._halted

    def resume(self, session: str | None = None) -> None:
        pass


def test_halt_survives_a_control_plane_restart(tmp_path: Path) -> None:
    m, _ = manager(tmp_path, gateway=FakeGateway(True))
    assert not m.halted
    m.sync_halt()
    assert m.halted
    with pytest.raises(WspxError, match="halted"):
        m.create(model="m")

    m, _ = manager(tmp_path, gateway=FakeGateway(WspxError("gateway unreachable")))
    m.sync_halt()  # an unreachable gateway leaves the state as it was
    assert not m.halted


def test_a_resume_during_the_halt_read_wins(tmp_path: Path) -> None:
    """A resume that wakes both services: the gateway answers the read first, but resume is newer."""
    gateway = FakeGateway(True)
    m, _ = manager(tmp_path, gateway=gateway)
    gateway.during = m.resume
    m.sync_halt()
    assert not m.halted


def test_gateway_admin_reads_the_halt_state() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/healthz"
        return httpx.Response(200, json={"ok": True, "halted": True})

    admin = GatewayAdmin("http://gw", "t", transport=httpx.MockTransport(handler))
    assert admin.halted()

    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>stopped</html>")

    admin = GatewayAdmin("http://gw", "t", transport=httpx.MockTransport(broken))
    with pytest.raises(WspxError, match="not JSON"):
        admin.halted()
