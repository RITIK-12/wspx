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
        self.fail: Exception | None = None
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
        if self.fail is not None:
            raise self.fail
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


def test_a_failed_start_fails_the_calls_queued_behind_it(tmp_path: Path) -> None:
    """Seen live: three calls queued on a lab that never got a GPU started three sandboxes."""
    m, backend = manager(tmp_path)
    backend.fail = RuntimeError("no GPU available")
    sid = m.create(model="m").id
    assert backend.starting.wait(5)
    results: list[Any] = []
    queued = [
        threading.Thread(target=lambda: results.append(m.tool(sid, "info", {}))) for _ in range(2)
    ]
    for t in queued:
        t.start()
    time.sleep(0.05)
    backend.ready.set()  # the start now fails
    for t in queued:
        t.join(timeout=5)
    assert backend.labs_started == 1
    assert [r["ok"] for r in results] == [False, False]
    assert all("failed to start: no GPU available" in r["summary"] for r in results)
    wait(lambda: m.get(sid).status == "error")
    assert m.tool(sid, "info", {}, source="human")["ok"] is False  # a new call tries again
    assert backend.labs_started == 2


def test_a_driver_session_is_starting_until_its_lab_is_up(tmp_path: Path) -> None:
    m, backend = manager(tmp_path)
    sid = m.create(model="m", driver="anthropic:x").id
    assert m.get(sid).status == "starting"  # the driver is queued, not waiting for a press
    assert m.create(model="m").status == "idle"  # no driver: nothing to start
    backend.ready.set()
    wait(lambda: m.get(sid).status == "running")
    assert len(backend.runners) == 1


class SilentLab(FakeLab):
    def __init__(self) -> None:
        self.stopped = False

    def heartbeat(self, lease_s: float) -> dict[str, Any]:
        raise TimeoutError("The read operation timed out")

    def stop(self) -> None:
        self.stopped = True


def test_a_lab_is_given_up_on_after_several_missed_heartbeats(tmp_path: Path) -> None:
    """One slow heartbeat dropped a live lab, which kept its GPU while a second one started."""
    from wspx.control.sessions import MISSED_HEARTBEATS

    m, _ = manager(tmp_path)
    sid = m.create(model="m").id
    lab = SilentLab()
    m.labs[sid] = lab
    for _ in range(MISSED_HEARTBEATS - 1):
        m.tick()
        assert m.labs.get(sid) is lab  # a missed beat or two is a busy lab, not a lost one
    m.tick()
    assert sid not in m.labs
    assert lab.stopped  # terminated, not just forgotten
    assert m.get(sid).tier == "cold"


# --------------------------------------------------------------------------- experiment guards


def test_denied_tools_are_refused_for_the_driver_and_recorded(tmp_path: Path) -> None:
    m, backend = manager(tmp_path)
    backend.ready.set()
    with pytest.raises(WspxError, match="no tool 'browse'"):
        m.create(model="x", tools_deny=["browse"])
    record = m.create(model="x", driver="anthropic:claude-opus-5-5", tools_deny=["run", "export"])
    m.lab(record.id)
    out = m.tool(record.id, "run", {"script": "a.py"}, source="driver")
    assert not out["ok"]
    assert "turned off" in out["summary"]
    assert m.tool(record.id, "run", {"script": "a.py"}, source="human")["ok"]
    assert m.tool(record.id, "info", {}, source="driver")["ok"]
    calls = [e for e in m.trace.events(record.id) if e["kind"] == "tool_call"]
    assert any(e["payload"]["tool"] == "run" and not e["payload"]["ok"] for e in calls)
    m.start_driver(record.id)
    created = next(e for e in m.trace.events(record.id) if e["kind"] == "session")
    assert created["payload"]["tools_deny"] == ["export", "run"]


def test_gpu_time_runs_out_and_stops_the_driver_and_lab(tmp_path: Path) -> None:
    m, backend = manager(tmp_path)
    backend.ready.set()
    record = m.create(model="x", driver="anthropic:claude-opus-5-5", max_lab_s=600)
    m.lab(record.id)
    m.start_driver(record.id)
    assert "[GPU time left: 10 min of 10 min]" in m.tool(record.id, "info", {})["summary"]
    assert record.lab_since is not None
    m.tick(now=record.lab_since + 300)
    assert record.id in m.labs
    m.tick(now=record.lab_since + 601)
    assert record.id not in m.labs
    assert record.status == "stopped"
    assert "GPU time" in record.reason
    assert not backend.runners[-1].running
    assert record.lab_seconds >= 600
    with pytest.raises(WspxError, match="used its GPU time"):
        m.lab(record.id)


def test_gpu_time_adds_up_over_labs(tmp_path: Path) -> None:
    m, backend = manager(tmp_path)
    backend.ready.set()
    record = m.create(model="x", max_lab_s=3600)
    m.lab(record.id)
    record.lab_since = time.time() - 100  # this lab ran 100 s
    m.stop_lab(record.id, "test")
    assert 99 < record.lab_seconds < 110
    assert record.lab_since is None
    m.lab(record.id)
    record.lab_since = time.time() - 50
    assert 149 < record.lab_used(time.time()) < 165
    stop = [
        e for e in m.trace.events(record.id) if e["kind"] == "lab" and "gpu_seconds" in e["payload"]
    ]
    assert stop
    assert stop[0]["payload"]["gpu_seconds"] >= 99


def test_the_lab_network_check_is_recorded(tmp_path: Path) -> None:
    class Checked(FakeLab):
        def heartbeat(self, lease_s: float) -> dict[str, Any]:
            return {"network": "blocked"}

    m, backend = manager(tmp_path)
    backend.ready.set()
    backend.start_lab = lambda record, cancelled: Checked()  # type: ignore[method-assign]
    record = m.create(model="x")
    m.lab(record.id)
    hot = [
        e for e in m.trace.events(record.id) if e["kind"] == "lab" and e["payload"]["tier"] == "hot"
    ]
    assert hot[-1]["payload"]["network"] == "blocked"
