"""The lab's HTTP server: a heartbeat must never wait behind a running tool call."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from tests.lab.test_session import tiny_llama
from wspx.lab import server as lab_server
from wspx.lab.sdk import open_lab
from wspx.lab.session import ToolResult


def test_a_heartbeat_answers_while_a_tool_call_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seen live: a 45 s test run held the lock, the heartbeat timed out, the lab was dropped."""
    entered, release = threading.Event(), threading.Event()

    def slow_call(lab: object, name: str, args: object) -> ToolResult:
        entered.set()
        release.wait(10)
        return ToolResult(f"{name} done")

    monkeypatch.setattr(lab_server, "call", slow_call)
    lab = open_lab(tiny_llama(), store=tmp_path / "s", workdir=tmp_path)
    with TestClient(lab_server.LabServer(lab, exit_on_lapse=False).app()) as client:
        results: list[dict[str, object]] = []
        worker = threading.Thread(
            target=lambda: results.append(client.post("/tool/test", json={"args": {}}).json())
        )
        worker.start()
        assert entered.wait(5)
        started = time.time()
        beat = client.post("/heartbeat", json={"lease_s": 300}).json()
        assert time.time() - started < 2
        assert beat["busy"] is True
        assert beat["lease_expires"] > time.time() + 250  # renewed all the same
        release.set()
        worker.join(5)
        assert results[0]["summary"] == "test done"
        assert "busy" not in client.post("/heartbeat", json={"lease_s": 300}).json()
