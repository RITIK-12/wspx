from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
from starlette.testclient import TestClient

from tests.driver.test_loop import Scripted, call, done
from tests.lab.test_session import tiny_llama
from wspx.auth import mint
from wspx.control.app import ControlSettings, create_app
from wspx.control.backend import LocalBackend, SessionRecord
from wspx.lab.ir.capture import build_ir
from wspx.lab.session import Lab
from wspx.store import Store
from wspx.tools import TOOLS
from wspx.vcs import Repo

TOKEN = "user-token"
SECRET = "shared-secret"
USER = {"authorization": f"Bearer {TOKEN}"}


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LiveServer:
    """The control plane on a real port, so a runner thread talks real HTTP to it."""

    def __init__(self, app: Any, port: int) -> None:
        import threading

        import uvicorn

        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> LiveServer:
        self.thread.start()
        wait(lambda: self.server.started, timeout=20)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def make(
    tmp_path: Path, script: list[Any] | None = None, *, port: int | None = None
) -> tuple[TestClient, Any]:
    root = tmp_path / "store"
    ir_cache: dict[str, Any] = {}

    def lab_factory(record: SessionRecord, store: Any) -> Lab:
        model = tiny_llama()
        if "ir" not in ir_cache:
            ir_cache["ir"] = build_ir(model)
        record.fingerprint = ir_cache["ir"].fingerprint
        from wspx.lab.sdk import _save_ir

        _save_ir(store, ir_cache["ir"])
        return Lab(
            model, ir_cache["ir"], store, session=record.id, branch=record.branch, workdir=tmp_path
        )

    def runner_factory(record: SessionRecord, env: dict[str, str]):  # type: ignore[no-untyped-def]
        from wspx.driver.run import run_from_env

        return lambda: run_from_env(env, provider=Scripted(list(script or [done()])))

    backend = LocalBackend(root, lab_factory=lab_factory, runner_factory=runner_factory)
    url = f"http://127.0.0.1:{port}" if port else "http://testserver"
    settings = ControlSettings(store_root=str(root), api_token=TOKEN, secret=SECRET, public_url=url)
    app = create_app(settings, backend, start_ticker=False)
    client = TestClient(app)
    return client, app.state.manager


def wait(predicate, timeout: float = 30.0) -> None:  # type: ignore[no-untyped-def]
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("timed out")


def test_tools_over_the_api_persist_through_the_journal(tmp_path: Path) -> None:
    client, manager = make(tmp_path)
    with client:
        sid = client.post("/api/sessions", json={"model": "tiny"}, headers=USER).json()["id"]
        wait(lambda: sid in manager.labs)

        def tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
            r = client.post(f"/api/sessions/{sid}/tools/{name}", json={"args": args}, headers=USER)
            assert r.status_code == 200, r.text
            return r.json()

        assert tool("arch", {})["ok"]
        grep = tool("grep", {"expression": "colnorm", "in": "L*.mlp.down", "top": 3})
        assert grep["handles"][0].startswith("@r1")
        assert tool("edit", {"op": "ablate", "args": {"targets": "L0.mlp.down[:, 1]"}})["ok"]
        commit = tool("vcs", {"action": "commit", "message": "ablate a unit"})
        assert commit["ok"], commit["summary"]

        # the control plane wrote the store: the commit is in the real store now
        store = Store(tmp_path / "store")
        repo = Repo(store, manager.get(sid).fingerprint)
        assert repo.log(repo.head(sid))[0].message == "ablate a unit"

        # the call log joins calls to their artefacts
        found = client.get(f"/api/calls/{commit['call_id']}", headers=USER).json()
        assert any(a["kind"] == "commit" for a in found["artefacts"])
        kinds = [
            e["kind"]
            for e in client.get(f"/api/sessions/{sid}/timeline", headers=USER).json()["events"]
        ]
        assert kinds.count("tool_call") == 4
        assert "lab" in kinds

        # a cold lab still answers store-served tools
        manager.stop_lab(sid, "test")
        cold = tool("arch", {})
        assert cold["data"]["served_by"] == "store"
        assert "lab cold" in cold["summary"]
        warm = tool("show", {"address": "L0.mlp.down[:, 1]"})
        assert "lab cold start" in warm["summary"]


def test_scripted_driver_runs_through_the_control_plane(tmp_path: Path) -> None:
    script = [
        call("arch"),
        call("edit", {"op": "ablate", "args": {"targets": "L1.mlp.down[:, 4]"}}, n=1),
        call("vcs", {"action": "commit", "message": "driver edit"}, n=2),
        done("all done"),
    ]
    port = free_port()
    client, manager = make(tmp_path, script, port=port)
    with LiveServer(client.app, port):  # the live server runs the lifespan; the client must not
        sid = client.post(
            "/api/sessions",
            json={"model": "tiny", "driver": "anthropic:claude-opus-5-5", "task": "ablate unit 4"},
            headers=USER,
        ).json()["id"]
        wait(lambda: manager.get(sid).status == "done", timeout=60)
        events = client.get(f"/api/sessions/{sid}/timeline", headers=USER).json()["events"]
        kinds = [e["kind"] for e in events]
        assert kinds.count("turn") == 4
        assert kinds.count("tool_call") == 3
        repo = Repo(Store(tmp_path / "store"), manager.get(sid).fingerprint)
        assert repo.log(repo.head(sid))[0].message == "driver edit"

        # the record starts at the start: what the model was given, then everything after it
        export = client.get(f"/api/sessions/{sid}/export", headers=USER).json()
        assert export["format"] == "wspx-session/1"
        kinds = [e["kind"] for e in export["events"]]
        assert kinds[0] == "session"
        (start,) = [e["payload"] for e in export["events"] if e["kind"] == "start"]
        assert start["system"]
        assert start["message"].startswith("ablate unit 4")
        assert "edit" in start["tools"]
        calls = [e["payload"] for e in export["events"] if e["kind"] == "tool_call"]
        assert calls[1]["data"]["handle"].startswith("p_")  # results in full, data included
        assert [c["message"] for c in export["commits"]["commits"]] == ["driver edit"]


class Preempted(BaseException):
    """The worker vanishing: nothing in the runner catches it."""


def test_a_preempted_runner_resumes_over_the_api(tmp_path: Path) -> None:
    """Seen live: Modal restarted the runner on the same input and it lost its history."""
    from wspx.driver.run import run_from_env

    class DiesAfter(Scripted):
        def __init__(self, script: list[Any], steps: int) -> None:
            super().__init__(script)
            self.steps = steps

        def step(self, *args: Any, **kw: Any) -> Any:
            if len(self.requests) == self.steps:
                raise Preempted
            return super().step(*args, **kw)

    envs: list[dict[str, str]] = []
    edit = {"op": "ablate", "args": {"targets": "L1.mlp.down[:, 4]"}}
    port = free_port()
    client, manager = make(tmp_path, port=port)

    def runner_factory(record: SessionRecord, env: dict[str, str]):  # type: ignore[no-untyped-def]
        envs.append(env)

        def target() -> None:
            with contextlib.suppress(Preempted):
                run_from_env(env, provider=DiesAfter([call("arch"), call("edit", edit, n=1)], 2))

        return target

    manager.backend.runner_factory = runner_factory
    with LiveServer(client.app, port):
        sid = client.post(
            "/api/sessions",
            json={"model": "tiny", "driver": "anthropic:claude-opus-5-5", "task": "ablate unit 4"},
            headers=USER,
        ).json()["id"]
        wait(lambda: bool(envs) and len(manager.trace.calls(sid)) == 2, timeout=60)
        time.sleep(0.5)  # the first runner has died by its third request
        (env,) = envs
        commit = call("vcs", {"action": "commit", "message": "after the restart"}, n=2)
        provider = Scripted([commit, done("resumed")])
        result = run_from_env(env, provider=provider)  # Modal reruns the same input
        assert (result.status, result.final_text) == ("done", "resumed")
        assert provider.requests[0][0].text.startswith("ablate unit 4")
        assert len(provider.requests[0]) == 5  # task, arch and its result, edit and its result
        assert any("restarted" in n for n in provider.requests[0][-1].notes)
        repo = Repo(Store(tmp_path / "store"), manager.get(sid).fingerprint)
        (latest, *_) = repo.log(repo.head(sid))
        assert latest.message == "after the restart"
        assert len(latest.patches) == 1  # the edit, applied once
        kinds = [e["kind"] for e in manager.trace.events(sid)]
        assert kinds.count("start") == 1
        assert kinds.count("resume") == 1
        # another start of the driver is a new runner: it does not pick up this checkpoint
        assert manager.trace.driver_state(sid, "r_other") is None


def test_tokens_are_scoped(tmp_path: Path) -> None:
    client, manager = make(tmp_path)
    with client:
        sid = client.post("/api/sessions", json={"model": "tiny"}, headers=USER).json()["id"]
        wait(lambda: sid in manager.labs)
        assert client.get("/api/sessions").status_code == 401
        mine = {"authorization": f"Bearer {mint(SECRET, sid, scope='runner')}"}
        other = {"authorization": f"Bearer {mint(SECRET, 's_other', scope='runner')}"}
        llm = {"authorization": f"Bearer {mint(SECRET, sid, scope='llm')}"}
        assert client.get(f"/api/sessions/{sid}/commands", headers=mine).status_code == 200
        assert client.get(f"/api/sessions/{sid}/commands", headers=other).status_code == 403
        assert client.get(f"/api/sessions/{sid}/commands", headers=llm).status_code == 401
        assert (
            client.get("/api/limits", headers=mine).status_code == 401
        )  # runners never see limits
        assert client.post("/api/kill", headers=mine).status_code == 401


def test_kill_resume_and_ingest(tmp_path: Path) -> None:
    client, manager = make(tmp_path)
    with client:
        sid = client.post("/api/sessions", json={"model": "tiny"}, headers=USER).json()["id"]
        wait(lambda: sid in manager.labs)
        r = client.post(f"/api/sessions/{sid}/tools/arch", json={"args": {}}, headers=USER).json()
        hook = {
            "hook_event_name": "PostToolUse",
            "tool_name": "mcp__wspx__arch",
            "tool_response": [{"type": "text", "text": f"… [call {r['call_id']}]"}],
        }
        joined = client.post(
            f"/api/ingest/claude-code?session={sid}", json=hook, headers=USER
        ).json()
        assert joined["joined"] == r["call_id"]
        assert client.post("/api/kill", json={}, headers=USER).json()["stopped"] == 1
        assert manager.halted
        assert not manager.labs
        assert client.post("/api/sessions", json={"model": "tiny"}, headers=USER).status_code == 400
        client.post("/api/resume", json={}, headers=USER)
        assert not manager.halted


def test_mcp_endpoint(tmp_path: Path) -> None:
    client, manager = make(tmp_path)
    with client:
        sid = client.post("/api/sessions", json={"model": "tiny"}, headers=USER).json()["id"]
        wait(lambda: sid in manager.labs)
        headers = {
            **USER,
            "accept": "application/json, text/event-stream",
            "x-wspx-session": sid,
            "content-type": "application/json",
        }
        assert (
            client.post(
                "/mcp/", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
            ).status_code
            == 401
        )
        init = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        }
        client.post("/mcp/", json=init, headers=headers)
        listed = client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers=headers,
        )
        assert listed.status_code == 200, listed.text
        names = [t["name"] for t in listed.json()["result"]["tools"]]
        assert "grep" in names
        assert len(names) == len(TOOLS)
        res = client.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "arch", "arguments": {}},
            },
            headers=headers,
        ).json()
        text = res["result"]["content"][0]["text"]
        assert "LlamaForCausalLM" in text
        assert "[call c_" in text
        assert json.loads(json.dumps(res))["result"]["isError"] is False


def test_ui_views_are_served_from_the_store(tmp_path: Path) -> None:
    client, manager = make(tmp_path)
    with client:
        sid = client.post("/api/sessions", json={"model": "tiny"}, headers=USER).json()["id"]
        wait(lambda: sid in manager.labs)

        def tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
            return client.post(
                f"/api/sessions/{sid}/tools/{name}", json={"args": args}, headers=USER
            ).json()

        tool("edit", {"op": "ablate", "args": {"targets": "L0.mlp.down[:, 1]"}})
        tool("vcs", {"action": "commit", "message": "first"})
        tool("test", {"suite": "canary"})
        trace = tool("trace", {"prompt": [1, 2, 3]})
        manager.stop_lab(sid, "views must not need the lab")

        arch = client.get(f"/api/sessions/{sid}/architecture", headers=USER).json()
        assert arch["available"]
        assert len(arch["blocks"]) == 3
        assert any(r["address"] == "L0.mlp.down" for r in arch["blocks"][0]["roles"])
        history = client.get(f"/api/sessions/{sid}/commits", headers=USER).json()
        assert history["commits"][0]["message"] == "first"
        assert history["commits"][0]["patches"][0]["op"] == "ablate"
        matrix = client.get(f"/api/sessions/{sid}/runs", headers=USER).json()
        canary = next(r for r in matrix["runs"] if r["suite"] == "canary")
        assert canary["commits"] == [history["commits"][0]["short"]]
        assert canary["checks"]
        handles = client.get(f"/api/sessions/{sid}/handles", headers=USER).json()["handles"]
        ref = next(h["ref"] for h in handles if h["key"] == trace["handles"][0])
        obj = client.get(f"/api/objects/{ref}", headers=USER).json()
        assert obj["kind"] == "forward"
        assert obj["arrays"]["lens_ids"]
        assert sid not in manager.labs


def test_projects_ship_their_files_and_settings_to_the_lab(tmp_path: Path) -> None:
    from wspx.config import WspxConfig
    from wspx.control.backend import project_workdir
    from wspx.project import Project

    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    (root / "wspx.toml").write_text('[project]\nname = "demo"\n\n[check]\nmax_rel_norm = 0.01\n')
    (root / "data" / "loops.jsonl").write_text('{"prompt": "Wait"}\n')
    found = Project.find(root)
    assert found is not None
    packed = found.pack()
    gzip = USER | {"content-type": "application/gzip"}

    client, manager = make(tmp_path)
    with client:
        up = client.post("/api/projects", content=packed, headers=gzip)
        assert up.status_code == 200, up.text
        assert up.json()["files"] == 2
        snapshot = up.json()["snapshot"]
        again = client.post("/api/projects", content=packed, headers=gzip).json()
        assert again["snapshot"] == snapshot  # stored once, by content
        assert client.post("/api/projects", content=b"junk", headers=gzip).status_code == 400

        unknown = {"model": "tiny", "snapshot": "sha256:" + "0" * 64}
        assert client.post("/api/sessions", json=unknown, headers=USER).status_code == 400
        body = {"model": "tiny", "project": "demo", "snapshot": snapshot}
        sid = client.post("/api/sessions", json=body, headers=USER).json()["id"]
        record = manager.get(sid)
        assert (record.project, record.snapshot) == ("demo", snapshot)
        listed = client.get("/api/sessions", headers=USER).json()["sessions"]
        assert [s["project"] for s in listed] == ["demo"]

    workdir = project_workdir(record, Store(tmp_path / "store"))
    assert workdir is not None
    assert (workdir / "data" / "loops.jsonl").read_text() == '{"prompt": "Wait"}\n'
    assert WspxConfig.discover(workdir).check.max_rel_norm == 0.01


def test_each_tool_call_is_counted_once_for_loop_detection(tmp_path: Path) -> None:
    """The gateway sees the built-in driver's calls in its responses; the control plane
    counts only other callers', so A, B, A is not mistaken for a loop."""
    seen: list[tuple[str, str]] = []

    class Gateway:
        def observe(self, sid: str, kind: str, key: Any, label: str = "") -> None:
            seen.append((kind, label))

    client, manager = make(tmp_path)
    with client:
        sid = client.post("/api/sessions", json={"model": "tiny"}, headers=USER).json()["id"]
        wait(lambda: sid in manager.labs)
        manager.gateway = Gateway()
        manager.tool(sid, "arch", {}, source="driver")
        assert ("call", "arch") not in seen
        manager.tool(sid, "arch", {}, source="mcp")
        assert seen.count(("call", "arch")) == 1


def test_lookups_that_miss_between_ones_that_match_are_not_a_loop(tmp_path: Path) -> None:
    """Exploring a new model, an agent tries names that do not exist (E101) between many
    that do; only the same diagnostic on calls in a row is a loop. One no-op edit after
    another is still caught."""
    from wspx.loops import LoopDetector

    detector = LoopDetector(threshold=3)
    actions: list[str] = []

    class Gateway:
        def observe(self, sid: str, kind: str, key: Any, label: str = "") -> Any:
            verdict = detector.observe(sid, kind, key, label=label)
            if verdict is None:
                return None
            actions.append(verdict.action)
            return verdict.to_dict()

        def pause(self, sid: str, reason: str) -> None:
            pass

    client, manager = make(tmp_path)
    with client:
        sid = client.post("/api/sessions", json={"model": "tiny"}, headers=USER).json()["id"]
        wait(lambda: sid in manager.labs)
        manager.gateway = Gateway()  # type: ignore[assignment]
        for _ in range(4):
            assert not manager.tool(sid, "arch", {"scope": "no.such.weights*"}, source="driver")[
                "ok"
            ]
            assert manager.tool(sid, "arch", {"scope": "L0.*"}, source="driver")["ok"]
        assert actions == []
        for _ in range(3):
            manager.tool(sid, "arch", {"scope": "no.such.weights*"}, source="driver")
        assert actions == ["nudge"]
        no_op = {"op": "scale", "args": {"targets": "L0.mlp.down", "factor": 1.0}}
        for _ in range(3):
            assert "E105" in manager.tool(sid, "edit", no_op, source="driver")["summary"]
        assert actions == ["nudge", "nudge"]


def test_lab_failures_are_results_not_crashes() -> None:
    import httpx

    from wspx.control.backend import _post_tool
    from wspx.errors import LabUnavailable

    failing = httpx.Client(
        base_url="http://lab",
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")),
    )
    result = _post_tool(failing, "edit", {}, "c_1", {})
    assert result["ok"] is False
    assert "HTTP 500" in result["summary"]

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    gone = httpx.Client(base_url="http://lab", transport=httpx.MockTransport(refuse))
    with pytest.raises(LabUnavailable):
        _post_tool(gone, "edit", {}, "c_1", {})


def test_the_open_model_trace_is_kept_for_the_user_only(tmp_path: Path) -> None:
    """What the open model was given and wrote is the user's record (held-out items
    included): it is stored as its own trace and never returned to the caller."""
    client, manager = make(tmp_path)
    with client:
        sid = client.post("/api/sessions", json={"model": "tiny"}, headers=USER).json()["id"]
        wait(lambda: sid in manager.labs)
        for source in ("driver", "mcp"):
            out = manager.tool(
                sid, "generate", {"prompt": [1, 2, 3], "max_tokens": 3}, source=source
            )
            assert out["ok"], out["summary"]
            assert "model_io" not in out
            assert "model_io" not in json.dumps(out)
        manager.tool(sid, "arch", {}, source="driver")  # no prompt to the model: no record
        events = client.get(f"/api/sessions/{sid}/export", headers=USER).json()["events"]
        traced = [e for e in events if e["kind"] == "model_io"]
        assert [e["source"] for e in traced] == ["open_model", "open_model"]
        assert [e["payload"]["tool"] for e in traced] == ["generate", "generate"]
        (record,) = traced[0]["payload"]["records"]
        assert (record["kind"], record["prompt"], record["output_tokens"]) == (
            "generate",
            "token ids [1, 2, 3]",
            3,
        )


def test_the_ui_gets_a_sessions_prompt_images_and_nothing_else(tmp_path: Path) -> None:
    """Images a prompt named are served to the user from the project snapshot, unpacked once;
    other files, paths outside the project and other callers are refused."""
    from wspx.project import Project

    root = tmp_path / "proj"
    (root / "prompts").mkdir(parents=True)
    (root / "wspx.toml").write_text('[project]\nname = "demo"\n')
    png = b"\x89PNG\r\n\x1a\n" + b"\0" * 64
    (root / "prompts" / "a.png").write_bytes(png)
    found = Project.find(root)
    assert found is not None
    client, _ = make(tmp_path)
    with client:
        gzip = USER | {"content-type": "application/gzip"}
        snapshot = client.post("/api/projects", content=found.pack(), headers=gzip).json()[
            "snapshot"
        ]
        sid = client.post(
            "/api/sessions",
            json={"model": "tiny", "project": "demo", "snapshot": snapshot},
            headers=USER,
        ).json()["id"]
        bare = client.post("/api/sessions", json={"model": "tiny"}, headers=USER).json()["id"]
        url = f"/api/sessions/{sid}/media"
        got = client.get(url, params={"path": "prompts/a.png"}, headers=USER)
        assert got.status_code == 200, got.text
        assert got.content == png
        assert got.headers["content-type"] == "image/png"
        assert "immutable" in got.headers["cache-control"]
        refused = {
            "../store/trace.db": 400,  # outside the project
            "/etc/passwd": 400,
            "wspx.toml": 400,  # not an image or video
            "prompts/missing.png": 404,
        }
        for path, status in refused.items():
            assert client.get(url, params={"path": path}, headers=USER).status_code == status, path
        assert client.get(url, params={"path": "prompts/a.png"}).status_code == 401  # no token
        runner = {"authorization": f"Bearer {mint(SECRET, sid, scope='runner')}"}
        assert client.get(url, params={"path": "prompts/a.png"}, headers=runner).status_code == 401
        nothing = client.get(f"/api/sessions/{bare}/media", params={"path": "a.png"}, headers=USER)
        assert nothing.status_code == 404


def test_snapshots_are_unpacked_once_and_the_oldest_are_dropped(tmp_path: Path) -> None:
    import io
    import tarfile

    from wspx.control import media

    def snapshot(name: str) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            data = name.encode()
            info = tarfile.TarInfo("prompts/a.png")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        return buf.getvalue()

    cache = media.MediaCache(tmp_path / "cache")
    reads: list[str] = []

    def loader(name: str) -> Any:
        return lambda: (reads.append(name), snapshot(name))[1]

    for _ in range(3):
        path, kind = cache.file(loader("one"), "sha256:one", "prompts/a.png")
    assert (path.read_text(), kind, reads) == ("one", "image/png", ["one"])
    for i in range(media.KEEP + 2):
        cache.file(loader(f"s{i}"), f"sha256:s{i}", "prompts/a.png")
    kept = [p for p in (tmp_path / "cache").iterdir() if (p / ".unpacked").exists()]
    assert len(kept) == media.KEEP
