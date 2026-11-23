"""The control plane's HTTP app: REST API, SSE timelines, hook ingest, MCP and the UI.

Who may call what:

* the **user token** (CLI, UI, MCP clients) may do anything the user can;
* a **runner token** may only call tools, post events and read commands and
  the workspace for its own session;
* nobody but the user reaches limits, kill and resume. No agent tool exposes them.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

from wspx import __version__
from wspx import project as project_snapshot
from wspx.auth import TokenError, check_bearer, verify
from wspx.control.backend import Backend
from wspx.control.gateway_client import GatewayAdmin
from wspx.control.mcp_server import build_server
from wspx.control.media import MediaCache
from wspx.control.sessions import SessionManager
from wspx.control.tracedb import TraceDB
from wspx.errors import WspxError
from wspx.store import Store

PROVIDER_CAPS = {
    "anthropic": "https://platform.claude.com/settings/limits",
    "openai": "https://platform.openai.com/settings/organization/limits",
    "google": "https://aistudio.google.com/spend",
    "modal": "https://modal.com/settings/usage",
}
_CALL_RE = re.compile(r"\[call (c_[0-9a-f]+)\]")


@dataclass
class ControlSettings:
    store_root: str
    api_token: str
    secret: str
    """Shared with the gateway; signs runner and LLM session tokens."""
    gateway_url: str = ""
    gateway_admin_token: str = ""
    public_url: str = ""
    tick_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> ControlSettings:
        return cls(
            store_root=os.environ.get("WSPX_STORE", "/store"),
            api_token=os.environ["WSPX_API_TOKEN"],
            secret=os.environ["WSPX_GATEWAY_SECRET"],
            gateway_url=os.environ.get("WSPX_GATEWAY_URL", ""),
            gateway_admin_token=os.environ.get("WSPX_ADMIN_TOKEN", ""),
            public_url=os.environ.get("WSPX_PUBLIC_URL", ""),
        )


class NewSession(BaseModel):
    model: str
    driver: str | None = None
    task: str = ""
    gpu: str | None = None
    branch: str | None = None
    revision: str | None = None
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    grace_s: float = Field(900.0, ge=0)
    keep_warm: str = "auto"
    project: str = ""
    snapshot: str | None = Field(None, pattern=r"^sha256:[0-9a-f]{64}$")
    tools_deny: list[str] = Field(default_factory=list)
    max_lab_s: float | None = Field(None, gt=0)


class ToolCall(BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None
    turn: int | None = None
    source: str = "driver"


def create_app(
    settings: ControlSettings,
    backend: Backend,
    *,
    gateway: GatewayAdmin | None = None,
    start_ticker: bool = True,
) -> FastAPI:
    store = Store(settings.store_root)
    trace = TraceDB(Path(settings.store_root) / "trace.db")
    media_cache = MediaCache()
    if gateway is None and settings.gateway_url and settings.gateway_admin_token:
        gateway = GatewayAdmin(settings.gateway_url, settings.gateway_admin_token)
    manager = SessionManager(
        store,
        trace,
        backend,
        gateway=gateway,
        secret=settings.secret,
        control_url=settings.public_url,
        gateway_url=settings.gateway_url,
    )

    def bearer(request: Request) -> str | None:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:]
        return request.query_params.get("token")

    def user(request: Request) -> None:
        if not check_bearer(settings.api_token, bearer(request)):
            raise HTTPException(401, "user token required")

    def session_caller(request: Request) -> str:
        """``"user"`` or ``"runner"``; a runner token only works for its own session."""
        token = bearer(request)
        if check_bearer(settings.api_token, token):
            return "user"
        try:
            claims = verify(settings.secret, token or "", scope="runner")
        except TokenError:
            raise HTTPException(401, "token required") from None
        if claims.session != request.path_params.get("sid"):
            raise HTTPException(403, "this runner token is for another session")
        return "runner"

    def guarded(fn: Callable[..., Any]) -> Any:
        try:
            return fn()
        except WspxError as exc:
            raise HTTPException(400, str(exc)) from None

    async def offload(fn: Callable[..., Any]) -> Any:
        """``guarded`` on a worker thread: for async endpoints whose work blocks (Modal, I/O)."""
        return await run_in_threadpool(guarded, fn)

    # ------------------------------------------------------------------ MCP

    def mcp_call(session: str | None, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if session is None:
            running = [
                r.id
                for r in manager.records.values()
                if r.status in ("starting", "idle", "running")
            ]
            if len(running) != 1:
                return {
                    "summary": "pick a session: set the X-Wspx-Session header (see `wspx connect`)",
                    "ok": False,
                }
            session = running[0]
        try:
            return manager.tool(session, name, args, source="mcp")
        except WspxError as exc:
            return {"summary": str(exc), "ok": False}

    from mcp.server.transport_security import TransportSecuritySettings

    mcp_server = build_server(mcp_call)
    mcp_inner = mcp_server.streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    async def mcp_app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            auth = headers.get("authorization", "")
            if not check_bearer(
                settings.api_token, auth[7:] if auth.lower().startswith("bearer ") else None
            ):
                response = JSONResponse({"error": "user token required"}, status_code=401)
                await response(scope, receive, send)
                return
        await mcp_inner(scope, receive, send)

    # ------------------------------------------------------------------ lifecycle

    stop = threading.Event()

    def ticker() -> None:
        manager.sync_halt()  # here, not at import: a cold gateway must not delay startup
        while not stop.wait(settings.tick_seconds):
            try:
                manager.tick()
            except Exception as exc:
                print(f"wspx control: tick failed: {exc}", flush=True)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async with mcp_inner.router.lifespan_context(mcp_inner):
            thread = threading.Thread(target=ticker, daemon=True) if start_ticker else None
            if thread is not None:
                thread.start()
            yield
            stop.set()

    app = FastAPI(
        title="wspx", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None
    )
    app.state.manager = manager
    app.state.trace = trace
    app.mount("/mcp", mcp_app)

    # ------------------------------------------------------------------ sessions

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "halted": manager.halted}

    @app.get("/api/sessions", dependencies=[Depends(user)])
    def list_sessions() -> dict[str, Any]:
        return {"sessions": [r.to_dict() for r in manager.records.values()]}

    @app.post("/api/sessions", dependencies=[Depends(user)])
    def new_session(body: NewSession) -> dict[str, Any]:
        record = guarded(lambda: manager.create(**body.model_dump()))
        return record.to_dict()  # type: ignore[no-any-return]

    @app.post("/api/projects", dependencies=[Depends(user)])
    async def upload_project(request: Request) -> dict[str, Any]:
        """Store a project snapshot (a gzipped tar from ``wspx.project``) by content hash."""
        data = await request.body()
        if len(data) > project_snapshot.MAX_BYTES:
            raise HTTPException(413, "project snapshot too large")
        infos = await offload(lambda: project_snapshot.members(data))
        return {"snapshot": await offload(lambda: store.objects.put(data)), "files": len(infos)}

    @app.get("/api/sessions/{sid}", dependencies=[Depends(user)])
    def get_session(sid: str) -> dict[str, Any]:
        record = guarded(lambda: manager.get(sid))
        out = record.to_dict()
        if manager.gateway is not None:
            try:
                out["usage"] = manager.gateway.usage(session=sid)
            except WspxError as exc:
                out["usage"] = {"error": str(exc)}
        return out  # type: ignore[no-any-return]

    @app.post("/api/sessions/{sid}/tools/{name}")
    def tool(
        sid: str, name: str, body: ToolCall, caller: str = Depends(session_caller)
    ) -> dict[str, Any]:
        source = (
            body.source
            if caller == "runner"
            else (body.source if body.source != "driver" else "human")
        )
        result: dict[str, Any] = guarded(
            lambda: manager.tool(
                sid, name, body.args, call_id=body.call_id, source=source, turn=body.turn
            )
        )
        return result

    @app.post("/api/sessions/{sid}/events")
    async def runner_event(
        sid: str, request: Request, _caller: str = Depends(session_caller)
    ) -> dict[str, Any]:
        event = await request.json()
        await offload(lambda: manager.runner_event(sid, event))
        return {"ok": True}

    @app.put("/api/sessions/{sid}/driver-state")
    async def put_driver_state(
        sid: str, runner: str, request: Request, _caller: str = Depends(session_caller)
    ) -> dict[str, Any]:
        body = await request.body()
        await offload(lambda: manager.trace.put_driver_state(sid, runner, body))
        return {"ok": True}

    @app.get("/api/sessions/{sid}/driver-state")
    def driver_state(sid: str, runner: str, _caller: str = Depends(session_caller)) -> Response:
        state = manager.trace.driver_state(sid, runner)
        if state is None:
            return Response(status_code=204)
        return Response(state, media_type="application/octet-stream")

    @app.get("/api/sessions/{sid}/commands")
    def commands(sid: str, _caller: str = Depends(session_caller)) -> dict[str, Any]:
        manager.touch(sid)
        return {"commands": manager.take_commands(sid)}

    @app.get("/api/sessions/{sid}/workspace")
    def workspace(sid: str, _caller: str = Depends(session_caller)) -> dict[str, Any]:
        return {"workspace": manager.workspace.get(sid, "")}

    @app.post("/api/sessions/{sid}/{action}", dependencies=[Depends(user)])
    async def control(sid: str, action: str, request: Request) -> dict[str, Any]:
        body = await request.json() if await request.body() else {}
        actions: dict[str, Callable[[], Any]] = {
            "pause": lambda: manager.pause(sid, body.get("reason", "paused by user")),
            "resume": lambda: manager.resume(sid),
            "stop": lambda: manager.kill(sid),
            "inject": lambda: manager.command(sid, "inject", body.get("text", "")),
            "start": lambda: manager.start_driver(sid),
        }
        if action not in actions:
            raise HTTPException(404, f"unknown action {action}")
        await offload(actions[action])
        return {"ok": True, "session": manager.get(sid).to_dict()}

    @app.get("/api/sessions/{sid}/timeline", dependencies=[Depends(user)])
    def timeline(sid: str, since: int = 0) -> dict[str, Any]:
        return {"events": trace.events(sid, since)}

    @app.get("/api/sessions/{sid}/stream", dependencies=[Depends(user)])
    async def stream(sid: str, request: Request, since: int = 0) -> StreamingResponse:
        guarded(lambda: manager.get(sid))

        async def events() -> AsyncIterator[bytes]:
            manager.viewers[sid] += 1
            last = since
            try:
                while not await request.is_disconnected():
                    manager.touch(sid)
                    rows = trace.events(sid, last)
                    for row in rows:
                        last = row["id"]
                        yield f"id: {row['id']}\nevent: {row['kind']}\ndata: {json.dumps(row)}\n\n".encode()
                    if not rows:
                        yield b": keep-alive\n\n"
                    await asyncio.sleep(0.5)
            finally:
                manager.viewers[sid] -= 1

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/api/calls/{call_id}", dependencies=[Depends(user)])
    def call(call_id: str) -> dict[str, Any]:
        found = trace.call(call_id)
        if found is None:
            raise HTTPException(404, "no such call")
        return found

    # ------------------------------------------------------------------ views for the UI

    from wspx.control import views

    def _fp(sid: str) -> str | None:
        return guarded(lambda: manager.get(sid)).fingerprint  # type: ignore[no-any-return]

    @app.get("/api/sessions/{sid}/architecture", dependencies=[Depends(user)])
    def architecture(sid: str) -> dict[str, Any]:
        return views.architecture(store, _fp(sid))

    @app.get("/api/sessions/{sid}/commits", dependencies=[Depends(user)])
    def commits(sid: str) -> dict[str, Any]:
        return views.commits(store, _fp(sid), sid)

    @app.get("/api/sessions/{sid}/runs", dependencies=[Depends(user)])
    def runs(sid: str) -> dict[str, Any]:
        return views.runs(store, _fp(sid), sid)

    @app.get("/api/sessions/{sid}/export", dependencies=[Depends(user)])
    def export_session(sid: str) -> dict[str, Any]:
        """The session's whole record: its trace from the start, commits, evals and spend."""
        record = guarded(lambda: manager.get(sid))
        out: dict[str, Any] = {
            "format": "wspx-session/1",
            "wspx": __version__,
            "exported": time.time(),
            "session": record.to_dict(),
            "events": trace.trace(sid),
        }
        parts: dict[str, Callable[[], Any]] = {
            "commits": lambda: views.commits(store, _fp(sid), sid),
            "evals": lambda: views.runs(store, _fp(sid), sid),
        }
        if manager.gateway is not None:
            parts["usage"] = lambda: manager.gateway.usage(session=sid)  # type: ignore[union-attr]
        for key, part in parts.items():
            try:
                out[key] = part()
            except Exception as exc:  # a missing part must not lose the rest of the record
                out[key] = {"error": str(exc)}
        return out

    @app.get("/api/sessions/{sid}/media", dependencies=[Depends(user)])
    async def session_media(sid: str, path: str) -> Response:
        """An image or video a prompt in this session named, from its project snapshot."""
        record = guarded(lambda: manager.get(sid))
        if not record.snapshot:
            raise HTTPException(404, "this session has no project files")
        snapshot = record.snapshot
        try:
            found, kind = await offload(
                lambda: media_cache.file(lambda: store.objects.get(snapshot), snapshot, path)
            )
        except FileNotFoundError:
            raise HTTPException(404, f"{path}: not in this session's project") from None
        except WspxError as exc:
            raise HTTPException(400, str(exc)) from None
        # a session's snapshot never changes, so the file at a path never does
        return FileResponse(
            found, media_type=kind, headers={"cache-control": "private, max-age=86400, immutable"}
        )

    @app.get("/api/sessions/{sid}/handles", dependencies=[Depends(user)])
    def handles(sid: str) -> dict[str, Any]:
        return views.handles(store, sid)

    @app.get("/api/objects/{oid}", dependencies=[Depends(user)])
    def json_object(oid: str) -> Any:
        return guarded(lambda: views.json_object(store, oid))

    # ------------------------------------------------------------------ spend and safety

    def need_gateway() -> GatewayAdmin:
        if manager.gateway is None:
            raise HTTPException(503, "no gateway configured")
        return manager.gateway

    @app.get("/api/keys", dependencies=[Depends(user)])
    def get_keys() -> dict[str, Any]:
        return {"keys": guarded(lambda: need_gateway().keys())}

    @app.post("/api/keys/reload", dependencies=[Depends(user)])
    def reload_keys() -> dict[str, Any]:
        return {"reloaded": guarded(lambda: need_gateway().recycle())}

    @app.get("/api/limits", dependencies=[Depends(user)])
    def get_limits() -> dict[str, Any]:
        return {"limits": guarded(lambda: need_gateway().limits())}

    @app.post("/api/limits", dependencies=[Depends(user)])
    async def set_limit(request: Request) -> dict[str, Any]:
        body = await request.json()
        return await offload(lambda: need_gateway().set_limit(body))  # type: ignore[no-any-return]

    @app.delete("/api/limits/{limit_id}", dependencies=[Depends(user)])
    def remove_limit(limit_id: str) -> dict[str, Any]:
        return {"removed": guarded(lambda: need_gateway().remove_limit(limit_id))}

    @app.post("/api/prices", dependencies=[Depends(user)])
    async def set_price(request: Request) -> dict[str, Any]:
        body = await request.json()
        await offload(
            lambda: need_gateway().price(body["model"], float(body["input"]), float(body["output"]))
        )
        return {"ok": True}

    @app.get("/api/usage", dependencies=[Depends(user)])
    def usage(session: str | None = None, since: float = 0.0) -> dict[str, Any]:
        return guarded(lambda: need_gateway().usage(session, since))  # type: ignore[no-any-return]

    @app.post("/api/kill", dependencies=[Depends(user)])
    async def kill(request: Request) -> dict[str, Any]:
        body = await request.json() if await request.body() else {}
        return {"stopped": await offload(lambda: manager.kill(body.get("session")))}

    @app.post("/api/resume", dependencies=[Depends(user)])
    async def resume(request: Request) -> dict[str, Any]:
        body = await request.json() if await request.body() else {}
        await offload(lambda: manager.resume(body.get("session")))
        return {"ok": True}

    @app.get("/api/doctor", dependencies=[Depends(user)])
    def doctor() -> dict[str, Any]:
        limits: list[dict[str, Any]] | str
        try:
            limits = need_gateway().limits()
        except HTTPException as exc:
            limits = str(exc.detail)
        except WspxError as exc:
            limits = str(exc)
        running = [
            r.to_dict()
            for r in manager.records.values()
            if r.tier != "cold" or r.status == "running"
        ]
        return {
            "halted": manager.halted,
            "limits": limits,
            "active": running,
            "provider_caps": PROVIDER_CAPS,
            "note": "wspx sets no limits by default; provider caps are set in each provider's console",
        }

    # ------------------------------------------------------------------ hooks from external agents

    @app.post("/api/ingest/{client}", dependencies=[Depends(user)])
    async def ingest(client: str, request: Request, session: str | None = None) -> dict[str, Any]:
        payload = await request.json()
        sid = session or request.headers.get("x-wspx-session") or "external"
        text = json.dumps(payload.get("tool_response") or payload.get("tool_output") or "")
        joined = _CALL_RE.search(text)
        trace.event(
            sid,
            "hook",
            {
                "client": client,
                "event": payload.get("hook_event_name") or payload.get("event"),
                "tool": payload.get("tool_name"),
                "call_id": joined.group(1) if joined else None,
                "payload": payload,
            },
            source=client,
        )
        return {"ok": True, "joined": joined.group(1) if joined else None}

    # ------------------------------------------------------------------ UI

    @app.get("/", response_class=HTMLResponse)
    def index() -> Response:
        ui = files("wspx.control").joinpath("ui/index.html")
        if ui.is_file():
            return HTMLResponse(ui.read_text(encoding="utf-8"))
        return HTMLResponse(
            "<!doctype html><title>wspx</title><p>wspx control plane is running.</p>"
        )

    @app.get("/assets/{name}")
    def asset(name: str) -> Response:
        path = files("wspx.control").joinpath(f"ui/assets/{name}")
        if "/" in name or not path.is_file():
            raise HTTPException(404)
        kind = (
            "text/javascript"
            if name.endswith(".js")
            else "text/css"
            if name.endswith(".css")
            else "application/octet-stream"
        )
        return Response(
            path.read_bytes(),
            media_type=kind,
            headers={"cache-control": "public, max-age=31536000"},
        )

    app.state.started = time.time()
    return app


def create_app_from_env() -> FastAPI:
    """Factory for ``uvicorn --factory`` (self-hosting with in-process labs)."""
    from wspx.control.backend import LocalBackend

    settings = ControlSettings.from_env()
    return create_app(settings, LocalBackend(settings.store_root))
