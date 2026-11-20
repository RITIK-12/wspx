"""The gateway's HTTP app: provider proxies, session state and an admin API.

Runner SDKs point their ``base_url`` here and present a session token instead
of a key:

=============  ============================================================
anthropic      ``{gateway}/v1/anthropic``          → api.anthropic.com
openai         ``{gateway}/v1/openai/v1``          → api.openai.com/v1
google         ``{gateway}/v1/google``             → generativelanguage.googleapis.com
hf-router      ``{gateway}/v1/hf-router/v1``       → router.huggingface.co/v1
compat         ``{gateway}/v1/compat/<name>``      → a configured endpoint
=============  ============================================================

The admin API (bearer admin token, held by the control plane and the user's
CLI, never by a runner or a lab) manages limits, prices, the halt flag and
pauses. No agent tool can reach it.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from wspx.auth import TokenError, check_bearer, verify
from wspx.gateway.ledger import Ledger, Limit
from wspx.gateway.prices import Price
from wspx.gateway.usage import (
    AnthropicStream,
    GeminiStream,
    Observed,
    OpenAIStream,
    SSEParser,
    Usage,
    anthropic_json,
    gemini_json,
    openai_json,
)
from wspx.loops import LoopDetector

UPSTREAMS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
    "google": "https://generativelanguage.googleapis.com",
    "hf-router": "https://router.huggingface.co",
}
KEY_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "hf-router": ("HF_TOKEN",),
}
DEFAULT_MAX_OUTPUT = 128_000
"""Worst case used when a request does not state its output cap."""
HOP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "accept-encoding",
        "authorization",
        "x-api-key",
        "x-goog-api-key",
        "keep-alive",
    }
)
SPEND_LIMIT_CODES = frozenset(
    {
        "insufficient_quota",
        "project_spend_limit_exceeded",
        "organization_spend_limit_exceeded",
        "billing_error",
        "billing_hard_limit_reached",
    }
)


@dataclass
class GatewayConfig:
    secret: str
    admin_token: str
    ledger_path: str
    keys: dict[str, str] = field(default_factory=dict)
    upstreams: dict[str, str] = field(default_factory=lambda: dict(UPSTREAMS))
    compat: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    """name → (base URL, optional key) for ``openai-compatible`` endpoints."""
    loop_threshold: int = 3
    loop_window: int = 10
    loops_enabled: bool = True
    transport: httpx.AsyncBaseTransport | None = None

    @classmethod
    def from_env(cls) -> GatewayConfig:
        keys = {}
        for provider, names in KEY_ENV.items():
            for name in names:
                if os.environ.get(name):
                    keys[provider] = os.environ[name]
                    break
        compat: dict[str, tuple[str, str | None]] = {}
        for name, value in os.environ.items():
            if name.startswith("WSPX_COMPAT_") and name.endswith("_URL"):
                label = name[len("WSPX_COMPAT_") : -len("_URL")].lower()
                compat[label] = (value, os.environ.get(f"WSPX_COMPAT_{label.upper()}_KEY"))
        return cls(
            secret=os.environ["WSPX_GATEWAY_SECRET"],
            admin_token=os.environ["WSPX_ADMIN_TOKEN"],
            ledger_path=os.environ.get("WSPX_LEDGER", "/ledger/ledger.db"),
            keys=keys,
            compat=compat,
        )


@dataclass
class Call:
    provider: str
    upstream: str
    model: str
    stream: bool
    max_output: int
    metered: bool


def create_app(config: GatewayConfig) -> Starlette:
    ledger = Ledger(config.ledger_path)
    loops = LoopDetector(
        threshold=config.loop_threshold, window=config.loop_window, enabled=config.loops_enabled
    )
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(900.0, connect=15.0), transport=config.transport
    )
    last_body: dict[str, tuple[bytes, int]] = {}

    def session_of(request: Request) -> str:
        token = (
            request.headers.get("x-api-key")
            or request.headers.get("x-goog-api-key")
            or request.query_params.get("key")
            or ""
        )
        auth = request.headers.get("authorization", "")
        if not token and auth.lower().startswith("bearer "):
            token = auth[7:]
        return verify(config.secret, token).session

    def error(status: int, kind: str, message: str) -> JSONResponse:
        return JSONResponse(
            {"type": "error", "error": {"type": kind, "message": message}},
            status_code=status,
            headers={"x-should-retry": "false"},
        )

    async def proxy(request: Request) -> Response:
        provider = request.path_params["provider"]
        path = request.path_params["path"]
        try:
            session = session_of(request)
        except TokenError as exc:
            return error(401, "wspx_auth", str(exc))
        if (reason := ledger.halted()) is not None:
            return error(
                503, "wspx_halted", f"wspx is halted ({reason}); `wspx resume` to continue"
            )
        if (reason := ledger.paused(session)) is not None:
            return error(403, "wspx_paused", f"session {session} is paused: {reason}")
        body = await request.body()
        try:
            call = describe(config, provider, path, body)
        except ValueError as exc:
            return error(404, "wspx_route", str(exc))
        key = config.keys.get(provider) if provider != "compat" else None
        if provider == "compat":
            key = config.compat[path.split("/", 1)[0]][1]
        elif key is None:
            return error(
                500, "wspx_no_key", f"no {provider} key is configured; `wspx keys set {provider}`"
            )

        reservation = None
        price = ledger.price_table().get(call.provider, call.model) if call.metered else None
        if call.metered:
            bound = input_bound(last_body.get(session), body)
            reservation, refusal = ledger.reserve(
                session,
                call.provider,
                call.model,
                input_bound=bound,
                max_output=call.max_output,
                price=price,
            )
            if refusal is not None:
                ledger.pause(session, f"limit reached: {refusal.message}")
                return error(429, "wspx_limit", refusal.message)

        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        headers.update(auth_headers(provider, key))
        url = call.upstream + ("?" + request.url.query if request.url.query else "")
        upstream = client.build_request(request.method, url, headers=headers, content=body)
        try:
            response = await client.send(upstream, stream=True)
        except httpx.HTTPError as exc:
            ledger.settle(
                reservation,
                session,
                call.provider,
                call.model,
                cost=0.0,
                status=502,
                error=str(exc),
            )
            return error(502, "wspx_upstream", f"{provider} unreachable: {exc}")

        def finish(usage: Usage, obs: Observed, status: int, err: str | None) -> list[str]:
            cost = usage.cost(price) if usage.complete else None
            if not call.metered:
                return []
            ledger.settle(
                reservation,
                session,
                call.provider,
                call.model,
                input=usage.input,
                output=usage.output,
                cache_read=usage.cache_read,
                cache_write=usage.cache_write,
                cost=cost if usage.complete else None,
                status=status,
                error=err,
            )
            if usage.complete:
                last_body[session] = (body, usage.input + usage.cache_read + usage.cache_write)
            return watch(session, obs, status, err)

        if response.status_code >= 400:
            raw = await response.aread()
            await response.aclose()
            err, spend_limit = classify_error(raw)
            spend_limit = (
                spend_limit or response.status_code == 402
            )  # Gemini: prepaid credit exhausted
            out_headers = passthrough(response.headers)
            if spend_limit:
                ledger.pause(session, f"provider spend limit: {err}")
                out_headers["x-should-retry"] = "false"
            finish(Usage(complete=True), Observed(), response.status_code, err)
            return Response(
                raw,
                status_code=response.status_code,
                headers=out_headers,
                media_type=response.headers.get("content-type"),
            )

        if not call.stream:
            raw = await response.aread()
            await response.aclose()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {}
            usage, obs = parse_json(provider, path, data)
            finish(usage, obs, response.status_code, None)
            return Response(
                raw,
                status_code=response.status_code,
                headers=passthrough(response.headers),
                media_type=response.headers.get("content-type"),
            )

        tracker = stream_tracker(provider, path)
        parser = SSEParser()

        async def relay() -> AsyncIterator[bytes]:
            done = False
            try:
                async for chunk in response.aiter_bytes():  # decoded: content-encoding is stripped
                    for name, data in parser.feed(chunk):
                        tracker.event(name, data)
                    yield chunk
                done = True
            finally:
                await response.aclose()
                usage, obs = tracker.finish()
                if not done:
                    usage.complete = False  # broken stream: the hold stays charged
                finish(usage, obs, response.status_code, None if done else "stream interrupted")

        return StreamingResponse(
            relay(),
            status_code=response.status_code,
            headers=passthrough(response.headers),
            media_type=response.headers.get("content-type", "text/event-stream"),
        )

    def watch(session: str, obs: Observed, status: int, err: str | None) -> list[str]:
        """Feed one response to the loop detector; record nudges and stops."""
        verdicts = []
        loops.turn(session)
        if err:
            verdicts.append(
                loops.observe(session, "error", f"{status}:{err}", label=f"HTTP {status}")
            )
        else:
            for name, args in obs.tool_calls:
                verdicts.append(
                    loops.observe(session, "call", {"tool": name, "args": args}, label=name)
                )
            if obs.text.strip():
                verdicts.append(loops.observe(session, "output", obs.text))
            if not obs.tool_calls and not obs.text.strip():
                verdicts.append(loops.observe(session, "empty", "empty", label="empty turn"))
        messages = []
        for v in verdicts:
            if v is None:
                continue
            ledger.loop_event(session, v.to_dict())
            if v.action == "stop":
                ledger.pause(session, f"loop detected: {v.message}")
            else:
                ledger.add_nudge(session, v.message)
                messages.append(v.message)
        return messages

    async def session_state(request: Request) -> Response:
        try:
            session = session_of(request)
        except TokenError as exc:
            return error(401, "wspx_auth", str(exc))
        return JSONResponse(
            {
                "session": session,
                "halted": ledger.halted(),
                "paused": ledger.paused(session),
                "nudges": ledger.take_nudges(session),
            }
        )

    # ------------------------------------------------------------------ admin

    def admin(handler: Callable[[Request], Any]) -> Callable[[Request], Any]:
        async def wrapped(request: Request) -> Response:
            auth = request.headers.get("authorization", "")
            if not check_bearer(
                config.admin_token, auth[7:] if auth.lower().startswith("bearer ") else None
            ):
                return error(401, "wspx_auth", "admin token required")
            try:
                result = await handler(request)
            except (ValueError, KeyError) as exc:
                return error(400, "wspx_bad_request", str(exc))
            return result if isinstance(result, Response) else JSONResponse(result)

        return wrapped

    async def limits(request: Request) -> Any:
        if request.method == "POST":
            limit = Limit.model_validate(await request.json())
            return ledger.set_limit(limit).model_dump()
        if request.method == "DELETE":
            return {"removed": ledger.remove_limit(request.query_params["id"])}
        return {
            "limits": [lim.model_dump() | {"describe": lim.describe()} for lim in ledger.limits()]
        }

    async def usage(request: Request) -> Any:
        q = request.query_params
        return ledger.usage_summary(session=q.get("session"), since=float(q.get("since", 0))) | {
            "notices": ledger.notices(float(q.get("since", 0)))
        }

    async def halt(request: Request) -> Any:
        data = await request.json() if await request.body() else {}
        ledger.set_halt(True, data.get("reason", "kill --all"))
        return {"halted": True}

    async def resume(request: Request) -> Any:
        data = await request.json() if await request.body() else {}
        session = data.get("session")
        ledger.resume(session)
        if session:
            loops.reset(session)
        return {"resumed": session or "all"}

    async def pause(request: Request) -> Any:
        data = await request.json()
        ledger.pause(request.path_params["session"], data.get("reason", "paused by user"))
        return {"paused": request.path_params["session"]}

    async def session_info(request: Request) -> Any:
        sid = request.path_params["session"]
        return {
            "session": sid,
            "paused": ledger.paused(sid),
            "usage": ledger.usage_summary(session=sid),
            "loop_events": ledger.loop_events(sid),
        }

    async def prices(request: Request) -> Any:
        data = await request.json()
        provider, _, model = data["model"].partition(":")
        ledger.set_price(
            provider,
            model,
            Price(
                float(data["input"]),
                float(data["output"]),
                data.get("cache_read"),
                data.get("cache_write"),
            ),
        )
        return {"ok": True}

    async def observe(request: Request) -> Any:
        """The control plane reports tool calls and outcomes; repetition there also counts."""
        data = await request.json()
        session = data["session"]
        verdict = loops.observe(session, data["kind"], data["key"], label=data.get("label", ""))
        if verdict is None:
            return {"verdict": None}
        ledger.loop_event(session, verdict.to_dict())
        if verdict.action == "stop":
            ledger.pause(session, f"loop detected: {verdict.message}")
        return {"verdict": verdict.to_dict()}

    async def health(_request: Request) -> Response:
        return JSONResponse({"ok": True, "halted": ledger.halted() is not None})

    routes = [
        Route("/healthz", health),
        Route("/v1/session", session_state),
        Route("/v1/{provider:str}/{path:path}", proxy, methods=["POST", "GET"]),
        Route("/admin/limits", admin(limits), methods=["GET", "POST", "DELETE"]),
        Route("/admin/usage", admin(usage)),
        Route("/admin/halt", admin(halt), methods=["POST"]),
        Route("/admin/resume", admin(resume), methods=["POST"]),
        Route("/admin/sessions/{session}/pause", admin(pause), methods=["POST"]),
        Route("/admin/sessions/{session}", admin(session_info)),
        Route("/admin/prices", admin(prices), methods=["POST"]),
        Route("/admin/observe", admin(observe), methods=["POST"]),
    ]
    app = Starlette(routes=routes)
    app.state.ledger = ledger
    app.state.loops = loops
    return app


# --------------------------------------------------------------------------- request shapes

_GEMINI_RE = re.compile(r"models/(?P<model>[^/:]+):(?P<method>\w+)")


def describe(config: GatewayConfig, provider: str, path: str, body: bytes) -> Call:
    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        data = {}
    if provider == "compat":
        name, _, rest = path.partition("/")
        if name not in config.compat:
            raise ValueError(f"no openai-compatible endpoint named {name!r}")
        base = config.compat[name][0].rstrip("/")
        return Call(
            "compat:" + name,
            f"{base}/{rest}",
            str(data.get("model", "")),
            bool(data.get("stream")),
            int(data.get("max_completion_tokens") or data.get("max_tokens") or DEFAULT_MAX_OUTPUT),
            rest.endswith(("chat/completions", "responses")),
        )
    if provider not in config.upstreams:
        raise ValueError(f"unknown provider {provider!r}")
    upstream = f"{config.upstreams[provider].rstrip('/')}/{path}"
    if provider == "anthropic":
        metered = path.endswith("messages")
        return Call(
            provider,
            upstream,
            str(data.get("model", "")),
            bool(data.get("stream")),
            int(data.get("max_tokens") or DEFAULT_MAX_OUTPUT),
            metered,
        )
    if provider == "google":
        m = _GEMINI_RE.search(path)
        model = m.group("model") if m else ""
        method = m.group("method") if m else ""
        cap = (data.get("generationConfig") or {}).get("maxOutputTokens") or DEFAULT_MAX_OUTPUT
        return Call(
            provider,
            upstream,
            model,
            method == "streamGenerateContent",
            int(cap),
            method in ("generateContent", "streamGenerateContent"),
        )
    cap = (
        data.get("max_output_tokens") or data.get("max_completion_tokens") or data.get("max_tokens")
    )
    return Call(
        provider,
        upstream,
        str(data.get("model", "")),
        bool(data.get("stream")),
        int(cap or DEFAULT_MAX_OUTPUT),
        path.endswith(("responses", "chat/completions")),
    )


def input_bound(previous: tuple[bytes, int] | None, body: bytes, slack: int = 256) -> int:
    """An upper bound on input tokens: every token covers at least one byte.

    When this body extends the previous one (a growing conversation), the bound is
    the previous exact count plus the new bytes; otherwise it is the body length.
    """
    if previous is not None:
        prev, tokens = previous
        cut = max(0, len(prev) - slack)
        if len(body) >= cut and body[:cut] == prev[:cut]:
            return tokens + (len(body) - cut)
    return len(body)


def auth_headers(provider: str, key: str | None) -> dict[str, str]:
    if key is None:
        return {}
    if provider == "anthropic":
        return {"x-api-key": key}
    if provider == "google":
        return {"x-goog-api-key": key}
    return {"authorization": f"Bearer {key}"}


def passthrough(headers: httpx.Headers) -> dict[str, str]:
    drop = {"content-length", "content-encoding", "transfer-encoding", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in drop}


def classify_error(raw: bytes) -> tuple[str, bool]:
    """``(code or message, is a spend limit)``. Spend limits are never retried."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:200].decode("utf-8", "replace"), False
    err = data.get("error") if isinstance(data, dict) else None
    if not isinstance(err, dict):
        return str(data)[:200], False
    code = str(err.get("code") or err.get("type") or err.get("status") or "")
    message = str(err.get("message") or code)
    spend = (
        code in SPEND_LIMIT_CODES
        or "spend limit" in message.lower()
        or (err.get("status") == "RESOURCE_EXHAUSTED" and "billing" in message.lower())
    )
    return code or message[:120], spend


def parse_json(provider: str, path: str, data: dict[str, Any]) -> tuple[Usage, Observed]:
    if provider == "anthropic":
        return anthropic_json(data)
    if provider == "google":
        return gemini_json(data)
    del path
    return openai_json(data)


def stream_tracker(provider: str, path: str) -> Any:
    del path
    if provider == "anthropic":
        return AnthropicStream()
    if provider == "google":
        return GeminiStream()
    return OpenAIStream()


def create_app_from_env() -> Starlette:
    """Factory for ``uvicorn --factory`` (self-hosting)."""
    return create_app(GatewayConfig.from_env())
