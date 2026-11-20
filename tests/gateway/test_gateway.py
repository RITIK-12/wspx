from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from wspx.auth import mint
from wspx.gateway.app import GatewayConfig, create_app, input_bound
from wspx.gateway.ledger import Ledger, Limit
from wspx.gateway.prices import BUILTIN, Price, PriceTable
from wspx.loops import LoopDetector

SECRET = "s3cret"
ADMIN = "admin-token"
OPUS = BUILTIN["anthropic:claude-opus-5-5"]


def anthropic_reply(
    text: str = "hi",
    tools: list[tuple[str, dict]] | None = None,  # type: ignore[type-arg]
    usage: tuple[int, int] = (100, 20),
) -> dict:  # type: ignore[type-arg]
    content = [{"type": "text", "text": text}] if text else []
    content += [
        {"type": "tool_use", "id": f"t{i}", "name": n, "input": a}
        for i, (n, a) in enumerate(tools or [])
    ]
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": content,
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": usage[0],
            "output_tokens": usage[1],
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def sse(events: list[tuple[str, dict]]) -> bytes:  # type: ignore[type-arg]
    return b"".join(f"event: {n}\ndata: {json.dumps(d)}\n\n".encode() for n, d in events)


class Upstream:
    """A fake provider: records requests, answers with whatever the test queues."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.reply: Callable[[httpx.Request], httpx.Response] = lambda r: httpx.Response(
            200, json=anthropic_reply()
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.reply(request)


@pytest.fixture
def setup(tmp_path: Path):  # type: ignore[no-untyped-def]
    upstream = Upstream()
    config = GatewayConfig(
        secret=SECRET,
        admin_token=ADMIN,
        ledger_path=str(tmp_path / "ledger.db"),
        keys={"anthropic": "sk-real", "openai": "sk-openai", "google": "g-key"},
        transport=httpx.MockTransport(upstream),
    )
    app = create_app(config)
    client = TestClient(app)
    return client, upstream, app.state.ledger


def call(
    client: TestClient,
    session: str = "s1",
    body: dict | None = None,
    path: str = "anthropic/v1/messages",  # type: ignore[type-arg]
    header: str = "x-api-key",
) -> httpx.Response:
    token = mint(SECRET, session)
    headers = {header: token} if header != "authorization" else {"authorization": f"Bearer {token}"}
    body = body or {
        "model": "claude-opus-5-5",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "x"}],
    }
    return client.post(f"/v1/{path}", json=body, headers=headers)


def admin(client: TestClient, method: str, path: str, **kw: object) -> httpx.Response:
    return client.request(
        method, f"/admin/{path}", headers={"authorization": f"Bearer {ADMIN}"}, **kw
    )  # type: ignore[arg-type]


class TestProxy:
    def test_auth_and_key_swap(self, setup) -> None:  # type: ignore[no-untyped-def]
        client, upstream, _ = setup
        assert client.post("/v1/anthropic/v1/messages", json={}).status_code == 401
        assert call(client).status_code == 200
        sent = upstream.requests[-1]
        assert sent.headers["x-api-key"] == "sk-real"
        assert "wspx_s_" not in str(sent.headers)
        assert str(sent.url) == "https://api.anthropic.com/v1/messages"

    def test_metering_matches_price(self, setup) -> None:  # type: ignore[no-untyped-def]
        client, upstream, ledger = setup
        upstream.reply = lambda r: httpx.Response(
            200, json=anthropic_reply(usage=(1_000_000, 100_000))
        )
        call(client)
        summary = ledger.usage_summary(session="s1")
        assert summary["total_charged"] == pytest.approx(OPUS.input + OPUS.output * 0.1)

    def test_anthropic_stream_is_relayed_and_metered(self, setup) -> None:  # type: ignore[no-untyped-def]
        client, upstream, ledger = setup
        events = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 500_000, "output_tokens": 1}},
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hello"},
                },
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 50_000},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        payload = sse(events)
        upstream.reply = lambda r: httpx.Response(
            200, content=payload, headers={"content-type": "text/event-stream"}
        )
        resp = call(
            client,
            body={"model": "claude-opus-5-5", "max_tokens": 64000, "stream": True, "messages": []},
        )
        assert resp.content == payload
        assert ledger.usage_summary()["total_charged"] == pytest.approx(
            0.5 * OPUS.input + 0.05 * OPUS.output
        )

    def test_openai_responses_stream(self, setup) -> None:  # type: ignore[no-untyped-def]
        client, upstream, ledger = setup
        done = {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{"type": "function_call", "name": "grep", "arguments": "{}"}],
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 300,
                    "input_tokens_details": {"cached_tokens": 200},
                    "output_tokens_details": {"reasoning_tokens": 100},
                },
            },
        }
        upstream.reply = lambda r: httpx.Response(
            200,
            content=sse([("response.completed", done)]),
            headers={"content-type": "text/event-stream"},
        )
        admin(
            client, "POST", "prices", json={"model": "openai:gpt-test", "input": 1.0, "output": 2.0}
        )
        call(
            client,
            path="openai/v1/responses",
            header="authorization",
            body={"model": "gpt-test", "stream": True, "input": "x", "max_output_tokens": 500},
        )
        assert upstream.requests[-1].headers["authorization"] == "Bearer sk-openai"
        row = ledger.usage_summary()["groups"][0]
        assert (row["i"], row["o"], row["cr"]) == (800, 300, 200)
        assert row["c"] == pytest.approx((800 * 1.0 + 300 * 2.0 + 200 * 0.1) / 1e6)

    def test_gemini_stream_and_unknown_price(self, setup) -> None:  # type: ignore[no-untyped-def]
        client, upstream, ledger = setup
        chunks = [
            {
                "candidates": [{"content": {"parts": [{"text": "a"}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1},
            },
            {
                "candidates": [{"content": {"parts": [{"text": "b"}]}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "thoughtsTokenCount": 7,
                },
            },
        ]
        upstream.reply = lambda r: httpx.Response(
            200,
            content=sse([("message", c) for c in chunks]),
            headers={"content-type": "text/event-stream"},
        )
        call(
            client,
            path="google/v1beta/models/gemini-x:streamGenerateContent",
            header="x-goog-api-key",
            body={"contents": []},
        )
        assert upstream.requests[-1].headers["x-goog-api-key"] == "g-key"
        row = ledger.usage_summary()["groups"][0]
        assert (row["model"], row["o"], row["c"]) == (
            "gemini-x",
            12,
            None,
        )  # output includes thoughts; no price


class TestLimits:
    def test_no_limits_by_default_and_long_runs_continue(self, setup) -> None:  # type: ignore[no-untyped-def]
        client, upstream, ledger = setup
        n = 0

        def fresh(r: httpx.Request) -> httpx.Response:
            nonlocal n
            n += 1
            return httpx.Response(
                200, json=anthropic_reply(f"step {n}: new work", [("grep", {"q": n})])
            )

        upstream.reply = fresh
        for _ in range(60):
            assert call(client).status_code == 200
        assert ledger.paused("s1") is None
        assert ledger.limits() == []

    def test_dollar_limit_never_overshoots_and_pauses(self, setup) -> None:  # type: ignore[no-untyped-def]
        client, upstream, ledger = setup
        n = iter(range(10_000))  # distinct replies, so loop detection stays out of it
        upstream.reply = lambda r: httpx.Response(
            200, json=anthropic_reply(f"reply {next(n)}", usage=(1000, 1000))
        )
        admin(
            client, "POST", "limits", json={"scope": "session", "metric": "dollars", "value": 0.05}
        )
        statuses = [call(client).status_code for _ in range(40)]
        assert 429 in statuses
        assert ledger.usage_summary(session="s1")["total_charged"] <= 0.05
        assert "limit reached" in (ledger.paused("s1") or "")
        state = client.get("/v1/session", headers={"x-api-key": mint(SECRET, "s1")}).json()
        assert state["paused"]

    def test_unknown_price_with_dollar_limit_asks_for_price(self, setup) -> None:  # type: ignore[no-untyped-def]
        client, _, _ = setup
        admin(client, "POST", "limits", json={"scope": "day", "metric": "dollars", "value": 5})
        resp = call(client, body={"model": "claude-unknown-9", "max_tokens": 10, "messages": []})
        assert resp.status_code == 429
        assert "price of anthropic:claude-unknown-9 is unknown" in resp.json()["error"]["message"]

    def test_concurrent_sessions_never_settle_above_the_limit(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "l.db")
        ledger.set_limit(Limit(scope="day", metric="dollars", value=1.0))
        price = Price(1.0, 1.0)

        def worker(i: int) -> None:
            for _ in range(50):
                rid, refusal = ledger.reserve(
                    f"s{i}", "anthropic", "m", input_bound=10_000, max_output=10_000, price=price
                )
                if refusal is None:
                    ledger.settle(
                        rid, f"s{i}", "anthropic", "m", input=8000, output=9000, cost=0.017
                    )

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert ledger.usage_summary()["total_charged"] <= 1.0

    def test_notices_and_validation(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "l.db")
        lim = ledger.set_limit(Limit(scope="session", metric="dollars", value=1.0))
        rid, _ = ledger.reserve(
            "s", "anthropic", "m", input_bound=10, max_output=10, price=Price(1, 1)
        )
        notes = ledger.settle(rid, "s", "anthropic", "m", cost=0.85)
        assert [n["level"] for n in notes] == [0.5, 0.8]
        assert lim.describe() == "dollars ≤ 1 per session"
        with pytest.raises(ValueError):  # noqa: PT011
            Limit(scope="day", metric="calls", value=3)
        with pytest.raises(ValueError):  # noqa: PT011
            Limit(scope="velocity", metric="dollars", value=3)

    def test_broken_stream_keeps_the_hold(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "l.db")
        ledger.set_limit(Limit(scope="session", metric="dollars", value=10))
        rid, _ = ledger.reserve(
            "s", "anthropic", "m", input_bound=1_000_000, max_output=0, price=Price(2, 2)
        )
        ledger.settle(rid, "s", "anthropic", "m", cost=None, error="stream interrupted")
        assert ledger.usage_summary()["total_charged"] == pytest.approx(
            2.5
        )  # worst case incl. cache writes

    def test_input_bound(self) -> None:
        prev = b'{"messages":[1,2,3]}' * 50
        assert input_bound((prev, 40), prev + b"more") == 40 + 4 + 256
        assert input_bound((prev, 40), b"different") == 9
        assert input_bound(None, b"abc") == 3


class TestSafety:
    def test_spend_limit_errors_are_never_retried(self, setup) -> None:  # type: ignore[no-untyped-def]
        client, upstream, ledger = setup
        upstream.reply = lambda r: httpx.Response(
            429,
            json={"error": {"code": "insufficient_quota", "message": "You exceeded your quota"}},
        )
        resp = call(
            client,
            path="openai/v1/responses",
            header="authorization",
            body={"model": "gpt-x", "input": "x"},
        )
        assert resp.status_code == 429
        assert resp.headers["x-should-retry"] == "false"
        assert "spend limit" in (ledger.paused("s1") or "")
        assert len(upstream.requests) == 1

    def test_ordinary_rate_limits_pass_through(self, setup) -> None:  # type: ignore[no-untyped-def]
        client, upstream, ledger = setup
        upstream.reply = lambda r: httpx.Response(
            429, json={"error": {"type": "rate_limit_error", "message": "slow down"}}
        )
        resp = call(client)
        assert resp.status_code == 429
        assert "x-should-retry" not in resp.headers
        assert ledger.paused("s1") is None

    def test_halt_and_resume(self, setup) -> None:  # type: ignore[no-untyped-def]
        client, _, _ = setup
        assert client.post("/admin/halt").status_code == 401  # admin only
        admin(client, "POST", "halt", json={"reason": "kill --all"})
        assert call(client).status_code == 503
        admin(client, "POST", "resume", json={})
        assert call(client).status_code == 200

    def test_loops_nudge_then_stop(self, setup) -> None:  # type: ignore[no-untyped-def]
        client, upstream, ledger = setup
        upstream.reply = lambda r: httpx.Response(
            200, json=anthropic_reply("", [("grep", {"q": "same"})])
        )
        for _ in range(3):
            assert call(client).status_code == 200
        state = client.get("/v1/session", headers={"x-api-key": mint(SECRET, "s1")}).json()
        assert state["nudges"]
        assert "identical arguments" in state["nudges"][0]
        assert call(client).status_code == 200  # the repeat after the nudge…
        assert "loop detected" in (ledger.paused("s1") or "")  # …stops the session
        assert call(client).status_code == 403
        admin(client, "POST", "resume", json={"session": "s1"})
        assert call(client).status_code == 200


class TestLoopDetector:
    def test_signals(self) -> None:
        d = LoopDetector(threshold=3, window=10)
        for i in range(2):
            d.turn("s")
            assert d.observe("s", "outcome", "W203") is None, i
        d.turn("s")
        v = d.observe("s", "outcome", "W203", label="W203")
        assert v is not None
        assert v.action == "nudge"
        d.turn("s")
        assert d.observe("s", "outcome", "E105") is None  # a different outcome breaks the run

    def test_window(self) -> None:
        d = LoopDetector(threshold=3, window=4)
        for i in range(12):
            d.turn("s")
            v = d.observe(
                "s", "call", {"tool": "grep", "args": {"q": "x"}} if i % 4 == 0 else {"i": i}
            )
            assert v is None  # identical calls, but never 3 inside any 4-turn window

    def test_disabled(self) -> None:
        d = LoopDetector(enabled=False)
        assert all(d.observe("s", "output", "same") is None for _ in range(10))


def test_price_table_fallbacks() -> None:
    table = PriceTable({"openai:gpt-x": Price(1, 2)})
    assert table.get("anthropic", "claude-opus-5-5") == OPUS
    assert table.get("anthropic", "claude-opus-5-5-fast") == OPUS
    assert table.get("openai", "gpt-x") == Price(1, 2)
    assert table.get("openai", "gpt-unknown") is None


def test_keys_are_checked_without_being_revealed(setup) -> None:  # type: ignore[no-untyped-def]
    client, upstream, _ = setup

    def reply(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.openai.com":
            return httpx.Response(401, json={"error": {"message": "invalid key"}})
        return httpx.Response(200, json={"data": []})

    upstream.reply = reply
    assert client.get("/admin/keys").status_code == 401
    keys = admin(client, "GET", "keys").json()["keys"]
    assert keys["anthropic"] == {"set": True, "ok": True}
    assert keys["google"] == {"set": True, "ok": True}
    assert keys["openai"]["ok"] is False
    assert "401" in keys["openai"]["detail"]
    assert keys["hf-router"] == {"set": False}
    assert "sk-real" not in json.dumps(keys)
    sent = {r.url.host: r for r in upstream.requests}
    assert sent["api.anthropic.com"].headers["x-api-key"] == "sk-real"
    assert sent["api.anthropic.com"].method == "GET"

    # outside Modal there is no container to recycle; the gateway says so
    assert admin(client, "POST", "recycle").json()["recycled"] is False
