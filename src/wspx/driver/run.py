"""Run a session's driver against a control plane and a gateway.

The runner holds a runner token (for its session's tool calls and events) and
an LLM token (for the gateway). It never holds a provider key, never runs
model-written code and never touches weights.

``python -m wspx.driver.run`` reads ``WSPX_*`` variables (see ``run_from_env``).
"""

from __future__ import annotations

import contextlib
import json
import os
import zlib
from collections.abc import Mapping
from typing import Any

import httpx

from wspx.driver.loop import Hooks, Runner, RunnerConfig, RunResult, ToolReply
from wspx.driver.providers import Provider, make_provider
from wspx.integrations import skill_text


class ControlHooks:
    def __init__(
        self,
        control_url: str,
        gateway_url: str,
        session: str,
        runner_token: str,
        llm_token: str,
        *,
        runner_id: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.session = session
        self.runner_id = runner_id
        """Names this runner's input: a restart of the same input resumes its checkpoint."""
        self.turn = 0
        self.control = httpx.Client(
            base_url=control_url.rstrip("/"),
            timeout=3600.0,
            transport=transport,
            headers={"authorization": f"Bearer {runner_token}"},
            follow_redirects=True,  # Modal's 303 for requests past 150 s (a cold lab)
            max_redirects=40,
        )
        self.gateway = (
            httpx.Client(
                base_url=gateway_url.rstrip("/"),
                timeout=30.0,
                transport=transport,
                headers={"x-api-key": llm_token},
            )
            if gateway_url
            else None
        )

    def tool(self, name: str, args: dict[str, Any], call_id: str) -> ToolReply:
        r = self.control.post(
            f"/api/sessions/{self.session}/tools/{name}",
            json={"args": args, "call_id": call_id, "turn": self.turn, "source": "driver"},
        )
        if r.status_code >= 400:
            return ToolReply(
                f"{name}: control plane error {r.status_code}: {r.text[:500]}", ok=False
            )
        data = r.json()
        return ToolReply(
            data.get("summary", ""), bool(data.get("ok", True)), list(data.get("attachments") or [])
        )

    def blob(self, oid: str) -> bytes:
        r = self.control.get(f"/api/sessions/{self.session}/attachments/{oid}")
        r.raise_for_status()
        return r.content

    def gateway_state(self) -> dict[str, Any]:
        if self.gateway is None:
            return {}
        try:
            data: dict[str, Any] = self.gateway.get("/v1/session").json()
            return data
        except httpx.HTTPError:
            return {"paused": "the gateway is unreachable; calls stop until it returns"}

    def commands(self) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = self.control.get(
            f"/api/sessions/{self.session}/commands"
        ).json()["commands"]
        return data

    def workspace(self) -> str:
        return str(
            self.control.get(f"/api/sessions/{self.session}/workspace").json().get("workspace", "")
        )

    def event(self, event: dict[str, Any]) -> None:
        if event.get("type") in ("turn", "resume"):
            self.turn = int(event.get("turn", event.get("turns", self.turn)))
        # The trace is best-effort; the call log on the control plane is authoritative.
        with contextlib.suppress(httpx.HTTPError):
            self.control.post(f"/api/sessions/{self.session}/events", json=event)

    def save_state(self, state: dict[str, Any]) -> None:
        if not self.runner_id:
            return
        body = zlib.compress(json.dumps(state, default=str).encode())
        # best-effort like the trace: a missed checkpoint costs at most one step on a restart
        with contextlib.suppress(httpx.HTTPError):
            self.control.put(
                f"/api/sessions/{self.session}/driver-state",
                params={"runner": self.runner_id},
                content=body,
                headers={"content-type": "application/octet-stream"},
            )

    def load_state(self) -> dict[str, Any] | None:
        if not self.runner_id:
            return None
        r = self.control.get(
            f"/api/sessions/{self.session}/driver-state", params={"runner": self.runner_id}
        )
        if r.status_code != 200 or not r.content:
            return None
        state: dict[str, Any] = json.loads(zlib.decompress(r.content))
        return state

    def hooks(self) -> Hooks:
        return Hooks(
            tool=self.tool,
            gateway_state=self.gateway_state,
            commands=self.commands,
            workspace=self.workspace,
            event=self.event,
            save_state=self.save_state,
            load_state=self.load_state,
            blob=self.blob,
        )


def run_from_env(
    env: Mapping[str, str] | None = None,
    *,
    provider: Provider | None = None,
    transport: httpx.BaseTransport | None = None,
) -> RunResult:
    env = env or os.environ
    hooks = ControlHooks(
        env["WSPX_CONTROL_URL"],
        env.get("WSPX_GATEWAY_URL", ""),
        env["WSPX_SESSION"],
        env["WSPX_RUNNER_TOKEN"],
        env["WSPX_LLM_TOKEN"],
        runner_id=env.get("WSPX_RUNNER_ID", ""),
        transport=transport,
    )
    if provider is None:
        provider = make_provider(
            env["WSPX_DRIVER"], gateway=env["WSPX_GATEWAY_URL"], token=env["WSPX_LLM_TOKEN"]
        )
    deny = frozenset(t for t in env.get("WSPX_TOOLS_DENY", "").split(",") if t)
    config = RunnerConfig(
        system=skill_text(),
        task=env.get("WSPX_TASK", ""),
        effort=env.get("WSPX_EFFORT", "high"),
        deny=deny,
        attachments=json.loads(env.get("WSPX_ATTACHMENTS") or "[]"),
        vision=env.get("WSPX_DRIVER_IMAGES", "auto") != "off",
        extra_tools=json.loads(env.get("WSPX_EXTRA_TOOLS") or "[]"),
    )
    return Runner(provider, hooks.hooks(), config).run()


def main() -> None:
    result = run_from_env()
    print(f"wspx driver: {result.status} after {result.turns} turn(s): {result.reason}", flush=True)


if __name__ == "__main__":
    main()
