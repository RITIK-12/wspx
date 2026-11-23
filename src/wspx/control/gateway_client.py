"""The control plane's view of the gateway's admin API."""

from __future__ import annotations

from typing import Any

import httpx

from wspx.errors import WspxError


class GatewayAdmin:
    def __init__(
        self, url: str, token: str, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.client = httpx.Client(
            base_url=url.rstrip("/"),
            headers={"authorization": f"Bearer {token}"},
            timeout=90.0,  # the gateway scales to zero; a cold start takes tens of seconds
            transport=transport,
        )

    def _req(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = self.client.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise WspxError(f"gateway unreachable: {exc}") from exc
        if r.status_code >= 400:
            raise WspxError(f"gateway {path}: {r.text[:300]}")
        try:
            return r.json()
        except ValueError:
            raise WspxError(f"gateway {path}: not JSON: {r.text[:300]}") from None

    def halted(self) -> bool:
        return bool(self._req("GET", "/healthz")["halted"])

    def keys(self) -> dict[str, Any]:
        data: dict[str, Any] = self._req("GET", "/admin/keys")["keys"]
        return data

    def recycle(self) -> bool:
        return bool(self._req("POST", "/admin/recycle")["recycled"])

    def limits(self) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = self._req("GET", "/admin/limits")["limits"]
        return data

    def set_limit(self, limit: dict[str, Any]) -> dict[str, Any]:
        data: dict[str, Any] = self._req("POST", "/admin/limits", json=limit)
        return data

    def remove_limit(self, limit_id: str) -> bool:
        return bool(self._req("DELETE", "/admin/limits", params={"id": limit_id})["removed"])

    def usage(self, session: str | None = None, since: float = 0.0) -> dict[str, Any]:
        params: dict[str, Any] = {"since": since}
        if session:
            params["session"] = session
        data: dict[str, Any] = self._req("GET", "/admin/usage", params=params)
        return data

    def halt(self, reason: str = "kill --all") -> None:
        self._req("POST", "/admin/halt", json={"reason": reason})

    def resume(self, session: str | None = None) -> None:
        self._req("POST", "/admin/resume", json={"session": session} if session else {})

    def pause(self, session: str, reason: str) -> None:
        self._req("POST", f"/admin/sessions/{session}/pause", json={"reason": reason})

    def session(self, session: str) -> dict[str, Any]:
        data: dict[str, Any] = self._req("GET", f"/admin/sessions/{session}")
        return data

    def price(self, model: str, inp: float, out: float) -> None:
        self._req("POST", "/admin/prices", json={"model": model, "input": inp, "output": out})

    def observe(
        self, session: str, kind: str, key: Any, label: str = "", step: int | None = None
    ) -> dict[str, Any] | None:
        verdict: dict[str, Any] | None = self._req(
            "POST",
            "/admin/observe",
            json={"session": session, "kind": kind, "key": key, "label": label, "step": step},
        )["verdict"]
        return verdict
