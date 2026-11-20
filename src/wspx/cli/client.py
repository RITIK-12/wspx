"""HTTP client for the control plane."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx

from wspx.errors import WspxError


class Control:
    def __init__(
        self, url: str, token: str, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.client = httpx.Client(
            base_url=self.url,
            headers={"authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(3600.0, connect=15.0),
            transport=transport,
        )

    def _req(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = self.client.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise WspxError(f"cannot reach the control plane at {self.url}: {exc}") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise WspxError(f"{method} {path}: {detail}")
        return r.json()

    def sessions(self) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = self._req("GET", "/api/sessions")["sessions"]
        return data

    def upload_project(self, snapshot: bytes) -> str:
        """Store a packed project; returns its snapshot id (uploads are deduplicated by content)."""
        data: dict[str, Any] = self._req(
            "POST", "/api/projects", content=snapshot, headers={"content-type": "application/gzip"}
        )
        return str(data["snapshot"])

    def new(self, **body: Any) -> dict[str, Any]:
        data: dict[str, Any] = self._req("POST", "/api/sessions", json=body)
        return data

    def session(self, sid: str) -> dict[str, Any]:
        data: dict[str, Any] = self._req("GET", f"/api/sessions/{sid}")
        return data

    def tool(self, sid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        data: dict[str, Any] = self._req(
            "POST", f"/api/sessions/{sid}/tools/{name}", json={"args": args, "source": "human"}
        )
        return data

    def action(self, sid: str, action: str, **body: Any) -> dict[str, Any]:
        data: dict[str, Any] = self._req("POST", f"/api/sessions/{sid}/{action}", json=body)
        return data

    def limits(self) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = self._req("GET", "/api/limits")["limits"]
        return data

    def set_limit(self, body: dict[str, Any]) -> dict[str, Any]:
        data: dict[str, Any] = self._req("POST", "/api/limits", json=body)
        return data

    def remove_limit(self, limit_id: str) -> bool:
        return bool(self._req("DELETE", f"/api/limits/{limit_id}")["removed"])

    def price(self, model: str, inp: float, out: float) -> None:
        self._req("POST", "/api/prices", json={"model": model, "input": inp, "output": out})

    def usage(self, session: str | None = None) -> dict[str, Any]:
        data: dict[str, Any] = self._req(
            "GET", "/api/usage", params={"session": session} if session else {}
        )
        return data

    def kill(self, session: str | None = None) -> int:
        return int(
            self._req("POST", "/api/kill", json={"session": session} if session else {})["stopped"]
        )

    def resume(self, session: str | None = None) -> None:
        self._req("POST", "/api/resume", json={"session": session} if session else {})

    def doctor(self) -> dict[str, Any]:
        data: dict[str, Any] = self._req("GET", "/api/doctor")
        return data

    def ingest(self, client: str, payload: dict[str, Any], session: str | None) -> dict[str, Any]:
        data: dict[str, Any] = self._req(
            "POST",
            f"/api/ingest/{client}",
            json=payload,
            params={"session": session} if session else {},
        )
        return data

    def stream(self, sid: str, since: int = 0) -> Iterator[dict[str, Any]]:
        import json

        with self.client.stream(
            "GET", f"/api/sessions/{sid}/stream", params={"since": since}, timeout=None
        ) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])
