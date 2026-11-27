"""HTTP client for the control plane."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
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
            # Modal answers a request still running after 150 s with a 303 to poll for the
            # result; a tool call that wakes a cold lab takes that long.
            follow_redirects=True,
            max_redirects=40,
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
        try:
            return r.json()
        except ValueError:
            raise WspxError(
                f"{method} {path}: the control plane at {self.url} answered with something "
                f"other than JSON (is the deployment stopped? run `wspx deploy`): {r.text[:200]}"
            ) from None

    def _raw(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            r = self.client.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise WspxError(f"cannot reach the control plane at {self.url}: {exc}") from exc
        if r.status_code >= 400:
            raise WspxError(f"{method} {path}: {r.text[:300]}")
        return r

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

    def upload(self, path: Path) -> dict[str, Any]:
        """An image or file to give a session (with ``new --attach`` or ``steer --attach``)."""
        import mimetypes

        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data: dict[str, Any] = self._req(
            "POST",
            "/api/attachments",
            params={"name": path.name},
            content=path.read_bytes(),
            headers={"content-type": kind},
        )
        return data

    def attachment(self, sid: str, oid: str) -> tuple[bytes, str]:
        r = self._raw("GET", f"/api/sessions/{sid}/attachments/{oid}")
        return r.content, r.headers.get("content-type", "application/octet-stream")

    def export_session(self, sid: str) -> dict[str, Any]:
        """The session's whole record (``wspx-session/1``): trace, commits, evals, spend."""
        data: dict[str, Any] = self._req("GET", f"/api/sessions/{sid}/export")
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

    def keys(self) -> dict[str, Any]:
        data: dict[str, Any] = self._req("GET", "/api/keys")["keys"]
        return data

    def reload_keys(self) -> bool:
        return bool(self._req("POST", "/api/keys/reload")["reloaded"])

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
