"""The trace DB (§7): every driver turn and every harness call, joined.

``sessions`` · ``turns`` · ``tool_calls`` · ``artefacts`` · ``events``. The
events table is the session timeline the UI streams over SSE; every wspx
result carries a ``call_id`` so external clients' hook payloads join exactly
to harness calls and the artefacts they produced.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, data TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL, idx INTEGER,
    role TEXT, text TEXT, reasoning TEXT, reasoning_available INTEGER, tokens TEXT, cost REAL, latency REAL,
    ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS tool_calls (call_id TEXT PRIMARY KEY, session TEXT NOT NULL, turn INTEGER,
    tool TEXT NOT NULL, args TEXT, summary TEXT, ok INTEGER, latency REAL, error TEXT, source TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS artefacts (id TEXT NOT NULL, kind TEXT NOT NULL, call_id TEXT, session TEXT,
    ts REAL, PRIMARY KEY (id, call_id));
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL,
    kind TEXT NOT NULL, source TEXT, payload TEXT NOT NULL, ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS events_session ON events(session, id);
CREATE INDEX IF NOT EXISTS calls_session ON tool_calls(session);
"""


def _kind_of(handle: str) -> str:
    if handle.startswith("@r"):
        return "result_set"
    if handle.startswith("dir:"):
        return "direction"
    if handle.startswith("p_"):
        return "patch"
    if handle.startswith("t_"):
        return "trace"
    return "commit"


class TraceDB:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            str(path), check_same_thread=False, isolation_level=None, timeout=30
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def _exec(self, sql: str, args: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, args)

    # sessions
    def put_session(self, record: dict[str, Any]) -> None:
        self._exec(
            "INSERT OR REPLACE INTO sessions VALUES (?,?,?)",
            (record["id"], json.dumps(record), time.time()),
        )

    def session(self, sid: str) -> dict[str, Any] | None:
        row = self._exec("SELECT data FROM sessions WHERE id = ?", (sid,)).fetchone()
        return json.loads(row["data"]) if row else None

    def sessions(self) -> list[dict[str, Any]]:
        return [
            json.loads(r["data"])
            for r in self._exec("SELECT data FROM sessions ORDER BY updated DESC")
        ]

    # timeline
    def event(
        self, session: str, kind: str, payload: dict[str, Any], *, source: str = "wspx"
    ) -> int:
        cur = self._exec(
            "INSERT INTO events (session, kind, source, payload, ts) VALUES (?,?,?,?,?)",
            (session, kind, source, json.dumps(payload, default=str), time.time()),
        )
        return int(cur.lastrowid or 0)

    def events(self, session: str, since: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        rows = self._exec(
            "SELECT * FROM events WHERE session = ? AND id > ? ORDER BY id LIMIT ?",
            (session, since, limit),
        ).fetchall()
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in rows]

    def turn(self, session: str, data: dict[str, Any]) -> int:
        self._exec(
            "INSERT INTO turns (session, idx, role, text, reasoning, reasoning_available, tokens, cost, "
            "latency, ts) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                session,
                data.get("turn"),
                "assistant",
                data.get("text"),
                data.get("reasoning"),
                int(bool(data.get("reasoning_available"))),
                json.dumps(data.get("usage") or {}),
                data.get("cost"),
                data.get("latency"),
                time.time(),
            ),
        )
        return self.event(session, "turn", data, source=data.get("source", "driver"))

    def tool_call(
        self,
        session: str,
        call_id: str,
        tool: str,
        args: dict[str, Any],
        result: dict[str, Any],
        *,
        turn: int | None,
        latency: float,
        source: str,
    ) -> int:
        self._exec(
            "INSERT OR REPLACE INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                call_id,
                session,
                turn,
                tool,
                json.dumps(args, default=str),
                result.get("summary", "")[:20000],
                int(bool(result.get("ok"))),
                latency,
                None if result.get("ok") else result.get("summary", "")[:500],
                source,
                time.time(),
            ),
        )
        for handle in result.get("handles") or []:
            self._exec(
                "INSERT OR IGNORE INTO artefacts VALUES (?,?,?,?,?)",
                (handle, _kind_of(handle), call_id, session, time.time()),
            )
        commit = (result.get("data") or {}).get("commit")
        if commit:
            self._exec(
                "INSERT OR IGNORE INTO artefacts VALUES (?,?,?,?,?)",
                (commit, "commit", call_id, session, time.time()),
            )
        return self.event(
            session,
            "tool_call",
            {
                "call_id": call_id,
                "tool": tool,
                "args": args,
                "turn": turn,
                "ok": result.get("ok"),
                "summary": result.get("summary", "")[:4000],
                "handles": result.get("handles") or [],
                "latency": latency,
            },
            source=source,
        )

    def call(self, call_id: str) -> dict[str, Any] | None:
        row = self._exec("SELECT * FROM tool_calls WHERE call_id = ?", (call_id,)).fetchone()
        if row is None:
            return None
        arts = self._exec("SELECT id, kind FROM artefacts WHERE call_id = ?", (call_id,)).fetchall()
        return dict(row) | {
            "args": json.loads(row["args"] or "{}"),
            "artefacts": [dict(a) for a in arts],
        }

    def calls(self, session: str) -> list[dict[str, Any]]:
        rows = self._exec(
            "SELECT * FROM tool_calls WHERE session = ? ORDER BY ts", (session,)
        ).fetchall()
        return [dict(r) for r in rows]
