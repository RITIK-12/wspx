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
    tool TEXT NOT NULL, args TEXT, summary TEXT, ok INTEGER, latency REAL, error TEXT, source TEXT, ts REAL,
    data TEXT);
CREATE TABLE IF NOT EXISTS artefacts (id TEXT NOT NULL, kind TEXT NOT NULL, call_id TEXT, session TEXT,
    ts REAL, PRIMARY KEY (id, call_id));
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL,
    kind TEXT NOT NULL, source TEXT, payload TEXT NOT NULL, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS driver_state (session TEXT PRIMARY KEY, runner TEXT NOT NULL,
    state BLOB NOT NULL, ts REAL NOT NULL);
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


MAX_DATA_BYTES = 1 << 20
"""Structured results above this are kept as a note; their full form is in the store."""


def _data_json(data: Any) -> str | None:
    if not data:
        return None
    text = json.dumps(data, default=str)
    if len(text) > MAX_DATA_BYTES:
        return json.dumps({"omitted": f"{len(text):,} bytes; the full result is in the store"})
    return text


class TraceDB:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            str(path), check_same_thread=False, isolation_level=None, timeout=30
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        columns = {r["name"] for r in self._db.execute("PRAGMA table_info(tool_calls)")}
        if "data" not in columns:  # trace databases from before results were kept in full
            self._db.execute("ALTER TABLE tool_calls ADD COLUMN data TEXT")
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

    # driver checkpoints: one per session, the latest runner's
    def put_driver_state(self, session: str, runner: str, state: bytes) -> None:
        self._exec(
            "INSERT OR REPLACE INTO driver_state VALUES (?,?,?,?)",
            (session, runner, state, time.time()),
        )

    def driver_state(self, session: str, runner: str) -> bytes | None:
        row = self._exec(
            "SELECT state FROM driver_state WHERE session = ? AND runner = ?", (session, runner)
        ).fetchone()
        return bytes(row["state"]) if row else None

    # timeline
    def event(
        self, session: str, kind: str, payload: dict[str, Any], *, source: str = "wspx"
    ) -> int:
        cur = self._exec(
            "INSERT INTO events (session, kind, source, payload, ts) VALUES (?,?,?,?,?)",
            (session, kind, source, json.dumps(payload, default=str), time.time()),
        )
        return int(cur.lastrowid or 0)

    def events(self, session: str, since: int = 0, limit: int | None = 500) -> list[dict[str, Any]]:
        rows = self._exec(
            "SELECT * FROM events WHERE session = ? AND id > ? ORDER BY id LIMIT ?",
            (session, since, -1 if limit is None else limit),
        ).fetchall()
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in rows]

    def trace(self, session: str) -> list[dict[str, Any]]:
        """Every event of the session from its start, each tool call with its full result."""
        data = {
            r["call_id"]: json.loads(r["data"]) if r["data"] else None
            for r in self._exec(
                "SELECT call_id, data FROM tool_calls WHERE session = ?", (session,)
            ).fetchall()
        }
        events = self.events(session, limit=None)
        for e in events:
            if e["kind"] == "tool_call":
                e["payload"]["data"] = data.get(e["payload"].get("call_id"))
        return events

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
            "INSERT OR REPLACE INTO tool_calls (call_id, session, turn, tool, args, summary, ok, "
            "latency, error, source, ts, data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                call_id,
                session,
                turn,
                tool,
                json.dumps(args, default=str),
                result.get("summary", ""),
                int(bool(result.get("ok"))),
                latency,
                None if result.get("ok") else result.get("summary", "")[:500],
                source,
                time.time(),
                _data_json(result.get("data")),
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
                "summary": result.get("summary", ""),
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
