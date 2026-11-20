"""The spend ledger (§7): SQLite on its own volume, written only by the gateway.

Tables: ``limits`` (user-set, empty by default), ``reservations`` (open
worst-case holds, used only while a dollar limit is set), ``usage`` (settled
cost per call), ``loop_events``, ``halt``, plus per-session state, pending
nudges, price overrides and limit notices.

Dollar limits never overshoot. Before a call the gateway reserves its
worst-case cost against every applicable limit inside one ``BEGIN
IMMEDIATE`` transaction; afterwards the hold settles to the actual cost. A
call that ends without usage (a broken stream) keeps its full reservation.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from wspx.gateway.prices import Price, PriceTable

Scope = Literal["call", "session", "day", "month", "rate", "velocity"]
Metric = Literal[
    "input_tokens",
    "output_tokens",
    "dollars",
    "calls",
    "turns",
    "tool_calls",
    "wall_clock_s",
    "gpu_minutes",
    "gpu_dollars",
    "concurrent_sessions",
]
GATEWAY_METRICS = frozenset({"input_tokens", "output_tokens", "dollars", "calls", "turns"})
NOTICE_LEVELS = (0.5, 0.8, 1.0)

SCHEMA = """
CREATE TABLE IF NOT EXISTS limits (
    id TEXT PRIMARY KEY, scope TEXT NOT NULL, metric TEXT NOT NULL, value REAL NOT NULL,
    session TEXT, provider TEXT, window_s REAL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS reservations (
    id TEXT PRIMARY KEY, session TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
    amount REAL NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, session TEXT NOT NULL, provider TEXT NOT NULL,
    model TEXT NOT NULL, input INTEGER, output INTEGER, cache_read INTEGER, cache_write INTEGER,
    cost REAL, charged REAL NOT NULL DEFAULT 0, status INTEGER, error TEXT);
CREATE INDEX IF NOT EXISTS usage_session ON usage(session);
CREATE INDEX IF NOT EXISTS usage_ts ON usage(ts);
CREATE TABLE IF NOT EXISTS loop_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, session TEXT NOT NULL, action TEXT NOT NULL,
    kind TEXT NOT NULL, key TEXT NOT NULL, evidence TEXT NOT NULL, message TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS halt (id INTEGER PRIMARY KEY CHECK (id = 1), active INTEGER NOT NULL,
    reason TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS sessions (session TEXT PRIMARY KEY, paused INTEGER NOT NULL DEFAULT 0,
    reason TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS nudges (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
    session TEXT NOT NULL, message TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS prices (key TEXT PRIMARY KEY, input REAL NOT NULL, output REAL NOT NULL,
    cache_read REAL, cache_write REAL);
CREATE TABLE IF NOT EXISTS notices (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
    limit_id TEXT NOT NULL, bucket TEXT NOT NULL, level REAL NOT NULL, value REAL NOT NULL,
    UNIQUE (limit_id, bucket, level));
"""


class Limit(BaseModel):
    """A user-set limit. None exist by default."""

    id: str = Field(default_factory=lambda: "lim_" + secrets.token_hex(4))
    scope: Scope
    metric: Metric
    value: float = Field(gt=0)
    session: str | None = None
    """For session-scoped limits: one session, or every session when unset."""
    provider: str | None = None
    window_s: float | None = Field(default=None, gt=0)
    """Rate and velocity window (rate defaults to 60 s)."""

    @model_validator(mode="after")
    def _shape(self) -> Limit:
        allowed = {
            "call": {"input_tokens", "output_tokens"},
            "session": {
                "dollars",
                "calls",
                "turns",
                "tool_calls",
                "wall_clock_s",
                "gpu_minutes",
                "gpu_dollars",
            },
            "day": {"dollars"},
            "month": {"dollars"},
            "rate": {"calls", "concurrent_sessions"},
            "velocity": {"dollars"},
        }[self.scope]
        if self.metric not in allowed:
            raise ValueError(
                f"{self.scope} limits take {', '.join(sorted(allowed))}, not {self.metric}"
            )
        if self.scope == "velocity" and self.window_s is None:
            raise ValueError("a velocity limit needs window_s, e.g. 600 for 'per 10 minutes'")
        return self

    def describe(self) -> str:
        where = f" for {self.session}" if self.session else ""
        prov = f" on {self.provider}" if self.provider else ""
        window = f" per {self.window_s:g}s" if self.window_s else ""
        return f"{self.metric} ≤ {self.value:g} per {self.scope}{window}{where}{prov}"


@dataclass
class Refusal:
    limit: Limit | None
    message: str


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            str(self.path), isolation_level=None, check_same_thread=False, timeout=30
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._lock = threading.RLock()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def close(self) -> None:
        self._db.close()

    # ------------------------------------------------------------------ limits

    def limits(self) -> list[Limit]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM limits ORDER BY created").fetchall()
        return [
            Limit(
                **{
                    k: r[k]
                    for k in ("id", "scope", "metric", "value", "session", "provider", "window_s")
                }
            )
            for r in rows
        ]

    def set_limit(self, limit: Limit) -> Limit:
        with self.tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO limits VALUES (?,?,?,?,?,?,?,?)",
                (
                    limit.id,
                    limit.scope,
                    limit.metric,
                    limit.value,
                    limit.session,
                    limit.provider,
                    limit.window_s,
                    time.time(),
                ),
            )
        return limit

    def remove_limit(self, limit_id: str) -> bool:
        with self.tx() as db:
            return db.execute("DELETE FROM limits WHERE id = ?", (limit_id,)).rowcount > 0

    # ------------------------------------------------------------------ prices

    def price_table(self) -> PriceTable:
        with self._lock:
            rows = self._db.execute("SELECT * FROM prices").fetchall()
        return PriceTable(
            {
                r["key"]: Price(r["input"], r["output"], r["cache_read"], r["cache_write"])
                for r in rows
            }
        )

    def set_price(self, provider: str, model: str, price: Price) -> None:
        with self.tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?)",
                (
                    f"{provider}:{model}",
                    price.input,
                    price.output,
                    price.cache_read,
                    price.cache_write,
                ),
            )

    # ------------------------------------------------------------------ halt / pause

    def halted(self) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT active, reason FROM halt WHERE id = 1").fetchone()
        return (row["reason"] or "halted") if row and row["active"] else None

    def set_halt(self, active: bool, reason: str = "") -> None:
        with self.tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO halt VALUES (1, ?, ?, ?)",
                (int(active), reason, time.time()),
            )

    def paused(self, session: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT paused, reason FROM sessions WHERE session = ?", (session,)
            ).fetchone()
        return (row["reason"] or "paused") if row and row["paused"] else None

    def pause(self, session: str, reason: str) -> None:
        with self.tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?, 1, ?, ?)",
                (session, reason, time.time()),
            )

    def resume(self, session: str | None = None) -> None:
        with self.tx() as db:
            if session is None:
                db.execute("UPDATE sessions SET paused = 0")
                db.execute("INSERT OR REPLACE INTO halt VALUES (1, 0, '', ?)", (time.time(),))
            else:
                db.execute("UPDATE sessions SET paused = 0 WHERE session = ?", (session,))

    # ------------------------------------------------------------------ nudges and loop events

    def add_nudge(self, session: str, message: str) -> None:
        with self.tx() as db:
            db.execute(
                "INSERT INTO nudges (ts, session, message) VALUES (?,?,?)",
                (time.time(), session, message),
            )

    def take_nudges(self, session: str) -> list[str]:
        with self.tx() as db:
            rows = db.execute(
                "SELECT id, message FROM nudges WHERE session = ? AND delivered = 0 ORDER BY id",
                (session,),
            ).fetchall()
            db.execute(
                "UPDATE nudges SET delivered = 1 WHERE session = ? AND delivered = 0", (session,)
            )
        return [r["message"] for r in rows]

    def loop_event(self, session: str, verdict: dict[str, Any]) -> None:
        with self.tx() as db:
            db.execute(
                "INSERT INTO loop_events (ts, session, action, kind, key, evidence, message) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    time.time(),
                    session,
                    verdict["action"],
                    verdict["kind"],
                    verdict["key"],
                    json.dumps(verdict["evidence"]),
                    verdict["message"],
                ),
            )

    def loop_events(self, session: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            sql = (
                "SELECT * FROM loop_events"
                + (" WHERE session = ?" if session else "")
                + " ORDER BY id"
            )
            rows = self._db.execute(sql, (session,) if session else ()).fetchall()
        return [dict(r) | {"evidence": json.loads(r["evidence"])} for r in rows]

    # ------------------------------------------------------------------ reserve / settle

    def reserve(
        self,
        session: str,
        provider: str,
        model: str,
        *,
        input_bound: int,
        max_output: int,
        price: Price | None,
        now: float | None = None,
    ) -> tuple[str | None, Refusal | None]:
        """Check every applicable limit; hold the worst-case cost if a dollar limit applies."""
        now = now or time.time()
        with self.tx() as db:
            limits = [
                lim
                for lim in self._load(db)
                if lim.metric in GATEWAY_METRICS
                and (lim.provider is None or lim.provider == provider)
                and (lim.session is None or lim.session == session)
            ]
            for lim in limits:
                if lim.scope == "call" and lim.metric == "input_tokens" and input_bound > lim.value:
                    return None, Refusal(
                        lim,
                        f"this call may send up to {input_bound:,} input tokens; "
                        f"limit {lim.describe()}",
                    )
                if lim.scope == "call" and lim.metric == "output_tokens" and max_output > lim.value:
                    return None, Refusal(
                        lim, f"max output {max_output:,} tokens exceeds {lim.describe()}"
                    )
                if lim.metric in ("calls", "turns") and lim.scope in ("session", "rate"):
                    since = now - (lim.window_s or 60) if lim.scope == "rate" else 0.0
                    where, args = self._where(lim, session, provider, since)
                    used = db.execute(f"SELECT COUNT(*) FROM usage WHERE {where}", args).fetchone()[
                        0
                    ]
                    held = db.execute(
                        f"SELECT COUNT(*) FROM reservations WHERE {where.replace('ts', 'created')}",
                        args,
                    ).fetchone()[0]
                    if used + held + 1 > lim.value:
                        return None, Refusal(
                            lim, f"{lim.describe()} reached ({used + held:g} used)"
                        )
            dollar = [lim for lim in limits if lim.metric == "dollars"]
            if not dollar:
                return None, None
            if price is None:
                return None, Refusal(
                    dollar[0],
                    f"price of {provider}:{model} is unknown, so the dollar limit "
                    f"{dollar[0].describe()} cannot be enforced; set it with "
                    f"`wspx limits price {provider}:{model} <input> <output>`",
                )
            worst = (input_bound * max(price.input, price.write) + max_output * price.output) / 1e6
            for lim in dollar:
                spent = self._spent(db, lim, session, provider, now)
                if spent + worst > lim.value + 1e-12:
                    return None, Refusal(
                        lim,
                        f"{lim.describe()}: ${spent:.4f} spent or held, this call may cost "
                        f"up to ${worst:.4f}",
                    )
            rid = "res_" + secrets.token_hex(6)
            db.execute(
                "INSERT INTO reservations VALUES (?,?,?,?,?,?)",
                (rid, session, provider, model, worst, now),
            )
            return rid, None

    def settle(
        self,
        reservation: str | None,
        session: str,
        provider: str,
        model: str,
        *,
        input: int = 0,
        output: int = 0,
        cache_read: int = 0,
        cache_write: int = 0,
        cost: float | None,
        status: int = 200,
        error: str | None = None,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        """Record a finished call. Returns any limit notices it triggered (50/80/100%)."""
        now = now or time.time()
        with self.tx() as db:
            held = 0.0
            if reservation is not None:
                row = db.execute(
                    "SELECT amount FROM reservations WHERE id = ?", (reservation,)
                ).fetchone()
                held = float(row["amount"]) if row else 0.0
                db.execute("DELETE FROM reservations WHERE id = ?", (reservation,))
            charged = cost if cost is not None else held  # no usage: keep the whole hold
            db.execute(
                "INSERT INTO usage (ts, session, provider, model, input, output, cache_read, cache_write, "
                "cost, charged, status, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    now,
                    session,
                    provider,
                    model,
                    input,
                    output,
                    cache_read,
                    cache_write,
                    cost,
                    charged,
                    status,
                    error,
                ),
            )
            return self._notices(db, session, provider, now)

    def _load(self, db: sqlite3.Connection) -> list[Limit]:
        rows = db.execute("SELECT * FROM limits").fetchall()
        return [
            Limit(
                **{
                    k: r[k]
                    for k in ("id", "scope", "metric", "value", "session", "provider", "window_s")
                }
            )
            for r in rows
        ]

    @staticmethod
    def _bucket_start(scope: str, now: float, window: float | None) -> float:
        if scope == "day":
            d = datetime.fromtimestamp(now, UTC)
            return datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()
        if scope == "month":
            d = datetime.fromtimestamp(now, UTC)
            return datetime(d.year, d.month, 1, tzinfo=UTC).timestamp()
        if scope in ("velocity", "rate"):
            return now - (window or 60)
        return 0.0

    def _where(
        self, lim: Limit, session: str, provider: str, since: float
    ) -> tuple[str, tuple[Any, ...]]:
        """Rows a limit counts: session limits see their session; the others are global
        unless the limit names a session or a provider."""
        clauses: list[str] = ["ts >= ?"]
        args: list[Any] = [since]
        if lim.scope == "session" or lim.session:
            clauses.append("session = ?")
            args.append(session)
        if lim.provider:
            clauses.append("provider = ?")
            args.append(provider)
        return " AND ".join(clauses), tuple(args)

    def _spent(
        self, db: sqlite3.Connection, lim: Limit, session: str, provider: str, now: float
    ) -> float:
        since = self._bucket_start(lim.scope, now, lim.window_s)
        where, args = self._where(lim, session, provider, since)
        used = db.execute(
            f"SELECT COALESCE(SUM(charged), 0) FROM usage WHERE {where}", args
        ).fetchone()[0]
        held = db.execute(
            f"SELECT COALESCE(SUM(amount), 0) FROM reservations WHERE {where.replace('ts', 'created')}",
            args,
        ).fetchone()[0]
        return float(used) + float(held)

    def _notices(
        self, db: sqlite3.Connection, session: str, provider: str, now: float
    ) -> list[dict[str, Any]]:
        out = []
        for lim in self._load(db):
            if lim.metric != "dollars" or (lim.provider and lim.provider != provider):
                continue
            if lim.session and lim.session != session:
                continue
            since = self._bucket_start(lim.scope, now, lim.window_s)
            where, args = self._where(lim, session, provider, since)
            spent = float(
                db.execute(
                    f"SELECT COALESCE(SUM(charged), 0) FROM usage WHERE {where}", args
                ).fetchone()[0]
            )
            bucket = f"{lim.scope}:{session if lim.scope == 'session' else int(since)}"
            for level in NOTICE_LEVELS:
                if spent >= level * lim.value:
                    cur = db.execute(
                        "INSERT OR IGNORE INTO notices (ts, limit_id, bucket, level, value) "
                        "VALUES (?,?,?,?,?)",
                        (now, lim.id, bucket, level, spent),
                    )
                    if cur.rowcount:
                        out.append(
                            {
                                "limit": lim.id,
                                "describe": lim.describe(),
                                "level": level,
                                "spent": spent,
                            }
                        )
        return out

    # ------------------------------------------------------------------ reporting

    def usage_summary(self, *, session: str | None = None, since: float = 0.0) -> dict[str, Any]:
        with self._lock:
            where = "ts >= ?" + (" AND session = ?" if session else "")
            args: tuple[Any, ...] = (since, session) if session else (since,)
            rows = self._db.execute(
                f"SELECT session, provider, model, COUNT(*) n, SUM(input) i, SUM(output) o, SUM(cache_read) cr, "
                f"SUM(cache_write) cw, SUM(cost) c, SUM(charged) ch, SUM(cost IS NULL) unknown "
                f"FROM usage WHERE {where} GROUP BY session, provider, model",
                args,
            ).fetchall()
            held = self._db.execute("SELECT COALESCE(SUM(amount), 0) FROM reservations").fetchone()[
                0
            ]
        groups = [dict(r) for r in rows]
        return {
            "groups": groups,
            "total_charged": sum(g["ch"] or 0 for g in groups),
            "unknown_cost_calls": sum(g["unknown"] or 0 for g in groups),
            "held": held,
        }

    def notices(self, since: float = 0.0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM notices WHERE ts >= ? ORDER BY id", (since,)
            ).fetchall()
        return [dict(r) for r in rows]
