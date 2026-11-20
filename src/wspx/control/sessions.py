"""Sessions and their lifecycle (§3.2, §5.13).

Activity is defined by the session, not by lab traffic: a session is active
while its driver is running (thinking counts) or someone has it open in the
UI. Active sessions keep their lab hot and its lease renewed; idle ones keep
it for the grace period, then release the GPU. Store-served tools answer
while the lab is cold, and a lab-bound call wakes it.
"""

from __future__ import annotations

import contextlib
import json
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import fields
from typing import Any

from wspx.auth import mint
from wspx.control.backend import Backend, LabHandle, RunnerHandle, SessionRecord
from wspx.control.gateway_client import GatewayAdmin
from wspx.control.tracedb import TraceDB
from wspx.errors import LabCancelled, WspxError
from wspx.store import Store, short
from wspx.store.overlay import apply_journal
from wspx.tools import TOOLS

LEASE_S = 300.0
RECENT_CALL_S = 120.0
STORE_SERVED = frozenset({"info", "ls"})
_CODE_RE = re.compile(r"\b([EW]\d{3})\b")


class SessionManager:
    def __init__(
        self,
        store: Store,
        trace: TraceDB,
        backend: Backend,
        *,
        gateway: GatewayAdmin | None = None,
        secret: str = "",
        control_url: str = "",
        gateway_url: str = "",
    ) -> None:
        self.store = store
        self.trace = trace
        self.backend = backend
        self.gateway = gateway
        self.secret = secret
        self.control_url = control_url
        self.gateway_url = gateway_url
        self.records: dict[str, SessionRecord] = {}
        self.labs: dict[str, LabHandle] = {}
        self.runners: dict[str, RunnerHandle] = {}
        self.commands: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self.viewers: dict[str, int] = defaultdict(int)
        self.last_activity: dict[str, float] = {}
        self.idle_since: dict[str, float] = {}
        self.workspace: dict[str, str] = {}
        self.halted = False
        self._lab_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._lab_gen: dict[str, int] = defaultdict(int)
        """Bumped by every lab stop: a start begun under an older value is abandoned."""
        self._lock = threading.RLock()
        for data in trace.sessions():
            known = {f.name for f in fields(SessionRecord)}
            record = SessionRecord(**{k: v for k, v in data.items() if k in known})
            record.tier = "cold"
            self.records[record.id] = record

    # ------------------------------------------------------------------ sessions

    def create(
        self,
        *,
        model: str,
        driver: str | None = None,
        task: str = "",
        gpu: str | None = None,
        branch: str | None = None,
        revision: str | None = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        grace_s: float = 900.0,
        keep_warm: str = "auto",
        project: str = "",
        snapshot: str | None = None,
    ) -> SessionRecord:
        if self.halted:
            raise WspxError("wspx is halted; `wspx resume` first")
        if snapshot is not None and not self.store.objects.has(snapshot):
            raise WspxError(f"project snapshot {snapshot[:12]} was not uploaded")
        sid = "s_" + secrets.token_hex(4)
        record = SessionRecord(
            sid,
            model,
            driver,
            task,
            gpu,
            branch or sid,
            revision,
            dtype,
            trust_remote_code,
            grace_s=grace_s,
            keep_warm=keep_warm,
            project=project,
            snapshot=snapshot,
            created=time.time(),
        )
        self.records[sid] = record
        self._save(record)
        self.trace.event(
            sid, "session", {"status": "created", "model": model, "driver": driver, "task": task}
        )
        self.touch(sid)
        threading.Thread(target=self._prewarm, args=(sid,), daemon=True).start()
        return record

    def get(self, sid: str) -> SessionRecord:
        if sid not in self.records:
            raise WspxError(f"no session {sid}")
        return self.records[sid]

    def _save(self, record: SessionRecord) -> None:
        self.trace.put_session(record.to_dict())

    def _prewarm(self, sid: str) -> None:
        record = self.get(sid)
        try:
            self.lab(sid)
        except LabCancelled:
            return  # stopped while the lab started: nothing to report
        except Exception as exc:
            record.status, record.reason = "error", f"lab failed to start: {exc}"
            self._save(record)
            self.trace.event(sid, "error", {"error": record.reason})
            return
        if record.driver and record.status in ("idle", "running"):
            self.start_driver(sid)

    def start_driver(self, sid: str) -> None:
        """Start the session's driver. A session never has two: starting a running one is a no-op."""
        record = self.get(sid)
        if not record.driver:
            raise WspxError(f"{sid} has no driver")
        env = {
            "WSPX_SESSION": sid,
            "WSPX_DRIVER": record.driver,
            "WSPX_TASK": record.task,
            "WSPX_CONTROL_URL": self.control_url,
            "WSPX_GATEWAY_URL": self.gateway_url,
            "WSPX_RUNNER_TOKEN": mint(self.secret, sid, scope="runner"),
            "WSPX_LLM_TOKEN": mint(self.secret, sid, scope="llm"),
        }
        with self._lock:
            runner = self.runners.get(sid)
            if runner is not None and runner.alive():
                if record.status == "running":
                    return
                runner.stop()  # a paused or stopped driver still winding down
            record.status = "running"
            self._save(record)
            self.runners[sid] = self.backend.start_runner(record, env)
        self.trace.event(sid, "session", {"status": "running"})

    # ------------------------------------------------------------------ labs

    def lab(self, sid: str) -> LabHandle:
        """The session's lab, started if cold (blocks until it is up).

        Stopping the lab abandons a start in progress, and fails the calls
        that were waiting for it, instead of starting the lab again for them.
        """
        gen = self._lab_gen[sid]
        with self._lab_locks[sid]:
            if sid in self.labs:
                return self.labs[sid]
            if self._lab_gen[sid] != gen:
                raise LabCancelled(f"{sid}'s lab was stopped")
            record = self.get(sid)
            self.trace.event(sid, "lab", {"tier": "starting"})
            started = time.time()
            try:
                handle = self.backend.start_lab(record, cancelled=lambda: self._lab_gen[sid] != gen)
            except LabCancelled:
                self.trace.event(sid, "lab", {"tier": "cold", "reason": "start cancelled"})
                raise
            record.load_seconds = time.time() - started
            record.tier = "hot"
            self.labs[sid] = handle
            self._save(record)
            self._persist(sid, handle.heartbeat(LEASE_S))
            self.trace.event(sid, "lab", {"tier": "hot", "load_seconds": record.load_seconds})
            return handle

    def stop_lab(self, sid: str, reason: str) -> None:
        self._lab_gen[sid] += 1  # before the lock: a start in progress holds it until it gives up
        with self._lab_locks[sid]:
            handle = self.labs.pop(sid, None)
            if handle is None:
                return
            with contextlib.suppress(Exception):
                self._persist(sid, handle.heartbeat(0))
            handle.stop()
            record = self.get(sid)
            record.tier = "cold"
            self._save(record)
            self.trace.event(sid, "lab", {"tier": "cold", "reason": reason})

    def _persist(self, sid: str, response: dict[str, Any]) -> None:
        journal = response.get("journal")
        if journal:
            apply_journal(self.store, journal)
        if response.get("workspace"):
            self.workspace[sid] = response["workspace"]
        record = self.records.get(sid)
        fingerprint = response.get("fingerprint")
        if record is not None and fingerprint and record.fingerprint != fingerprint:
            record.fingerprint = fingerprint
            self._save(record)

    # ------------------------------------------------------------------ tools

    def tool(
        self,
        sid: str,
        name: str,
        args: dict[str, Any],
        *,
        call_id: str | None = None,
        source: str = "driver",
        turn: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = self.get(sid)
        if name not in TOOLS:
            return {"summary": f"unknown tool {name!r}", "ok": False, "handles": [], "data": {}}
        if self.halted:
            return {
                "summary": "wspx is halted; `wspx resume` to continue",
                "ok": False,
                "handles": [],
                "data": {},
            }
        if source == "driver" and record.status == "stopped":
            # a driver call still in flight when the session was stopped must not wake the lab
            return {"summary": f"{sid} was stopped", "ok": False, "handles": [], "data": {}}
        call_id = call_id or "c_" + secrets.token_hex(5)
        self.touch(sid)
        started = time.time()
        result = self._store_served(record, name, args) if sid not in self.labs else None
        note = ""
        if result is None:
            if sid not in self.labs:
                note = "(lab cold start {:.0f} s)"
            ctx = {"caller": source, "driver": record.driver, **(context or {})}
            try:
                result = self.lab(sid).call(name, args, call_id, ctx)
            except WspxError as exc:  # recorded like any other failed call, never dropped
                result = {"summary": f"{name}: {exc}", "ok": False, "handles": [], "data": {}}
                note = ""
            else:
                self._persist(sid, result)
                result.pop("journal", None)
            if note:
                note = note.format(time.time() - started)
        latency = time.time() - started
        result["summary"] = (result.get("summary") or "") + (f"\n{note}" if note else "")
        result["call_id"] = call_id
        result["summary"] += self._observe(sid, name, args, result, source)
        self.trace.tool_call(
            sid, call_id, name, args, result, turn=turn, latency=latency, source=source
        )
        return result

    def _store_served(
        self, record: SessionRecord, name: str, args: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Answer from the store while the lab is cold, when the tool allows it."""
        if name not in STORE_SERVED or record.fingerprint is None:
            return None
        raw = self.store.read_file(f"models/{record.fingerprint.replace(':', '_')}/ir.json")
        if raw is None:
            return None
        from wspx.lab.ir.model_ir import ModelIR

        ir = ModelIR.from_dict(json.loads(raw))
        if name == "info":
            head = self.store.refs.read(f"sessions/{record.id}/HEAD")
            return {
                "summary": ir.render_info()
                + f"\nHEAD {short(head) if head else '(base)'} · lab cold",
                "ok": True,
                "handles": [],
                "data": {"summary": ir.summary(), "served_by": "store"},
            }
        scope = args.get("scope") or "L*.*"
        try:
            targets = ir.resolve(scope)
        except WspxError as exc:
            return {"summary": f"ls: {exc}", "ok": False, "handles": [], "data": {}}
        lines = [f"{t.address:<28} {t.param}" for t in targets[: int(args.get("limit", 100))]]
        return {
            "summary": "\n".join(lines) + "\n(lab cold: served from the store)",
            "ok": True,
            "handles": [],
            "data": {"served_by": "store"},
        }

    def _observe(
        self, sid: str, name: str, args: dict[str, Any], result: dict[str, Any], source: str
    ) -> str:
        """Repetition across tool calls also counts as a loop (the gateway decides).

        The gateway already sees the built-in driver's calls in the model's responses, so
        only other callers' calls are counted here (each call is counted once). Outcomes
        are only visible here.
        """
        if self.gateway is None:
            return ""
        notes = []
        try:
            verdicts = (
                []
                if source == "driver"
                else [self.gateway.observe(sid, "call", {"tool": name, "args": args}, label=name)]
            )
            codes = sorted(set(_CODE_RE.findall(result.get("summary") or "")))
            if codes:
                verdicts.append(
                    self.gateway.observe(sid, "outcome", ",".join(codes), label=",".join(codes))
                )
            no_op = "E105" in codes
            if no_op:
                verdicts.append(
                    self.gateway.observe(sid, "outcome", "no-op edit", label="no-op edit")
                )
        except WspxError:
            return ""
        for verdict in verdicts:
            if verdict is None:
                continue
            self.trace.event(sid, "loop", verdict)
            if verdict["action"] == "stop":
                self.pause(sid, f"loop detected: {verdict['message']}", source="wspx")
                notes.append(f"\n⚠ session paused: {verdict['message']}")
            else:
                notes.append(f"\n⚠ {verdict['message']}")
        return "".join(notes)

    # ------------------------------------------------------------------ control

    def touch(self, sid: str) -> None:
        self.last_activity[sid] = time.time()
        self.idle_since.pop(sid, None)

    def command(
        self, sid: str, kind: str, text: str | None = None, *, source: str = "human"
    ) -> None:
        self.get(sid)
        self.commands[sid].append({"type": kind, **({"text": text} if text else {})})
        self.trace.event(sid, "command", {"type": kind, "text": text}, source=source)

    def take_commands(self, sid: str) -> list[dict[str, Any]]:
        out = []
        while self.commands[sid]:
            out.append(self.commands[sid].popleft())
        return out

    def pause(self, sid: str, reason: str, *, source: str = "human") -> None:
        record = self.get(sid)
        record.status, record.reason = "paused", reason
        self._save(record)
        self.command(sid, "pause", source=source)
        if self.gateway is not None:
            with contextlib.suppress(WspxError):
                self.gateway.pause(sid, reason)

    def sync_halt(self) -> None:
        """Take the halt state from the gateway, which keeps it across control-plane restarts."""
        if self.gateway is not None:
            with contextlib.suppress(WspxError):
                self.halted = self.gateway.halted()

    def resume(self, sid: str | None = None) -> None:
        targets = [sid] if sid else [s for s, r in self.records.items() if r.status == "paused"]
        if sid is None:
            self.halted = False
        if self.gateway is not None:
            self.gateway.resume(sid)
        for s in targets:
            record = self.get(s)
            record.status, record.reason = "idle", ""
            self._save(record)
            self.trace.event(s, "session", {"status": "resumed"})
            if record.driver:
                self.start_driver(s)

    def kill(self, sid: str | None = None) -> int:
        """Stop drivers, revoke leases and terminate labs. ``None`` kills everything and halts."""
        targets = [sid] if sid else list(self.records)
        if sid is None:
            self.halted = True
            if self.gateway is not None:
                with contextlib.suppress(WspxError):
                    self.gateway.halt("kill --all")
        for s in targets:
            record = self.get(s)
            record.status = "stopped"
            self._save(record)
            self.command(s, "stop")
            with self._lock:
                runner = self.runners.pop(s, None)
            if runner is not None:
                runner.stop()
            self.stop_lab(s, "killed")
        return len(targets)

    def runner_event(self, sid: str, event: dict[str, Any]) -> None:
        self.touch(sid)
        kind = event.get("type")
        if kind == "turn":
            self.trace.turn(sid, event)
        elif kind == "end":
            record = self.get(sid)
            record.status = {
                "done": "done",
                "stopped": "stopped",
                "paused": "paused",
                "refused": "done",
                "halted": "paused",
                "error": "error",
            }.get(event.get("status", ""), "idle")
            record.reason = event.get("reason", "")
            self._save(record)
            self.trace.event(sid, "session", {"status": record.status, "reason": record.reason})
        elif kind != "tool_call":  # tool calls are recorded when the control plane runs them
            self.trace.event(sid, str(kind), event, source="driver")

    # ------------------------------------------------------------------ lifecycle

    def active(self, sid: str, now: float) -> bool:
        runner = self.runners.get(sid)
        if runner is not None and runner.alive():
            return True
        if self.viewers.get(sid, 0) > 0:
            return True
        return now - self.last_activity.get(sid, 0.0) < RECENT_CALL_S

    def tick(self, now: float | None = None) -> None:
        """Renew leases of hot and grace labs; release labs whose grace has run out."""
        now = now or time.time()
        for sid in list(self.labs):
            record = self.records[sid]
            if self.active(sid, now) or record.keep_warm == "always":
                self.idle_since.pop(sid, None)
                record.tier = "hot"
            else:
                since = self.idle_since.setdefault(sid, now)
                if now - since >= record.grace_s:
                    self.stop_lab(sid, "idle past the grace period")
                    continue
                record.tier = "grace"
            try:
                self._persist(sid, self.labs[sid].heartbeat(LEASE_S))
            except Exception as exc:
                self.labs.pop(sid, None)
                record.tier = "cold"
                self.trace.event(sid, "lab", {"tier": "cold", "reason": f"lost: {exc}"})
            self._save(record)
