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
from wspx.control.extra_tools import Extra, Remote, project_tools
from wspx.control.gateway_client import GatewayAdmin
from wspx.control.tracedb import TraceDB
from wspx.errors import LabCancelled, LabUnavailable, WspxError
from wspx.store import Store, short
from wspx.store.overlay import apply_journal
from wspx.tools import TOOLS
from wspx.tools.dispatch import PROJECT_TOOL
from wspx.tools.schemas import JOB_TOOLS

LEASE_S = 300.0
MISSED_HEARTBEATS = 3
"""Consecutive failed heartbeats before a lab is given up on (and terminated)."""
RECENT_CALL_S = 120.0
STORE_SERVED = frozenset({"arch"})
_CODE_RE = re.compile(r"\b([EW]\d{3})\b")
_DIAG_RE = re.compile(r"^\s*([EW]\d{3}\b.*)$", re.MULTILINE)


def outcome_of(summary: str) -> tuple[str, str]:
    """A tool result's outcome for the loop detector, and its label. One outcome per call: a
    no-op edit's E105 and its "no-op edit" reported apart alternated, so successive no-op
    edits never added up to a run. Otherwise the same result is the same diagnostics, not
    the same codes. Seen live: a W203 at 67% (refused), the fixed edit's W203 at 15%, and
    check repeating that one counted as one result three times running."""
    codes = sorted(set(_CODE_RE.findall(summary)))
    if "E105" in codes:
        return "no-op edit", "no-op edit"
    label = ",".join(codes)
    lines = sorted({line.strip() for line in _DIAG_RE.findall(summary)})
    return (" | ".join(lines) or label), label


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
        self.mcp_connect: Any = None
        """How remote MCP servers are reached (tests connect in-process); ``None``: over HTTP."""
        self.halted = False
        self._halt_epoch = 0
        """Bumped by every kill --all and resume, so a stale halt read cannot undo them."""
        self._lab_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._lab_gen: dict[str, int] = defaultdict(int)
        self._inflight: dict[str, threading.Event] = {}
        """Driver calls running now, by call id: a restarted runner re-sending one waits for it."""
        self._missed: dict[str, int] = defaultdict(int)
        self._epoch: dict[str, int] = defaultdict(int)
        """State-changing tool calls so far, per session: part of a repeated call's key."""
        self._outcome_run: set[str] = set()
        """Sessions whose last tool call reported a diagnostic: a clean call ends that run."""
        self._lab_why: dict[str, str] = {}
        """Why the last generation ended: calls that waited on it fail with this."""
        """Bumped by every lab stop: a start begun under an older value is abandoned."""
        self._lock = threading.RLock()
        for data in trace.sessions():
            known = {f.name for f in fields(SessionRecord)}
            record = SessionRecord(**{k: v for k, v in data.items() if k in known})
            record.tier = "cold"
            if record.status == "starting":  # nothing is queued across a restart
                record.status = "idle"
            if record.lab_since is not None:  # its lab did not survive the restart: count it all
                record.lab_seconds = record.lab_used(time.time())
                record.lab_since = None
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
        engine: str = "auto",
        driver_images: str = "auto",
        cpu: float | None = None,
        memory: int | None = None,
        packages: list[str] | None = None,
        grace_s: float = 900.0,
        keep_warm: str = "auto",
        project: str = "",
        snapshot: str | None = None,
        tools_deny: list[str] | None = None,
        max_lab_s: float | None = None,
        attachments: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        mcp: list[dict[str, Any]] | None = None,
    ) -> SessionRecord:
        if self.halted:
            raise WspxError("wspx is halted; `wspx resume` first")
        project_tools(tools or [])  # names are checked before anything is created
        check_engine_gpu(engine, gpu)
        if snapshot is not None and not self.store.objects.has(snapshot):
            raise WspxError(f"project snapshot {snapshot[:12]} was not uploaded")
        unknown = sorted(set(tools_deny or []) - set(TOOLS))
        if unknown:
            raise WspxError(f"no tool {unknown[0]!r} to turn off; tools: {', '.join(TOOLS)}")
        if max_lab_s is not None and max_lab_s <= 0:
            raise WspxError("the GPU time limit must be positive")
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
            status="starting" if driver else "idle",
            tools_deny=sorted(set(tools_deny or [])),
            max_lab_s=max_lab_s,
            engine=engine,
            driver_images=driver_images,
            cpu=cpu,
            memory=memory,
            packages=_packages(packages or []),
        )
        self.records[sid] = record
        if attachments:
            self.allow(record, attachments)
            record.extra["task_attachments"] = [a["oid"] for a in attachments]
        if tools:
            record.extra["project_tools"] = tools
        if mcp:
            record.extra["mcp"] = mcp
        self._save(record)
        self.trace.event(
            sid,
            "session",
            {
                "status": "created",
                "model": model,
                "driver": driver,
                "task": task,
                "gpu": gpu,
                "engine": engine,
                "grace_s": grace_s,
                "tools_deny": record.tools_deny,
                "max_lab_s": max_lab_s,
            },
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
        if record.driver and record.status in ("starting", "running"):
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
            "WSPX_TOOLS_DENY": ",".join(record.tools_deny),
            "WSPX_ATTACHMENTS": json.dumps(self.task_attachments(record)),
            "WSPX_DRIVER_IMAGES": record.driver_images,
            "WSPX_EXTRA_TOOLS": json.dumps(
                [t.offered() for t in self.extra_tools(record).values()]
            ),
            # a cloud runner restarted on this same input resumes from its checkpoint
            "WSPX_RUNNER_ID": "r_" + secrets.token_hex(6),
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
                raise LabCancelled(self._lab_why.get(sid, f"{sid}'s lab was stopped"))
            record = self.get(sid)
            if record.max_lab_s is not None and record.lab_seconds >= record.max_lab_s:
                raise LabUnavailable(
                    f"{sid} has used its GPU time ({_minutes(record.max_lab_s)}); "
                    "start a new session to continue"
                )
            self.trace.event(sid, "lab", {"tier": "starting"})
            started = time.time()
            try:
                handle = self.backend.start_lab(record, cancelled=lambda: self._lab_gen[sid] != gen)
            except LabCancelled:
                record.lab_seconds += time.time() - started
                self._save(record)
                self.trace.event(sid, "lab", {"tier": "cold", "reason": "start cancelled"})
                raise
            except Exception as exc:
                # fail the calls queued behind this start too, rather than each starting a lab
                record.lab_seconds += time.time() - started
                self._save(record)
                self._lab_why[sid] = f"the lab failed to start: {exc}"
                self._lab_gen[sid] += 1
                self.trace.event(sid, "lab", {"tier": "cold", "reason": self._lab_why[sid]})
                raise LabUnavailable(self._lab_why[sid]) from exc
            record.load_seconds = time.time() - started
            record.tier = "hot"
            record.lab_since = started
            self.labs[sid] = handle
            self._save(record)
            beat = handle.heartbeat(LEASE_S)
            self._persist(sid, beat)
            self.trace.event(
                sid,
                "lab",
                {
                    "tier": "hot",
                    "load_seconds": record.load_seconds,
                    **({"network": beat["network"]} if beat.get("network") else {}),
                },
            )
            return handle

    def stop_lab(self, sid: str, reason: str, *, now: float | None = None) -> None:
        self._lab_why[sid] = f"{sid}'s lab was stopped: {reason}"
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
            record.lab_seconds = record.lab_used(now or time.time())
            record.lab_since = None
            self._save(record)
            self.trace.event(
                sid, "lab", {"tier": "cold", "reason": reason, "gpu_seconds": record.lab_seconds}
            )

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
        extra = self.extra_tools(record) if name not in TOOLS else {}
        if name not in TOOLS and name not in extra:
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
        if source == "driver" and name in record.tools_deny:
            denied_id = call_id or "c_" + secrets.token_hex(5)
            refusal: dict[str, Any] = {
                "summary": f"{name} is turned off for this session",
                "ok": False,
                "handles": [],
                "data": {},
                "call_id": denied_id,
            }
            self.trace.tool_call(
                sid, denied_id, name, args, refusal, turn=turn, latency=0.0, source=source
            )
            return refusal
        if call_id and source == "driver":
            earlier = self._recorded(sid, call_id)
            if earlier is not None:
                return earlier
        call_id = call_id or "c_" + secrets.token_hex(5)
        try:
            return self._run_tool(record, name, args, call_id, source, turn, context)
        finally:
            done = self._inflight.pop(call_id, None)
            if done is not None:
                done.set()

    def _recorded(self, sid: str, call_id: str) -> dict[str, Any] | None:
        """The result of a driver call already run (or running) under this id, else ``None``.

        Only a restarted runner re-sends an id; answering from the record keeps an edit
        or a commit from being applied twice. A call not seen before is marked running.
        """
        with self._lock:
            running = self._inflight.get(call_id)
            row = None if running is not None else self.trace.call(call_id)
            if running is None and row is None:
                self._inflight[call_id] = threading.Event()
                return None
        if running is not None:
            running.wait()
            row = self.trace.call(call_id)
        if row is None or row.get("session") != sid:
            return {"summary": f"call {call_id} did not complete", "ok": False, "handles": []}
        handles = [a["id"] for a in row.get("artefacts") or [] if a.get("kind") != "commit"]
        return {
            "summary": (row.get("summary") or "")
            + "\n(answered from the record: this call id already ran)",
            "ok": bool(row.get("ok")),
            "handles": handles,
            "data": json.loads(row["data"]) if row.get("data") else {},
            "call_id": call_id,
        }

    def _run_tool(
        self,
        record: SessionRecord,
        name: str,
        args: dict[str, Any],
        call_id: str,
        source: str,
        turn: int | None,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        sid = record.id
        self.touch(sid)
        started = time.time()
        result = self._store_served(record, name, args) if sid not in self.labs else None
        note = ""
        if result is None:
            if sid not in self.labs:
                note = "(lab cold start {:.0f} s)"
            ctx = {"caller": source, "driver": record.driver, **(context or {})}
            tool = self.extra_tools(record).get(name) if name not in TOOLS else None
            try:
                if name == "jobs":
                    result = self._jobs(record, args, call_id, ctx)
                    note = ""
                elif tool is not None and tool.kind == "mcp":
                    result = self._remote(record, tool, args)
                    note = ""
                elif tool is not None:  # the project's own: run in the lab like a script
                    inner = {"call": tool.call, "args": args, "name": tool.name}
                    result = self.lab(sid).call(PROJECT_TOOL, inner, call_id, ctx)
                else:
                    result = self.lab(sid).call(name, args, call_id, ctx)
            except WspxError as exc:  # recorded like any other failed call, never dropped
                why = str(exc)
                if sid not in self.labs and sid in self._lab_why:
                    # seen live: the GPU cap stopped the lab mid-call and the call said only
                    # "Server disconnected"
                    why = f"{self._lab_why[sid]} (while this call ran)"
                result = {"summary": f"{name}: {why}", "ok": False, "handles": [], "data": {}}
                note = ""
            else:
                self._persist(sid, result)
                result.pop("journal", None)
                if result.get("attachments"):  # replayed into the store with the journal
                    self.allow(record, result["attachments"])
                    self._save(record)
            if note:
                note = note.format(time.time() - started)
        latency = time.time() - started
        # The open model's exchanges are the user's record only: they include held-out and
        # blind suite items, so they never go back to the caller.
        model_io = result.pop("model_io", None)
        if source == "driver" and record.max_lab_s is not None:
            left = max(0.0, record.max_lab_s - record.lab_used(time.time()))
            note = (note + " · " if note else "") + (
                f"[GPU time left: {_minutes(left)} of {_minutes(record.max_lab_s)}]"
            )
        result["summary"] = (result.get("summary") or "") + (f"\n{note}" if note else "")
        result["call_id"] = call_id
        result["summary"] += self._observe(sid, name, args, result, source, turn)
        self.trace.tool_call(
            sid, call_id, name, args, result, turn=turn, latency=latency, source=source
        )
        if model_io and (model_io.get("records") or model_io.get("omitted")):
            self.trace.event(
                sid,
                "model_io",
                {"call_id": call_id, "tool": name, "turn": turn, **model_io},
                source="open_model",
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
        scope = args.get("scope")
        if not scope:
            head = self.store.refs.read(f"sessions/{record.id}/HEAD")
            return {
                "summary": ir.render_info()
                + f"\nHEAD {short(head) if head else '(base)'} · lab cold",
                "ok": True,
                "handles": [],
                "data": {"summary": ir.summary(), "served_by": "store"},
            }
        if re.fullmatch(r"L[\d*,\-]+", scope):
            scope += ".*"
        try:
            targets = ir.resolve(scope)
        except WspxError as exc:
            return {"summary": f"arch: {exc}", "ok": False, "handles": [], "data": {}}
        limit = int(args.get("limit", 200))
        lines = [f"{len(targets)} tensor(s) match {scope}"] + [
            f"{ir.key_of(t.param)}  {tuple(ir.params[t.param].shape)}  {t.param}"
            for t in targets[:limit]
        ]
        return {
            "summary": "\n".join(lines) + "\n(lab cold: served from the store)",
            "ok": True,
            "handles": [],
            "data": {"served_by": "store"},
        }

    def _observe(
        self,
        sid: str,
        name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        source: str,
        turn: int | None = None,
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
            key = {"tool": name, "args": args, "state": self._epoch[sid]}
            if TOOLS[name].mutates and result.get("ok", True):
                self._epoch[sid] += 1
            verdicts = (
                []
                if source == "driver"
                else [self.gateway.observe(sid, "call", key, label=name, step=turn)]
            )
            outcome, label = outcome_of(result.get("summary") or "")
            if outcome:
                self._outcome_run.add(sid)
                # the turn makes one response's parallel calls one step, not a run
                verdicts.append(
                    self.gateway.observe(sid, "outcome", outcome, label=label, step=turn)
                )
            elif sid in self._outcome_run:
                # a clean call ends the run: one request when a run ends, not one per call
                self._outcome_run.discard(sid)
                verdicts.append(self.gateway.observe(sid, "outcome", None))
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
        self,
        sid: str,
        kind: str,
        text: str | None = None,
        *,
        source: str = "human",
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        record = self.get(sid)
        entry: dict[str, Any] = {"type": kind, **({"text": text} if text else {})}
        if attachments:
            self.allow(record, attachments)
            entry["attachments"] = attachments
        self.commands[sid].append(entry)
        self.trace.event(sid, "command", entry, source=source)

    # ------------------------------------------------------------------ background jobs

    def _jobs(
        self, record: SessionRecord, args: dict[str, Any], call_id: str, ctx: dict[str, Any]
    ) -> dict[str, Any]:
        """Start, list, collect or cancel a session's background jobs. Their state lives in
        the session's record, so it survives a control-plane restart; a job's result waits in
        the lab until collected, and here after."""
        jobs = record.extra.setdefault("jobs", {})
        action = args.get("action")

        def reply(summary: str, ok: bool = True, **extra: Any) -> dict[str, Any]:
            return {"summary": summary, "ok": ok, "handles": [], "data": {}, **extra}

        if action == "start":
            tool = args.get("tool")
            if tool not in JOB_TOOLS:
                return reply(f"jobs: start runs one of {', '.join(JOB_TOOLS)}", ok=False)
            view = self.lab(record.id).job(
                "POST",
                "/jobs",
                {"tool": tool, "args": args.get("args") or {}, "call_id": call_id, "context": ctx},
            )
            jobs[view["job"]] = {
                "tool": tool,
                "args": args.get("args") or {},
                "status": view["status"],
                "started": time.time(),
            }
            self._save(record)
            return reply(
                f"started job {view['job']}: {tool}; `jobs` status shows progress, collect returns the result",
                data={"job": view["job"]},
            )
        if action == "status":
            self._refresh_jobs(record)
            if not jobs:
                return reply("no background jobs")
            lines = []
            for jid, j in jobs.items():
                progress = j.get("progress") or {}
                done = (
                    f" ({progress.get('done')}/{progress.get('of')})" if progress.get("of") else ""
                )
                lines.append(f"{jid}  {j['tool']}  {j['status']}{done}")
            return reply("\n".join(lines), data={"jobs": jobs})
        jid = str(args.get("job") or "")
        if jid not in jobs:
            return reply(f"jobs: no job {jid!r}; `jobs` status lists them", ok=False)
        if action == "cancel":
            if jobs[jid]["status"] in ("queued", "running") and record.id in self.labs:
                self.labs[record.id].job("POST", f"/jobs/{jid}/cancel")
                jobs[jid]["status"] = "cancelling"
                self._save(record)
            return reply(f"{jid}: {jobs[jid]['status']}")
        if action == "collect":
            self._refresh_jobs(record, collect=jid)
            j = jobs[jid]
            if "result" in j:
                return dict(j["result"])
            return reply(f"{jid}: still {j['status']}; collect again later", ok=False)
        return reply("jobs: action is start, status, collect or cancel", ok=False)

    def _refresh_jobs(self, record: SessionRecord, *, collect: str | None = None) -> None:
        """Ask the lab how its jobs are doing; take a finished job's result (once) when asked."""
        jobs = record.extra.get("jobs", {})
        handle = self.labs.get(record.id)
        for jid, j in jobs.items():
            if "result" in j or j["status"] in ("lost",):
                continue
            if handle is None:
                j["status"] = "lost (the lab stopped before it finished)"
                continue
            try:
                view = handle.job("GET", f"/jobs/{jid}")
            except WspxError:
                j["status"] = "lost (the lab restarted)"
                continue
            j["status"], j["progress"] = view["status"], view.get("progress")
            if view.get("result") is not None and (collect == jid or view["status"] != "running"):
                result = dict(view["result"])
                self._persist(record.id, result)
                handle.job("POST", f"/jobs/{jid}/ack")
                result.pop("journal", None)
                result.pop("model_io", None)
                if result.get("attachments"):
                    self.allow(record, result["attachments"])
                j["result"] = {
                    k: result.get(k) for k in ("summary", "ok", "handles", "data", "attachments")
                }
        self._save(record)

    def _job_states(self, record: SessionRecord, states: dict[str, str]) -> None:
        """Job states from a heartbeat, so a job that finished unwatched lets its lab go idle."""
        jobs = record.extra.get("jobs", {})
        changed = False
        for jid, state in states.items():
            if jid in jobs and "result" not in jobs[jid] and jobs[jid].get("status") != state:
                jobs[jid]["status"] = state
                changed = True
        if changed:
            self._save(record)

    def _running_jobs(self, record: SessionRecord) -> bool:
        return any(
            j.get("status") in ("queued", "running", "cancelling")
            for j in record.extra.get("jobs", {}).values()
        )

    # ------------------------------------------------------------------ extra tools

    def extra_tools(self, record: SessionRecord) -> dict[str, Extra]:
        """The project's tools and the allowed tools of its remote MCP servers (listed once
        per session; a server that cannot be reached offers nothing, and the trace says so)."""
        found = {t.name: t for t in project_tools(record.extra.get("project_tools") or [])}
        servers = record.extra.get("mcp") or []
        if servers:
            if "mcp_tools" not in record.extra:
                listed: list[dict[str, Any]] = []
                for spec in servers:
                    try:
                        remote = Remote(
                            [spec], **({"connect": self.mcp_connect} if self.mcp_connect else {})
                        )
                        listed += [t.to_dict() for t in remote.discover()]
                    except Exception as exc:
                        self.trace.event(
                            record.id,
                            "mcp",
                            {"server": spec.get("name"), "error": f"{type(exc).__name__}: {exc}"},
                        )
                record.extra["mcp_tools"] = listed
                self._save(record)
            found.update({t["name"]: Extra(**t) for t in record.extra["mcp_tools"]})
        return found

    def _remote(self, record: SessionRecord, tool: Extra, args: dict[str, Any]) -> dict[str, Any]:
        """A remote MCP tool, called from here: its images become this session's attachments."""
        servers = record.extra.get("mcp") or []
        remote = Remote(servers, **({"connect": self.mcp_connect} if self.mcp_connect else {}))
        try:
            text, images, failed = remote.call(tool, args)
        except Exception as exc:
            raise WspxError(f"MCP server {tool.server}: {type(exc).__name__}: {exc}") from exc
        metas = [
            self.upload(data, name=f"{tool.remote}-{i + 1}.{kind.split('/')[-1]}", media_type=kind)
            for i, (data, kind) in enumerate(images)
        ]
        cut = text if len(text) <= REMOTE_CHARS else text[:REMOTE_CHARS] + "\n… (cut)"
        return {"summary": cut, "ok": not failed, "handles": [], "data": {}, "attachments": metas}

    # ------------------------------------------------------------------ attachments

    def upload(self, data: bytes, *, name: str, media_type: str) -> dict[str, Any]:
        """Store an image or file a session may be given; it is shown to a driver only once
        a session is given it (a task, a steering message)."""
        if len(data) > MAX_ATTACHMENT:
            raise WspxError(f"{name}: {len(data) / 2**20:.0f} MB; attachments are at most 20 MB")
        meta: dict[str, Any] = {
            "oid": self.store.objects.put(data),
            "media_type": media_type,
            "name": name,
            "size": len(data),
        }
        if media_type.startswith("image/"):
            with contextlib.suppress(Exception):
                import io

                from PIL import Image

                with Image.open(io.BytesIO(data)) as im:
                    meta["width"], meta["height"] = im.size
        return meta

    def allow(self, record: SessionRecord, metas: list[dict[str, Any]]) -> None:
        """Attachments this session's driver may fetch: what its tools returned, what it was
        given. Nothing else in the store (held-out suites live there too)."""
        allowed = record.extra.setdefault("attachments", {})
        for m in metas:
            if (
                isinstance(m, dict)
                and isinstance(m.get("oid"), str)
                and self.store.objects.has(m["oid"])
            ):
                allowed[m["oid"]] = {
                    k: m[k]
                    for k in ("media_type", "name", "size", "width", "height", "caption")
                    if k in m
                }

    def attachment(self, sid: str, oid: str) -> tuple[bytes, str]:
        record = self.get(sid)
        meta = record.extra.get("attachments", {}).get(oid)
        if meta is None:
            raise WspxError(f"{oid[:16]}: not an attachment of {sid}")
        return self.store.objects.get(oid), str(
            meta.get("media_type") or "application/octet-stream"
        )

    def task_attachments(self, record: SessionRecord) -> list[dict[str, Any]]:
        allowed = record.extra.get("attachments", {})
        return [
            {"oid": o, **allowed[o]}
            for o in record.extra.get("task_attachments", [])
            if o in allowed
        ]

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
        """Take the halt state from the gateway, which keeps it across control-plane restarts.

        A kill or resume that lands while the gateway answers (both often wake it
        together) is newer than the answer, and wins.
        """
        if self.gateway is None:
            return
        epoch = self._halt_epoch
        try:
            halted = self.gateway.halted()
        except WspxError:
            return
        with self._lock:
            if self._halt_epoch == epoch:
                self.halted = halted

    def resume(self, sid: str | None = None) -> None:
        targets = [sid] if sid else [s for s, r in self.records.items() if r.status == "paused"]
        if sid is None:
            with self._lock:
                self.halted = False
                self._halt_epoch += 1
        if self.gateway is not None:
            self.gateway.resume(sid)
        for s in targets:
            record = self.get(s)
            record.status, record.reason = "idle", ""
            self._save(record)
            self.trace.event(s, "session", {"status": "resumed"})
            if record.driver:
                self.start_driver(s)

    def _end_driver(self, sid: str, reason: str) -> None:
        """Stop the session's driver for good (its GPU time is used up)."""
        record = self.get(sid)
        record.status, record.reason = "stopped", reason
        self._save(record)
        self.command(sid, "stop", source="wspx")
        with self._lock:
            runner = self.runners.pop(sid, None)
        if runner is not None:
            runner.stop()
        self.trace.event(sid, "session", {"status": "stopped", "reason": reason})

    def kill(self, sid: str | None = None) -> int:
        """Stop drivers, revoke leases and terminate labs. ``None`` kills everything and halts."""
        targets = [sid] if sid else list(self.records)
        if sid is None:
            with self._lock:
                self.halted = True
                self._halt_epoch += 1
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
            if record.max_lab_s is not None and record.lab_used(now) >= record.max_lab_s:
                reason = f"used its GPU time ({_minutes(record.max_lab_s)})"
                self._end_driver(sid, reason)
                self.stop_lab(sid, reason, now=now)
                record.lab_seconds = max(record.lab_seconds, record.max_lab_s)
                self._save(record)
                continue
            if self.active(sid, now) or record.keep_warm == "always" or self._running_jobs(record):
                self.idle_since.pop(sid, None)
                record.tier = "hot"
            else:
                since = self.idle_since.setdefault(sid, now)
                if now - since >= record.grace_s:
                    self.stop_lab(sid, "idle past the grace period")
                    continue
                record.tier = "grace"
            self._save(record)
            try:
                beat = self.labs[sid].heartbeat(LEASE_S)
                self._persist(sid, beat)
                self._job_states(record, beat.get("jobs") or {})
                self._missed.pop(sid, None)
            except Exception as exc:
                self._missed[sid] += 1
                if self._missed[sid] < MISSED_HEARTBEATS:
                    continue  # a slow answer is not a lost lab; its lease still has time
                self._missed.pop(sid, None)
                # terminate it rather than forget it, or it keeps its GPU until the lease lapses
                self.stop_lab(sid, f"lost: no heartbeat answered {MISSED_HEARTBEATS} times ({exc})")


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds >= 60 else f"{seconds:.0f} s"


MAX_ATTACHMENT = 20 * 2**20
REMOTE_CHARS = 12_000
"""A remote MCP tool's text, as much as a wspx tool's summary may hold."""

GPU_CAPABILITY = {
    "T4": 7.5,
    "L4": 8.9,
    "A10G": 8.6,
    "A10": 8.6,
    "A100": 8.0,
    "A100-40GB": 8.0,
    "A100-80GB": 8.0,
    "L40S": 8.9,
    "H100": 9.0,
    "H100!": 9.0,
    "H200": 9.0,
    "B200": 10.0,
}
"""CUDA compute capability of Modal's GPU types."""
VLLM_MIN_CAPABILITY = 8.0


def _capability(gpu: str) -> float | None:
    kind = gpu.split(":", maxsplit=1)[0]
    return GPU_CAPABILITY.get(kind.upper()) or GPU_CAPABILITY.get(kind)


def check_engine_gpu(engine: str, gpu: str | None) -> None:
    """Refuse at creation what the lab could only fail at minutes later."""
    if engine not in ("auto", "hf", "vllm"):
        raise WspxError(f"unknown engine {engine!r}; use 'auto', 'hf' or 'vllm'")
    if engine != "vllm":
        return
    if not gpu:
        raise WspxError("engine = 'vllm' needs a GPU: set [lab] gpu, e.g. 'H100' or 'H100:2'")
    capability = _capability(gpu)
    if capability is not None and capability < VLLM_MIN_CAPABILITY:
        raise WspxError(
            f"engine = 'vllm' needs compute capability {VLLM_MIN_CAPABILITY} or newer, and a "
            f"{gpu.split(':')[0].upper()} has {capability}; use an L4, A100, L40S, H100 or newer"
        )


def wants_vllm(engine: str, gpu: str | None) -> bool:
    """Whether the lab starts from the vLLM lab image: ``vllm``, or ``auto`` on a GPU vLLM
    runs on (the lab then decides whether vLLM serves this model)."""
    if engine == "vllm":
        return True
    if engine != "auto" or not gpu:
        return False
    capability = _capability(gpu)
    return capability is None or capability >= VLLM_MIN_CAPABILITY


_PACKAGE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?([<>=!~]=?[A-Za-z0-9.*+!_-]+(,[<>=!~]=?[A-Za-z0-9.*+!_-]+)*)?$"
)


def _packages(specs: list[str]) -> list[str]:
    """Requirement specifiers only: a name, extras and versions (no URLs, paths or flags)."""
    for spec in specs:
        if not _PACKAGE.match(spec.replace(" ", "")):
            raise WspxError(
                f"[lab] packages: {spec!r} is not a package requirement (e.g. 'scikit-learn==1.8.0')"
            )
    return sorted({p.replace(" ", "") for p in specs})
