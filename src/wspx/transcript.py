"""A session's record as files: the full trace, a readable transcript and the session itself.

Works on the document the control plane serves at ``/api/sessions/{sid}/export``
(``format: wspx-session/1``), so it runs on any machine with the CLI, no lab needed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

FORMAT = "wspx-session/1"


def write(export: dict[str, Any], out: Path) -> Path:
    """Write ``trace.jsonl``, ``transcript.md`` and ``session.json`` into ``out/<session id>``."""
    sid = str(export["session"]["id"])
    folder = Path(out) / sid
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "trace.jsonl").open("w") as f:
        for event in export["events"]:
            f.write(json.dumps(event, default=str) + "\n")
    (folder / "transcript.md").write_text(render(export))
    meta = {k: v for k, v in export.items() if k != "events"}
    (folder / "session.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return folder


def render(export: dict[str, Any]) -> str:
    """The whole session as Markdown, in the order it happened."""
    s = export["session"]
    lines = [
        f"# Session {s['id']}",
        "",
        f"- **Task:** {s.get('task') or '(interactive)'}",
        f"- **Model:** {s['model']} · {s.get('dtype', '')} · GPU {s.get('gpu') or 'CPU'}",
        f"- **Driver:** {s.get('driver') or 'none'}",
        f"- **Project:** {s.get('project') or '-'} · branch {s.get('branch') or '-'}",
        f"- **Status:** {s.get('status')}" + (f" ({s['reason']})" if s.get("reason") else ""),
        f"- **Created:** {_when(s.get('created'))} · exported {_when(export.get('exported'))} "
        f"with wspx {export.get('wspx', '?')}",
        "",
    ]
    lines += _spend(export.get("usage"))
    lines += ["## Trace", ""]
    for event in export["events"]:
        lines += _event(event)
    lines += _commits(export.get("commits"))
    return "\n".join(lines).rstrip() + "\n"


def _event(e: dict[str, Any]) -> list[str]:
    p, kind, at = e["payload"], e["kind"], _clock(e.get("ts"))
    if kind == "start":
        return [
            f"### Driver started · {p.get('provider')}:{p.get('model')} · effort {p.get('effort')}"
            f" · {at}",
            "",
            f"Tools: {', '.join(p.get('tools') or [])}",
            "",
            "<details><summary>System prompt</summary>",
            "",
            _fence(p.get("system", "")),
            "",
            "</details>",
            "",
            "First message to the model:",
            "",
            _fence(p.get("message", "")),
            "",
        ]
    if kind == "turn":
        out = [
            f"### Turn {p.get('turn')} · {p.get('model')} · {_tokens(p.get('usage'))} · "
            f"{p.get('latency', 0):.1f} s · {at}",
            "",
        ]
        if p.get("reasoning"):
            out += ["**Reasoning**", "", _quote(p["reasoning"]), ""]
        if p.get("text"):
            out += [p["text"], ""]
        for call in p.get("tool_calls") or []:
            out.append(f"- calls `{call['name']}` `{json.dumps(call.get('args'))}`")
        return [*out, ""] if p.get("tool_calls") else out
    if kind == "tool_call":
        status = "ok" if p.get("ok") else "failed"
        return [
            f"#### `{p.get('tool')}` · {status} · {e.get('source')} · {p.get('latency', 0):.2f} s"
            f" · {p.get('call_id')} · {at}",
            "",
            f"`{json.dumps(p.get('args'))}`",
            "",
            _fence(p.get("summary", "")),
            "",
        ]
    if kind == "notes":
        return ["**Delivered to the model:**", "", *(_quote(n) + "\n" for n in p.get("notes", []))]
    if kind == "compaction":
        return [
            f"**History compacted** ({p.get('dropped_messages')} messages, "
            f"{_tokens(p.get('usage'))}):",
            "",
            _quote(p.get("summary", "")),
            "",
        ]
    if kind == "command":
        return [f"- {at} **human:** {p.get('type')}" + (f" “{p['text']}”" if p.get("text") else "")]
    if kind == "session":
        return [
            f"- {at} session {p.get('status')}" + (f": {p['reason']}" if p.get("reason") else "")
        ]
    if kind == "lab":
        extra = f" in {p['load_seconds']:.0f} s" if p.get("load_seconds") else ""
        if p.get("network"):
            extra += f" · outbound network {p['network']}"
        if p.get("gpu_seconds") is not None:
            extra += f" · {p['gpu_seconds'] / 60:.1f} GPU min used"
        return [
            f"- {at} lab {p.get('tier')}{extra}" + (f": {p['reason']}" if p.get("reason") else "")
        ]
    return [f"- {at} {kind}: {json.dumps(p, default=str)[:500]}"]


def _spend(usage: Any) -> list[str]:
    if not isinstance(usage, dict) or not usage.get("groups"):
        return []
    lines = [
        "## Spend",
        "",
        "| provider:model | calls | input | cache read | cache write | output | cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in usage["groups"]:
        cost = "unknown" if g.get("c") is None else f"${g['c']:.4f}"
        lines.append(
            f"| {g.get('provider')}:{g.get('model')} | {g.get('n', 0)} | {g.get('i', 0):,} | "
            f"{g.get('cr', 0):,} | {g.get('cw', 0):,} | {g.get('o', 0):,} | {cost} |"
        )
    return [*lines, "", f"Total charged: ${usage.get('total_charged', 0):.4f}", ""]


def _commits(commits: Any) -> list[str]:
    rows = commits.get("commits") if isinstance(commits, dict) else None
    if not rows:
        return []
    lines = ["", "## Commits", ""]
    for c in rows:
        ops = ", ".join(p.get("op", "?") for p in c.get("patches") or []) or "merge or revert"
        flag = "" if c.get("status") in (None, "ok") else f" [{c['status']}]"
        lines.append(
            f"- `{c.get('short')}` {_when(c.get('time'))}{flag} {c.get('message')} ({ops})"
        )
    return lines


def _tokens(usage: Any) -> str:
    u = usage or {}
    cached = u.get("cache_read", 0) + u.get("cache_write", 0)
    text = f"in {u.get('input', 0) + cached:,}"
    if cached:
        text += f" ({u.get('cache_read', 0):,} cache read, {u.get('cache_write', 0):,} written)"
    text += f" → out {u.get('output', 0):,}"
    if u.get("reasoning"):
        text += f" ({u['reasoning']:,} reasoning)"
    return text + " tokens"


def _fence(text: str) -> str:
    ticks = "````" if "```" in text else "```"
    return f"{ticks}\n{text}\n{ticks}"


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in str(text).splitlines())


def _when(ts: Any) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "-"


def _clock(ts: Any) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts)) if ts else ""
