"""A session's record as files: two traces, each raw and readable, and the session itself.

The leader trace is the driver model's: what it was given, its turns with reasoning, the
tools it called and the results it got. The open model trace is the open-weight model's:
every prompt it was given, as it saw it, and what it wrote back. Works on the document the
control plane serves at ``/api/sessions/{sid}/export`` (``format: wspx-session/1``), so it
runs on any machine with the CLI, no lab needed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

FORMAT = "wspx-session/1"


FILES = ("leader.jsonl", "leader.md", "open-model.jsonl", "open-model.md", "session.json")


def write(export: dict[str, Any], out: Path) -> Path:
    """Write both traces (``.jsonl`` and ``.md``) and ``session.json`` into ``out/<session id>``."""
    sid = str(export["session"]["id"])
    folder = Path(out) / sid
    folder.mkdir(parents=True, exist_ok=True)
    leader = [e for e in export["events"] if e["kind"] != "model_io"]
    model = [e for e in export["events"] if e["kind"] == "model_io"]
    for name, events in (("leader", leader), ("open-model", model)):
        with (folder / f"{name}.jsonl").open("w") as f:
            for event in events:
                f.write(json.dumps(event, default=str) + "\n")
    (folder / "leader.md").write_text(render(export))
    (folder / "open-model.md").write_text(render_open_model(export))
    meta = {k: v for k, v in export.items() if k != "events"}
    (folder / "session.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return folder


def render(export: dict[str, Any]) -> str:
    """The leader's trace as Markdown, in the order it happened."""
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
    lines += ["## Leader trace", ""]
    for event in export["events"]:
        if event["kind"] != "model_io":
            lines += _event(event)
    lines += _commits(export.get("commits"))
    return "\n".join(lines).rstrip() + "\n"


def render_open_model(export: dict[str, Any]) -> str:
    """The open model's trace as Markdown: each tool call's prompts and generations."""
    s = export["session"]
    events = [e for e in export["events"] if e["kind"] == "model_io"]
    records = [r for e in events for r in e["payload"].get("records") or []]
    made = [r for r in records if r.get("kind") == "generate"]
    lines = [
        f"# Open model trace · session {s['id']}",
        "",
        f"- **Model:** {s['model']} · {s.get('dtype', '')}",
        f"- **Exchanges:** {len(made)} generation(s), {len(records) - len(made)} forward pass(es)"
        + (
            f", {sum(e['payload'].get('omitted', 0) for e in events)} more not kept"
            if events
            else ""
        ),
        f"- **Tokens:** prompts {sum(r.get('prompt_tokens', 0) for r in records):,} → output "
        f"{sum(r.get('output_tokens') or 0 for r in made):,}",
        "",
    ]
    if not events:
        return "\n".join([*lines, "The open model was not prompted in this session.", ""])
    for e in events:
        p = e["payload"]
        turn = f" · turn {p['turn']}" if p.get("turn") is not None else ""
        lines += [f"## `{p.get('tool')}`{turn} · {p.get('call_id')} · {_clock(e.get('ts'))}", ""]
        for i, r in enumerate(p.get("records") or [], 1):
            lines += _exchange(i, r)
    return "\n".join(lines).rstrip() + "\n"


def _exchange(i: int, r: dict[str, Any]) -> list[str]:
    what = "generation" if r.get("kind") == "generate" else "forward pass"
    head = f"### {i}. {what} · {r.get('purpose')} · weights {r.get('weights')} · "
    head += f"{r.get('prompt_tokens')} prompt tokens"
    if r.get("kind") == "generate":
        head += f" → {r.get('output_tokens')} · stop {r.get('stop')}"
    if r.get("split"):
        head += f" · {r['split']} (shown to the user only)"
    out = [head, "", "Prompt as the model saw it:", "", _fence(r.get("prompt", "")), ""]
    if r.get("reasoning"):
        out += ["**Reasoning**", "", _quote(r["reasoning"]), ""]
    for call in r.get("tool_calls") or []:
        args = call.get("arguments", call.get("raw"))
        out.append(f"- calls `{call.get('name') or 'tool'}` `{json.dumps(args)}`")
    if r.get("tool_calls"):
        out.append("")
    if r.get("kind") == "generate":
        out += ["**Answer**", "", r.get("answer") or "(no answer text)", ""]
        if r.get("output") != r.get("answer"):
            out += [
                "<details><summary>Raw output</summary>",
                "",
                _fence(r.get("output", "")),
                "",
                "</details>",
                "",
            ]
    return out


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
