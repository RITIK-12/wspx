import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { formatTokens, post, stream, type Session, type TimelineEvent } from "../api";
import { Markdown } from "../markdown";
import { MediaStrip, mediaParts } from "../media";

// The session as a conversation: your messages on the right (the task, and anything you send
// while it runs); on the left, the leader's replies with their thinking folded away and the
// tools they called, and the open model's answers in boxes of their own. Session and lab
// changes are small notes in between. The full record, as spans, is in Traces.
export function SessionView({ session, onChange }: { session: Session; onChange: () => void }) {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const log = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    setEvents([]);
    stick.current = true;
    const close = stream(session.id, 0, (e) =>
      setEvents((prev) => (prev.some((p) => p.id === e.id) ? prev : [...prev, e])),
    );
    return close;
  }, [session.id]);

  // Follow new messages only while the reader is at the bottom.
  useEffect(() => {
    const panel = log.current?.closest(".panel");
    if (!panel) return;
    const onScroll = () => {
      stick.current = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 80;
    };
    panel.addEventListener("scroll", onScroll);
    return () => panel.removeEventListener("scroll", onScroll);
  }, []);
  useEffect(() => {
    const panel = log.current?.closest(".panel");
    if (panel && stick.current) panel.scrollTop = panel.scrollHeight;
  }, [events.length]);

  const items = useMemo(() => conversation(events, session), [events, session]);
  const act = async (action: string, body: object = {}) => {
    await post(`/api/sessions/${session.id}/${action}`, body);
    onChange();
  };
  const send = async () => {
    const text = draft.trim();
    if (!text || sending) return;
    setSending(true);
    try {
      await act("inject", { text });
      setDraft("");
      stick.current = true;
    } finally {
      setSending(false);
    }
  };

  const usage = events.filter((e) => e.kind === "turn").map((e) => e.payload.usage ?? {});
  const tokens = usage.reduce((n, u) => n + (u.input ?? 0) + (u.cache_read ?? 0) + (u.cache_write ?? 0) + (u.output ?? 0), 0);
  const running = session.status === "running";
  // A message waits in the control plane's memory until the leader next reads its commands,
  // so it is only offered while a leader is running, starting or paused (and resumes soon).
  const listening = !!session.driver && ["running", "starting", "paused"].includes(session.status);

  return (
    <div className="session-view chat">
      <div className="chat-head">
        <div className="chat-title">
          <strong>{session.task || "Interactive session"}</strong>
          <div className="muted small">
            <span className={`status-dot s-${session.status}`} /> {session.status} · {session.model} · leader{" "}
            {session.driver ?? "none"} · {tokens.toLocaleString()} tokens
            {session.reason && <> · {session.reason}</>}
          </div>
        </div>
        <div className="chat-actions">
          {running && <button onClick={() => act("pause")}>Pause</button>}
          {session.status === "paused" && <button onClick={() => act("resume")}>Resume</button>}
          {session.driver && ["idle", "done", "stopped", "error"].includes(session.status) && (
            <button onClick={() => act("start")}>Run leader</button>
          )}
          <button className="danger" onClick={() => act("stop")}>
            Stop
          </button>
        </div>
      </div>

      <div className="chat-log" ref={log}>
        {items.map((item) => (
          <ChatItem key={item.key} item={item} session={session.id} />
        ))}
        {running && <div className="chat-working muted small">working…</div>}
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <textarea
          rows={1}
          placeholder={
            !session.driver
              ? "This session has no leader"
              : listening
                ? "Message the leader… (Enter to send, Shift+Enter for a new line)"
                : "The leader is not running; Run leader to continue"
          }
          value={draft}
          disabled={!listening}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              void send();
            }
          }}
        />
        <button type="submit" className="primary" disabled={!draft.trim() || sending || !listening}>
          Send
        </button>
      </form>
    </div>
  );
}

// --------------------------------------------------------------------------- model

interface ToolUse {
  id: string;
  name: string;
  args: any;
  result?: any;
}

type Item =
  | { key: string; type: "user"; text: string; ts?: number; note?: string }
  | { key: string; type: "system"; text: string; ts?: number; tone?: "warn" | "danger"; body?: string }
  | { key: string; type: "leader"; turn: any; tools: ToolUse[]; ts: number }
  | { key: string; type: "tools"; tools: ToolUse[]; source: string; ts: number }
  | { key: string; type: "open"; io: any; ts: number; model: string };

function conversation(events: TimelineEvent[], session: Session): Item[] {
  const items: Item[] = [];
  const pending = new Map<string, ToolUse>();
  let asked = false;
  const ask = (ts?: number) => {
    if (asked || !session.task) return;
    asked = true;
    items.push({ key: "task", type: "user", text: session.task, ts });
  };
  for (const e of events) {
    const p = e.payload;
    const key = `e${e.id}`;
    switch (e.kind) {
      case "session":
        if (p.status === "created") {
          items.push({ key, type: "system", text: "Session created", ts: e.ts });
          ask(e.ts);
        } else if (p.status !== "running") {
          items.push({ key, type: "system", text: `Session ${p.status}${p.reason ? `: ${p.reason}` : ""}`, ts: e.ts, tone: p.status === "error" ? "danger" : undefined });
        }
        break;
      case "lab":
        items.push({ key, type: "system", text: `Lab ${p.tier}${p.load_seconds ? ` in ${p.load_seconds.toFixed(0)} s` : ""}${p.reason ? `: ${p.reason}` : ""}`, ts: e.ts });
        break;
      case "start":
        ask(e.ts);
        items.push({ key, type: "system", text: `Leader ${p.provider}:${p.model} started · effort ${p.effort}`, ts: e.ts, body: p.system });
        break;
      case "turn": {
        const tools = (p.tool_calls ?? []).map((c: any) => {
          const use: ToolUse = { id: c.id, name: c.name, args: c.args };
          pending.set(c.id, use);
          return use;
        });
        if (p.text || p.reasoning || tools.length) items.push({ key, type: "leader", turn: p, tools, ts: e.ts });
        break;
      }
      case "tool_call": {
        const use = pending.get(p.call_id);
        if (use) {
          use.result = p;
          pending.delete(p.call_id);
        } else {
          const last = items[items.length - 1];
          const loose: ToolUse = { id: p.call_id, name: p.tool, args: p.args, result: p };
          if (last?.type === "tools" && last.source === e.source) last.tools.push(loose);
          else items.push({ key, type: "tools", tools: [loose], source: e.source ?? "", ts: e.ts });
        }
        break;
      }
      case "model_io":
        if ((p.records ?? []).length) items.push({ key, type: "open", io: p, ts: e.ts, model: session.model });
        break;
      case "command":
        if (p.text) items.push({ key, type: "user", text: p.text, ts: e.ts, note: p.type !== "inject" ? p.type : undefined });
        else items.push({ key, type: "system", text: `You: ${p.type}`, ts: e.ts });
        break;
      case "notes":
        items.push({ key, type: "system", text: "Workspace notes delivered to the leader", ts: e.ts, body: (p.notes ?? []).join("\n\n") });
        break;
      case "compaction":
        items.push({ key, type: "system", text: `Leader history compacted (${formatTokens(p.usage)} tokens)`, ts: e.ts, body: p.summary });
        break;
      case "loop":
        items.push({ key, type: "system", text: `Loop ${p.action}: ${p.message}`, ts: e.ts, tone: "warn" });
        break;
      case "error":
        items.push({ key, type: "system", text: `Error: ${p.error ?? JSON.stringify(p)}`, ts: e.ts, tone: "danger" });
        break;
      case "hook":
        break; // an external client's own hook echo: its tool calls are shown already
      case "end":
        break;
      default:
        items.push({ key, type: "system", text: e.kind, ts: e.ts });
    }
  }
  ask();
  return items;
}

// --------------------------------------------------------------------------- view

function ChatItem({ item, session }: { item: Item; session: string }) {
  if (item.type === "user") {
    return (
      <div className="msg-row user">
        <div className="bubble user">
          {item.note && <div className="who">{item.note}</div>}
          <div className="plain">{item.text}</div>
        </div>
      </div>
    );
  }
  if (item.type === "system") {
    return (
      <div className={`msg-system ${item.tone ?? ""}`}>
        {item.body ? (
          <details>
            <summary>
              {item.text} <Clock ts={item.ts} />
            </summary>
            <pre className="result">{item.body}</pre>
          </details>
        ) : (
          <span>
            {item.text} <Clock ts={item.ts} />
          </span>
        )}
      </div>
    );
  }
  if (item.type === "leader") {
    const p = item.turn;
    return (
      <div className="msg-row">
        <Avatar kind="leader" />
        <div className="bubble leader">
          <div className="who">
            Leader · {p.model || p.provider} <span className="muted">· turn {p.turn} · {formatTokens(p.usage)} tokens · {(p.latency ?? 0).toFixed(1)} s</span>
          </div>
          {p.reasoning && <Thinking text={p.reasoning} tokens={p.usage?.reasoning} />}
          {p.text && <Markdown text={p.text} />}
          {item.tools.length > 0 && <Tools tools={item.tools} />}
        </div>
      </div>
    );
  }
  if (item.type === "tools") {
    return (
      <div className="msg-row">
        <Avatar kind="client" />
        <div className="bubble client">
          <div className="who">{item.source || "client"}</div>
          <Tools tools={item.tools} />
        </div>
      </div>
    );
  }
  return <OpenModel io={item.io} model={item.model} session={session} />;
}

function Tools({ tools }: { tools: ToolUse[] }) {
  return (
    <div className="tool-uses">
      {tools.map((t) => {
        const r = t.result;
        const state = !r ? "running" : r.ok ? "ok" : "failed";
        return (
          <details key={t.id} className={`tool-use ${state}`}>
            <summary>
              <span className="tool-mark">{state === "running" ? "…" : state === "ok" ? "✓" : "✕"}</span>
              <code>{t.name}</code>
              <span className="tool-args">{preview(t.args)}</span>
              {r && <span className="muted small">{(r.latency ?? 0).toFixed(2)} s</span>}
            </summary>
            <pre className="tool-io">{JSON.stringify(t.args, null, 2)}</pre>
            {r ? <pre className="result">{r.summary}</pre> : <div className="muted small">running…</div>}
          </details>
        );
      })}
    </div>
  );
}

function OpenModel({ io, model, session }: { io: any; model: string; session: string }) {
  const records: any[] = io.records ?? [];
  const made = records.filter((r) => r.kind === "generate");
  const read = records.length - made.length;
  const purposes = [...new Set(records.map((r) => r.purpose))].join(", ");
  if (!made.length) {
    return (
      <div className="msg-system">
        <span>
          Open model read {read} prompt{read === 1 ? "" : "s"} for {purposes}
          {io.omitted ? ` (+${io.omitted} not kept)` : ""}
        </span>
      </div>
    );
  }
  const first = made.slice(0, 2);
  const rest = made.slice(2);
  return (
    <div className="msg-row">
      <Avatar kind="open" />
      <div className="bubble open">
        <div className="who">
          Open model · {model}{" "}
          <span className="muted">
            · for {io.tool}
            {io.turn != null ? ` (turn ${io.turn})` : ""} · {made.length} answer{made.length === 1 ? "" : "s"}
            {read ? ` · ${read} more prompt${read === 1 ? "" : "s"} read` : ""}
          </span>
        </div>
        {first.map((r, i) => (
          <Answer key={i} r={r} session={session} />
        ))}
        {rest.length > 0 && (
          <details className="more">
            <summary>
              {rest.length} more answer{rest.length === 1 ? "" : "s"}
            </summary>
            {rest.map((r, i) => (
              <Answer key={i} r={r} session={session} />
            ))}
          </details>
        )}
      </div>
    </div>
  );
}

function Answer({ r, session }: { r: any; session: string }) {
  const system = (r.messages ?? []).find((m: any) => m.role === "system");
  const user = [...(r.messages ?? [])].reverse().find((m: any) => m.role === "user");
  const asked = user ? contentText(user.content, false) : "";
  return (
    <div className="answer">
      <div className="answer-meta muted small">
        {r.purpose} · weights {r.weights} · {r.prompt_tokens} → {r.output_tokens} tokens
        {r.seconds != null && <> · {r.seconds.toFixed(2)} s</>}
        {r.split && <span className="badge warn">{r.split}: visible to you only</span>}
      </div>
      {(system || asked || r.item) && (
        <div className="asked small">
          {r.item && <div><span className="muted">stored prompt</span> {r.item}</div>}
          {system && <div><span className="muted">instruction</span> {contentText(system.content)}</div>}
          {asked && <div><span className="muted">asked</span> {asked}</div>}
        </div>
      )}
      <MediaStrip session={session} parts={mediaParts(r.messages)} />
      {r.reasoning && <Thinking text={r.reasoning} />}
      {(r.tool_calls ?? []).map((c: any, i: number) => (
        <div key={i} className="tool-use ok inline">
          <span className="tool-mark">→</span> <code>{c.name ?? "tool call"}</code>{" "}
          <span className="tool-args">{JSON.stringify(c.arguments ?? c.raw)}</span>
        </div>
      ))}
      <div className="plain">{r.answer || <span className="muted">(no answer text)</span>}</div>
      <details className="prompt">
        <summary>prompt as the model saw it</summary>
        <pre className="reasoning">{r.prompt}</pre>
      </details>
    </div>
  );
}

function Thinking({ text, tokens }: { text: string; tokens?: number }) {
  return (
    <details className="thinking">
      <summary>
        Thinking{tokens ? ` · ${tokens.toLocaleString()} tokens` : ""}
      </summary>
      <div className="thinking-body">{text}</div>
    </details>
  );
}

function Avatar({ kind }: { kind: "leader" | "open" | "client" }) {
  const glyph: Record<string, ReactNode> = {
    leader: <path d="M8 2.5v11M2.5 8h11M4.2 4.2l7.6 7.6M11.8 4.2l-7.6 7.6" />,
    open: (
      <>
        <path d="M8 2.5 13 5.2v5.6L8 13.5 3 10.8V5.2z" />
        <path d="M3 5.2 8 8l5-2.8M8 8v5.5" />
      </>
    ),
    client: <path d="M3 4.5h10v7H3zM6 13.5h4" />,
  };
  return (
    <span className={`avatar ${kind}`} aria-hidden>
      <svg viewBox="0 0 16 16" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        {glyph[kind]}
      </svg>
    </span>
  );
}

function Clock({ ts }: { ts?: number }) {
  return ts ? <span className="clock">{new Date(ts * 1000).toLocaleTimeString()}</span> : null;
}

function preview(args: any): string {
  const parts = Object.entries(args ?? {})
    .filter(([, v]) => typeof v === "string" || typeof v === "number" || typeof v === "boolean")
    .map(([k, v]) => `${k}=${typeof v === "string" ? JSON.stringify(v.length > 40 ? `${v.slice(0, 39)}…` : v) : v}`);
  const text = parts.join(" ");
  return text.length > 90 ? `${text.slice(0, 89)}…` : text;
}

function contentText(content: any, media = true): string {
  if (typeof content === "string") return content;
  return (content ?? [])
    .filter((part: any) => media || part.type === "text")
    .map((part: any) => (part.type === "text" ? part.text : `[${part.type}]`))
    .join(" ");
}
