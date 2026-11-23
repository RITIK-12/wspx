import { useEffect, useMemo, useState, type ReactNode } from "react";
import { api, formatTokens, type Session } from "../api";

// Two traces of one session, each a tree of spans. Leader: the driver model's run — every
// LLM call with its reasoning, tokens and cost, every tool call with its full result, and
// the session and lab events around them. Open model: everything the open-weight model being
// edited was prompted with and wrote back, grouped by the tool call that prompted it. Click a
// span to read it. Export saves the same document `wspx sessions export` reads.
const VIEWS = ["Leader", "Open model"] as const;
type View = (typeof VIEWS)[number];

type Kind = "workflow" | "run" | "llm" | "tool" | "model" | "generate" | "forward" | "note" | "event";

interface Span {
  id: string;
  kind: Kind;
  title: string;
  subtitle?: string;
  start?: number;
  end?: number;
  seconds?: number;
  tokens?: { in: number; out: number };
  cost?: number | null;
  failed?: boolean;
  badge?: string;
  children: Span[];
  open?: boolean;
  detail: () => ReactNode;
}

export function TracesView({ session }: { session: Session }) {
  const [record, setRecord] = useState<any>(null);
  const [error, setError] = useState("");
  const [handles, setHandles] = useState<Record<string, any>>({});
  const [view, setView] = useState<View>("Leader");
  const [forward, setForward] = useState(false);
  const live = ["starting", "running"].includes(session.status);

  useEffect(() => {
    let gone = false;
    const load = () =>
      api(`/api/sessions/${session.id}/export`)
        .then((r) => !gone && (setRecord(r), setError("")))
        .catch((e) => !gone && setError(String(e)));
    load();
    api(`/api/sessions/${session.id}/handles`)
      .then((d) => !gone && setHandles(Object.fromEntries(d.handles.map((h: any) => [h.key, h]))))
      .catch(() => {});
    const timer = live ? setInterval(load, 5000) : undefined;
    return () => {
      gone = true;
      if (timer) clearInterval(timer);
    };
  }, [session.id, live]);

  const root = useMemo(
    () => (record ? (view === "Leader" ? leaderTree(record, handles) : openModelTree(record, forward)) : null),
    [record, handles, view, forward],
  );

  if (error && !record) return <div className="error">{error}</div>;
  if (!record || !root) return <p className="muted">Loading…</p>;

  const save = () => {
    const blob = new Blob([JSON.stringify(record, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${session.id}.wspx-session.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  return (
    <div className="full-trace">
      <div className="controls">
        <nav className="tabs subtabs">
          {VIEWS.map((v) => (
            <button key={v} className={v === view ? "tab active" : "tab"} onClick={() => setView(v)}>
              {v}
            </button>
          ))}
        </nav>
        {view === "Open model" && (
          <label className="toggle muted small">
            <input type="checkbox" checked={forward} onChange={(e) => setForward(e.target.checked)} />
            show forward passes
          </label>
        )}
        <span className="spacer" />
        <button onClick={save} title="Both traces, as `wspx sessions export` writes them">
          Export
        </button>
      </div>
      <SpanTree key={view} root={root} />
    </div>
  );
}

// --------------------------------------------------------------------------- the tree

function SpanTree({ root }: { root: Span }) {
  const [selected, setSelected] = useState<string>(root.id);
  const [closed, setClosed] = useState<Set<string>>(() => initiallyClosed(root));
  const byId = useMemo(() => index(root), [root]);
  const chosen = byId.get(selected) ?? root;
  const toggle = (id: string) =>
    setClosed((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const all = (open: boolean) => setClosed(open ? new Set() : new Set([...byId.values()].filter((s) => s.children.length && s.id !== root.id).map((s) => s.id)));

  return (
    <div className="trace-split">
      <div className="span-tree">
        <Row span={root} selected={selected} closed={closed} onSelect={setSelected} onToggle={toggle} root>
          <button className="icon-btn" title="Expand all" onClick={(e) => (e.stopPropagation(), all(true))}>
            <Chevrons open />
          </button>
          <button className="icon-btn" title="Collapse all" onClick={(e) => (e.stopPropagation(), all(false))}>
            <Chevrons />
          </button>
        </Row>
        {root.children.length > 0 && (
          <Children spans={root.children} selected={selected} closed={closed} onSelect={setSelected} onToggle={toggle} />
        )}
      </div>
      <aside className="span-detail">
        <div className="detail-head">
          <Icon kind={chosen.kind} failed={chosen.failed} />
          <div>
            <div className="span-title">{chosen.title}</div>
            <Meta span={chosen} />
          </div>
        </div>
        {chosen.detail()}
      </aside>
    </div>
  );
}

function Children(props: {
  spans: Span[];
  selected: string;
  closed: Set<string>;
  onSelect: (id: string) => void;
  onToggle: (id: string) => void;
}) {
  return (
    <div className="span-children">
      {props.spans.map((s) => (
        <div key={s.id}>
          <Row span={s} {...props} />
          {s.children.length > 0 && !props.closed.has(s.id) && <Children {...props} spans={s.children} />}
        </div>
      ))}
    </div>
  );
}

function Row({
  span,
  selected,
  closed,
  onSelect,
  onToggle,
  root,
  children,
}: {
  span: Span;
  selected: string;
  closed: Set<string>;
  onSelect: (id: string) => void;
  onToggle: (id: string) => void;
  root?: boolean;
  children?: ReactNode;
}) {
  const open = !closed.has(span.id);
  return (
    <div
      className={`span-row ${span.id === selected ? "selected" : ""} ${root ? "root" : ""}`}
      onClick={() => onSelect(span.id)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onSelect(span.id))}
    >
      <Icon kind={span.kind} failed={span.failed} />
      <div className="span-main">
        <div className="span-title">
          {span.title}
          {span.subtitle && <span className="span-sub"> {span.subtitle}</span>}
          {span.badge && <span className="badge warn">{span.badge}</span>}
        </div>
        <Meta span={span} />
      </div>
      {children}
      {!root && span.children.length > 0 && (
        <button
          className="icon-btn"
          title={open ? "Collapse" : "Expand"}
          aria-expanded={open}
          onClick={(e) => (e.stopPropagation(), onToggle(span.id))}
        >
          <Chevron open={open} />
        </button>
      )}
    </div>
  );
}

function Meta({ span }: { span: Span }) {
  const parts: string[] = [];
  const secs = duration(span);
  if (secs != null) parts.push(seconds(secs));
  if (span.tokens) {
    const { in: i, out: o } = span.tokens;
    parts.push(`${i.toLocaleString()} → ${o.toLocaleString()} (∑ ${(i + o).toLocaleString()})`);
  }
  if (span.cost != null) parts.push(money(span.cost));
  if (span.failed) parts.push("failed");
  return <div className="span-meta">{parts.join("   ")}</div>;
}

function initiallyClosed(root: Span): Set<string> {
  const out = new Set<string>();
  const walk = (s: Span) => {
    if (s !== root && s.children.length && !s.open) out.add(s.id);
    s.children.forEach(walk);
  };
  walk(root);
  return out;
}

function index(root: Span): Map<string, Span> {
  const out = new Map<string, Span>();
  const walk = (s: Span) => (out.set(s.id, s), s.children.forEach(walk));
  walk(root);
  return out;
}

function duration(s: Span): number | null {
  if (s.seconds != null) return s.seconds;
  if (s.start != null && s.end != null) return Math.max(0, s.end - s.start);
  return null;
}

function seconds(s: number): string {
  if (s < 60) return `${s.toFixed(2)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${String(Math.round(s - m * 60)).padStart(2, "0")}s`;
}

function money(c: number): string {
  return `$${c < 0.01 ? c.toFixed(6) : c.toFixed(4)}`;
}

// --------------------------------------------------------------------------- leader

function leaderTree(record: any, handles: Record<string, any>): Span {
  const events: any[] = record.events.filter((e: any) => e.kind !== "model_io");
  const io = new Map<string, any>(
    record.events.filter((e: any) => e.kind === "model_io").map((e: any) => [e.payload.call_id, e.payload]),
  );
  const prices = priceTable(record.usage);
  const s = record.session ?? {};
  const root: Span = {
    id: "root",
    kind: "workflow",
    title: `Session ${s.id}`,
    subtitle: s.task ? `· ${clipText(s.task, 80)}` : undefined,
    start: events[0]?.ts,
    end: events[events.length - 1]?.ts,
    cost: typeof record.usage?.total_charged === "number" ? record.usage.total_charged : undefined,
    children: [],
    open: true,
    detail: () => <SessionDetail record={record} />,
  };
  let run: Span | null = null;
  const add = (span: Span, toRun: boolean) => (toRun && run ? run.children : root.children).push(span);
  for (const e of events) {
    const p = e.payload;
    const id = `e${e.id}`;
    if (e.kind === "start") {
      run = {
        id,
        kind: "run",
        title: `Driver run: ${p.provider}:${p.model}`,
        subtitle: `· effort ${p.effort}`,
        start: e.ts,
        end: e.ts,
        children: [],
        open: true,
        detail: () => <RunDetail p={p} />,
      };
      root.children.push(run);
    } else if (e.kind === "turn") {
      const usage = p.usage ?? {};
      const span: Span = {
        id,
        kind: "llm",
        title: `LLM call · ${p.model || p.provider || "model"}`,
        subtitle: `· turn ${p.turn}`,
        start: e.ts - (p.latency ?? 0),
        end: e.ts,
        tokens: { in: (usage.input ?? 0) + (usage.cache_read ?? 0) + (usage.cache_write ?? 0), out: usage.output ?? 0 },
        cost: costOf(prices.get(`${p.provider}:${p.model}`), usage),
        children: [],
        detail: () => <TurnDetail p={p} />,
      };
      add(span, true);
    } else if (e.kind === "tool_call") {
      const exchanges = io.get(p.call_id)?.records ?? [];
      const made = exchanges.filter((r: any) => r.kind === "generate").length;
      add(
        {
          id,
          kind: "tool",
          title: `Tool: ${p.tool}`,
          subtitle: argsPreview(p.args) + (e.source && e.source !== "driver" ? ` · ${e.source}` : ""),
          start: e.ts - (p.latency ?? 0),
          end: e.ts,
          failed: !p.ok,
          badge: exchanges.length ? `open model: ${made} gen, ${exchanges.length - made} fwd` : undefined,
          children: [],
          detail: () => <ToolDetail p={p} handles={handles} />,
        },
        e.source === "driver",
      );
    } else if (e.kind === "notes" || e.kind === "compaction") {
      const usage = p.usage;
      add(
        {
          id,
          kind: e.kind === "compaction" ? "llm" : "note",
          title: e.kind === "notes" ? "Delivered to the model" : "History compacted",
          end: e.ts,
          tokens: usage ? { in: (usage.input ?? 0) + (usage.cache_read ?? 0) + (usage.cache_write ?? 0), out: usage.output ?? 0 } : undefined,
          children: [],
          detail: () => <pre className="result">{e.kind === "notes" ? (p.notes ?? []).join("\n\n") : p.summary}</pre>,
        },
        true,
      );
    } else {
      add(
        {
          id,
          kind: "event",
          title: eventLabel(e),
          seconds: e.kind === "lab" && p.load_seconds ? p.load_seconds : undefined,
          failed: e.kind === "error",
          children: [],
          detail: () => <pre className="result">{JSON.stringify(p, null, 2)}</pre>,
        },
        e.kind === "command" || e.kind === "error",
      );
    }
    if (run && e.kind !== "session" && e.kind !== "lab") run.end = e.ts;
  }
  const llm = [...index(root).values()].filter((x) => x.kind === "llm" && x.tokens);
  const sum = (key: "in" | "out") => llm.reduce((n, x) => n + (x.tokens?.[key] ?? 0), 0);
  if (llm.length) {
    root.tokens = { in: sum("in"), out: sum("out") };
    const r = run as Span | null;
    if (r) {
      const inRun = r.children.filter((x) => x.kind === "llm");
      r.tokens = {
        in: inRun.reduce((n, x) => n + (x.tokens?.in ?? 0), 0),
        out: inRun.reduce((n, x) => n + (x.tokens?.out ?? 0), 0),
      };
      const costs = inRun.map((x) => x.cost);
      r.cost = costs.every((c) => c != null) ? costs.reduce((n: number, c) => n + (c ?? 0), 0) : undefined;
    }
  }
  return root;
}

function eventLabel(e: any): string {
  const p = e.payload;
  if (e.kind === "lab") return `Lab ${p.tier}${p.reason ? `: ${p.reason}` : ""}`;
  if (e.kind === "session") return `Session ${p.status}${p.reason ? `: ${p.reason}` : ""}`;
  if (e.kind === "command") return `Human: ${p.type}${p.text ? ` "${clipText(p.text, 60)}"` : ""}`;
  if (e.kind === "loop") return `Loop ${p.action}: ${clipText(p.message ?? "", 80)}`;
  if (e.kind === "hook") return `${p.client} ${p.event ?? ""} ${p.tool ?? ""}`;
  if (e.kind === "error") return `Error: ${clipText(p.error ?? "", 80)}`;
  return e.kind;
}

function argsPreview(args: any): string {
  const pairs = Object.entries(args ?? {})
    .filter(([, v]) => typeof v === "string" || typeof v === "number" || typeof v === "boolean")
    .map(([k, v]) => `${k}=${typeof v === "string" ? JSON.stringify(clipText(v, 40)) : v}`);
  return pairs.length ? ` · ${clipText(pairs.join(" "), 70)}` : "";
}

function clipText(text: string, n: number): string {
  const one = String(text).replace(/\s+/g, " ").trim();
  return one.length > n ? `${one.slice(0, n - 1)}…` : one;
}

function priceTable(usage: any): Map<string, any> {
  return new Map((usage?.groups ?? []).filter((g: any) => g.price).map((g: any) => [`${g.provider}:${g.model}`, g.price]));
}

function costOf(price: any, u: any): number | null {
  if (!price) return null;
  return (
    ((u.input ?? 0) * price.input +
      (u.output ?? 0) * price.output +
      (u.cache_read ?? 0) * price.cache_read +
      (u.cache_write ?? 0) * price.cache_write) /
    1e6
  );
}

function SessionDetail({ record }: { record: any }) {
  const s = record.session ?? {};
  const rows: [string, ReactNode][] = [
    ["model", `${s.model} · ${s.dtype ?? ""}`],
    ["driver", s.driver ?? "none"],
    ["task", s.task || "(interactive)"],
    ["status", `${s.status}${s.reason ? ` (${s.reason})` : ""}`],
    ["gpu", s.gpu ?? "CPU"],
  ];
  if (typeof record.usage?.total_charged === "number") rows.push(["charged", money(record.usage.total_charged)]);
  return (
    <table className="kv">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <th>{k}</th>
            <td>{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function RunDetail({ p }: { p: any }) {
  return (
    <>
      <div className="muted small">
        tools: {(p.tools ?? []).join(", ")} · max {p.max_tokens?.toLocaleString()} tokens
      </div>
      <details>
        <summary>system prompt</summary>
        <pre className="reasoning">{p.system}</pre>
      </details>
      <details open>
        <summary>first message</summary>
        <pre className="reasoning">{p.message}</pre>
      </details>
    </>
  );
}

function TurnDetail({ p }: { p: any }) {
  return (
    <>
      <div className="muted small">
        {formatTokens(p.usage)} tokens · stop {p.stop}
      </div>
      {p.reasoning && (
        <details open>
          <summary>reasoning</summary>
          <pre className="reasoning">{p.reasoning}</pre>
        </details>
      )}
      {p.text && <div className="text">{p.text}</div>}
      {(p.tool_calls ?? []).map((c: any) => (
        <div key={c.id} className="call">
          → <code>{c.name}</code> <code className="args">{JSON.stringify(c.args)}</code>
        </div>
      ))}
    </>
  );
}

function ToolDetail({ p, handles }: { p: any; handles: Record<string, any> }) {
  const trace = (p.handles ?? []).map((h: string) => handles[h]).find((h: any) => h?.kind === "trace");
  return (
    <>
      <div className="muted small">
        <code>{p.call_id}</code>
        {p.turn != null && <> · turn {p.turn}</>}
      </div>
      <pre className="reasoning">{JSON.stringify(p.args, null, 2)}</pre>
      <pre className="result">{p.summary}</pre>
      {trace && <TraceObject handle={trace} />}
    </>
  );
}

// --------------------------------------------------------------------------- open model

function openModelTree(record: any, forward: boolean): Span {
  const events: any[] = record.events.filter((e: any) => e.kind === "model_io");
  const calls = new Map<string, any>(
    record.events.filter((e: any) => e.kind === "tool_call").map((e: any) => [e.payload.call_id, e]),
  );
  const s = record.session ?? {};
  const all = events.flatMap((e) => e.payload.records ?? []);
  const made = all.filter((r: any) => r.kind === "generate").length;
  const sum = (rs: any[], key: string) => rs.reduce((n: number, r: any) => n + (r[key] ?? 0), 0);
  const first = events[0] ? calls.get(events[0].payload.call_id) : undefined;
  const root: Span = {
    id: "root",
    kind: "workflow",
    title: `Open model: ${s.model}`,
    subtitle: `· ${made} generation(s), ${all.length - made} forward pass(es)`,
    start: first ? first.ts - (first.payload.latency ?? 0) : events[0]?.ts,
    end: events[events.length - 1]?.ts,
    tokens: all.length ? { in: sum(all, "prompt_tokens"), out: sum(all, "output_tokens") } : undefined,
    children: [],
    open: true,
    detail: () =>
      events.length ? (
        <p className="muted">Every prompt the open model was given and what it wrote, by the tool call that prompted it.</p>
      ) : (
        <p className="muted">The open model has not been prompted yet in this session.</p>
      ),
  };
  for (const e of events) {
    const p = e.payload;
    const records: any[] = p.records ?? [];
    const shown = records.filter((r) => forward || r.kind === "generate");
    if (!shown.length && !p.omitted) continue;
    const call = calls.get(p.call_id);
    const gen = records.filter((r) => r.kind === "generate").length;
    root.children.push({
      id: `e${e.id}`,
      kind: "model",
      title: `${p.tool}${p.turn != null ? ` · turn ${p.turn}` : ""}`,
      subtitle: `· ${gen} generation(s), ${records.length - gen} forward${p.omitted ? `, ${p.omitted} not kept` : ""}`,
      seconds: call?.payload.latency,
      tokens: { in: sum(records, "prompt_tokens"), out: sum(records, "output_tokens") },
      children: shown.map((r, i) => ({
        id: `e${e.id}.${i}`,
        kind: r.kind === "generate" ? "generate" : "forward",
        title: `${r.kind === "generate" ? "Generation" : "Forward pass"} · ${r.purpose}`,
        subtitle: r.kind === "generate" && r.answer ? `· ${clipText(r.answer, 48)}` : undefined,
        seconds: r.seconds,
        tokens: { in: r.prompt_tokens ?? 0, out: r.output_tokens ?? 0 },
        badge: r.split ? `${r.split}: visible here only` : undefined,
        children: [],
        detail: () => <Exchange record={r} />,
      })),
      open: shown.length <= 6,
      detail: () => (
        <div className="muted small">
          <code>{p.call_id}</code> · {records.length} exchange(s){p.omitted ? `, ${p.omitted} more not kept` : ""}
          {call && <pre className="result">{call.payload.summary}</pre>}
        </div>
      ),
    });
  }
  return root;
}

function Exchange({ record: r }: { record: any }) {
  const system = (r.messages ?? []).filter((m: any) => m.role === "system");
  return (
    <div className={`exchange ${r.kind}`}>
      <div className="meta">
        {r.kind === "generate" ? "generation" : "forward pass"} · {r.purpose} · weights {r.weights} ·{" "}
        {r.prompt_tokens} prompt tokens
        {r.kind === "generate" && (
          <>
            {" "}
            → {r.output_tokens} · stop {r.stop}
            {r.batch > 1 && <> · batch of {r.batch}</>}
          </>
        )}
        {r.split && <span className="badge warn">{r.split}: visible here only</span>}
      </div>
      {system.map((m: any, i: number) => (
        <div key={i} className="instruction">
          <span className="muted small">instruction</span> {contentText(m.content)}
        </div>
      ))}
      <details open>
        <summary>prompt as the model saw it</summary>
        <pre className="reasoning">{r.prompt}</pre>
      </details>
      {r.reasoning && (
        <details open>
          <summary>reasoning</summary>
          <pre className="reasoning">{r.reasoning}</pre>
        </details>
      )}
      {(r.tool_calls ?? []).map((c: any, i: number) => (
        <div key={i} className="call">
          → <code>{c.name ?? "tool call"}</code>{" "}
          <code className="args">{JSON.stringify(c.arguments ?? c.raw)}</code>
        </div>
      ))}
      {r.kind === "generate" && <div className="text">{r.answer || <span className="muted">(no answer text)</span>}</div>}
      {r.kind === "generate" && r.output !== r.answer && (
        <details>
          <summary>raw output</summary>
          <pre className="reasoning">{r.output}</pre>
        </details>
      )}
    </div>
  );
}

function contentText(content: any): string {
  if (typeof content === "string") return content;
  return (content ?? [])
    .map((part: any) => (part.type === "text" ? part.text : `[${part.type}${part[part.type] ? `: ${part[part.type]}` : ""}]`))
    .join(" ");
}

// --------------------------------------------------------------------------- icons

const GLYPHS: Record<Kind, ReactNode> = {
  workflow: (
    <>
      <path d="M3 4h2M3 8h2M3 12h2" />
      <path d="M8 4h5M8 8h5M8 12h5" />
    </>
  ),
  run: <path d="M2.5 8h11M5 5.5 2.5 8 5 10.5M11 5.5 13.5 8 11 10.5" />,
  llm: <path d="M8 2.5v11M2.5 8h11M4.2 4.2l7.6 7.6M11.8 4.2l-7.6 7.6" />,
  tool: <path d="M6 3.5c-1.5 0-1.5 1-1.5 2v1c0 1-.7 1.5-1.5 1.5.8 0 1.5.5 1.5 1.5v1c0 1 0 2 1.5 2M10 3.5c1.5 0 1.5 1 1.5 2v1c0 1 .7 1.5 1.5 1.5-.8 0-1.5.5-1.5 1.5v1c0 1 0 2-1.5 2" />,
  model: (
    <>
      <path d="M8 2.5 13 5.2v5.6L8 13.5 3 10.8V5.2z" />
      <path d="M3 5.2 8 8l5-2.8M8 8v5.5" />
    </>
  ),
  generate: <path d="M8 2.5 9.4 6.6 13.5 8 9.4 9.4 8 13.5 6.6 9.4 2.5 8 6.6 6.6z" />,
  forward: <path d="M3 8h9M9 5l3 3-3 3" />,
  note: <path d="M4 3.5h8v9H4zM6 6.5h4M6 9h4" />,
  event: <circle cx="8" cy="8" r="2" />,
};

function Icon({ kind, failed }: { kind: Kind; failed?: boolean }) {
  return (
    <span className={`span-icon k-${kind} ${failed ? "failed" : ""}`} aria-hidden>
      <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
        {GLYPHS[kind]}
      </svg>
    </span>
  );
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={open ? "M4 6l4 4 4-4" : "M6 4l4 4-4 4"} />
    </svg>
  );
}

function Chevrons({ open }: { open?: boolean }) {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={open ? "M5 6l3-3 3 3M5 10l3 3 3-3" : "M5 3l3 3 3-3M5 13l3-3 3 3"} />
    </svg>
  );
}

function TraceObject({ handle }: { handle: any }) {
  const [trace, setTrace] = useState<any>(null);
  if (!trace) {
    return (
      <button className="link" onClick={() => api(`/api/objects/${handle.ref}`).then(setTrace)}>
        show {handle.key}
      </button>
    );
  }
  return (
    <div className="trace">
      {trace.kind === "forward" && <LogitLens arrays={trace.arrays} />}
      {trace.kind === "diff" && <Divergence values={trace.arrays.divergence} calls={trace.arrays.calls} />}
      {trace.kind === "attribution" && (
        <ol className="ranking">
          {trace.data.ranking.map(([addr, score]: [string, number]) => (
            <li key={addr}>
              <code>{addr}</code>{" "}
              <span className="bar" style={{ width: `${Math.min(100, Math.abs(score) * 400)}%` }} /> {score.toFixed(4)}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function LogitLens({ arrays }: { arrays: any }) {
  const max = Math.max(...arrays.resid_norm, 1e-9);
  return (
    <table className="lens">
      <thead>
        <tr>
          <th>call</th>
          <th>|h|</th>
          <th>top tokens at the last position</th>
        </tr>
      </thead>
      <tbody>
        {arrays.lens_ids.map((ids: number[], i: number) => (
          <tr key={i}>
            <td>{arrays.calls?.[i] ?? `L${i}`}</td>
            <td>
              <span className="bar" style={{ width: `${(100 * arrays.resid_norm[i]) / max}%` }} />
            </td>
            <td>
              {ids.map((t, j) => (
                <span key={j} className="token" style={{ opacity: 0.35 + 0.65 * Math.min(1, arrays.lens_p[i][j] * 3) }}>
                  #{t} <span className="muted small">{arrays.lens_p[i][j].toFixed(2)}</span>
                </span>
              ))}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Divergence({ values, calls }: { values: number[]; calls?: string[] }) {
  const max = Math.max(...values, 1e-9);
  return (
    <div className="divergence">
      {values.map((v, i) => (
        <div key={i} className="div-row">
          <span>{calls?.[i] ?? `L${i}`}</span>
          <span className="bar" style={{ width: `${(100 * v) / max}%` }} />
          <span className="muted small">{v.toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
}
