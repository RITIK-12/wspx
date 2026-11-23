import { useEffect, useState } from "react";
import { api, formatTokens, type Session } from "../api";

// Two traces of one session. Leader: the driver model's record from creation — what it was
// given, every turn with its reasoning and token counts, every tool call with its full
// result, every human action and lab change. Open model: everything the open-weight model
// being edited was prompted with (chat template, system instructions and all) and what it
// wrote back — reasoning, tool calls and answer — per tool call, with the weights it ran on.
// Export saves the same document `wspx sessions export` reads.
const VIEWS = ["Leader", "Open model"] as const;
type View = (typeof VIEWS)[number];

export function TracesView({ session }: { session: Session }) {
  const [record, setRecord] = useState<any>(null);
  const [error, setError] = useState("");
  const [handles, setHandles] = useState<Record<string, any>>({});
  const [view, setView] = useState<View>("Leader");
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

  if (error && !record) return <div className="error">{error}</div>;
  if (!record) return <p className="muted">Loading…</p>;

  const events: any[] = record.events;
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
        <span className="spacer" />
        <button onClick={save} title="Both traces, as `wspx sessions export` writes them">
          Export
        </button>
      </div>
      {view === "Leader" ? (
        <LeaderTrace events={events.filter((e) => e.kind !== "model_io")} usage={record.usage} handles={handles} />
      ) : (
        <OpenModelTrace events={events.filter((e) => e.kind === "model_io")} model={record.session?.model} />
      )}
    </div>
  );
}

function LeaderTrace({ events, usage, handles }: { events: any[]; usage: any; handles: Record<string, any> }) {
  const turns = events.filter((e) => e.kind === "turn");
  const calls = events.filter((e) => e.kind === "tool_call");
  const sum = (key: string) =>
    events
      .filter((e) => e.kind === "turn" || e.kind === "compaction")
      .reduce((n, e) => n + (e.payload.usage?.[key] ?? 0), 0);
  return (
    <>
      <div className="muted small">
        {turns.length} turns · {calls.length} tool calls ({calls.filter((c) => !c.payload.ok).length} failed) ·{" "}
        {formatTokens({
          input: sum("input"),
          cache_read: sum("cache_read"),
          cache_write: sum("cache_write"),
          output: sum("output"),
          reasoning: sum("reasoning"),
        })}{" "}
        tokens
        {typeof usage?.total_charged === "number" && <> · ${usage.total_charged.toFixed(4)}</>}
      </div>
      <ol className="timeline">
        {events.map((e) => (
          <Entry key={e.id} event={e} handles={handles} />
        ))}
      </ol>
    </>
  );
}

function OpenModelTrace({ events, model }: { events: any[]; model?: string }) {
  const [all, setAll] = useState(false);
  const records = events.flatMap((e) => e.payload.records ?? []);
  const generations = records.filter((r: any) => r.kind === "generate");
  const omitted = events.reduce((n, e) => n + (e.payload.omitted ?? 0), 0);
  const tokens = (key: string) => records.reduce((n: number, r: any) => n + (r[key] ?? 0), 0);
  if (!events.length) {
    return <p className="muted">The open model has not been prompted yet in this session.</p>;
  }
  return (
    <>
      <div className="muted small">
        {model} · {generations.length} generation(s), {records.length - generations.length} forward pass(es)
        {omitted ? ` (+${omitted} not kept)` : ""} · prompts {tokens("prompt_tokens").toLocaleString()} tokens → output{" "}
        {tokens("output_tokens").toLocaleString()} tokens ·{" "}
        <label>
          <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} /> show forward passes
        </label>
      </div>
      <ol className="timeline">
        {events.map((e) => {
          const shown = (e.payload.records ?? []).filter((r: any) => all || r.kind === "generate");
          if (!shown.length && !all) return null;
          return (
            <li key={e.id} className="ev model">
              <div className="meta">
                <code>{e.payload.tool}</code>
                {e.payload.turn != null && <> · turn {e.payload.turn}</>} · <code>{e.payload.call_id}</code> ·{" "}
                {new Date(e.ts * 1000).toLocaleTimeString()} · {(e.payload.records ?? []).length} exchange(s)
                {e.payload.omitted ? `, ${e.payload.omitted} more not kept` : ""}
              </div>
              {shown.map((r: any, i: number) => (
                <Exchange key={i} record={r} />
              ))}
            </li>
          );
        })}
      </ol>
    </>
  );
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
          </>
        )}
        {r.split && <span className="badge warn">{r.split}: visible here only</span>}
      </div>
      {system.map((m: any, i: number) => (
        <div key={i} className="instruction">
          <span className="muted small">instruction</span> {contentText(m.content)}
        </div>
      ))}
      <details>
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

function Entry({ event, handles }: { event: any; handles: Record<string, any> }) {
  const p = event.payload;
  const time = new Date(event.ts * 1000).toLocaleTimeString();
  if (event.kind === "start") {
    return (
      <li className="ev turn">
        <div className="meta">
          driver started · {p.provider}:{p.model} · effort {p.effort} · max {p.max_tokens?.toLocaleString()} tokens ·{" "}
          {time}
        </div>
        <details>
          <summary>system prompt</summary>
          <pre className="reasoning">{p.system}</pre>
        </details>
        <details open>
          <summary>first message</summary>
          <pre className="reasoning">{p.message}</pre>
        </details>
      </li>
    );
  }
  if (event.kind === "turn") {
    return (
      <li className="ev turn">
        <div className="meta">
          turn {p.turn} · {p.model} · {formatTokens(p.usage)} tokens · {(p.latency ?? 0).toFixed(1)} s · stop{" "}
          {p.stop} · {time}
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
      </li>
    );
  }
  if (event.kind === "tool_call") {
    const trace = (p.handles ?? []).map((h: string) => handles[h]).find((h: any) => h?.kind === "trace");
    return (
      <li className={`ev tool ${p.ok ? "" : "failed"}`}>
        <div className="meta">
          <code>{p.tool}</code> · {event.source} · {(p.latency ?? 0).toFixed(2)} s · <code>{p.call_id}</code> · {time}
        </div>
        <code className="args">{JSON.stringify(p.args)}</code>
        <pre className="result">{p.summary}</pre>
        {trace && <TraceObject handle={trace} />}
      </li>
    );
  }
  if (event.kind === "notes" || event.kind === "compaction") {
    return (
      <li className="ev note notes">
        {event.kind === "notes" ? "delivered to the model" : `history compacted (${formatTokens(p.usage)} tokens)`} ·{" "}
        {time}
        <pre className="result">{event.kind === "notes" ? (p.notes ?? []).join("\n\n") : p.summary}</pre>
      </li>
    );
  }
  const label =
    event.kind === "lab"
      ? `lab ${p.tier}${p.load_seconds ? ` (${p.load_seconds.toFixed(0)} s)` : ""}${p.reason ? `: ${p.reason}` : ""}`
      : event.kind === "session"
        ? `session ${p.status}${p.reason ? `: ${p.reason}` : ""}`
        : event.kind === "command"
          ? `human: ${p.type}${p.text ? ` "${p.text}"` : ""}`
          : `${event.kind}: ${JSON.stringify(p).slice(0, 300)}`;
  return (
    <li className={`ev note ${event.kind}`}>
      {label} <span className="muted small">{time}</span>
    </li>
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
