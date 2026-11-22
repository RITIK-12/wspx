import { useEffect, useState } from "react";
import { api, formatTokens, type Session } from "../api";

// The session's whole record, in order, from creation: what the driver was given, every turn
// with its reasoning and token counts, every tool call with its full result, every human
// action and lab change. Export saves the same document `wspx sessions export` reads.
export function TracesView({ session }: { session: Session }) {
  const [record, setRecord] = useState<any>(null);
  const [error, setError] = useState("");
  const [handles, setHandles] = useState<Record<string, any>>({});
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
  const turns = events.filter((e) => e.kind === "turn");
  const calls = events.filter((e) => e.kind === "tool_call");
  const sum = (key: string) =>
    events
      .filter((e) => e.kind === "turn" || e.kind === "compaction")
      .reduce((n, e) => n + (e.payload.usage?.[key] ?? 0), 0);
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
          {typeof record.usage?.total_charged === "number" && <> · ${record.usage.total_charged.toFixed(4)}</>}
        </div>
        <span className="spacer" />
        <button onClick={save} title="The same record `wspx sessions export` writes as files">
          Export
        </button>
      </div>
      <ol className="timeline">
        {events.map((e) => (
          <Entry key={e.id} event={e} handles={handles} />
        ))}
      </ol>
    </div>
  );
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
