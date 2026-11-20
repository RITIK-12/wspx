import { useEffect, useRef, useState } from "react";
import { post, stream, type Session, type TimelineEvent } from "../api";

export function SessionView({ session, onChange }: { session: Session; onChange: () => void }) {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [steer, setSteer] = useState("");
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setEvents([]);
    const close = stream(session.id, 0, (e) => setEvents((prev) => (prev.some((p) => p.id === e.id) ? prev : [...prev, e])));
    return close;
  }, [session.id]);

  useEffect(() => {
    // scrollIntoView may return a Promise in newer browsers: never return it as an effect cleanup.
    bottom.current?.scrollIntoView({ block: "end" });
  }, [events.length]);

  const act = async (action: string, body: object = {}) => {
    await post(`/api/sessions/${session.id}/${action}`, body);
    onChange();
  };

  const cost = events
    .filter((e) => e.kind === "turn")
    .reduce((sum, e) => sum + (e.payload.usage?.input ?? 0) + (e.payload.usage?.output ?? 0), 0);

  return (
    <div className="session-view">
      <div className="controls">
        <div>
          <strong>{session.task || "Interactive session"}</strong>
          <div className="muted small">
            {session.model} · {session.driver ?? "no driver"} · {cost.toLocaleString()} tokens so far
            {session.reason && <> · {session.reason}</>}
          </div>
        </div>
        <span className="spacer" />
        {session.status === "running" && <button onClick={() => act("pause")}>Pause</button>}
        {session.status === "paused" && <button onClick={() => act("resume")}>Resume</button>}
        {session.driver && session.status !== "running" && session.status !== "paused" && (
          <button onClick={() => act("start")}>Run driver</button>
        )}
        <button className="danger" onClick={() => act("stop")}>
          Stop
        </button>
      </div>
      <form className="steer" onSubmit={(e) => (e.preventDefault(), steer && act("inject", { text: steer }).then(() => setSteer("")))}>
        <input placeholder="Steer the driver…" value={steer} onChange={(e) => setSteer(e.target.value)} />
        <button type="submit" disabled={!steer}>
          Send
        </button>
      </form>
      <ol className="timeline">
        {events.map((e) => (
          <Event key={e.id} event={e} />
        ))}
        <div ref={bottom} />
      </ol>
    </div>
  );
}

function Event({ event }: { event: TimelineEvent }) {
  const p = event.payload;
  const time = new Date(event.ts * 1000).toLocaleTimeString();
  if (event.kind === "turn") {
    return (
      <li className="ev turn">
        <div className="meta">
          turn {p.turn} · {p.model} · {p.usage?.input ?? 0}→{p.usage?.output ?? 0} tok · {(p.latency ?? 0).toFixed(1)} s · {time}
        </div>
        {p.reasoning ? (
          <details>
            <summary>reasoning</summary>
            <pre className="reasoning">{p.reasoning}</pre>
          </details>
        ) : p.reasoning_available === false ? (
          <div className="muted small">reasoning not returned by this provider</div>
        ) : null}
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
    return (
      <li className={`ev tool ${p.ok ? "" : "failed"}`}>
        <div className="meta">
          <code>{p.tool}</code> · {event.source} · {(p.latency ?? 0).toFixed(2)} s · <code>{p.call_id}</code>
          {p.handles?.length > 0 && <> · {p.handles.map((h: string) => <code key={h} className="handle">{h}</code>)}</>}
        </div>
        <pre className="result">{p.summary}</pre>
      </li>
    );
  }
  const label =
    event.kind === "lab"
      ? `lab ${p.tier}${p.load_seconds ? ` (${p.load_seconds.toFixed(0)} s)` : ""}${p.reason ? `: ${p.reason}` : ""}`
      : event.kind === "loop"
        ? `loop ${p.action}: ${p.message}`
        : event.kind === "session"
          ? `session ${p.status}${p.reason ? `: ${p.reason}` : ""}`
          : event.kind === "command"
            ? `human: ${p.type}${p.text ? ` "${p.text}"` : ""}`
            : event.kind === "hook"
              ? `${p.client} ${p.event ?? ""} ${p.tool ?? ""}${p.call_id ? ` → ${p.call_id}` : ""}`
              : `${event.kind}: ${JSON.stringify(p).slice(0, 200)}`;
  return <li className={`ev note ${event.kind}`}>{label} <span className="muted small">{time}</span></li>;
}
