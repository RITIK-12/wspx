import { useCallback, useEffect, useState } from "react";
import { api, ApiError, hasToken, initialSession, post, setToken, type Session } from "./api";
import { SessionView } from "./views/SessionView";
import { ArchitectureView } from "./views/ArchitectureView";
import { CommitsView } from "./views/CommitsView";
import { EvalsView } from "./views/EvalsView";
import { TracesView } from "./views/TracesView";
import { LabView } from "./views/LabView";

const TABS = ["Session", "Architecture", "Commits", "Evals", "Traces", "Lab & cost"] as const;
type Tab = (typeof TABS)[number];

export function App() {
  const [authed, setAuthed] = useState(hasToken());
  const [sessions, setSessions] = useState<Session[]>([]);
  const [current, setCurrent] = useState<string | undefined>(initialSession);
  const [tab, setTab] = useState<Tab>("Session");
  const [error, setError] = useState<string>("");
  const [halted, setHalted] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const data = await api<{ sessions: Session[] }>("/api/sessions");
      setSessions(data.sessions);
      const health = await api<{ halted: boolean }>("/healthz");
      setHalted(health.halted);
      setCurrent((c) => c ?? data.sessions[0]?.id);
      setError("");
    } catch (e) {
      if (e instanceof ApiError && e.message.startsWith("401")) setAuthed(false);
      else setError(String(e));
    }
  }, []);

  useEffect(() => {
    if (!authed) return;
    refresh();
    const timer = setInterval(refresh, 4000);
    return () => clearInterval(timer);
  }, [authed, refresh]);

  if (!authed) return <Login onToken={(t) => (setToken(t), setAuthed(true))} />;

  const session = sessions.find((s) => s.id === current);

  const stopEverything = async () => {
    if (!confirm("Stop every session, release every GPU and halt wspx until you resume?")) return;
    await post("/api/kill", {});
    refresh();
  };

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">wspx</span>
        {session && <TierBanner session={session} />}
        {halted && (
          <span className="badge danger">
            halted · <button className="link" onClick={() => post("/api/resume", {}).then(refresh)}>resume</button>
          </span>
        )}
        <span className="spacer" />
        <button className="danger" onClick={stopEverything} title="wspx kill --all">
          Stop everything
        </button>
      </header>
      <div className="body">
        <aside className="sidebar">
          <NewSession onCreated={(id) => (setCurrent(id), refresh())} />
          <h3>Sessions</h3>
          {sessions.length === 0 && <p className="muted">No sessions yet.</p>}
          <ul className="sessions">
            {sessions.map((s) => (
              <li key={s.id} className={s.id === current ? "active" : ""} onClick={() => setCurrent(s.id)}>
                <div className="row">
                  <code>{s.id}</code>
                  <span className={`dot ${s.tier}`} title={`lab ${s.tier}`} />
                </div>
                <div className="muted small">{s.model}</div>
                <div className="small">
                  <span className={`status ${s.status}`}>{s.status}</span> {s.driver ?? "interactive"}
                </div>
              </li>
            ))}
          </ul>
        </aside>
        <main className="main">
          {error && <div className="error">{error}</div>}
          {!session ? (
            <div className="empty">Start a session to see it here.</div>
          ) : (
            <>
              <nav className="tabs">
                {TABS.map((t) => (
                  <button key={t} className={t === tab ? "tab active" : "tab"} onClick={() => setTab(t)}>
                    {t}
                  </button>
                ))}
              </nav>
              <section className="panel">
                {tab === "Session" && <SessionView session={session} onChange={refresh} />}
                {tab === "Architecture" && <ArchitectureView session={session} />}
                {tab === "Commits" && <CommitsView session={session} />}
                {tab === "Evals" && <EvalsView session={session} />}
                {tab === "Traces" && <TracesView session={session} />}
                {tab === "Lab & cost" && <LabView session={session} />}
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function TierBanner({ session }: { session: Session }) {
  const text =
    session.tier === "hot"
      ? `Lab hot${session.gpu ? ` · ${session.gpu}` : ""}`
      : session.tier === "grace"
        ? "Lab idle (grace period)"
        : session.load_seconds
          ? `Lab cold · wakes in ~${Math.round(session.load_seconds)} s`
          : "Lab cold";
  return <span className={`badge tier-${session.tier}`}>{text}</span>;
}

function Login({ onToken }: { onToken: (t: string) => void }) {
  const [value, setValue] = useState("");
  return (
    <div className="login">
      <h1>wspx</h1>
      <p className="muted">Open the UI with <code>wspx open</code>, or paste your API token.</p>
      <form onSubmit={(e) => (e.preventDefault(), value && onToken(value.trim()))}>
        <input type="password" placeholder="wspx_…" value={value} onChange={(e) => setValue(e.target.value)} />
        <button type="submit">Continue</button>
      </form>
    </div>
  );
}

function NewSession({ onCreated }: { onCreated: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  const [model, setModel] = useState("");
  const [driver, setDriver] = useState("anthropic:claude-opus-5-5");
  const [gpu, setGpu] = useState("");
  const [task, setTask] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  if (!open)
    return (
      <button className="wide" onClick={() => setOpen(true)}>
        New session
      </button>
    );
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      const rec = await post<Session>("/api/sessions", {
        model,
        driver: driver || null,
        gpu: gpu || null,
        task,
      });
      setOpen(false);
      onCreated(rec.id);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <form className="new-session" onSubmit={submit}>
      <label>
        Model <input required placeholder="Qwen/Qwen3.5-4B" value={model} onChange={(e) => setModel(e.target.value)} />
      </label>
      <label>
        Driver <input placeholder="none: interactive" value={driver} onChange={(e) => setDriver(e.target.value)} />
      </label>
      <label>
        GPU <input placeholder="none: CPU" value={gpu} onChange={(e) => setGpu(e.target.value)} />
      </label>
      <label>
        Task <textarea rows={3} value={task} onChange={(e) => setTask(e.target.value)} />
      </label>
      {error && <div className="error">{error}</div>}
      <div className="row">
        <button type="submit" disabled={busy}>
          {busy ? "Starting…" : "Start"}
        </button>
        <button type="button" className="ghost" onClick={() => setOpen(false)}>
          Cancel
        </button>
      </div>
    </form>
  );
}
