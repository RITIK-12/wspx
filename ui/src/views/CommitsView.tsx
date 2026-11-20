import { useEffect, useState } from "react";
import { api, toolCall, type Session } from "../api";

export function CommitsView({ session }: { session: Session }) {
  const [data, setData] = useState<any>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [diff, setDiff] = useState("");

  useEffect(() => {
    api(`/api/sessions/${session.id}/commits`).then(setData).catch((e) => setData({ error: String(e) }));
  }, [session.id]);

  if (!data) return <p className="muted">Loading…</p>;
  if (data.error) return <div className="error">{data.error}</div>;

  const toggle = (short: string) =>
    setPicked((p) => (p.includes(short) ? p.filter((x) => x !== short) : [...p, short].slice(-2)));
  const branchesAt: Record<string, string[]> = {};
  for (const [name, oid] of Object.entries<string>(data.branches)) (branchesAt[oid] ??= []).push(name);

  const compare = async () => {
    const [a, b] = picked;
    const r = await toolCall(session.id, "vcs", { action: "diff", a, b });
    setDiff(r.summary);
  };

  return (
    <div className="commits">
      <div className="row">
        <span className="muted">
          on <strong>{data.branch ?? "(detached)"}</strong> · select two commits to compare
        </span>
        <span className="spacer" />
        <button disabled={picked.length !== 2} onClick={compare}>
          Compare {picked.join(" ↔ ")}
        </button>
      </div>
      {data.commits.length === 0 && <p className="muted">No commits yet.</p>}
      <ol className="commit-list">
        {data.commits.map((c: any) => (
          <li key={c.oid} className={picked.includes(c.short) ? "picked" : ""} onClick={() => toggle(c.short)}>
            <span className={`graph ${c.parents.length > 1 ? "merge" : ""}`} />
            <code>{c.short}</code>
            {(branchesAt[c.oid] ?? []).map((b) => (
              <span key={b} className="badge">
                {b}
              </span>
            ))}
            {c.status !== "ok" && <span className="badge danger">{c.status}</span>}
            {c.forced && <span className="badge warn">forced</span>}
            <span className="message">{c.message.split("\n")[0]}</span>
            <span className="muted small">
              {c.patches.map((p: any) => p.op).join(", ") || (c.merge ? "merge" : "")} · {c.tensors} tensor(s) ·{" "}
              {new Date(c.time * 1000).toLocaleString()}
            </span>
          </li>
        ))}
      </ol>
      {diff && <pre className="result">{diff}</pre>}
    </div>
  );
}
