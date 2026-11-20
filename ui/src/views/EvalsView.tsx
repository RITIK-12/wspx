import { useEffect, useState } from "react";
import { api, type Session } from "../api";

export function EvalsView({ session }: { session: Session }) {
  const [data, setData] = useState<any>(null);
  const [cell, setCell] = useState<any>(null);

  useEffect(() => {
    api(`/api/sessions/${session.id}/runs`).then(setData).catch((e) => setData({ error: String(e) }));
  }, [session.id]);

  if (!data) return <p className="muted">Loading…</p>;
  if (data.error) return <div className="error">{data.error}</div>;
  if (data.runs.length === 0) return <p className="muted">No test runs yet. Run <code>wspx test</code>.</p>;

  const suites = [...new Set<string>(data.runs.map((r: any) => r.suite))].sort();
  const trees = [...new Set<string>(data.runs.map((r: any) => r.tree))];
  const label = (tree: string) => data.runs.find((r: any) => r.tree === tree).commits.join(", ");

  return (
    <div className="evals">
      <p className="muted">
        Commits × suites. Held-out suites show per-item results here only; the driver sees aggregates.
        {data.heldout_queries > 0 && <> Held-out queries so far: <strong>{data.heldout_queries}</strong>.</>}
      </p>
      <table className="matrix">
        <thead>
          <tr>
            <th>weights</th>
            {suites.map((s) => (
              <th key={s}>{s}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {trees.map((tree) => (
            <tr key={tree}>
              <td>
                <code>{label(tree)}</code>
              </td>
              {suites.map((s) => {
                const run = data.runs.find((r: any) => r.tree === tree && r.suite === s);
                if (!run) return <td key={s} className="muted">–</td>;
                const pct = Math.round(100 * (run.pass_rate ?? 0));
                return (
                  <td key={s} className="cell" onClick={() => setCell(run)} style={{ ["--pct" as string]: `${pct}%` }}>
                    {pct}% <span className="muted small">({run.passed}/{run.n})</span>
                    {run.split === "heldout" && <span className="badge">held-out</span>}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {cell && (
        <div className="items">
          <h3>
            {cell.suite} on <code>{cell.commits.join(", ")}</code>
          </h3>
          <table>
            <thead>
              <tr>
                <th>item</th>
                <th>result</th>
                <th>check</th>
                <th>output</th>
              </tr>
            </thead>
            <tbody>
              {cell.checks.map((c: any) => (
                <tr key={c.item} className={c.passed ? "" : "failed"}>
                  <td>{c.item}</td>
                  <td>{c.passed ? "pass" : "fail"}</td>
                  <td>
                    {c.kind} {c.detail}
                  </td>
                  <td>
                    <code>{(c.output ?? (typeof c.score === "number" ? c.score.toFixed(4) : "")).slice(0, 200)}</code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
