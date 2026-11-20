import { useEffect, useState } from "react";
import { api, type Session } from "../api";

export function TracesView({ session }: { session: Session }) {
  const [traces, setTraces] = useState<any[]>([]);
  const [trace, setTrace] = useState<any>(null);

  useEffect(() => {
    api(`/api/sessions/${session.id}/handles`)
      .then((d) => setTraces(d.handles.filter((h: any) => h.kind === "trace").reverse()))
      .catch(() => setTraces([]));
  }, [session.id]);

  const open = async (h: any) => setTrace({ handle: h, ...(await api(`/api/objects/${h.ref}`)) });

  return (
    <div className="traces">
      <ul className="trace-list">
        {traces.length === 0 && <li className="muted">No traces yet. Run <code>wspx trace</code>.</li>}
        {traces.map((h) => (
          <li key={h.key} onClick={() => open(h)} className={trace?.handle?.key === h.key ? "active" : ""}>
            <code>{h.key}</code> <span className="muted small">{h.description}</span>
          </li>
        ))}
      </ul>
      {trace && (
        <div className="trace">
          {trace.kind === "forward" && <LogitLens arrays={trace.arrays} />}
          {trace.kind === "diff" && <Divergence values={trace.arrays.divergence} />}
          {trace.kind === "attribution" && (
            <ol className="ranking">
              {trace.data.ranking.map(([addr, score]: [string, number]) => (
                <li key={addr}>
                  <code>{addr}</code> <span className="bar" style={{ width: `${Math.min(100, Math.abs(score) * 400)}%` }} />{" "}
                  {score.toFixed(4)}
                </li>
              ))}
            </ol>
          )}
          <pre className="result">{trace.summary}</pre>
        </div>
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
          <th>block</th>
          <th>|h|</th>
          <th>top tokens at the last position</th>
        </tr>
      </thead>
      <tbody>
        {arrays.lens_ids.map((ids: number[], i: number) => (
          <tr key={i}>
            <td>L{i}</td>
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

function Divergence({ values }: { values: number[] }) {
  const max = Math.max(...values, 1e-9);
  return (
    <div className="divergence">
      {values.map((v, i) => (
        <div key={i} className="div-row">
          <span>L{i}</span>
          <span className="bar" style={{ width: `${(100 * v) / max}%` }} />
          <span className="muted small">{v.toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
}
