import { useEffect, useState } from "react";
import { api, type Session } from "../api";

export function LabView({ session }: { session: Session }) {
  const [usage, setUsage] = useState<any>(null);
  const [limits, setLimits] = useState<any>(null);
  const [doctor, setDoctor] = useState<any>(null);

  useEffect(() => {
    api(`/api/usage?session=${session.id}`).then(setUsage).catch((e) => setUsage({ error: String(e) }));
    api("/api/limits").then(setLimits).catch((e) => setLimits({ error: String(e) }));
    api("/api/doctor").then(setDoctor).catch(() => setDoctor(null));
  }, [session.id]);

  return (
    <div className="lab">
      <h3>Lab</h3>
      <dl>
        <dt>tier</dt>
        <dd>{session.tier}</dd>
        <dt>GPU</dt>
        <dd>{session.gpu ?? "CPU"}</dd>
        <dt>model</dt>
        <dd>{session.model}</dd>
        <dt>last load</dt>
        <dd>{session.load_seconds ? `${session.load_seconds.toFixed(1)} s` : "–"}</dd>
        <dt>fingerprint</dt>
        <dd>
          <code>{session.fingerprint?.slice(0, 23) ?? "–"}</code>
        </dd>
        <dt>isolation</dt>
        <dd>on Modal: no outbound network, no secrets, reachable only from the control plane</dd>
      </dl>
      <h3>Spend</h3>
      {usage?.error ? (
        <p className="muted">{noGateway(usage.error)}</p>
      ) : usage ? (
        <>
          <table>
            <thead>
              <tr>
                <th>provider:model</th>
                <th>calls</th>
                <th>input</th>
                <th>output</th>
                <th>cost</th>
              </tr>
            </thead>
            <tbody>
              {usage.groups.map((g: any) => (
                <tr key={g.provider + g.model}>
                  <td>
                    {g.provider}:{g.model}
                  </td>
                  <td>{g.n}</td>
                  <td>{(g.i ?? 0).toLocaleString()}</td>
                  <td>{(g.o ?? 0).toLocaleString()}</td>
                  <td>{g.c == null ? "unknown" : `$${g.c.toFixed(4)}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted">
            charged ${usage.total_charged.toFixed(4)} · held ${usage.held.toFixed(4)}
          </p>
        </>
      ) : (
        <p className="muted">Loading…</p>
      )}
      <h3>Limits</h3>
      {limits?.error ? (
        <p className="muted">{noGateway(limits.error)}</p>
      ) : limits?.limits?.length ? (
        <ul>
          {limits.limits.map((l: any) => (
            <li key={l.id}>
              <code>{l.id}</code> {l.describe}
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">No limits set. wspx never sets any by default; add them with <code>wspx limits set</code>.</p>
      )}
      {doctor && (
        <>
          <h3>Provider caps</h3>
          <ul>
            {Object.entries<string>(doctor.provider_caps).map(([p, url]) => (
              <li key={p}>
                {p}: <a href={url} target="_blank" rel="noreferrer">{url}</a>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

function noGateway(error: string): string {
  return error.includes("503")
    ? "Metering and limits live in the LLM gateway; this deployment runs without one."
    : error;
}
