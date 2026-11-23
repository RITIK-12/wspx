import { useEffect, useState } from "react";
import { api, toolCall, type Session } from "../api";

type Hunk = {
  where: string;
  changed: number;
  values: { at: number[]; old: number; new: number }[];
  more?: number;
};
type DiffRow = {
  tensor: string;
  address: string;
  norm: number;
  rel: number;
  rank_text: string;
  touched: number;
  slices: string;
  precision: string;
  hunks?: Hunk[];
};
type DiffResult = { a: string; b: string; summary: string; ok: boolean; data: any };

const fmt = (v: number) => (Math.abs(v) >= 1e-3 || v === 0 ? v.toFixed(6) : v.toExponential(4));

/** A weight diff laid out as an IDE shows a git diff: changed tensors on the left, the
 *  selected one's hunks on the right, each value before (−) and after (+). */
function DiffView({ result }: { result: DiffResult }) {
  const rows: DiffRow[] = result.data?.tensors ?? [];
  const [open, setOpen] = useState<string>(rows[0]?.tensor ?? "");
  const row = rows.find((r) => r.tensor === open) ?? rows[0];
  const behaviour: string[] = result.data?.behaviour ?? [];
  const cancelled: string[] = result.data?.cancelled ?? [];
  if (!result.ok) return <div className="error">{result.summary}</div>;
  return (
    <div className="gitdiff">
      <div className="gitdiff-head">
        <code>{result.a}</code> → <code>{result.b}</code> · {rows.length} tensor(s) changed
        {cancelled.length > 0 && <span className="muted"> · {cancelled.length} patched with no net change</span>}
      </div>
      {rows.length === 0 ? (
        <p className="muted">No weights differ.</p>
      ) : (
        <div className="gitdiff-split">
          <ul className="gitdiff-files" role="listbox" aria-label="Changed tensors">
            {rows.map((r) => (
              <li
                key={r.tensor}
                role="option"
                aria-selected={r === row}
                className={r === row ? "on" : ""}
                onClick={() => setOpen(r.tensor)}
              >
                <span className="gitdiff-mark">M</span>
                <code className="gitdiff-name">{r.address}</code>
                <span className="gitdiff-stat">
                  {(r.rel * 100).toFixed(2)}% · {r.touched.toLocaleString()}
                </span>
              </li>
            ))}
          </ul>
          {row && (
            <div className="gitdiff-body">
              <div className="gitdiff-file">
                <code>{row.address}</code>
                <span className="muted">
                  ‖Δ‖ {row.norm.toPrecision(3)} ({(row.rel * 100).toFixed(2)}%) · rank {row.rank_text} ·{" "}
                  {row.touched.toLocaleString()} changed in {row.slices}
                  {row.precision === "adapter" ? " · adapter" : ""}
                </span>
              </div>
              {(row.hunks ?? []).map((h) => (
                <div key={h.where} className="gitdiff-hunk">
                  <div className="gitdiff-at">
                    @@ {h.where} · {h.changed.toLocaleString()} changed @@
                  </div>
                  <div className="gitdiff-row gitdiff-cols">
                    <span>index</span>
                    <span>− {result.a}</span>
                    <span>+ {result.b}</span>
                  </div>
                  {h.values.map((v) => (
                    <div key={v.at.join(",")} className="gitdiff-row">
                      <span className="idx">[{v.at.join(", ")}]</span>
                      <span className="del">
                        <span className="sign">−</span>
                        {fmt(v.old)}
                      </span>
                      <span className="add">
                        <span className="sign">+</span>
                        {fmt(v.new)}
                      </span>
                    </div>
                  ))}
                  {h.changed > h.values.length && (
                    <div className="gitdiff-more">… {(h.changed - h.values.length).toLocaleString()} more changed values</div>
                  )}
                  {h.more ? <div className="gitdiff-more">… {h.more.toLocaleString()} more hunks</div> : null}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      {behaviour.length > 0 && <pre className="result">{behaviour.join("\n")}</pre>}
    </div>
  );
}

export function CommitsView({ session }: { session: Session }) {
  const [data, setData] = useState<any>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [diff, setDiff] = useState<DiffResult | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api(`/api/sessions/${session.id}/commits`).then(setData).catch((e) => setData({ error: String(e) }));
  }, [session.id]);

  if (!data) return <p className="muted">Loading…</p>;
  if (data.error) return <div className="error">{data.error}</div>;

  const toggle = (short: string) =>
    setPicked((p) => (p.includes(short) ? p.filter((x) => x !== short) : [...p, short].slice(-2)));
  const branchesAt: Record<string, string[]> = {};
  for (const [name, oid] of Object.entries<string>(data.branches)) (branchesAt[oid] ??= []).push(name);

  // older → newer, as git diff shows a change: the list is newest first
  const order: string[] = data.commits.map((c: any) => c.short);
  const ordered = [...picked].sort((x, y) => order.indexOf(y) - order.indexOf(x));
  const compare = async () => {
    const [a, b] = ordered;
    setBusy(true);
    try {
      const r = await toolCall(session.id, "vcs", { action: "diff", a, b });
      setDiff({ a, b, ...r });
    } catch (e) {
      setDiff({ a, b, summary: String(e), ok: false, data: null });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="commits">
      <div className="row">
        <span className="muted">
          on <strong>{data.branch ?? "(detached)"}</strong> · select two commits to compare
        </span>
        <span className="spacer" />
        <button disabled={picked.length !== 2 || busy} onClick={compare} aria-busy={busy}>
          {busy ? "Comparing…" : `Compare ${ordered.join(" → ")}`}
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
      {diff && <DiffView key={`${diff.a}..${diff.b}`} result={diff} />}
    </div>
  );
}
