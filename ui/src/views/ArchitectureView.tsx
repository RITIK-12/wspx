import { useEffect, useState } from "react";
import { api, toolCall, type Session } from "../api";

interface Role {
  address: string;
  param: string;
  kind: string;
  shape: number[];
  split: number[][] | null;
  experts: number | null;
}

export function ArchitectureView({ session }: { session: Session }) {
  const [arch, setArch] = useState<any>(null);
  const [selected, setSelected] = useState<Role | null>(null);
  const [detail, setDetail] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api(`/api/sessions/${session.id}/architecture`).then(setArch).catch((e) => setArch({ error: String(e) }));
  }, [session.id, session.fingerprint]);

  if (!arch) return <p className="muted">Loading…</p>;
  if (arch.error) return <div className="error">{arch.error}</div>;
  if (!arch.available) return <p className="muted">The architecture appears once the lab has loaded the model.</p>;

  const show = async (role: Role) => {
    setSelected(role);
    setDetail("");
  };
  const inspect = async () => {
    if (!selected) return;
    setBusy(true);
    try {
      const r = await toolCall(session.id, "show", { address: selected.address });
      setDetail(r.summary);
    } catch (e) {
      setDetail(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="arch">
      <p className="muted">
        {arch.arch} · {(arch.numel / 1e6).toFixed(1)}M params in {arch.params} tensors · {arch.blocks.length} blocks on
        the residual stream{arch.tied ? " · embeddings tied" : ""}
        {arch.unexercised.length > 0 && <span className="badge danger"> {arch.unexercised.length} unexercised</span>}
      </p>
      <div className="arch-layout">
        <div className="blocks">
          <div className="stream-cap">
            {arch.specials
              .filter((r: Role) => r.kind === "embed" || r.kind === "pos_embed" || r.kind === "stem")
              .map((r: Role) => (
                <Chip key={r.address + r.param} role={r} onClick={show} active={selected === r} />
              ))}
          </div>
          {arch.blocks.map((b: any) => (
            <div key={b.index} className="block">
              <div className="block-head">
                <strong>L{b.index}</strong> <span className="muted small">{b.module}</span>
              </div>
              {b.branches.map((branch: string) => (
                <div key={branch} className={`branch ${branch.split(".")[0]}`}>
                  <span className="branch-name">{branch}</span>
                  {b.roles
                    .filter((r: Role) => r.address.startsWith(`L${b.index}.${branch}.`))
                    .map((r: Role) => (
                      <Chip key={r.address + r.param} role={r} onClick={show} active={selected === r} />
                    ))}
                </div>
              ))}
            </div>
          ))}
          <div className="stream-cap">
            {arch.specials
              .filter((r: Role) => r.kind === "final_norm" || r.kind === "unembed" || r.kind === "bias")
              .map((r: Role) => (
                <Chip key={r.address + r.param} role={r} onClick={show} active={selected === r} />
              ))}
          </div>
        </div>
        <div className="inspector">
          {selected ? (
            <>
              <h3>
                <code>{selected.address}</code>
              </h3>
              <dl>
                <dt>parameter</dt>
                <dd>
                  <code>{selected.param}</code>
                </dd>
                <dt>role</dt>
                <dd>{selected.kind}</dd>
                <dt>shape</dt>
                <dd>{selected.shape.join(" × ")}</dd>
                {selected.split && (
                  <>
                    <dt>fused rows</dt>
                    <dd>{selected.split.map((r) => `${r[0]}:${r[1]}${r[2] !== 1 ? `:${r[2]}` : ""}`).join(", ")}</dd>
                  </>
                )}
                {selected.experts && (
                  <>
                    <dt>experts</dt>
                    <dd>{selected.experts}</dd>
                  </>
                )}
              </dl>
              <button onClick={inspect} disabled={busy}>
                {busy ? "Asking the lab…" : "Stats, histogram and vocabulary"}
              </button>
              {detail && <pre className="result">{detail}</pre>}
            </>
          ) : (
            <p className="muted">Select a weight to inspect it.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function Chip({ role, onClick, active }: { role: Role; onClick: (r: Role) => void; active: boolean }) {
  const name = role.address.split(".").slice(2).join(".") || role.address;
  return (
    <button className={`chip ${role.kind} ${active ? "active" : ""}`} onClick={() => onClick(role)} title={role.param}>
      {name}
    </button>
  );
}
