import { useEffect, useMemo, useRef, useState } from "react";
import { api, toolCall, type Session } from "../api";

// The compiled map as a diagram, read top to bottom like the classic transformer figure:
// tokens → embedding → the layer stack (a loop drawn around layers that run more than once)
// → final norm → unembedding. Pick a layer to see its sublayers in the order they run, on the
// residual stream; pick a weight to see its shape, axes and kind, and load its values (a
// magnitude map, histogram and spectrum) from the lab. Everything comes from the module tree
// and the traced ops, so it is the model as it ran, not a template.

type Arch = any;

const VARIANT_COLORS = ["#6366f1", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#a855f7", "#14b8a6", "#f43f5e"];

export function ArchitectureView({ session }: { session: Session }) {
  const [arch, setArch] = useState<Arch>(null);
  const [layer, setLayer] = useState(0);
  const [tensor, setTensor] = useState<string | null>(null);
  const [view, setView] = useState<"diagram" | "3d">("diagram");

  useEffect(() => {
    api(`/api/sessions/${session.id}/architecture`).then(setArch).catch((e) => setArch({ error: String(e) }));
  }, [session.id, session.fingerprint]);

  const index = useMemo(() => (arch?.available ? buildIndex(arch) : null), [arch]);

  if (!arch) return <p className="muted">Loading…</p>;
  if (arch.error) return <div className="error">{arch.error}</div>;
  if (!arch.available || !index) return <p className="muted">The architecture appears once the lab has loaded the model.</p>;

  const pick = (param: string) => {
    setTensor(param);
    const l = arch.tensors[param]?.layer;
    if (l != null) setLayer(l);
  };

  return (
    <div className="archx">
      <Summary arch={arch} onFind={pick} index={index} />
      <div className="archx-grid">
        <section className="archx-flow">
          <ModelFlow arch={arch} index={index} layer={layer} onLayer={setLayer} tensor={tensor} onTensor={pick} view={view} onView={setView} />
        </section>
        <section className="archx-layer">
          {arch.stack ? (
            <LayerDetail arch={arch} index={index} layer={layer} onLayer={setLayer} tensor={tensor} onTensor={pick} />
          ) : (
            <p className="muted">No layer stack (module list) in this model: its tensors are listed on the left.</p>
          )}
        </section>
        <section className="archx-weights">
          <WeightPanel arch={arch} session={session} param={tensor} />
        </section>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- data

interface Index {
  byKey: Map<string, string>;
  variantOf: number[];
  layerParams: (i: number, inner: string) => string | undefined;
}

function buildIndex(arch: Arch): Index {
  const byKey = new Map<string, string>();
  for (const [param, t] of Object.entries<any>(arch.tensors)) byKey.set(t.key, param);
  const variantOf: number[] = [];
  (arch.stack?.variants ?? []).forEach((v: any, vi: number) => v.layers.forEach((l: number) => (variantOf[l] = vi)));
  return { byKey, variantOf, layerParams: (i, inner) => byKey.get(inner ? `L${i}.${inner}` : `L${i}`) };
}

function letter(i: number): string {
  return String.fromCharCode(65 + (i % 26));
}

function human(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function numel(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

function spans(layers: number[]): string {
  const out: string[] = [];
  let start = layers[0];
  let prev = layers[0];
  for (const l of [...layers.slice(1), Infinity]) {
    if (l !== prev + 1) {
      out.push(start === prev ? `L${start}` : `L${start}-${prev}`);
      start = l;
    }
    prev = l;
  }
  return out.join(", ");
}

// --------------------------------------------------------------------------- summary

function Summary({ arch, index, onFind }: { arch: Arch; index: Index; onFind: (param: string) => void }) {
  const [q, setQ] = useState("");
  const keys = useMemo(() => [...index.byKey.keys()], [index]);
  const hits = q.length >= 2 ? keys.filter((k) => k.toLowerCase().includes(q.toLowerCase())).slice(0, 12) : [];
  const s = arch.stack;
  const looped = s ? s.segments.filter((g: number[]) => g[2] > 1) : [];
  return (
    <div className="archx-summary">
      <div>
        <strong>{arch.arch}</strong>
        <div className="muted small">
          {human(arch.numel)} params in {arch.params} tensors · width {arch.hidden ?? "?"}
          {s && (
            <>
              {" "}
              · {s.size} layers, {s.calls} layer calls per forward pass
              {looped.length > 0 && <> · loop {looped.map((g: number[]) => `L${g[0]}-${g[1]} ×${g[2]}`).join(", ")}</>}
            </>
          )}
          {arch.tied && " · embeddings tied"}
          {arch.unexercised.length > 0 && <span className="badge danger">{arch.unexercised.length} unexercised</span>}
        </div>
      </div>
      <span className="spacer" />
      <div className="archx-find">
        <input placeholder="Find a weight by key…" value={q} onChange={(e) => setQ(e.target.value)} />
        {hits.length > 0 && (
          <div className="archx-hits">
            {hits.map((k) => (
              <button
                key={k}
                className="link"
                onClick={() => {
                  onFind(index.byKey.get(k)!);
                  setQ("");
                }}
              >
                <code>{k}</code>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- the model, top to bottom

function ModelFlow(props: {
  arch: Arch;
  index: Index;
  layer: number;
  onLayer: (l: number) => void;
  tensor: string | null;
  onTensor: (p: string) => void;
  view: "diagram" | "3d";
  onView: (v: "diagram" | "3d") => void;
}) {
  const { arch, index, layer, onLayer, tensor, onTensor, view, onView } = props;
  const t = arch.tensors;
  const before = [arch.embed, ...arch.pos_embed, ...arch.stem].filter(Boolean);
  const after = [...arch.final_norm_params, ...arch.final.filter((n: string) => !arch.final_norm_params.includes(n))];
  const vision = arch.others.filter((o: any) => o.vision_only);
  const extra = arch.others.filter((o: any) => !o.vision_only);
  const chips = (names: string[]) => (
    <div className="flow-chips">
      {names.map((n) => (
        <button key={n} className={`wchip ${n === tensor ? "active" : ""}`} onClick={() => onTensor(n)} title={n}>
          {t[n].key} <span className="muted">{t[n].shape.join("×")}</span>
        </button>
      ))}
    </div>
  );
  return (
    <div className="flow">
      <div className="flow-io">input tokens</div>
      <Arrow />
      {vision.map((o: any) => (
        <div key={o.path} className="flow-side">
          <div className="fbox c-other">
            <div className="fbox-title">Vision tower</div>
            <div className="muted small">
              {o.path} · {o.tensors.length} tensors · {human(o.numel)}
              {o.stack && <> · its own stack of {o.stack.size} layers</>}
            </div>
            <details>
              <summary className="small">tensors</summary>
              {chips(o.tensors.slice(0, 60))}
              {o.tensors.length > 60 && <div className="muted small">… {o.tensors.length - 60} more (use Find)</div>}
            </details>
          </div>
          <div className="flow-join muted small">↘ image tokens join the sequence</div>
        </div>
      ))}
      {before.length > 0 && (
        <div className="fbox c-embed">
          <div className="fbox-title">Embedding</div>
          {chips(before)}
        </div>
      )}
      <Arrow />
      {arch.stack ? (
        <div className="fstack">
          <div className="fstack-head">
            <div>
              <div className="fbox-title">Layer stack</div>
              <div className="muted small">
                {arch.stack.path} · {arch.stack.size} layers
              </div>
            </div>
            <span className="spacer" />
            <div className="seg">
              <button className={view === "diagram" ? "on" : ""} onClick={() => onView("diagram")}>
                2D
              </button>
              <button className={view === "3d" ? "on" : ""} onClick={() => onView("3d")}>
                3D
              </button>
            </div>
          </div>
          {view === "diagram" ? (
            <StackDiagram arch={arch} index={index} layer={layer} onLayer={onLayer} />
          ) : (
            <Stack3D arch={arch} index={index} layer={layer} onLayer={onLayer} />
          )}
          <div className="variants">
            {arch.stack.variants.map((v: any, vi: number) => (
              <button key={vi} className="variant" onClick={() => onLayer(v.layers[0])} title={v.class}>
                <span className="swatch" style={{ background: VARIANT_COLORS[vi % VARIANT_COLORS.length] }}>
                  {letter(vi)}
                </span>
                <span>
                  {spans(v.layers)} <span className="muted">· {v.class} · {v.tensors.length} tensors</span>
                </span>
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="fbox c-other">{chips(Object.keys(t).filter((n) => !before.includes(n) && !after.includes(n)))}</div>
      )}
      <Arrow />
      {arch.final_norm_params.length > 0 && (
        <div className="fbox c-norm">
          <div className="fbox-title">Final norm</div>
          {chips(arch.final_norm_params)}
        </div>
      )}
      {after.filter((n) => !arch.final_norm_params.includes(n)).length > 0 && (
        <>
          <div className="fbox c-other">
            <div className="fbox-title">After the stack</div>
            {chips(after.filter((n) => !arch.final_norm_params.includes(n)))}
          </div>
        </>
      )}
      {arch.unembed && (
        <>
          <Arrow />
          <div className="fbox c-linear">
            <div className="fbox-title">Unembedding{arch.tied ? " (tied to the embedding)" : ""}</div>
            {chips([arch.unembed])}
          </div>
          <Arrow />
          <div className="fbox c-out">
            <div className="fbox-title">Next-token probabilities</div>
            <div className="muted small">softmax over {t[arch.unembed].shape[0].toLocaleString()} tokens</div>
          </div>
        </>
      )}
      {extra.length > 0 && (
        <div className="flow-extra">
          <div className="muted small">Outside the stack</div>
          {extra.map((o: any) => (
            <div key={o.path} className="fbox c-other slim">
              <div className="small">
                <code>{o.path}</code> · {o.tensors.length} tensor{o.tensors.length === 1 ? "" : "s"}
              </div>
              {o.tensors.length <= 8 && chips(o.tensors)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Arrow() {
  return <div className="flow-arrow" aria-hidden />;
}

function StackDiagram({ arch, index, layer, onLayer }: { arch: Arch; index: Index; layer: number; onLayer: (l: number) => void }) {
  return (
    <div className="segments">
      {arch.stack.segments.map(([a, b, times]: number[]) => {
        const layers = Array.from({ length: b - a + 1 }, (_, k) => a + k);
        return (
          <div key={a} className={`segment ${times > 1 ? "looped" : ""}`}>
            {times > 1 && (
              <div className="loop-tag" title={`L${a}-${b} run ${times} times per forward pass, with the same weights`}>
                ↻ ×{times}
              </div>
            )}
            <div className="layer-chips">
              {layers.map((l) => (
                <button
                  key={l}
                  className={`lchip ${l === layer ? "active" : ""}`}
                  style={{ ["--v" as any]: VARIANT_COLORS[index.variantOf[l] % VARIANT_COLORS.length] }}
                  onClick={() => onLayer(l)}
                  title={`L${l} · type ${letter(index.variantOf[l])}`}
                >
                  L{l}
                </button>
              ))}
            </div>
          </div>
        );
      })}
      <div className="muted small">
        run order: {arch.stack.order.map((r: number[]) => (r[0] === r[1] ? `L${r[0]}` : `L${r[0]}-${r[1]}`)).join(" → ")}
      </div>
    </div>
  );
}

function Stack3D({ arch, index, layer, onLayer }: { arch: Arch; index: Index; layer: number; onLayer: (l: number) => void }) {
  const size = arch.stack.size;
  const gap = Math.max(5, Math.min(16, 320 / size));
  const looped = arch.stack.segments.filter((g: number[]) => g[2] > 1);
  return (
    <div className="stack3d" style={{ height: size * gap * 0.9 + 190 }}>
      <div className="muted small stack3d-note">
        L0 at the bottom · colour = layer type
        {looped.map((g: number[]) => ` · glowing: L${g[0]}-${g[1]} run ×${g[2]}`).join("")}
      </div>
      <div className="stack3d-scene">
        {Array.from({ length: size }, (_, l) => {
          const loop = looped.find((g: number[]) => l >= g[0] && l <= g[1]);
          return (
            <button
              key={l}
              className={`slab ${l === layer ? "active" : ""} ${loop ? "looped" : ""}`}
              style={{
                ["--v" as any]: VARIANT_COLORS[index.variantOf[l] % VARIANT_COLORS.length],
                transform: `translateZ(${l * gap}px)`,
              }}
              onClick={() => onLayer(l)}
              title={`L${l} · type ${letter(index.variantOf[l])}${loop ? ` · runs ×${loop[2]}` : ""}`}
            >
              <span>L{l}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- one layer

const FLAG_LABEL: Record<string, string> = {
  attends: "attention",
  routes: "mixture of experts",
  convolves: "convolution",
  gates: "gate",
  norms: "norm",
};

function blockColor(b: any): string {
  if (b.flags.includes("routes")) return "c-ffn";
  if (b.flags.includes("attends") && !b.flags.includes("norms")) return "c-attn";
  if (b.flags.includes("gates")) return "c-gate";
  if (b.flags.includes("convolves")) return "c-conv";
  if (b.flags.includes("norms")) return "c-norm";
  if (b.writes) return "c-ffn";
  return "c-other";
}

function LayerDetail(props: {
  arch: Arch;
  index: Index;
  layer: number;
  onLayer: (l: number) => void;
  tensor: string | null;
  onTensor: (p: string) => void;
}) {
  const { arch, index, layer, onLayer, tensor, onTensor } = props;
  const vi = index.variantOf[layer] ?? 0;
  const variant = arch.stack.variants[vi];
  const runs = arch.stack.runs[layer];
  const total = (variant.blocks ?? []).reduce((n: number, b: any) => n + b.numel, 0);
  return (
    <div className="layer-detail">
      <div className="layer-head">
        <div className="layer-title">
          <span className="swatch" style={{ background: VARIANT_COLORS[vi % VARIANT_COLORS.length] }}>
            {letter(vi)}
          </span>
          <div>
            <strong>L{layer}</strong> <span className="muted">· {variant.class}</span>
            <div className="muted small">
              {human(total)} params · {variant.tensors.length} tensors
              {runs > 1 && <> · runs {runs}× per forward pass (passes L{layer}@1…@{runs})</>}
            </div>
          </div>
        </div>
        <div className="layer-nav">
          <button disabled={layer === 0} onClick={() => onLayer(layer - 1)} aria-label="Previous layer">
            ‹
          </button>
          <button disabled={layer === arch.stack.size - 1} onClick={() => onLayer(layer + 1)} aria-label="Next layer">
            ›
          </button>
        </div>
      </div>
      <div className="rail-legend muted small">
        Sublayers in the order the model runs them, on the residual stream. → reads the stream · ⊕ adds back to it
      </div>
      <ol className="rail">
        {(variant.blocks ?? []).map((b: any) => (
          <li key={b.name} className="rail-row">
            <div className={`rail-mark ${b.writes ? "writes" : ""} ${b.reads ? "reads" : ""}`} aria-hidden>
              {b.writes ? "⊕" : b.reads ? "→" : ""}
            </div>
            <div className={`fbox ${blockColor(b)} block-card`}>
              <div className="block-head">
                <span className="fbox-title">{b.name}</span>
                <span className="muted small">
                  {b.flags.map((f: string) => FLAG_LABEL[f] ?? f).join(" · ")}
                  {b.flags.length > 0 && " · "}
                  {human(b.numel)}
                  {!b.ran && " · did not run when traced"}
                </span>
              </div>
              <div className="flow-chips">
                {b.tensors.map((inner: string) => {
                  const param = index.layerParams(layer, inner);
                  if (!param) return null;
                  const tt = arch.tensors[param];
                  return (
                    <button
                      key={inner}
                      className={`wchip k-${tt.kind} ${param === tensor ? "active" : ""}`}
                      onClick={() => onTensor(param)}
                      title={`${tt.key} · ${tt.kind}`}
                    >
                      {inner.startsWith(`${b.name}.`) ? inner.slice(b.name.length + 1) : inner || b.name}{" "}
                      <span className="muted">{tt.shape.join("×")}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </li>
        ))}
      </ol>
      {variant.layers.length > 1 && (
        <div className="muted small same-type">
          Same type:{" "}
          {variant.layers
            .filter((l: number) => l !== layer)
            .map((l: number) => (
              <button key={l} className="link" onClick={() => onLayer(l)}>
                L{l}
              </button>
            ))}
        </div>
      )}
    </div>
  );
}

// --------------------------------------------------------------------------- one weight

const KIND_TEXT: Record<string, string> = {
  writer: "output is model-width and reaches the stream",
  reader: "input is model-width",
  gate: "model-width output that only passes through sigmoid, tanh or the like",
  expert: "a stack of experts on dim 0",
  router: "picks the experts",
  embed: "token embedding table",
  unembed: "unembedding: width → vocabulary",
  final_norm: "the final norm before the unembedding",
  norm: "a norm's scale",
  vector: "a vector",
  bias: "a bias",
  matrix: "a matrix (neither side is model-width)",
};

function WeightPanel({ arch, session, param }: { arch: Arch; session: Session; param: string | null }) {
  const [shown, setShown] = useState<{ param: string; data?: any; error?: string; busy?: boolean } | null>(null);
  useEffect(() => setShown(null), [param]);
  if (!param) {
    return (
      <div className="weights">
        <p className="muted">Select a weight (a chip in the diagram, or Find) to see its shape, axes and values.</p>
      </div>
    );
  }
  const t = arch.tensors[param];
  const cold = session.tier !== "hot";
  const load = async () => {
    setShown({ param, busy: true });
    try {
      const r = await toolCall(session.id, "show", { address: t.key, grid: 48 });
      if (!r.ok) throw new Error(r.summary);
      setShown({ param, data: r.data.show?.[0] });
    } catch (e: any) {
      setShown({ param, error: String(e.message ?? e) });
    }
  };
  const axes =
    t.in != null && t.out != null
      ? `reads dim ${t.in} (${t.shape[t.in]}) → writes dim ${t.out} (${t.shape[t.out]})`
      : t.in != null
        ? `contracted on dim ${t.in}`
        : null;
  return (
    <div className="weights">
      <div className="weights-head">
        <code className="wkey">{t.key}</code>
        <div className="muted small">{param}</div>
      </div>
      <Block3D shape={t.shape} experts={t.experts} />
      <dl className="wfacts">
        <dt>kind</dt>
        <dd>
          {t.kind} <span className="muted small">{KIND_TEXT[t.kind] ?? ""}</span>
        </dd>
        <dt>shape</dt>
        <dd>
          {t.shape.join(" × ")} <span className="muted small">· {human(numel(t.shape))} · {t.dtype}</span>
        </dd>
        {axes && (
          <>
            <dt>axes</dt>
            <dd>{axes}</dd>
          </>
        )}
        {t.experts && (
          <>
            <dt>experts</dt>
            <dd>{t.experts} on dim 0</dd>
          </>
        )}
        {t.runs > 1 && (
          <>
            <dt>runs</dt>
            <dd>{t.runs}× per forward pass: an edit changes every pass</dd>
          </>
        )}
        <dt>traced</dt>
        <dd>{t.exercised ? (t.phases?.length ? t.phases.join(", ") : "yes") : <span className="badge danger">never read</span>}</dd>
      </dl>
      {shown?.param === param && shown.data ? (
        <WeightValues data={shown.data} />
      ) : (
        <button className="primary" onClick={load} disabled={shown?.busy}>
          {shown?.busy ? "Reading the weights…" : cold ? "Load values (starts the lab)" : "Load values"}
        </button>
      )}
      {shown?.error && <div className="error small">{shown.error}</div>}
    </div>
  );
}

function Block3D({ shape, experts }: { shape: number[]; experts?: number | null }) {
  // A box drawn to the tensor's proportions (log scale), as weight blocks in 3D LLM diagrams.
  const dims = shape.length >= 3 ? shape.slice(-2) : shape.length === 2 ? shape : [1, shape[0] ?? 1];
  const depthN = shape.length >= 3 ? shape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
  const s = (n: number) => 10 + 26 * Math.log10(Math.max(1, n));
  const w = s(dims[1]);
  const h = s(dims[0]);
  const d = depthN > 1 ? Math.min(60, 6 + 14 * Math.log10(depthN)) : 6;
  const dx = d * 0.8;
  const dy = d * 0.5;
  const W = w + dx + 70;
  const H = h + dy + 34;
  const x0 = 8;
  const y0 = dy + 8;
  return (
    <svg className="block3d" viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`a ${shape.join(" by ")} tensor`}>
      <polygon className="face top" points={`${x0},${y0} ${x0 + dx},${y0 - dy} ${x0 + w + dx},${y0 - dy} ${x0 + w},${y0}`} />
      <polygon className="face side" points={`${x0 + w},${y0} ${x0 + w + dx},${y0 - dy} ${x0 + w + dx},${y0 + h - dy} ${x0 + w},${y0 + h}`} />
      <rect className="face front" x={x0} y={y0} width={w} height={h} />
      <text x={x0 + w / 2} y={y0 + h + 14} textAnchor="middle">
        {dims[1].toLocaleString()}
      </text>
      <text x={x0 + w + dx + 6} y={y0 + h / 2} dominantBaseline="middle">
        {dims[0].toLocaleString()}
      </text>
      {depthN > 1 && (
        <text x={x0 + w + dx + 6} y={y0 - dy + 2} dominantBaseline="hanging">
          ×{depthN.toLocaleString()}
          {experts ? " experts" : ""}
        </text>
      )}
    </svg>
  );
}

function WeightValues({ data }: { data: any }) {
  const s = data.stats;
  const peak = Math.max(...(s.hist_counts ?? [1]), 1);
  const top = Math.max(...(s.spectrum ?? [1]), 1e-12);
  return (
    <div className="wvalues">
      {data.map && <Heatmap map={data.map} />}
      <div className="wstats">
        {[
          ["norm", s.norm],
          ["mean", s.mean],
          ["std", s.std],
          ["absmax", s.absmax],
          ["kurtosis", s.kurtosis],
          ["zeros", `${(s.zeros * 100).toFixed(1)}%`],
        ].map(([k, v]) => (
          <div key={k as string}>
            <span className="muted small">{k}</span>
            <strong>{typeof v === "number" ? fmt(v) : v}</strong>
          </div>
        ))}
      </div>
      {s.hist_counts?.length > 0 && (
        <figure className="mini-chart">
          <figcaption className="muted small">
            values from {fmt(s.hist_edges[0])} to {fmt(s.hist_edges[s.hist_edges.length - 1])}
          </figcaption>
          <div className="bars">
            {s.hist_counts.map((c: number, i: number) => (
              <span key={i} style={{ height: `${(100 * c) / peak}%` }} title={`${c.toLocaleString()} values`} />
            ))}
          </div>
        </figure>
      )}
      {s.spectrum?.length > 0 && (
        <figure className="mini-chart">
          <figcaption className="muted small">top singular values</figcaption>
          <div className="bars spectrum">
            {s.spectrum.map((v: number, i: number) => (
              <span key={i} style={{ height: `${(100 * v) / top}%` }} title={fmt(v)} />
            ))}
          </div>
        </figure>
      )}
      {data.vocab?.length > 0 && (
        <div className="small">
          <span className="muted">{data.vocab_label}</span>
          <div className="flow-chips">
            {data.vocab.map(([tok, score]: [string, number], i: number) => (
              <span key={i} className="wchip">
                {tok} <span className="muted">{score.toFixed(2)}</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Heatmap({ map }: { map: any }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    canvas.width = map.cols;
    canvas.height = map.rows;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(map.cols, map.rows);
    const max = map.max || 1;
    map.cells.forEach((row: number[], r: number) =>
      row.forEach((v: number, c: number) => {
        const [R, G, B] = viridis(Math.min(1, v / max));
        const i = (r * map.cols + c) * 4;
        img.data[i] = R;
        img.data[i + 1] = G;
        img.data[i + 2] = B;
        img.data[i + 3] = 255;
      }),
    );
    ctx.putImageData(img, 0, 0);
  }, [map]);
  const wide = map.cols >= map.rows;
  return (
    <figure className="heatmap">
      <canvas ref={ref} style={wide ? { width: "100%" } : { height: 220 }} />
      <figcaption className="muted small">
        |w| (root mean square) in {map.rows}×{map.cols} blocks of a {map.of[0].toLocaleString()}×{map.of[1].toLocaleString()} view ·
        brighter is larger, up to {fmt(map.max)}
      </figcaption>
    </figure>
  );
}

function viridis(t: number): [number, number, number] {
  const stops: [number, number, number][] = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ];
  const x = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  return a !== 0 && (a < 1e-3 || a >= 1e4) ? v.toExponential(2) : v.toFixed(a < 1 ? 4 : 2);
}
