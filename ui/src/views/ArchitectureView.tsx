import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { api, toolCall, type Session } from "../api";

// The compiled map as a block diagram, drawn like the classic transformer figure: boxes joined
// by arrows from the input tokens to the output probabilities, the layer stack in a frame
// marked N×, each sublayer on the residual stream with a skip arrow into ⊕ where its output is
// added back, loops marked where layers run more than once, and a vision tower feeding in from
// the side. The adds, the order and the loops are read off the traced op graph, not assumed.
// Click a box to open it into its weights; click a weight to see its shape, axes and kind, and
// load its values from the lab.

type Arch = any;

const VARIANT_COLORS = ["#6366f1", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#a855f7", "#14b8a6", "#f43f5e"];

export function ArchitectureView({ session }: { session: Session }) {
  const [arch, setArch] = useState<Arch>(null);
  const [layer, setLayer] = useState(0);
  const [tensor, setTensor] = useState<string | null>(null);
  const [open, setOpen] = useState<Set<string>>(() => new Set());

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
    const box = index.boxOf(param);
    if (box) setOpen((o) => new Set(o).add(box));
  };
  const toggle = (id: string) =>
    setOpen((o) => {
      const n = new Set(o);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  const vi = index.variantOf[layer] ?? 0;
  const layerBoxes = arch.stack ? index.blocksOf(vi).map((b: any) => `L${vi}:${b.name}`) : [];
  const extra = arch.others.filter((o: any) => !o.vision_only);

  return (
    <div className="archx">
      <Summary arch={arch} onFind={pick} index={index} />
      <div className="archx-grid">
        <div className="archx-main">
          {arch.stack && <LayerDock arch={arch} index={index} layer={layer} onLayer={setLayer} />}
          <section className="archx-diagram">
            <div className="archx-tools">
              <span className="muted small">Click a box to open its weights{arch.stack ? " · pick a layer from the tabs on the left" : ""}</span>
              <span className="spacer" />
              {layerBoxes.length > 0 && (
                <button onClick={() => setOpen((o) => new Set([...o, ...layerBoxes]))}>Open this layer</button>
              )}
              <button onClick={() => setOpen(new Set())} disabled={open.size === 0}>
                Close all
              </button>
            </div>
            <Diagram arch={arch} index={index} layer={layer} open={open} onToggle={toggle} tensor={tensor} onTensor={pick} />
            {extra.length > 0 && (
              <div className="archx-extra">
                <div className="muted small">Outside the stack, not on the path above</div>
                {extra.map((o: any) => (
                  <div key={o.path} className="small">
                    <code>{o.path}</code> <span className="muted">· {o.tensors.length} weight{o.tensors.length === 1 ? "" : "s"} · {human(o.numel)}</span>
                    <div className="archx-chips">
                      {o.tensors.slice(0, 12).map((n: string) => (
                        <button key={n} className={`wchip ${n === tensor ? "active" : ""}`} onClick={() => pick(n)} title={n}>
                          {arch.tensors[n].key} <span className="muted">{arch.tensors[n].shape.join("×")}</span>
                        </button>
                      ))}
                      {o.tensors.length > 12 && <span className="muted small">+{o.tensors.length - 12} more · use Find</span>}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
        <aside className="archx-weights">
          <WeightPanel arch={arch} session={session} param={tensor} />
        </aside>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- data

interface Index {
  byKey: Map<string, string>;
  variantOf: number[];
  layerParam: (i: number, inner: string) => string | undefined;
  blocksOf: (vi: number) => any[];
  boxOf: (param: string) => string | null;
}

function buildIndex(arch: Arch): Index {
  const byKey = new Map<string, string>();
  for (const [param, t] of Object.entries<any>(arch.tensors)) byKey.set(t.key, param);
  const variantOf: number[] = [];
  const variants = arch.stack?.variants ?? [];
  variants.forEach((v: any, vi: number) => v.layers.forEach((l: number) => (variantOf[l] = vi)));
  const layerParam = (i: number, inner: string) => byKey.get(inner ? `L${i}.${inner}` : `L${i}`);
  const blocks = variants.map((v: any) => v.blocks ?? groupTensors(v, (inner) => layerParam(v.layers[0], inner), arch));
  const blocksOf = (vi: number) => blocks[vi] ?? [];
  const boxOf = (param: string): string | null => {
    const t = arch.tensors[param];
    if (!t) return null;
    if (t.layer != null && arch.stack) {
      const vi = variantOf[t.layer];
      const head = `L${t.layer}`;
      const inner = t.key === head ? "" : t.key.startsWith(`${head}.`) ? t.key.slice(head.length + 1) : null;
      const b = blocksOf(vi).find((b: any) => b.tensors.includes(inner));
      return b ? `L${vi}:${b.name}` : null;
    }
    if (param === arch.embed || arch.stem.includes(param)) return "embed";
    if (arch.pos_embed.includes(param)) return "pos";
    if (arch.final_norm_params.includes(param)) return "final_norm";
    if (param === arch.unembed) return "unembed";
    if (arch.final.includes(param)) return "after";
    const o = arch.others.find((o: any) => o.tensors.includes(param));
    if (o) return `o:${o.path}`;
    return "body";
  };
  return { byKey, variantOf, layerParam, blocksOf, boxOf };
}

/** Sublayers for a map compiled before they were recorded: grouped by top-level module. */
function groupTensors(variant: any, param: (inner: string) => string | undefined, arch: Arch): any[] {
  const out = new Map<string, any>();
  for (const inner of variant.tensors as string[]) {
    const name = inner.split(".")[0] || "(own)";
    const b = out.get(name) ?? { name, tensors: [], flags: [], writes: false, reads: false, numel: 0, ran: true };
    b.tensors.push(inner);
    const p = param(inner);
    if (p) b.numel += numel(arch.tensors[p].shape);
    out.set(name, b);
  }
  return [...out.values()];
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

// --------------------------------------------------------------------------- layer tabs

/** Every layer as a tab, coloured by its type, docked beside the diagram and centred in view;
 * layers that run more than once are bracketed together. */
function LayerDock({ arch, index, layer, onLayer }: { arch: Arch; index: Index; layer: number; onLayer: (l: number) => void }) {
  const stack = arch.stack;
  const size: number = stack.size;
  const tabs = useRef<(HTMLButtonElement | null)[]>([]);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    // Keep the open tab in view inside the dock, without scrolling the page.
    const el = tabs.current[layer];
    const b = box.current;
    if (!el || !b) return;
    if (el.offsetTop < b.scrollTop) b.scrollTop = el.offsetTop - 8;
    else if (el.offsetTop + el.offsetHeight > b.scrollTop + b.clientHeight) b.scrollTop = el.offsetTop + el.offsetHeight - b.clientHeight + 8;
  }, [layer]);
  const go = (l: number) => {
    const next = Math.max(0, Math.min(size - 1, l));
    onLayer(next);
    tabs.current[next]?.focus();
  };
  const keys = (e: KeyboardEvent) => {
    const to: Record<string, number> = { ArrowUp: layer - 1, ArrowDown: layer + 1, PageUp: layer - 5, PageDown: layer + 5, Home: 0, End: size - 1 };
    if (e.key in to) {
      e.preventDefault();
      go(to[e.key]);
    }
  };
  return (
    <nav className="archx-dock" aria-label="Layers">
      <div className="dock-box" ref={box}>
        <div className="dock-list" role="tablist" aria-orientation="vertical" onKeyDown={keys}>
          <div className="dock-head">{size} layers</div>
          {stack.segments.map(([a, b, times]: number[]) => (
            <div
              key={a}
              className={`dock-seg${times > 1 ? " looped" : ""}`}
              title={times > 1 ? `L${a}-L${b} run ${times} times per forward pass, with the same weights` : undefined}
            >
              {times > 1 && <div className="dock-loop">↻ ×{times}</div>}
              {Array.from({ length: b - a + 1 }, (_, k) => a + k).map((l) => {
                const v = index.variantOf[l] ?? 0;
                return (
                  <button
                    key={l}
                    ref={(el) => {
                      tabs.current[l] = el;
                    }}
                    role="tab"
                    aria-selected={l === layer}
                    tabIndex={l === layer ? 0 : -1}
                    className={`dock-tab${l === layer ? " on" : ""}`}
                    style={{ ["--v" as any]: VARIANT_COLORS[v % VARIANT_COLORS.length] }}
                    onClick={() => onLayer(l)}
                    title={`L${l} · ${stack.variants[v].class}${stack.variants.length > 1 ? ` (type ${letter(v)})` : ""}${times > 1 ? ` · runs ×${times}` : ""}`}
                  >
                    L{l}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </nav>
  );
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

// --------------------------------------------------------------------------- the diagram

const BW = 300; // main column box width
const LANE = 30; // skip-arrow lane, left of the column
const GAP = 22; // arrow between boxes
const HEAD = 44; // a closed box
const CHIP_H = 20;
const CW = 6.6; // a monospace character at 11px
const SIDE_W = 200; // the vision column
const CAP = 40; // weights shown in one box before "use Find"

const FLAG_LABEL: Record<string, string> = {
  attends: "attention",
  routes: "mixture of experts",
  convolves: "convolution",
  gates: "gate",
  norms: "norm",
};

function blockTone(b: any): string {
  if (b.flags.includes("routes")) return "ffn";
  if (b.flags.includes("attends") && !b.flags.includes("norms")) return "attn";
  if (b.flags.includes("gates")) return "gate";
  if (b.flags.includes("convolves")) return "conv";
  if (b.flags.includes("norms")) return "norm";
  if (b.writes) return "ffn";
  return "other";
}

function blockSub(b: any): string {
  const words = b.flags.map((f: string) => FLAG_LABEL[f] ?? f);
  if (!words.length && b.writes) words.push("writes the stream");
  return [...words, human(b.numel), b.ran ? null : "did not run when traced"].filter(Boolean).join(" · ");
}

function fit(text: string, chars: number, fromLeft = false): string {
  if (text.length <= chars) return text;
  return fromLeft ? `…${text.slice(text.length - chars + 1)}` : `${text.slice(0, chars - 1)}…`;
}

function activate(fn: () => void) {
  return (e: KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fn();
    }
  };
}

interface Item {
  param: string;
  label: string;
}

interface BoxSpec {
  id: string;
  x: number;
  y: number;
  w: number;
  title: string;
  sub: string;
  tone: string;
  items: Item[];
  more?: number;
  dashed?: boolean;
}

function chipRows(items: Item[], arch: Arch, width: number) {
  const chips: (Item & { x: number; y: number; w: number; shape: string })[] = [];
  let x = 0;
  let y = 0;
  for (const it of items) {
    const shape = arch.tensors[it.param].shape.join("×");
    const room = Math.floor((width - 16) / CW) - shape.length - 1;
    const label = fit(it.label, Math.max(4, room), true);
    const w = Math.min(width, 14 + (label.length + 1 + shape.length) * CW);
    if (x > 0 && x + w > width) {
      x = 0;
      y += CHIP_H + 5;
    }
    chips.push({ ...it, label, shape, x, y, w });
    x += w + 5;
  }
  return { chips, h: items.length ? y + CHIP_H : 0 };
}

/** At most CAP items, keeping the selected one in view. */
function capped(items: Item[], selected: string | null): { items: Item[]; more: number } {
  if (items.length <= CAP) return { items, more: 0 };
  const head = items.slice(0, CAP);
  const sel = items.find((i) => i.param === selected);
  if (sel && !head.includes(sel)) head[CAP - 1] = sel;
  return { items: head, more: items.length - CAP };
}

function Diagram(props: {
  arch: Arch;
  index: Index;
  layer: number;
  open: Set<string>;
  onToggle: (id: string) => void;
  tensor: string | null;
  onTensor: (p: string) => void;
}) {
  const { arch, index, layer, open, onToggle, tensor, onTensor } = props;
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    ref.current?.querySelector(".bd-chip.on")?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [tensor]);

  const T = arch.tensors;
  const stack = arch.stack;
  const vision = arch.others
    .filter((o: any) => o.vision_only)
    .sort((a: any, b: any) => Number(!!b.stack) - Number(!!a.stack) || b.numel - a.numel);
  const hidden = arch.hidden;

  // Columns: [vision | N×] [frame: lane, boxes] [position embedding]
  const leftZone = Math.max(vision.length ? SIDE_W + 40 : 12, stack ? 76 : 12);
  const frameX = leftZone;
  const laneX = frameX + 18;
  const colX = laneX + LANE;
  const cx = colX + BW / 2;
  const frameR = colX + BW + 18;
  const posW = 160;
  const posX = colX + BW + 36;

  const frames: ReactNode[] = [];
  const wires: ReactNode[] = [];
  const nodes: ReactNode[] = [];
  let n = 0;
  const key = () => `k${n++}`;

  const wire = (x1: number, y1: number, x2: number, y2: number, arrow = false) =>
    wires.push(<line key={key()} className="bd-wire" x1={x1} y1={y1} x2={x2} y2={y2} markerEnd={arrow ? "url(#bd-arrow)" : undefined} />);
  const route = (d: string, cls: string, arrow: boolean) =>
    wires.push(<path key={key()} className={cls} d={d} markerEnd={arrow ? `url(#${cls === "bd-skip" ? "bd-arrow-skip" : "bd-arrow"})` : undefined} />);
  const text = (x: number, y: number, s: string, cls = "bd-note", anchor: "start" | "middle" | "end" = "start") =>
    nodes.push(
      <text key={key()} className={cls} x={x} y={y} textAnchor={anchor}>
        {s}
      </text>,
    );
  const pill = (x: number, y: number, label: string) => {
    const w = Math.max(96, label.length * 6.6 + 28);
    nodes.push(
      <g key={key()} className="bd-pill">
        <rect x={x - w / 2} y={y} width={w} height={24} rx={12} />
        <text x={x} y={y + 16} textAnchor="middle">
          {label}
        </text>
      </g>,
    );
    return 24;
  };
  const plus = (x: number, y: number, what: string) =>
    nodes.push(
      <g key={key()} className="bd-add" transform={`translate(${x} ${y})`}>
        <circle r={10} />
        <path d="M-5 0H5M0-5V5" />
        <title>{what}</title>
      </g>,
    );

  const box = (s: BoxSpec): number => {
    const can = s.items.length > 0;
    const isOpen = can && open.has(s.id);
    const { chips, h: ch } = isOpen ? chipRows(s.items, arch, s.w - 20) : { chips: [], h: 0 };
    const moreH = isOpen && s.more ? 18 : 0;
    const h = HEAD + (isOpen ? ch + 10 + moreH : 0);
    const sel = tensor != null && s.items.some((i) => i.param === tensor);
    const count = s.items.length + (s.more ?? 0);
    const caret = can ? `${count} weight${count === 1 ? "" : "s"} ${isOpen ? "▾" : "▸"}` : "";
    const room = (s.w - 26 - caret.length * 6.2) / 7.4;
    const toggle = () => can && onToggle(s.id);
    nodes.push(
      <g key={s.id} className={`bd-box t-${s.tone}${s.dashed ? " dashed" : ""}${sel ? " sel" : ""}`}>
        <rect className="bd-rect" x={s.x} y={s.y} width={s.w} height={h} rx={8} />
        <g
          className={`bd-head${can ? "" : " fixed"}`}
          role={can ? "button" : undefined}
          tabIndex={can ? 0 : undefined}
          aria-expanded={can ? isOpen : undefined}
          aria-label={`${s.title}: ${s.sub}`}
          onClick={toggle}
          onKeyDown={activate(toggle)}
        >
          <rect className="bd-hit" x={s.x} y={s.y} width={s.w} height={HEAD} rx={8} />
          <text className="bd-title" x={s.x + 12} y={s.y + 19}>
            {fit(s.title, Math.floor(room))}
          </text>
          <text className="bd-sub" x={s.x + 12} y={s.y + 35}>
            {fit(s.sub, Math.floor((s.w - 24) / 6))}
          </text>
          {can && (
            <text className="bd-caret" x={s.x + s.w - 12} y={s.y + 19} textAnchor="end">
              {caret}
            </text>
          )}
          <title>{`${s.title} · ${s.sub}`}</title>
        </g>
        {chips.map((c) => {
          const t = T[c.param];
          const choose = () => onTensor(c.param);
          return (
            <g
              key={c.param}
              className={`bd-chip k-${t.kind}${c.param === tensor ? " on" : ""}${t.exercised ? "" : " cold"}`}
              role="button"
              tabIndex={0}
              aria-pressed={c.param === tensor}
              aria-label={`${t.key}, ${t.shape.join(" by ")}`}
              transform={`translate(${s.x + 10 + c.x} ${s.y + HEAD + c.y})`}
              onClick={choose}
              onKeyDown={activate(choose)}
            >
              <rect className="bd-chip-bg" width={c.w} height={CHIP_H} rx={4} />
              <rect className="bd-kind" width={3} height={CHIP_H} rx={1.5} />
              <text x={8} y={14}>
                {c.label} <tspan className="bd-shape">{c.shape}</tspan>
              </text>
              <title>{`${t.key} · ${t.kind} · ${t.shape.join("×")}${t.exercised ? "" : " · never read when traced"}`}</title>
            </g>
          );
        })}
        {moreH > 0 && (
          <text className="bd-note" x={s.x + 12} y={s.y + HEAD + ch + 20}>
            +{s.more} more · use Find
          </text>
        )}
      </g>,
    );
    return h;
  };
  const items = (params: string[], label: (p: string) => string) => capped(params.map((param) => ({ param, label: label(param) })), tensor);

  // ---- inputs
  let y = 10;
  y += pill(cx, y, "input tokens");
  let vy = 10;
  const vx = 8 + SIDE_W / 2;
  if (vision.length) {
    vy += pill(vx, vy, "images · video");
    for (const o of vision) {
      wire(vx, vy, vx, vy + GAP, true);
      vy += GAP;
      const it = items(o.tensors, (p) => p.slice(o.path.length + 1).replace(/\.weight$/, ""));
      vy += box({
        id: `o:${o.path}`,
        x: 8,
        y: vy,
        w: SIDE_W,
        title: o.path.split(".").pop(),
        sub: [o.stack ? `${o.stack.size} layers` : null, human(o.numel), "ran on images"].filter(Boolean).join(" · "),
        tone: "vision",
        ...it,
      });
    }
  }
  const embedParams = [arch.embed, ...arch.stem].filter(Boolean);
  if (embedParams.length) {
    wire(cx, y, cx, y + GAP, true);
    y += GAP;
    const shape = arch.embed ? T[arch.embed].shape : null;
    y += box({
      id: "embed",
      x: colX,
      y,
      w: BW,
      title: "Embedding",
      sub: shape?.length === 2 ? `${shape[0].toLocaleString()} tokens × width ${shape[1].toLocaleString()}` : `${embedParams.length} weights`,
      tone: "embed",
      ...items(embedParams, (p) => T[p].key),
    });
  }
  if (arch.pos_embed.length) {
    const ay = y + GAP + 10;
    wire(cx, y, cx, ay - 10, true);
    plus(cx, ay, "add: token embedding plus position embedding");
    const ph = box({
      id: "pos",
      x: posX,
      y: ay - HEAD / 2,
      w: posW,
      title: "Position embedding",
      sub: T[arch.pos_embed[0]].shape.join(" × "),
      tone: "embed",
      ...items(arch.pos_embed, (p) => T[p].key),
    });
    wire(posX, ay, cx + 10, ay, true);
    const below = Math.max(ay + 10, ay - HEAD / 2 + ph + 6);
    wire(cx, ay + 10, cx, below);
    y = below;
  }
  if (vision.length) {
    const jy = Math.max(y + GAP + 8, vy + 26);
    wire(cx, y, cx, jy - 8, true);
    nodes.push(
      <g key={key()} className="bd-join" transform={`translate(${cx} ${jy}) rotate(45)`}>
        <rect x={-6} y={-6} width={12} height={12} />
        <title>image tokens join the sequence</title>
      </g>,
    );
    route(`M ${vx} ${vy} V ${jy} H ${cx - 10}`, "bd-wire", true);
    text(cx - 16, jy - 6, "image tokens", "bd-note", "end");
    y = jy + 8;
  }

  // ---- the layer stack
  if (stack) {
    const vi = index.variantOf[layer] ?? 0;
    const variant = stack.variants[vi];
    const blocks = index.blocksOf(vi);
    const runs = stack.runs[layer] ?? 1;
    const total = blocks.reduce((s: number, b: any) => s + b.numel, 0);
    const top = y + GAP;
    const by = top + 54;
    if (hidden) text(cx + 8, y + GAP / 2 + 8, `width ${hidden.toLocaleString()}`);
    wire(cx, y, cx, by);

    const known = blocks.some((b: any) => "adds" in b);
    const adds = (b: any) => (known ? b.adds : b.writes);
    const groups: any[][] = [];
    let cur: any[] = [];
    for (const b of blocks) {
      cur.push(b);
      if (adds(b)) {
        groups.push(cur);
        cur = [];
      }
    }
    const spec = (b: any, at: number, dashed = false): BoxSpec => {
      const params = b.tensors
        .map((inner: string) => ({ inner, param: index.layerParam(layer, inner) }))
        .filter((p: any) => p.param);
      const labelOf = new Map(params.map((p: any) => [p.param, p.inner]));
      const label = (p: string) => {
        const inner = labelOf.get(p) as string;
        if (!inner || inner === b.name) return "weight";
        return inner.startsWith(`${b.name}.`) ? inner.slice(b.name.length + 1) : inner;
      };
      return {
        id: `L${vi}:${b.name}`,
        x: colX,
        y: at,
        w: BW,
        title: b.name,
        sub: blockSub(b),
        tone: blockTone(b),
        dashed: dashed || !b.ran,
        ...items(
          params.map((p: any) => p.param),
          label,
        ),
      };
    };

    let iy = by;
    for (const g of groups) {
      const branch = iy + 6;
      wire(cx, iy, cx, branch);
      nodes.push(<circle key={key()} className="bd-dot" cx={cx} cy={branch} r={3} />);
      let gy = branch;
      for (const b of g) {
        wire(cx, gy, cx, gy + GAP, true);
        gy += GAP;
        gy += box(spec(b, gy));
      }
      const ay = gy + GAP;
      wire(cx, gy, cx, ay - 10, true);
      plus(cx, ay, `add: the residual stream plus ${g[g.length - 1].name}'s output (traced)`);
      route(`M ${cx} ${branch} H ${laneX} V ${ay} H ${cx - 10}`, "bd-skip", true);
      iy = ay + 10;
    }
    if (cur.length && groups.length) {
      // Sublayers after the last add: their output is not added to the stream, which goes past.
      const ty = iy + 30;
      text(colX, ty - 8, "also in this layer · output not added to the stream");
      let gy = ty;
      for (const b of cur) gy += box(spec(b, gy, true)) + 8;
      route(`M ${cx} ${iy} V ${iy + 10} H ${laneX} V ${gy + 4} H ${cx} V ${gy + 12}`, "bd-wire", false);
      iy = gy + 12;
    } else {
      for (const b of cur) {
        wire(cx, iy, cx, iy + GAP, true);
        iy += GAP;
        iy += box(spec(b, iy));
      }
    }

    const size = stack.size;
    const bottom = iy + 16;
    wire(cx, iy, cx, bottom);
    frames.push(<rect key="frame" className="bd-frame" x={frameX} y={top} width={frameR - frameX} height={bottom - top} rx={14} />);
    const headRoom = Math.floor((cx - 14 - frameX - 14) / 6.2);
    text(frameX + 14, top + 23, `L${layer}`, "bd-ltitle");
    text(frameX + 14 + `L${layer}`.length * 9.5 + 6, top + 23, `of ${size}`);
    text(frameX + 14, top + 40, fit(variant.class, headRoom));
    text(cx + 12, top + 23, `${human(total)} params`);
    text(
      cx + 12,
      top + 40,
      runs > 1 ? `runs ×${runs} per pass` : stack.variants.length > 1 ? `type ${letter(vi)} of ${stack.variants.length}` : `${blocks.length} sublayers`,
    );
    const mid = (top + bottom) / 2;
    text(frameX - 12, mid + 2, `${size}×`, "bd-n", "end");
    text(frameX - 12, mid + 20, stack.variants.length > 1 ? `${stack.variants.length} types` : "layers", "bd-note", "end");
    y = bottom;
  } else {
    const ends = new Set([...embedParams, ...arch.pos_embed, ...arch.final, ...arch.final_norm_params, arch.unembed]);
    const inOthers = new Set(arch.others.flatMap((o: any) => o.tensors));
    const body = Object.keys(T).filter((p) => !ends.has(p) && !inOthers.has(p));
    if (body.length) {
      wire(cx, y, cx, y + GAP, true);
      y += GAP;
      y += box({ id: "body", x: colX, y, w: BW, title: "Model", sub: `no layer stack found · ${body.length} weights`, tone: "other", ...items(body, (p) => T[p].key) });
    }
  }

  // ---- outputs
  const after = arch.final.filter((p: string) => !arch.final_norm_params.includes(p));
  const next = (spec: Omit<BoxSpec, "x" | "y" | "w">) => {
    wire(cx, y, cx, y + GAP, true);
    y += GAP;
    y += box({ ...spec, x: colX, y, w: BW });
  };
  if (arch.final_norm_params.length)
    next({ id: "final_norm", title: "Final norm", sub: arch.final_norm ? `norm · ${arch.final_norm}` : "norm", tone: "norm", ...items(arch.final_norm_params, (p) => T[p].key) });
  if (after.length) next({ id: "after", title: "After the stack", sub: `${after.length} weight${after.length === 1 ? "" : "s"}`, tone: "other", ...items(after, (p) => T[p].key) });
  if (arch.unembed) {
    const [V, d] = T[arch.unembed].shape;
    next({
      id: "unembed",
      title: "Unembedding",
      sub: `linear · width ${Number(d).toLocaleString()} → ${Number(V).toLocaleString()} tokens${arch.tied ? " · tied to the embedding" : ""}`,
      tone: "linear",
      ...items([arch.unembed], (p) => T[p].key),
    });
    next({ id: "softmax", title: "Softmax", sub: "over the vocabulary", tone: "out", items: [] });
    wire(cx, y, cx, y + GAP, true);
    y += GAP;
    y += pill(cx, y, "next-token probabilities");
  } else {
    wire(cx, y, cx, y + GAP, true);
    y += GAP;
    y += pill(cx, y, "output");
  }

  const W = Math.max(frameR + 12, arch.pos_embed.length ? posX + posW + 12 : 0, colX + BW + 24);
  const H = Math.max(y, vy) + 12;
  return (
    <div className="archx-canvas">
      <svg
        ref={ref}
        className="bd"
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        style={{ minWidth: Math.min(W, 440) }}
        role="group"
        aria-label={`Block diagram of ${arch.arch}`}
      >
        <defs>
          <marker id="bd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path className="bd-mark-wire" d="M0,0 L10,5 L0,10 z" />
          </marker>
          <marker id="bd-arrow-skip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path className="bd-mark-skip" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>
        {frames}
        {wires}
        {nodes}
      </svg>
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

const LEGEND: [string, string][] = [
  ["attn", "attention"],
  ["ffn", "writes the stream: feed-forward, experts"],
  ["norm", "norm"],
  ["conv", "convolution"],
  ["gate", "gate"],
  ["embed", "embedding"],
  ["linear", "unembedding"],
  ["vision", "ran only on images"],
];

function WeightPanel({ arch, session, param }: { arch: Arch; session: Session; param: string | null }) {
  const [shown, setShown] = useState<{ param: string; data?: any; error?: string; busy?: boolean } | null>(null);
  useEffect(() => setShown(null), [param]);
  if (!param) {
    return (
      <div className="weights">
        <p className="muted small">Open a box, then pick a weight to see its shape, axes and kind, and load its values.</p>
        <ul className="legend">
          {LEGEND.map(([tone, label]) => (
            <li key={tone}>
              <span className={`lg t-${tone}`} />
              {label}
            </li>
          ))}
          <li>
            <span className="lg-add">+</span>added back to the residual stream, as the traced ops show
          </li>
          <li>
            <span className="lg dashed" />
            did not run when traced, or not added back
          </li>
          <li>
            <span className="lg-loop">↻3</span>these layers run 3 times per pass, same weights
          </li>
        </ul>
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
          <div className="archx-chips">
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
