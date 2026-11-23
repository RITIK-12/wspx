import { useEffect, useRef, useState, type RefObject } from "react";
import { apiBlob } from "./api";

// Images and videos a prompt named, shown from the session's project files. Each is fetched
// only once it is on screen (videos only when played), with the token in a header, and its
// blob is freed when it leaves the page, so a long session stays light.

interface Part {
  type: "image" | "video";
  ref: string | string[];
  frames?: number;
}

export function mediaParts(messages: any[] | undefined, roles = ["user"]): Part[] {
  const out: Part[] = [];
  for (const m of messages ?? []) {
    if (!roles.includes(m.role) || !Array.isArray(m.content)) continue;
    for (const part of m.content) {
      if (part.type === "image" || part.type === "video") {
        out.push({ type: part.type, ref: part[part.type], frames: part.frames });
      }
    }
  }
  return out;
}

export function MediaStrip({ session, parts }: { session: string; parts: Part[] }) {
  if (!parts.length) return null;
  return (
    <div className="media-strip">
      {parts.map((p, i) =>
        p.ref === "(inline)" ? (
          <span key={i} className="media-missing">
            {p.type} passed inline (not kept)
          </span>
        ) : Array.isArray(p.ref) ? (
          <Frames key={i} session={session} frames={p.ref} total={p.frames ?? p.ref.length} />
        ) : p.type === "video" ? (
          <Video key={i} session={session} path={p.ref} />
        ) : (
          <Thumb key={i} session={session} path={p.ref} />
        ),
      )}
    </div>
  );
}

function url(session: string, path: string): string {
  return `/api/sessions/${session}/media?path=${encodeURIComponent(path)}`;
}

function useBlob(session: string, path: string, wanted: boolean): { src?: string; error?: string } {
  const [state, setState] = useState<{ src?: string; error?: string }>({});
  useEffect(() => {
    if (!wanted) return;
    let src: string | undefined;
    let gone = false;
    apiBlob(url(session, path))
      .then((blob) => {
        if (gone) return;
        src = URL.createObjectURL(blob);
        setState({ src });
      })
      .catch((e) => !gone && setState({ error: String(e.message ?? e) }));
    return () => {
      gone = true;
      if (src) URL.revokeObjectURL(src);
    };
  }, [session, path, wanted]);
  return state;
}

function useOnScreen<T extends Element>(): [RefObject<T | null>, boolean] {
  const ref = useRef<T | null>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || seen) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setSeen(true);
          io.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [seen]);
  return [ref, seen];
}

function Thumb({ session, path }: { session: string; path: string }) {
  const [ref, seen] = useOnScreen<HTMLButtonElement>();
  const { src, error } = useBlob(session, path, seen);
  const [big, setBig] = useState(false);
  return (
    <>
      <button ref={ref} className="thumb" title={path} onClick={() => src && setBig(true)} disabled={!src}>
        {src ? <img src={src} alt={path} /> : <span className="thumb-note">{error ? "unavailable" : "…"}</span>}
        <span className="thumb-name">{error ? `${base(path)}: ${error}` : base(path)}</span>
      </button>
      {big && src && <Lightbox src={src} label={path} onClose={() => setBig(false)} />}
    </>
  );
}

function Frames({ session, frames, total }: { session: string; frames: string[]; total: number }) {
  const shown = frames.slice(0, 4);
  return (
    <div className="frames">
      {shown.map((f) => (
        <Thumb key={f} session={session} path={f} />
      ))}
      <span className="thumb-name">
        video · {total} frame{total === 1 ? "" : "s"}
        {total > shown.length ? ` (first ${shown.length} shown)` : ""}
      </span>
    </div>
  );
}

function Video({ session, path }: { session: string; path: string }) {
  const [play, setPlay] = useState(false);
  const { src, error } = useBlob(session, path, play);
  if (!play) {
    return (
      <button className="thumb video" onClick={() => setPlay(true)} title={`Load and play ${path}`}>
        <span className="thumb-note">▶</span>
        <span className="thumb-name">{base(path)}</span>
      </button>
    );
  }
  if (error) return <span className="media-missing">{base(path)}: {error}</span>;
  return src ? <video className="clip" src={src} controls autoPlay /> : <span className="thumb-note">loading…</span>;
}

function Lightbox({ src, label, onClose }: { src: string; label: string; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="lightbox" role="dialog" aria-label={label} onClick={onClose}>
      <img src={src} alt={label} />
      <div className="lightbox-label">{label} · click or Esc to close</div>
    </div>
  );
}

function base(path: string): string {
  return path.split("/").pop() ?? path;
}
