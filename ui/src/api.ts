// Talks to the control plane with the user token. The token arrives in the URL
// fragment (never sent to the server in a request line) and is kept locally.

const KEY = "wspx.token";

function readFragment(): { token?: string; session?: string } {
  const params = new URLSearchParams(window.location.hash.slice(1));
  const out = { token: params.get("token") ?? undefined, session: params.get("session") ?? undefined };
  if (out.token) history.replaceState(null, "", window.location.pathname);
  return out;
}

const fragment = readFragment();
let token = fragment.token ?? "";
try {
  if (fragment.token) localStorage.setItem(KEY, fragment.token);
  else token = localStorage.getItem(KEY) ?? "";
} catch {
  // storage may be unavailable; the token from the fragment still works for this page
}

export const initialSession = fragment.session;
export const hasToken = () => token.length > 0;
export function setToken(value: string) {
  token = value;
  try {
    localStorage.setItem(KEY, value);
  } catch {
    /* ignore */
  }
}

export class ApiError extends Error {}

/** A file (an image or video a prompt named) as a blob, with the token in a header. */
export async function apiBlob(path: string): Promise<Blob> {
  const res = await fetch(path, { headers: { authorization: `Bearer ${token}` } });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(detail);
  }
  return res.blob();
}

export async function api<T = any>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { authorization: `Bearer ${token}`, "content-type": "application/json", ...(init.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? body.error ?? detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const post = <T = any>(path: string, body: unknown = {}) =>
  api<T>(path, { method: "POST", body: JSON.stringify(body) });

// Every kind the trace records; the stream names each event by its kind.
export const EVENT_KINDS = [
  "session", "start", "turn", "tool_call", "notes", "compaction", "lab", "loop", "command", "error", "hook", "end",
  "model_io",
];

/** "in 6,054 (6,000 cache read, 50 written) → out 90 tokens" */
export function formatTokens(u: any): string {
  const read = u?.cache_read ?? 0;
  const written = u?.cache_write ?? 0;
  let text = `in ${((u?.input ?? 0) + read + written).toLocaleString()}`;
  if (read || written) text += ` (${read.toLocaleString()} cache read, ${written.toLocaleString()} written)`;
  text += ` → out ${(u?.output ?? 0).toLocaleString()}`;
  if (u?.reasoning) text += ` (${u.reasoning.toLocaleString()} reasoning)`;
  return text;
}

export function stream(session: string, since: number, onEvent: (e: TimelineEvent) => void): () => void {
  const source = new EventSource(`/api/sessions/${session}/stream?since=${since}&token=${encodeURIComponent(token)}`);
  // EventSource fires its own `error` (no data) when the connection drops and while it
  // reconnects; only events the server sent carry data.
  const handler = (msg: MessageEvent) => {
    if (typeof msg.data === "string") onEvent(JSON.parse(msg.data));
  };
  for (const kind of EVENT_KINDS) {
    source.addEventListener(kind, handler as EventListener);
  }
  return () => source.close();
}

export interface Session {
  id: string;
  model: string;
  driver: string | null;
  task: string;
  gpu: string | null;
  status: string;
  tier: string;
  reason: string;
  fingerprint: string | null;
  load_seconds: number | null;
  created: number;
}

export interface TimelineEvent {
  id: number;
  kind: string;
  source: string;
  ts: number;
  payload: any;
}

export const toolCall = (session: string, name: string, args: Record<string, unknown>) =>
  post<{ summary: string; ok: boolean; handles: string[]; data: any }>(`/api/sessions/${session}/tools/${name}`, {
    args,
    source: "human",
  });
