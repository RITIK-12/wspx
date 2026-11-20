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

export function stream(session: string, since: number, onEvent: (e: TimelineEvent) => void): () => void {
  const source = new EventSource(`/api/sessions/${session}/stream?since=${since}&token=${encodeURIComponent(token)}`);
  const handler = (msg: MessageEvent) => onEvent(JSON.parse(msg.data));
  for (const kind of ["turn", "tool_call", "lab", "loop", "command", "session", "error", "compaction", "hook", "end"]) {
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
