/**
 * Shared HTTP layer for ALL backend calls — single source for the API base, the
 * request timeout and the low-level fetch. Everything goes through the vite proxy
 * at `/api/*` → :8010.
 *
 *  - `api.ts`         (dataset selector / open-consultation submit) builds on `rawFetch`;
 *  - `analysisApi.ts` (analysis / insights / citations / avis / flags) uses `rawFetch` directly.
 *
 * Keeping BASE + TIMEOUT_MS + rawFetch here removes the previous duplication
 * (the 180 s timeout used to be declared in both clients).
 */

/** Vite proxy prefix — every endpoint path is resolved against this base. */
export const BASE = '/api';

/** Généreux : certaines analyses/embeddings sont lents au 1ᵉʳ appel (chargement lazy). */
export const TIMEOUT_MS = 180000;

/** Raw fetch result: HTTP status + parsed JSON body (`null` if the body isn't JSON). */
export interface RawResult {
  status: number;
  body: any;
}

/**
 * Fetch `BASE + path`, returning the status code and parsed JSON body. Aborts
 * after `timeoutMs`. NEVER throws on a non-2xx response (callers inspect `status`);
 * a network failure / abort still rejects, as with `fetch`.
 */
export async function rawFetch(
  path: string,
  init?: RequestInit,
  timeoutMs = TIMEOUT_MS,
): Promise<RawResult> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(BASE + path, { ...init, signal: ctrl.signal });
    const body = await r.json().catch(() => null);
    return { status: r.status, body };
  } finally {
    clearTimeout(t);
  }
}
