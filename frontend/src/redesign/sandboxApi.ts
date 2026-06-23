/**
 * Client for the analyst RECLUSTER sandbox (`/sandbox`, `/explain`) behind the
 * vite `/api/*` proxy → :8010. Drives the "console de mixage": every fader move
 * POSTs the fader set and gets back a fresh recluster + decision trace.
 *
 * Lane B's real endpoint arrives in parallel, so this client tries the REAL
 * backend first and falls back to the seeded MOCK on any failure (unreachable,
 * not-ready, bad shape) — OR forces the mock under `VITE_FORCE_MOCK=1` for fully
 * offline dev. Each call reports whether the answer was `live` or `mock` so the
 * console can badge it. The `/explain` calls are resolved locally against the last
 * `/sandbox` response when the backend can't answer (the mock has no server).
 */
import type {
  ExplainCluster,
  ExplainPair,
  SandboxParams,
  SandboxResponse,
} from './sandboxContract';
import { mockExplainCluster, mockExplainPair, mockSandbox } from './sandboxMock';

const FORCE_MOCK = import.meta.env.VITE_FORCE_MOCK === '1';
const TIMEOUT_MS = 15000;

export type SandboxSource = 'live' | 'mock';

export interface SandboxResult {
  data: SandboxResponse;
  source: SandboxSource;
}

async function rawFetch(url: string, init?: RequestInit): Promise<{ status: number; body: any }> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const r = await fetch(url, { ...init, signal: ctrl.signal });
    const body = await r.json().catch(() => null);
    return { status: r.status, body };
  } finally {
    clearTimeout(t);
  }
}

/** Shape guard — a real /sandbox body must carry clusters + a trace. */
function looksLikeSandbox(b: any): b is SandboxResponse {
  return (
    b &&
    Array.isArray(b.clusters) &&
    b.trace &&
    Array.isArray(b.trace.pairs) &&
    Array.isArray(b.trace.nodes)
  );
}

/** POST /sandbox {dataset, ...faders} → recluster + trace (real, else mock fallback). */
export async function postSandbox(
  dataset: string,
  params: SandboxParams,
): Promise<SandboxResult> {
  if (FORCE_MOCK) return { data: mockSandbox(dataset, params), source: 'mock' };
  try {
    const { status, body } = await rawFetch('/api/sandbox', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset, ...params }),
    });
    if (status === 200 && looksLikeSandbox(body)) return { data: body, source: 'live' };
  } catch {
    /* fall through to mock */
  }
  return { data: mockSandbox(dataset, params), source: 'mock' };
}

/**
 * GET /explain?cluster=nX — a node's neighbourhood + criteria. Falls back to a
 * local explanation derived from `last` (the most recent /sandbox response) so the
 * trace panel works offline / before the real endpoint lands.
 */
export async function explainCluster(
  dataset: string,
  clusterId: string,
  last: SandboxResponse,
): Promise<ExplainCluster | null> {
  if (!FORCE_MOCK) {
    try {
      const qs = new URLSearchParams({ dataset, cluster: clusterId });
      const { status, body } = await rawFetch(`/api/explain?${qs}`);
      if (status === 200 && body && Array.isArray(body.neighbors) && body.node) {
        return body as ExplainCluster;
      }
    } catch {
      /* fall through to local */
    }
  }
  return mockExplainCluster(last, clusterId);
}

/** GET /explain?pair=nX,nY — one pair's merge criteria (real, else local fallback). */
export async function explainPair(
  dataset: string,
  a: string,
  b: string,
  last: SandboxResponse,
): Promise<ExplainPair | null> {
  if (!FORCE_MOCK) {
    try {
      const qs = new URLSearchParams({ dataset, pair: `${a},${b}` });
      const { status, body } = await rawFetch(`/api/explain?${qs}`);
      if (status === 200 && body && Array.isArray(body.pair)) return body as ExplainPair;
    } catch {
      /* fall through to local */
    }
  }
  return mockExplainPair(last, a, b);
}
