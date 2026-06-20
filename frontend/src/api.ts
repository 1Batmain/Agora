import type { GraphPayload, GraphStats, KnobSpec, Knobs, Theme } from './types';

/**
 * Backend client. Everything goes through the vite proxy at `/api/*` → :8010.
 * When the backend is down we degrade to the static `public/graph.json` and the
 * knobs panel becomes read-only.
 */

/** Defaults + bounds from the FROZEN cross-lane contract (nomic-v2 winner). */
export const DEFAULT_KNOBS: KnobSpec[] = [
  { key: 'dedup', label: 'dedup (cosine)', value: 0.95, min: 0.9, max: 0.99, step: 0.01, hint: 'fusion near-dups' },
  { key: 'min_chars', label: 'min_chars', value: 12, min: 0, max: 40, step: 1, hint: 'filtre avis courts' },
  { key: 'k', label: 'k (voisins)', value: 12, min: 5, max: 30, step: 1, hint: 'densité k-NN' },
  { key: 'threshold', label: 'threshold (cosine)', value: 0.6, min: 0.4, max: 0.85, step: 0.01, hint: 'coupe les arêtes' },
  { key: 'resolution_macro', label: 'resolution_macro', value: 1.0, min: 0.3, max: 3.0, step: 0.1, hint: 'granularité macros' },
  { key: 'resolution_sub', label: 'resolution_sub', value: 1.5, min: 0.5, max: 4.0, step: 0.1, hint: 'granularité sous-thèmes' },
  { key: 'min_sub_size', label: 'min_sub_size', value: 18, min: 5, max: 40, step: 1, hint: 'fusion des miettes' },
];

const TIMEOUT_MS = 8000;

async function jsonFetch(url: string, init?: RequestInit): Promise<any> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const r = await fetch(url, { ...init, signal: ctrl.signal });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } finally {
    clearTimeout(t);
  }
}

/**
 * Build the knob specs. Tries `GET /api/params` and merges any
 * default/min/max/step it returns over our contract defaults. Tolerant of a few
 * response shapes ({k:{default,min,max,step}} or {k:{value,...}}); falls back to
 * pure defaults on any error → throws so the caller knows backend is unreachable.
 */
export async function fetchParams(): Promise<KnobSpec[]> {
  const raw = await jsonFetch('/api/params');
  const params = raw?.params ?? raw ?? {};
  return DEFAULT_KNOBS.map((spec) => {
    const p = params[spec.key];
    if (!p || typeof p !== 'object') return { ...spec };
    const value = num(p.default ?? p.value ?? p.def, spec.value);
    return {
      ...spec,
      value,
      min: num(p.min, spec.min),
      max: num(p.max, spec.max),
      step: num(p.step, spec.step),
    };
  });
}

function num(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

/** POST /api/recluster {knobs} → fresh GraphPayload. */
export async function recluster(knobs: Knobs): Promise<GraphPayload> {
  return (await jsonFetch('/api/recluster', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(knobs),
  })) as GraphPayload;
}

/** Static fallback: REAL consultation first, committed fixture otherwise. */
export async function fetchStatic(): Promise<GraphPayload> {
  try {
    return (await jsonFetch('/graph.json')) as GraphPayload;
  } catch {
    return (await jsonFetch('/graph.sample.json')) as GraphPayload;
  }
}

/**
 * Derive the stats bar from a payload. Uses `meta.stats` when the backend
 * provides it, else reconstructs from themes/nodes/meta so the static fallback
 * still shows meaningful numbers.
 */
export function deriveStats(payload: GraphPayload): GraphStats {
  const s = payload.meta?.stats ?? {};
  const macros = payload.themes.filter((t: Theme) => t.level === 0);
  const subs = payload.themes.filter((t: Theme) => t.level === 1);
  const lh = payload.meta?.clustering?.leiden_hierarchy ?? {};
  return {
    n_macros: num(s.n_macros, macros.length),
    n_subs: num(s.n_subs, subs.length),
    n_nodes: num(s.n_nodes, payload.nodes.length),
    modularity:
      typeof s.modularity === 'number'
        ? s.modularity
        : typeof lh.macro_modularity === 'number'
          ? lh.macro_modularity
          : null,
    took_ms: typeof s.took_ms === 'number' ? s.took_ms : null,
  };
}
