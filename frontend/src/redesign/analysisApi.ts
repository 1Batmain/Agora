/**
 * Client for the redesigned-front endpoints (`/analysis`, `/insights`,
 * `/citations`) behind the vite `/api/*` proxy → :8010.
 *
 * SERVE-only backend: these endpoints ONLY read the precomputed analysis cache.
 * When an analysis isn't ready yet the backend answers `{status: building|absent|
 * error}` (HTTP 202/503) while a BUILD runs in the background. This client maps
 * that to a `building`/`error` `DataSource` so the shell shows "Analyse en cours…"
 * (and polls), instead of silently swapping in mock data.
 *
 * Mock is ONLY used when `VITE_FORCE_MOCK=1` (isolated dev without a backend) —
 * never as a hidden prod fallback. Each call reports its `DataSource` so the shell
 * can badge live / build / mock / error.
 */
import type {
  AnalysisPayload,
  AvisProvenance,
  Backend,
  BuildProgress,
  Citation,
  DataSource,
  InsightLevel,
} from './contract';
import { mockAnalysis, mockAvis, mockCitations, mockInsights } from './mock';

const FORCE_MOCK = import.meta.env.VITE_FORCE_MOCK === '1';
const TIMEOUT_MS = 180000;

export interface Sourced<T> {
  data: T | null;
  source: DataSource;
  progress?: BuildProgress;
}

interface RawResult {
  status: number;
  body: any;
}

/** Fetch + parse JSON, returning status code and body (never throws on non-2xx). */
async function rawFetch(url: string, init?: RequestInit): Promise<RawResult> {
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

/** Map a not-ready backend body to a building/error source (no mock fallback). */
function notReady<T>(body: any): Sourced<T> {
  const status: string = (body && body.status) || 'building';
  const source: DataSource = status === 'error' ? 'error' : 'building';
  return { data: null, source, progress: (body as BuildProgress) ?? undefined };
}

/** POST /analysis {dataset} → spatial map (themes x,y + edges), or building/error. */
export async function fetchAnalysis(
  dataset: string,
  backend: Backend = 'auto',
): Promise<Sourced<AnalysisPayload>> {
  if (FORCE_MOCK) return { data: mockAnalysis(dataset, backend), source: 'mock' };
  try {
    const { status, body } = await rawFetch('/api/analysis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset, backend }),
    });
    if (status === 200 && body && Array.isArray(body.themes)) {
      return { data: body as AnalysisPayload, source: 'live' };
    }
    return notReady<AnalysisPayload>(body);
  } catch (e) {
    // Backend unreachable — surface an error, do NOT silently show mock.
    return { data: null, source: 'error', progress: { status: 'error', error: String(e) } };
  }
}

/** GET /insights {dataset, level, id?} → { markdown }, or building/error. */
export async function fetchInsights(
  dataset: string,
  level: InsightLevel,
  themeId?: string,
  themeForMock?: import('./contract').SpatialTheme,
): Promise<Sourced<string>> {
  if (FORCE_MOCK) {
    return { data: mockInsights(dataset, level, themeForMock).markdown, source: 'mock' };
  }
  try {
    const qs = new URLSearchParams({ dataset, level });
    if (themeId) qs.set('id', themeId);
    const { status, body } = await rawFetch(`/api/insights?${qs}`);
    if (status === 200 && body && typeof body.markdown === 'string') {
      return { data: body.markdown, source: 'live' };
    }
    return notReady<string>(body);
  } catch (e) {
    return { data: null, source: 'error', progress: { status: 'error', error: String(e) } };
  }
}

/** GET /citations {dataset, theme_id} → Citation[] (centroid-sorted), or building/error. */
export async function fetchCitations(
  dataset: string,
  themeId: string,
): Promise<Sourced<Citation[]>> {
  if (FORCE_MOCK) return { data: mockCitations(dataset, themeId), source: 'mock' };
  try {
    const qs = new URLSearchParams({ dataset, theme_id: themeId });
    const { status, body } = await rawFetch(`/api/citations?${qs}`);
    if (status === 200 && Array.isArray(body)) {
      return { data: body as Citation[], source: 'live' };
    }
    return notReady<Citation[]>(body);
  } catch (e) {
    return { data: null, source: 'error', progress: { status: 'error', error: String(e) } };
  }
}

/**
 * Feedback FLAGS — Bob signals a badly cut / mis-targeted / mis-extracted avis
 * with a free-text comment, persisted server-side (upsert by avis_id) and editable.
 * These are LIGHT and independent of the analysis cache, so they never go through
 * the building/error mapping — a plain throw is enough for the UI to ignore.
 */

/** One persisted flag — free-text feedback on an avis, timestamped (UTC ISO-8601). */
export interface AvisFlag {
  avis_id: string;
  text: string;
  created_at?: string;
  updated_at?: string;
}

/** GET /flags {dataset} → all flags of a dataset (to restore state on load). */
export async function fetchFlags(dataset: string): Promise<AvisFlag[]> {
  if (FORCE_MOCK) return [];
  const qs = new URLSearchParams({ dataset });
  const { status, body } = await rawFetch(`/api/flags?${qs}`);
  if (status === 200 && body && Array.isArray(body.flags)) return body.flags as AvisFlag[];
  return [];
}

/** POST /flag {dataset, avis_id, text} → upsert the avis flag, returns the saved flag. */
export async function upsertFlag(
  dataset: string,
  avisId: string,
  text: string,
): Promise<AvisFlag | null> {
  if (FORCE_MOCK) return { avis_id: avisId, text };
  const { status, body } = await rawFetch('/api/flag', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset, avis_id: avisId, text }),
  });
  if (status === 200 && body && body.ok && body.flag) return body.flag as AvisFlag;
  return null;
}

/** DELETE /flag/{avis_id} {dataset} → remove the avis flag, returns whether it existed. */
export async function deleteFlag(dataset: string, avisId: string): Promise<boolean> {
  if (FORCE_MOCK) return true;
  const qs = new URLSearchParams({ dataset });
  const { status, body } = await rawFetch(`/api/flag/${encodeURIComponent(avisId)}?${qs}`, {
    method: 'DELETE',
  });
  return status === 200 && Boolean(body && body.ok && body.removed);
}

/** GET /avis/{id} {dataset} → full avis text + claims (spans + target), or building/error. */
export async function fetchAvis(
  dataset: string,
  avisId: string,
): Promise<Sourced<AvisProvenance>> {
  if (FORCE_MOCK) return { data: mockAvis(dataset, avisId), source: 'mock' };
  try {
    const qs = new URLSearchParams({ dataset });
    const { status, body } = await rawFetch(`/api/avis/${encodeURIComponent(avisId)}?${qs}`);
    if (status === 200 && body && typeof body.text === 'string' && Array.isArray(body.claims)) {
      return { data: body as AvisProvenance, source: 'live' };
    }
    return notReady<AvisProvenance>(body);
  } catch (e) {
    return { data: null, source: 'error', progress: { status: 'error', error: String(e) } };
  }
}
