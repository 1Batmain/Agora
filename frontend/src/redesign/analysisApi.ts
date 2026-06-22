/**
 * Client for the redesigned-front endpoints (`/analysis`, `/insights`,
 * `/citations`) behind the vite `/api/*` proxy → :8010.
 *
 * Strategy: try the REAL backend first; on any failure (endpoint not built yet,
 * backend down) fall back to the seeded MOCK so the UI stays navigable. A build
 * flag `VITE_FORCE_MOCK=1` forces the mock path (dev without a backend). Each
 * call reports its `DataSource` so the shell can badge mock vs live.
 */
import type {
  AnalysisPayload,
  Backend,
  Citation,
  DataSource,
  InsightLevel,
} from './contract';
import { mockAnalysis, mockCitations, mockInsights } from './mock';

const FORCE_MOCK = import.meta.env.VITE_FORCE_MOCK === '1';
const TIMEOUT_MS = 180000;

export interface Sourced<T> {
  data: T;
  source: DataSource;
}

async function jsonFetch(url: string, init?: RequestInit): Promise<any> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const r = await fetch(url, { ...init, signal: ctrl.signal });
    if (!r.ok) {
      const detail = await r
        .clone()
        .json()
        .then((b) => (b && typeof b.detail === 'string' ? b.detail : null))
        .catch(() => null);
      throw new Error(detail ? `HTTP ${r.status} — ${detail}` : `HTTP ${r.status}`);
    }
    return await r.json();
  } finally {
    clearTimeout(t);
  }
}

/** POST /analysis {dataset, backend?} → spatial map (themes x,y + edges). */
export async function fetchAnalysis(
  dataset: string,
  backend: Backend = 'auto',
): Promise<Sourced<AnalysisPayload>> {
  if (!FORCE_MOCK) {
    try {
      const data = (await jsonFetch('/api/analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset, backend }),
      })) as AnalysisPayload;
      if (data && Array.isArray(data.themes)) return { data, source: 'live' };
    } catch {
      /* fall through to mock */
    }
  }
  return { data: mockAnalysis(dataset, backend), source: 'mock' };
}

/** GET /insights {dataset, level, id?} → { markdown }. */
export async function fetchInsights(
  dataset: string,
  level: InsightLevel,
  themeId?: string,
  themeForMock?: import('./contract').SpatialTheme,
): Promise<Sourced<string>> {
  if (!FORCE_MOCK) {
    try {
      const qs = new URLSearchParams({ dataset, level });
      if (themeId) qs.set('id', themeId);
      const data = await jsonFetch(`/api/insights?${qs}`);
      if (data && typeof data.markdown === 'string') return { data: data.markdown, source: 'live' };
    } catch {
      /* fall through to mock */
    }
  }
  return { data: mockInsights(dataset, level, themeForMock).markdown, source: 'mock' };
}

/** GET /citations {dataset, theme_id} → Citation[] sorted by centroid distance. */
export async function fetchCitations(
  dataset: string,
  themeId: string,
): Promise<Sourced<Citation[]>> {
  if (!FORCE_MOCK) {
    try {
      const qs = new URLSearchParams({ dataset, theme_id: themeId });
      const data = await jsonFetch(`/api/citations?${qs}`);
      if (Array.isArray(data)) return { data: data as Citation[], source: 'live' };
    } catch {
      /* fall through to mock */
    }
  }
  return { data: mockCitations(dataset, themeId), source: 'mock' };
}
