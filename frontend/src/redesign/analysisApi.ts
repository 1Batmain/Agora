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
  AvisListResponse,
  AvisProvenance,
  Backend,
  BuildProgress,
  Citation,
  DataSource,
  InsightLevel,
} from './contract';
import { mockAnalysis, mockAvis, mockCitations, mockInsights } from './mock';
import { rawFetch } from './http';

const FORCE_MOCK = import.meta.env.VITE_FORCE_MOCK === '1';

export interface Sourced<T> {
  data: T | null;
  source: DataSource;
  progress?: BuildProgress;
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
    const { status, body } = await rawFetch('/analysis', {
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
    const { status, body } = await rawFetch(`/insights?${qs}`);
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
    const { status, body } = await rawFetch(`/citations?${qs}`);
    if (status === 200 && Array.isArray(body)) {
      return { data: body as Citation[], source: 'live' };
    }
    return notReady<Citation[]>(body);
  } catch (e) {
    return { data: null, source: 'error', progress: { status: 'error', error: String(e) } };
  }
}

/** GET /cost {dataset} → coût LLM du traitement (tokens, $ estimé, durées) ; null si non mesuré. */
export async function fetchCost(
  dataset: string,
): Promise<import('./contract').CostPayload | null> {
  try {
    const qs = new URLSearchParams({ dataset });
    const { status, body } = await rawFetch(`/cost?${qs}`);
    if (status === 200 && body && body.total) return body as import('./contract').CostPayload;
  } catch {
    /* non mesuré / réseau */
  }
  return null;
}

/**
 * GET /opinion {dataset} → répartition d'opinion par thème feuille (objet de clivage +
 * stance agrégée). Artefact À PART, indépendant de l'analyse : chargé UNE fois puis
 * lookup par theme_id côté front. Léger comme les flags — pas de mapping building/error :
 * une liste vide (non bakée / réseau indispo) fait simplement disparaître la barre.
 */
export async function fetchOpinion(
  dataset: string,
): Promise<import('./contract').ThemeOpinion[]> {
  if (FORCE_MOCK) return [];
  try {
    const qs = new URLSearchParams({ dataset });
    const { status, body } = await rawFetch(`/opinion?${qs}`);
    if (status === 200 && body && Array.isArray(body.themes)) {
      return body.themes as import('./contract').ThemeOpinion[];
    }
  } catch {
    /* réseau indisponible → pas de répartition d'opinion */
  }
  return [];
}

/**
 * GET /arguments {dataset} → arguments minés par thème (synthèses LLM sourcées sur
 * contributions réelles). Artefact À PART et OPTIONNEL (les datasets déjà analysés
 * n'en ont pas) : même contrat que fetchOpinion — chargé UNE fois, lookup par
 * theme_id, liste vide = pas de panneau, jamais d'état bloquant.
 */
export async function fetchArguments(
  dataset: string,
): Promise<import('./contract').ThemeArguments[]> {
  if (FORCE_MOCK) return [];
  try {
    const qs = new URLSearchParams({ dataset });
    const { status, body } = await rawFetch(`/arguments?${qs}`);
    if (status === 200 && body && Array.isArray(body.themes)) {
      return body.themes as import('./contract').ThemeArguments[];
    }
  } catch {
    /* réseau indisponible → pas d'arguments minés */
  }
  return [];
}

/**
 * GET /demographics {dataset} → profil démographique du panel (global + majorité
 * par thème). Artefact À PART et OPTIONNEL : null (absent/réseau) = pas d'affichage,
 * les datasets sans données démographiques sont strictement inchangés.
 */
export async function fetchDemographics(
  dataset: string,
): Promise<import('./contract').DemographicsPayload | null> {
  if (FORCE_MOCK) return null;
  try {
    const qs = new URLSearchParams({ dataset });
    const { status, body } = await rawFetch(`/demographics?${qs}`);
    if (status === 200 && body && Array.isArray(body.themes) && body.themes.length) {
      return body as import('./contract').DemographicsPayload;
    }
  } catch {
    /* réseau indisponible → pas de profil démographique */
  }
  return null;
}

/**
 * Feedback FLAGS — Bob signals a badly cut / mis-targeted / mis-extracted avis
 * with a free-text comment, persisted server-side (upsert by avis_id) and editable.
 * These are LIGHT and independent of the analysis cache, so they never go through
 * the building/error mapping — a plain throw is enough for the UI to ignore.
 */

/**
 * One persisted flag — free-text feedback on an AVIS or a theme SYNTHESIS,
 * timestamped (UTC ISO-8601). Themes additionally carry `layer` (depth) +
 * `category`; avis flags keep `avis_id` (== target_id) for the existing avis UI.
 */
export interface Flag {
  target_type: 'avis' | 'theme';
  target_id: string;
  avis_id?: string; // present on avis flags (retro-compat)
  layer?: number | null;
  category?: string | null;
  text: string;
  created_at?: string;
  updated_at?: string;
}
/** @deprecated kept as an alias — use Flag. */
export type AvisFlag = Flag;

/** GET /flags {dataset} → all flags of a dataset, every type (front filters). */
export async function fetchFlags(dataset: string): Promise<Flag[]> {
  if (FORCE_MOCK) return [];
  const qs = new URLSearchParams({ dataset });
  const { status, body } = await rawFetch(`/flags?${qs}`);
  if (status === 200 && body && Array.isArray(body.flags)) return body.flags as Flag[];
  return [];
}

/** POST /flag — generic upsert; returns the saved flag (or null on failure). */
async function postFlag(payload: Record<string, unknown>): Promise<Flag | null> {
  const { status, body } = await rawFetch('/flag', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (status === 200 && body && body.ok && body.flag) return body.flag as Flag;
  return null;
}

/** DELETE /flag/{id} — generic remove; returns whether it existed. */
async function deleteFlagOf(
  dataset: string,
  targetType: 'avis' | 'theme',
  targetId: string,
): Promise<boolean> {
  const qs = new URLSearchParams({ dataset, target_type: targetType });
  const { status, body } = await rawFetch(
    `/flag/${encodeURIComponent(targetId)}?${qs}`,
    { method: 'DELETE' },
  );
  return status === 200 && Boolean(body && body.ok && body.removed);
}

/** Upsert the AVIS flag (free-text), returns the saved flag. */
export async function upsertFlag(
  dataset: string,
  avisId: string,
  text: string,
): Promise<Flag | null> {
  if (FORCE_MOCK) return { target_type: 'avis', target_id: avisId, avis_id: avisId, text };
  return postFlag({ dataset, target_type: 'avis', target_id: avisId, text });
}

/** Remove the AVIS flag, returns whether it existed. */
export async function deleteFlag(dataset: string, avisId: string): Promise<boolean> {
  if (FORCE_MOCK) return true;
  return deleteFlagOf(dataset, 'avis', avisId);
}

/** Upsert a THEME-synthesis flag (category + free-text + layer/depth). */
export async function upsertThemeFlag(
  dataset: string,
  themeId: string,
  layer: number | null,
  category: string,
  text: string,
): Promise<Flag | null> {
  if (FORCE_MOCK)
    return { target_type: 'theme', target_id: themeId, layer, category, text };
  return postFlag({ dataset, target_type: 'theme', target_id: themeId, layer, category, text });
}

/** Remove a theme-synthesis flag, returns whether it existed. */
export async function deleteThemeFlag(dataset: string, themeId: string): Promise<boolean> {
  if (FORCE_MOCK) return true;
  return deleteFlagOf(dataset, 'theme', themeId);
}

/**
 * GET /avis_list {dataset, theme_id?, q?, limit, offset} → a paginated/filtered
 * page of avis for the exploration page, or building/error while the analysis cooks.
 */
export async function fetchAvisList(
  dataset: string,
  opts: {
    themeId?: string | null; q?: string;
    stance?: 'favorable' | 'defavorable' | null;
    limit?: number; offset?: number;
  } = {},
): Promise<Sourced<AvisListResponse>> {
  const { themeId, q, stance, limit = 50, offset = 0 } = opts;
  if (FORCE_MOCK) return { data: { total: 0, items: [] }, source: 'mock' };
  try {
    const qs = new URLSearchParams({ dataset, limit: String(limit), offset: String(offset) });
    if (themeId) qs.set('theme_id', themeId);
    if (q && q.trim()) qs.set('q', q.trim());
    if (stance) qs.set('stance', stance);
    const { status, body } = await rawFetch(`/avis_list?${qs}`);
    if (status === 200 && body && Array.isArray(body.items) && typeof body.total === 'number') {
      return { data: body as AvisListResponse, source: 'live' };
    }
    return notReady<AvisListResponse>(body);
  } catch (e) {
    return { data: null, source: 'error', progress: { status: 'error', error: String(e) } };
  }
}

/** GET /avis/{id} {dataset} → full avis text + claims (spans + target), or building/error. */
export async function fetchAvis(
  dataset: string,
  avisId: string,
): Promise<Sourced<AvisProvenance>> {
  if (FORCE_MOCK) return { data: mockAvis(dataset, avisId), source: 'mock' };
  try {
    const qs = new URLSearchParams({ dataset });
    const { status, body } = await rawFetch(`/avis/${encodeURIComponent(avisId)}?${qs}`);
    if (status === 200 && body && typeof body.text === 'string' && Array.isArray(body.claims)) {
      return { data: body as AvisProvenance, source: 'live' };
    }
    return notReady<AvisProvenance>(body);
  } catch (e) {
    return { data: null, source: 'error', progress: { status: 'error', error: String(e) } };
  }
}
