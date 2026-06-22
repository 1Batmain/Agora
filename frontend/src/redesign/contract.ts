/**
 * FROZEN cross-lane contract (queue/front-redesign.md). The redesigned "Agora pour
 * députés" front codes EXACTLY against these shapes. The backend computes the
 * variance-driven hierarchy; the front only navigates it.
 *
 *   POST /analysis  {dataset, backend?} -> AnalysisPayload
 *   GET  /insights  {dataset, level, id?} -> { markdown }
 *   GET  /citations {dataset, theme_id}   -> Citation[]
 */

/** Which extraction backend the insights/claims pipeline should use. */
export type Backend = 'api' | 'mac' | 'auto';

/** A theme node positioned on the 2D spatial map (UMAP of centroids). */
export interface SpatialTheme {
  id: string;
  label: string;
  /**
   * Optional LLM-generated human title — the preferred bubble caption when
   * present. Falls back to `label` (the keyword stub) until the backend emits it.
   */
  title?: string;
  /** Optional keyword stubs (legacy caption / hover detail). */
  keywords?: string[];
  x: number; // UMAP-2D position (semantic proximity)
  y: number;
  n_avis: number;
  n_claims: number;
  weight: number;
  consensus: number; // 0..1
  dispersion: number; // internal spread; drives adaptive subdivision (backend-side)
  parent_id: string | null; // null = root (global level)
  has_children: boolean; // true → drillable; false → leaf (→ citations)
  color: string; // cluster colour (by macro); single source: backend palette.py
}

/** A co-occurrence edge between two themes (avis whose claims bridge a↔b). */
export interface SpatialEdge {
  a: string;
  b: string;
  weight: number;
}

/**
 * One headline indicator about the whole dataset (a card/gauge in the dashboard
 * under the map). The backend may emit these as objects (preferred) or, more
 * loosely, as a flat `{key: number}` record — the dashboard normalises both.
 */
export interface DatasetStat {
  key: string;
  label: string; // human-readable name ("Diversité des opinions")
  value: number; // raw value
  /** 0..1 fill for a gauge bar; omit → rendered as a plain count card. */
  gauge?: number;
  /** preformatted value for display ("72 %", "1 234"); else `value` is shown. */
  display?: string;
  /** one-line explanation of what the indicator means. */
  hint?: string;
}

/** Dataset-level indicators. Either a ready list, or a loose record of numbers. */
export type DatasetStats = DatasetStat[] | Record<string, number>;

/** `POST /analysis` → the whole spatial map (full adaptive tree + edges). */
export interface AnalysisPayload {
  themes: SpatialTheme[];
  edges: SpatialEdge[];
  params: Record<string, unknown>;
  backend_used: Backend;
  /** Optional headline indicators for the dashboard under the map (graceful if absent). */
  dataset_stats?: DatasetStats;
}

/** `GET /insights` → LLM Markdown synthesis for the current zoom level. */
export interface InsightsPayload {
  markdown: string;
}

export type InsightLevel = 'global' | 'theme';

/** One citation (verbatim claim) at a leaf theme, sorted by centroid proximity. */
export interface Citation {
  text: string;
  dist_to_centroid: number; // smaller = more representative
  weight: number;
  avis_id?: string; // source avis — opens its full text with highlights
}

/** A verbatim portion of an avis, anchored + coloured by its (macro) cluster. */
export interface AvisSpan {
  start: number;
  end: number;
  cluster_id: string | null;
  color: string;
  theme_label: string;
}

/** `GET /avis/{id}` → one avis in full, with its extractive spans to highlight. */
export interface AvisProvenance {
  id: string;
  text: string;
  spans: AvisSpan[];
}

/**
 * Where the data came from — surfaced in the UI so it's clear what's on screen:
 *  - `live`     : real precomputed analysis served from the backend cache;
 *  - `building` : backend is still precomputing (BUILD in background) — show "Analyse en cours…";
 *  - `error`    : backend reachable but the build failed / endpoint errored;
 *  - `mock`     : seeded demo data (ONLY under VITE_FORCE_MOCK, never a silent prod fallback).
 */
export type DataSource = 'live' | 'mock' | 'building' | 'error';

/** Progress of a backend BUILD, surfaced while an analysis isn't ready yet. */
export interface BuildProgress {
  status: string; // building | absent | error | ready
  phase?: string | null;
  detail?: string | null;
  done?: number | null;
  total?: number | null;
  error?: string | null;
}
