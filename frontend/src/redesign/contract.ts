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
  x: number; // UMAP-2D position (semantic proximity)
  y: number;
  n_avis: number;
  n_claims: number;
  weight: number;
  consensus: number; // 0..1
  dispersion: number; // internal spread; drives adaptive subdivision (backend-side)
  parent_id: string | null; // null = root (global level)
  has_children: boolean; // true → drillable; false → leaf (→ citations)
}

/** A co-occurrence edge between two themes (avis whose claims bridge a↔b). */
export interface SpatialEdge {
  a: string;
  b: string;
  weight: number;
}

/** `POST /analysis` → the whole spatial map (full adaptive tree + edges). */
export interface AnalysisPayload {
  themes: SpatialTheme[];
  edges: SpatialEdge[];
  params: Record<string, unknown>;
  backend_used: Backend;
}

/** `GET /insights` → LLM Markdown synthesis for the current zoom level. */
export interface InsightsPayload {
  markdown: string;
}

export type InsightLevel = 'global' | 'theme';

/** One citation (raw avis text) at a leaf theme, sorted by centroid proximity. */
export interface Citation {
  text: string;
  dist_to_centroid: number; // smaller = more representative
  weight: number;
}

/** Where the data came from — surfaced in the UI so it's clear mock vs live. */
export type DataSource = 'live' | 'mock';
