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
  /** Contributions représentatives (proches du centroïde), verbatim. */
  representative_claims?: string[];
  /**
   * Optional LLM hover fields (emitted by the backend in parallel — graceful when
   * absent). `hook` = one-line accroche; `description` = a short Markdown synthesis
   * shown ONLY in the hover tooltip (never baked onto the bubble); `convergence` =
   * 0..1 "convergence des idées" within the cluster.
   */
  hook?: string;
  description?: string;
  convergence?: number;
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
  /**
   * Optional dataset-level intro shown in the GLOBAL view (above the map): a short
   * description of the consultation and the context in which contributions were
   * collected. Emitted by the backend in parallel — both fields graceful if absent.
   */
  dataset_description?: string;
  dataset_context?: string;
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

/** A character range `[start, end)` into the avis text (verbatim gate applies). */
export interface CharRange {
  start: number;
  end: number;
}

/**
 * claim-v2 — one claim extracted from an avis. A claim is made of 1..N verbatim
 * `spans` (possibly NON-contiguous) all painted in the claim's `color` (its
 * cluster colour), plus an optional `target`: the verbatim CIBLE, a sub-range
 * that lives INSIDE one of the claim's spans, underlined so it stands out within
 * the highlight. Every range (spans AND target) is an EXACT substring of the avis
 * text — the backend's verbatim gate guarantees no drift.
 */
export interface AvisClaim {
  id: string;
  cluster_id: string | null;
  color: string;
  spans: CharRange[];
  target: CharRange | null;
  theme_title: string;
}

/**
 * `GET /avis/{id}` → one avis in full, with its claims (spans + target) to render.
 *  - `text`    : the ORIGINAL (claims' spans/target are offsets into it — verbatim gate);
 *  - `text_fr` : French translation precomputed at build (`null` if already FR / untranslated);
 *  - `lang`    : the avis' language code. When `lang !== 'fr'` the UI shows `text_fr` by
 *    default with a « voir l'original » toggle (highlights render on the original `text`).
 */
export interface AvisProvenance {
  id: string;
  text: string;
  text_fr?: string | null;
  lang?: string;
  claims: AvisClaim[];
}

/**
 * One row of the avis-exploration list (`GET /avis_list`). Carries the WHOLE avis
 * (`text`/`text_fr`/`lang`/`claims`, same shape as `AvisProvenance`) so the explorer
 * renders each avis INLINE (full text + verbatim highlights) without a per-card
 * `/avis/{id}` round-trip — plus a flattened `excerpt` and the distinct theme chips.
 */
export interface AvisListItem {
  avis_id: string;
  /** ~220-char preview of the avis text (whitespace-flattened). */
  excerpt: string;
  /** Distinct themes carried by the avis' claims (chips), in first-seen order. */
  themes: { id: string; title: string; color: string }[];
  /** Full avis text (claims' spans/target are offsets into it — verbatim gate). */
  text: string;
  /** French translation precomputed at build (`null`/absent if already FR). */
  text_fr?: string | null;
  /** Avis language code (`'fr'` default). */
  lang?: string;
  /** Claims (verbatim spans + target) to render the inline highlights. */
  claims: AvisClaim[];
}

/** `GET /avis_list` → a paginated/filtered page of avis (`total` = before paging). */
export interface AvisListResponse {
  total: number;
  items: AvisListItem[];
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

/*
 * ───────────────────────────────────────────────────────────────────────────
 * Dataset selector / open-consultation participation
 *
 *   GET  /datasets          -> Consultation[]
 *   POST /submit {id, text} -> SubmitResult
 *
 * Source de vérité = backend ; on ne garde ici que les shapes réellement
 * consommées par l'UI (Consultation, Theme, SubmitResult).
 * ───────────────────────────────────────────────────────────────────────────
 */

/** Theme-naming methods served by the backend (capacités exposées par /datasets). */
export type NamingMethod = 'ctfidf' | 'centroid' | 'llm';

export interface Theme {
  cluster_id: number;
  level: 0 | 1; // 0 = macro, 1 = sub-theme
  parent_id: number | null;
  children: number[]; // sub-theme cluster_ids (macros only)
  member_ids: string[];
  size: number;
  weight_sum: number;
  diversity?: number;
  consensus?: number;
  label: string;
  keywords?: string[];
  color: string;
}

/**
 * One consultation, from `GET /api/datasets`. MIROIR EXACT du TypedDict
 * `Consultation` backend (backend/consultation_schema.py) — source de vérité
 * unique, construite par `dataset_descriptor` / `open_consultation_descriptor`.
 * Le endpoint y ajoute aussi les capacités serveur `namings`/`default_naming`.
 */
export interface Consultation {
  id: string;
  label: string;
  /** Consultation status: 'open' (participation en cours) | 'closed' (analyse seule). */
  status: 'open' | 'closed';
  /** Échantillon réellement analysé (= ancien n_nodes). */
  n_sample: number;
  /** Nombre RÉEL de contributions reçues (avant cap d'échantillonnage). */
  n_contributions: number;
  /** Rétro-compat : alias historique de n_sample (toujours == n_sample). */
  n_nodes: number;
  languages: string[];
  lang_counts: Record<string, number>;
  source: string;
  /** Consultations OUVERTES (et clôturées qui en exposent un) : sujet affiché. */
  question?: string;
  context?: string;
  /** Capacités serveur (méthodes de nommage) — ajoutées par le endpoint /datasets. */
  namings?: NamingMethod[];
  default_naming?: NamingMethod;
}

/** `POST /submit` → corrélation instantanée d'une contribution citoyenne. */
export interface SubmitResult {
  ok: boolean;
  n_similar: number;
  nearest_excerpt: string | null;
  message: string;
}
