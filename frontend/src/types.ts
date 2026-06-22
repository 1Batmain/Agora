/**
 * Shapes of the GraphPayload returned by the recluster backend (:8010) and of
 * the static `public/graph.json` fallback. Kept loose on purpose — the backend
 * is the source of truth and may add fields; we only read what the console needs.
 */

export interface GraphNode {
  id: string;
  type: string;
  label: string;
  props: {
    text: string;
    text_clean?: string;
    ts?: string;
    lang?: string;
    author_hash?: string;
    source?: string;
    weight?: number;
  };
  cluster_id: number; // sub-theme id (Leiden level 1) / flat cluster (HDBSCAN; -1 = noise)
  macro_id: number; // macro id (level 0)
  color?: string;
  x?: number; // UMAP-2D coord (HDBSCAN method only)
  y?: number;
}

/** Clustering methods the backend can switch between. */
export type ClusterMethod = 'leiden' | 'hdbscan';

/** Theme-naming methods (orthogonal to the clustering method). */
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

export interface GraphStats {
  method: ClusterMethod;
  naming: NamingMethod; // naming method ACTUALLY applied (may differ if LLM fell back)
  naming_requested: NamingMethod | null; // what the user asked for
  naming_fallback: boolean; // true when LLM fell back to c-TF-IDF
  n_macros: number;
  n_subs: number;
  n_nodes: number;
  n_clusters: number | null; // HDBSCAN: flat cluster count
  n_noise: number | null; // HDBSCAN: unclassified (cluster_id -1) count
  modularity: number | null;
  took_ms: number | null;
}

export interface GraphPayload {
  meta: Record<string, any> & { stats?: Partial<GraphStats> };
  nodes: GraphNode[];
  links?: unknown[];
  themes: Theme[];
}

/** `POST /api/synthesize` → short Markdown report (Mistral). */
export interface SynthesisResult {
  report_markdown: string;
  meta: {
    model?: string;
    took_ms?: number;
    n_clusters?: number;
    fallback?: boolean; // true when no key / API error → report is a notice
    reason?: string;
    lang?: string;
    truncated?: boolean;
  };
}

/** One selectable dataset, from `GET /api/datasets`. */
export interface Dataset {
  id: string;
  label: string;
  n_nodes: number;
  languages: string[];
  lang_counts?: Record<string, number>;
  source?: string;
}

/** View mode: existing cluster views vs the emergent-claims map. */
export type ViewMode = 'clusters' | 'claims';

/** One emergent theme (cluster of claims) from `POST /api/claims`. */
export interface ClaimTheme {
  cluster_id: number;
  name: string; // c-TF-IDF label
  keywords: string[];
  n_claims: number;
  n_avis: number;
  weight: number;
  consensus: number;
  diversity: number;
  representative_claims: string[];
}

/** A co-occurrence edge: `count` avis whose claims bridge themes `a` and `b`. */
export interface ClaimsCooc {
  a: number;
  b: number;
  count: number;
}

/** `POST /api/claims` → emergent themes + co-occurrence map. */
export interface ClaimsPayload {
  themes: ClaimTheme[];
  cooccurrence: ClaimsCooc[];
  params: Record<string, any>;
  meta?: Record<string, any>;
}

/** One tunable knob, used to build a slider in the panel. */
export interface KnobSpec {
  key: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  hint?: string;
}

export type Knobs = Record<string, number>;
