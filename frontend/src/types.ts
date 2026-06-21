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
  cluster_id: number; // sub-theme id (level 1)
  macro_id: number; // macro id (level 0)
  color?: string;
}

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
  n_macros: number;
  n_subs: number;
  n_nodes: number;
  modularity: number | null;
  took_ms: number | null;
}

export interface GraphPayload {
  meta: Record<string, any> & { stats?: Partial<GraphStats> };
  nodes: GraphNode[];
  links?: unknown[];
  themes: Theme[];
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
