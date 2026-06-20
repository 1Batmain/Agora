/**
 * Canonical batch data shapes — aligned on the cross-lane contract
 * (`queue/cross-lane.md`) and the reference artefact
 * `pipeline/cluster/fixtures/graph.sample.json`.
 *
 * ⚠️ Contract precision (post-merge nlp): `cluster_id` (int Leiden community)
 * and `color` (hex palette) live at the TOP LEVEL of a node — NOT inside
 * `props`. The swarm is coloured by `node.color`.
 */

export interface GraphNode {
  id: string;
  type: string; // "idea" (extensible)
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
  cluster_id: number; // top-level — leaf (sub-theme) Leiden community
  macro_id?: number; // top-level — parent macro-theme (hierarchical mode)
  color: string; // top-level — hex, palette colour for the swarm (macro colour)
}

export interface GraphLink {
  source: string;
  target: string;
  type: string; // "knn"
  props: { weight: number }; // cosine similarity
}

export interface Theme {
  cluster_id: number;
  member_ids: string[];
  size: number;
  weight_sum: number;
  diversity: number;
  consensus: number;
  centroid?: number[];
  label: string;
  keywords: string[];
  color: string;
  // Hierarchy (cross-lane contract): macro = level 0, sub-theme = level 1.
  level: number;
  parent_id: number | null; // macro id for a sub-theme, null for a macro
  children: number[]; // leaf ids for a macro, [] for a sub-theme
}

export interface GraphPayload {
  meta: Record<string, unknown>;
  nodes: GraphNode[];
  links: GraphLink[];
  themes: Theme[];
}

/** A node enriched with its dense index into the positions buffer. */
export interface IndexedNode extends GraphNode {
  index: number;
}

/**
 * Pre-indexed view of a {@link GraphPayload} for the renderer: dense node array
 * (each carrying its buffer index), id→index and id→node lookups. The force
 * worker reads positions by this same index order.
 */
export interface GraphIndex {
  nodes: IndexedNode[];
  links: GraphLink[];
  themes: Theme[];
  meta: Record<string, unknown>;
  indexOf: Map<string, number>;
  byId: Map<string, IndexedNode>;
  themesById: Map<number, Theme>;
}

/** A macro-theme paired with its resolved sub-themes — the drill-down tree. */
export interface MacroNode {
  macro: Theme;
  subs: Theme[];
}

/**
 * Build the macro→sub drill-down tree from a flat `themes[]` list. Macros
 * (`level === 0`) are sorted by weight; each macro's children are resolved via
 * `themesById` and likewise weight-sorted. Falls back gracefully on flat
 * payloads (no `level`): every theme is treated as its own macro with no subs.
 */
export function buildThemeTree(index: GraphIndex): MacroNode[] {
  const byWeight = (a: Theme, b: Theme) => b.weight_sum - a.weight_sum;
  const macros = index.themes.filter((t) => (t.level ?? 0) === 0);
  return macros.sort(byWeight).map((macro) => ({
    macro,
    subs: (macro.children ?? [])
      .map((id) => index.themesById.get(id))
      .filter((t): t is Theme => t != null)
      .sort(byWeight),
  }));
}

export function buildIndex(payload: GraphPayload): GraphIndex {
  const nodes: IndexedNode[] = payload.nodes.map((n, index) => ({ ...n, index }));
  const indexOf = new Map<string, number>();
  const byId = new Map<string, IndexedNode>();
  for (const n of nodes) {
    indexOf.set(n.id, n.index);
    byId.set(n.id, n);
  }
  const themesById = new Map<number, Theme>();
  for (const t of payload.themes) themesById.set(t.cluster_id, t);
  return {
    nodes,
    links: payload.links,
    themes: payload.themes,
    meta: payload.meta,
    indexOf,
    byId,
    themesById,
  };
}
