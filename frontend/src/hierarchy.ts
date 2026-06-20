import { hierarchy, pack, type HierarchyCircularNode } from 'd3-hierarchy';
import type { GraphNode, GraphPayload, Theme } from './types';

/**
 * The circle-packing hierarchy: root → macros → sub-themes → avis.
 * Each datum carries enough to render a circle and drive the side panels.
 */
export interface PackDatum {
  kind: 'root' | 'macro' | 'sub' | 'avis';
  id: string;
  label: string;
  color: string;
  value: number; // weight (avis) — internal nodes sum their children
  theme?: Theme;
  node?: GraphNode;
  children?: PackDatum[];
}

export type PackNode = HierarchyCircularNode<PackDatum>;

/** Turn a GraphPayload into a laid-out circle-pack hierarchy fit to `size`. */
export function buildPack(payload: GraphPayload, size: number): PackNode {
  const root = toDatum(payload);
  const h = hierarchy<PackDatum>(root)
    .sum((d) => (d.children && d.children.length ? 0 : d.value))
    .sort((a, b) => (b.value ?? 0) - (a.value ?? 0));
  return pack<PackDatum>().size([size, size]).padding(3)(h);
}

function toDatum(payload: GraphPayload): PackDatum {
  const macros = payload.themes.filter((t) => t.level === 0);
  const subs = payload.themes.filter((t) => t.level === 1);
  const subsByParent = new Map<number, Theme[]>();
  for (const s of subs) {
    const arr = subsByParent.get(s.parent_id ?? -1) ?? [];
    arr.push(s);
    subsByParent.set(s.parent_id ?? -1, arr);
  }
  const nodesBySub = new Map<number, GraphNode[]>();
  for (const n of payload.nodes) {
    const arr = nodesBySub.get(n.cluster_id) ?? [];
    arr.push(n);
    nodesBySub.set(n.cluster_id, arr);
  }

  return {
    kind: 'root',
    id: 'root',
    label: 'Consultation',
    color: '#1a1c22',
    value: 0,
    children: macros
      .sort((a, b) => b.weight_sum - a.weight_sum)
      .map((m) => ({
        kind: 'macro' as const,
        id: `macro:${m.cluster_id}`,
        label: m.label,
        color: m.color,
        value: m.weight_sum,
        theme: m,
        children: (subsByParent.get(m.cluster_id) ?? [])
          .sort((a, b) => b.weight_sum - a.weight_sum)
          .map((s) => ({
            kind: 'sub' as const,
            id: `sub:${s.cluster_id}`,
            label: s.label,
            color: s.color,
            value: s.weight_sum,
            theme: s,
            children: (nodesBySub.get(s.cluster_id) ?? []).map((n) => ({
              kind: 'avis' as const,
              id: n.id,
              label: n.label,
              color: n.color ?? s.color,
              value: n.props.weight ?? 1,
              node: n,
            })),
          })),
      })),
  };
}
