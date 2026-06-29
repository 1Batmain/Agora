import { useMemo, useState } from 'react';
import type { SpatialTheme } from './contract';

/**
 * Navigateur de thèmes ARBORESCENT (accordéon récursif) — remplace le `Poll`
 * statique dans le template de synthèse. On voit les gros thèmes en barres de %
 * (part des voix), on SCROLLE vers les moins représentés, et on CLIQUE un thème
 * à enfants pour EXPANDRE ses sous-thèmes indentés (chacun à son tour expandable).
 *
 * 100 % FRONT : tout vient du payload `/analysis` (liste plate de thèmes reliés
 * par `parent_id`). Le `%` d'une ligne = `n_avis / total_du_parent` (chaque niveau
 * ≈ 100 % de son parent : racine ÷ `total`, sous-thème ÷ `n_avis` du parent).
 */
export function ThemeNavigator({
  themes,
  total,
  onSelect,
}: {
  /** Liste COMPLÈTE des thèmes du payload (tous niveaux confondus). */
  themes: SpatialTheme[];
  /** Voix du niveau racine (dénominateur des thèmes `parent_id == null`). */
  total: number;
  /** Optionnel : drill la vue d'analyse sur le thème cliqué (l'accordéon marche seul). */
  onSelect?: (themeId: string) => void;
}) {
  // parent_id → enfants triés par n_avis décroissant (un seul passage).
  const childrenOf = useMemo(() => {
    const map = new Map<string | null, SpatialTheme[]>();
    for (const t of themes) {
      const pid = t.parent_id ?? null;
      const arr = map.get(pid);
      if (arr) arr.push(t);
      else map.set(pid, [t]);
    }
    for (const arr of map.values()) arr.sort((a, b) => (b.n_avis ?? 0) - (a.n_avis ?? 0));
    return map;
  }, [themes]);

  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const roots = childrenOf.get(null) ?? [];
  if (!roots.length || total <= 0) return null;

  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  // Aplatit l'arbre en lignes visibles selon l'état d'expansion (DFS pré-ordre).
  const rows: { theme: SpatialTheme; depth: number; pct: number }[] = [];
  const walk = (parentId: string | null, parentTotal: number, depth: number) => {
    const kids = childrenOf.get(parentId) ?? [];
    for (const t of kids) {
      const pct = parentTotal > 0 ? Math.round(((t.n_avis ?? 0) / parentTotal) * 100) : 0;
      rows.push({ theme: t, depth, pct });
      if (t.has_children && expanded.has(t.id)) walk(t.id, t.n_avis ?? 0, depth + 1);
    }
  };
  walk(null, total, 0);

  return (
    <div className="tnav" aria-label="Navigateur de thèmes">
      {rows.map(({ theme: t, depth, pct }) => {
        const name = t.title || t.label;
        const open = expanded.has(t.id);
        return (
          <button
            type="button"
            key={t.id}
            className="tnav__row"
            style={{ paddingLeft: `${depth * 1.1}rem` }}
            aria-expanded={t.has_children ? open : undefined}
            title={name}
            onClick={() => {
              if (t.has_children) toggle(t.id);
              onSelect?.(t.id);
            }}
          >
            <span className="tnav__caret" aria-hidden>
              {t.has_children ? (open ? '▾' : '▸') : ''}
            </span>
            <span className="tnav__label">{name}</span>
            <span className="tnav__track">
              <span className="tnav__fill" style={{ width: `${pct}%` }} />
            </span>
            <span className="tnav__pct">{pct}%</span>
          </button>
        );
      })}
    </div>
  );
}
