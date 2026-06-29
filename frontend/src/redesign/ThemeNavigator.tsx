import { useEffect, useMemo, useState } from 'react';
import type { SpatialTheme } from './contract';

/** Combien d'enfants on montre avant le bouton « Voir plus ». */
const TOP_N = 5;

/**
 * Navigateur de clusters FLUIDE — le cluster cliqué ne devient JAMAIS un titre :
 * il reste un item de menu, juste MARQUÉ, ÉPINGLÉ EN HAUT, avec ses enfants
 * déroulés/indentés juste en dessous. Le chemin racine→courant reste affiché
 * (chaque ancêtre est un item cliquable) pour ne JAMAIS perdre la trace de où
 * l'on est. Cliquer un enfant l'épingle à son tour + ouvre ses enfants : on
 * descend ainsi la hiérarchie sans rechargement brutal.
 *
 * 100 % FRONT : tout vient du payload `/analysis` (liste plate de thèmes reliés
 * par `parent_id`). Le `%` d'une ligne = `n_avis / n_avis_du_parent` (ou `/total`
 * à la racine). Le composant ne garde en interne QUE l'état « Voir plus » du
 * niveau ; le cluster courant est piloté par le parent (`currentId`/`onSelect`).
 */
export function ThemeNavigator({
  themes,
  total,
  currentId = null,
  onDrill,
  onBack,
  onSelect,
}: {
  /** Liste COMPLÈTE des thèmes du payload (tous niveaux confondus). */
  themes: SpatialTheme[];
  /** Voix du niveau racine (dénominateur des thèmes `parent_id == null`). */
  total: number;
  /** Cluster courant : `null` = racine ; sinon on l'épingle + montre ses enfants. */
  currentId?: string | null;
  /** Épingler/sélectionner un cluster (alias historique de `onSelect`). */
  onDrill?: (themeId: string) => void;
  /** Remonter d'un niveau (vers le parent du cluster courant). */
  onBack?: () => void;
  /** Suivre la synthèse sur le cluster cliqué. */
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

  const byId = useMemo(() => {
    const m = new Map<string, SpatialTheme>();
    for (const t of themes) m.set(t.id, t);
    return m;
  }, [themes]);

  // Seul état interne : « Voir plus » du niveau. On replie quand le niveau change.
  const [showAll, setShowAll] = useState(false);
  useEffect(() => setShowAll(false), [currentId]);

  const roots = childrenOf.get(null) ?? [];
  if (!roots.length || total <= 0) return null;

  const current = currentId != null ? byId.get(currentId) ?? null : null;

  // Chemin racine→courant (inclus) — vide à la racine. Chaque ancêtre reste un
  // item de menu épinglé (jamais un titre).
  const path: SpatialTheme[] = [];
  {
    let c: SpatialTheme | null = current;
    let guard = 0;
    while (c && guard++ < 64) {
      path.unshift(c);
      c = c.parent_id != null ? byId.get(c.parent_id) ?? null : null;
    }
  }

  // Enfants du cluster courant (ou racines à l'accueil), denom = voix du parent.
  const kids = childrenOf.get(currentId ?? null) ?? [];
  const denom = current ? current.n_avis ?? 0 : total;
  const visible = showAll ? kids : kids.slice(0, TOP_N);

  // Dénominateur du % d'un item du chemin = voix de SON parent (ou total racine).
  const pathDenom = (t: SpatialTheme) => {
    const p = t.parent_id != null ? byId.get(t.parent_id) ?? null : null;
    return p ? p.n_avis ?? 0 : total;
  };

  const grandParent =
    current && current.parent_id != null ? byId.get(current.parent_id) ?? null : null;
  const backLabel = grandParent ? grandParent.title || grandParent.label : 'Vue générale';

  const row = (
    t: SpatialTheme,
    rowDenom: number,
    opts: { selected?: boolean; child?: boolean; open?: boolean },
  ) => {
    const name = t.title || t.label;
    const pct = rowDenom > 0 ? Math.round(((t.n_avis ?? 0) / rowDenom) * 100) : 0;
    const caret = opts.open ? '▾' : t.has_children ? '▸' : '';
    return (
      <button
        type="button"
        key={t.id}
        className={
          'tnav__row' +
          (opts.selected ? ' tnav__row--selected' : '') +
          (opts.child ? ' tnav__row--child' : '')
        }
        aria-current={opts.selected ? 'true' : undefined}
        title={name}
        onClick={() => {
          onSelect?.(t.id);
          onDrill?.(t.id);
        }}
      >
        <span className="tnav__caret" aria-hidden>
          {caret}
        </span>
        <span className="tnav__label">{name}</span>
        <span className="tnav__track">
          <span className="tnav__fill" style={{ width: `${pct}%` }} />
        </span>
        <span className="tnav__pct">{pct}%</span>
      </button>
    );
  };

  return (
    <div className="tnav" aria-label="Navigateur de clusters">
      {/* Chemin épinglé en haut : ancêtres + cluster courant marqué (jamais un titre). */}
      {path.length > 0 && (
        <div className="tnav__path">
          <button type="button" className="tnav__back" onClick={() => onBack?.()}>
            ← {backLabel}
          </button>
          {path.map((t) => row(t, pathDenom(t), { selected: t.id === currentId, open: true }))}
        </div>
      )}

      {/* Enfants du courant (ou racines à l'accueil), indentés sous le chemin. */}
      {visible.length > 0 ? (
        visible.map((t) => row(t, denom, { child: path.length > 0 }))
      ) : path.length > 0 ? (
        <p className="tnav__empty">Cluster terminal — aucun sous-cluster.</p>
      ) : null}

      {kids.length > TOP_N && (
        <button type="button" className="tnav__more" onClick={() => setShowAll((s) => !s)}>
          {showAll ? 'Voir moins' : `Voir plus (${kids.length - TOP_N})`}
        </button>
      )}
    </div>
  );
}
