import { useEffect, useMemo, useState } from 'react';
import type { SpatialTheme } from './contract';

/** Combien d'enfants on montre avant le bouton « Voir plus ». */
const TOP_N = 5;

/**
 * Navigateur de clusters par DRILL (un niveau à la fois) — remplace l'ancien
 * accordéon scrollable. On affiche les enfants du NIVEAU COURANT (`currentId`),
 * triés par nombre d'avis : les 5 plus gros + un bouton « Voir plus » qui déplie
 * tout le niveau (pas de scroll). Cliquer un cluster DESCEND dedans (il devient le
 * niveau courant) ; un contrôle de RETOUR au-dessus remonte d'un niveau.
 *
 * 100 % FRONT : tout vient du payload `/analysis` (liste plate de thèmes reliés
 * par `parent_id`). Le `%` d'une ligne = `n_avis / n_avis_du_parent` (ou `/total`
 * à la racine). Le composant ne garde en interne QUE l'état « Voir plus » du
 * niveau ; le niveau courant est piloté par le parent (`currentId`/`onDrill`/`onBack`).
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
  /** Niveau courant : `null` = racine ; sinon on montre les enfants de ce thème. */
  currentId?: string | null;
  /** Descendre dans un cluster (il devient le niveau courant). */
  onDrill?: (themeId: string) => void;
  /** Remonter d'un niveau (vers le parent du parent ; depuis un macro → racine). */
  onBack?: () => void;
  /** Suivre la synthèse sur le cluster cliqué (peut être identique à `onDrill`). */
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
  const kids = childrenOf.get(currentId ?? null) ?? [];
  // Dénominateur du % : voix du parent courant (ou voix totales à la racine).
  const denom = current ? current.n_avis ?? 0 : total;
  const visible = showAll ? kids : kids.slice(0, TOP_N);

  // En-tête : nom du cluster parent courant. Retour : on remonte vers son parent
  // (le grand-parent) — affiché en libellé si connu, sinon « Vue générale ».
  const parentName = current ? current.title || current.label : null;
  const grandParent =
    current && current.parent_id != null ? byId.get(current.parent_id) ?? null : null;
  const backLabel = grandParent ? grandParent.title || grandParent.label : 'Vue générale';

  return (
    <div className="tnav" aria-label="Navigateur de clusters">
      {currentId != null && (
        <div className="tnav__nav">
          <button type="button" className="tnav__back" onClick={() => onBack?.()}>
            ← {backLabel}
          </button>
          {parentName && (
            <span className="tnav__here" title={parentName}>
              {parentName}
            </span>
          )}
        </div>
      )}

      {visible.length > 0 ? (
        visible.map((t) => {
          const name = t.title || t.label;
          const pct = denom > 0 ? Math.round(((t.n_avis ?? 0) / denom) * 100) : 0;
          return (
            <button
              type="button"
              key={t.id}
              className="tnav__row"
              title={name}
              onClick={() => {
                onSelect?.(t.id);
                onDrill?.(t.id);
              }}
            >
              <span className="tnav__caret" aria-hidden>
                {t.has_children ? '▸' : ''}
              </span>
              <span className="tnav__label">{name}</span>
              <span className="tnav__track">
                <span className="tnav__fill" style={{ width: `${pct}%` }} />
              </span>
              <span className="tnav__pct">{pct}%</span>
            </button>
          );
        })
      ) : (
        <p className="tnav__empty">Cluster terminal — aucun sous-cluster.</p>
      )}

      {kids.length > TOP_N && (
        <button type="button" className="tnav__more" onClick={() => setShowAll((s) => !s)}>
          {showAll ? 'Voir moins' : `Voir plus (${kids.length - TOP_N})`}
        </button>
      )}
    </div>
  );
}
