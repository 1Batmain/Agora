import { useMemo } from 'react';
import type { SpatialTheme } from './contract';

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

  const roots = childrenOf.get(null) ?? [];
  if (!roots.length || total <= 0) return null;

  const current = currentId != null ? byId.get(currentId) ?? null : null;

  // Chemin racine→courant (pour le fil d'Ariane / retour).
  const path: SpatialTheme[] = [];
  {
    let c: SpatialTheme | null = current;
    let guard = 0;
    while (c && guard++ < 64) {
      path.unshift(c);
      c = c.parent_id != null ? byId.get(c.parent_id) ?? null : null;
    }
  }

  // Enfants du cluster courant (ou racines à l'accueil), triés par voix décroissantes
  // (les 3 plus gros arrivent donc en tête). Dénominateur du % = voix du parent.
  const kids = childrenOf.get(currentId ?? null) ?? [];
  const denom = current ? current.n_avis ?? 0 : total;

  const grandParent =
    current && current.parent_id != null ? byId.get(current.parent_id) ?? null : null;
  const backLabel = grandParent ? grandParent.title || grandParent.label : 'Vue générale';

  // Une CARTE de cluster : nom · % du parent · nombre de voix · cohésion (0-100).
  const card = (t: SpatialTheme) => {
    const name = t.title || t.label;
    const pct = denom > 0 ? Math.round(((t.n_avis ?? 0) / denom) * 100) : 0;
    const coh = Math.round((t.consensus ?? 0) * 100);
    const selected = t.id === currentId;
    return (
      <button
        type="button"
        key={t.id}
        role="listitem"
        className={`tnav-card${selected ? ' tnav-card--selected' : ''}`}
        aria-current={selected ? 'true' : undefined}
        title={name}
        onClick={() => {
          onSelect?.(t.id);
          onDrill?.(t.id);
        }}
      >
        <span className="tnav-card__name">{name}</span>
        <div className="tnav-card__figs">
          <span className="tnav-card__pct">{pct}%</span>
          <span className="tnav-card__voix">{(t.n_avis ?? 0).toLocaleString('fr-FR')} voix</span>
        </div>
        <span className="tnav-card__track" aria-hidden>
          <span className="tnav-card__fill" style={{ width: `${pct}%` }} />
        </span>
        <span className="tnav-card__coh">cohésion {coh}%</span>
        {t.has_children && <span className="tnav-card__drill" aria-hidden>▸ sous-thèmes</span>}
      </button>
    );
  };

  return (
    <div className="tnav" aria-label="Navigateur de clusters">
      {/* Fil d'Ariane / retour quand on est descendu dans un cluster. */}
      {path.length > 0 && (
        <div className="tnav__crumb">
          <button type="button" className="tnav__back" onClick={() => onBack?.()}>
            ← {backLabel}
          </button>
          {current && <span className="tnav__here">{current.title || current.label}</span>}
        </div>
      )}

      {/* Cartes côte à côte : les 3 plus gros en tête, DÉFILEMENT HORIZONTAL pour tous. */}
      {kids.length > 0 ? (
        <div className="tnav-cards" role="list" aria-label="Clusters (défilement horizontal)">
          {kids.map(card)}
        </div>
      ) : path.length > 0 ? (
        <p className="tnav__empty">Cluster terminal — aucun sous-cluster.</p>
      ) : null}
    </div>
  );
}
