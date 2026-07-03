import { useEffect, useMemo, useState } from 'react';
import type { Citation, SpatialTheme, ThemeOpinion } from './contract';
import { fetchCitations, fetchInsights } from './analysisApi';
import { OpinionBar } from './OpinionBar';
import { Markdown } from './Markdown';
import { LOCALE } from './strings';

/** Retire la section « ## Points saillants » de la synthèse LLM (choix produit : on ne
 * l'affiche plus). De la ligne de titre jusqu'au prochain « ## » (ou la fin). Exporté
 * pour être partagé avec la synthèse GLOBALE (ConsultationOverview). */
export function stripSaillants(md: string | null): string | null {
  if (!md) return md;
  return md
    .replace(/(^|\n)#{1,3}\s*Points?\s+saillants[^\n]*\n[\s\S]*?(?=\n#{1,3}\s|$)/i, '$1')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/**
 * OUTLINE DE CLUSTERS — accordéon récursif « par niveau » affiché SOUS la synthèse
 * globale, sur la même page (plus de navigation qui remplace le contenu). Chaque
 * cluster est une ligne dépliable : l'ouvrir révèle EN PLACE sa synthèse riche
 * (volume + opinion + mots-clés + avis) PUIS ses sous-clusters, eux-mêmes dépliables
 * et indentés → la profondeur se lit à l'œil.
 *
 * Accordéon PAR NIVEAU : un seul frère ouvert à la fois. L'état est un CHEMIN
 * racine→cluster courant (`openPath`) : ouvrir un cluster déplie tout son chemin
 * d'ancêtres et referme les branches voisines. Les synthèses/citations sont chargées
 * PARESSEUSEMENT au montage de chaque panneau (donc au dépliage) — jamais 55 d'un coup.
 */
export function ClusterOutline({
  dataset,
  themes,
  total,
  opinions,
  navTotal,
  openPath,
  onOpenPath,
  onViewGraph,
  onExploreTheme,
  onExploreAvis,
}: {
  dataset: string;
  /** Liste COMPLÈTE des thèmes (tous niveaux, reliés par `parent_id`). */
  themes: SpatialTheme[];
  /** Voix du niveau racine (dénominateur des macros). */
  total: number;
  /** Répartitions d'opinion (lookup par theme_id ; gracieux si absent). */
  opinions: ThemeOpinion[];
  /** Total « panel » pour le % de volume affiché dans chaque panneau. */
  navTotal: number;
  /** Chemin ouvert racine→courant, PILOTÉ par le parent (permet d'ouvrir un cluster
   *  depuis les raccourcis « principaux thèmes » de la synthèse globale). */
  openPath: string[];
  onOpenPath: (path: string[]) => void;
  onViewGraph: (themeId: string | null) => void;
  onExploreTheme: (themeId: string | null, stance?: 'favorable' | 'defavorable' | null) => void;
  onExploreAvis: (avisId: string) => void;
}) {
  // parent_id → enfants triés par n_avis décroissant (les plus gros en tête).
  const childrenOf = useMemo(() => {
    const map = new Map<string | null, SpatialTheme[]>();
    for (const t of themes) {
      const pid = t.parent_id ?? null;
      (map.get(pid) ?? map.set(pid, []).get(pid)!).push(t);
    }
    for (const arr of map.values()) arr.sort((a, b) => (b.n_avis ?? 0) - (a.n_avis ?? 0));
    return map;
  }, [themes]);

  const byId = useMemo(() => {
    const m = new Map<string, SpatialTheme>();
    for (const t of themes) m.set(t.id, t);
    return m;
  }, [themes]);

  // Chemin ouvert racine→courant, piloté par le parent. `isOpen` = présent dans le chemin.
  const ancestorsAndSelf = (id: string): string[] => {
    const chain: string[] = [];
    let cur: SpatialTheme | undefined = byId.get(id);
    let guard = 0;
    while (cur && guard++ < 64) {
      chain.unshift(cur.id);
      cur = cur.parent_id != null ? byId.get(cur.parent_id) : undefined;
    }
    return chain;
  };

  const toggle = (id: string) => {
    const i = openPath.indexOf(id);
    if (i !== -1) onOpenPath(openPath.slice(0, i)); // déjà ouvert → referme (et ses descendants)
    else onOpenPath(ancestorsAndSelf(id)); // sinon → ouvre son chemin (referme les branches voisines)
  };

  // On n'affiche que les TOP_N plus gros clusters par défaut ; « voir plus » déroule le reste.
  const [showAll, setShowAll] = useState(false);
  const TOP_N = 5;

  const roots = childrenOf.get(null) ?? [];
  if (!roots.length || total <= 0) return null;
  const shownRoots = showAll ? roots : roots.slice(0, TOP_N);

  const renderNodes = (nodes: SpatialTheme[], denom: number): JSX.Element[] =>
    nodes.map((t) => {
      const open = openPath.includes(t.id);
      const kids = childrenOf.get(t.id) ?? [];
      const name = t.title || t.label;
      const pct = denom > 0 ? Math.round(((t.n_avis ?? 0) / denom) * 100) : 0;
      const coh = Math.round((t.cohesion ?? t.consensus ?? 0) * 100);
      return (
        <div key={t.id} id={`clout-node-${t.id}`} className={`clout__node${open ? ' clout__node--open' : ''}`}>
          <button
            type="button"
            className="clout__row"
            aria-expanded={open}
            onClick={() => toggle(t.id)}
          >
            <span className="clout__caret" aria-hidden>
              {t.has_children ? (open ? '▾' : '▸') : '•'}
            </span>
            <span className="clout__name">{name}</span>
            <span className="clout__figs">
              <span className="clout__pct">{pct}%</span>
              <span className="clout__voix">{(t.n_avis ?? 0).toLocaleString(LOCALE)} voix</span>
              <span className="clout__coh">cohésion {coh}%</span>
            </span>
            <span className="clout__track" aria-hidden>
              <span className="clout__fill" style={{ width: `${pct}%` }} />
            </span>
          </button>
          {open && (
            <div className="clout__body">
              <ClusterPanel
                dataset={dataset}
                theme={t}
                opinion={opinions.find((o) => o.theme_id === t.id) ?? null}
                navTotal={navTotal}
                onViewGraph={onViewGraph}
                onExploreTheme={onExploreTheme}
                onExploreAvis={onExploreAvis}
              />
              {kids.length > 0 && (
                <div className="clout__children">
                  {renderNodes(kids, t.n_avis ?? 0)}
                </div>
              )}
            </div>
          )}
        </div>
      );
    });

  return (
    <div className="clout" aria-label="Synthèses par cluster (dépliables)">
      {renderNodes(shownRoots, total)}
      {roots.length > TOP_N && (
        <button type="button" className="clout__more" onClick={() => setShowAll((v) => !v)}>
          {showAll
            ? 'Voir moins'
            : `Voir plus — ${roots.length - TOP_N} autres thèmes`}
        </button>
      )}
    </div>
  );
}

/**
 * PANNEAU RICHE d'un cluster déplié — monté à l'ouverture (⇒ fetch paresseux de la
 * synthèse + des avis représentatifs). Reprend le contenu de l'ancienne vue focalisée :
 * dashboard de volume, barre d'opinion (thème feuille), synthèse Markdown, mots-clés
 * cliquables (filtrent les avis), avis représentatifs cliquables, et les accès
 * graphe/témoignages scopés au thème.
 */
function ClusterPanel({
  dataset,
  theme,
  opinion,
  navTotal,
  onViewGraph,
  onExploreTheme,
  onExploreAvis,
}: {
  dataset: string;
  theme: SpatialTheme;
  opinion: ThemeOpinion | null;
  navTotal: number;
  onViewGraph: (themeId: string | null) => void;
  onExploreTheme: (themeId: string | null, stance?: 'favorable' | 'defavorable' | null) => void;
  onExploreAvis: (avisId: string) => void;
}) {
  const [synthesis, setSynthesis] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [citations, setCitations] = useState<Citation[] | null>(null);
  const [selectedKeyword, setSelectedKeyword] = useState<string | null>(null);

  // Fetch PARESSEUX au montage (= au dépliage). Annule proprement si on referme avant la fin.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setSynthesis(null);
    setCitations(null);
    setSelectedKeyword(null);
    fetchInsights(dataset, 'theme', theme.id, theme)
      .catch(() => null)
      .then((s) => {
        if (cancelled) return;
        setSynthesis(s?.data ?? null);
        setLoading(false);
      });
    fetchCitations(dataset, theme.id)
      .catch(() => null)
      .then((r) => {
        if (!cancelled) setCitations(r?.data ?? null);
      });
    return () => {
      cancelled = true;
    };
  }, [dataset, theme.id, theme]);

  const dynSource = stripSaillants(synthesis);
  const keywords = theme.keywords ?? [];
  const avisN = theme.n_avis ?? 0;
  const claimsN = theme.n_claims ?? 0;
  const pct = navTotal > 0 ? Math.round((avisN / navTotal) * 100) : null;
  const allAvis = citations ?? [];
  const repAvis = (selectedKeyword
    ? allAvis.filter((c) => (c.text || '').toLowerCase().includes(selectedKeyword.toLowerCase()))
    : allAvis
  ).slice(0, selectedKeyword ? 8 : 5);

  return (
    <div className={`overview__dynsynth clout__panel${loading ? ' is-loading' : ''}`} aria-live="polite" aria-busy={loading}>
      {/* Dashboard de VOLUME du cluster. */}
      {avisN > 0 && (
        <div className="overview__dash" aria-label="Volume de ce cluster">
          <span className="overview__dash-item">
            <strong>{avisN.toLocaleString(LOCALE)}</strong> témoignages
          </span>
          {pct != null && (
            <span className="overview__dash-item overview__dash-pct">{pct}% du panel</span>
          )}
          {claimsN > 0 && (
            <span className="overview__dash-item">
              <strong>{claimsN.toLocaleString(LOCALE)}</strong> idées
            </span>
          )}
        </div>
      )}

      {/* Répartition d'opinion (thème feuille avec objet de clivage baké). */}
      {opinion && (
        <OpinionBar
          opinion={opinion}
          onSelectStance={(stance) => onExploreTheme(theme.id, stance)}
        />
      )}

      {/* Synthèse Markdown du cluster. */}
      {dynSource ? (
        <div className="overview__synthbody">
          <Markdown source={dynSource} />
          {loading && <p className="overview__synthloading">Actualisation…</p>}
        </div>
      ) : loading ? (
        <p className="overview__loading">Chargement de la synthèse…</p>
      ) : (
        <p className="overview__loading">Synthèse indisponible.</p>
      )}

      {/* Mots-clés cliquables → filtrent les avis représentatifs. */}
      {keywords.length > 0 && (
        <div className="kw-chips kw-chips--clickable" aria-label="Mots-clés — cliquer pour filtrer les avis">
          {keywords.map((kw) => {
            const on = selectedKeyword === kw;
            return (
              <button
                key={kw}
                type="button"
                className={`kw-chip kw-chip--btn${on ? ' kw-chip--on' : ''}`}
                aria-pressed={on}
                title={on ? 'Retirer le filtre' : `Avis mentionnant « ${kw} »`}
                onClick={() => setSelectedKeyword(on ? null : kw)}
              >
                {kw}
              </button>
            );
          })}
        </div>
      )}

      {/* Avis représentatifs (cliquables → exploration focalisée). */}
      {allAvis.length > 0 && (
        <div className="overview__avis">
          <h4 className="synthesis__subhead">
            {selectedKeyword ? `Avis mentionnant « ${selectedKeyword} »` : 'Avis représentatifs'}
            {selectedKeyword && (
              <button type="button" className="overview__kwclear" onClick={() => setSelectedKeyword(null)}>
                × tous
              </button>
            )}
          </h4>
          {repAvis.length > 0 ? (
            repAvis.map((c, i) => {
              const id = c.avis_id;
              return (
                <blockquote
                  key={id ?? i}
                  className={`overview__avis-quote${id ? ' overview__avis-quote--open' : ''}`}
                  role={id ? 'button' : undefined}
                  tabIndex={id ? 0 : undefined}
                  onClick={id ? () => onExploreAvis(id) : undefined}
                  onKeyDown={id ? (e) => { if (e.key === 'Enter') onExploreAvis(id); } : undefined}
                >
                  « {c.text} »
                </blockquote>
              );
            })
          ) : (
            <p className="overview__loading">Aucun avis ne mentionne « {selectedKeyword} ».</p>
          )}
        </div>
      )}

      {/* Accès graphe + explorateur scopés à CE cluster. */}
      <div className="overview__actions">
        <button type="button" className="btn-primary" onClick={() => onViewGraph(theme.id)}>
          Voir le graphe du thème →
        </button>
        <button type="button" className="btn-secondary" onClick={() => onExploreTheme(theme.id)}>
          Consulter les témoignages du thème →
        </button>
      </div>
    </div>
  );
}
