import { useEffect, useMemo, useState } from 'react';
import type {
  Citation, SpatialTheme, ThemeArguments, ThemeDemographics, ThemeOpinion,
} from './contract';
import { fetchCitations, fetchInsights } from './analysisApi';
import { ArgumentsPanel } from './ArgumentsPanel';
import { OpinionBar } from './OpinionBar';
import { Markdown } from './Markdown';
import { LOCALE, stripMd } from './strings';

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

/** Extrait le corps de la 1re section « ## <heading> » trouvée (teste plusieurs alias
 * pour tolérer l'ANCIEN et le NOUVEAU format de synthèse), ou null. Miroir front de
 * `insights._section_of` : le harness structure la synthèse de thématique en
 * « Vue générale » (identité) puis « À relever » (tensions/consensus), avec la liste des
 * sous-thématiques intercalée entre les deux. */
function sectionOf(md: string | null, headings: string[]): string | null {
  if (!md) return null;
  for (const h of headings) {
    const esc = h.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const m = md.match(new RegExp(`(^|\\n)#{1,4}\\s*${esc}\\b[^\\n]*\\n`, 'i'));
    if (m && m.index != null) {
      const rest = md.slice(m.index + m[0].length);
      const nxt = rest.match(/\n#{1,4}\s/);
      const body = (nxt ? rest.slice(0, nxt.index ?? undefined) : rest).trim();
      if (body) return body;
    }
  }
  return null;
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
  themeArgs = [],
  demographics = [],
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
  /** Arguments minés (artefact OPTIONNEL ; lookup par theme_id, gracieux si absent). */
  themeArgs?: ThemeArguments[];
  /** Profil démographique par thème (artefact OPTIONNEL ; gracieux si absent). */
  demographics?: ThemeDemographics[];
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
      const name = stripMd(t.title || t.label);
      const pct = denom > 0 ? Math.round(((t.n_avis ?? 0) / denom) * 100) : 0;
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
            </span>
            <span className="clout__track" aria-hidden>
              <span className="clout__fill" style={{ width: `${pct}%` }} />
            </span>
          </button>
          {open && (
            <div className="clout__body">
              {/* Rail gauche cliquable : replie la thématique + matérialise l'imbrication. */}
              <button
                type="button"
                className="clout__collapse-rail"
                aria-label={`Replier « ${name} »`}
                title="Replier"
                onClick={() => toggle(t.id)}
              />
              <div className="clout__body-inner">
              <ClusterPanel
                dataset={dataset}
                theme={t}
                opinion={opinions.find((o) => o.theme_id === t.id) ?? null}
                args={themeArgs.find((a) => a.theme_id === t.id) ?? null}
                demog={demographics.find((d) => d.theme_id === t.id) ?? null}
                navTotal={navTotal}
                subclusters={
                  kids.length > 0 ? (
                    <div className="clout__sub">
                      <h3 className="synth-h">Thèmes distincts</h3>
                      <p className="overview__clusters-lead--sub">
                        {kids.length} sous-thématique{kids.length > 1 ? 's' : ''} identifiée
                        {kids.length > 1 ? 's' : ''}.
                      </p>
                      <div className="clout__children">{renderNodes(kids, t.n_avis ?? 0)}</div>
                    </div>
                  ) : null
                }
                onViewGraph={onViewGraph}
                onExploreTheme={onExploreTheme}
                onExploreAvis={onExploreAvis}
              />
              </div>
            </div>
          )}
        </div>
      );
    });

  return (
    <div className="clout" aria-label="Synthèses par thématique (dépliables)">
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
  args,
  demog,
  navTotal,
  subclusters,
  onViewGraph,
  onExploreTheme,
  onExploreAvis,
}: {
  dataset: string;
  theme: SpatialTheme;
  opinion: ThemeOpinion | null;
  /** Arguments minés du thème (artefact optionnel — null = pas de panneau). */
  args: ThemeArguments | null;
  /** Profil démographique du thème (artefact optionnel — null = pas de chip). */
  demog: ThemeDemographics | null;
  navTotal: number;
  /** Liste des sous-thématiques (lead + accordéon imbriqué), intercalée entre le corps
   *  de la synthèse et la section « À retenir ». Null si thématique feuille. */
  subclusters: JSX.Element | null;
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

  // Harness : « Vue générale » (identité) → sous-thématiques → « À relever » (tensions/
  // consensus). Alias inclus pour tolérer l'ancien format tant qu'on n'a pas re-baké.
  const md = stripSaillants(synthesis);
  const vue = sectionOf(md, ['Vue générale', 'Ce que disent les citoyens']);
  const relever = sectionOf(md, ['À relever', 'À retenir']);
  const body = vue ?? (relever ? null : md); // vieux format sans section connue → tout en corps
  const keywords = theme.keywords ?? [];
  const avisN = theme.n_avis ?? 0;
  const pct = navTotal > 0 ? Math.round((avisN / navTotal) * 100) : null;
  const allAvis = citations ?? [];
  const repAvis = (selectedKeyword
    ? allAvis.filter((c) => (c.text || '').toLowerCase().includes(selectedKeyword.toLowerCase()))
    : allAvis
  ).slice(0, selectedKeyword ? 8 : 5);
  // HERO : l'avis au score composite (`theme.hero_avis_id`, calculé au build) s'il est
  // présent ET trouvé dans les citations ; sinon REPLI gracieux sur le 1er représentatif.
  // Un filtre mot-clé actif prime (on montre le meilleur avis mentionnant le mot-clé).
  const heroCit = selectedKeyword
    ? repAvis[0]
    : (theme.hero_avis_id
        ? (allAvis.find((c) => c.avis_id === theme.hero_avis_id) ?? repAvis[0])
        : repAvis[0]);

  return (
    <div className={`overview__dynsynth clout__panel${loading ? ' is-loading' : ''}`} aria-live="polite" aria-busy={loading}>
      {/* HERO — le témoignage le plus représentatif, EN PREMIER. Cliquable → TOUS les
          témoignages de la thématique (explorateur d'avis filtré sur le thème). */}
      {heroCit && (
        <figure
          className="overview__hero"
          role="button"
          tabIndex={0}
          title="Voir tous les témoignages de cette thématique"
          onClick={() => onExploreTheme(theme.id)}
          onKeyDown={(e) => { if (e.key === 'Enter') onExploreTheme(theme.id); }}
        >
          <figcaption className="overview__hero-label">
            Témoignage représentatif{selectedKeyword ? ` · « ${selectedKeyword} »` : ''}
          </figcaption>
          <blockquote className="overview__hero-quote">« {heroCit.text} »</blockquote>
          <span className="overview__hero-more">Voir tous les témoignages →</span>
        </figure>
      )}

      {/* Dashboard de VOLUME de la thématique. */}
      {avisN > 0 && (
        <div className="overview__dash" aria-label="Volume de cette thématique">
          <span className="overview__dash-item">
            <strong>{avisN.toLocaleString(LOCALE)}</strong> témoignages
          </span>
          {pct != null && (
            <span className="overview__dash-item overview__dash-pct">{pct}% du panel</span>
          )}
          {demog && Object.entries(demog.majority).map(([axis, m]) => (
            <span
              key={axis}
              className="overview__dash-item overview__dash-demog"
              title={`Groupe majoritaire (${axis === 'age' ? 'âge' : axis}) parmi les répondants de ce thème — profil déclaré`}
            >
              <strong>{m.label}</strong> {Math.round(m.share * 100)}%
            </span>
          ))}
        </div>
      )}

      {/* VUE GÉNÉRALE — ce qui fait l'identité de la thématique. */}
      {body ? (
        <div className="overview__synthbody">
          <Markdown source={body} />
          {loading && <p className="overview__synthloading">Actualisation…</p>}
        </div>
      ) : loading ? (
        <p className="overview__loading">Chargement de la synthèse…</p>
      ) : !relever ? (
        <p className="overview__loading">Synthèse indisponible.</p>
      ) : null}

      {/* THÈMES DISTINCTS — sous-thématiques (lead « N sous-thématiques identifiées : » +
          accordéon imbriqué), présentées COMME la vue globale, AVANT « À relever ». */}
      {subclusters}

      {/* À RELEVER — ce qui ressort (consensus/clivage, adapté au signal), APRÈS les sous-thématiques. */}
      {relever && (
        <div className="overview__synthbody overview__retenir">
          <Markdown source={`## À relever\n${relever}`} />
        </div>
      )}

      {/* ANALYSE DE STANCE — répartition d'opinion (objet de clivage + fav/déf), À LA FIN
          de la synthèse (thème avec opinion bakée). */}
      {opinion && (
        <div className="clout__stance">
          <OpinionBar
            opinion={opinion}
            onSelectStance={(stance) => onExploreTheme(theme.id, stance)}
          />
        </div>
      )}

      {/* ARGUMENTS MINÉS (artefact OPTIONNEL) : les arguments les plus mis en avant
          pour / contre, chacun sourcé sur des contributions réelles (extraits
          verbatim cliquables). Rendu aussi sans OpinionBar (mode neutre). */}
      {args && <ArgumentsPanel args={args} onExploreAvis={onExploreAvis} />}

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

    </div>
  );
}
