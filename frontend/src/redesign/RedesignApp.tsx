import { useCallbackRef } from '../useCallbackRef';
import { useEffect, useCallback, useMemo, useRef, useState } from 'react';
import { fetchDatasets } from '../api';
import type { Consultation } from './contract';
import type {
  AnalysisPayload,
  BuildProgress,
  Citation,
  DataSource,
  SpatialTheme,
} from './contract';
import { fetchAnalysis, fetchCitations, fetchFlags, fetchInsights } from './analysisApi';
import { fetchScatter, type ScatterPayload, type ScatterRealPoint } from './scatterApi';
import { Header } from './Header';
import { PieChart } from './PieChart';
import { AnswersTable } from './AnswersTable';
import { Scatter3D } from './Scatter3D';
import { InsightsPanel, type ThemeFlagState } from './InsightsPanel';
import { CitationsPanel } from './CitationsPanel';
import { IndicesDashboard } from './IndicesDashboard';
import { themeCaption } from './labels';

/** Mode de visualisation de la carte d'analyse. */
type VizMode = 'graph' | 'scatter3d';
/** État du cache scatter (UMAP précalculé) servant la vue Nuage 3D. */
type ScatterStatus = 'idle' | 'loading' | 'ready' | 'unavailable';

/** Onglets du toggle de visualisation (ordre d'affichage). */
const VIZ_TABS: { id: VizMode; label: string }[] = [
  { id: 'graph', label: 'Graphe' },
  { id: 'scatter3d', label: 'Nuage 3D' },
];

// Right panel width (px) — drag-resizable, persisted, with sane bounds.
const RIGHT_MIN = 300;
const RIGHT_MAX = 760;
const RIGHT_KEY = 'agora.rightWidth';
// Whether the right panel (« Synthèse globale » / citations) is rolled up — persisted
// like the width, so a reader's choice survives navigation/reload.
const RIGHT_COLLAPSED_KEY = 'agora.rightCollapsed';

/**
 * Redesigned "Agora pour députés". DSFR-inspired shell (recoloured orange). The
 * left tool column is gone: the dataset picker lives in the HEADER, and the page
 * SCROLLS vertically — map on top, a dashboard of dataset indices beneath it. The
 * right column (insights → leaf citations) follows the drill level and is
 * drag-resizable. Navigation is an adaptive drill on the bubbles.
 *
 * Single public view: the Députés/Analystes tabs are gone — the site is open to
 * all, with one unified view (no backend/extraction knob in the header).
 *
 * Embedded under the app shell: `initialDataset` selects which consultation to
 * open (else the first discovered), and `onBack` renders a « ← Consultations »
 * link back to the landing grid.
 */
export default function RedesignApp({
  initialDataset = null,
  initialThemeId = null,
  onBack,
  onOpenAvis,
  onAbout,
}: {
  initialDataset?: string | null;
  /** Thème sur lequel OUVRIR l'analyse (bouton « Voir l'analyse du thème ») : on pré-drill. */
  initialThemeId?: string | null;
  onBack?: () => void;
  /** Ouvre l'avis en entier dans l'explorateur (lien « voir l'avis complet » de la table). */
  onOpenAvis?: (avisId: string) => void;
  /** Ouvre la page « À propos » (présentation du projet). */
  onAbout?: () => void;
} = {}) {
  const [datasets, setDatasets] = useState<Consultation[]>([]);
  const [dataset, setDataset] = useState<string | null>(null);

  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null);
  const [analysisSource, setAnalysisSource] = useState<DataSource | null>(null);
  const [buildProgress, setBuildProgress] = useState<BuildProgress | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // navigation: drill path (themes we've descended into) + selected slice (camembert)
  const [path, setPath] = useState<SpatialTheme[]>([]);
  const [selected, setSelected] = useState<SpatialTheme | null>(null);

  // Mode de visualisation de la carte (graphe à bulles · nuage 3D).
  const [vizMode, setVizMode] = useState<VizMode>('graph');
  // Nuage 3D réel (une VRAIE contribution par point) — chargé paresseusement.
  const [scatterPayload, setScatterPayload] = useState<ScatterPayload | null>(null);
  const [scatterStatus, setScatterStatus] = useState<ScatterStatus>('idle');

  // right-column content state
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [insightsSource, setInsightsSource] = useState<DataSource | null>(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [citations, setCitations] = useState<Citation[] | null>(null);
  const [citationsSource, setCitationsSource] = useState<DataSource | null>(null);
  const [citationsLoading, setCitationsLoading] = useState(false);
  // Theme-synthesis flags, keyed by theme id (restored from the dataset-wide /flags).
  const [themeFlags, setThemeFlags] = useState<Record<string, ThemeFlagState>>({});

  // resizable right panel — width persisted in localStorage, clamped to bounds.
  const [rightWidth, setRightWidth] = useState<number>(() => {
    const saved = Number(localStorage.getItem(RIGHT_KEY));
    return Number.isFinite(saved) && saved > 0
      ? Math.max(RIGHT_MIN, Math.min(RIGHT_MAX, saved))
      : 380;
  });
  useEffect(() => {
    localStorage.setItem(RIGHT_KEY, String(rightWidth));
  }, [rightWidth]);

  // Roll/unroll the WHOLE right panel — a horizontal slide (the panel's content
  // keeps its natural width and gets progressively CLIPPED from its left edge as
  // the panel rolls up, so it visibly "empties" left→right rather than an
  // instant top-to-bottom show/hide of its content). `rightDragging` briefly
  // disables the transition while the user is actively resizing (drag must
  // track the pointer 1:1, not ease).
  const [rightCollapsed, setRightCollapsed] = useState<boolean>(
    () => localStorage.getItem(RIGHT_COLLAPSED_KEY) === '1',
  );
  const [rightDragging, setRightDragging] = useState(false);
  useEffect(() => {
    localStorage.setItem(RIGHT_COLLAPSED_KEY, rightCollapsed ? '1' : '0');
  }, [rightCollapsed]);

  const currentParentId = path.length ? path[path.length - 1].id : null;
  const contextTheme = selected ?? (path.length ? path[path.length - 1] : null);
  const showCitations = selected != null && !selected.has_children;
  const atGlobal = path.length === 0 && !selected;

  // Pré-drill sur `initialThemeId` (bouton « Voir l'analyse du thème ») : dès que
  // l'analyse est chargée, on reconstruit le chemin racine→thème. Un thème à enfants →
  // on DESCEND dedans (ses enfants s'affichent) ; une feuille → on la SÉLECTIONNE à son
  // niveau parent (ses citations s'ouvrent). Appliqué UNE seule fois.
  const focusApplied = useRef(false);
  useEffect(() => {
    if (focusApplied.current || !initialThemeId) return;
    const nodes = analysis?.themes;
    if (!nodes || !nodes.length) return;
    const byId = new Map(nodes.map((t) => [t.id, t]));
    const target = byId.get(initialThemeId);
    if (!target) return;
    const chain: SpatialTheme[] = [];
    let cur: SpatialTheme | undefined = target;
    let guard = 0;
    while (cur && guard++ < 64) {
      chain.unshift(cur);
      cur = cur.parent_id ? byId.get(cur.parent_id) : undefined;
    }
    if (target.has_children) {
      setPath(chain);
      setSelected(null);
    } else {
      setPath(chain.slice(0, -1));
      setSelected(target);
    }
    focusApplied.current = true;
  }, [initialThemeId, analysis]);

  // Template de synthèse unifié : mots-clés représentatifs + « sondage » des sous-thèmes
  // dominants (barres de %) du niveau courant. Global → macros ; thème → ses enfants.
  const levelKeywords = contextTheme
    ? contextTheme.keywords
    : (analysis?.dataset_stats as { keywords?: string[] } | undefined)?.keywords;
  // Dénominateur racine du navigateur : voix totales (= somme des thèmes racine).
  const themesTotal = useMemo(() => {
    const all = analysis?.themes ?? [];
    const totals = (analysis?.dataset_stats as { totals?: Record<string, number> } | undefined)?.totals ?? {};
    const roots = all.filter((t) => !t.parent_id);
    return (totals.participants ?? totals.n_avis ?? roots.reduce((s, t) => s + (t.n_avis ?? 0), 0)) || 0;
  }, [analysis]);

  // `poll=true` is a background re-check while the backend is still BUILDING:
  // it must not flash the busy spinner nor reset the user's drill path/selection.
  const loadAnalysis = useCallbackRef(async (ds: string | null, poll = false) => {
    if (!ds) return;
    if (!poll) {
      setBusy(true);
      setError(null);
      setPath([]);
      setSelected(null);
    }
    try {
      const { data, source, progress } = await fetchAnalysis(ds);
      setAnalysisSource(source);
      setBuildProgress(progress ?? null);
      if (data) setAnalysis(data);
      else if (!poll) setAnalysis(null);
    } catch (e) {
      setError(`chargement de la carte impossible : ${String(e)}`);
    } finally {
      if (!poll) setBusy(false);
    }
  });

  // Boot: discover datasets then load the first one's map.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const ds = await fetchDatasets().catch(() => [] as Consultation[]);
      if (cancelled) return;
      // Mock-friendly: if the backend has no datasets, offer a synthetic one so
      // the redesigned UI is navigable offline.
      const list: Consultation[] = ds.length ? ds : [{
        id: 'demo', label: 'Consultation (démo)', status: 'closed',
        n_sample: 0, n_contributions: 0, n_nodes: 0,
        languages: [], lang_counts: {}, source: 'demo',
      }];
      setDatasets(list);
      // Open the consultation requested by the shell (landing card), else the first.
      const start = (initialDataset && list.some((d) => d.id === initialDataset))
        ? initialDataset
        : list[0].id;
      setDataset(start);
      await loadAnalysis(start);
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll while the backend is BUILDING: re-check every few seconds until the real
  // analysis is ready. `buildProgress` gets a fresh object each poll, so this effect
  // re-arms itself; it stops as soon as the source flips away from 'building'.
  useEffect(() => {
    if (analysisSource !== 'building' || !dataset) return;
    const t = setTimeout(() => loadAnalysis(dataset, true), 2500);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysisSource, buildProgress, dataset]);

  // Insights effect — follows the zoom level (skipped when showing citations).
  useEffect(() => {
    if (!dataset || showCitations) return;
    let cancelled = false;
    setInsightsLoading(true);
    const level = contextTheme ? 'theme' : 'global';
    fetchInsights(dataset, level, contextTheme?.id, contextTheme ?? undefined)
      .then(({ data, source }) => {
        if (cancelled) return;
        setMarkdown(data);
        setInsightsSource(source);
      })
      .finally(() => !cancelled && setInsightsLoading(false));
    return () => {
      cancelled = true;
    };
  }, [dataset, contextTheme?.id, showCitations]); // eslint-disable-line react-hooks/exhaustive-deps

  // Citations effect — only when a leaf is selected.
  useEffect(() => {
    if (!dataset || !showCitations || !selected) return;
    let cancelled = false;
    setCitationsLoading(true);
    fetchCitations(dataset, selected.id)
      .then(({ data, source }) => {
        if (cancelled) return;
        setCitations(data);
        setCitationsSource(source);
      })
      .finally(() => !cancelled && setCitationsLoading(false));
    return () => {
      cancelled = true;
    };
  }, [dataset, showCitations, selected?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Restore theme-synthesis flags when the dataset changes, so the « Signaler »
  // button reflects the persisted state at load (avis flags load in CitationsPanel).
  useEffect(() => {
    if (!dataset) return;
    let cancelled = false;
    fetchFlags(dataset).then((list) => {
      if (cancelled) return;
      const map: Record<string, ThemeFlagState> = {};
      for (const f of list) {
        if (f.target_type === 'theme') map[f.target_id] = { category: f.category, text: f.text };
      }
      setThemeFlags(map);
    });
    return () => {
      cancelled = true;
    };
  }, [dataset]);

  // Changement de dataset → on repart sur le graphe et on invalide le cache
  // scatter (rechargement paresseux à la prochaine sélection).
  useEffect(() => {
    setVizMode('graph');
    setScatterPayload(null);
    setScatterStatus('idle');
  }, [dataset]);

  // Chargement PARESSEUX du nuage de points RÉEL (une vraie contribution par point).
  // Déclenché quand l'utilisateur bascule sur « Nuage 3D ». Lit `/scatter` qui sert
  // les coords UMAP 2D réelles + cluster/couleur depuis le cache précalculé.
  useEffect(() => {
    if (!dataset) return;
    if (vizMode !== 'scatter3d') return;
    if (scatterStatus !== 'idle') return;

    let cancelled = false;
    setScatterStatus('loading');
    fetchScatter(dataset)
      .then((data) => {
        if (cancelled) return;
        if (data && data.points.length > 0) {
          setScatterPayload(data);
          setScatterStatus('ready');
        } else {
          setScatterStatus('unavailable');
        }
      })
      .catch(() => {
        if (cancelled) return;
        setScatterStatus('unavailable');
      });
    return () => {
      cancelled = true;
    };
    // Do NOT add scatterStatus to deps — same pattern as the old density fetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, vizMode]);

  // Vrais points scatter (une contribution par point, coords UMAP réelles),
  // FILTRÉS par le contexte de navigation (thème sélectionné ou drill path).
  // Si on est sur "Vue globale" → tous les points ; si on a descendu dans un
  // thème → seulement les points des feuilles de ce sous-arbre.
  //
  // RECOLORATION : les couleurs du backend sont par MACRO (toutes les feuilles
  // d'un macro ont la même couleur). Quand on descend dans un thème, on recolore
  // chaque point selon le sous-thème ENFANT DIRECT qui le contient, en générant
  // une palette distincte (HSL réparti sur le cercle) pour les distinguer.
  const realScatterPoints: ScatterRealPoint[] = useMemo(() => {
    const all = scatterPayload?.points ?? [];
    if (!all.length) return all;

    // Pas de drill = vue globale → tous les points, couleurs d'origine.
    const contextId = selected?.id ?? (path.length ? path[path.length - 1].id : null);
    if (!contextId) return all;

    const themes = analysis?.themes ?? [];
    if (!themes.length) return all;

    const childrenMap = new Map<string | null, SpatialTheme[]>();
    for (const t of themes) {
      const arr = childrenMap.get(t.parent_id) ?? [];
      arr.push(t);
      childrenMap.set(t.parent_id, arr);
    }

    // Enfants DIRECTS du contexte courant (c'est par eux qu'on recolore).
    const directChildren = childrenMap.get(contextId) ?? [];
    // Si le contexte est une feuille → pas de recoloration (un seul groupe).
    const contextNode = themes.find((t) => t.id === contextId);
    if (contextNode && !contextNode.has_children) {
      return all.filter((p) => p.cluster_id === contextId);
    }

    // Map leaf_id → enfant direct qui le contient.
    const leafToDirectChild = new Map<string, string>();
    for (const child of directChildren) {
      const stack = [child];
      while (stack.length) {
        const t = stack.pop()!;
        if (t.has_children) {
          stack.push(...(childrenMap.get(t.id) ?? []));
        } else {
          leafToDirectChild.set(t.id, child.id);
        }
      }
    }

    // Palette distincte pour les enfants directs (HSL réparti sur le cercle).
    const childColorMap = new Map<string, string>();
    const nChildren = directChildren.length || 1;
    for (let i = 0; i < directChildren.length; i++) {
      const hue = Math.round((i * 360) / nChildren);
      childColorMap.set(directChildren[i].id, `hsl(${hue}, 65%, 50%)`);
    }

    // Filtr + recoloration.
    const leafIds = new Set(leafToDirectChild.keys());
    if (!leafIds.size) return all;
    return all
      .filter((p) => p.cluster_id != null && leafIds.has(p.cluster_id))
      .map((p) => ({
        ...p,
        color: childColorMap.get(leafToDirectChild.get(p.cluster_id!)!) ?? p.color,
      }));
  }, [scatterPayload, analysis, selected, path]);

  // Map cluster_id (feuille) → ID du sous-thème enfant direct qui le contient.
  // Sert au Scatter3D pour le hover de groupe + le double-clic drill.
  const clusterToChildTheme = useMemo(() => {
    const themes = analysis?.themes ?? [];
    if (!themes.length) return new Map<string, string>();
    // Vue globale (contextId null) → enfants = thèmes racines (parent_id null).
    const contextId = selected?.id ?? (path.length ? path[path.length - 1].id : null);
    const childrenMap = new Map<string | null, SpatialTheme[]>();
    for (const t of themes) {
      const arr = childrenMap.get(t.parent_id) ?? [];
      arr.push(t);
      childrenMap.set(t.parent_id, arr);
    }
    // directChildren = enfants du contexte, ou racines si vue globale.
    const directChildren = contextId
      ? (childrenMap.get(contextId) ?? [])
      : (childrenMap.get(null) ?? []);
    const contextNode = contextId ? themes.find(t => t.id === contextId) : null;
    if (contextNode && !contextNode.has_children) {
      const m = new Map<string, string>();
      m.set(contextId!, contextId!);
      return m;
    }
    const m = new Map<string, string>();
    for (const child of directChildren) {
      const stack = [child];
      while (stack.length) {
        const t = stack.pop()!;
        if (t.has_children) stack.push(...(childrenMap.get(t.id) ?? []));
        else m.set(t.id, child.id);
      }
    }
    return m;
  }, [analysis, selected, path]);

  // Thèmes enfants DIRECTS du contexte courant.
  const childThemesForScatter: SpatialTheme[] = useMemo(() => {
    const themes = analysis?.themes ?? [];
    if (!themes.length) return [];
    const contextId = selected?.id ?? (path.length ? path[path.length - 1].id : null);
    if (selected && !selected.has_children) return [];
    // Vue globale (contextId null) → thèmes racines (parent_id null).
    return themes.filter(t => t.parent_id === contextId);
  }, [analysis, selected, path]);

  // Handler de drill pour le Nuage 3D : trouve le thème et descend.
  const onScatterDrill = useCallback((themeId: string) => {
    const themes = analysis?.themes ?? [];
    const t = themes.find(x => x.id === themeId);
    if (t && t.has_children) onDrill(t);
  }, [analysis]);
  const onDataset = useCallbackRef(async (id: string) => {
    if (id === dataset) return;
    setDataset(id);
    await loadAnalysis(id);
  });

  function onDrill(t: SpatialTheme) {
    if (!t.has_children) return;
    setPath((p) => [...p, t]);
    setSelected(null);
  }

  function gotoCrumb(index: number) {
    // index -1 = global; otherwise pop to that depth
    setPath((p) => (index < 0 ? [] : p.slice(0, index + 1)));
    setSelected(null);
  }

  // --- right-panel resize: drag the handle, clamp, persist on release. ---
  const dragging = useRef(false);
  const onResizeStart = useCallback((e: React.PointerEvent) => {
    dragging.current = true;
    setRightDragging(true);
    e.currentTarget.setPointerCapture(e.pointerId);
  }, []);
  const onResizeMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current) return;
    // panel hugs the right edge → width = distance from cursor to viewport right.
    const w = window.innerWidth - e.clientX;
    setRightWidth(Math.max(RIGHT_MIN, Math.min(RIGHT_MAX, w)));
  }, []);
  const onResizeEnd = useCallback((e: React.PointerEvent) => {
    dragging.current = false;
    setRightDragging(false);
    e.currentTarget.releasePointerCapture(e.pointerId);
  }, []);

  const themes = analysis?.themes ?? [];
  const insightTitle = contextTheme ? themeCaption(contextTheme) : 'Synthèse globale';

  // F2 — the collection context is the START of the GLOBAL synthesis (one
  // synthesis, no separate intro block). The backend (B2) ideally already folds it
  // in; this is the FRONT repli: at the global level, if the synthesis doesn't
  // already open with the context, prefix it as the first paragraph.
  const panelMarkdown = useMemo(() => {
    const ctx = analysis?.dataset_context?.trim();
    if (!atGlobal || !ctx || !markdown) return markdown;
    // Ignore les marqueurs d'emphase Markdown (_italique_/**gras**) en tête : le
    // backend B2 plie le contexte EN ITALIQUE (`_ctx_`), donc sans ça la dédup ratait
    // (`_x-stance` ne « startsWith » pas `x-stance`) → contexte affiché en double.
    const norm = (s: string) => s.replace(/[_*]/g, '').replace(/\s+/g, ' ').trim().toLowerCase();
    return norm(markdown).startsWith(norm(ctx)) ? markdown : `${ctx}\n\n${markdown}`;
  }, [atGlobal, analysis?.dataset_context, markdown]);

  const crumbs = useMemo(
    () => [
      { label: 'Vue globale', idx: -1 },
      ...path.map((t, i) => ({ label: themeCaption(t), idx: i })),
    ],
    [path],
  );

  const buildLine = buildProgress?.detail
    ? `${buildProgress.detail}${
        buildProgress.total ? ` (${buildProgress.done ?? 0}/${buildProgress.total})` : ''
      }`
    : 'le backend précalcule les thèmes, les citations et les synthèses…';

  return (
    <div className="agora">
      <Header
        onHome={onBack}
        onAbout={onAbout}
        right={
          <span className="header-consultation" title="Consultation en cours">
            {datasets.find((d) => d.id === dataset)?.label ?? dataset}
          </span>
        }
      />

      <div
        className="agora__body"
        style={{ '--right-w': `${rightCollapsed ? 0 : rightWidth}px` } as React.CSSProperties}
      >
        <main className="agora__center">
          {/* Fil d'Ariane du drill (camembert → sous-camembert → …). */}
          {!busy && themes.length > 0 && (
            <div className="viewtoggle" role="tablist" aria-label="Mode de visualisation">
              {VIZ_TABS.map((t) => {
                return (
                  <button
                    key={t.id}
                    role="tab"
                    aria-selected={vizMode === t.id}
                    className={`viewtoggle__tab${vizMode === t.id ? ' viewtoggle__tab--active' : ''}`}
                    onClick={() => setVizMode(t.id)}
                  >
                    {t.label}
                  </button>
                );
              })}
            </div>
          )}

          {/* Fil d'Ariane = navigation du GRAPHE et du Nuage 3D (le drill
              filtre le nuage par sous-thème). */}
          {(vizMode === 'graph' || vizMode === 'scatter3d') && (
            <nav className="breadcrumb">
              {crumbs.map((c, i) => (
                <span key={c.idx}>
                  {i > 0 && <span className="breadcrumb__sep">›</span>}
                  <button
                    className={`breadcrumb__item${i === crumbs.length - 1 ? ' breadcrumb__item--active' : ''}`}
                    onClick={() => gotoCrumb(c.idx)}
                  >
                    {c.label}
                  </button>
                </span>
              ))}
            </nav>
          )}

          {/* F2 — no separate intro block anymore. The collection context is folded
              into the START of the GLOBAL synthesis (right panel), so the global view
              shows a SINGLE synthesis. See `panelMarkdown` below. */}

          <div className="agora__canvas">
            {busy ? (
              <div className="agora__loading">
                <span className="spinner" /> calcul de la carte…
              </div>
            ) : themes.length ? (
              vizMode === 'graph' ? (
                <PieChart
                  themes={themes}
                  currentParentId={currentParentId}
                  selectedId={selected?.id ?? null}
                  onSelect={setSelected}
                  onDrill={onDrill}
                />
              ) : vizMode === 'scatter3d' ? (
                scatterStatus === 'loading' || scatterStatus === 'idle' ? (
                  <div className="agora__loading">
                    <span className="spinner" /> chargement du nuage 3D…
                  </div>
                ) : scatterStatus === 'unavailable' || !scatterPayload ? (
                  <div className="agora__loading agora__build-error">
                    <strong>Nuage 3D indisponible</strong>
                    <p>la projection UMAP n'est pas précalculée pour cette consultation.</p>
                  </div>
                ) : (
                  <Scatter3D
                    points={realScatterPoints}
                    clusterToThemeId={clusterToChildTheme}
                    childThemes={childThemesForScatter}
                    onDrill={onScatterDrill}
                  />
                )
              ) : null
            ) : analysisSource === 'building' ? (
              <div className="agora__loading agora__building">
                <span className="spinner" />
                <strong>Analyse en cours…</strong>
                <p>{buildLine}</p>
              </div>
            ) : analysisSource === 'error' ? (
              <div className="agora__loading agora__build-error">
                <strong>Backend indisponible</strong>
                <p>{buildProgress?.error || error || "l'analyse n'a pas pu être chargée."}</p>
              </div>
            ) : (
              <div className="agora__loading">{error ?? 'aucune donnée'}</div>
            )}
          </div>

          {/* F8 — dataset indices under the map (graceful when absent). */}
          {!busy && themes.length > 0 && <IndicesDashboard stats={analysis?.dataset_stats} />}

          {/* F9 — table des réponses des citoyens qui composent le cluster courant
              (obligatoire : quel que soit le niveau de drill, on voit les avis concrets). */}
          {!busy && themes.length > 0 && dataset && (
            <AnswersTable
              dataset={dataset}
              themeId={contextTheme?.id ?? null}
              title={contextTheme ? `Avis — ${themeCaption(contextTheme)}` : 'Avis — toute la consultation'}
              onOpenAvis={onOpenAvis}
            />
          )}

          {error && <p className="agora__error">{error}</p>}
        </main>

        {/* `agora__divider` : conteneur PARTAGÉ entre la zone de drag (resizer) et le
            bouton roll/unroll — les deux sont des ENFANTS SÉPARÉS (siblings), jamais
            l'un dans l'autre. Le bouton avait été placé DANS le resizer, dont le
            `pointerdown` appelle `setPointerCapture` (drag) : même avec
            `stopPropagation`, ce genre d'imbrication reste fragile d'un navigateur à
            l'autre — les siblings l'évitent une fois pour toutes. */}
        <div className={`agora__divider${rightCollapsed ? ' agora__divider--collapsed' : ''}`}>
          <div
            className="agora__resizer"
            role="separator"
            aria-orientation="vertical"
            aria-label="Redimensionner le panneau"
            onPointerDown={rightCollapsed ? undefined : onResizeStart}
            onPointerMove={rightCollapsed ? undefined : onResizeMove}
            onPointerUp={rightCollapsed ? undefined : onResizeEnd}
          >
            {!rightCollapsed && <span className="agora__resizer-grip" />}
          </div>
          <button
            type="button"
            className="agora__right-toggle"
            aria-expanded={!rightCollapsed}
            title={rightCollapsed ? 'Afficher le panneau de synthèse' : 'Masquer le panneau de synthèse'}
            onClick={() => setRightCollapsed((c) => !c)}
          >
            <span aria-hidden>{rightCollapsed ? '‹' : '›'}</span>
          </button>
        </div>

        <aside
          className={`agora__right${rightCollapsed ? ' agora__right--collapsed' : ''}${
            rightDragging ? ' agora__right--dragging' : ''
          }`}
          aria-hidden={rightCollapsed}
        >
          <div className="agora__right-inner" style={{ width: rightWidth }}>
            {showCitations && selected ? (
              <CitationsPanel
                dataset={dataset}
                themeLabel={themeCaption(selected)}
                themeColor={selected.color}
                hook={selected.hook}
                description={selected.description}
                convergence={selected.convergence}
                citations={citations}
                loading={citationsLoading}
                source={citationsSource}
                onBack={() => setSelected(null)}
              />
            ) : (
              <InsightsPanel
                title={insightTitle}
                markdown={panelMarkdown}
                loading={insightsLoading}
                source={insightsSource}
                keywords={levelKeywords}
                themes={themes}
                themesTotal={themesTotal}
                navCurrentId={contextTheme?.id ?? null}
                onSelectTheme={(id) => {
                  const t = themes.find((x) => x.id === id);
                  if (t) setSelected(t);
                }}
                onDrillTheme={(id) => {
                  const t = themes.find((x) => x.id === id);
                  if (t) onDrill(t);
                }}
                onBackTheme={() => {
                  setSelected(null);
                  setPath((p) => p.slice(0, -1));
                }}
                flagTarget={
                  dataset && contextTheme
                    ? {
                        dataset,
                        themeId: contextTheme.id,
                        // depth of the synthesised theme: a selected bubble sits at the
                        // current level (path.length); a drilled-into theme one above.
                        layer: selected ? path.length : path.length - 1,
                        flag: themeFlags[contextTheme.id],
                        onChange: (id, flag) =>
                          setThemeFlags((prev) => {
                            const next = { ...prev };
                            if (flag) next[id] = flag;
                            else delete next[id];
                            return next;
                          }),
                      }
                    : undefined
                }
              />
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
