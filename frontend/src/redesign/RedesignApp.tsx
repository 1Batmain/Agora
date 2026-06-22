import { useCallbackRef } from '../useCallbackRef';
import { useEffect, useMemo, useState } from 'react';
import { fetchDatasets } from '../api';
import type { Dataset } from '../types';
import type {
  AnalysisPayload,
  Backend,
  BuildProgress,
  Citation,
  DataSource,
  SpatialTheme,
} from './contract';
import { fetchAnalysis, fetchCitations, fetchInsights } from './analysisApi';
import { SpatialMap } from './SpatialMap';
import { InsightsPanel } from './InsightsPanel';
import { CitationsPanel } from './CitationsPanel';
import { ToolsPanel, type ClusterMethod } from './ToolsPanel';

type Tab = 'deputes' | 'analystes';

/** Human badge label per data source (live / build / mock / error). */
const SOURCE_LABEL: Record<DataSource, string> = {
  live: 'backend live',
  building: 'analyse en cours',
  mock: 'données mock',
  error: 'backend indisponible',
};

/**
 * Redesigned "Agora pour députés". DSFR-inspired shell (recoloured orange), two
 * tabs (Députés épuré / Analystes + réglages), 3 columns: tools | spatial map |
 * insights. Navigation is adaptive drill on the map; the right column follows the
 * zoom level (global synthesis → theme synthesis → leaf citations).
 */
export default function RedesignApp() {
  const [tab, setTab] = useState<Tab>('deputes');
  const analyst = tab === 'analystes';

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [dataset, setDataset] = useState<string | null>(null);

  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null);
  const [analysisSource, setAnalysisSource] = useState<DataSource | null>(null);
  const [buildProgress, setBuildProgress] = useState<BuildProgress | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // analyst knobs
  const [backend, setBackend] = useState<Backend>('auto');
  const [method, setMethod] = useState<ClusterMethod>('leiden');
  const [resolution, setResolution] = useState(1.0);

  // filters
  const [query, setQuery] = useState('');
  const [minConsensus, setMinConsensus] = useState(0);

  // navigation: drill path (themes we've descended into) + selected bubble
  const [path, setPath] = useState<SpatialTheme[]>([]);
  const [selected, setSelected] = useState<SpatialTheme | null>(null);

  // right-column content state
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [insightsSource, setInsightsSource] = useState<DataSource | null>(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [citations, setCitations] = useState<Citation[] | null>(null);
  const [citationsSource, setCitationsSource] = useState<DataSource | null>(null);
  const [citationsLoading, setCitationsLoading] = useState(false);

  const currentParentId = path.length ? path[path.length - 1].id : null;
  const contextTheme = selected ?? (path.length ? path[path.length - 1] : null);
  const showCitations = selected != null && !selected.has_children;

  // `poll=true` is a background re-check while the backend is still BUILDING:
  // it must not flash the busy spinner nor reset the user's drill path/selection.
  const loadAnalysis = useCallbackRef(async (ds: string | null, be: Backend, poll = false) => {
    if (!ds) return;
    if (!poll) {
      setBusy(true);
      setError(null);
      setPath([]);
      setSelected(null);
    }
    try {
      const { data, source, progress } = await fetchAnalysis(ds, be);
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
      const ds = await fetchDatasets().catch(() => [] as Dataset[]);
      if (cancelled) return;
      // Mock-friendly: if the backend has no datasets, offer a synthetic one so
      // the redesigned UI is navigable offline.
      const list = ds.length ? ds : [{ id: 'demo', label: 'Consultation (démo)', n_nodes: 0, languages: [] }];
      setDatasets(list);
      const first = list[0].id;
      setDataset(first);
      await loadAnalysis(first, backend);
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
    const t = setTimeout(() => loadAnalysis(dataset, backend, true), 2500);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysisSource, buildProgress, dataset, backend]);

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

  const onDataset = useCallbackRef(async (id: string) => {
    if (id === dataset) return;
    setDataset(id);
    await loadAnalysis(id, backend);
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

  const themes = analysis?.themes ?? [];
  const edges = analysis?.edges ?? [];
  const insightTitle = contextTheme ? contextTheme.label : 'Synthèse globale';

  const crumbs = useMemo(
    () => [{ label: 'Vue globale', idx: -1 }, ...path.map((t, i) => ({ label: t.label, idx: i }))],
    [path],
  );

  const buildLine = buildProgress?.detail
    ? `${buildProgress.detail}${
        buildProgress.total ? ` (${buildProgress.done ?? 0}/${buildProgress.total})` : ''
      }`
    : 'le backend précalcule les thèmes, les citations et les synthèses…';

  return (
    <div className="agora">
      <header className="gov-header">
        <div className="gov-header__brand">
          <div className="gov-logo" aria-hidden>
            <span className="gov-logo__mark">◆</span>
          </div>
          <div className="gov-header__title">
            <strong>Agora</strong>
            <span>Analyse des consultations citoyennes</span>
          </div>
        </div>
        <div className="gov-header__right">
          {analysisSource && (
            <span className={`badge badge--${analysisSource}`}>{SOURCE_LABEL[analysisSource]}</span>
          )}
          <nav className="tabs">
            <button
              className={`tab${tab === 'deputes' ? ' tab--active' : ''}`}
              onClick={() => setTab('deputes')}
            >
              Députés
            </button>
            <button
              className={`tab${tab === 'analystes' ? ' tab--active' : ''}`}
              onClick={() => setTab('analystes')}
            >
              Analystes
            </button>
          </nav>
        </div>
      </header>

      <div className="agora__body">
        <aside className="agora__left">
          <ToolsPanel
            analyst={analyst}
            datasets={datasets}
            dataset={dataset}
            onDataset={onDataset}
            query={query}
            onQuery={setQuery}
            minConsensus={minConsensus}
            onMinConsensus={setMinConsensus}
            backend={backend}
            onBackend={setBackend}
            resolution={resolution}
            onResolution={setResolution}
            method={method}
            onMethod={setMethod}
            onRerun={() => loadAnalysis(dataset, backend)}
            busy={busy}
          />
          {error && <p className="agora__error">{error}</p>}
        </aside>

        <main className="agora__center">
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
          {busy ? (
            <div className="agora__loading">
              <span className="spinner" /> calcul de la carte…
            </div>
          ) : themes.length ? (
            <SpatialMap
              themes={themes}
              edges={edges}
              currentParentId={currentParentId}
              selectedId={selected?.id ?? null}
              onSelect={setSelected}
              onDrill={onDrill}
              query={query}
              minConsensus={minConsensus}
            />
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
        </main>

        <aside className="agora__right">
          {showCitations && selected ? (
            <CitationsPanel
              dataset={dataset}
              themeLabel={selected.label}
              citations={citations}
              loading={citationsLoading}
              source={citationsSource}
              onBack={() => setSelected(null)}
            />
          ) : (
            <InsightsPanel
              title={insightTitle}
              markdown={markdown}
              loading={insightsLoading}
              source={insightsSource}
            />
          )}
        </aside>
      </div>
    </div>
  );
}
