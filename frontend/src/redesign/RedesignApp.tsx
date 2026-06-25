import { useCallbackRef } from '../useCallbackRef';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchDatasets } from '../api';
import type { Dataset } from '../types';
import type {
  AnalysisPayload,
  BuildProgress,
  Citation,
  DataSource,
  SpatialTheme,
} from './contract';
import { fetchAnalysis, fetchCitations, fetchFlags, fetchInsights } from './analysisApi';
import { SpatialMap } from './SpatialMap';
import { InsightsPanel, type ThemeFlagState } from './InsightsPanel';
import { CitationsPanel } from './CitationsPanel';
import { IndicesDashboard } from './IndicesDashboard';
import { themeCaption } from './labels';

// Right panel width (px) — drag-resizable, persisted, with sane bounds.
const RIGHT_MIN = 300;
const RIGHT_MAX = 760;
const RIGHT_KEY = 'agora.rightWidth';

/**
 * Redesigned "Agora pour députés". DSFR-inspired shell (recoloured orange). The
 * left tool column is gone: the dataset picker lives in the HEADER, and the page
 * SCROLLS vertically — map on top, a dashboard of dataset indices beneath it. The
 * right column (insights → leaf citations) follows the drill level and is
 * drag-resizable. Navigation is an adaptive drill on the bubbles.
 *
 * Single public view: the Députés/Analystes tabs are gone — the site is open to
 * all, with one unified view (no backend/extraction knob in the header).
 */
export default function RedesignApp() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [dataset, setDataset] = useState<string | null>(null);

  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null);
  const [analysisSource, setAnalysisSource] = useState<DataSource | null>(null);
  const [buildProgress, setBuildProgress] = useState<BuildProgress | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const currentParentId = path.length ? path[path.length - 1].id : null;
  const contextTheme = selected ?? (path.length ? path[path.length - 1] : null);
  const showCitations = selected != null && !selected.has_children;
  const atGlobal = path.length === 0 && !selected;

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
      const ds = await fetchDatasets().catch(() => [] as Dataset[]);
      if (cancelled) return;
      // Mock-friendly: if the backend has no datasets, offer a synthetic one so
      // the redesigned UI is navigable offline.
      const list = ds.length ? ds : [{ id: 'demo', label: 'Consultation (démo)', n_nodes: 0, languages: [] }];
      setDatasets(list);
      const first = list[0].id;
      setDataset(first);
      await loadAnalysis(first);
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
    e.currentTarget.releasePointerCapture(e.pointerId);
  }, []);

  const themes = analysis?.themes ?? [];
  const edges = analysis?.edges ?? [];
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
          {/* Dataset picker moved here from the (removed) left tool column. */}
          <label className="header-dataset">
            <span>Consultation</span>
            <select
              className="header-dataset__select"
              value={dataset ?? ''}
              disabled={busy || datasets.length === 0}
              onChange={(e) => onDataset(e.target.value)}
            >
              {datasets.length === 0 && <option value="">(aucun dataset)</option>}
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                  {d.n_nodes ? ` (${d.n_nodes})` : ''}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      <div className="agora__body" style={{ '--right-w': `${rightWidth}px` } as React.CSSProperties}>
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

          {/* F2 — no separate intro block anymore. The collection context is folded
              into the START of the GLOBAL synthesis (right panel), so the global view
              shows a SINGLE synthesis. See `panelMarkdown` below. */}

          <div className="agora__canvas">
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
          </div>

          {/* F8 — dataset indices under the map (graceful when absent). */}
          {!busy && themes.length > 0 && <IndicesDashboard stats={analysis?.dataset_stats} />}

          {error && <p className="agora__error">{error}</p>}
        </main>

        <div
          className="agora__resizer"
          role="separator"
          aria-orientation="vertical"
          aria-label="Redimensionner le panneau"
          onPointerDown={onResizeStart}
          onPointerMove={onResizeMove}
          onPointerUp={onResizeEnd}
        >
          <span className="agora__resizer-grip" />
        </div>

        <aside className="agora__right">
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
        </aside>
      </div>
    </div>
  );
}
