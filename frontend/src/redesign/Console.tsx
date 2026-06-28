import { useEffect, useMemo, useRef, useState } from 'react';
import { useCallbackRef } from '../useCallbackRef';
import type { SpatialTheme } from './contract';
import { fetchRecluster, type ReclusterPayload } from './reclusterApi';
import { Header } from './Header';
import { Density3D } from './Density3D';
import { Scatter2D } from './Scatter2D';
import { SpatialMap } from './SpatialMap';
import { IndicesDashboard } from './IndicesDashboard';
import { themeCaption } from './labels';

/**
 * PAGE « Console » — un atelier de re-clustering LIVE, distinct de la page
 * d'analyse. Un seul potard (« Seuil k-NN ») reconstruit la carte des thèmes à la
 * volée (`POST /recluster`, zéro LLM) ; quatre lectures du même résultat sont
 * offertes côte à côte :
 *
 *   1. Paysage 3D densité (`/density`) — FIXE, indépendant du seuil ;
 *   2. Nuage UMAP 2D (`Scatter2D`) — les `points` colorés par cluster ;
 *   3. Graphe à bulles (`SpatialMap`) — les `themes` du re-cluster ;
 *   4. Tableau d'indices (`IndicesDashboard`) — les `indices` du re-cluster.
 *
 * Les vues lourdes (3D / Nuage / Graphe) partagent une zone à onglets ; le tableau
 * d'indices vit à côté et se recalcule à chaque re-cluster. Le slider pilote
 * Nuage + Graphe + Indices ; le 3D reste fixe (densité PRÉ-clustering).
 */

// Bornes du potard : nombre de voisins k du graphe k-NN (entier). Le défaut vient
// du 1er /recluster (k dérivé du dataset).
const K_MIN = 2;
const K_MAX = 200;          // = plafond backend (Field le=200)
const K_STEP = 1;
const DEBOUNCE_MS = 300;

type Tab = 'landscape' | 'scatter' | 'graph';

export function Console({
  dataset,
  label,
  onHome,
}: {
  dataset: string;
  label?: string;
  onHome?: () => void;
}) {
  const [payload, setPayload] = useState<ReclusterPayload | null>(null);
  const [k, setK] = useState<number | null>(null);
  const [loading, setLoading] = useState(true); // 1er chargement (bloquant)
  const [reclustering, setReclustering] = useState(false); // re-cluster (overlay léger)
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('graph');

  // Navigation dans le graphe : chemin de drill + bulle sélectionnée (local à la Console).
  const [path, setPath] = useState<SpatialTheme[]>([]);
  const [selected, setSelected] = useState<SpatialTheme | null>(null);

  // Re-cluster au seuil `thr` (null au 1er appel → défaut dérivé par le backend).
  const runRecluster = useCallbackRef(async (kVal: number | null, initial = false) => {
    if (initial) setLoading(true);
    else setReclustering(true);
    setError(null);
    try {
      const data = await fetchRecluster(dataset, kVal);
      if (!data) {
        if (initial) setError('re-clustering indisponible pour cette consultation.');
        return;
      }
      setPayload(data);
      // k reconstruit change la hiérarchie → on repart de la vue globale.
      setPath([]);
      setSelected(null);
      if (initial && data.meta.k_default != null) {
        setK(data.meta.k_default);
      }
    } catch (e) {
      if (initial) setError(`re-clustering impossible : ${String(e)}`);
    } finally {
      if (initial) setLoading(false);
      else setReclustering(false);
    }
  });

  // Boot : 1er re-cluster au seuil DÉFAUT (le backend le dérive du dataset).
  useEffect(() => {
    runRecluster(null, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset]);

  // Slider DEBOUNCÉ : bouge en continu (état immédiat) ; le re-cluster part ~300 ms
  // après le dernier mouvement, pour ne pas marteler le backend pendant le glissé.
  const timer = useRef<number | undefined>(undefined);
  const onSlider = (v: number) => {
    setK(v);
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => runRecluster(v), DEBOUNCE_MS);
  };
  useEffect(() => () => window.clearTimeout(timer.current), []);

  const themes = payload?.themes ?? [];
  const points = payload?.points ?? [];
  const currentParentId = path.length ? path[path.length - 1].id : null;

  const crumbs = useMemo(
    () => [
      { label: 'Vue globale', idx: -1 },
      ...path.map((t, i) => ({ label: themeCaption(t), idx: i })),
    ],
    [path],
  );

  function onDrill(t: SpatialTheme) {
    if (!t.has_children) return;
    setPath((p) => [...p, t]);
    setSelected(null);
  }
  function gotoCrumb(idx: number) {
    setPath((p) => (idx < 0 ? [] : p.slice(0, idx + 1)));
    setSelected(null);
  }

  const TABS: { id: Tab; label: string }[] = [
    { id: 'landscape', label: 'Paysage 3D' },
    { id: 'scatter', label: 'Nuage 2D' },
    { id: 'graph', label: 'Graphe' },
  ];

  return (
    <div className="agora">
      <Header
        onHome={onHome}
        right={
          <span className="header-consultation" title="Console — consultation en cours">
            ⚙ Console · {label ?? dataset}
          </span>
        }
      />

      {/* Barre de contrôle : le potard « voisins k » pilote tout le re-clustering. */}
      <div className="console__controls">
        <label className="console__slider" htmlFor="knn-k">
          <span className="console__slider-label">Voisins (k)</span>
          <input
            id="knn-k"
            type="range"
            min={K_MIN}
            max={K_MAX}
            step={K_STEP}
            value={k ?? K_MIN}
            disabled={loading || k == null}
            onChange={(e) => onSlider(Number(e.target.value))}
          />
          <output className="console__slider-value">
            {k != null ? k : '—'}
          </output>
        </label>
        <span className="console__meta">
          {reclustering ? (
            <>
              <span className="spinner" /> re-clustering…
            </>
          ) : payload ? (
            `${payload.meta.n_themes} thèmes · ${payload.meta.n_macros} macros · ${payload.meta.n_ideas} idées`
          ) : null}
        </span>
      </div>

      <div className="console__body">
        {/* Zone principale : onglets pour les 3 vues lourdes. */}
        <main className="console__main">
          <div className="viewtoggle" role="tablist" aria-label="Visualisation">
            {TABS.map((t) => (
              <button
                key={t.id}
                role="tab"
                aria-selected={tab === t.id}
                className={`viewtoggle__tab${tab === t.id ? ' viewtoggle__tab--active' : ''}`}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>

          {tab === 'graph' && (
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

          <div className="console__canvas">
            {loading ? (
              <div className="agora__loading">
                <span className="spinner" /> calcul de la carte…
              </div>
            ) : error ? (
              <div className="agora__loading agora__build-error">
                <strong>Indisponible</strong>
                <p>{error}</p>
              </div>
            ) : (
              <>
                {/* 3D : densité PRÉ-clustering, indépendante du seuil → toujours montée
                    (cachée hors onglet) pour éviter de reconstruire la scène three. */}
                <div hidden={tab !== 'landscape'} className="console__pane">
                  <Density3D dataset={dataset} />
                </div>
                {tab === 'scatter' && <Scatter2D points={points} />}
                {tab === 'graph' &&
                  (themes.length ? (
                    <SpatialMap
                      themes={themes}
                      edges={[]}
                      currentParentId={currentParentId}
                      selectedId={selected?.id ?? null}
                      onSelect={setSelected}
                      onDrill={onDrill}
                      live
                    />
                  ) : (
                    <div className="agora__loading">aucun thème à ce seuil</div>
                  ))}
                {/* Overlay léger pendant le re-cluster (pas de flash, sauf 3D fixe). */}
                {reclustering && tab !== 'landscape' && (
                  <div className="console__reclustering">
                    <span className="spinner" />
                  </div>
                )}
              </>
            )}
          </div>
        </main>

        {/* Tableau d'indices À CÔTÉ — se recalcule à chaque re-cluster. */}
        <aside className="console__indices">
          {payload && payload.indices ? (
            <IndicesDashboard stats={payload.indices} />
          ) : (
            <p className="console__hint">les indices s'afficheront après le 1er calcul.</p>
          )}
        </aside>
      </div>
    </div>
  );
}
