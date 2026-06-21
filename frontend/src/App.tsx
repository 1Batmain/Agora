import { useCallbackRef } from './useCallbackRef';
import { useEffect, useMemo, useRef, useState } from 'react';
import { CirclePack } from './CirclePack';
import { KnobsPanel } from './KnobsPanel';
import { DatasetPicker } from './DatasetPicker';
import { MethodPicker } from './MethodPicker';
import { NamingPicker } from './NamingPicker';
import { StatsBar } from './StatsBar';
import { AvisPanel } from './AvisPanel';
import { SynthesisPanel } from './SynthesisPanel';
import {
  DEFAULT_KNOBS,
  deriveStats,
  fetchDatasets,
  fetchParams,
  fetchStatic,
  recluster,
} from './api';
import type { PackNode } from './hierarchy';
import type { ClusterMethod, Dataset, GraphPayload, KnobSpec, Knobs, NamingMethod } from './types';

const DEBOUNCE_MS = 300;

/**
 * Agora console. On boot it tries the live backend (:8010 via the /api proxy):
 * `GET /api/params` for the knob bounds + `POST /api/recluster` for the first
 * graph. If that fails it falls back to the static graph.json (read-only knobs).
 * Moving a knob debounces (~300 ms) then reclusters and re-renders.
 */
export default function App() {
  const [specs, setSpecs] = useState<KnobSpec[]>(DEFAULT_KNOBS);
  const [values, setValues] = useState<Knobs>(() => knobsFrom(DEFAULT_KNOBS));
  const [payload, setPayload] = useState<GraphPayload | null>(null);
  const [live, setLive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<PackNode | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [dataset, setDataset] = useState<string | null>(null);
  const [method, setMethod] = useState<ClusterMethod>('leiden');
  const [naming, setNaming] = useState<NamingMethod>('ctfidf');

  // Boot: discover datasets, then load the default one's knobs + first graph.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const ds = await fetchDatasets().catch(() => [] as Dataset[]);
        if (cancelled) return;
        setDatasets(ds);
        const first = ds[0]?.id ?? null;
        setDataset(first);
        const ps = await fetchParams(first ?? undefined, 'leiden');
        if (cancelled) return;
        setSpecs(ps);
        const v = knobsFrom(ps);
        setValues(v);
        const g = await recluster(v, first ?? undefined, 'leiden', 'ctfidf');
        if (cancelled) return;
        setPayload(g);
        setLive(true);
      } catch {
        try {
          const g = await fetchStatic();
          if (cancelled) return;
          setPayload(g);
          setLive(false);
        } catch (e) {
          if (!cancelled) setError(`Impossible de charger les données : ${String(e)}`);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Debounced recluster on knob change (live mode only). Always tied to the
  // currently-selected dataset.
  const timer = useRef<number | null>(null);
  const runRecluster = useCallbackRef(
    async (v: Knobs, ds: string | null, m: ClusterMethod, nm: NamingMethod) => {
      setBusy(true);
      try {
        const g = await recluster(v, ds ?? undefined, m, nm);
        setPayload(g);
        setError(null);
      } catch (e) {
        setError(`recluster a échoué : ${String(e)}`);
      } finally {
        setBusy(false);
      }
    },
  );

  function onKnob(key: string, value: number) {
    if (!live) return;
    const next = { ...values, [key]: value };
    setValues(next);
    if (timer.current) clearTimeout(timer.current);
    timer.current = window.setTimeout(() => runRecluster(next, dataset, method, naming), DEBOUNCE_MS);
  }

  function onReset() {
    if (!live) return;
    const v = knobsFrom(specs);
    setValues(v);
    if (timer.current) clearTimeout(timer.current);
    timer.current = window.setTimeout(() => runRecluster(v, dataset, method, naming), DEBOUNCE_MS);
  }

  // Load a (dataset, method) pair: pull THAT combination's derived knobs from
  // /params, reset values, and recluster. Shared by the dataset and method
  // switches — both re-pull knobs because the panel adapts to the method.
  const loadConfig = useCallbackRef(async (id: string | null, m: ClusterMethod, nm: NamingMethod) => {
    setSelected(null);
    setBusy(true);
    if (timer.current) clearTimeout(timer.current);
    try {
      const ps = await fetchParams(id ?? undefined, m);
      setSpecs(ps);
      const v = knobsFrom(ps);
      setValues(v);
      const g = await recluster(v, id ?? undefined, m, nm);
      setPayload(g);
      setError(null);
    } catch (e) {
      setError(`changement de configuration a échoué : ${String(e)}`);
    } finally {
      setBusy(false);
    }
  });

  // Switch dataset (keeps the current method + naming).
  const onDataset = useCallbackRef(async (id: string) => {
    if (!live || id === dataset) return;
    setDataset(id);
    await loadConfig(id, method, naming);
  });

  // Switch clustering method (keeps the current dataset + naming). The knobs
  // panel re-renders with the new method's sliders.
  const onMethod = useCallbackRef(async (m: ClusterMethod) => {
    if (!live || m === method) return;
    setMethod(m);
    await loadConfig(dataset, m, naming);
  });

  // Switch naming method (orthogonal: same clusters/knobs, just re-titled).
  // No knob re-pull — naming doesn't change the slider set.
  const onNaming = useCallbackRef(async (nm: NamingMethod) => {
    if (!live || nm === naming) return;
    setNaming(nm);
    setSelected(null);
    if (timer.current) clearTimeout(timer.current);
    await runRecluster(values, dataset, method, nm);
  });

  const stats = useMemo(() => (payload ? deriveStats(payload) : null), [payload]);

  return (
    <div className="app">
      <aside className="app__left">
        <h1 className="brand">
          Agora <span>· console</span>
        </h1>
        <DatasetPicker
          datasets={datasets}
          current={dataset}
          disabled={!live || busy}
          onChange={onDataset}
        />
        <MethodPicker current={method} disabled={!live || busy} onChange={onMethod} />
        <NamingPicker
          current={naming}
          fallback={stats?.naming_fallback ?? false}
          disabled={!live || busy}
          onChange={onNaming}
        />
        <KnobsPanel
          specs={specs}
          values={values}
          disabled={!live}
          busy={busy}
          onChange={onKnob}
          onReset={onReset}
        />
        <SynthesisPanel dataset={dataset} method={method} naming={naming} disabled={!live || busy} />
        {error && <p className="app__error">{error}</p>}
      </aside>

      <main className="app__center">
        {stats && <StatsBar stats={stats} live={live} />}
        {payload ? (
          <CirclePack payload={payload} onSelect={setSelected} selectedId={selected?.data.id ?? null} />
        ) : (
          <div className="app__loading">{error ?? 'chargement…'}</div>
        )}
      </main>

      <aside className="app__right">
        <AvisPanel selected={selected} />
      </aside>
    </div>
  );
}

function knobsFrom(specs: KnobSpec[]): Knobs {
  return Object.fromEntries(specs.map((s) => [s.key, s.value]));
}
