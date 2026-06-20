import { useCallbackRef } from './useCallbackRef';
import { useEffect, useMemo, useRef, useState } from 'react';
import { CirclePack } from './CirclePack';
import { KnobsPanel } from './KnobsPanel';
import { StatsBar } from './StatsBar';
import { AvisPanel } from './AvisPanel';
import { DEFAULT_KNOBS, deriveStats, fetchParams, fetchStatic, recluster } from './api';
import type { PackNode } from './hierarchy';
import type { GraphPayload, KnobSpec, Knobs } from './types';

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

  // Boot.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const ps = await fetchParams();
        if (cancelled) return;
        setSpecs(ps);
        const v = knobsFrom(ps);
        setValues(v);
        const g = await recluster(v);
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

  // Debounced recluster on knob change (live mode only).
  const timer = useRef<number | null>(null);
  const runRecluster = useCallbackRef(async (v: Knobs) => {
    setBusy(true);
    try {
      const g = await recluster(v);
      setPayload(g);
      setError(null);
    } catch (e) {
      setError(`recluster a échoué : ${String(e)}`);
    } finally {
      setBusy(false);
    }
  });

  function onKnob(key: string, value: number) {
    if (!live) return;
    const next = { ...values, [key]: value };
    setValues(next);
    if (timer.current) clearTimeout(timer.current);
    timer.current = window.setTimeout(() => runRecluster(next), DEBOUNCE_MS);
  }

  function onReset() {
    if (!live) return;
    const v = knobsFrom(specs);
    setValues(v);
    if (timer.current) clearTimeout(timer.current);
    timer.current = window.setTimeout(() => runRecluster(v), DEBOUNCE_MS);
  }

  const stats = useMemo(() => (payload ? deriveStats(payload) : null), [payload]);

  return (
    <div className="app">
      <aside className="app__left">
        <h1 className="brand">
          Agora <span>· console</span>
        </h1>
        <KnobsPanel
          specs={specs}
          values={values}
          disabled={!live}
          busy={busy}
          onChange={onKnob}
          onReset={onReset}
        />
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
