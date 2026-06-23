import { useEffect, useMemo, useRef, useState } from 'react';
import type { DatasetStats, SpatialTheme } from './contract';
import { openLiveStream } from './liveStream';
import { SpatialMap } from './SpatialMap';
import { IndicesDashboard } from './IndicesDashboard';
import { Markdown } from './Markdown';

/** Sentinel parent so the whole live frontier renders as ONE flat level. */
const LIVE_LEVEL = '__live__';

const FORCE_MOCK = import.meta.env.VITE_FORCE_MOCK === '1';

type LiveStatus = 'connecting' | 'streaming' | 'done' | 'error';

const STATUS: Record<LiveStatus, { label: string; badge: string }> = {
  connecting: { label: 'connexion…', badge: 'building' },
  streaming: { label: 'diffusion en direct', badge: 'building' },
  done: { label: 'replay terminé', badge: 'live' },
  error: { label: 'flux indisponible', badge: 'error' },
};

/**
 * LIVE replay view — the same d3-pack as `/analysis`, but the map is *built before
 * your eyes* from an SSE stream: it starts at the snapshot, bubbles **swell** as
 * voices arrive (`claim_added`), and a theme **divides** into children
 * (`theme_split`) until `done`. Position is non-semantic (deterministic pack); only
 * SIZE / COLOUR / LABEL carry meaning, exactly as in the static view.
 *
 * Source is the real backend `/stream`, with a clean repli on the seeded MOCK
 * stream (auto-fallback if the backend has no live endpoint yet, or on demand via
 * the toggle). "▶ Rejouer" restarts the replay; "Fermer" returns to `/analysis`.
 */
export function LiveView({
  dataset,
  datasetLabel,
  onClose,
}: {
  dataset: string;
  datasetLabel?: string;
  onClose: () => void;
}) {
  const [themes, setThemes] = useState<SpatialTheme[]>([]);
  const [stats, setStats] = useState<DatasetStats | undefined>();
  const [context, setContext] = useState<string | undefined>();
  const [status, setStatus] = useState<LiveStatus>('connecting');
  const [selected, setSelected] = useState<SpatialTheme | null>(null);
  const [useMock, setUseMock] = useState(FORCE_MOCK);
  const [runId, setRunId] = useState(0);

  // Auto-fallback to the mock stream ONCE if the real backend has no /stream yet.
  const autoFellBack = useRef(false);

  useEffect(() => {
    setThemes([]);
    setSelected(null);
    setStatus('connecting');

    const close = openLiveStream(
      dataset,
      {
        onSnapshot: (e) => {
          setThemes(e.themes);
          setStats(e.dataset_stats);
          setContext(e.dataset_context);
          setStatus('streaming');
        },
        onClaimAdded: (e) =>
          setThemes((prev) =>
            prev.map((t) =>
              t.id === e.theme_id
                ? {
                    ...t,
                    n_avis: e.n_avis,
                    n_claims: e.n_claims,
                    weight: e.weight,
                    dispersion: e.dispersion,
                    consensus: e.consensus,
                    convergence: e.convergence ?? t.convergence,
                  }
                : t,
            ),
          ),
        onThemeSplit: (e) =>
          setThemes((prev) => {
            const fresh = e.children.filter((c) => !prev.some((p) => p.id === c.id));
            return [
              ...prev.map((t) => (t.id === e.parent_id ? { ...t, has_children: true } : t)),
              ...fresh,
            ];
          }),
        onDone: () => setStatus('done'),
        onError: () => {
          // Repli propre : if the real stream fails before any data, silently fall
          // back to the seeded mock replay (once) so the view always has something.
          if (!useMock && !autoFellBack.current) {
            autoFellBack.current = true;
            setUseMock(true);
            setRunId((n) => n + 1);
          } else {
            setStatus('error');
          }
        },
      },
      { mock: useMock },
    );
    return close;
  }, [dataset, runId, useMock]); // eslint-disable-line react-hooks/exhaustive-deps

  // The live FRONTIER: themes that have no revealed children yet — i.e. the leaves
  // of the forest built so far. A theme_split moves a parent off the frontier and
  // its children onto it, so the map shows exactly "what is currently a bubble".
  const display = useMemo(() => {
    const hasChild = new Set(
      themes.map((t) => t.parent_id).filter((p): p is string => !!p),
    );
    return themes
      .filter((t) => !hasChild.has(t.id))
      // Flatten to one level (sentinel parent) and neutralise drilling — the live
      // view is a spectacle, not a navigation; clicks just highlight a bubble.
      .map((t) => ({ ...t, parent_id: LIVE_LEVEL, has_children: false }));
  }, [themes]);

  const totalVoices = display.reduce((s, t) => s + t.n_avis, 0);
  const st = STATUS[status];
  const title = datasetLabel ?? dataset;

  const restart = () => {
    autoFellBack.current = false;
    setRunId((n) => n + 1);
  };
  const toggleSource = () => {
    autoFellBack.current = false;
    setUseMock((m) => !m);
    setRunId((n) => n + 1);
  };

  return (
    <div className="agora">
      <header className="gov-header">
        <div className="gov-header__brand">
          <div className="gov-logo" aria-hidden>
            <span className="gov-logo__mark">◆</span>
          </div>
          <div className="gov-header__title">
            <strong>Agora · Direct</strong>
            <span>Construction de la carte en temps réel — {title}</span>
          </div>
        </div>
        <div className="gov-header__right">
          <span className={`badge badge--${st.badge}`}>{st.label}</span>
          {useMock && <span className="badge badge--mock">flux simulé</span>}
          <button className="live-btn" onClick={toggleSource} title="Basculer la source du flux">
            {useMock ? 'Essayer le flux réel' : 'Rejouer en démo'}
          </button>
          <button className="live-btn live-btn--primary" onClick={restart}>
            ▶ Rejouer
          </button>
          <button className="live-btn" onClick={onClose}>
            ✕ Fermer
          </button>
        </div>
      </header>

      <div className="agora__body live-body">
        <main className="agora__center">
          <div className="live-hud">
            <span className="live-hud__metric">
              <strong>{totalVoices.toLocaleString('fr-FR')}</strong> voix
            </span>
            <span className="live-hud__metric">
              <strong>{display.length}</strong> thème{display.length > 1 ? 's' : ''}
            </span>
            {status === 'streaming' && <span className="live-hud__pulse" aria-hidden />}
            <span className="live-hud__hint">
              les bulles grossissent à mesure que les voix arrivent ; un thème se
              divise quand il se diversifie.
            </span>
          </div>

          {context && (
            <p className="dataset-intro__context live-context">
              <span className="dataset-intro__label">Contexte</span>
              {context}
            </p>
          )}

          <div className="agora__canvas">
            {status === 'error' ? (
              <div className="agora__loading agora__build-error">
                <strong>Flux live indisponible</strong>
                <p>
                  le backend n'expose pas (encore) <code>/stream</code>. Utilisez
                  « Rejouer en démo » pour visualiser la construction.
                </p>
              </div>
            ) : display.length ? (
              <SpatialMap
                live
                themes={display}
                edges={[]}
                currentParentId={LIVE_LEVEL}
                selectedId={selected?.id ?? null}
                onSelect={setSelected}
                onDrill={setSelected}
              />
            ) : (
              <div className="agora__loading">
                <span className="spinner" /> en attente du flux…
              </div>
            )}
          </div>

          {display.length > 0 && <IndicesDashboard stats={stats} />}
        </main>

        {selected && (
          <aside className="agora__right live-aside">
            <button className="link-back" onClick={() => setSelected(null)}>
              ← fermer
            </button>
            <h2 className="panel__title">{selected.title?.trim() || selected.label}</h2>
            {selected.hook && <p className="live-aside__hook">{selected.hook}</p>}
            {selected.description && <Markdown source={selected.description} />}
            <p className="panel__empty">
              {selected.n_avis.toLocaleString('fr-FR')} voix
              {typeof selected.convergence === 'number'
                ? ` · convergence ${Math.round(selected.convergence * 100)}%`
                : ` · consensus ${Math.round(selected.consensus * 100)}%`}
            </p>
          </aside>
        )}
      </div>
    </div>
  );
}
