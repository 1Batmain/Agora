import { useMemo } from 'react';
import type { ClaimsPayload, ClaimTheme } from './types';

/**
 * Emergent-themes map (minimal): a list of themes sorted by social weight (bar =
 * weight) and a co-occurrence section (bar thickness = number of avis bridging
 * the two themes). Click a theme to see its representative claims in the right
 * panel. Deliberately list-based, not the circle-pack — the console will be
 * redesigned; this just has to work and stay switchable.
 */
export function ClaimsView({
  payload,
  selectedId,
  onSelect,
}: {
  payload: ClaimsPayload;
  selectedId: number | null;
  onSelect: (t: ClaimTheme | null) => void;
}) {
  const themes = payload.themes;
  const maxWeight = useMemo(() => Math.max(1, ...themes.map((t) => t.weight)), [themes]);
  const nameById = useMemo(() => {
    const m = new Map<number, string>();
    themes.forEach((t) => m.set(t.cluster_id, t.name));
    return m;
  }, [themes]);
  const cooc = payload.cooccurrence ?? [];
  const maxCount = useMemo(() => Math.max(1, ...cooc.map((e) => e.count)), [cooc]);

  if (!themes.length) {
    return <div className="app__loading">Aucun thème — lance le calcul à gauche.</div>;
  }

  return (
    <div className="claims" onClick={() => onSelect(null)}>
      <div className="claims__col">
        <h3 className="claims__h">Thèmes ({themes.length}) — triés par poids social</h3>
        <ul className="claims__list">
          {themes.map((t, i) => {
            const sel = t.cluster_id === selectedId;
            return (
              <li key={t.cluster_id}>
                <button
                  type="button"
                  className={`claims__theme ${sel ? 'is-active' : ''}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelect(sel ? null : t);
                  }}
                >
                  <div className="claims__themehead">
                    <span className="claims__rank">{i + 1}</span>
                    <span className="claims__name" title={t.name}>
                      {t.name}
                    </span>
                    <span className="claims__w">{t.weight}</span>
                  </div>
                  <div className="claims__bar">
                    <div
                      className="claims__barfill"
                      style={{
                        width: `${(t.weight / maxWeight) * 100}%`,
                        background: themeColor(t.cluster_id),
                      }}
                    />
                  </div>
                  <div className="claims__meta">
                    {t.n_claims} claims · {t.n_avis} avis · consensus {t.consensus} · diversité{' '}
                    {t.diversity}
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="claims__col">
        <h3 className="claims__h">Co-occurrence — avis qui pontent deux thèmes</h3>
        {cooc.length ? (
          <ul className="claims__cooc">
            {cooc.slice(0, 30).map((e) => (
              <li key={`${e.a}-${e.b}`} className="claims__coocrow">
                <div className="claims__coocnames">
                  <span title={nameById.get(e.a)}>{truncate(nameById.get(e.a) ?? `#${e.a}`, 28)}</span>
                  <span className="claims__coocsep">↔</span>
                  <span title={nameById.get(e.b)}>{truncate(nameById.get(e.b) ?? `#${e.b}`, 28)}</span>
                  <span className="claims__cooccount">{e.count}</span>
                </div>
                <div className="claims__bar">
                  <div
                    className="claims__barfill"
                    style={{ width: `${(e.count / maxCount) * 100}%`, background: '#6b7afb' }}
                  />
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="avis__empty">
            Aucune co-occurrence : à cette résolution, chaque avis reste dans un seul thème.
          </p>
        )}
      </div>
    </div>
  );
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n - 1) + '…';
}

/** Stable color per cluster_id (golden-angle hue) — no hardcoded palette. */
function themeColor(id: number): string {
  const hue = (id * 137.508) % 360;
  return `hsl(${hue.toFixed(0)} 65% 60%)`;
}
