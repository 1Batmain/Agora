import type { ClaimsPayload } from './types';

/**
 * Left-panel controls for the emergent-claims view: a Leiden RESOLUTION slider
 * (granularity of the emergent themes) and a run button. The first run on a cold
 * dataset is long (claims are extracted on the Mac); a resolution change replays
 * from cache and is fast. Minimal on purpose — the console will be redesigned.
 */
export function ClaimsControls({
  resolution,
  onResolution,
  onRun,
  busy,
  disabled,
  payload,
}: {
  resolution: number;
  onResolution: (r: number) => void;
  onRun: () => void;
  busy: boolean;
  disabled: boolean;
  payload: ClaimsPayload | null;
}) {
  const cache = payload?.meta?.cache;
  return (
    <div className="claims-ctrl">
      <div className="panel__head">
        <h2>Thèmes émergents</h2>
      </div>
      <p className="method__hint">
        Les avis sont décomposés en <strong>claims atomiques</strong> (LLM local, souverain) puis
        regroupés du bas — aucune taxonomie imposée.
      </p>
      <div className="knob">
        <div className="knob__row">
          <span className="knob__name">résolution</span>
          <span className="knob__val">{resolution.toFixed(1)}</span>
        </div>
        <input
          type="range"
          min={0.3}
          max={3}
          step={0.1}
          value={resolution}
          disabled={disabled || busy}
          onChange={(e) => onResolution(Number(e.target.value))}
        />
        <p className="knob__hint">granularité : bas = peu de grands thèmes, haut = sous-facettes</p>
      </div>
      <button type="button" className="btn synth__go" disabled={disabled || busy} onClick={onRun}>
        {busy ? 'calcul en cours…' : payload ? 'recalculer' : 'calculer les thèmes'}
      </button>
      {busy && (
        <p className="knobs__busy">
          1er run : extraction des claims sur le Mac (~1,3 s/avis) ; ensuite c'est mis en cache.
        </p>
      )}
      {payload && !busy && (
        <p className="method__hint">
          {payload.themes.length} thèmes · {payload.params?.n_claims ?? '?'} claims ·{' '}
          {payload.params?.n_avis ?? '?'} avis
          {cache
            ? ` · ${cache.claims_extracted ? `${cache.claims_extracted} extraits` : 'cache claims'}`
            : ''}
        </p>
      )}
    </div>
  );
}
