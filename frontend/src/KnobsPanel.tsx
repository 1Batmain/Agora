import type { KnobSpec } from './types';

interface Props {
  specs: KnobSpec[];
  values: Record<string, number>;
  disabled: boolean;
  busy: boolean;
  onChange: (key: string, value: number) => void;
  onReset: () => void;
}

/**
 * Knobs panel built from `GET /api/params` (or contract defaults). Each change
 * is debounced upstream (see App) → `POST /api/recluster`. Disabled = read-only
 * fallback when the backend is unreachable.
 */
export function KnobsPanel({ specs, values, disabled, busy, onChange, onReset }: Props) {
  return (
    <section className="panel knobs">
      <header className="panel__head">
        <h2>Knobs</h2>
        <button className="btn" onClick={onReset} disabled={disabled}>
          reset
        </button>
      </header>
      {disabled && <p className="knobs__ro">backend :8010 indisponible — lecture seule</p>}
      <div className="knobs__list" aria-disabled={disabled}>
        {specs.map((s) => {
          const v = values[s.key] ?? s.value;
          return (
            <label className="knob" key={s.key}>
              <span className="knob__row">
                <span className="knob__name">{s.label}</span>
                <span className="knob__val">{format(v, s.step)}</span>
              </span>
              <input
                type="range"
                min={s.min}
                max={s.max}
                step={s.step}
                value={v}
                disabled={disabled}
                onChange={(e) => onChange(s.key, Number(e.target.value))}
              />
              {s.hint && <span className="knob__hint">{s.hint}</span>}
            </label>
          );
        })}
      </div>
      {busy && <div className="knobs__busy">re-clustering…</div>}
    </section>
  );
}

function format(v: number, step: number): string {
  return step < 1 ? v.toFixed(2) : String(Math.round(v));
}
