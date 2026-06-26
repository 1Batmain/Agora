import type { DatasetStat, DatasetStats } from './contract';
import { INDEX_LABELS, indexExplanation, LOCALE, type IndexDetail } from './strings';

/**
 * F8 — dashboard of dataset-level INDICES under the map. Renders `dataset_stats`
 * from `/analysis` as intuitive cards/gauges (label + value + short explanation).
 *
 * The backend ships PURE DATA ({key, value, detail}); ALL copy (labels +
 * explanations) lives in `strings.ts` — the single source of truth. This module
 * holds only rendering logic.
 *
 * Robust to two backend shapes (the contract is permissive): a ready list of
 * `DatasetStat` objects, OR a loose `{key: number}` record. Both are keyed into
 * `strings.ts`; unknown keys are humanised so the panel never breaks on an
 * unexpected field. Absent/empty → renders nothing.
 */
export function IndicesDashboard({ stats }: { stats?: DatasetStats }) {
  const cards = normalise(stats);
  if (!cards.length) return null;
  return (
    <section className="dashboard" aria-label="Indices du dataset">
      <h3 className="dashboard__title">Indices de la consultation</h3>
      <div className="dashboard__grid">
        {cards.map((c) => (
          <article className="indic" key={c.key}>
            <header className="indic__head">
              <span className="indic__label">{c.label}</span>
              <strong className="indic__value">{c.display ?? fmt(c.value)}</strong>
            </header>
            {c.gauge != null && (
              <div className="indic__gauge" aria-hidden>
                <span
                  className="indic__gaugefill"
                  style={{ width: `${Math.round(clamp01(c.gauge) * 100)}%` }}
                />
              </div>
            )}
            {c.hint && <p className="indic__hint">{c.hint}</p>}
          </article>
        ))}
      </div>
    </section>
  );
}

type Card = DatasetStat;

/** Build a display card from a backend `{key, value, detail}` indicator. */
function toCard(key: string, value: number, detail?: IndexDetail): Card {
  const isPct = value >= 0 && value <= 1;
  return {
    key,
    label: INDEX_LABELS[key] ?? humanise(key),
    value,
    display: isPct ? `${Math.round(clamp01(value) * 100)} %` : fmt(value),
    gauge: isPct ? clamp01(value) : undefined,
    hint: indexExplanation(key, detail),
  };
}

/** Coerce either contract shape into a clean list of display cards. */
function normalise(stats?: DatasetStats): Card[] {
  if (!stats) return [];
  if (Array.isArray(stats)) {
    return stats
      .filter((s) => s && typeof s.value === 'number' && Number.isFinite(s.value))
      .map((s) => toCard(s.key ?? s.label, s.value, (s as { detail?: IndexDetail }).detail));
  }
  if (typeof stats === 'object') {
    return Object.entries(stats)
      .filter(([, v]) => typeof v === 'number' && Number.isFinite(v))
      .map(([key, value]) => toCard(key, value));
  }
  return [];
}

function humanise(key: string): string {
  const s = key.replace(/^n_/, 'nombre de ').replace(/_/g, ' ').trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function fmt(v: number): string {
  if (Number.isInteger(v)) return v.toLocaleString(LOCALE);
  return v.toLocaleString(LOCALE, { maximumFractionDigits: 2 });
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}
