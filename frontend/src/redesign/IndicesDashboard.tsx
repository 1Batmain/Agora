import type { DatasetStat, DatasetStats } from './contract';

/**
 * F8 — dashboard of dataset-level INDICES under the map. Renders `dataset_stats`
 * from `/analysis` as intuitive cards/gauges (label + value + short explanation).
 *
 * Robust to two backend shapes (the contract is permissive): a ready list of
 * `DatasetStat` objects, OR a loose `{key: number}` record. The latter is
 * normalised here using a small dictionary of known indicators (label, hint,
 * formatting, whether it reads as a 0..1 gauge); unknown keys are humanised so
 * the panel never breaks on an unexpected field. Absent/empty → renders nothing.
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

/** Known indicators: label, explanation, and how to render (gauge vs count). */
const DICT: Record<
  string,
  { label: string; hint: string; pct?: boolean; gaugeFromValue?: boolean }
> = {
  diversity: { label: 'Diversité des opinions', hint: 'Variété des thèmes exprimés (0 = unanime, 1 = très éclaté).', pct: true, gaugeFromValue: true },
  diversite: { label: 'Diversité des opinions', hint: 'Variété des thèmes exprimés (0 = unanime, 1 = très éclaté).', pct: true, gaugeFromValue: true },
  consensus: { label: 'Consensus global', hint: 'Degré d’accord moyen sur l’ensemble des avis.', pct: true, gaugeFromValue: true },
  consensus_global: { label: 'Consensus global', hint: 'Degré d’accord moyen sur l’ensemble des avis.', pct: true, gaugeFromValue: true },
  concentration: { label: 'Concentration', hint: 'Part des avis captée par les plus gros thèmes (1 = très concentré).', pct: true, gaugeFromValue: true },
  polarization: { label: 'Polarisation', hint: 'Opposition entre pôles d’opinion.', pct: true, gaugeFromValue: true },
  polarisation: { label: 'Polarisation', hint: 'Opposition entre pôles d’opinion.', pct: true, gaugeFromValue: true },
  coverage: { label: 'Couverture', hint: 'Part des avis rattachés à un thème.', pct: true, gaugeFromValue: true },
  n_avis: { label: 'Avis analysés', hint: 'Nombre total de contributions citoyennes.' },
  n_themes: { label: 'Thèmes émergents', hint: 'Nombre de thèmes de premier niveau.' },
  n_claims: { label: 'Arguments extraits', hint: 'Nombre de prises de position verbatim.' },
  n_clusters: { label: 'Clusters', hint: 'Nombre de regroupements détectés.' },
};

type Card = DatasetStat;

/** Coerce either contract shape into a clean list of display cards. */
function normalise(stats?: DatasetStats): Card[] {
  if (!stats) return [];
  if (Array.isArray(stats)) {
    return stats
      .filter((s) => s && typeof s.value === 'number' && Number.isFinite(s.value))
      .map((s) => ({ ...s, key: s.key ?? s.label }));
  }
  if (typeof stats === 'object') {
    return Object.entries(stats)
      .filter(([, v]) => typeof v === 'number' && Number.isFinite(v))
      .map(([key, value]) => {
        const d = DICT[key];
        const isPct = d?.pct ?? (value >= 0 && value <= 1);
        const asGauge = d?.gaugeFromValue ?? isPct;
        return {
          key,
          label: d?.label ?? humanise(key),
          value,
          display: isPct ? `${Math.round(clamp01(value) * 100)} %` : fmt(value),
          gauge: asGauge ? clamp01(value) : undefined,
          hint: d?.hint,
        } as Card;
      });
  }
  return [];
}

function humanise(key: string): string {
  const s = key.replace(/^n_/, 'nombre de ').replace(/_/g, ' ').trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function fmt(v: number): string {
  if (Number.isInteger(v)) return v.toLocaleString('fr-FR');
  return v.toLocaleString('fr-FR', { maximumFractionDigits: 2 });
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}
