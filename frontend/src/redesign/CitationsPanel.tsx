import type { Citation, DataSource } from './contract';

/** Short badge text per data source. */
const BADGE: Record<DataSource, string> = {
  live: 'live',
  building: 'en cours',
  mock: 'mock',
  error: 'erreur',
};

/**
 * F6 — leaf-level citations. When a leaf theme is selected the right column shows
 * its source avis, sorted by proximity to the cluster centroid (most
 * representative first). Pure navigation/reading — no LLM. Adapts the spirit of
 * the legacy AvisPanel to the new contract shape.
 */
export function CitationsPanel({
  themeLabel,
  citations,
  loading,
  source,
  onBack,
}: {
  themeLabel: string;
  citations: Citation[] | null;
  loading: boolean;
  source: DataSource | null;
  onBack: () => void;
}) {
  return (
    <section className="panel citations">
      <header className="panel__head">
        <h2 title={themeLabel}>{themeLabel}</h2>
        {source && <span className={`badge badge--${source}`}>{BADGE[source]}</span>}
      </header>
      <button className="link-back" onClick={onBack}>
        ← retour aux thèmes
      </button>
      {loading ? (
        <div className="insights__loading">
          <span className="spinner" /> chargement des citations…
        </div>
      ) : source === 'building' ? (
        <div className="insights__loading">
          <span className="spinner" /> Analyse en cours…
        </div>
      ) : citations && citations.length ? (
        <>
          <p className="citations__meta">
            {citations.length} citations · triées par proximité au centroïde
          </p>
          <ol className="citations__list">
            {citations.map((c, i) => (
              <li className="citations__item" key={i}>
                <p>{c.text}</p>
                <span className="citations__sub">
                  proximité {(1 - Math.min(1, c.dist_to_centroid)).toFixed(2)} · poids {c.weight}
                </span>
              </li>
            ))}
          </ol>
        </>
      ) : (
        <p className="panel__empty">Aucune citation pour ce thème.</p>
      )}
    </section>
  );
}
