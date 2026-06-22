import { Markdown } from './Markdown';
import type { DataSource } from './contract';

/** Short badge text per data source. */
const BADGE: Record<DataSource, string> = {
  live: 'live',
  building: 'en cours',
  mock: 'mock',
  error: 'erreur',
};

/**
 * F4 — right column. Renders the LLM Markdown synthesis for the CURRENT zoom
 * level (global vs selected theme). Shows a spinner during generation. At a leaf
 * the parent swaps this for the citations panel (no LLM there).
 */
export function InsightsPanel({
  title,
  markdown,
  loading,
  source,
}: {
  title: string;
  markdown: string | null;
  loading: boolean;
  source: DataSource | null;
}) {
  return (
    <section className="panel insights">
      <header className="panel__head">
        <h2 title={title}>{title}</h2>
        {source && <span className={`badge badge--${source}`}>{BADGE[source]}</span>}
      </header>
      {loading ? (
        <div className="insights__loading">
          <span className="spinner" /> génération de la synthèse…
        </div>
      ) : markdown ? (
        <Markdown source={markdown} />
      ) : source === 'building' ? (
        <div className="insights__loading">
          <span className="spinner" /> Analyse en cours…
        </div>
      ) : source === 'error' ? (
        <p className="panel__empty">Synthèse indisponible (backend).</p>
      ) : (
        <p className="panel__empty">Aucune synthèse pour ce niveau.</p>
      )}
    </section>
  );
}
