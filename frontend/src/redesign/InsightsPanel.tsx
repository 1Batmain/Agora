import { Markdown } from './Markdown';
import type { DataSource } from './contract';

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
        {source && (
          <span className={`badge badge--${source}`}>{source === 'mock' ? 'mock' : 'live'}</span>
        )}
      </header>
      {loading ? (
        <div className="insights__loading">
          <span className="spinner" /> génération de la synthèse…
        </div>
      ) : markdown ? (
        <Markdown source={markdown} />
      ) : (
        <p className="panel__empty">Aucune synthèse pour ce niveau.</p>
      )}
    </section>
  );
}
