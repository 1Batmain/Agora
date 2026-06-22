import { useEffect, useState } from 'react';
import type { AvisProvenance, Citation, DataSource } from './contract';
import { fetchAvis } from './analysisApi';
import { AvisDetail } from './AvisDetail';
import { Markdown } from './Markdown';

/** Short badge text per data source. */
const BADGE: Record<DataSource, string> = {
  live: 'live',
  building: 'en cours',
  mock: 'mock',
  error: 'erreur',
};

/**
 * F6 — leaf-level citations. When a leaf theme is selected the right column shows
 * its source claims (verbatim portions), sorted by proximity to the cluster
 * centroid. Clicking a claim opens its full avis with the extractive portions
 * HIGHLIGHTED in their cluster colour — one panel, two states (list ↔ avis), so
 * the avis view is a single consolidated flow.
 */
export function CitationsPanel({
  dataset,
  themeLabel,
  themeColor,
  hook,
  description,
  convergence,
  citations,
  loading,
  source,
  onBack,
}: {
  dataset: string | null;
  themeLabel: string;
  /** Selected leaf cluster colour — tints the header so the panel reads as "this cluster". */
  themeColor?: string;
  /** LLM accroche for this exact cluster (graceful if absent). */
  hook?: string;
  /** LLM synthesis for this exact cluster, rendered as markdown above the list. */
  description?: string;
  /** 0..1 convergence of ideas inside this cluster (graceful if absent). */
  convergence?: number;
  citations: Citation[] | null;
  loading: boolean;
  source: DataSource | null;
  onBack: () => void;
}) {
  // Selected avis (opened from a citation) — fetched lazily with highlights.
  const [avisId, setAvisId] = useState<string | null>(null);
  const [avis, setAvis] = useState<AvisProvenance | null>(null);
  const [avisLoading, setAvisLoading] = useState(false);

  // Reset the opened avis whenever the theme/citation list changes.
  useEffect(() => {
    setAvisId(null);
    setAvis(null);
  }, [themeLabel, citations]);

  useEffect(() => {
    if (!dataset || !avisId) return;
    let cancelled = false;
    setAvisLoading(true);
    setAvis(null);
    fetchAvis(dataset, avisId)
      .then(({ data }) => !cancelled && setAvis(data))
      .finally(() => !cancelled && setAvisLoading(false));
    return () => {
      cancelled = true;
    };
  }, [dataset, avisId]);

  if (avisId) {
    return <AvisDetail avis={avis} loading={avisLoading} onBack={() => setAvisId(null)} />;
  }

  return (
    <section className="panel citations">
      <header
        className="panel__head"
        style={themeColor ? { borderBottomColor: themeColor } : undefined}
      >
        <h2 title={themeLabel}>
          {themeColor && (
            <i className="citations__dot" style={{ background: themeColor }} aria-hidden />
          )}
          {themeLabel}
        </h2>
        {source && <span className={`badge badge--${source}`}>{BADGE[source]}</span>}
      </header>
      <button className="link-back" onClick={onBack}>
        ← retour aux thèmes
      </button>
      {/* Cluster synthesis — a short LLM note about THIS exact cluster, shown ABOVE
          the testimonials so the reader has the gist before the verbatims. */}
      {(hook || description || typeof convergence === 'number') && (
        <aside
          className="cluster-note"
          style={themeColor ? { borderLeftColor: themeColor } : undefined}
        >
          <span className="cluster-note__tag">Synthèse du cluster</span>
          {hook && <p className="cluster-note__hook">{hook}</p>}
          {description && (
            <div className="cluster-note__body">
              <Markdown source={description} />
            </div>
          )}
          {typeof convergence === 'number' && Number.isFinite(convergence) && (
            <p className="cluster-note__conv">
              Convergence des idées : <strong>{Math.round(convergence * 100)} %</strong>
            </p>
          )}
        </aside>
      )}
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
            {citations.length} témoignage{citations.length > 1 ? 's' : ''} · triés par
            proximité au cœur du thème
          </p>
          <ol className="citations__list">
            {citations.map((c, i) => {
              const openable = Boolean(c.avis_id);
              return (
                <li
                  className={`citations__item${openable ? ' citations__item--open' : ''}`}
                  key={i}
                  onClick={() => openable && setAvisId(c.avis_id!)}
                  title={openable ? "voir l'avis complet" : undefined}
                >
                  <p>“{c.text}”</p>
                  <span className="citations__sub">
                    proximité {(1 - Math.min(1, c.dist_to_centroid)).toFixed(2)} · poids {c.weight}
                    {openable && <span className="citations__open"> · voir l'avis →</span>}
                  </span>
                </li>
              );
            })}
          </ol>
        </>
      ) : (
        <p className="panel__empty">Aucun témoignage pour ce thème.</p>
      )}
    </section>
  );
}
