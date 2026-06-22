import { useEffect, useState } from 'react';
import type { AvisProvenance, Citation, DataSource } from './contract';
import { fetchAvis } from './analysisApi';
import { AvisDetail } from './AvisDetail';

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
  citations,
  loading,
  source,
  onBack,
}: {
  dataset: string | null;
  themeLabel: string;
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
        <p className="panel__empty">Aucune citation pour ce thème.</p>
      )}
    </section>
  );
}
