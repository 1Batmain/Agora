import { useEffect, useState } from 'react';
import type { Dataset } from '../types';
import type { AnalysisPayload } from './contract';
import { fetchAnalysis, fetchInsights } from './analysisApi';
import { Header } from './Header';
import { Markdown } from './Markdown';

/**
 * Page d'APERÇU d'une consultation CLOSE (sous-page entre la landing et le graphe).
 * Présente la consultation — questions/contexte, panel (langues), nombre de réponses,
 * nombre de thèmes identifiés, et la SYNTHÈSE générale — puis un bouton « Voir le
 * graphe » qui ouvre la vue d'analyse interactive.
 */
export function ConsultationOverview({
  dataset,
  onHome,
  onViewGraph,
}: {
  dataset: Dataset;
  onHome: () => void;
  onViewGraph: () => void;
}) {
  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null);
  const [synthesis, setSynthesis] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([
      fetchAnalysis(dataset.id).catch(() => null),
      fetchInsights(dataset.id, 'global').catch(() => null),
    ]).then(([a, s]) => {
      if (cancelled) return;
      setAnalysis(a?.data ?? null);
      setSynthesis(s?.data ?? null);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [dataset.id]);

  const totals = (analysis?.dataset_stats as { totals?: Record<string, number> } | undefined)?.totals ?? {};
  const context = analysis?.dataset_context || dataset.context || '';
  const nReponses = totals.participants ?? totals.n_avis ?? dataset.n_nodes ?? null;
  const nThemes = totals.n_themes ?? null;
  const langues = (dataset.languages ?? []).map((l) => l.toUpperCase()).join(' · ');

  return (
    <div className="agora overview">
      <Header onHome={onHome} right={<span className="ds-card__badge ds-card__badge--closed">Clos</span>} />

      <main className="overview__body">
        <section className="overview__head">
          <h1 className="overview__title">{dataset.label}</h1>
          {context && <p className="overview__context">{context}</p>}
        </section>

        <section className="overview__stats" aria-label="Chiffres de la consultation">
          <div className="overview__stat">
            <strong>{nReponses != null ? nReponses.toLocaleString('fr-FR') : '—'}</strong>
            <span>réponses</span>
          </div>
          <div className="overview__stat">
            <strong>{nThemes ?? '—'}</strong>
            <span>thèmes identifiés</span>
          </div>
          <div className="overview__stat">
            <strong>{langues || '—'}</strong>
            <span>panel · langues</span>
          </div>
        </section>

        <button type="button" className="btn-primary overview__cta" onClick={onViewGraph}>
          Voir le graphe →
        </button>

        <section className="overview__synthesis">
          <h2>Synthèse générale</h2>
          {loading ? (
            <p className="overview__loading">Chargement de la synthèse…</p>
          ) : synthesis ? (
            <Markdown source={synthesis} />
          ) : (
            <p className="overview__loading">Synthèse indisponible.</p>
          )}
        </section>
      </main>
    </div>
  );
}
