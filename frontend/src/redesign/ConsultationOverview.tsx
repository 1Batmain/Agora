import { useEffect, useState } from 'react';
import type { Consultation } from './contract';
import type { AnalysisPayload } from './contract';
import { fetchAnalysis, fetchInsights } from './analysisApi';
import { Header } from './Header';
import { Markdown } from './Markdown';
import { ThemeNavigator } from './ThemeNavigator';
import { LOCALE } from './strings';

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
  dataset: Consultation;
  onHome: () => void;
  onViewGraph: () => void;
}) {
  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null);
  const [synthesis, setSynthesis] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  // Synthèse DYNAMIQUE : null = vue globale ; sinon synthèse du thème sélectionné.
  const [selectedThemeId, setSelectedThemeId] = useState<string | null>(null);
  const [themeSynthesis, setThemeSynthesis] = useState<string | null>(null);
  const [themeLoading, setThemeLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setSelectedThemeId(null);
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

  // Fetch PARESSEUX de la synthèse du thème sélectionné ; annule le fetch précédent
  // au changement de sélection. null = vue globale (déjà chargée), pas de fetch.
  const selectedTheme = analysis?.themes?.find((t) => t.id === selectedThemeId) ?? null;
  useEffect(() => {
    if (selectedThemeId == null) return;
    let cancelled = false;
    setThemeLoading(true);
    setThemeSynthesis(null);
    fetchInsights(dataset.id, 'theme', selectedThemeId, selectedTheme ?? undefined)
      .catch(() => null)
      .then((s) => {
        if (cancelled) return;
        setThemeSynthesis(s?.data ?? null);
        setThemeLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // selectedTheme dérive de selectedThemeId — pas besoin de le suivre.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset.id, selectedThemeId]);

  const totals = (analysis?.dataset_stats as { totals?: Record<string, number> } | undefined)?.totals ?? {};
  const keywords = (analysis?.dataset_stats as { keywords?: string[] } | undefined)?.keywords ?? [];
  const context = analysis?.dataset_context || dataset.context || '';
  const nReponses = dataset.n_contributions ?? totals.participants ?? totals.n_avis ?? dataset.n_nodes ?? null;
  const nThemes = totals.n_themes ?? null;
  const langues = (dataset.languages ?? []).map((l) => l.toUpperCase()).join(' · ');
  // Arbre navigable : tous les thèmes du payload, dénominateur racine = voix totales.
  const themes = analysis?.themes ?? [];
  const macros = themes.filter((t) => !t.parent_id);
  const navTotal =
    (totals.participants ?? totals.n_avis ?? macros.reduce((s, m) => s + (m.n_avis ?? 0), 0)) || 0;

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
            <strong>{nReponses != null ? nReponses.toLocaleString(LOCALE) : '—'}</strong>
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

          {themes.length > 0 && (
            <>
              <h3 className="synthesis__subhead">Points de convergence</h3>
              <ThemeNavigator
                themes={themes}
                total={navTotal}
                selectedId={selectedThemeId}
                onSelect={setSelectedThemeId}
              />
            </>
          )}

          {/* Synthèse DYNAMIQUE : globale si rien de sélectionné, sinon celle du thème. */}
          {(() => {
            const dynLoading = selectedThemeId == null ? loading : themeLoading;
            const dynSource = selectedThemeId == null ? synthesis : themeSynthesis;
            const dynTitle = selectedTheme
              ? selectedTheme.title || selectedTheme.label
              : "Vue d'ensemble";
            // Mots-clés DU FOCUS : globaux si rien de sélectionné, sinon ceux du thème.
            const focusKeywords = selectedTheme ? (selectedTheme.keywords ?? []) : keywords;
            // Avis représentatifs (centroïde) du focus — à TOUS les niveaux (macro inclus).
            const avis = selectedTheme?.representative_claims ?? [];
            return (
              <div className="overview__dynsynth" aria-live="polite">
                <h3 className="synthesis__subhead">{dynTitle}</h3>
                {focusKeywords.length > 0 && (
                  <div className="kw-chips" aria-label="Mots-clés du focus">
                    {focusKeywords.map((kw) => (
                      <span key={kw} className="kw-chip">{kw}</span>
                    ))}
                  </div>
                )}
                {dynLoading ? (
                  <p className="overview__loading">Chargement de la synthèse…</p>
                ) : dynSource ? (
                  <Markdown source={dynSource} />
                ) : (
                  <p className="overview__loading">Synthèse indisponible.</p>
                )}
                {avis.length > 0 && (
                  <div className="overview__avis">
                    <h4 className="synthesis__subhead">Avis représentatifs</h4>
                    {avis.map((a, i) => (
                      <blockquote key={i} className="overview__avis-quote">« {a} »</blockquote>
                    ))}
                  </div>
                )}
              </div>
            );
          })()}
        </section>
      </main>
    </div>
  );
}
