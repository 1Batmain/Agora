import { useEffect, useState } from 'react';
import type { AnalysisPayload, Citation, Consultation } from './contract';
import { fetchAnalysis, fetchCitations, fetchInsights } from './analysisApi';
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
  onExploreAvis,
}: {
  dataset: Consultation;
  onHome: () => void;
  onViewGraph: () => void;
  /** Clic sur une citation représentative → page d'exploration FOCALISÉE sur l'avis. */
  onExploreAvis: (avisId: string) => void;
}) {
  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null);
  const [synthesis, setSynthesis] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  // Synthèse DYNAMIQUE : null = vue globale ; sinon synthèse du thème sélectionné.
  const [selectedThemeId, setSelectedThemeId] = useState<string | null>(null);
  const [themeSynthesis, setThemeSynthesis] = useState<string | null>(null);
  const [themeLoading, setThemeLoading] = useState(false);
  // Avis représentatifs du focus = citations triées centroïde (cliquables → exploration).
  const [citations, setCitations] = useState<Citation[] | null>(null);
  // Mot-clé cliqué → on ne montre que les avis qui le mentionnent (les plus proches).
  const [selectedKeyword, setSelectedKeyword] = useState<string | null>(null);

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
    if (selectedThemeId == null) {
      setCitations(null);
      return;
    }
    let cancelled = false;
    setThemeLoading(true);
    setThemeSynthesis(null);
    setCitations(null);
    fetchInsights(dataset.id, 'theme', selectedThemeId, selectedTheme ?? undefined)
      .catch(() => null)
      .then((s) => {
        if (cancelled) return;
        setThemeSynthesis(s?.data ?? null);
        setThemeLoading(false);
      });
    fetchCitations(dataset.id, selectedThemeId)
      .catch(() => null)
      .then((r) => {
        if (!cancelled) setCitations(r?.data ?? null);
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
          {themes.length > 0 && (
            <>
              <h3 className="synthesis__subhead">Clusters identifiés</h3>
              <ThemeNavigator
                themes={themes}
                total={navTotal}
                currentId={selectedThemeId}
                onDrill={setSelectedThemeId}
                onSelect={setSelectedThemeId}
                onBack={() => {
                  const cur = themes.find((t) => t.id === selectedThemeId);
                  setSelectedThemeId(cur?.parent_id ?? null);
                }}
              />
            </>
          )}

          {(() => {
            const dynLoading = selectedThemeId == null ? loading : themeLoading;
            const dynSource = selectedThemeId == null ? synthesis : themeSynthesis;
            const dynTitle = selectedTheme
              ? selectedTheme.title || selectedTheme.label
              : "Vue d'ensemble";
            // Mots-clés DU FOCUS : globaux si rien de sélectionné, sinon ceux du thème.
            const focusKeywords = selectedTheme ? (selectedTheme.keywords ?? []) : keywords;
            // Avis du focus : si un mot-clé est cliqué, on ne garde que ceux qui le
            // mentionnent (les plus proches) ; sinon les représentatifs (centroïde).
            const allAvis = citations ?? [];
            const repAvis = (selectedKeyword
              ? allAvis.filter((c) => (c.text || '').toLowerCase().includes(selectedKeyword.toLowerCase()))
              : allAvis
            ).slice(0, selectedKeyword ? 8 : 5);
            return (
              <div className="overview__dynsynth" aria-live="polite">
                <h3 className="synthesis__subhead">{dynTitle}</h3>
                {selectedTheme && (
                  <button
                    type="button"
                    className="overview__backgen"
                    onClick={() => setSelectedThemeId(null)}
                  >
                    ← Vue générale
                  </button>
                )}
                {dynLoading ? (
                  <p className="overview__loading">Chargement de la synthèse…</p>
                ) : dynSource ? (
                  <Markdown source={dynSource} />
                ) : (
                  <p className="overview__loading">Synthèse indisponible.</p>
                )}
                {/* Mots-clés CLIQUABLES, juste au-dessus des avis : un clic filtre les avis
                    sur ceux qui mentionnent ce mot-clé (les plus proches). */}
                {selectedTheme && focusKeywords.length > 0 && (
                  <div className="kw-chips kw-chips--clickable" aria-label="Mots-clés — cliquer pour filtrer les avis">
                    {focusKeywords.map((kw) => {
                      const on = selectedKeyword === kw;
                      return (
                        <button
                          key={kw}
                          type="button"
                          className={`kw-chip kw-chip--btn${on ? ' kw-chip--on' : ''}`}
                          aria-pressed={on}
                          title={on ? 'Retirer le filtre' : `Avis mentionnant « ${kw} »`}
                          onClick={() => setSelectedKeyword(on ? null : kw)}
                        >
                          {kw}
                        </button>
                      );
                    })}
                  </div>
                )}
                {selectedTheme && allAvis.length > 0 && (
                  <div className="overview__avis">
                    <h4 className="synthesis__subhead">
                      {selectedKeyword ? `Avis mentionnant « ${selectedKeyword} »` : 'Avis représentatifs'}
                      {selectedKeyword && (
                        <button
                          type="button"
                          className="overview__kwclear"
                          onClick={() => setSelectedKeyword(null)}
                        >
                          × tous
                        </button>
                      )}
                    </h4>
                    {repAvis.length > 0 ? (
                      repAvis.map((c, i) => {
                        const id = c.avis_id;
                        return (
                          <blockquote
                            key={id ?? i}
                            className={`overview__avis-quote${id ? ' overview__avis-quote--open' : ''}`}
                            role={id ? 'button' : undefined}
                            tabIndex={id ? 0 : undefined}
                            onClick={id ? () => onExploreAvis(id) : undefined}
                            onKeyDown={id ? (e) => { if (e.key === 'Enter') onExploreAvis(id); } : undefined}
                          >
                            « {c.text} »
                          </blockquote>
                        );
                      })
                    ) : (
                      <p className="overview__loading">Aucun avis ne mentionne « {selectedKeyword} ».</p>
                    )}
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
