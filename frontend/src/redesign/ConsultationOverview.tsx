import { useEffect, useState } from 'react';
import type { AnalysisPayload, Consultation, CostPayload, ThemeOpinion } from './contract';
import { fetchAnalysis, fetchCost, fetchInsights, fetchOpinion } from './analysisApi';
import { Header } from './Header';
import { Markdown } from './Markdown';
import { ClusterOutline, stripSaillants } from './ClusterOutline';
import { LOCALE } from './strings';

/** Retire une phrase d'AMORCE résiduelle en toute fin de synthèse globale (ex. « Les
 * principaux thèmes identifiés sont : ») : le front la remplace par son propre lead
 * cliquable (« Agora a identifié N thèmes distincts : »), donc une amorce laissée par
 * le LLM ferait DOUBLON. On ne coupe que si la toute fin est une courte ligne en « : ». */
function stripDanglingLead(md: string | null): string | null {
  if (!md) return md;
  return md.replace(/\n+[^\n]{0,120}:\s*$/, '').trim();
}

/**
 * Page d'APERÇU d'une consultation CLOSE (sous-page entre la landing et le graphe).
 * Présente la consultation — questions/contexte, panel (langues), nombre de réponses,
 * nombre de thèmes — puis la SYNTHÈSE : la vue d'ensemble GLOBALE (fixe, en tête) suivie
 * de l'OUTLINE des clusters (`ClusterOutline`), un accordéon récursif où chaque cluster
 * se déplie EN PLACE avec sa propre synthèse riche (plus de navigation qui remplace la
 * page → on garde le fil de la profondeur). Un bouton « Voir le graphe » ouvre la vue
 * d'analyse interactive.
 */
export function ConsultationOverview({
  dataset,
  onHome,
  onViewGraph,
  onExploreTheme,
  onExploreAvis,
}: {
  dataset: Consultation;
  onHome: () => void;
  /** Ouvre le graphe, focalisé sur le thème courant (null = graphe complet). */
  onViewGraph: (themeId: string | null) => void;
  /** Ouvre l'explorateur d'avis, filtré sur le thème courant (null = tous les avis). */
  onExploreTheme: (themeId: string | null, stance?: 'favorable' | 'defavorable' | null) => void;
  /** Clic sur une citation représentative → page d'exploration FOCALISÉE sur l'avis. */
  onExploreAvis: (avisId: string) => void;
}) {
  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null);
  const [synthesis, setSynthesis] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  // Répartition d'opinion : chargée UNE fois par dataset, lookup par theme_id (gracieux si absent).
  const [opinions, setOpinions] = useState<ThemeOpinion[]>([]);
  // Coût LLM du traitement (transparence) — null si non mesuré.
  const [cost, setCost] = useState<CostPayload | null>(null);
  // Chemin ouvert de l'outline de clusters (accordéon par niveau).
  const [openPath, setOpenPath] = useState<string[]>([]);
  useEffect(() => {
    let cancelled = false;
    setCost(null);
    fetchCost(dataset.id).then((c) => !cancelled && setCost(c));
    return () => {
      cancelled = true;
    };
  }, [dataset.id]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setOpinions([]);
    setOpenPath([]);
    Promise.all([
      fetchAnalysis(dataset.id).catch(() => null),
      fetchInsights(dataset.id, 'global').catch(() => null),
      fetchOpinion(dataset.id).catch(() => []),
    ]).then(([a, s, op]) => {
      if (cancelled) return;
      setAnalysis(a?.data ?? null);
      setSynthesis(s?.data ?? null);
      setOpinions(op ?? []);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [dataset.id]);

  const totals = (analysis?.dataset_stats as { totals?: Record<string, number> } | undefined)?.totals ?? {};
  const nReponses = dataset.n_contributions ?? totals.participants ?? totals.n_avis ?? dataset.n_nodes ?? null;
  // Distinction honnête : participants (lignes reçues) / réponses à LA question (voix,
  // doublons regroupés) / textes uniques analysés. `n_responses` sert de dénominateur vrai.
  const nResp = dataset.n_responses ?? nReponses;
  const isSampled = dataset.n_sample != null && nResp != null && dataset.n_sample < 0.98 * nResp;
  const hasNonRespondents = nResp != null && nReponses != null && nResp < nReponses;
  const nThemes = totals.n_themes ?? null;
  const langues = (dataset.languages ?? []).map((l) => l.toUpperCase()).join(' · ');
  // Arbre navigable : tous les thèmes du payload, dénominateur racine = voix totales.
  const themes = analysis?.themes ?? [];
  const macros = themes.filter((t) => !t.parent_id);
  const navTotal =
    (totals.participants ?? totals.n_avis ?? macros.reduce((s, m) => s + (m.n_avis ?? 0), 0)) || 0;
  // Témoignages RÉELLEMENT analysés = avis DISTINCTS (`n_sample`). `navTotal` (somme des
  // `n_avis` par thème) DOUBLE-COMPTE les avis présents dans plusieurs thèmes → ne JAMAIS
  // l'afficher comme total honnête (c'est le bug des « 4377 » vs 3000 analysés / 28384 total).
  const nAnalyzed = dataset.n_sample ?? navTotal;
  const globalSource = stripDanglingLead(stripSaillants(synthesis));

  return (
    <div className="agora overview">
      <Header onHome={onHome} right={<span className="ds-card__badge ds-card__badge--closed">Clos</span>} />

      <main className="overview__body">
        <section className="overview__head">
          {/* Nom de la consultation en titre MODESTE, une courte intro, puis la question
              posée en sous-titre italique entre guillemets (centrés). */}
          <h1 className="overview__name">{dataset.label}</h1>
          {dataset.question && (
            <p className="overview__question">
              <span className="overview__question-lead">Analyse des réponses à la question&nbsp;:</span>
              <span className="overview__question-text">« {dataset.question} »</span>
            </p>
          )}
          {dataset.official_url && (
            <p className="overview__official">
              <a href={dataset.official_url} target="_blank" rel="noreferrer">
                Voir la consultation officielle ↗
              </a>
            </p>
          )}
        </section>

        <section className="overview__stats" aria-label="Chiffres de la consultation">
          <div className="overview__stat">
            <strong>{nAnalyzed > 0 ? nAnalyzed.toLocaleString(LOCALE) : '—'}</strong>
            <span>témoignages analysés</span>
          </div>
        </section>

        {/* Transparence : quand l'analyse porte sur un ÉCHANTILLON (coût), on le dit
            clairement — pas de faux « tout est analysé ». Masqué si couverture 100 %. */}
        {isSampled && dataset.n_sample != null && nResp != null ? (
          <p className="overview__sample" role="note">
            <span className="overview__sample-tag">Échantillon</span>
            L'analyse porte sur un échantillon représentatif de{' '}
            <strong>{dataset.n_sample.toLocaleString(LOCALE)}</strong> textes —{' '}
            <strong>{Math.round((dataset.n_sample / nResp) * 100)}&nbsp;%</strong> des{' '}
            {nResp.toLocaleString(LOCALE)} réponses à la question
            {hasNonRespondents && nReponses != null && (
              <> ({nReponses.toLocaleString(LOCALE)} participants, dont{' '}
              {(nReponses - nResp).toLocaleString(LOCALE)} sans réponse à cette question)</>
            )}
            . Les proportions par thème sont celles de cet échantillon.
          </p>
        ) : dataset.n_sample != null && nResp != null && hasNonRespondents ? (
          <p className="overview__sample overview__sample--full" role="note">
            <span className="overview__sample-tag">Couverture complète</span>
            <strong>{dataset.n_sample.toLocaleString(LOCALE)}</strong> textes uniques analysés,
            couvrant <strong>{nResp.toLocaleString(LOCALE)}</strong> réponses (doublons
            strictement identiques regroupés, chaque voix comptée)&nbsp;;{' '}
            {nReponses != null && (
              <>{(nReponses - nResp).toLocaleString(LOCALE)} participants n'ont pas répondu à
              cette question.</>
            )}
          </p>
        ) : null}

        <section className="overview__synthesis">
          {/* 1) VUE D'ENSEMBLE — synthèse globale FIXE, toujours en tête. */}
          <div className={`overview__dynsynth${loading ? ' is-loading' : ''}`} aria-live="polite" aria-busy={loading}>
            {globalSource ? (
              <div className="overview__synthbody">
                <Markdown source={globalSource} />
                {loading && <p className="overview__synthloading">Actualisation…</p>}
              </div>
            ) : loading ? (
              <p className="overview__loading">Chargement de la synthèse…</p>
            ) : (
              <p className="overview__loading">Synthèse indisponible.</p>
            )}
          </div>

          {/* 2) LES CLUSTERS — introduits par la phrase qui prolonge la synthèse globale.
              L'outline n'affiche que les 5 plus gros par défaut (+ « voir plus »), chacun
              déployable EN PLACE (accordéon récursif par niveau) avec sa synthèse riche. */}
          {themes.length > 0 && (
            <>
              <h3 className="synth-h">Thèmes identifiés</h3>
              <p className="overview__clusters-lead--sub">
                Agora a identifié <strong>{nThemes ?? macros.length}</strong> thèmes distincts.
              </p>
              <ClusterOutline
                dataset={dataset.id}
                themes={themes}
                total={navTotal}
                navTotal={navTotal}
                opinions={opinions}
                openPath={openPath}
                onOpenPath={setOpenPath}
                onViewGraph={onViewGraph}
                onExploreTheme={onExploreTheme}
                onExploreAvis={onExploreAvis}
              />
            </>
          )}

          {/* Accès graphe + explorateur TOUT EN BAS de la synthèse (vue globale). */}
          {themes.length > 0 && (
            <div className="overview__actions overview__actions--bottom">
              <button type="button" className="btn-primary" onClick={() => onViewGraph(null)}>
                Voir le graphe →
              </button>
              <button type="button" className="btn-secondary" onClick={() => onExploreTheme(null)}>
                Consulter les témoignages →
              </button>
            </div>
          )}
        </section>

        {/* PIED DE PAGE — transparence des coûts : tokens · $ · durée RÉELLE du traitement
            (somme des phases mesurées + estimées marquées), versus le dispositif officiel
            sourcé quand le descripteur en porte un. */}
        {(cost || langues) && (
          <footer className="overview__footer" role="contentinfo">
            {cost && (
            <p className="overview__cost">
              <span className="overview__cost-agora">
                Traitement Agora&nbsp;:{' '}
                <strong>{(cost.total.total_tokens / 1e6).toLocaleString(LOCALE, { maximumFractionDigits: 1 })}&nbsp;M tokens</strong>
                {' · '}
                <strong>≈&nbsp;{cost.total.estimated_usd.toLocaleString(LOCALE, { maximumFractionDigits: 2 })}&nbsp;$</strong>
                {(() => {
                  const secs =
                    (cost.total as { duration_seconds?: number }).duration_seconds ||
                    (cost.durations?.analysis_seconds ?? 0) + (cost.durations?.opinion_seconds ?? 0);
                  if (!secs) return null;
                  const label = secs >= 5400
                    ? `~${(secs / 3600).toLocaleString(LOCALE, { maximumFractionDigits: 1 })} h`
                    : `~${Math.max(1, Math.round(secs / 60))} min`;
                  const estimated = Object.keys(cost.phases ?? {}).some((k) => k.endsWith('_estimee'));
                  return (
                    <>
                      {' · '}
                      <strong>{label} de traitement</strong>
                      {estimated && <span className="overview__cost-est"> (durée en partie estimée)</span>}
                    </>
                  );
                })()}
              </span>
              {dataset.official_baseline && (
                <span className="overview__cost-baseline">
                  {' '}versus {dataset.official_baseline.label}&nbsp;:{' '}
                  {dataset.official_baseline.cost}
                  {dataset.official_baseline.duration ? ` · ${dataset.official_baseline.duration}` : ''}
                  {dataset.official_baseline.source_url && (
                    <>
                      {' '}
                      <a href={dataset.official_baseline.source_url} target="_blank" rel="noreferrer">
                        (source)
                      </a>
                    </>
                  )}
                  {dataset.official_baseline.note && (
                    <span className="overview__cost-note"> {dataset.official_baseline.note}</span>
                  )}
                </span>
              )}
            </p>
            )}
            {/* Petite info additionnelle : langues du panel. */}
            {langues && (
              <p className="overview__langs">Langues du panel&nbsp;: {langues}</p>
            )}
          </footer>
        )}
      </main>
    </div>
  );
}
