import { Header } from './Header';
import type { Consultation } from './contract';
import { LOCALE } from './strings';

/**
 * Vue d'accueil d'Agora : hero explicite, repères d'usage et grille de
 * consultations (depuis `/datasets`). Le clic sur une carte remonte au shell,
 * qui route vers l'analyse (close) ou la participation (ouverte).
 */
export function Landing({
  datasets,
  loading,
  onOpen,
}: {
  datasets: Consultation[];
  loading: boolean;
  onOpen: (d: Consultation) => void;
}) {
  const openCount = datasets.filter((d) => d.status === 'open').length;
  const closedCount = datasets.length - openCount;
  const features = [
    {
      title: 'Voir les grands thèmes',
      body: 'Comprendre rapidement les sujets qui reviennent le plus dans les réponses citoyennes.',
    },
    {
      title: 'Vérifier les résultats',
      body: 'Retrouver les réponses citoyennes utilisées pour produire l’analyse.',
    },
  ];
  const steps = [
    {
      title: 'Lire les réponses',
      body: 'Agora analyse les réponses libres envoyées par les citoyens.',
    },
    {
      title: 'Regrouper les idées proches',
      body: 'Les réponses qui parlent du même sujet sont regroupées en thèmes.',
    },
    {
      title: 'Résumer avec des exemples',
      body: 'Chaque thème est accompagné de chiffres clés et de réponses sources.',
    },
  ];

  return (
    <div className="agora landing">
      <Header />

      <main className="landing__body">
        <section className="hero">
          <h1 className="hero__title">Comprendre ce que disent les citoyens</h1>
          <p className="hero__tagline">
            Agora organise les réponses libres d’une consultation en grands thèmes,
            chiffres clés et exemples vérifiables.
          </p>
          <ul className="hero__meta" aria-hidden={datasets.length === 0}>
            <li>
              <strong>{datasets.length}</strong> consultations
            </li>
            <li>
              <strong>{openCount}</strong> ouvertes
            </li>
            <li>
              <strong>{closedCount}</strong> analysées
            </li>
          </ul>
        </section>

        <section className="landing__features">
          <header className="sec-head">
            <h2>Ce que vous pouvez faire</h2>
            <span className="sec-head__hint">Deux repères simples pour comprendre rapidement une consultation</span>
          </header>
          <ul className="feature-grid">
            {features.map((feature) => (
              <li key={feature.title} className="feature-card">
                <strong>{feature.title}</strong>
                <p>{feature.body}</p>
              </li>
            ))}
          </ul>
        </section>

        <section className="landing__list">
          <header className="sec-head">
            <h2>Consultations disponibles</h2>
            <span className="sec-head__hint">Choisissez une consultation à analyser ou à rejoindre</span>
          </header>
          {loading ? (
            <div className="landing__loading">
              <span className="spinner" /> chargement des consultations…
            </div>
          ) : datasets.length === 0 ? (
            <p className="landing__empty">Aucune consultation disponible.</p>
          ) : (
            <ul className="landing__grid">
              {datasets.map((d) => {
                const open = d.status === 'open';
                return (
                  <li key={d.id}>
                    <button
                      className="ds-card"
                      onClick={() => onOpen(d)}
                      aria-label={`${d.label} — ${open ? 'consultation ouverte' : 'analyse disponible'}`}
                    >
                      <span className={`ds-card__badge ds-card__badge--${open ? 'open' : 'closed'}`}>
                        <span className="ds-card__dot" aria-hidden />
                        {open ? 'Consultation ouverte' : 'Analyse disponible'}
                      </span>
                      <span className="ds-card__eyebrow">Question posée</span>
                      <span className="ds-card__title">{d.label}</span>
                      <span className="ds-card__meta">
                        {(() => {
                          const n = d.n_contributions ?? d.n_nodes;
                          return n != null ? `${n.toLocaleString(LOCALE)} réponses citoyennes` : '—';
                        })()}
                      </span>
                      <span className="ds-card__source">Source : consultation publique</span>
                      <span className="ds-card__cta">
                        {open ? 'Répondre à la consultation →' : 'Comprendre les résultats →'}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
          <p className="landing__note">
            Analyse automatique : les résultats doivent être vérifiés avec les réponses
            sources.
          </p>
        </section>

        <section className="landing__problem">
          <header className="sec-head">
            <h2>Pourquoi Agora existe</h2>
            <span className="sec-head__hint">Lire des milliers de réponses à la main prend beaucoup de temps</span>
          </header>
          <div className="section-prose">
            <p>
              Une consultation publique peut produire des milliers de réponses libres. Les
              lire une par une prend beaucoup de temps. Agora aide à faire émerger les grands
              thèmes, tout en gardant un lien avec les réponses citoyennes d’origine.
            </p>
          </div>
        </section>

        <section className="landing__how">
          <header className="sec-head">
            <h2>Comment ça marche ?</h2>
            <span className="sec-head__hint">Une version simple de la méthode</span>
          </header>
          <ol className="how__steps">
            {steps.map((step) => (
              <li key={step.title} className="how__step">
                <div>
                  <strong>{step.title}</strong>
                  <p>{step.body}</p>
                </div>
              </li>
            ))}
          </ol>
          <p className="how__footer">
            Agora ne remplace pas la lecture humaine : l’outil aide à organiser les
            réponses, à repérer les grands thèmes et à retrouver les exemples qui les
            justifient.
          </p>
        </section>

        <section className="landing__collab">
          <header className="sec-head">
            <h2>Projet ouvert</h2>
            <span className="sec-head__hint">Code source et contribution</span>
          </header>
          <p className="how__lead">
            Le code source est disponible sur GitHub pour contribuer au projet.
          </p>
          <div className="collab__actions">
            <a
              className="collab__link"
              href="https://github.com/1Batmain/Analyse-des-consultations-citoyennes"
              target="_blank"
              rel="noreferrer"
            >
              <span className="collab__link-arrow" aria-hidden>→</span>
              <span className="collab__link-label">Repo GitHub</span>
              <span className="collab__link-sub">Consulter le dépôt et proposer une contribution</span>
            </a>
          </div>
        </section>
      </main>
    </div>
  );
}
