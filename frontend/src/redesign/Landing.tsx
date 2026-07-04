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
  const formatResponseCount = (n: number | null | undefined) =>
    n != null ? `${n.toLocaleString(LOCALE)} réponses citoyennes` : 'Nombre de réponses indisponible';
  const features = [
    {
      title: 'Les thèmes identifiés',
      body: 'Les grands sujets qui reviennent dans les contributions citoyennes.',
    },
    {
      title: 'Les chiffres clés',
      body: 'Le nombre de contributions analysées, le nombre de thèmes identifiés et l’état de la consultation.',
    },
    {
      title: 'Des exemples d’avis',
      body: 'Des extraits de contributions pour garder un lien avec ce que les citoyens ont réellement exprimé.',
    },
  ];
  const steps = [
    {
      title: '1. Une question est posée',
      body: 'Une consultation publique invite les citoyens à donner leur avis sur un sujet précis.',
    },
    {
      title: '2. Les citoyens contribuent',
      body: 'Ils expriment leurs avis, leurs préoccupations ou leurs propositions dans des textes libres.',
    },
    {
      title: '3. Agora regroupe les sujets récurrents',
      body: 'Les contributions qui parlent de sujets proches sont rapprochées.',
    },
    {
      title: '4. Les résultats sont présentés clairement',
      body: 'Agora affiche les thèmes identifiés, les chiffres clés et des exemples d’avis exprimés.',
    },
  ];

  return (
    <div className="agora landing">
      <Header />

      <main className="landing__body">
        <section className="hero">
          <p className="hero__kicker">Outil d’analyse de consultations publiques</p>
          <h1 className="hero__title">Comprendre ce que disent les citoyens dans une consultation</h1>
          <p className="hero__tagline">
            Agora analyse les contributions citoyennes publiées dans une
            consultation et fait ressortir les sujets qui reviennent le plus
            souvent.
          </p>
          <a className="hero__cta btn-primary" href="#consultations">
            Voir les consultations
          </a>
          <ul className="hero__meta" aria-hidden={datasets.length === 0}>
            <li>
              <strong>{datasets.length}</strong> consultations disponibles
            </li>
            <li>
              <strong>{openCount}</strong> consultations ouvertes
            </li>
            <li>
              <strong>{closedCount}</strong> analyses disponibles
            </li>
          </ul>
        </section>

        <section className="landing__how">
          <header className="sec-head">
            <h2>Comment ça fonctionne</h2>
            <span className="sec-head__hint">De la question posée à une lecture plus claire des contributions.</span>
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
        </section>

        <section className="landing__features">
          <header className="sec-head">
            <h2>Ce que vous pouvez voir</h2>
            <span className="sec-head__hint">Trois repères pour comprendre rapidement le contenu d’une consultation.</span>
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

        <section id="consultations" className="landing__list">
          <header className="sec-head">
            <h2>Consultations disponibles</h2>
            <span className="sec-head__hint">Choisissez une consultation pour voir la question posée, la source officielle, les chiffres clés et l’analyse des thèmes identifiés.</span>
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
                        {formatResponseCount(d.n_contributions ?? d.n_nodes)}
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
          <p className="landing__summary">
            Les résultats sont une aide à la lecture. Ils ne remplacent pas une
            analyse humaine ni une synthèse officielle.
          </p>
        </section>

        <section className="landing__collab">
          <header className="sec-head">
            <h2>Projet ouvert</h2>
            <span className="sec-head__hint">Code source et contribution.</span>
          </header>
          <p className="collab__lead">
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
