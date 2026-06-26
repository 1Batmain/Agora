import { Header } from './Header';
import type { Consultation } from './contract';
import { LOCALE } from './strings';

/**
 * Vue d'accueil d'Agora — hero « façon framework » (grand titre mono, tagline,
 * beaucoup de blanc, curseur typewriter clignotant en déco), puis la grille de
 * consultations (depuis `/datasets`). Chaque carte porte un badge Ouvert/Clos,
 * son titre et son nombre de contributions. Le clic remonte au shell, qui route
 * vers l'analyse (clos) ou la participation (ouvert).
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

  return (
    <div className="agora landing">
      <Header />

      <main className="landing__body">
        <section className="hero">
          <h1 className="hero__title">
            <span className="hero__accent">Libérez </span>
            la parole citoyenne
            <br />
          </h1>
          <p className="hero__tagline">
            Agora fait émerger les thèmes et les idées communes des questions ouvertes
            et fournit une analyse des opinions de manière automatisée, traçable et
            transparente.
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

        <section className="landing__list">
          <h2>Consultations</h2>
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
                      aria-label={`${d.label} — ${open ? 'consultation ouverte' : 'consultation close'}`}
                    >
                      <span className={`ds-card__badge ds-card__badge--${open ? 'open' : 'closed'}`}>
                        <span className="ds-card__dot" aria-hidden />
                        {open ? 'Ouvert' : 'Clos'}
                      </span>
                      <span className="ds-card__title">{d.label}</span>
                      <span className="ds-card__meta">
                        {(() => {
                          const n = d.n_contributions ?? d.n_nodes;
                          return n ? `${n.toLocaleString(LOCALE)} contributions` : '—';
                        })()}
                      </span>
                      <span className="ds-card__cta">
                        {open ? 'Participer →' : 'Voir l’analyse →'}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section className="landing__how">
          <h2>Comment ça marche</h2>
          <p className="how__lead">
            Agora utilise l'IA pour identifier les différentes thématiques abordées
            dans une contribution. Il segmente l'avis sans en modifier le contenu.
            Chaque segment est ensuite encodé dans un espace latent pour identifier les
            thèmes les plus proches.
          </p>
          <ol className="how__steps">
            <li className="how__step">
              <span className="how__num">1</span>
              <div>
                <strong>Extraction des thèmes abordés</strong>
                <p>
                  Chaque contribution est segmentée par thématique{' '}
                  <em>sans en modifier le contenu</em> — cette tâche est effectuée par
                  un grand modèle de langue (Mistral).
                </p>
              </div>
            </li>
            <li className="how__step">
              <span className="how__num">2</span>
              <div>
                <strong>Regroupement automatique</strong>
                <p>
                  Les idées sont transformees en vecteur d'embedding puis sont regroupées
                  par thème <em>automatiquement</em> — les sujets émergent des données,
                  aucune catégorie n'est imposée.
                </p>
              </div>
            </li>
            <li className="how__step">
              <span className="how__num">3</span>
              <div>
                <strong>Synthèse traçable</strong>
                <p>
                  Chaque thème reçoit une synthèse lisible, et chaque idée reste{' '}
                  <em>traçable</em> jusqu'à la contribution d'origine — on peut
                  toujours remonter à la source.
                </p>
              </div>
            </li>
          </ol>
          <p className="how__footer">
            Modèles ouverts · traitement souverain · multilingue.
          </p>
        </section>
      </main>
    </div>
  );
}
