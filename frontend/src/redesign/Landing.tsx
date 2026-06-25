import { Header } from './Header';
import type { Dataset } from '../types';

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
  datasets: Dataset[];
  loading: boolean;
  onOpen: (d: Dataset) => void;
}) {
  const openCount = datasets.filter((d) => d.status === 'open').length;
  const closedCount = datasets.length - openCount;

  return (
    <div className="agora landing">
      <Header />

      <main className="landing__body">
        <section className="hero">
          <h1 className="hero__title">
            La parole citoyenne,
            <br />
            <span className="hero__accent">réstituée fidèlement.</span>
          </h1>
          <p className="hero__tagline">
            Agora fait émerger les thèmes des grandes consultations sans trahir
            ce qui a été dit — puis vous laisse explorer l'analyse ou contribuer
            aux débats encore ouverts.
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
          <h2>
            <span className="landing__list-idx">01</span> Consultations
          </h2>
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
                        {d.n_nodes ? `${d.n_nodes.toLocaleString('fr-FR')} contributions` : '—'}
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
          <h2>
            <span className="landing__list-idx">02</span> Comment ça marche
          </h2>
          <p className="how__lead">
            Agora transforme des milliers de contributions en une carte des idées —
            fidèlement, sans grille imposée et sans reformuler la parole citoyenne.
          </p>
          <ol className="how__steps">
            <li className="how__step">
              <span className="how__num">1</span>
              <div>
                <strong>Extraction fidèle</strong>
                <p>
                  Chaque contribution est découpée en idées reprises <em>mot à mot</em>{' '}
                  — jamais reformulées. La position du citoyen n'est pas déformée.
                </p>
              </div>
            </li>
            <li className="how__step">
              <span className="how__num">2</span>
              <div>
                <strong>Regroupement émergent</strong>
                <p>
                  Les idées sont placées dans un espace sémantique et regroupées par
                  thème <em>automatiquement</em> — les sujets émergent des données,
                  aucune catégorie n'est imposée à l'avance.
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
