import type { Dataset } from '../types';

/**
 * Vue d'accueil par défaut d'Agora : un pitch court et accrocheur, puis une
 * grille de cartes de consultations (depuis `/datasets`). Chaque carte porte un
 * badge Ouvert/Clos, son titre et son nombre de contributions. Le clic remonte
 * au shell, qui route vers l'analyse (clos) ou la participation (ouvert).
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
  return (
    <div className="agora landing">
      <header className="gov-header">
        <div className="gov-header__brand">
          <div className="gov-logo" aria-hidden>
            <span className="gov-logo__mark">◆</span>
          </div>
          <div className="gov-header__title">
            <strong>Agora</strong>
            <span>Analyse des consultations citoyennes</span>
          </div>
        </div>
      </header>

      <main className="landing__body">
        <section className="landing__hero">
          <h1>Agora — donnez du sens à la parole citoyenne</h1>
          <p>
            L'IA qui fait émerger les thèmes des consultations, fidèlement.
            Explorez ce que les citoyens ont vraiment dit, ou contribuez aux
            débats encore ouverts.
          </p>
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
      </main>
    </div>
  );
}
