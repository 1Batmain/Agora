import type { ClaimTheme } from './types';

/**
 * Right panel for the emergent-claims view: the selected theme's keywords and its
 * representative claims (closest to the cluster centroid). Mirrors AvisPanel's
 * look (reuses its classes) but lists CLAIMS, not raw avis.
 */
export function ClaimsThemePanel({ theme }: { theme: ClaimTheme | null }) {
  if (!theme) {
    return (
      <section className="panel avis">
        <header className="panel__head">
          <h2>Thème</h2>
        </header>
        <p className="avis__empty">Clique un thème pour voir ses claims représentatives.</p>
      </section>
    );
  }
  return (
    <section className="panel avis">
      <header className="panel__head">
        <h2 title={theme.name}>{theme.name}</h2>
        <span className="avis__count">{theme.n_claims}</span>
      </header>
      <p className="claims__panelmeta">
        {theme.n_avis} avis · poids {theme.weight} · consensus {theme.consensus} · diversité{' '}
        {theme.diversity}
      </p>
      {theme.keywords.length > 0 && (
        <div className="avis__kw">
          {theme.keywords.map((k) => (
            <span className="kw" key={k}>
              {k}
            </span>
          ))}
        </div>
      )}
      <ul className="avis__list">
        {theme.representative_claims.map((c, i) => (
          <li className="avis__item" key={i}>
            <p>{c}</p>
          </li>
        ))}
      </ul>
    </section>
  );
}
