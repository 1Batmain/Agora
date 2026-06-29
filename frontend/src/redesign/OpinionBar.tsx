import type { ThemeOpinion } from './contract';

/**
 * Répartition d'opinion d'un thème FEUILLE : l'objet de clivage (proposition polaire
 * débattable), une barre fav / défavorable / nuance (vert / rouge / gris discrets), le
 * `% favorable parmi les engagés`, et un badge clivant/consensuel.
 *
 * Honnêteté : un thème `impur` (signal trop diffus, garde-fou de pureté côté backend)
 * n'a PAS de répartition fiable — on ne montre alors aucune barre (rien de trompeur).
 * Un thème `consensuel` surface tout de même la minorité de sceptiques (segment rouge).
 */
export function OpinionBar({ opinion }: { opinion: ThemeOpinion }) {
  const { proposition, fav, def, nuance, pct_favorable, profil } = opinion;
  if (profil === 'impur') return null;

  const total = fav + def + nuance || 1;
  const pct = (x: number) => `${(100 * x) / total}%`;
  const pctFav = Math.round(100 * pct_favorable);
  const clivant = profil === 'clivant';

  return (
    <div className="opinion" aria-label="Répartition d'opinion du thème">
      <div className="opinion__head">
        <span className="opinion__label">Objet de clivage</span>
        <span className={`opinion__badge opinion__badge--${profil}`}>
          {clivant ? 'Clivant' : 'Consensuel'}
        </span>
      </div>
      <p className="opinion__proposition">« {proposition} »</p>

      <div
        className="opinion__bar"
        role="img"
        aria-label={`${fav} favorables, ${def} défavorables, ${nuance} nuancés`}
      >
        {fav > 0 && (
          <span className="opinion__seg opinion__seg--fav" style={{ width: pct(fav) }} title={`Favorables : ${fav}`} />
        )}
        {def > 0 && (
          <span className="opinion__seg opinion__seg--def" style={{ width: pct(def) }} title={`Défavorables : ${def}`} />
        )}
        {nuance > 0 && (
          <span className="opinion__seg opinion__seg--nu" style={{ width: pct(nuance) }} title={`Nuancés / hors-sujet : ${nuance}`} />
        )}
      </div>

      <div className="opinion__legend">
        <span className="opinion__key opinion__key--fav">Favorable {fav}</span>
        <span className="opinion__key opinion__key--def">Défavorable {def}</span>
        <span className="opinion__key opinion__key--nu">Nuance {nuance}</span>
      </div>

      <p className="opinion__pct">
        <strong>{pctFav}%</strong> favorables parmi les contributions engagées
        {clivant
          ? ' — opinion partagée.'
          : ' — large adhésion, une minorité reste sceptique.'}
      </p>
    </div>
  );
}
