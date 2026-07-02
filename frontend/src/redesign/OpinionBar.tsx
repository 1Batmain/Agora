import type { ThemeOpinion } from './contract';

/**
 * Répartition d'opinion d'un thème FEUILLE, en DEUX niveaux d'insight :
 *  1. ENGAGEMENT : quelle part des contributions a pris une position CLAIRE (sentiment positif
 *     ou négatif) envers l'objet de clivage — vs « nuance / hors-sujet ». C'est le
 *     premier insight : la stance ne porte que sur les avis engagés.
 *  2. RÉPARTITION : PARMI les engagées, le partage positif / négatif.
 *  (3. à venir : argument mining — les arguments les plus mis en avant pour / contre.)
 *
 * Honnêteté : un thème `impur` (signal trop diffus, garde-fou backend) n'a pas de
 * répartition fiable → on ne montre rien. Un thème `consensuel` surface tout de même la
 * minorité de sceptiques.
 */
export function OpinionBar({ opinion }: { opinion: ThemeOpinion }) {
  const { proposition, fav, def, nuance, n, engagement, pct_favorable, profil } = opinion;
  if (profil === 'impur') return null;

  const engaged = fav + def;
  const total = n || engaged + nuance || 1;
  const engPct = Math.round(100 * (engagement ?? engaged / total));
  const pctFav = Math.round(100 * pct_favorable);
  const pctDef = 100 - pctFav;
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

      {/* 1 ── ENGAGEMENT : part des contributions qui expriment un sentiment net */}
      <div className="opinion__metric">
        <p className="opinion__lead">
          <strong>{engPct}%</strong> des contributions expriment un sentiment net{' '}
          <span className="opinion__sub">({engaged} sur {total})</span>
        </p>
        <div className="opinion__bar" role="img" aria-label={`${engaged} avec sentiment, ${nuance} neutres`}>
          <span className="opinion__seg opinion__seg--engaged" style={{ width: `${engPct}%` }} title={`Avec sentiment : ${engaged}`} />
          <span className="opinion__seg opinion__seg--nu" style={{ width: `${100 - engPct}%` }} title={`Neutres / hors-sujet : ${nuance}`} />
        </div>
      </div>

      {/* 2 ── SENTIMENT : positif / négatif envers l'objet de clivage, PARMI les engagées */}
      {engaged > 0 && (
        <div className="opinion__metric">
          <p className="opinion__lead">
            Parmi elles : <strong className="opinion__txt--fav">{pctFav}% positifs</strong>
            {' · '}
            <strong className="opinion__txt--def">{pctDef}% négatifs</strong>
          </p>
          <div className="opinion__bar" role="img" aria-label={`${fav} positifs, ${def} négatifs`}>
            <span className="opinion__seg opinion__seg--fav" style={{ width: `${pctFav}%` }} title={`Positifs : ${fav}`} />
            <span className="opinion__seg opinion__seg--def" style={{ width: `${pctDef}%` }} title={`Négatifs : ${def}`} />
          </div>
          <p className="opinion__note">
            {clivant ? 'Sentiment partagé.' : 'Sentiment majoritairement positif, une minorité négative.'}
          </p>
        </div>
      )}

      {/* 3 ── (à venir) argument mining : arguments les plus mis en avant pour / contre */}
    </div>
  );
}
