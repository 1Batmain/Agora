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
export function OpinionBar({
  opinion,
  onSelectStance,
}: {
  opinion: ThemeOpinion;
  /** Clic sur la carte positive/négative → ouvrir les avis de ce sentiment (câblé par le shell). */
  onSelectStance?: (stance: 'favorable' | 'defavorable') => void;
}) {
  const {
    proposition, fav, def, nuance, n, engagement, pct_favorable, profil,
    is_aggregate, n_children, child_propositions,
  } = opinion;
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
        <span className="opinion__label">Analyse des positions</span>
        <span className={`opinion__badge opinion__badge--${profil}`}>
          {clivant ? 'Clivant' : 'Consensuel'}
        </span>
      </div>
      <p className="opinion__proposition">
        Par rapport à la question{is_aggregate ? ` (synthèse de ${n_children ?? 0} sous-thèmes)` : ''}&nbsp;:
        <span className="opinion__cleavage">{proposition}</span>
      </p>
      {is_aggregate && child_propositions && child_propositions.length > 0 && (
        <details className="opinion__children">
          <summary>Voir les {child_propositions.length} objets de clivage des sous-thèmes</summary>
          <ul>
            {child_propositions.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </details>
      )}

      {/* Dashboard de cartes : engagement (grande) + positif / négatif (cliquables → avis). */}
      <div className="opinion__cards">
        <div className="opinion__card opinion__card--eng">
          <strong className="opinion__card-pct">{engPct}%</strong>
          <span className="opinion__card-label">des contributions expriment un sentiment net</span>
          <span className="opinion__card-sub">{engaged} sur {total}</span>
        </div>
        {engaged > 0 && (
          <>
            <button
              type="button"
              className="opinion__card opinion__card--pos"
              onClick={() => onSelectStance?.('favorable')}
              disabled={!onSelectStance}
              title={`Voir les ${fav} avis au sentiment positif`}
            >
              <strong className="opinion__card-pct">{pctFav}%</strong>
              <span className="opinion__card-label">sentiment positif</span>
              <span className="opinion__card-sub">{fav} avis{onSelectStance ? ' · voir →' : ''}</span>
            </button>
            <button
              type="button"
              className="opinion__card opinion__card--neg"
              onClick={() => onSelectStance?.('defavorable')}
              disabled={!onSelectStance}
              title={`Voir les ${def} avis au sentiment négatif`}
            >
              <strong className="opinion__card-pct">{pctDef}%</strong>
              <span className="opinion__card-label">sentiment négatif</span>
              <span className="opinion__card-sub">{def} avis{onSelectStance ? ' · voir →' : ''}</span>
            </button>
          </>
        )}
      </div>

      {/* Honnêteté : ces % décrivent les CONTRIBUTIONS REÇUES (participation volontaire,
          classement automatique) — pas l'opinion de la population. */}
      <p className="opinion__disclaimer" role="note">
        Ceci n'est pas un sondage : ces proportions décrivent les contributions reçues
        (participation volontaire, classement automatique par IA), pas la population générale.
      </p>

      {/* 3 ── (à venir) argument mining : arguments les plus mis en avant pour / contre */}
    </div>
  );
}
