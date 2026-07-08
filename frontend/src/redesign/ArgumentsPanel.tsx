import { useEffect, useMemo, useState } from 'react';
import type { MinedArgument, ThemeArguments } from './contract';
import { stripMd } from './strings';

/**
 * Arguments minés d'un thème — le « 3. » annoncé par OpinionBar : les arguments les
 * plus mis en avant pour / contre (ou une liste neutre quand le thème n'a pas de
 * clivage baké). Quand les DEUX polarités existent, un sélecteur pour/contre
 * n'affiche qu'un côté à la fois, la POSITION MAJORITAIRE (au support) par défaut.
 *
 * Honnêteté : chaque argument est une contribution citoyenne VERBATIM (sélectionnée
 * par IA, jamais reformulée — V-SELECT), et il n'existe que s'il est adossé à
 * d'autres contributions réelles (fail-closed côté build) — les sources verbatim
 * sont dépliables et cliquables vers l'avis d'origine.
 */

// Même langage visuel de stance que AvisDetail.STANCE_META (vert positif / rouge négatif).
const STANCE_UI: Record<string, { label: string; tab: string; className: string }> = {
  pour: { label: 'Arguments pour', tab: 'Pour', className: 'argmine__col--pour' },
  contre: { label: 'Arguments contre', tab: 'Contre', className: 'argmine__col--contre' },
  neutre: { label: 'Arguments les plus avancés', tab: 'Tous', className: 'argmine__col--neutre' },
};

function ArgumentCard({
  arg,
  onExploreAvis,
}: {
  arg: MinedArgument;
  onExploreAvis?: (avisId: string) => void;
}) {
  return (
    <li className="argmine__arg">
      <p className="argmine__text">{stripMd(arg.argument)}</p>
      <span className="argmine__badge" title="Contributions distinctes soutenant cet argument">
        {arg.n_support} contribution{arg.n_support > 1 ? 's' : ''}
        {arg.share != null ? ` · ${Math.round(arg.share * 100)}%` : ''}
      </span>
      {arg.sources.length > 0 && (
        <details className="argmine__sources">
          <summary>
            Voir {arg.sources.length} extrait{arg.sources.length > 1 ? 's' : ''} source
          </summary>
          <ul>
            {arg.sources.map((s) => (
              <li key={s.claim_id}>
                <button
                  type="button"
                  className="argmine__source"
                  onClick={() => onExploreAvis?.(s.avis_id)}
                  disabled={!onExploreAvis}
                  title={onExploreAvis ? "Ouvrir l'avis source" : undefined}
                >
                  <q>{s.text}</q>
                  {onExploreAvis && <span className="argmine__source-go"> · voir l'avis →</span>}
                </button>
              </li>
            ))}
          </ul>
        </details>
      )}
    </li>
  );
}

export function ArgumentsPanel({
  args,
  onExploreAvis,
}: {
  args: ThemeArguments;
  /** Clic sur un extrait source → page d'exploration FOCALISÉE sur l'avis. */
  onExploreAvis?: (avisId: string) => void;
}) {
  // En mode pour/contre, les DEUX pilules sont toujours proposées (même si un côté
  // n'a aucun argument suffisamment sourcé — on l'explique plutôt que de le cacher),
  // triées par SUPPORT décroissant → la 1re est la position majoritaire, par défaut.
  const stances = useMemo(() => {
    const support = (s: string) =>
      args.arguments.filter((a) => a.stance === s).reduce((t, a) => t + a.n_support, 0);
    const base = args.mode === 'pour_contre' ? ['pour', 'contre'] : ['neutre'];
    return base
      .filter((s) => s === 'pour' || s === 'contre' || args.arguments.some((a) => a.stance === s))
      .sort((a, b) => support(b) - support(a));
  }, [args.arguments, args.mode]);
  const [stance, setStance] = useState(stances[0]);
  useEffect(() => setStance(stances[0]), [args.theme_id, stances]);

  if (!args.arguments.length || !stance) return null;
  const shown = args.arguments.filter((a) => a.stance === stance);
  return (
    <div className="argmine" aria-label="Arguments principaux du thème">
      <div className="argmine__head">
        <span className="argmine__label">
          {args.is_aggregate
            ? `Arguments principaux · synthèse de ${args.n_children ?? 0} sous-thèmes`
            : 'Arguments principaux'}
        </span>
        {stances.length > 1 && (
          <div className="argmine__tabs" role="tablist" aria-label="Polarité des arguments">
            {stances.map((s) => {
              const n = args.arguments
                .filter((a) => a.stance === s)
                .reduce((t, a) => t + a.n_support, 0);
              return (
                <button
                  key={s}
                  type="button"
                  role="tab"
                  aria-selected={stance === s}
                  className={`argmine__tab argmine__tab--${s}${stance === s ? ' argmine__tab--on' : ''}`}
                  onClick={() => setStance(s)}
                >
                  {STANCE_UI[s].tab} · {n}
                </button>
              );
            })}
          </div>
        )}
      </div>
      <section className={`argmine__col ${STANCE_UI[stance].className}`}>
        <h4 className="argmine__coltitle">{STANCE_UI[stance].label}</h4>
        {shown.length > 0 ? (
          <ul className="argmine__list">
            {shown.map((a) => (
              <ArgumentCard key={a.id} arg={a} onExploreAvis={onExploreAvis} />
            ))}
          </ul>
        ) : (
          <p className="argmine__empty">
            Aucun argument {stance === 'pour' ? 'favorable' : 'défavorable'} n'est ressorti avec
            assez de contributions pour être affiché (seuil de sourçage non atteint).
          </p>
        )}
      </section>
      <p className="argmine__disclaimer" role="note">
        Arguments reformulés par IA à partir des contributions : chacun est adossé à des
        contributions réelles — dépliez les extraits pour les auditer mot à mot.
      </p>
    </div>
  );
}
