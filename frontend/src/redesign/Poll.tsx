/**
 * « Sondage » partagé : les sous-thèmes dominants d'un niveau en barres de pourcentage
 * (part des voix). Brique du TEMPLATE de synthèse, réutilisée à chaque niveau (synthèse
 * globale = top macros ; synthèse d'un thème = ses sous-thèmes). Couleur = celle du thème.
 */
export type PollItem = { label: string; value: number; color?: string };

export function Poll({ items, total }: { items: PollItem[]; total: number }) {
  if (!items.length || total <= 0) return null;
  return (
    <div className="poll" aria-label="Répartition des voix par sous-thème">
      {items.map((b) => {
        const pct = Math.round((b.value / total) * 100);
        return (
          <div className="poll__row" key={b.label}>
            <span className="poll__label" title={b.label}>{b.label}</span>
            <span className="poll__track">
              <span
                className="poll__fill"
                style={{ width: `${pct}%`, background: b.color || 'var(--agora, #000091)' }}
              />
            </span>
            <span className="poll__pct">{pct}%</span>
          </div>
        );
      })}
    </div>
  );
}
