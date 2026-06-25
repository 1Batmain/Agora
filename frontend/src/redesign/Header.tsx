import type { ReactNode } from 'react';

/**
 * Header PARTAGÉ d'Agora (issu de la landing). Réutilisé sur TOUTES les vues
 * (landing, analyse, participer) pour une identité cohérente.
 *
 *  - `onHome` : si fourni, le brand (logo + titre) devient cliquable et ramène à
 *    l'accueil (curseur main). Absent (sur la landing) → brand statique.
 *  - `right`  : contenu optionnel à droite (nom de la consultation, badge statut…).
 */
export function Header({ onHome, right }: { onHome?: () => void; right?: ReactNode }) {
  const brandInner = (
    <>
      <div className="gov-logo" aria-hidden>
        <span className="gov-logo__mark">◆</span>
      </div>
      <div className="gov-header__title">
        <strong>Agora</strong>
        <span>L'IA au service de la démocratie</span>
      </div>
    </>
  );
  return (
    <header className="gov-header">
      {onHome ? (
        <button
          type="button"
          className="gov-header__brand gov-header__brand--home"
          onClick={onHome}
          title="Retour à l’accueil"
        >
          {brandInner}
        </button>
      ) : (
        <div className="gov-header__brand">{brandInner}</div>
      )}
      {right && <div className="gov-header__right">{right}</div>}
    </header>
  );
}
