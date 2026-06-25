import { useState } from 'react';
import type { Dataset } from '../types';

/**
 * Vue PARTICIPER (placeholder) pour une consultation OUVERTE : affiche le sujet
 * de la consultation et un formulaire « Donnez votre avis » (textarea + bouton).
 *
 * TODO(participation) : le formulaire n'est PAS encore câblé. À brancher sur le
 * futur endpoint `POST /submit` (puis corrélation de la contribution à la carte
 * d'analyse). Pour l'instant « Envoyer » ne fait qu'un retour visuel local.
 */
export function Participate({ dataset, onBack }: { dataset: Dataset; onBack: () => void }) {
  const [text, setText] = useState('');
  const [sent, setSent] = useState(false);

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    // TODO(participation) : appeler POST /submit avec { dataset: dataset.id, text }.
    setSent(true);
  }

  return (
    <div className="agora participate">
      <header className="gov-header">
        <div className="gov-header__brand">
          <button className="gov-header__back" onClick={onBack} title="Retour aux consultations">
            ← Consultations
          </button>
          <div className="gov-logo" aria-hidden>
            <span className="gov-logo__mark">◆</span>
          </div>
          <div className="gov-header__title">
            <strong>Agora</strong>
            <span>Participer à la consultation</span>
          </div>
        </div>
        <div className="gov-header__right">
          <span className="ds-card__badge ds-card__badge--open">Ouvert</span>
        </div>
      </header>

      <main className="participate__body">
        <section className="participate__topic">
          <h1>{dataset.label}</h1>
          <p className="participate__lead">
            Cette consultation est ouverte : votre contribution rejoindra les{' '}
            {dataset.n_nodes ? dataset.n_nodes.toLocaleString('fr-FR') : ''} avis déjà recueillis.
          </p>
        </section>

        <form className="participate__form" onSubmit={onSubmit}>
          <label htmlFor="participate-text">Donnez votre avis</label>
          <textarea
            id="participate-text"
            rows={8}
            placeholder="Exprimez votre point de vue sur le sujet de la consultation…"
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setSent(false);
            }}
          />
          <div className="participate__actions">
            <button type="submit" className="btn-primary" disabled={!text.trim()}>
              Envoyer
            </button>
            {sent && (
              <span className="participate__note">
                Merci ! (démo — la contribution n'est pas encore enregistrée)
              </span>
            )}
          </div>
          <p className="participate__todo">
            Formulaire de démonstration : l'envoi sera prochainement relié au
            backend et corrélé à l'analyse.
          </p>
        </form>
      </main>
    </div>
  );
}
