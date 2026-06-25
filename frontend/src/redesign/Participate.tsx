import { useState } from 'react';
import { submitContribution } from '../api';
import type { Dataset, SubmitResult } from '../types';

/**
 * Vue PARTICIPER d'une consultation OUVERTE : affiche le sujet de la consultation
 * (titre + question/contexte) et un formulaire. À l'envoi, la contribution part
 * sur `POST /submit` : le backend l'embedde (nomic local, aucun LLM) et la corrèle
 * INSTANTANÉMENT aux retours déjà reçus → on affiche « N personnes ont déjà évoqué
 * un sujet proche : « … » » (ou « parmi les premiers ») + un remerciement.
 */
type Status = 'idle' | 'sending' | 'done' | 'error';

export function Participate({ dataset, onBack }: { dataset: Dataset; onBack: () => void }) {
  const [text, setText] = useState('');
  const [status, setStatus] = useState<Status>('idle');
  const [result, setResult] = useState<SubmitResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const value = text.trim();
    if (!value || status === 'sending') return;
    setStatus('sending');
    setError(null);
    try {
      const res = await submitContribution(dataset.id, value);
      setResult(res);
      setStatus('done');
      setText('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Envoi impossible.');
      setStatus('error');
    }
  }

  return (
    <div className="agora participate">
      <header className="gov-header">
        <button
          type="button"
          className="gov-header__brand gov-header__brand--home"
          onClick={onBack}
          title="Retour à l’accueil"
        >
          <div className="gov-logo" aria-hidden>
            <span className="gov-logo__mark">◆</span>
          </div>
          <div className="gov-header__title">
            <strong>Agora</strong>
            <span>Participer à la consultation</span>
          </div>
        </button>
        <div className="gov-header__right">
          <span className="ds-card__badge ds-card__badge--open">Ouvert</span>
        </div>
      </header>

      <main className="participate__body">
        <section className="participate__topic">
          <h1>{dataset.label}</h1>
          {dataset.question && <p className="participate__question">{dataset.question}</p>}
          {dataset.context && <p className="participate__lead">{dataset.context}</p>}
        </section>

        <form className="participate__form" onSubmit={onSubmit}>
          <label htmlFor="participate-text" className="term__label">votre_avis</label>
          <div className={`term${text ? ' term--typing' : ''}`}>
            <span className="term__prompt" aria-hidden>❯</span>
            <textarea
              id="participate-text"
              className="term__input"
              rows={8}
              placeholder="exprimez votre point de vue sur le sujet…"
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                if (status !== 'sending') setStatus('idle');
              }}
            />
            {!text && <span className="term__cursor" aria-hidden />}
          </div>
          <div className="participate__actions">
            <button
              type="submit"
              className="btn-primary"
              disabled={!text.trim() || status === 'sending'}
            >
              {status === 'sending' ? 'Analyse en cours…' : 'Envoyer'}
            </button>
            {status === 'sending' && (
              <span className="participate__note">
                <span className="spinner" /> corrélation à la parole déjà recueillie…
              </span>
            )}
            {status === 'error' && error && (
              <span className="participate__error">{error}</span>
            )}
          </div>
        </form>

        {status === 'done' && result && (
          <section className="participate__result" aria-live="polite">
            {result.n_similar > 0 ? (
              <>
                <p className="participate__insight">
                  <strong>{result.n_similar} personne{result.n_similar > 1 ? 's' : ''}</strong>{' '}
                  {result.n_similar > 1 ? 'ont' : 'a'} déjà évoqué un sujet proche du vôtre.
                </p>
                {result.nearest_excerpt && (
                  <blockquote className="participate__excerpt">
                    « {result.nearest_excerpt} »
                  </blockquote>
                )}
              </>
            ) : (
              <p className="participate__insight">
                Vous êtes parmi les premiers à soulever ce point ! 🎉
              </p>
            )}
            <p className="participate__thanks">
              Merci pour votre contribution — elle a bien été enregistrée et rejoint le débat.
            </p>
          </section>
        )}
      </main>
    </div>
  );
}
