import { useState } from 'react';
import { submitContribution } from '../api';
import { Header } from './Header';
import type { Consultation, SubmitResult } from './contract';

/**
 * Vue PARTICIPER d'une consultation OUVERTE : affiche le sujet de la consultation
 * (titre + question/contexte) et un formulaire. À l'envoi, la contribution part
 * sur `POST /submit` : le backend l'embedde (nomic local, aucun LLM) et la corrèle
 * INSTANTANÉMENT aux retours déjà reçus → on affiche « N personnes ont déjà évoqué
 * un sujet proche : « … » » (ou « parmi les premiers ») + un remerciement.
 */
type Status = 'idle' | 'sending' | 'done' | 'error';

export function Participate({ dataset, onBack }: { dataset: Consultation; onBack: () => void }) {
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
      <Header
        onHome={onBack}
        right={<span className="ds-card__badge ds-card__badge--open">Ouvert</span>}
      />

      <main className="participate__body">
        <section className="participate__topic">
          <h1>{dataset.label}</h1>
          {dataset.question && <p className="participate__question">{dataset.question}</p>}
          {dataset.context && <p className="participate__lead">{dataset.context}</p>}
        </section>

        <form className="participate__form" onSubmit={onSubmit}>
          <textarea
            id="participate-text"
            className="participate__input"
            rows={6}
            placeholder="Écrivez votre avis ici…"
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              if (status !== 'sending') setStatus('idle');
            }}
          />
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
