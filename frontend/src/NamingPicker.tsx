import type { NamingMethod } from './types';

/**
 * Theme-NAMING switch (c-TF-IDF ⇄ Centroïde ⇄ LLM), next to the method/dataset
 * pickers. Orthogonal to the clustering method: it only changes how each cluster
 * is *titled*, not which clusters exist — so switching reclusters but keeps the
 * same knobs.
 *   - c-TF-IDF : distinctive keywords derived from the corpus (default).
 *   - Centroïde: the most representative citizen verbatim (medoid).
 *   - LLM      : short titles from the Mistral API (batched); falls back to
 *                c-TF-IDF if the key is missing / API is down (signalled discreetly).
 */
const NAMINGS: { id: NamingMethod; label: string; hint: string }[] = [
  { id: 'ctfidf', label: 'c-TF-IDF', hint: 'mots-clés distinctifs (défaut)' },
  { id: 'centroid', label: 'Centroïde', hint: 'verbatim citoyen représentatif' },
  { id: 'llm', label: 'LLM (Mistral)', hint: 'titre court via Mistral, repli c-TF-IDF' },
];

export function NamingPicker({
  current,
  fallback,
  disabled,
  onChange,
}: {
  current: NamingMethod;
  fallback: boolean;
  disabled: boolean;
  onChange: (n: NamingMethod) => void;
}) {
  const hint = NAMINGS.find((n) => n.id === current)?.hint ?? '';
  // Discreet notice when the user asked for LLM but Mistral was unavailable → c-TF-IDF.
  const fellBack = fallback && current === 'llm';
  return (
    <div className="method">
      <div className="panel__head">
        <h2>Nommage</h2>
      </div>
      <div className="method__toggle" role="group" aria-label="Méthode de nommage">
        {NAMINGS.map((n) => (
          <button
            key={n.id}
            type="button"
            className={`method__btn ${current === n.id ? 'is-active' : ''}`}
            disabled={disabled}
            aria-pressed={current === n.id}
            onClick={() => onChange(n.id)}
          >
            {n.label}
          </button>
        ))}
      </div>
      <p className="method__hint">
        {hint}
        {fellBack && <span className="method__warn"> · Mistral indisponible → repli c-TF-IDF</span>}
      </p>
    </div>
  );
}
