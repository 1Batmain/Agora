import type { ClusterMethod } from './types';

/**
 * Clustering-method switch (Leiden ⇄ HDBSCAN), next to the dataset picker.
 * Switching pulls THAT method's knobs from `/params?method=…` and reclusters,
 * so you can compare the two on the same corpus/embeddings by eye.
 *   - Leiden  : hierarchical macro → sub-themes (default).
 *   - HDBSCAN : UMAP-5D → flat clusters + an "unclassified" (noise) group.
 */
const METHODS: { id: ClusterMethod; label: string; hint: string }[] = [
  { id: 'leiden', label: 'Leiden', hint: 'hiérarchique (macro → sous-thèmes)' },
  { id: 'hdbscan', label: 'HDBSCAN', hint: 'UMAP-5D → clusters plats + bruit' },
];

export function MethodPicker({
  current,
  disabled,
  onChange,
}: {
  current: ClusterMethod;
  disabled: boolean;
  onChange: (m: ClusterMethod) => void;
}) {
  const hint = METHODS.find((m) => m.id === current)?.hint ?? '';
  return (
    <div className="method">
      <div className="panel__head">
        <h2>Méthode</h2>
      </div>
      <div className="method__toggle" role="group" aria-label="Méthode de clustering">
        {METHODS.map((m) => (
          <button
            key={m.id}
            type="button"
            className={`method__btn ${current === m.id ? 'is-active' : ''}`}
            disabled={disabled}
            aria-pressed={current === m.id}
            onClick={() => onChange(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>
      <p className="method__hint">{hint}</p>
    </div>
  );
}
