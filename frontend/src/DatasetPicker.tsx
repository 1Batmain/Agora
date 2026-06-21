import type { Dataset } from './types';

/**
 * Dataset selector — populated from `GET /api/datasets`. Changing it reclusters
 * with the new `dataset` and re-renders. Below the dropdown we surface the
 * current dataset's metadata (n avis, langues). When a dataset is multilingual
 * we highlight the linguistic mix — the whole point of x-stance is that themes
 * blend the languages instead of separating by them.
 */
export function DatasetPicker({
  datasets,
  current,
  disabled,
  onChange,
}: {
  datasets: Dataset[];
  current: string | null;
  disabled: boolean;
  onChange: (id: string) => void;
}) {
  const ds = datasets.find((d) => d.id === current) ?? null;
  const langs = ds?.languages ?? [];
  const multilingual = langs.length > 1;

  return (
    <div className="dataset">
      <div className="panel__head">
        <h2>Jeu de données</h2>
      </div>
      <select
        className="dataset__select"
        value={current ?? ''}
        disabled={disabled || datasets.length === 0}
        onChange={(e) => onChange(e.target.value)}
      >
        {datasets.map((d) => (
          <option key={d.id} value={d.id}>
            {d.label}
          </option>
        ))}
      </select>

      {ds && (
        <div className="dataset__meta">
          <span className="dataset__n">{ds.n_nodes.toLocaleString('fr-FR')} avis</span>
          <div className="dataset__langs">
            {langs.map((lg) => (
              <span key={lg} className="lang-chip">
                {lg}
                {ds.lang_counts?.[lg] != null && (
                  <em>{ds.lang_counts[lg].toLocaleString('fr-FR')}</em>
                )}
              </span>
            ))}
          </div>
          {multilingual && (
            <p className="dataset__mix">
              Multilingue — les thèmes <strong>mélangent les langues</strong> (regroupement
              par sujet, pas par langue).
            </p>
          )}
        </div>
      )}
    </div>
  );
}
