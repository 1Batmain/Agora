import type { ViewMode } from './types';

/**
 * View switch: the existing CLUSTER views (Leiden/HDBSCAN circle-pack) vs the
 * emergent-CLAIMS map (avis → claims → bottom-up themes). Switching only changes
 * what the center/right render — it never touches the cluster state, so the
 * existing views keep working untouched.
 */
const VIEWS: { id: ViewMode; label: string; hint: string }[] = [
  { id: 'clusters', label: 'Clusters', hint: 'avis regroupés (Leiden / HDBSCAN)' },
  { id: 'claims', label: 'Thèmes émergents', hint: 'claims atomiques → thèmes (ouvert, sans taxo)' },
];

export function ViewPicker({
  current,
  disabled,
  onChange,
}: {
  current: ViewMode;
  disabled: boolean;
  onChange: (v: ViewMode) => void;
}) {
  const hint = VIEWS.find((v) => v.id === current)?.hint ?? '';
  return (
    <div className="method">
      <div className="panel__head">
        <h2>Vue</h2>
      </div>
      <div className="method__toggle" role="group" aria-label="Mode de vue">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            type="button"
            className={`method__btn ${current === v.id ? 'is-active' : ''}`}
            disabled={disabled}
            aria-pressed={current === v.id}
            onClick={() => onChange(v.id)}
          >
            {v.label}
          </button>
        ))}
      </div>
      <p className="method__hint">{hint}</p>
    </div>
  );
}
