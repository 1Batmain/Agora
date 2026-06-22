import type { Dataset } from '../types';
import type { Backend } from './contract';

export type ClusterMethod = 'leiden' | 'hdbscan';

/**
 * F5 — left column. Députés: dataset picker, search, filters (épuré). Analystes:
 * the same plus clustering knobs (resolution, backend, method) that re-run
 * `/analysis`.
 */
export function ToolsPanel({
  analyst,
  datasets,
  dataset,
  onDataset,
  query,
  onQuery,
  minConsensus,
  onMinConsensus,
  backend,
  onBackend,
  resolution,
  onResolution,
  method,
  onMethod,
  onRerun,
  busy,
}: {
  analyst: boolean;
  datasets: Dataset[];
  dataset: string | null;
  onDataset: (id: string) => void;
  query: string;
  onQuery: (q: string) => void;
  minConsensus: number;
  onMinConsensus: (v: number) => void;
  backend: Backend;
  onBackend: (b: Backend) => void;
  resolution: number;
  onResolution: (v: number) => void;
  method: ClusterMethod;
  onMethod: (m: ClusterMethod) => void;
  onRerun: () => void;
  busy: boolean;
}) {
  return (
    <div className="tools">
      <fieldset className="tools__group">
        <legend>Consultation</legend>
        <select
          className="tools__select"
          value={dataset ?? ''}
          disabled={busy || datasets.length === 0}
          onChange={(e) => onDataset(e.target.value)}
        >
          {datasets.length === 0 && <option value="">(aucun dataset)</option>}
          {datasets.map((d) => (
            <option key={d.id} value={d.id}>
              {d.label} ({d.n_nodes})
            </option>
          ))}
        </select>
      </fieldset>

      <fieldset className="tools__group">
        <legend>Recherche</legend>
        <input
          className="tools__input"
          type="search"
          placeholder="filtrer les thèmes…"
          value={query}
          onChange={(e) => onQuery(e.target.value)}
        />
      </fieldset>

      <fieldset className="tools__group">
        <legend>Filtres</legend>
        <label className="tools__knob">
          <span>
            consensus min <b>{Math.round(minConsensus * 100)}%</b>
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={minConsensus}
            onChange={(e) => onMinConsensus(Number(e.target.value))}
          />
        </label>
      </fieldset>

      {analyst && (
        <fieldset className="tools__group tools__group--analyst">
          <legend>Réglages (analystes)</legend>
          <label className="tools__field">
            <span>backend</span>
            <select
              className="tools__select"
              value={backend}
              disabled={busy}
              onChange={(e) => onBackend(e.target.value as Backend)}
            >
              <option value="auto">auto (API → repli)</option>
              <option value="api">API Mistral</option>
              <option value="mac">Mac (Ollama)</option>
            </select>
          </label>
          <label className="tools__field">
            <span>méthode</span>
            <select
              className="tools__select"
              value={method}
              disabled={busy}
              onChange={(e) => onMethod(e.target.value as ClusterMethod)}
            >
              <option value="leiden">Leiden</option>
              <option value="hdbscan">HDBSCAN</option>
            </select>
          </label>
          <label className="tools__knob">
            <span>
              résolution <b>{resolution.toFixed(1)}</b>
            </span>
            <input
              type="range"
              min={0.3}
              max={3}
              step={0.1}
              value={resolution}
              disabled={busy}
              onChange={(e) => onResolution(Number(e.target.value))}
            />
          </label>
          <button className="btn btn--primary" disabled={busy} onClick={onRerun}>
            {busy ? 'calcul…' : 'recalculer la carte'}
          </button>
        </fieldset>
      )}
    </div>
  );
}
