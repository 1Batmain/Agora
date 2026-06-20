import type { GraphStats } from './types';

/** Live read-out of `meta.stats` — builds the intuition "this knob does that". */
export function StatsBar({ stats, live }: { stats: GraphStats; live: boolean }) {
  return (
    <div className="stats">
      <Stat label="macros" value={String(stats.n_macros)} />
      <Stat label="sous-thèmes" value={String(stats.n_subs)} />
      <Stat label="avis" value={String(stats.n_nodes)} />
      <Stat
        label="modularité"
        value={stats.modularity != null ? stats.modularity.toFixed(3) : '—'}
      />
      <Stat label="took" value={stats.took_ms != null ? `${Math.round(stats.took_ms)} ms` : '—'} />
      <span className={`stats__src ${live ? 'is-live' : ''}`}>{live ? '● live :8010' : '○ statique'}</span>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span className="stat__value">{value}</span>
      <span className="stat__label">{label}</span>
    </div>
  );
}
