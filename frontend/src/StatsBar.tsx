import type { GraphStats } from './types';

/** Live read-out of `meta.stats` — builds the intuition "this knob does that".
 * The stats shown adapt to the clustering method: Leiden is hierarchical
 * (macros/sub-themes/modularity), HDBSCAN is flat (clusters/noise). */
export function StatsBar({ stats, live }: { stats: GraphStats; live: boolean }) {
  const hdbscan = stats.method === 'hdbscan';
  return (
    <div className="stats">
      {hdbscan ? (
        <>
          <Stat label="clusters" value={String(stats.n_clusters ?? stats.n_macros)} />
          <Stat label="non classés" value={String(stats.n_noise ?? 0)} />
        </>
      ) : (
        <>
          <Stat label="macros" value={String(stats.n_macros)} />
          <Stat label="sous-thèmes" value={String(stats.n_subs)} />
        </>
      )}
      <Stat label="avis" value={String(stats.n_nodes)} />
      {!hdbscan && (
        <Stat
          label="modularité"
          value={stats.modularity != null ? stats.modularity.toFixed(3) : '—'}
        />
      )}
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
