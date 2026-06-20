import { useMemo } from 'react';
import type { GraphIndex } from '../lib/graphData';
import { useGraphStore } from '../state/useGraphStore';

interface ThemesPanelProps {
  graph: GraphIndex;
}

/**
 * Generic side panel — the only HUD we keep from the fork (dummy's business HUD
 * is jettisoned). Two jobs, both jury-facing:
 *
 *  1. Themes list, sorted by weight — label, keywords, size, weight_sum,
 *     diversity, consensus (read straight from `themes`).
 *  2. Auditable drill-down (transparency = jury criterion): clicking a theme
 *     reveals the REAL member ideas (`node.props.text`) of that Leiden cluster.
 */
export function ThemesPanel({ graph }: ThemesPanelProps) {
  const selectedClusterId = useGraphStore((s) => s.selectedClusterId);
  const selectCluster = useGraphStore((s) => s.selectCluster);

  const themes = useMemo(
    () => [...graph.themes].sort((a, b) => b.weight_sum - a.weight_sum),
    [graph],
  );

  const membersById = useMemo(() => graph.byId, [graph]);

  return (
    <aside className="panel">
      <header className="panel__head">
        <h1>Agora · Essaim citoyen</h1>
        <p className="panel__sub">
          {graph.nodes.length} avis · {themes.length} thèmes (communautés Leiden)
        </p>
      </header>

      <div className="panel__scroll">
        {themes.map((t) => {
          const open = t.cluster_id === selectedClusterId;
          return (
            <section key={t.cluster_id} className={`theme ${open ? 'theme--open' : ''}`}>
              <button
                type="button"
                className="theme__head"
                onClick={() => selectCluster(t.cluster_id)}
                aria-expanded={open}
              >
                <span className="theme__swatch" style={{ background: t.color }} />
                <span className="theme__title">{t.label}</span>
                <span className="theme__size">{t.size}</span>
              </button>

              <div className="theme__keywords">
                {t.keywords.map((k) => (
                  <span key={k} className="kw">
                    {k}
                  </span>
                ))}
              </div>

              <dl className="theme__scores">
                <div>
                  <dt>poids</dt>
                  <dd>{t.weight_sum.toFixed(1)}</dd>
                </div>
                <div>
                  <dt>diversité</dt>
                  <dd>{t.diversity.toFixed(2)}</dd>
                </div>
                <div>
                  <dt>consensus</dt>
                  <dd>{t.consensus.toFixed(2)}</dd>
                </div>
              </dl>

              {open && (
                <ul className="theme__members">
                  {t.member_ids.map((id) => {
                    const n = membersById.get(id);
                    if (!n) return null;
                    return (
                      <li key={id} className="member">
                        <span className="member__id">{id}</span>
                        <span className="member__text">{n.props.text}</span>
                      </li>
                    );
                  })}
                </ul>
              )}
            </section>
          );
        })}
      </div>

      <footer className="panel__foot">
        Cliquez un thème (ou un nœud) → ses avis sources. Phase 1 batch ·
        données <code>graph.sample.json</code>.
      </footer>
    </aside>
  );
}
