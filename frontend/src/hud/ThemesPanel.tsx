import { useMemo } from 'react';
import { buildThemeTree, type GraphIndex, type Theme } from '../lib/graphData';
import { useGraphStore } from '../state/useGraphStore';

interface ThemesPanelProps {
  graph: GraphIndex;
}

/**
 * Generic side panel — the only HUD we keep from the fork (dummy's business HUD
 * is jettisoned). The drill-down is now a macro→sub→avis TREE (hierarchical
 * Leiden, cross-lane contract):
 *
 *  1. Macro-themes, sorted by weight — label, keywords, size, weight_sum.
 *  2. Expanding a macro reveals its sub-themes (also weight-sorted).
 *  3. Clicking a sub-theme reveals the REAL member ideas (`node.props.text`) of
 *     that leaf community — transparency is a jury criterion.
 *
 * The swarm stays coloured by macro; emphasis follows the open macro / sub-theme.
 */
export function ThemesPanel({ graph }: ThemesPanelProps) {
  const expandedMacroId = useGraphStore((s) => s.expandedMacroId);
  const selectedClusterId = useGraphStore((s) => s.selectedClusterId);
  const toggleMacro = useGraphStore((s) => s.toggleMacro);
  const selectCluster = useGraphStore((s) => s.selectCluster);

  const tree = useMemo(() => buildThemeTree(graph), [graph]);
  const membersById = useMemo(() => graph.byId, [graph]);

  return (
    <aside className="panel">
      <header className="panel__head">
        <h1>Agora · Essaim citoyen</h1>
        <p className="panel__sub">
          {graph.nodes.length} avis · {tree.length} macro-thèmes (Leiden hiérarchique)
        </p>
      </header>

      <div className="panel__scroll">
        {tree.map(({ macro, subs }) => {
          const open = macro.cluster_id === expandedMacroId;
          return (
            <section key={macro.cluster_id} className={`macro ${open ? 'macro--open' : ''}`}>
              <button
                type="button"
                className="macro__head"
                onClick={() => toggleMacro(macro.cluster_id)}
                aria-expanded={open}
              >
                <span className="macro__caret">{open ? '▾' : '▸'}</span>
                <span className="theme__swatch" style={{ background: macro.color }} />
                <span className="macro__title">{macro.label}</span>
                <span className="theme__size">{macro.size}</span>
              </button>

              <div className="theme__keywords">
                {macro.keywords.map((k) => (
                  <span key={k} className="kw">
                    {k}
                  </span>
                ))}
              </div>

              {open && (
                <div className="macro__subs">
                  <p className="macro__meta">
                    poids {macro.weight_sum.toFixed(1)} · {subs.length} sous-thèmes
                  </p>
                  {subs.map((sub) => (
                    <SubTheme
                      key={sub.cluster_id}
                      sub={sub}
                      open={sub.cluster_id === selectedClusterId}
                      onToggle={() => selectCluster(sub.cluster_id)}
                      membersById={membersById}
                    />
                  ))}
                </div>
              )}
            </section>
          );
        })}
      </div>

      <footer className="panel__foot">
        Macro → sous-thème → avis sources. Phase 1 batch · données{' '}
        <code>graph.json</code>.
      </footer>
    </aside>
  );
}

interface SubThemeProps {
  sub: Theme;
  open: boolean;
  onToggle: () => void;
  membersById: GraphIndex['byId'];
}

function SubTheme({ sub, open, onToggle, membersById }: SubThemeProps) {
  return (
    <section className={`theme ${open ? 'theme--open' : ''}`}>
      <button type="button" className="theme__head" onClick={onToggle} aria-expanded={open}>
        <span className="theme__swatch" style={{ background: sub.color }} />
        <span className="theme__title">{sub.label}</span>
        <span className="theme__size">{sub.size}</span>
      </button>

      <div className="theme__keywords">
        {sub.keywords.map((k) => (
          <span key={k} className="kw">
            {k}
          </span>
        ))}
      </div>

      <dl className="theme__scores">
        <div>
          <dt>poids</dt>
          <dd>{sub.weight_sum.toFixed(1)}</dd>
        </div>
        <div>
          <dt>diversité</dt>
          <dd>{sub.diversity.toFixed(2)}</dd>
        </div>
        <div>
          <dt>consensus</dt>
          <dd>{sub.consensus.toFixed(2)}</dd>
        </div>
      </dl>

      {open && (
        <ul className="theme__members">
          {sub.member_ids.map((id) => {
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
}
