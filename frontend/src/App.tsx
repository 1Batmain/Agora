import { useEffect, useState } from 'react';
import { buildIndex, type GraphIndex, type GraphPayload } from './lib/graphData';
import { GraphScene } from './scene/GraphScene';
import { ThemesPanel } from './hud/ThemesPanel';

/**
 * Phase 1 (batch, no backend): load the static GraphPayload from /public and
 * render the swarm + themes panel. Phase 2 swaps this fetch for a WS snapshot
 * + incremental `addNodes` (see frontend/README.md).
 */
export default function App() {
  const [graph, setGraph] = useState<GraphIndex | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/graph.sample.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<GraphPayload>;
      })
      .then((payload) => setGraph(buildIndex(payload)))
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="app">
      <div className="app__scene">
        {graph && <GraphScene graph={graph} />}
        {!graph && (
          <div className="app__status">{error ? `Erreur : ${error}` : 'Chargement de l’essaim…'}</div>
        )}
      </div>
      {graph && <ThemesPanel graph={graph} />}
    </div>
  );
}
