import { useEffect, useState } from 'react';
import { buildIndex, type GraphIndex, type GraphPayload } from './lib/graphData';
import { GraphScene } from './scene/GraphScene';
import { ThemesPanel } from './hud/ThemesPanel';

/**
 * Phase 1 (batch, no backend): load the static GraphPayload from /public and
 * render the swarm + themes panel. Phase 2 swaps this fetch for a WS snapshot
 * + incremental `addNodes` (see frontend/README.md).
 *
 * Source priority: the REAL consultation (`/graph.json`, produced by
 * `pipeline.cluster.build`) first, falling back to the committed fixture
 * (`/graph.sample.json`) when the real artefact is absent (e.g. fresh clone
 * before the pipeline has run).
 */
async function fetchPayload(url: string): Promise<GraphPayload> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as GraphPayload;
}

export default function App() {
  const [graph, setGraph] = useState<GraphIndex | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPayload('/graph.json')
      .catch(() => fetchPayload('/graph.sample.json'))
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
