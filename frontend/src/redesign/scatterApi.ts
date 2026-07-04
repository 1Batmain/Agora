/**
 * Client for `GET /scatter?dataset=X` — the real UMAP 2D scatter (one point per
 * contribution). Goes through the shared `http.ts` layer (vite `/api/*` proxy
 * → :8010), like the other redesign clients.
 *
 * Each point is a REAL contribution with its UMAP 2D coordinates, cluster,
 * color, id and a text excerpt. All precomputed from cache — zero computation
 * at request time.
 */
import { rawFetch } from './http';

/** A single scatter point — one real contribution in UMAP 2D space. */
export interface ScatterRealPoint {
  x: number;
  z: number;
  id: string;
  cluster_id: string | null;
  color: string;
  text: string;
}

/** Payload returned by `/scatter`. */
export interface ScatterPayload {
  points: ScatterRealPoint[];
  total: number;
  returned: number;
}

/**
 * Fetch the real scatter points for a dataset. Returns `null` when unavailable
 * (e.g. 503: UMAP not installed and no cache) so the caller can show a graceful
 * message instead of throwing.
 */
export async function fetchScatter(dataset: string): Promise<ScatterPayload | null> {
  const { status, body } = await rawFetch(`/scatter?dataset=${encodeURIComponent(dataset)}`);
  if (status === 200 && body && Array.isArray(body.points)) {
    return body as ScatterPayload;
  }
  return null;
}
