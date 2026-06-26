/**
 * Client for `POST /recluster {dataset, knn_threshold}` — the LIVE re-clustering
 * that powers the Console. The backend rebuilds the theme map on the fly by varying
 * the k-NN edge threshold (Leiden + variance-adaptive subdivision, c-TF-IDF naming,
 * M5 indices, UMAP-2D points) with ZERO LLM calls, so it answers in ~1-2 s.
 *
 * Goes through the shared `http.ts` layer (vite `/api/*` proxy → :8010) like the
 * other redesign clients. `knn_threshold=null` → the backend DERIVES the default
 * threshold from the dataset (the Console boots exactly like `/analysis`).
 */
import type { DatasetStats, SpatialTheme } from './contract';
import { rawFetch } from './http';

/** One UMAP-2D point (one idea), coloured by its macro cluster. */
export interface ScatterPoint {
  x: number;
  z: number;
  cluster_id: string | null;
  color: string;
}

/** Meta block of a `/recluster` response (only the fields the Console consumes). */
export interface ReclusterMeta {
  dataset: string;
  knn_threshold: number | null;
  knn_threshold_default: number | null;
  n_themes: number;
  n_macros: number;
  n_ideas: number;
  n_points: number;
  took_ms: number;
}

/** `POST /recluster` → live theme map + scatter points + indices + meta. */
export interface ReclusterPayload {
  themes: SpatialTheme[];
  points: ScatterPoint[];
  indices: DatasetStats;
  meta: ReclusterMeta;
}

/**
 * Re-cluster a dataset at the given k-NN threshold. `knn_threshold=null` asks the
 * backend for the derived default. Returns `null` when the backend can't serve the
 * map (e.g. 503: vectors not cached) so the Console can show a graceful message.
 */
export async function fetchRecluster(
  dataset: string,
  knnThreshold: number | null,
): Promise<ReclusterPayload | null> {
  const { status, body } = await rawFetch('/recluster', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset, knn_threshold: knnThreshold }),
  });
  if (status === 200 && body && Array.isArray(body.themes) && Array.isArray(body.points)) {
    return body as ReclusterPayload;
  }
  return null;
}
