/**
 * Client for `GET /density?dataset=X` — the 3D density landscape (UMAP 2D + KDE).
 * Goes through the shared `http.ts` layer (vite `/api/*` proxy → :8010), like the
 * other redesign clients. The backend computes lazily then caches to disk, so the
 * FIRST call on a dataset can be slow (UMAP) — the generous `TIMEOUT_MS` covers it.
 */
import { rawFetch } from './http';

/** Grid surface returned by `/density`: heights[iz][ix] over a nx×nz lattice. */
export interface DensityPayload {
  nx: number;
  nz: number;
  x_range: [number, number];
  z_range: [number, number];
  heights: number[][];
  zmax: number;
}

/**
 * Fetch the density landscape for a dataset. Returns `null` when the surface is
 * unavailable (e.g. 503: UMAP not installed and no cache) so the caller can show a
 * graceful message instead of throwing.
 */
export async function fetchDensity(dataset: string): Promise<DensityPayload | null> {
  const { status, body } = await rawFetch(`/density?dataset=${encodeURIComponent(dataset)}`);
  if (status === 200 && body && Array.isArray(body.heights)) {
    return body as DensityPayload;
  }
  return null;
}
