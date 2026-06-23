import type { Dataset } from './types';

/**
 * Backend client. Everything goes through the vite proxy at `/api/*` → :8010.
 * The redesign UI reads the precomputed spatial analysis via `redesign/analysisApi.ts`;
 * this module only exposes the dataset selector source.
 */

// Généreux : la découverte de datasets est rapide, mais on garde une marge.
const TIMEOUT_MS = 180000;

async function jsonFetch(url: string, init?: RequestInit, timeoutMs = TIMEOUT_MS): Promise<any> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { ...init, signal: ctrl.signal });
    if (!r.ok) {
      // Surface the backend's `detail` (e.g. 503 = Mac unreachable) when present.
      const detail = await r
        .clone()
        .json()
        .then((b) => (b && typeof b.detail === 'string' ? b.detail : null))
        .catch(() => null);
      throw new Error(detail ? `HTTP ${r.status} — ${detail}` : `HTTP ${r.status}`);
    }
    return await r.json();
  } finally {
    clearTimeout(t);
  }
}

/** List the datasets the backend has a cache for (populates the selector). */
export async function fetchDatasets(): Promise<Dataset[]> {
  const raw = await jsonFetch('/api/datasets');
  return Array.isArray(raw) ? (raw as Dataset[]) : [];
}
