import type { Consultation, SubmitResult } from './types';

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
export async function fetchDatasets(): Promise<Consultation[]> {
  const raw = await jsonFetch('/api/datasets');
  return Array.isArray(raw) ? (raw as Consultation[]) : [];
}

/**
 * Envoie une contribution sur une consultation OUVERTE. Le backend l'embedde
 * (nomic local) et la corrèle aux retours déjà reçus → `{n_similar, nearest_excerpt}`.
 * L'embedding du modèle peut prendre quelques secondes au 1ᵉʳ appel (chargement lazy).
 */
export async function submitContribution(
  consultationId: string,
  text: string,
): Promise<SubmitResult> {
  return (await jsonFetch('/api/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ consultation_id: consultationId, text }),
  })) as SubmitResult;
}
