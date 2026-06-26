import type { Consultation, SubmitResult } from './redesign/contract';
import { rawFetch } from './redesign/http';

/**
 * Backend client. Everything goes through the vite proxy at `/api/*` → :8010.
 * The redesign UI reads the precomputed spatial analysis via `redesign/analysisApi.ts`;
 * this module only exposes the dataset selector source.
 *
 * Shares the base / timeout / `rawFetch` primitive with `analysisApi.ts` (see
 * `redesign/http.ts`); on top of it this client throws on a non-2xx response,
 * surfacing the backend's `detail` when present.
 */

/** rawFetch + throw on non-2xx (surfacing the backend `detail`); returns the JSON body. */
async function jsonFetch(path: string, init?: RequestInit): Promise<any> {
  const { status, body } = await rawFetch(path, init);
  if (status < 200 || status >= 300) {
    // Surface the backend's `detail` (e.g. 503 = Mac unreachable) when present.
    const detail = body && typeof body.detail === 'string' ? body.detail : null;
    throw new Error(detail ? `HTTP ${status} — ${detail}` : `HTTP ${status}`);
  }
  return body;
}

/** List the datasets the backend has a cache for (populates the selector). */
export async function fetchDatasets(): Promise<Consultation[]> {
  const raw = await jsonFetch('/datasets');
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
  return (await jsonFetch('/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ consultation_id: consultationId, text }),
  })) as SubmitResult;
}
