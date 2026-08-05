/**
 * Client de la vue LIVE — pipeline expérimental de synthèse d'un débat rejoué.
 *
 * Cible un backend SÉPARÉ (`live.server`, :8020), pas l'API des consultations (:8010).
 * La séparation est volontaire : le pipeline live est un chantier de recherche, il ne doit
 * ni toucher aux caches servis ni pouvoir dégrader la Console. Le proxy vite `/live-api`
 * garde l'appel same-origin (cf. `vite.config.ts`).
 *
 * Le backend live ne CALCULE rien : il sert des instantanés écrits par `live.replay`.
 * Si aucun rejeu n'a tourné, `/snapshot` répond 404 — c'est un état normal, pas une panne.
 */

/** Préfixe proxy vite → :8020. */
export const LIVE_BASE = '/live-api';

/** Court : cette vue rafraîchit toutes les 2 s, une requête qui traîne doit céder la place. */
const TIMEOUT_MS = 8000;

export interface LiveTheme {
  id: string;
  label: string;
  keywords: string[];
  cleavage: string;
  cleavage_justif: string;
  n_bootstrap: number;
}

export interface ThemeOpinionRow {
  theme_id: string;
  label: string;
  cleavage: string;
  n_claims: number;
  favorable: number;
  defavorable: number;
  nuance: number;
  /** `null` (et non 0) quand aucun claim n'est rattaché : un dénominateur inventé mentirait. */
  part_favorable: number | null;
}

export interface SpeakerRow {
  speaker_id: string;
  speaker: string;
  theme_id: string;
  n_claims: number;
  favorable: number;
  defavorable: number;
  nuance: number;
  position: 'favorable' | 'défavorable' | 'partagé' | 'sans position';
}

export interface Drift {
  global: number;
  recent: number;
  window: number;
  n_uncovered: number;
  /** Part de claims hors thème ATTENDUE même sans dérive (construction du seuil). */
  baseline: number;
  /** `recent - baseline`, borné à 0 : le SEUL chiffre qui signale une dérive thématique. */
  excess: number;
}

export interface LiveSnapshot {
  session: string;
  topic: string;
  turns_seen: number;
  claims_seen: number;
  coverage_threshold: number;
  themes: LiveTheme[];
  theme_counts: Record<string, number>;
  theme_opinion: ThemeOpinionRow[];
  drift: Drift;
  speakers: SpeakerRow[];
  recent_claims: LiveClaim[];
}

export interface LiveClaim {
  claim_id: string;
  turn_id: string;
  seq: number;
  stime: number | null;
  speaker: string;
  speaker_id: string | null;
  text: string;
  theme_id: string | null;
  similarity: number;
  stance: 'favorable' | 'defavorable' | 'nuance' | null;
  confidence: 'high' | 'medium' | 'low' | null;
  justif: string;
}

export interface LiveTurn {
  id: string;
  seq: number;
  stime: number | null;
  speaker: string;
  speaker_id: string | null;
  text: string;
  n_reactions: number;
}

async function get<T>(path: string): Promise<T | null> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const r = await fetch(LIVE_BASE + path, { signal: ctrl.signal });
    if (!r.ok) return null;           // 404 = aucun rejeu en cours, pas une erreur
    return (await r.json()) as T;
  } catch {
    return null;                      // backend live éteint : la vue le dit, elle ne casse pas
  } finally {
    clearTimeout(t);
  }
}

export const fetchLiveSnapshot = () => get<LiveSnapshot>('/snapshot');

export const fetchLiveTranscript = (limit = 25) =>
  get<{ turns: LiveTurn[]; total?: number }>(`/transcript?limit=${limit}`);
