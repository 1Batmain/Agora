/**
 * Shared front-end types. Source de vérité = backend ; on ne garde ici que les
 * shapes réellement consommées par l'UI redesign (Consultation, Theme, SubmitResult).
 */

/** Theme-naming methods served by the backend (capacités exposées par /datasets). */
export type NamingMethod = 'ctfidf' | 'centroid' | 'llm';

export interface Theme {
  cluster_id: number;
  level: 0 | 1; // 0 = macro, 1 = sub-theme
  parent_id: number | null;
  children: number[]; // sub-theme cluster_ids (macros only)
  member_ids: string[];
  size: number;
  weight_sum: number;
  diversity?: number;
  consensus?: number;
  label: string;
  keywords?: string[];
  color: string;
}

/**
 * One consultation, from `GET /api/datasets`. MIROIR EXACT du TypedDict
 * `Consultation` backend (backend/consultation_schema.py) — source de vérité
 * unique, construite par `dataset_descriptor` / `open_consultation_descriptor`.
 * Le endpoint y ajoute aussi les capacités serveur `namings`/`default_naming`.
 */
export interface Consultation {
  id: string;
  label: string;
  /** Consultation status: 'open' (participation en cours) | 'closed' (analyse seule). */
  status: 'open' | 'closed';
  /** Échantillon réellement analysé (= ancien n_nodes). */
  n_sample: number;
  /** Nombre RÉEL de contributions reçues (avant cap d'échantillonnage). */
  n_contributions: number;
  /** Rétro-compat : alias historique de n_sample (toujours == n_sample). */
  n_nodes: number;
  languages: string[];
  lang_counts: Record<string, number>;
  source: string;
  /** Consultations OUVERTES (et clôturées qui en exposent un) : sujet affiché. */
  question?: string;
  context?: string;
  /** Capacités serveur (méthodes de nommage) — ajoutées par le endpoint /datasets. */
  namings?: NamingMethod[];
  default_naming?: NamingMethod;
}

/** `POST /submit` → corrélation instantanée d'une contribution citoyenne. */
export interface SubmitResult {
  ok: boolean;
  n_similar: number;
  nearest_excerpt: string | null;
  message: string;
}
