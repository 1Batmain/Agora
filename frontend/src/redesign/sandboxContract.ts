/**
 * Cross-lane contract for the LIVE RECLUSTER sandbox (analyst "console de
 * mixage"). Mirrors `/tmp/contract-sandbox.md` EXACTLY so the front codes against
 * the frozen shape; the backend (lane B) computes the recluster (blend → kNN →
 * Leiden → coarsen → subdivide → c-TF-IDF, NO LLM, ~1s for 3000 claims) and the
 * front only drives the faders and renders the result + its decision trace.
 *
 *   POST /sandbox  {dataset, alpha?, k?, resolution?, coarsen_mult?, tau_mult?} -> SandboxResponse
 *   GET  /explain  {dataset, cluster=nX}      -> ExplainCluster   (k nearest clusters + node criteria)
 *   GET  /explain  {dataset, pair=nX,nY}      -> ExplainPair      (sim / threshold / cohesions)
 */

/**
 * The five mixing-console faders. ALL optional → the backend derives sane defaults
 * from the data (returned under `params.derived`). Values are the raw fader knobs:
 *   - alpha        0..1   weight of the CIBLE (target) in the blended embedding;
 *   - k            int    kNN graph connectivity;
 *   - resolution   float  Leiden resolution (↑ → more, finer clusters);
 *   - coarsen_mult float  multiplier on the μ+σ merge threshold (↑ → merges more → fewer);
 *   - tau_mult     float  multiplier on the subdivision threshold τ (↑ → less subdivision).
 */
export interface SandboxParams {
  alpha?: number;
  k?: number;
  resolution?: number;
  coarsen_mult?: number;
  tau_mult?: number;
}

/** The fader set with everything resolved (defaults filled), echoed back by /sandbox. */
export interface ResolvedParams extends Required<SandboxParams> {
  /** Backend-derived raw values behind the multipliers (μ+σ seuil, τ, default k…). */
  derived: Record<string, number>;
}

/** One cluster of the reclustered tree (top-level or a subdivided child). */
export interface SandboxCluster {
  id: string;
  parent_id: string | null;
  n_claims: number;
  n_avis: number;
  keywords: string[];
  sample_claims: string[];
  cohesion: number; // 0..1 internal coherence (c-TF-IDF / centroid tightness)
}

/** One candidate MERGE between two clusters, with the criteria that decided it. */
export interface TracePair {
  a: string;
  b: string;
  sim: number; // centroid cosine similarity
  threshold: number; // the coarsen threshold it was tested against
  cohesion_min: number; // min(cohesion_a, cohesion_b) — the merge guard
  merged: boolean; // did a+b actually get coarsened together?
}

/** One node's SUBDIVISION decision (dispersion vs τ). */
export interface TraceNode {
  id: string;
  dispersion: number; // internal spread
  tau: number; // the subdivision threshold it was tested against
  subdivided: boolean; // did it split into children?
}

/** The decision trace — why the recluster merged / split the way it did. */
export interface SandboxTrace {
  pairs: TracePair[];
  nodes: TraceNode[];
}

/** `POST /sandbox` → the reclustered map + timing + decision trace. */
export interface SandboxResponse {
  params: ResolvedParams;
  n_claims: number;
  ms: number;
  clusters: SandboxCluster[];
  trace: SandboxTrace;
}

/** `GET /explain?cluster=nX` → a node's neighbourhood + the criteria at that node. */
export interface ExplainCluster {
  cluster: string;
  /** k nearest clusters by centroid similarity (the merge candidates of this node). */
  neighbors: { id: string; sim: number; cohesion: number; merged: boolean }[];
  node: TraceNode;
}

/** `GET /explain?pair=nX,nY` → the merge criteria of one specific pair. */
export interface ExplainPair {
  pair: [string, string];
  sim: number;
  threshold: number;
  cohesion_a: number;
  cohesion_b: number;
  cohesion_min: number;
  merged: boolean;
}
