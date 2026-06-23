/**
 * Mock `/sandbox` recluster — produces data in the FROZEN sandboxContract shape so
 * the analyst CONSOLE can be built and driven before lane B's real endpoint lands.
 *
 * The whole point of a mixing console is that the faders MOVE THE MAP, so this mock
 * is deliberately REACTIVE: the number of clusters and their sizes/cohesions are a
 * deterministic function of (resolution, coarsen_mult, tau_mult, k, alpha). Same
 * params → same map (stable, like a fixed seed); a fader nudge → a visibly
 * different recluster. The trace (merge pairs + subdivision nodes) is derived from
 * the same draws so it stays coherent with the bubbles on screen.
 *
 * This is PLACEHOLDER data — clearly synthetic, never corpus truth.
 */
import type {
  ExplainCluster,
  ExplainPair,
  ResolvedParams,
  SandboxCluster,
  SandboxParams,
  SandboxResponse,
  SandboxTrace,
  TraceNode,
  TracePair,
} from './sandboxContract';

/** Tiny deterministic PRNG (mulberry32) — stable map per (params) draw. */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashStr(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** Mix the fader values into the seed so each knob actually perturbs the draw. */
function paramSeed(dataset: string, p: Required<SandboxParams>): number {
  const q =
    Math.round(p.resolution * 100) * 1_000_003 +
    Math.round(p.coarsen_mult * 100) * 9_001 +
    Math.round(p.tau_mult * 100) * 131 +
    p.k * 17 +
    Math.round(p.alpha * 100);
  return (hashStr(dataset) ^ (q >>> 0)) >>> 0;
}

// Generic civic vocabulary — placeholder keywords only, NOT a corpus taxonomy.
const VOCAB = [
  'financement', 'accès', 'délais', 'transparence', 'territoires', 'formation',
  'numérique', 'fiscalité', 'mobilité', 'logement', 'santé', 'sécurité',
  'concertation', 'simplification', 'écologie', 'emploi', 'service', 'public',
  'local', 'national', 'aide', 'contrôle', 'qualité', 'égalité',
];

function pick<T>(arr: T[], r: () => number): T {
  return arr[Math.floor(r() * arr.length)];
}

/** Defaults the backend would DERIVE — kept here so the mock and UI agree offline. */
export const SANDBOX_DEFAULTS: Required<SandboxParams> = {
  alpha: 0.5,
  k: 12,
  resolution: 1.0,
  coarsen_mult: 1.0,
  tau_mult: 1.0,
};

function resolve(p: SandboxParams): Required<SandboxParams> {
  return {
    alpha: p.alpha ?? SANDBOX_DEFAULTS.alpha,
    k: p.k ?? SANDBOX_DEFAULTS.k,
    resolution: p.resolution ?? SANDBOX_DEFAULTS.resolution,
    coarsen_mult: p.coarsen_mult ?? SANDBOX_DEFAULTS.coarsen_mult,
    tau_mult: p.tau_mult ?? SANDBOX_DEFAULTS.tau_mult,
  };
}

/**
 * Generate a reclustering for a fader setting. Cluster COUNT rises with resolution
 * and with low coarsening / low τ; cohesion improves with α (target-aware blend)
 * and with higher k (denser graph). All bounded so the map stays legible.
 */
export function mockSandbox(dataset: string, params: SandboxParams = {}): SandboxResponse {
  const p = resolve(params);
  const r = rng(paramSeed(dataset || 'default', p));

  const N_CLAIMS = 3000;

  // #clusters ~ resolution, damped by coarsening (merges) and τ (subdivision gate).
  // base 8 at resolution 1; coarsen_mult>1 removes clusters, tau_mult>1 removes splits.
  const base = 8 * p.resolution;
  const afterCoarsen = base / (0.6 + 0.7 * p.coarsen_mult);
  const subdivBoost = 1 + Math.max(0, (1 - p.tau_mult)) * 0.8; // low τ → more nodes
  const nClusters = Math.max(3, Math.min(18, Math.round(afterCoarsen * subdivBoost)));

  // α (target weight) and k (graph density) lift cohesion; resolution (finer cuts)
  // also tightens clusters. A small per-cluster jitter keeps them distinct.
  const cohesionBase = 0.42 + 0.28 * p.alpha + 0.12 * Math.min(1, p.k / 20) + 0.08 * Math.min(1, p.resolution);

  // Partition the claims into nClusters buckets with a long-tail size profile.
  const rawSizes = Array.from({ length: nClusters }, (_, i) => {
    const zipf = 1 / Math.pow(i + 1, 0.7);
    return zipf * (0.7 + r() * 0.6);
  });
  const sumRaw = rawSizes.reduce((s, v) => s + v, 0);

  const clusters: SandboxCluster[] = rawSizes.map((sz, i) => {
    const id = `n${i}`;
    const nClaims = Math.max(12, Math.round((sz / sumRaw) * N_CLAIMS));
    const nAvis = Math.max(8, Math.round(nClaims * (0.55 + r() * 0.35)));
    const cohesion = Math.max(0.18, Math.min(0.97, cohesionBase + (r() - 0.5) * 0.22));
    const kw = Array.from({ length: 3 + Math.floor(r() * 2) }, () => pick(VOCAB, r));
    return {
      id,
      parent_id: null,
      n_claims: nClaims,
      n_avis: nAvis,
      keywords: Array.from(new Set(kw)),
      sample_claims: [
        `Il faudrait revoir ${kw[0]} pour le rendre plus juste.`,
        `Sur ${kw[1] ?? kw[0]}, les attentes sont fortes et concrètes.`,
        `Beaucoup soulignent l'importance de ${kw[0]} dans ce dossier.`,
      ].slice(0, 2 + Math.floor(r() * 2)),
      cohesion: Number(cohesion.toFixed(3)),
    };
  });

  const trace = mockTrace(clusters, p, r);
  // Latency model: blend + kNN + Leiden scale with k and #clusters; ~1s target.
  const ms = Math.round(280 + p.k * 14 + nClusters * 22 + r() * 120);

  const resolved: ResolvedParams = {
    ...p,
    derived: {
      coarsen_threshold: Number((0.5 * p.coarsen_mult).toFixed(3)), // μ+σ stand-in
      tau: Number((0.55 * p.tau_mult).toFixed(3)),
      default_k: SANDBOX_DEFAULTS.k,
      median_size: clusters.length ? clusters[Math.floor(clusters.length / 2)].n_claims : 0,
    },
  };

  return { params: resolved, n_claims: N_CLAIMS, ms, clusters, trace };
}

/**
 * Merge pairs + subdivision nodes coherent with the cluster set. Coarsening tests
 * the closest pairs against the threshold (high coarsen_mult → more `merged:true`);
 * each node's subdivision compares its dispersion (≈ 1 − cohesion) to τ.
 */
function mockTrace(
  clusters: SandboxCluster[],
  p: Required<SandboxParams>,
  r: () => number,
): SandboxTrace {
  const threshold = 0.5 * p.coarsen_mult;
  const tau = 0.55 * p.tau_mult;

  const pairs: TracePair[] = [];
  // A handful of candidate merges between consecutive clusters (closest centroids).
  for (let i = 0; i + 1 < clusters.length && pairs.length < 8; i++) {
    const a = clusters[i];
    const b = clusters[i + 1];
    const sim = Number((0.35 + r() * 0.5).toFixed(3));
    const cohesionMin = Math.min(a.cohesion, b.cohesion);
    // Merge when similar enough AND neither cluster is too loose to absorb the other.
    const merged = sim >= threshold && cohesionMin >= 0.3;
    pairs.push({ a: a.id, b: b.id, sim, threshold: Number(threshold.toFixed(3)), cohesion_min: Number(cohesionMin.toFixed(3)), merged });
  }

  const nodes: TraceNode[] = clusters.map((c) => {
    const dispersion = Number(Math.max(0.05, 1 - c.cohesion + (r() - 0.5) * 0.1).toFixed(3));
    return { id: c.id, dispersion, tau: Number(tau.toFixed(3)), subdivided: dispersion > tau };
  });

  return { pairs, nodes };
}

/** Mock `GET /explain?cluster=nX` — k nearest clusters + this node's criteria. */
export function mockExplainCluster(
  resp: SandboxResponse,
  clusterId: string,
): ExplainCluster | null {
  const node = resp.trace.nodes.find((n) => n.id === clusterId);
  if (!node) return null;
  const self = resp.clusters.find((c) => c.id === clusterId);
  const r = rng(hashStr(clusterId));
  const neighbors = resp.clusters
    .filter((c) => c.id !== clusterId)
    .map((c) => {
      // sim reuses the trace pair if one exists, else a stable synthetic draw.
      const pair = resp.trace.pairs.find(
        (pp) => (pp.a === clusterId && pp.b === c.id) || (pp.b === clusterId && pp.a === c.id),
      );
      return {
        id: c.id,
        sim: pair ? pair.sim : Number((0.2 + r() * 0.55).toFixed(3)),
        cohesion: c.cohesion,
        merged: pair ? pair.merged : false,
      };
    })
    .sort((x, y) => y.sim - x.sim)
    .slice(0, Math.min(5, Math.max(1, Math.round(resp.params.k / 3))));
  void self;
  return { cluster: clusterId, neighbors, node };
}

/** Mock `GET /explain?pair=nX,nY` — the merge criteria of one specific pair. */
export function mockExplainPair(
  resp: SandboxResponse,
  a: string,
  b: string,
): ExplainPair | null {
  const ca = resp.clusters.find((c) => c.id === a);
  const cb = resp.clusters.find((c) => c.id === b);
  if (!ca || !cb) return null;
  const pair = resp.trace.pairs.find(
    (pp) => (pp.a === a && pp.b === b) || (pp.a === b && pp.b === a),
  );
  const r = rng(hashStr(a + ':' + b));
  const sim = pair ? pair.sim : Number((0.2 + r() * 0.55).toFixed(3));
  const threshold = pair ? pair.threshold : Number((0.5 * resp.params.coarsen_mult).toFixed(3));
  const cohesion_min = Math.min(ca.cohesion, cb.cohesion);
  return {
    pair: [a, b],
    sim,
    threshold,
    cohesion_a: ca.cohesion,
    cohesion_b: cb.cohesion,
    cohesion_min: Number(cohesion_min.toFixed(3)),
    merged: pair ? pair.merged : sim >= threshold && cohesion_min >= 0.3,
  };
}
