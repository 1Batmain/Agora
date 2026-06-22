/**
 * Mock backend for the redesigned front. Returns data in the FROZEN contract
 * shape so F1-F6 can be built and navigated before the real `/analysis`,
 * `/insights`, `/citations` endpoints land. Everything is seeded from the dataset
 * id so a given dataset always produces the same map (stable positions, like the
 * real UMAP seed). This is placeholder data — clearly synthetic, never corpus
 * truth.
 */
import type {
  AnalysisPayload,
  Backend,
  Citation,
  InsightLevel,
  InsightsPayload,
  SpatialEdge,
  SpatialTheme,
} from './contract';

/** Tiny deterministic PRNG (mulberry32) so mocks are stable per dataset. */
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

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

// Generic civic vocabulary — placeholder labels only, NOT a corpus taxonomy.
const TOPICS = [
  'Transparence',
  'Fiscalité',
  'Mobilités',
  'Transition écologique',
  'Service public',
  'Démocratie locale',
  'Numérique',
  'Logement',
  'Santé',
  'Éducation',
  'Sécurité',
  'Emploi',
];
const FACETS = [
  'cadre légal',
  'financement',
  'mise en œuvre',
  'gouvernance',
  'évaluation',
  'accès',
  'simplification',
  'concertation',
];

function pick<T>(arr: T[], r: () => number): T {
  return arr[Math.floor(r() * arr.length)];
}

/**
 * Build a variance-driven tree of themes. A root subdivides when its dispersion
 * exceeds a DERIVED cut (the median dispersion of the level) — mirroring the
 * backend's adaptive logic so the front exercises the real navigation.
 */
export function mockAnalysis(dataset: string, backend: Backend = 'auto'): AnalysisPayload {
  const r = rng(hash(dataset || 'default'));
  const themes: SpatialTheme[] = [];
  const usedTopics = new Set<string>();

  const nRoots = 5 + Math.floor(r() * 3); // 5..7
  const rootSpecs = Array.from({ length: nRoots }, () => ({
    angle: r() * Math.PI * 2,
    radius: 0.45 + r() * 0.5,
    dispersion: r(),
    n_avis: 40 + Math.floor(r() * 360),
  }));
  const medianDisp = [...rootSpecs.map((s) => s.dispersion)].sort((a, b) => a - b)[
    Math.floor(rootSpecs.length / 2)
  ];

  let idn = 0;
  const nextId = () => `t${idn++}`;
  const labelFor = (r2: () => number) => {
    let topic = pick(TOPICS, r2);
    let guard = 0;
    while (usedTopics.has(topic) && guard++ < 12) topic = pick(TOPICS, r2);
    usedTopics.add(topic);
    return topic;
  };

  rootSpecs.forEach((spec) => {
    const id = nextId();
    const x = Math.cos(spec.angle) * spec.radius;
    const y = Math.sin(spec.angle) * spec.radius;
    const hasChildren = spec.dispersion > medianDisp;
    const label = labelFor(r);
    themes.push({
      id,
      label,
      x,
      y,
      n_avis: spec.n_avis,
      n_claims: spec.n_avis + Math.floor(r() * spec.n_avis),
      weight: spec.n_avis,
      consensus: 0.3 + r() * 0.65,
      dispersion: spec.dispersion,
      parent_id: null,
      has_children: hasChildren,
    });

    if (!hasChildren) return;
    const nKids = 2 + Math.floor(r() * 3); // 2..4
    for (let k = 0; k < nKids; k++) {
      const kid = nextId();
      const ka = spec.angle + (r() - 0.5) * 1.2;
      const kr = 0.12 + r() * 0.18;
      const kdisp = r() * 0.7; // children rarely subdivide further
      const grand = kdisp > 0.55;
      const kAvis = Math.max(8, Math.floor(spec.n_avis / nKids) + Math.floor((r() - 0.5) * 30));
      themes.push({
        id: kid,
        label: `${label} — ${pick(FACETS, r)}`,
        x: x + Math.cos(ka) * kr,
        y: y + Math.sin(ka) * kr,
        n_avis: kAvis,
        n_claims: kAvis + Math.floor(r() * kAvis),
        weight: kAvis,
        consensus: 0.3 + r() * 0.65,
        dispersion: kdisp,
        parent_id: id,
        has_children: grand,
      });
      if (!grand) continue;
      const nG = 2 + Math.floor(r() * 2);
      for (let g = 0; g < nG; g++) {
        const gAvis = Math.max(5, Math.floor(kAvis / nG));
        themes.push({
          id: nextId(),
          label: `${label} · ${pick(FACETS, r)} (${g + 1})`,
          x: x + Math.cos(ka) * kr + (r() - 0.5) * 0.08,
          y: y + Math.sin(ka) * kr + (r() - 0.5) * 0.08,
          n_avis: gAvis,
          n_claims: gAvis + Math.floor(r() * gAvis),
          weight: gAvis,
          consensus: 0.3 + r() * 0.65,
          dispersion: r() * 0.4,
          parent_id: kid,
          has_children: false,
        });
      }
    }
  });

  // Co-occurrence edges among roots (sparse).
  const roots = themes.filter((t) => t.parent_id === null);
  const edges: SpatialEdge[] = [];
  for (let i = 0; i < roots.length; i++) {
    for (let j = i + 1; j < roots.length; j++) {
      if (r() < 0.35) {
        edges.push({ a: roots[i].id, b: roots[j].id, weight: Math.round(5 + r() * 40) });
      }
    }
  }

  return {
    themes,
    edges,
    params: { mock: true, seed: hash(dataset || 'default'), median_dispersion: medianDisp },
    backend_used: backend,
  };
}

/** Mock LLM insights, keyed to the zoom level (global vs a selected theme). */
export function mockInsights(
  dataset: string,
  level: InsightLevel,
  theme?: SpatialTheme,
): InsightsPayload {
  if (level === 'global' || !theme) {
    return {
      markdown: [
        `## Synthèse globale`,
        ``,
        `Vue d'ensemble de la consultation **${dataset}**. La carte ci-contre projette les`,
        `thèmes émergents : la **proximité** spatiale traduit une proximité sémantique, la`,
        `**taille** des bulles le volume d'avis.`,
        ``,
        `### Points saillants`,
        `- Plusieurs **pôles d'opinion** se détachent nettement.`,
        `- Les thèmes les plus volumineux concentrent l'essentiel des contributions.`,
        `- Le **consensus** varie fortement d'un thème à l'autre.`,
        ``,
        `> _Synthèse de démonstration (mock). Cliquez une bulle pour zoomer sur un thème._`,
      ].join('\n'),
    };
  }
  return {
    markdown: [
      `## ${theme.label}`,
      ``,
      `**${theme.n_avis} avis** · **${theme.n_claims} claims** · consensus`,
      `**${Math.round(theme.consensus * 100)} %** · dispersion ${theme.dispersion.toFixed(2)}.`,
      ``,
      `### Ce que disent les contributeurs`,
      `Ce thème regroupe les contributions portant sur **${theme.label.toLowerCase()}**.`,
      theme.has_children
        ? `Sa **dispersion interne élevée** justifie une subdivision — zoomez pour explorer ses sous-thèmes.`
        : `Sa cohérence interne en fait une **feuille** : explorez les citations représentatives.`,
      ``,
      `> _Synthèse de démonstration (mock) liée au niveau de zoom courant._`,
    ].join('\n'),
  };
}

/** Mock citations for a leaf theme, already sorted by distance to centroid. */
export function mockCitations(dataset: string, themeId: string): Citation[] {
  const r = rng(hash(dataset + ':' + themeId));
  const n = 8 + Math.floor(r() * 14);
  const out: Citation[] = Array.from({ length: n }, (_, i) => ({
    text:
      `Contribution citoyenne ${i + 1} — extrait représentatif du thème. ` +
      `Le contributeur exprime un point de vue argumenté sur la question posée par la consultation.`,
    dist_to_centroid: i * 0.05 + r() * 0.04,
    weight: 1 + Math.floor(r() * 4),
  }));
  return out.sort((a, b) => a.dist_to_centroid - b.dist_to_centroid);
}
