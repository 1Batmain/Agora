import { useEffect, useMemo, useRef, useState } from 'react';
import { select, type Selection } from 'd3-selection';
import { zoom, zoomIdentity, type ZoomBehavior, type ZoomTransform } from 'd3-zoom';
import { scaleLinear, scaleSqrt } from 'd3-scale';
import 'd3-transition';
import type { SpatialEdge, SpatialTheme } from './contract';

/**
 * F3 — spatial theme map, rendered with **D3** (d3-selection data-join, d3-zoom,
 * d3-scale). Themes are anchored at their UMAP (x,y); proximity = semantic
 * similarity. A deterministic collision relaxation (fixed iterations, no RNG)
 * then nudges overlapping bubbles apart while a weak spring keeps them near the
 * UMAP anchor — readable, non-overlapping, yet faithful to the embedding.
 *
 * Bubble area ∝ weight (volume of avis). Colour encodes TWO things at once:
 *   - HUE   = cluster identity (categorical, stable per theme id → coherent with
 *             highlights and across drill levels);
 *   - SATURATION/LIGHTNESS = `consensus_eff`, the population-weighted consensus
 *             (Bayesian shrinkage of consensus by evidence volume, N = n_avis).
 *             A theme backed by a single avis stays PALE; a large consensual
 *             theme is vivid — so volume can no longer masquerade as agreement.
 *
 * Navigation is ADAPTIVE drill: you only ever see ONE level (the children of the
 * current parent); double-clicking a drillable bubble (`has_children`) descends;
 * a leaf bubble selects → its citations. NO raw avis are shown on the map — only
 * at the leaf, in the side panel.
 *
 * Free wheel-zoom + drag-pan (d3-zoom) inspect a dense level; the SEMANTIC zoom
 * (which level is shown) is driven by the parent via `currentParentId` — on a
 * level change the d3-zoom transform resets. (Co-occurrence edges were removed:
 * they were visual noise; the data may stay in the contract, we just don't draw it.)
 */
const VB = 1000; // internal viewBox size (square); SVG scales to the container.
const PAD = 120;
const COLLIDE_PAD = 5; // px gap enforced between bubble rims
const RELAX_ITERS = 400; // fixed → deterministic, stable layout across renders
const COLLIDE_PASSES = 3; // collision sub-iterations per tick (helps dense levels converge)
const SPRING_K = 0.03; // pull back toward the UMAP anchor each iteration

interface Pos {
  cx: number;
  cy: number;
  r: number;
}

interface Tip {
  x: number;
  y: number;
  theme: SpatialTheme;
}

export function SpatialMap({
  themes,
  edges,
  currentParentId,
  selectedId,
  onSelect,
  onDrill,
  query = '',
  minConsensus = 0,
}: {
  themes: SpatialTheme[];
  edges: SpatialEdge[];
  currentParentId: string | null;
  selectedId: string | null;
  onSelect: (t: SpatialTheme) => void;
  onDrill: (t: SpatialTheme) => void;
  query?: string;
  minConsensus?: number;
}) {
  const q = query.trim().toLowerCase();

  const visible = useMemo(
    () => themes.filter((t) => t.parent_id === currentParentId),
    [themes, currentParentId],
  );

  // Population-weighted consensus (Bayesian shrinkage):
  //   consensus_eff = (N/(N+k))*consensus + (k/(N+k))*prior
  // N = n_avis; k = median n_avis over the visible level (derived "typical sample
  // size", no magic number) → a theme with the median volume gets exactly half
  // confidence. We shrink toward a LOW prior (the brief's "ou bas" branch) rather
  // than the mean: the consensus metric is compressed (~0.6–0.8 here), so
  // regressing to the mean would leave a 1-avis fluke looking middling. With a
  // low prior, consensus_eff ≈ confidence × consensus → thin-evidence themes go
  // PALE and only big-AND-consensual themes are vivid (exactly "gros & consensuels
  // = vifs"; a single témoignage can never read as full consensus).
  const PRIOR = 0;
  const consensusEff = useMemo(() => {
    const eff = new Map<string, number>();
    if (visible.length === 0) return eff;
    const sortedN = visible.map((t) => t.n_avis).sort((a, b) => a - b);
    const mid = Math.floor(sortedN.length / 2);
    const medianN =
      sortedN.length % 2 === 0
        ? (sortedN[mid - 1] + sortedN[mid]) / 2
        : sortedN[mid];
    const k = Math.max(1, medianN); // shrinkage strength ~ a typical theme's size
    for (const t of visible) {
      const N = Math.max(0, t.n_avis);
      eff.set(t.id, (N / (N + k)) * t.consensus + (k / (N + k)) * PRIOR);
    }
    return eff;
  }, [visible]);

  // Layout: normalise the visible themes' x,y into the padded viewBox via d3
  // scales, then run a DETERMINISTIC collision relaxation (fixed iterations, no
  // RNG) so bubbles separate while staying near their UMAP anchor.
  const layout = useMemo(() => {
    const pos = new Map<string, Pos>();
    if (visible.length === 0) return pos;

    const xs = visible.map((t) => t.x);
    const ys = visible.map((t) => t.y);
    const minX = Math.min(...xs),
      maxX = Math.max(...xs);
    const minY = Math.min(...ys),
      maxY = Math.max(...ys);
    const maxW = Math.max(...visible.map((t) => t.weight), 1);

    const sx = scaleLinear().domain([minX, maxX]).range([PAD, VB - PAD]);
    // SVG y grows downward; invert the range so "up" matches UMAP up.
    const sy = scaleLinear().domain([minY, maxY]).range([VB - PAD, PAD]);
    const sr = scaleSqrt().domain([0, maxW]).range([26, 96]);

    // Nodes carry an immutable UMAP anchor (ax,ay) + a mutable position (x,y).
    const nodes = visible.map((t) => {
      const ax = sx(t.x);
      const ay = sy(t.y);
      return { id: t.id, ax, ay, x: ax, y: ay, r: sr(t.weight) };
    });

    // Keep a bubble fully inside the viewBox. Applied as a CONSTRAINT *inside* the
    // loop (before & after collision) — not just once at the end — so collision
    // resolves against the walls and converges to a packed, non-overlapping state.
    // (A single trailing clamp would re-stack bubbles it shoves off a crammed edge.)
    const clamp = (n: { x: number; y: number; r: number }) => {
      n.x = Math.max(n.r, Math.min(VB - n.r, n.x));
      n.y = Math.max(n.r, Math.min(VB - n.r, n.y));
    };

    for (let it = 0; it < RELAX_ITERS; it++) {
      // weak spring back toward the UMAP anchor (keeps the map ~faithful)
      for (const n of nodes) {
        n.x += (n.ax - n.x) * SPRING_K;
        n.y += (n.ay - n.y) * SPRING_K;
        clamp(n);
      }
      // pairwise collision: push overlapping bubbles apart, half each
      for (let pass = 0; pass < COLLIDE_PASSES; pass++) {
        for (let i = 0; i < nodes.length; i++) {
          for (let j = i + 1; j < nodes.length; j++) {
            const a = nodes[i];
            const b = nodes[j];
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let dist = Math.sqrt(dx * dx + dy * dy);
            const minDist = a.r + b.r + COLLIDE_PAD;
            if (dist < minDist) {
              if (dist < 1e-6) {
                // coincident anchors: deterministic nudge derived from indices
                dx = ((i % 3) - 1) || 1;
                dy = ((j % 3) - 1) || 1;
                dist = Math.sqrt(dx * dx + dy * dy);
              }
              const push = (minDist - dist) / 2;
              const ux = dx / dist;
              const uy = dy / dist;
              a.x -= ux * push;
              a.y -= uy * push;
              b.x += ux * push;
              b.y += uy * push;
            }
          }
        }
      }
      for (const n of nodes) clamp(n);
    }

    for (const n of nodes) {
      pos.set(n.id, { cx: n.x, cy: n.y, r: n.r });
    }
    return pos;
  }, [visible]);

  const isDim = (t: SpatialTheme) =>
    (q !== '' && !t.label.toLowerCase().includes(q)) || t.consensus < minConsensus;

  const svgRef = useRef<SVGSVGElement | null>(null);
  const zoomRef = useRef<ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const gRef = useRef<SVGGElement | null>(null);
  const [tip, setTip] = useState<Tip | null>(null);

  // Keep the latest callbacks/data in refs so the d3 event handlers always see
  // fresh values without re-binding the whole zoom behaviour each render.
  const cb = useRef({ onSelect, onDrill });
  cb.current = { onSelect, onDrill };

  // --- d3-zoom: set up once; drives the inner <g> transform (free pan/zoom). ---
  useEffect(() => {
    const svg = select(svgRef.current!);
    const g = select(gRef.current!);
    const zb = zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 6])
      .on('zoom', (event: { transform: ZoomTransform }) => {
        g.attr('transform', event.transform.toString());
        setTip(null); // a moving map invalidates the tooltip anchor
      });
    svg.call(zb);
    // d3-zoom binds its own dblclick-to-zoom; we use dblclick for semantic drill.
    svg.on('dblclick.zoom', null);
    zoomRef.current = zb;
    return () => {
      svg.on('.zoom', null);
    };
  }, []);

  // Reset the free zoom/pan whenever the semantic level changes.
  useEffect(() => {
    const svg = select(svgRef.current!);
    if (zoomRef.current) {
      svg
        .transition()
        .duration(450)
        .call(zoomRef.current.transform, zoomIdentity);
    }
    setTip(null);
  }, [currentParentId]);

  // --- render: data-join bubbles. (Co-occurrence edges intentionally not drawn.) ---
  useEffect(() => {
    const g = select<SVGGElement, unknown>(gRef.current!);

    // bubbles layer --------------------------------------------------------
    const nodeLayer = ensureLayer(g, 'map__nodes');
    const nodes = nodeLayer
      .selectAll<SVGGElement, SpatialTheme>('g.bubble')
      .data(visible, (t) => t.id)
      .join(
        (enter) => {
          const ge = enter.append('g');
          ge.append('circle').attr('class', 'bubble__ring');
          ge.append('circle').attr('class', 'bubble__circle');
          ge.append('text').attr('class', 'bubble__stat').attr('y', 4);
          ge.append('text').attr('class', 'bubble__label');
          return ge;
        },
        (update) => update,
        (exit) => exit.remove(),
      );

    nodes
      .attr(
        'class',
        (t) =>
          `bubble${t.id === selectedId ? ' bubble--sel' : ''}${
            t.has_children ? ' bubble--drill' : ' bubble--leaf'
          }${isDim(t) ? ' bubble--dim' : ''}`,
      )
      .attr('transform', (t) => {
        const p = layout.get(t.id)!;
        return `translate(${p.cx},${p.cy})`;
      })
      .on('click', (_e, t) => cb.current.onSelect(t))
      .on('dblclick', (event: MouseEvent, t) => {
        event.stopPropagation();
        if (t.has_children) cb.current.onDrill(t);
        else cb.current.onSelect(t);
      })
      .on('mouseenter', (_e, t) => {
        const p = layout.get(t.id)!;
        setTip({ x: p.cx, y: p.cy - p.r, theme: t });
      })
      .on('mouseleave', () => setTip(null));

    nodes
      .select<SVGCircleElement>('circle.bubble__circle')
      .attr('r', (t) => layout.get(t.id)!.r)
      .style('fill', (t) =>
        themeColor(t.id, consensusEff.get(t.id) ?? t.consensus),
      );

    nodes
      .select<SVGCircleElement>('circle.bubble__ring')
      .attr('r', (t) => layout.get(t.id)!.r + 6)
      .style('display', (t) => (t.has_children ? null : 'none'));

    nodes
      .select<SVGTextElement>('text.bubble__stat')
      .text((t) => String(t.n_avis));

    nodes
      .select<SVGTextElement>('text.bubble__label')
      .attr('y', (t) => layout.get(t.id)!.r + 22)
      .text((t) => truncate(t.label, 26));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, layout, consensusEff, selectedId, q, minConsensus]);

  return (
    <div className="map">
      <svg
        ref={svgRef}
        className="map__svg"
        viewBox={`0 0 ${VB} ${VB}`}
        preserveAspectRatio="xMidYMid meet"
      >
        <g ref={gRef} />
      </svg>
      {tip && (
        <MapTooltip tip={tip} />
      )}
      <div className="map__legend">
        <span className="map__legenditem">
          <i className="dot dot--pale" /> consensus pondéré faible
        </span>
        <span className="map__legenditem">
          <i className="dot dot--vivid" /> consensus pondéré fort
        </span>
        <span className="map__legenditem map__legendhint">
          teinte = thème · pâleur = consensus pondéré population · taille = volume
          d'avis · molette = zoom · double-clic = explorer
        </span>
      </div>
    </div>
  );
}

/** HTML tooltip overlay, positioned in viewBox space → % of the square SVG. */
function MapTooltip({ tip }: { tip: Tip }) {
  const t = tip.theme;
  return (
    <div
      className="map__tooltip"
      style={{ left: `${(tip.x / VB) * 100}%`, top: `${(tip.y / VB) * 100}%` }}
    >
      <strong>{t.label}</strong>
      <span>
        {t.n_avis} avis · consensus {Math.round(t.consensus * 100)}%
      </span>
      <span className="map__tooltiphint">
        {t.has_children ? 'double-clic pour explorer' : 'clic → citations'}
      </span>
    </div>
  );
}

/** Get-or-create a named child <g> layer (stable across renders). */
function ensureLayer(
  g: Selection<SVGGElement, unknown, null, undefined>,
  cls: string,
): Selection<SVGGElement, unknown, null, undefined> {
  let layer = g.select<SVGGElement>(`g.${cls}`);
  if (layer.empty()) layer = g.append('g').attr('class', cls);
  return layer;
}

/**
 * Bubble fill: HUE = stable categorical identity of the theme (hash of its id),
 * SATURATION + LIGHTNESS = population-weighted consensus. Low consensus_eff →
 * pale (washed out, near-white); high → vivid and darker. The hue is invariant
 * to consensus, so a theme keeps its colour across drill levels / highlights.
 */
function themeColor(id: string, consensusEff: number): string {
  const c = Math.max(0, Math.min(1, consensusEff));
  const hue = hashHue(id);
  const sat = Math.round(18 + 82 * c); // 18% (washed, near-grey) → 100% (vivid)
  const light = Math.round(88 - 44 * c); // 88% (near-white) → 44% (deep)
  return `hsl(${hue}, ${sat}%, ${light}%)`;
}

/** Deterministic hue (0..359) from a theme id — stable categorical identity. */
function hashHue(id: string): number {
  let h = 0;
  for (let i = 0; i < id.length; i++) {
    h = (h * 31 + id.charCodeAt(i)) | 0;
  }
  // golden-angle spread on the hash keeps neighbouring ids visually distinct
  return Math.abs(h * 137) % 360;
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}
