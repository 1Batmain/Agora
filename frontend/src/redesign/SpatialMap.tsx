import { useEffect, useMemo, useRef, useState } from 'react';
import { select, type Selection } from 'd3-selection';
import { zoom, zoomIdentity, type ZoomBehavior, type ZoomTransform } from 'd3-zoom';
import 'd3-transition';
import type { SpatialEdge, SpatialTheme } from './contract';
import { themeCaption } from './labels';
import { Markdown } from './Markdown';

/**
 * F3 — theme BUBBLES, rendered with **D3** (d3-selection data-join, d3-zoom,
 * d3-hierarchy pack). The UMAP "map" was dropped: at every level the bubbles are
 * laid out by a deterministic **circle pack** (no semantic x,y — those positions
 * read as noise to non-experts). Position carries NO meaning; only SIZE, COLOUR
 * and the LABEL do:
 *   - AREA  = `n_avis` (volume of voices) — the only thing the packing encodes;
 *   - HUE   = cluster identity (categorical, stable per theme id → coherent with
 *             highlights and across drill levels);
 *   - SATURATION/LIGHTNESS = `consensus_eff`, the population-weighted consensus
 *             (Bayesian shrinkage of consensus by evidence volume, N = n_avis).
 *             A theme backed by a single avis stays PALE; a large consensual
 *             theme is vivid — so volume can no longer masquerade as agreement.
 *
 * The default caption is DELIBERATELY sparse: the voice count (+ the LLM `title`
 * when readable). Keywords/consensus live only in the HOVER tooltip.
 *
 * Navigation is ADAPTIVE drill: you only ever see ONE level (the children of the
 * current parent). A single CLICK on a drillable bubble (`has_children`)
 * descends, hiding its siblings and revealing its children; a click on a leaf
 * selects it → its citations. Breadcrumbs (in the parent) climb back up. NO raw
 * avis are shown on the map — only at the leaf, in the side panel.
 *
 * Free wheel-zoom + drag-pan (d3-zoom) inspect a dense level; the SEMANTIC zoom
 * (which level is shown) is driven by the parent via `currentParentId` — on a
 * level change the d3-zoom transform resets. (Co-occurrence edges are not drawn.)
 */
const VB = 1000; // internal viewBox size (square); SVG scales to the container.
const PAD = 80; // outer margin so rim labels stay inside the viewBox.
const COLLIDE_PAD = 14; // gap (viewBox units) enforced between bubbles.
const HOVER_SCALE = 1.1; // how much a hovered bubble grows (smooth zoom).

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

  // Layout: a DETERMINISTIC force relaxation (no RNG, no extra dependency). The
  // old circle-pack clumped the big bubble dead-centre and crammed the small ones
  // around it, leaving empty corners; this seeds the bubbles on a phyllotaxis
  // (sunflower) spiral — which fills the disc EVENLY — then resolves overlaps with
  // a few hundred collision passes. Result: a harmonious, breathable spread that
  // uses the whole canvas, still stable across renders and non-overlapping. As in
  // the pack, position carries NO meaning — only AREA ∝ n_avis is encoded.
  const layout = useMemo(() => computeLayout(visible), [visible]);

  // Preferred caption: LLM `title` when present, else the keyword `label` stub
  // (single source of truth — `themeCaption`).
  const captionOf = themeCaption;

  const isDim = (t: SpatialTheme) =>
    (q !== '' && !captionOf(t).toLowerCase().includes(q)) || t.consensus < minConsensus;

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
      // Single click is the ONE navigation gesture: drill if the bubble has
      // children (hide siblings, reveal children), otherwise select → citations.
      .on('click', (event: MouseEvent, t) => {
        event.stopPropagation();
        if (t.has_children) cb.current.onDrill(t);
        else cb.current.onSelect(t);
      })
      // Hover = smooth ZOOM: the bubble grows slightly and is RAISED to the front
      // (z-order) so it never hides behind a neighbour; the tooltip anchors to it.
      .on('mouseenter', function (this: SVGGElement, _e: MouseEvent, t) {
        const p = layout.get(t.id)!;
        setTip({ x: p.cx, y: p.cy - p.r * HOVER_SCALE, theme: t });
        select(this)
          .raise()
          .transition('hover')
          .duration(180)
          .attr('transform', `translate(${p.cx},${p.cy}) scale(${HOVER_SCALE})`);
      })
      .on('mouseleave', function (this: SVGGElement, _e: MouseEvent, t) {
        const p = layout.get(t.id)!;
        setTip(null);
        select(this)
          .transition('hover')
          .duration(180)
          .attr('transform', `translate(${p.cx},${p.cy})`);
      });

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

    // Caption ON the bubble when it's big enough to fit it; otherwise the title
    // lives only in the tooltip so small bubbles stay uncluttered.
    nodes
      .select<SVGTextElement>('text.bubble__label')
      .attr('y', (t) => layout.get(t.id)!.r + 22)
      .text((t) => (layout.get(t.id)!.r >= 30 ? truncate(captionOf(t), 26) : ''));
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
          taille = nombre de voix · teinte = thème · pâleur = consensus pondéré ·
          clic = explorer / voir les témoignages · molette = zoom
        </span>
      </div>
    </div>
  );
}

/**
 * HTML tooltip overlay, positioned in viewBox space → % of the square SVG. This
 * is where ALL the rich/synthetic detail lives — the bubble itself stays sparse
 * (voices + short title), so the LLM synthesis NEVER clutters the graph:
 *   - `hook`        : one-line accroche (when present);
 *   - `description` : short LLM synthesis, rendered as MARKDOWN;
 *   - `convergence` : 0..1 convergence of ideas inside the cluster (+ voices).
 * All three are graceful: absent fields simply don't render (repli on keywords /
 * consensus from the legacy contract).
 */
function MapTooltip({ tip }: { tip: Tip }) {
  const t = tip.theme;
  const title = themeCaption(t);
  // Keyword fallback (only when there's no LLM description): explicit keywords if
  // the backend sent them, else the keyword stub `label` when it differs.
  const keywords = t.keywords?.length
    ? t.keywords.join(' · ')
    : title !== t.label
      ? t.label
      : '';
  const hasConv = typeof t.convergence === 'number' && Number.isFinite(t.convergence);
  return (
    <div
      className="map__tooltip"
      style={{ left: `${(tip.x / VB) * 100}%`, top: `${(tip.y / VB) * 100}%` }}
    >
      <strong>{title}</strong>
      {t.hook && <span className="map__tooltiphook">{t.hook}</span>}
      {t.description ? (
        <div className="map__tooltipdesc">
          <Markdown source={t.description} />
        </div>
      ) : (
        keywords && <span className="map__tooltipkw">{keywords}</span>
      )}
      <span className="map__tooltipmeta">
        {t.n_avis} voix
        {hasConv
          ? ` · convergence ${Math.round((t.convergence as number) * 100)}%`
          : ` · consensus ${Math.round(t.consensus * 100)}%`}
      </span>
      <span className="map__tooltiphint">
        {t.has_children ? 'clic pour explorer' : 'clic → témoignages'}
      </span>
    </div>
  );
}

interface Node {
  id: string;
  r: number;
  x: number;
  y: number;
}

/**
 * Deterministic force layout: phyllotaxis seed + collision relaxation. No RNG, so
 * the same level always yields the same arrangement (stable, like a UMAP seed).
 */
function computeLayout(visible: SpatialTheme[]): Map<string, Pos> {
  const pos = new Map<string, Pos>();
  const n = visible.length;
  if (n === 0) return pos;

  const side = VB - 2 * PAD;
  const cx0 = VB / 2;
  const cy0 = VB / 2;

  // Radius ∝ √n_avis (area ∝ n_avis), scaled so the bubbles fill a target FRACTION
  // of the canvas — leaving breathing room (no entassement). A floor keeps the
  // smallest bubble legible.
  const totalN = visible.reduce((s, t) => s + Math.max(1, t.n_avis), 0);
  const FILL = 0.4;
  const scale = Math.sqrt((FILL * side * side) / (Math.PI * totalN));
  const rMin = side * 0.05;
  const rMax = side * 0.34;
  const rOf = (t: SpatialTheme) =>
    Math.min(rMax, Math.max(rMin, Math.sqrt(Math.max(1, t.n_avis)) * scale));

  // Seed on a sunflower spiral (golden angle) → even coverage of the disc.
  const GOLDEN = Math.PI * (3 - Math.sqrt(5));
  const nodes: Node[] = visible.map((t, i) => {
    const r = rOf(t);
    const frac = (i + 0.5) / n;
    const rad = Math.sqrt(frac) * (side / 2 - r);
    const theta = i * GOLDEN;
    return { id: t.id, r, x: cx0 + rad * Math.cos(theta), y: cy0 + rad * Math.sin(theta) };
  });

  // Relax: push overlapping pairs apart, then clamp inside the padded square.
  const ITER = 360;
  for (let it = 0; it < ITER; it++) {
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const a = nodes[i];
        const b = nodes[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let d = Math.hypot(dx, dy) || 0.01;
        const min = a.r + b.r + COLLIDE_PAD;
        if (d < min) {
          const push = (min - d) / 2;
          dx /= d;
          dy /= d;
          a.x -= dx * push;
          a.y -= dy * push;
          b.x += dx * push;
          b.y += dy * push;
        }
      }
    }
    for (const nd of nodes) {
      nd.x = Math.max(PAD + nd.r, Math.min(VB - PAD - nd.r, nd.x));
      nd.y = Math.max(PAD + nd.r, Math.min(VB - PAD - nd.r, nd.y));
    }
  }

  for (const nd of nodes) pos.set(nd.id, { cx: nd.x, cy: nd.y, r: nd.r });
  return pos;
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
