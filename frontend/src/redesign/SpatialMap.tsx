import { useEffect, useMemo, useRef, useState } from 'react';
import { select, type Selection } from 'd3-selection';
import { zoom, zoomIdentity, type ZoomBehavior, type ZoomTransform } from 'd3-zoom';
import { hierarchy, pack } from 'd3-hierarchy';
import 'd3-transition';
import type { SpatialEdge, SpatialTheme } from './contract';
import { themeCaption } from './labels';

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
const COLLIDE_PAD = 8; // px gap enforced between packed bubbles.

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

  // Layout: a DETERMINISTIC circle pack (d3-hierarchy). No semantic x,y — the
  // only thing encoded is bubble AREA ∝ n_avis. The pack is stable across renders
  // (no RNG) and naturally non-overlapping, so the old UMAP relaxation is gone.
  const layout = useMemo(() => {
    const pos = new Map<string, Pos>();
    if (visible.length === 0) return pos;

    const root = hierarchy<{ children?: SpatialTheme[] } & Partial<SpatialTheme>>(
      { children: visible },
    ).sum((d) => ('n_avis' in d && d.n_avis != null ? Math.max(1, d.n_avis) : 0));

    const packed = pack<{ children?: SpatialTheme[] } & Partial<SpatialTheme>>()
      .size([VB - 2 * PAD, VB - 2 * PAD])
      .padding(COLLIDE_PAD)(root);

    for (const leaf of packed.leaves()) {
      const t = leaf.data as SpatialTheme;
      if (!t.id) continue;
      pos.set(t.id, { cx: leaf.x + PAD, cy: leaf.y + PAD, r: leaf.r });
    }
    return pos;
  }, [visible]);

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
 * is where the RICH detail lives (the bubble itself only shows voices + title):
 * title, keyword stubs, voices and consensus.
 */
function MapTooltip({ tip }: { tip: Tip }) {
  const t = tip.theme;
  const title = themeCaption(t);
  // Keyword line: explicit keywords if the backend sent them, else the keyword
  // stub `label` when it differs from the (LLM) title shown above.
  const keywords = t.keywords?.length
    ? t.keywords.join(' · ')
    : title !== t.label
      ? t.label
      : '';
  return (
    <div
      className="map__tooltip"
      style={{ left: `${(tip.x / VB) * 100}%`, top: `${(tip.y / VB) * 100}%` }}
    >
      <strong>{title}</strong>
      {keywords && <span className="map__tooltipkw">{keywords}</span>}
      <span>
        {t.n_avis} voix · consensus {Math.round(t.consensus * 100)}%
      </span>
      <span className="map__tooltiphint">
        {t.has_children ? 'clic pour explorer' : 'clic → témoignages'}
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
