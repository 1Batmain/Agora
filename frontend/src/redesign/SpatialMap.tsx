import { useEffect, useMemo, useRef, useState } from 'react';
import { select, type Selection } from 'd3-selection';
import { zoom, zoomIdentity, type ZoomBehavior, type ZoomTransform } from 'd3-zoom';
import { scaleLinear, scaleSqrt } from 'd3-scale';
import 'd3-transition';
import type { SpatialEdge, SpatialTheme } from './contract';

/**
 * F3 — spatial theme map, rendered with **D3** (d3-selection data-join, d3-zoom,
 * d3-scale). Themes are placed at their UMAP (x,y); proximity = semantic
 * similarity. Bubble area ∝ weight (volume of avis), colour = consensus.
 * Edges = co-occurrence, drawn between the *centres* of the source/target nodes
 * via a shared position map, so they stay attached at any zoom/pan level (this
 * is the FIX over the hand-rolled SVG: links never detach from clusters).
 *
 * Navigation is ADAPTIVE drill: you only ever see ONE level (the children of the
 * current parent); double-clicking a drillable bubble (`has_children`) descends;
 * a leaf bubble selects → its citations. NO raw avis are shown on the map — only
 * at the leaf, in the side panel.
 *
 * Free wheel-zoom + drag-pan (d3-zoom) inspect a dense level; the SEMANTIC zoom
 * (which level is shown) is driven by the parent via `currentParentId` — on a
 * level change the d3-zoom transform resets.
 */
const VB = 1000; // internal viewBox size (square); SVG scales to the container.
const PAD = 120;

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

  // Layout: normalise the visible themes' x,y into the padded viewBox via d3 scales.
  const layout = useMemo(() => {
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

    const pos = new Map<string, Pos>();
    for (const t of visible) {
      pos.set(t.id, { cx: sx(t.x), cy: sy(t.y), r: sr(t.weight) });
    }
    return pos;
  }, [visible]);

  const visibleIds = useMemo(() => new Set(visible.map((t) => t.id)), [visible]);
  const shownEdges = useMemo(
    () => edges.filter((e) => visibleIds.has(e.a) && visibleIds.has(e.b)),
    [edges, visibleIds],
  );
  const maxEdge = useMemo(
    () => Math.max(1, ...shownEdges.map((e) => e.weight)),
    [shownEdges],
  );

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

  // --- render: data-join edges + bubbles. Edges read node centres from `layout`
  //     so they are ALWAYS anchored to the bubbles, at any zoom/pan. ---
  useEffect(() => {
    const g = select<SVGGElement, unknown>(gRef.current!);

    // edges layer ----------------------------------------------------------
    const edgeLayer = ensureLayer(g, 'map__edges');
    edgeLayer
      .selectAll<SVGLineElement, SpatialEdge>('line.map__edge')
      .data(shownEdges, (e) => `${e.a}--${e.b}`)
      .join(
        (enter) => enter.append('line').attr('class', 'map__edge'),
        (update) => update,
        (exit) => exit.remove(),
      )
      .attr('x1', (e) => layout.get(e.a)!.cx)
      .attr('y1', (e) => layout.get(e.a)!.cy)
      .attr('x2', (e) => layout.get(e.b)!.cx)
      .attr('y2', (e) => layout.get(e.b)!.cy)
      .attr('stroke-width', (e) => 0.5 + (e.weight / maxEdge) * 3);

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
      .style('fill', (t) => consensusColor(t.consensus));

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
  }, [visible, shownEdges, layout, maxEdge, selectedId, q, minConsensus]);

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
          <i className="dot dot--low" /> consensus faible
        </span>
        <span className="map__legenditem">
          <i className="dot dot--high" /> consensus fort
        </span>
        <span className="map__legenditem map__legendhint">
          taille = volume d'avis · molette = zoom · double-clic = explorer
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

/** Consensus → orange ramp (pale = low, saturated Agora orange = high). */
function consensusColor(c: number): string {
  const t = Math.max(0, Math.min(1, c));
  // interpolate from pale sand to deep Agora orange
  const lerp = (a: number, b: number) => Math.round(a + (b - a) * t);
  const r = lerp(0xf6, 0xc8);
  const g = lerp(0xd9, 0x53);
  const b = lerp(0xb8, 0x12);
  return `rgb(${r}, ${g}, ${b})`;
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}
