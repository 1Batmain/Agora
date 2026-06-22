import { useEffect, useMemo, useRef, useState } from 'react';
import type { SpatialEdge, SpatialTheme } from './contract';

/**
 * F3 — spatial theme map. Themes are placed at their UMAP (x,y); proximity =
 * semantic similarity. Bubble area ∝ weight (volume of avis), colour = consensus.
 * Edges = co-occurrence. Navigation is ADAPTIVE drill: you only ever see ONE
 * level (the children of the current parent); double-clicking a drillable bubble
 * (`has_children`) descends; a leaf bubble selects → its citations. NO raw avis
 * are ever shown on the map itself — only at the leaf, in the side panel.
 *
 * Free wheel-zoom + drag-pan are layered on top for inspecting a dense level, but
 * the SEMANTIC zoom (which level is shown) is driven by the parent via
 * `currentParentId`.
 */
const VB = 1000; // internal viewBox size (square); SVG scales to the container.
const PAD = 120;

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
  const isDim = (t: SpatialTheme) =>
    (q !== '' && !t.label.toLowerCase().includes(q)) || t.consensus < minConsensus;
  const visible = useMemo(
    () => themes.filter((t) => t.parent_id === currentParentId),
    [themes, currentParentId],
  );

  // Layout: normalise the visible themes' x,y into the padded viewBox.
  const layout = useMemo(() => {
    const xs = visible.map((t) => t.x);
    const ys = visible.map((t) => t.y);
    const minX = Math.min(...xs),
      maxX = Math.max(...xs);
    const minY = Math.min(...ys),
      maxY = Math.max(...ys);
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const maxW = Math.max(...visible.map((t) => t.weight), 1);
    const pos = new Map<string, { cx: number; cy: number; r: number }>();
    for (const t of visible) {
      const cx = PAD + ((t.x - minX) / spanX) * (VB - 2 * PAD);
      // SVG y grows downward; flip so "up" matches UMAP up.
      const cy = PAD + (1 - (t.y - minY) / spanY) * (VB - 2 * PAD);
      const r = 26 + Math.sqrt(t.weight / maxW) * 70;
      pos.set(t.id, { cx, cy, r });
    }
    return pos;
  }, [visible]);

  const visibleIds = useMemo(() => new Set(visible.map((t) => t.id)), [visible]);
  const shownEdges = useMemo(
    () => edges.filter((e) => visibleIds.has(e.a) && visibleIds.has(e.b)),
    [edges, visibleIds],
  );
  const maxEdge = Math.max(1, ...shownEdges.map((e) => e.weight));

  // Pan/zoom transform (resets when the level changes).
  const [view, setView] = useState({ k: 1, tx: 0, ty: 0 });
  useEffect(() => setView({ k: 1, tx: 0, ty: 0 }), [currentParentId]);

  const svgRef = useRef<SVGSVGElement | null>(null);
  const drag = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);

  function toSvg(clientX: number, clientY: number) {
    const rect = svgRef.current!.getBoundingClientRect();
    return {
      x: ((clientX - rect.left) / rect.width) * VB,
      y: ((clientY - rect.top) / rect.height) * VB,
    };
  }

  function onWheel(e: React.WheelEvent) {
    e.preventDefault();
    const p = toSvg(e.clientX, e.clientY);
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    setView((v) => {
      const k = Math.min(6, Math.max(0.5, v.k * factor));
      // keep the cursor point stable
      const tx = p.x - ((p.x - v.tx) * k) / v.k;
      const ty = p.y - ((p.y - v.ty) * k) / v.k;
      return { k, tx, ty };
    });
  }

  function onPointerDown(e: React.PointerEvent) {
    if (e.button !== 0) return;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty };
  }
  function onPointerMove(e: React.PointerEvent) {
    if (!drag.current) return;
    const rect = svgRef.current!.getBoundingClientRect();
    const dx = ((e.clientX - drag.current.x) / rect.width) * VB;
    const dy = ((e.clientY - drag.current.y) / rect.height) * VB;
    setView((v) => ({ ...v, tx: drag.current!.tx + dx, ty: drag.current!.ty + dy }));
  }
  function onPointerUp() {
    drag.current = null;
  }

  return (
    <div className="map">
      <svg
        ref={svgRef}
        className="map__svg"
        viewBox={`0 0 ${VB} ${VB}`}
        preserveAspectRatio="xMidYMid meet"
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        <g transform={`translate(${view.tx} ${view.ty}) scale(${view.k})`}>
          {/* edges */}
          {shownEdges.map((e, i) => {
            const a = layout.get(e.a)!;
            const b = layout.get(e.b)!;
            return (
              <line
                key={i}
                className="map__edge"
                x1={a.cx}
                y1={a.cy}
                x2={b.cx}
                y2={b.cy}
                strokeWidth={0.5 + (e.weight / maxEdge) * 3}
              />
            );
          })}
          {/* bubbles */}
          {visible.map((t) => {
            const p = layout.get(t.id)!;
            const selected = t.id === selectedId;
            return (
              <g
                key={t.id}
                className={`bubble${selected ? ' bubble--sel' : ''}${
                  t.has_children ? ' bubble--drill' : ' bubble--leaf'
                }${isDim(t) ? ' bubble--dim' : ''}`}
                style={{ transform: `translate(${p.cx}px, ${p.cy}px)` }}
                onClick={() => onSelect(t)}
                onDoubleClick={() => (t.has_children ? onDrill(t) : onSelect(t))}
              >
                <title>
                  {t.label} — {t.n_avis} avis · consensus {Math.round(t.consensus * 100)}%
                  {t.has_children ? ' · double-clic pour zoomer' : ' · feuille (citations)'}
                </title>
                <circle
                  className="bubble__circle"
                  r={p.r}
                  style={{ fill: consensusColor(t.consensus) }}
                />
                {t.has_children && <circle className="bubble__ring" r={p.r + 6} />}
                <text className="bubble__label" y={p.r + 22}>
                  {truncate(t.label, 26)}
                </text>
                <text className="bubble__stat" y={4}>
                  {t.n_avis}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
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
