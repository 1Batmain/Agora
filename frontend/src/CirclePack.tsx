import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { buildPack, type PackNode } from './hierarchy';
import type { GraphPayload } from './types';

interface View {
  x: number; // focus centre x (layout space)
  y: number;
  r: number; // focus diameter (layout space) to fit the viewport
}

interface Props {
  payload: GraphPayload;
  onSelect: (node: PackNode | null) => void;
  selectedId: string | null;
}

const ZOOM_MS = 720;
const SIZE = 1000; // layout space; mapped to screen by `k = side / view.r`

/**
 * Zoomable circle packing (macro → sub-theme → avis). Zoom is done by tweening a
 * `view` [cx, cy, diameter] over requestAnimationFrame and projecting every
 * circle to screen space each frame — this keeps labels at constant screen size
 * (no counter-scaling jank). Click a circle to zoom in; click the background to
 * zoom back out one level.
 */
export function CirclePack({ payload, onSelect, selectedId }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [side, setSide] = useState(720);

  // Measure the available square.
  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSide(Math.max(320, Math.min(r.width, r.height)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const root = useMemo(() => buildPack(payload, SIZE), [payload]);

  const [focus, setFocus] = useState<PackNode>(root);
  const [view, setView] = useState<View>({ x: root.x, y: root.y, r: root.r * 2 });
  const rafRef = useRef<number | null>(null);

  // Reset to the top whenever the data changes (a recluster).
  useEffect(() => {
    setFocus(root);
    setView({ x: root.x, y: root.y, r: root.r * 2 });
    onSelect(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [root]);

  function zoomTo(target: PackNode) {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    const from = view;
    const to: View = { x: target.x, y: target.y, r: target.r * 2 };
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ZOOM_MS);
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; // easeInOutQuad
      setView({
        x: from.x + (to.x - from.x) * e,
        y: from.y + (to.y - from.y) * e,
        r: from.r + (to.r - from.r) * e,
      });
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    setFocus(target);
  }

  useEffect(() => () => void (rafRef.current && cancelAnimationFrame(rafRef.current)), []);

  function handleClick(d: PackNode, e: React.MouseEvent) {
    e.stopPropagation();
    if (d.data.kind === 'avis') {
      onSelect(d); // leaf: show in panel, no zoom
      return;
    }
    // macro / sub: zoom in. Surface avis in the panel whenever this circle holds
    // avis directly (a sub-theme in Leiden, OR a flat cluster in HDBSCAN).
    zoomTo(d);
    onSelect(holdsAvis(d) ? d : null);
  }

  function handleBackground() {
    const up = focus.parent ?? root;
    zoomTo(up);
    onSelect(holdsAvis(up) ? up : null);
  }

  const k = side / view.r;
  const project = (d: PackNode) => ({
    x: (d.x - view.x) * k + side / 2,
    y: (d.y - view.y) * k + side / 2,
    r: d.r * k,
  });

  // All circles except the root (root is the backdrop).
  const circles = root.descendants().filter((d) => d.depth > 0);

  return (
    <div className="viz" ref={wrapRef}>
      <svg
        className="viz__svg"
        width={side}
        height={side}
        viewBox={`0 0 ${side} ${side}`}
        onClick={handleBackground}
      >
        <rect x={0} y={0} width={side} height={side} fill="transparent" />
        {circles.map((d) => {
          const p = project(d);
          if (p.r < 0.4 || p.x + p.r < 0 || p.x - p.r > side || p.y + p.r < 0 || p.y - p.r > side)
            return null; // cull off-screen / sub-pixel
          const isLeaf = d.data.kind === 'avis';
          const selected = d.data.id === selectedId;
          return (
            <circle
              key={d.data.id}
              cx={p.x}
              cy={p.y}
              r={p.r}
              fill={d.data.color}
              fillOpacity={isLeaf ? 0.92 : d.depth === 1 ? 0.18 : 0.32}
              stroke={selected ? '#ffffff' : d.data.color}
              strokeOpacity={selected ? 1 : isLeaf ? 0.5 : 0.85}
              strokeWidth={selected ? 2.5 : 1}
              style={{ cursor: 'pointer' }}
              onClick={(e) => handleClick(d, e)}
            >
              <title>{`${d.data.label}${d.data.theme ? ` · ${d.data.theme.size} avis` : ''}`}</title>
            </circle>
          );
        })}
        {/* Labels: only the direct children of the current focus, Bostock-style. */}
        {circles
          .filter((d) => d.parent === focus && d.data.kind !== 'avis')
          .map((d) => {
            const p = project(d);
            if (p.r < 18) return null;
            return (
              <text
                key={`l-${d.data.id}`}
                x={p.x}
                y={p.y}
                className="viz__label"
                textAnchor="middle"
                dominantBaseline="middle"
                pointerEvents="none"
              >
                {truncate(d.data.label, Math.max(6, Math.round(p.r / 4)))}
              </text>
            );
          })}
      </svg>
      <div className="viz__hint">
        clic = zoom · fond = retour · {focus.data.kind === 'root' ? 'macro-thèmes' : focus.data.label}
      </div>
    </div>
  );
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n - 1) + '…';
}

/** True if this circle's direct children are avis (sub-theme or flat cluster). */
function holdsAvis(d: PackNode): boolean {
  return !!d.children?.some((c) => c.data.kind === 'avis');
}
