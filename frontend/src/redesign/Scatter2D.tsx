import { useEffect, useRef } from 'react';

/** One UMAP-2D point (one idea / one contribution), coloured by its cluster. */
export interface ScatterPoint {
  x: number;
  z: number;
  cluster_id: string | null;
  color: string;
}

/**
 * « Nuage UMAP 2D » — a flat scatter of the consultation's ideas. Each `point`
 * is one idea at its UMAP-2D position `(x, z)`, painted in `point.color` (its
 * macro-cluster colour, same palette as the bubbles and the 3D landscape). Unlike
 * the bubble graph, position HERE is semantic (UMAP proximity = similar ideas), so
 * clusters read as visual clouds.
 *
 * Rendered on a `<canvas>` (cheap for a few thousand dots, redraws on every
 * re-cluster without churning the DOM). Data bounds are computed from the points
 * and mapped to the device-pixel canvas with a small margin; a `ResizeObserver`
 * keeps the drawing crisp as the container resizes. The slider drives the colours
 * (cluster membership) — the dots themselves stay put (UMAP is fixed).
 */
const MARGIN = 18; // px gap between the cloud and the canvas edge.
const DOT_R = 2.6; // dot radius in CSS px.

export function Scatter2D({
  points,
  legend = 'Nuage UMAP 2D des contributions · couleur = cluster (mise à jour au re-clustering)',
}: {
  points: ScatterPoint[];
  /** Légende personnalisable (la source des couleurs varie selon l'appelant). */
  legend?: string;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;

    const draw = () => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cssW = wrap.clientWidth || 1;
      const cssH = wrap.clientHeight || 1;
      // Size the backing store for the device pixel ratio (crisp dots), then draw
      // in CSS-pixel space so MARGIN/DOT_R read intuitively.
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${cssH}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);
      if (!points.length) return;

      // Data bounds → fit the cloud into the inner (margined) rect, preserving the
      // UMAP aspect ratio so clusters aren't sheared.
      let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
      for (const p of points) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.z < minZ) minZ = p.z;
        if (p.z > maxZ) maxZ = p.z;
      }
      const spanX = maxX - minX || 1;
      const spanZ = maxZ - minZ || 1;
      const innerW = Math.max(1, cssW - 2 * MARGIN);
      const innerH = Math.max(1, cssH - 2 * MARGIN);
      const scale = Math.min(innerW / spanX, innerH / spanZ);
      // Centre the scaled cloud within the inner rect.
      const offX = MARGIN + (innerW - spanX * scale) / 2;
      const offY = MARGIN + (innerH - spanZ * scale) / 2;

      ctx.globalAlpha = 0.82;
      for (const p of points) {
        const px = offX + (p.x - minX) * scale;
        // canvas y grows downward → flip z so the cloud isn't mirrored vertically.
        const py = offY + (maxZ - p.z) * scale;
        ctx.fillStyle = p.color || '#000091';
        ctx.beginPath();
        ctx.arc(px, py, DOT_R, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    };

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [points]);

  return (
    <div ref={wrapRef} className="scatter2d">
      <canvas ref={canvasRef} className="scatter2d__canvas" />
      {!points.length && (
        <div className="scatter2d__overlay">nuage indisponible pour cette consultation.</div>
      )}
      <p className="scatter2d__legend">{legend}</p>
    </div>
  );
}
