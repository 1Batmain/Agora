import { useEffect, useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { GraphIndex } from '../lib/graphData';
import { useGraphStore } from '../state/useGraphStore';

interface LinksProps {
  graph: GraphIndex;
  getPositions: () => Float32Array | null;
}

/**
 * The k-NN edges (`links`, type "knn") as a single LineSegments. Endpoint
 * positions track the worker buffer each frame; per-vertex colour encodes the
 * edge weight (`props.weight`, cosine similarity) as brightness, and dims when
 * neither endpoint is in the selected theme.
 */
export function Links({ graph, getPositions }: LinksProps) {
  const segments = graph.links.length;

  // Pre-resolve each link's endpoint indices into the positions buffer.
  const endpoints = useMemo(() => {
    const arr = new Int32Array(segments * 2);
    for (let i = 0; i < segments; i++) {
      arr[i * 2] = graph.indexOf.get(graph.links[i].source) ?? -1;
      arr[i * 2 + 1] = graph.indexOf.get(graph.links[i].target) ?? -1;
    }
    return arr;
  }, [graph, segments]);

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(segments * 2 * 3), 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(segments * 2 * 3), 3));
    return g;
  }, [segments]);

  const material = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.5,
        depthWrite: false,
      }),
    [],
  );
  useEffect(() => () => geometry.dispose(), [geometry]);
  useEffect(() => () => material.dispose(), [material]);

  // Sentinel so the per-vertex weight colours are written on the first frame
  // (the default no-selection key is '' — an empty initial ref would skip it,
  // leaving the lines black).
  const lastColorKey = useRef('<init>');

  useFrame(() => {
    const positions = getPositions();
    if (!positions) return;
    const pos = geometry.getAttribute('position') as THREE.BufferAttribute;
    const colAttr = geometry.getAttribute('color') as THREE.BufferAttribute;
    const parr = pos.array as Float32Array;

    for (let i = 0; i < segments; i++) {
      const a = endpoints[i * 2];
      const b = endpoints[i * 2 + 1];
      if (a < 0 || b < 0) continue;
      parr[i * 6] = positions[a * 3];
      parr[i * 6 + 1] = positions[a * 3 + 1];
      parr[i * 6 + 2] = positions[a * 3 + 2];
      parr[i * 6 + 3] = positions[b * 3];
      parr[i * 6 + 4] = positions[b * 3 + 1];
      parr[i * 6 + 5] = positions[b * 3 + 2];
    }
    pos.needsUpdate = true;

    // Recolour only when the selected theme changes.
    const { selectedClusterId } = useGraphStore.getState();
    const key = `${selectedClusterId ?? ''}`;
    if (key !== lastColorKey.current) {
      const carr = colAttr.array as Float32Array;
      for (let i = 0; i < segments; i++) {
        const link = graph.links[i];
        const w = link.props?.weight ?? 0.5;
        // Weight (cosine) → brightness; dim edges outside the selected theme.
        const srcC = graph.byId.get(link.source)?.cluster_id;
        const tgtC = graph.byId.get(link.target)?.cluster_id;
        const inSel =
          selectedClusterId == null ||
          srcC === selectedClusterId ||
          tgtC === selectedClusterId;
        const v = (0.25 + 0.55 * w) * (inSel ? 1 : 0.12);
        for (let k = 0; k < 6; k += 3) {
          carr[i * 6 + k] = v;
          carr[i * 6 + k + 1] = v;
          carr[i * 6 + k + 2] = v * 1.15; // faint cool tint
        }
      }
      colAttr.needsUpdate = true;
      lastColorKey.current = key;
    }
  });

  return <lineSegments geometry={geometry} material={material} frustumCulled={false} />;
}
