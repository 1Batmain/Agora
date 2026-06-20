import { useEffect, useMemo, useRef } from 'react';
import { useFrame, type ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import type { GraphIndex } from '../lib/graphData';
import { useGraphStore } from '../state/useGraphStore';

interface NodesProps {
  graph: GraphIndex;
  getPositions: () => Float32Array | null;
}

const NODE_RADIUS = 4.5;
const _dummy = new THREE.Object3D();
const _col = new THREE.Color();
const _base = new THREE.Color();

/**
 * The swarm itself: one InstancedMesh of spheres, each placed from the worker's
 * live positions buffer and coloured by `node.color` (the Leiden-theme palette,
 * per the cross-lane contract). Selecting a theme dims everything outside it so
 * the cluster pops; hovering a node soft-brightens it.
 */
export function Nodes({ graph, getPositions }: NodesProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const count = graph.nodes.length;

  const focusNode = useGraphStore((s) => s.focusNode);
  const hover = useGraphStore((s) => s.hover);

  // Per-node base colours from node.color — parsed once.
  const baseColors = useMemo(
    () => graph.nodes.map((n) => new THREE.Color(n.color)),
    [graph],
  );

  const geometry = useMemo(() => new THREE.SphereGeometry(NODE_RADIUS, 24, 18), []);
  const material = useMemo(
    () => new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.05 }),
    [],
  );
  useEffect(() => () => geometry.dispose(), [geometry]);
  useEffect(() => () => material.dispose(), [material]);

  // Repaint colours only when selection/hover changes, not every frame. Seeded
  // with a sentinel that no real key matches, so the theme colours are painted
  // on the very first frame (the default key "|" would otherwise be skipped).
  const lastPaintKey = useRef('<init>');

  useFrame(() => {
    const mesh = meshRef.current;
    const positions = getPositions();
    if (!mesh || !positions) return;

    // (1) Positions — every frame while the layout settles.
    for (let i = 0; i < count; i++) {
      _dummy.position.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      _dummy.updateMatrix();
      mesh.setMatrixAt(i, _dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
    // three caches the raycast bounding sphere on first pick; instances move
    // every frame, so null it or nodes outside the stale sphere stop being
    // clickable.
    mesh.boundingSphere = null;

    // (2) Colours — only when emphasis state changes.
    const { expandedMacroId, selectedClusterId, hoveredNodeId } = useGraphStore.getState();
    const key = `${expandedMacroId ?? ''}|${selectedClusterId ?? ''}|${hoveredNodeId ?? ''}`;
    if (key !== lastPaintKey.current) {
      for (let i = 0; i < count; i++) {
        const node = graph.nodes[i];
        _base.copy(baseColors[i]);
        // A picked sub-theme narrows emphasis to that leaf; otherwise an open
        // macro lights its whole branch; otherwise everything is full colour.
        const inSelection =
          selectedClusterId != null
            ? node.cluster_id === selectedClusterId
            : expandedMacroId != null
              ? node.macro_id === expandedMacroId
              : true;
        if (!inSelection) {
          // Dim out-of-theme nodes toward the dark background.
          _col.copy(_base).lerp(_base.clone().multiplyScalar(0.18), 0.85);
        } else if (node.id === hoveredNodeId) {
          _col.copy(_base).lerp(new THREE.Color('#ffffff'), 0.45);
        } else {
          _col.copy(_base);
        }
        mesh.setColorAt(i, _col);
      }
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      lastPaintKey.current = key;
    }
  });

  const onPointerMove = (e: ThreeEvent<PointerEvent>) => {
    if (e.instanceId == null) return;
    e.stopPropagation();
    hover(graph.nodes[e.instanceId].id);
  };
  const onPointerOut = () => hover(null);
  const onClick = (e: ThreeEvent<MouseEvent>) => {
    if (e.instanceId == null) return;
    e.stopPropagation();
    // Click a node → open its macro branch then its sub-theme drill-down.
    const node = graph.nodes[e.instanceId];
    focusNode(node.macro_id ?? null, node.cluster_id);
  };

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, material, count]}
      frustumCulled={false}
      onPointerMove={onPointerMove}
      onPointerOut={onPointerOut}
      onClick={onClick}
    />
  );
}
