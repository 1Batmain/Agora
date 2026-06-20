import { useEffect, useRef, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import type { GraphIndex } from '../lib/graphData';
import { ForceLayoutClient } from '../workers/forceLayout.client';
import { Nodes } from './Nodes';
import { Links } from './Links';

interface GraphSceneProps {
  graph: GraphIndex;
}

/**
 * The 3D force-directed swarm. Forks dummy's force-layout WORKER verbatim
 * (`workers/forceLayout.*`) — the stable, contract-frozen engine — and renders
 * a clean, self-contained scene on top (no business-domain HUD coupling).
 *
 * The worker is created ONCE per mount and fed via `init` (batch). Its `addNodes`
 * entry point is preserved intact for Phase 2 live streaming — we just don't
 * call it yet (no WS wired in Phase 1).
 */
export function GraphScene({ graph }: GraphSceneProps) {
  const clientRef = useRef<ForceLayoutClient | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const client = new ForceLayoutClient();
    clientRef.current = client;
    client.init(
      graph.nodes.map((n) => ({ id: n.id, type: n.type })),
      graph.links.map((l) => ({ source: l.source, target: l.target, type: l.type })),
    );
    client.ready.then(() => setReady(true));
    // Phase 2 hook: subscribe a WS here and call `client.addNodes(...)` on each
    // wave of incoming ideas — positions are preserved, the swarm grows live.
    return () => {
      client.dispose();
      clientRef.current = null;
    };
  }, [graph]);

  const getPositions = () => clientRef.current?.getPositions() ?? null;

  return (
    <Canvas
      camera={{ position: [0, 0, 320], fov: 55, near: 0.1, far: 6000 }}
      gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
      onCreated={({ gl }) => gl.setClearColor(0x0a0a0b, 1)}
    >
      <ambientLight intensity={0.55} />
      <directionalLight position={[200, 240, 320]} intensity={1.6} color="#ffffff" />
      <pointLight position={[-260, -80, -200]} intensity={0.8} color="#cfd6e6" />
      {ready && (
        <>
          <Links graph={graph} getPositions={getPositions} />
          <Nodes graph={graph} getPositions={getPositions} />
        </>
      )}
      <OrbitControls
        enablePan
        enableDamping
        dampingFactor={0.08}
        minDistance={40}
        maxDistance={3000}
      />
    </Canvas>
  );
}
