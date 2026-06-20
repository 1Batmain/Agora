/**
 * Force-layout worker protocol — STABLE.
 *
 * This boundary is what the future Rust/WASM force-layout worker has to honor.
 * Any change here is a breaking change to both the JS sim and any future swap.
 */

/**
 * Live-tunable d3-force-3d parameters (the config panel writes these). Each maps
 * onto the existing force model as a global knob on top of the per-node
 * gravity/radius the worker already computes — defaults reproduce the baked-in
 * layout exactly, so an untouched panel changes nothing.
 */
export interface PhysicsParams {
  /** forceManyBody strength — repulsion (negative) / attraction (positive). */
  chargeStrength: number;
  /** Multiplier on each spring's collision-aware rest length (>1 = looser). */
  linkDistance: number;
  /** forceCenter pull toward the origin (0 = no centering). */
  centerStrength: number;
  /** Multiplier on each node's collision radius (>1 = more spacing). */
  collideRadius: number;
}

/** Defaults reproduce the previously hard-coded constants verbatim. */
export const DEFAULT_PHYSICS: PhysicsParams = {
  chargeStrength: -18,
  linkDistance: 1,
  centerStrength: 1,
  collideRadius: 1,
};

export type WorkerInbound =
  | { type: 'init'; nodes: { id: string; type: string }[]; links: { source: string; target: string; type: string }[]; sab?: SharedArrayBuffer | null }
  | {
      // Extend an already-running sim with new nodes and links. Existing
      // positions are preserved; the SAB transport (if any) is dropped and
      // the worker switches to transferable-buffer ticks (which can grow).
      type: 'addNodes';
      nodes: { id: string; type: string }[];
      links: { source: string; target: string; type: string }[];
    }
  | { type: 'focus'; subgraphIds: string[] }
  | { type: 'pin'; id: string; pos: [number, number, number] }
  | { type: 'unpin'; id: string }
  | { type: 'reheat'; alpha: number }
  // Live physics retune from the config panel — reapplies forces + reheats.
  | { type: 'setParams'; params: PhysicsParams }
  | { type: 'dispose' };

export type WorkerOutbound =
  | { type: 'ready'; usingSAB: boolean }
  | { type: 'tick'; positions?: Float32Array; alpha: number }
  | { type: 'cooled' };
