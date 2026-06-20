/// <reference lib="webworker" />
import {
  forceSimulation,
  forceManyBody,
  forceLink,
  forceCenter,
  forceCollide,
  type SimulationNodeDatum3,
  type SimulationLinkDatum3,
} from 'd3-force-3d';
import {
  DEFAULT_PHYSICS,
  type PhysicsParams,
  type WorkerInbound,
  type WorkerOutbound,
} from './forceLayout.protocol';

interface SimNode extends SimulationNodeDatum3 {
  id: string;
  type: string;
  // Per-node gravity score in [0, 1]. Derived from the specificity of the
  // node's incident relations: a node sitting on many rare relation types
  // exerts more pull on its neighbors and keeps them at a larger floor.
  gravity: number;
  // Pre-computed minimum-distance radius (used by forceCollide and to set
  // each link's rest length so the spring force doesn't fight collision).
  radius: number;
}

const COLLIDE_BASE_RADIUS = 5;
const COLLIDE_GRAVITY_GAIN = 28;
const BASE_LINK_STRENGTH = 0.35;
const LINK_GRAVITY_GAIN = 1.6;
const LINK_DISTANCE_MARGIN = 6;

type SimLink = SimulationLinkDatum3<SimNode> & { type: string };

let sim: ReturnType<typeof forceSimulation<SimNode, SimLink>> | null = null;
let nodes: SimNode[] = [];
let linkInputs: SimLink[] = [];
let positions: Float32Array | null = null;
let sharedPositions: Float32Array | null = null;
let tickHandle: number | null = null;
// Live-tunable physics (config panel). Defaults reproduce the old constants.
let params: PhysicsParams = { ...DEFAULT_PHYSICS };
// Per-relation-type specificity, recomputed with the graph; read by the link
// strength accessor. Module-scoped so applyForces() can rebuild forces freely.
let typeSpecificity = new Map<string, number>();

const FRAME_INTERVAL_MS = 33; // ~30 fps emission

/** Recompute gravity, radius, link strength/distance from the current
 * `nodes` + `linkInputs` set. Used by both `init` and `addNodes`. */
const recomputeForces = () => {
  const typeCounts = new Map<string, number>();
  for (const l of linkInputs) typeCounts.set(l.type, (typeCounts.get(l.type) ?? 0) + 1);
  const rawSpec = new Map<string, number>();
  let maxSpec = 0;
  for (const [t, count] of typeCounts) {
    const s = 1 / Math.log(2 + count);
    rawSpec.set(t, s);
    if (s > maxSpec) maxSpec = s;
  }
  typeSpecificity = new Map<string, number>();
  for (const [t, s] of rawSpec) typeSpecificity.set(t, s / (maxSpec || 1));

  const idToNode = new Map(nodes.map((n) => [n.id, n]));
  const gravityById = new Map<string, number>();
  for (const l of linkInputs) {
    const sId = typeof l.source === 'string' ? l.source : (l.source as SimNode).id;
    const tId = typeof l.target === 'string' ? l.target : (l.target as SimNode).id;
    const s = typeSpecificity.get(l.type) ?? 0;
    gravityById.set(sId, (gravityById.get(sId) ?? 0) + s);
    gravityById.set(tId, (gravityById.get(tId) ?? 0) + s);
  }
  let maxG = 0;
  for (const g of gravityById.values()) if (g > maxG) maxG = g;
  for (const n of nodes) {
    n.gravity = (gravityById.get(n.id) ?? 0) / (maxG || 1);
    n.radius = COLLIDE_BASE_RADIUS + COLLIDE_GRAVITY_GAIN * n.gravity;
  }
  return { idToNode };
};

/** (Re)build the four forces from the current `params` + computed
 * gravity/radius/specificity, and attach them to `sim`. Called by `init` and on
 * every live `setParams` retune so the panel's knobs take effect immediately. */
const applyForces = () => {
  if (!sim) return;
  sim
    .force(
      'link',
      forceLink<SimNode, SimLink>(linkInputs)
        .id((d) => d.id)
        .distance((l) => {
          const a = l.source as SimNode;
          const b = l.target as SimNode;
          return (a.radius + b.radius + LINK_DISTANCE_MARGIN) * params.linkDistance;
        })
        .strength((l) => {
          const a = l.source as SimNode;
          const b = l.target as SimNode;
          const spec = typeSpecificity.get(l.type) ?? 0.5;
          const g = Math.max(a.gravity, b.gravity);
          return Math.min(1, BASE_LINK_STRENGTH * spec * (1 + LINK_GRAVITY_GAIN * g));
        })
    )
    .force(
      'collide',
      forceCollide<SimNode>()
        .radius((n) => n.radius * params.collideRadius)
        .strength(0.9)
        .iterations(2)
    )
    .force('charge', forceManyBody().strength(params.chargeStrength).distanceMax(220))
    .force('center', forceCenter(0, 0, 0).strength(params.centerStrength));
};

const post = (msg: WorkerOutbound, transfer?: Transferable[]) => {
  (self as unknown as Worker).postMessage(msg, transfer ?? []);
};

const writePositions = () => {
  if (!positions) return;
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    // Pinned nodes are being dragged from the main thread — don't overwrite
    // their slots. Otherwise our 30 Hz tick fights the 60 Hz pointer events
    // and the dragged node bounces between cursor pos and physics pos.
    if (n.fx != null) continue;
    positions[i * 3] = n.x ?? 0;
    positions[i * 3 + 1] = n.y ?? 0;
    positions[i * 3 + 2] = n.z ?? 0;
  }
  if (sharedPositions) {
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      if (n.fx != null) continue;
      const j = i * 3;
      sharedPositions[j] = positions[j];
      sharedPositions[j + 1] = positions[j + 1];
      sharedPositions[j + 2] = positions[j + 2];
    }
  }
};

const emitTick = () => {
  if (!sim) return;
  writePositions();
  if (sharedPositions) {
    post({ type: 'tick', alpha: sim.alpha() });
  } else if (positions) {
    // Transfer ownership for zero-copy; allocate a fresh buffer next time.
    const buf = positions;
    positions = new Float32Array(buf.length);
    post({ type: 'tick', positions: buf, alpha: sim.alpha() }, [buf.buffer]);
  }
};

const startTickPump = () => {
  if (tickHandle != null) return;
  tickHandle = self.setInterval(() => {
    if (!sim) return;
    emitTick();
    if (sim.alpha() < sim.alphaMin()) {
      stopTickPump();
      post({ type: 'cooled' });
    }
  }, FRAME_INTERVAL_MS);
};

const stopTickPump = () => {
  if (tickHandle != null) {
    self.clearInterval(tickHandle);
    tickHandle = null;
  }
};

self.addEventListener('message', (e: MessageEvent<WorkerInbound>) => {
  const msg = e.data;
  switch (msg.type) {
    case 'init': {
      nodes = msg.nodes.map((n) => ({ id: n.id, type: n.type, gravity: 0, radius: COLLIDE_BASE_RADIUS }));
      linkInputs = msg.links.map((l) => ({
        source: l.source,
        target: l.target,
        type: l.type,
      }));

      recomputeForces();

      const N = nodes.length;
      if (msg.sab && msg.sab instanceof SharedArrayBuffer && msg.sab.byteLength >= N * 3 * 4) {
        sharedPositions = new Float32Array(msg.sab);
        positions = new Float32Array(N * 3);
      } else {
        sharedPositions = null;
        positions = new Float32Array(N * 3);
      }

      sim = forceSimulation<SimNode, SimLink>(nodes, 3)
        .alpha(1)
        // ~150 iterations to cool (≈half the prior ~300) — the layout still
        // settles well but reaches rest noticeably faster, so the load doesn't
        // sit on "Simulating forces" for long.
        .alphaDecay(0.045)
        .velocityDecay(0.35)
        .on('tick', () => {});
      applyForces(); // build link/collide/charge/center from the current params

      post({ type: 'ready', usingSAB: !!sharedPositions });
      startTickPump();
      break;
    }
    case 'addNodes': {
      if (!sim) return;

      const known = new Set(nodes.map((n) => n.id));
      const incomingNodes = msg.nodes.filter((n) => !known.has(n.id));
      const incomingLinks: SimLink[] = msg.links
        .filter((l) => {
          const sOk = known.has(l.source) || incomingNodes.some((n) => n.id === l.source);
          const tOk = known.has(l.target) || incomingNodes.some((n) => n.id === l.target);
          return sOk && tOk;
        })
        .map((l) => ({ source: l.source, target: l.target, type: l.type }));

      if (incomingNodes.length === 0 && incomingLinks.length === 0) break;

      // We drop SAB transport when the sim grows — its size is baked in at
      // init time. From here on the worker emits transferable buffers each
      // tick (writePositions already handles both paths).
      sharedPositions = null;

      const oldN = nodes.length;
      const oldPositions = positions;

      // Where each already-placed node sits, so a new node can spawn NEXT TO its
      // parent/neighbour instead of at the origin — it appears where it belongs
      // and a gentle reheat settles it locally, leaving the existing layout (and
      // its relations) almost untouched. New nodes are added to the map as we go,
      // so a chain of new nodes seeds off earlier ones.
      const placedPos = new Map<string, [number, number, number]>();
      if (oldPositions) {
        for (let i = 0; i < oldN; i++) {
          placedPos.set(nodes[i].id, [oldPositions[i * 3], oldPositions[i * 3 + 1], oldPositions[i * 3 + 2]]);
        }
      }

      for (const inc of incomingNodes) {
        // Find an already-placed node this one links to, and spawn beside it.
        // (incomingLinks were just built from the message, so the endpoints are
        // still plain ids — but the type is widened, so resolve defensively.)
        const endId = (e: string | SimNode) => (typeof e === 'string' ? e : e.id);
        let seed: [number, number, number] | undefined;
        for (const l of incomingLinks) {
          const s = endId(l.source);
          const t = endId(l.target);
          if (s === inc.id && placedPos.has(t)) { seed = placedPos.get(t); break; }
          if (t === inc.id && placedPos.has(s)) { seed = placedPos.get(s); break; }
        }
        const [bx, by, bz] = seed ?? [0, 0, 0];
        const n: SimNode = { id: inc.id, type: inc.type, gravity: 0, radius: COLLIDE_BASE_RADIUS };
        n.x = bx + (Math.random() - 0.5) * 40;
        n.y = by + (Math.random() - 0.5) * 40;
        n.z = bz + (Math.random() - 0.5) * 40;
        nodes.push(n);
        placedPos.set(inc.id, [n.x, n.y, n.z]);
      }

      // Note: link.source/target may have been mutated to node refs by d3.
      // Use the canonical id when we rebuild link inputs so the link force
      // re-resolves them against the new nodes array.
      const existing = ((sim.force('link') as any)?.links?.() as SimLink[] | undefined) ?? linkInputs;
      const carried: SimLink[] = existing.map((l) => ({
        source: typeof l.source === 'string' ? l.source : ((l.source as SimNode).id),
        target: typeof l.target === 'string' ? l.target : ((l.target as SimNode).id),
        type: l.type,
      }));
      linkInputs = [...carried, ...incomingLinks];

      recomputeForces();

      const N = nodes.length;
      const newPositions = new Float32Array(N * 3);
      if (oldPositions) newPositions.set(oldPositions.subarray(0, oldN * 3), 0);
      // Seed new node slots from the jittered SimNode coords assigned above.
      for (let i = oldN; i < N; i++) {
        newPositions[i * 3] = nodes[i].x ?? 0;
        newPositions[i * 3 + 1] = nodes[i].y ?? 0;
        newPositions[i * 3 + 2] = nodes[i].z ?? 0;
      }
      positions = newPositions;

      sim.nodes(nodes);
      (sim.force('link') as any).links(linkInputs);
      // Gentle reheat — new nodes spawned beside their neighbour only need a
      // light settle; a strong reheat would re-fling the whole (settled) graph
      // and scramble its relations.
      sim.alpha(0.35).restart();
      startTickPump();
      break;
    }
    case 'reheat': {
      if (!sim) return;
      sim.alpha(msg.alpha).restart();
      startTickPump();
      break;
    }
    case 'setParams': {
      params = msg.params;
      if (!sim) break;
      // Rebuild the forces from the new knobs and reheat so the layout
      // re-settles into the new shape (the panel tweaks it live).
      applyForces();
      sim.alpha(0.5).restart();
      startTickPump();
      break;
    }
    case 'focus': {
      // Reserved for a future "focused subgraph" mode where non-focus nodes get
      // pinned in place to reduce sim cost. For now: just reheat the global sim.
      if (!sim) return;
      sim.alpha(0.4).restart();
      startTickPump();
      break;
    }
    case 'pin': {
      const idx = nodes.findIndex((nn) => nn.id === msg.id);
      if (idx >= 0) {
        const n = nodes[idx];
        n.fx = msg.pos[0];
        n.fy = msg.pos[1];
        n.fz = msg.pos[2];
        n.x = msg.pos[0];
        n.y = msg.pos[1];
        n.z = msg.pos[2];
        // Push the pin position into the buffers immediately. Without this,
        // the very first frame after `pin` arrives can still show the
        // pre-pin physics result (because the prior tick has already written
        // to the SAB and writePositions now skips this slot forever).
        const j = idx * 3;
        if (positions) {
          positions[j] = msg.pos[0];
          positions[j + 1] = msg.pos[1];
          positions[j + 2] = msg.pos[2];
        }
        if (sharedPositions) {
          sharedPositions[j] = msg.pos[0];
          sharedPositions[j + 1] = msg.pos[1];
          sharedPositions[j + 2] = msg.pos[2];
        }
      }
      break;
    }
    case 'unpin': {
      const n = nodes.find((nn) => nn.id === msg.id);
      if (n) {
        n.fx = null;
        n.fy = null;
        n.fz = null;
      }
      break;
    }
    case 'dispose': {
      stopTickPump();
      sim?.stop();
      sim = null;
      nodes = [];
      positions = null;
      sharedPositions = null;
      break;
    }
  }
});

export {};
