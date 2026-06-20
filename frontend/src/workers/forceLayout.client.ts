import type { PhysicsParams, WorkerInbound, WorkerOutbound } from './forceLayout.protocol';

/**
 * Thin client wrapping the force-layout worker.
 *
 * IMPORTANT: This is the seam that lets us replace the JS d3 simulation with a
 * Rust/WASM worker later without touching the renderer. Anyone reading positions
 * goes through {@link getPositions} — they don't know or care who computed them.
 */
export class ForceLayoutClient {
  private worker: Worker;
  private sab: SharedArrayBuffer | null = null;
  private sharedView: Float32Array | null = null;
  private fallbackView: Float32Array | null = null;
  private nodeCount = 0;

  ready: Promise<{ usingSAB: boolean }>;

  private onCooled?: () => void;

  constructor() {
    this.worker = new Worker(new URL('./forceLayout.worker.ts', import.meta.url), {
      type: 'module',
    });
    this.ready = new Promise((resolve) => {
      const handle = (e: MessageEvent<WorkerOutbound>) => {
        if (e.data.type === 'ready') {
          this.worker.removeEventListener('message', handle);
          resolve({ usingSAB: e.data.usingSAB });
        }
      };
      this.worker.addEventListener('message', handle);
    });

    this.worker.addEventListener('message', (e: MessageEvent<WorkerOutbound>) => {
      const msg = e.data;
      if (msg.type === 'tick' && msg.positions) {
        this.fallbackView = msg.positions;
      }
      if (msg.type === 'cooled') {
        this.onCooled?.();
      }
    });
  }

  init(nodes: { id: string; type: string }[], links: { source: string; target: string; type: string }[]) {
    this.nodeCount = nodes.length;
    const bytes = nodes.length * 3 * Float32Array.BYTES_PER_ELEMENT;

    let sab: SharedArrayBuffer | null = null;
    try {
      if (typeof SharedArrayBuffer !== 'undefined' && crossOriginIsolated) {
        sab = new SharedArrayBuffer(bytes);
      }
    } catch {
      sab = null;
    }
    this.sab = sab;
    this.sharedView = sab ? new Float32Array(sab) : null;
    this.fallbackView = sab ? null : new Float32Array(nodes.length * 3);

    const msg: WorkerInbound = { type: 'init', nodes, links, sab };
    this.worker.postMessage(msg);
  }

  /** Extend the running sim with new nodes and links. Drops SAB transport
   * in favour of transferable-buffer ticks (positions grow). */
  addNodes(
    nodes: { id: string; type: string }[],
    links: { source: string; target: string; type: string }[],
  ) {
    // The worker will drop SAB next tick; client must stop reading the old
    // shared view immediately, else getPositions() would return stale data
    // truncated to the original size.
    this.sab = null;
    this.sharedView = null;
    this.nodeCount += nodes.length;
    const msg: WorkerInbound = { type: 'addNodes', nodes, links };
    this.worker.postMessage(msg);
  }

  /** Get the latest known positions buffer. Always non-null after the first tick. */
  getPositions(): Float32Array | null {
    return this.sharedView ?? this.fallbackView;
  }

  getNodeCount() {
    return this.nodeCount;
  }

  pin(id: string, pos: [number, number, number]) {
    const msg: WorkerInbound = { type: 'pin', id, pos };
    this.worker.postMessage(msg);
  }

  unpin(id: string) {
    const msg: WorkerInbound = { type: 'unpin', id };
    this.worker.postMessage(msg);
  }

  reheat(alpha = 0.3) {
    const msg: WorkerInbound = { type: 'reheat', alpha };
    this.worker.postMessage(msg);
  }

  /** Push live physics knobs (config panel) — the worker rebuilds forces +
   * reheats so the layout re-settles into the new shape. */
  setParams(params: PhysicsParams) {
    const msg: WorkerInbound = { type: 'setParams', params };
    this.worker.postMessage(msg);
  }

  onCooledOnce(cb: () => void) {
    this.onCooled = cb;
  }

  dispose() {
    const msg: WorkerInbound = { type: 'dispose' };
    this.worker.postMessage(msg);
    this.worker.terminate();
  }
}
