/**
 * LIVE stream client (SSE) for the "replay en direct" view.
 *
 * Subscribes to `GET /stream?dataset=<id>` (text/event-stream) and dispatches the
 * incremental build events from `/tmp/contract-live.md`:
 *   - `snapshot`    → initial state (themes without x,y + dataset stats/context);
 *   - `claim_added` → a theme's metrics grew (the bubble swells);
 *   - `theme_split` → a theme divided into children (children appear, parent reorgs);
 *   - `done`        → end of replay.
 *
 * Two sources, same handler shape:
 *   - REAL : an `EventSource` on the backend `/stream` (proxied `/api/stream`).
 *   - MOCK : a simulated replay (deterministic, seeded from the dataset id) so the
 *            view is fully exerciseable before the backend lands. The mock derives
 *            its tree from `mockAnalysis`, then scripts a populate→split→populate
 *            schedule that emits the exact same event shapes as the real stream.
 *
 * `openLiveStream` returns a cancel function (closes the EventSource / clears the
 * mock timers) so the caller can stop or restart cleanly.
 */
import type { DatasetStats, SpatialTheme } from './contract';
import { mockAnalysis } from './mock';

/** Initial state: the themes present at t0 (no x,y — UMAP dropped) + dataset meta. */
export interface SnapshotEvent {
  type: 'snapshot';
  themes: SpatialTheme[];
  dataset_stats?: DatasetStats;
  dataset_context?: string;
}

/** A theme's metrics grew (more voices/claims) — its bubble swells. Absolute values. */
export interface ClaimAddedEvent {
  type: 'claim_added';
  theme_id: string;
  n_avis: number;
  n_claims: number;
  weight: number;
  dispersion: number;
  consensus: number;
  convergence?: number;
}

/** A theme divided: its children appear (entry animation), the parent reorganises. */
export interface ThemeSplitEvent {
  type: 'theme_split';
  parent_id: string;
  children: SpatialTheme[];
}

/** End of the replay. */
export interface DoneEvent {
  type: 'done';
  n_avis: number;
  n_themes: number;
}

export type LiveEvent = SnapshotEvent | ClaimAddedEvent | ThemeSplitEvent | DoneEvent;

export interface LiveHandlers {
  onSnapshot: (e: SnapshotEvent) => void;
  onClaimAdded: (e: ClaimAddedEvent) => void;
  onThemeSplit: (e: ThemeSplitEvent) => void;
  onDone: (e: DoneEvent) => void;
  onError: (err: unknown) => void;
}

export interface LiveOptions {
  /** Replay the simulated mock stream instead of the real backend. */
  mock?: boolean;
  /** Delay between mock events (ms). Ignored for the real stream. */
  intervalMs?: number;
}

/** Dispatch one decoded event to the right handler. */
function dispatch(ev: LiveEvent, h: LiveHandlers): void {
  switch (ev.type) {
    case 'snapshot':
      h.onSnapshot(ev);
      break;
    case 'claim_added':
      h.onClaimAdded(ev);
      break;
    case 'theme_split':
      h.onThemeSplit(ev);
      break;
    case 'done':
      h.onDone(ev);
      break;
  }
}

/**
 * Subscribe to the live build of `dataset`. Returns a cancel function.
 *
 * Real mode: opens an `EventSource`; each `data:` line is JSON-parsed and
 * dispatched. We close on `done` (the contract is one-shot) so the browser does
 * not auto-reconnect and replay forever.
 */
export function openLiveStream(
  dataset: string,
  handlers: LiveHandlers,
  opts: LiveOptions = {},
): () => void {
  if (opts.mock) return openMockStream(dataset, handlers, opts.intervalMs ?? 280);

  let es: EventSource | null = null;
  let closed = false;
  try {
    es = new EventSource(`/api/stream?dataset=${encodeURIComponent(dataset)}`);
  } catch (e) {
    handlers.onError(e);
    return () => {};
  }
  const close = () => {
    closed = true;
    es?.close();
  };
  es.onmessage = (msg: MessageEvent) => {
    if (closed) return;
    let ev: LiveEvent | null = null;
    try {
      ev = JSON.parse(msg.data) as LiveEvent;
    } catch {
      return; // ignore keep-alives / unparseable frames
    }
    // The real stream omits x,y (UMAP dropped); the d3-pack ignores them anyway,
    // but the shared SpatialTheme still types them — default so TS/consumers are happy.
    if (ev.type === 'snapshot') ev.themes = ev.themes.map(withXY);
    if (ev.type === 'theme_split') ev.children = ev.children.map(withXY);
    dispatch(ev, handlers);
    if (ev.type === 'done') close();
  };
  es.onerror = (e) => {
    if (closed) return;
    handlers.onError(e);
    close(); // do not let EventSource silently retry; the caller decides.
  };
  return close;
}

/** Ensure x,y exist (the d3-pack never reads them, but the type requires them). */
function withXY(t: SpatialTheme): SpatialTheme {
  return { ...t, x: t.x ?? 0, y: t.y ?? 0 };
}

// --------------------------------------------------------------------------
// MOCK stream — a scripted, deterministic replay in the contract's event shape.
// --------------------------------------------------------------------------

/** Estimate claims from a current voice count, preserving the theme's claim ratio. */
function claimsAt(t: SpatialTheme, n: number): number {
  const ratio = t.n_avis > 0 ? t.n_claims / t.n_avis : 1.3;
  return Math.max(n, Math.round(n * ratio));
}

/**
 * Build the full event list for a mock replay, then return a thunk-cancellable
 * player. Derives the tree from `mockAnalysis` and scripts:
 *   snapshot (roots, small) → grow roots → split roots → grow children → split
 *   children with grandchildren → finish growing → done.
 * Every emitted theme starts with `has_children:false` so it reads as a live leaf
 * until its own `theme_split` arrives (matching the real contract examples).
 */
export function buildMockEvents(dataset: string): LiveEvent[] {
  const { themes, dataset_stats, dataset_context } = mockAnalysis(dataset);
  const byId = new Map(themes.map((t) => [t.id, t]));
  const childrenOf = (id: string) => themes.filter((t) => t.parent_id === id);
  const roots = themes.filter((t) => t.parent_id === null);

  const events: LiveEvent[] = [];
  const cur = new Map<string, number>();
  const startN = (t: SpatialTheme) => Math.max(1, Math.round(t.n_avis * 0.12));
  const target = (t: SpatialTheme) => Math.max(1, t.n_avis);

  /** A frontier theme as broadcast: full shape, current voices, no children yet. */
  const frame = (t: SpatialTheme): SpatialTheme => {
    const n = cur.get(t.id)!;
    return { ...withXY(t), has_children: false, n_avis: n, n_claims: claimsAt(t, n), weight: n };
  };

  roots.forEach((t) => cur.set(t.id, startN(t)));
  events.push({
    type: 'snapshot',
    themes: roots.map(frame),
    dataset_stats,
    dataset_context,
  });

  const frontier = new Set(roots.map((t) => t.id));
  const split = new Set<string>();

  const claimEvent = (t: SpatialTheme, n: number): ClaimAddedEvent => ({
    type: 'claim_added',
    theme_id: t.id,
    n_avis: n,
    n_claims: claimsAt(t, n),
    weight: n,
    dispersion: t.dispersion,
    consensus: t.consensus,
    convergence: t.convergence,
  });

  // One growth wave: nudge every still-growing frontier bubble toward its target.
  const grow = (frac: number) => {
    for (const id of frontier) {
      const t = byId.get(id)!;
      const c = cur.get(id)!;
      const tg = target(t);
      if (c >= tg) continue;
      const next = Math.min(tg, c + Math.max(1, Math.ceil((tg - c) * frac)));
      cur.set(id, next);
      events.push(claimEvent(t, next));
    }
  };

  // Split every frontier theme that has (not-yet-revealed) children.
  const doSplits = () => {
    for (const id of [...frontier]) {
      if (split.has(id)) continue;
      const kids = childrenOf(id);
      if (!kids.length) continue;
      split.add(id);
      frontier.delete(id);
      kids.forEach((k) => cur.set(k.id, startN(k)));
      events.push({ type: 'theme_split', parent_id: id, children: kids.map(frame) });
      kids.forEach((k) => frontier.add(k.id));
    }
  };

  grow(0.5);
  grow(0.6);
  doSplits(); // roots → their children
  grow(0.5);
  grow(0.6);
  doSplits(); // children → grandchildren
  grow(0.7);
  grow(1);
  // Make sure everything reaches its target before we finish.
  for (const id of frontier) {
    const t = byId.get(id)!;
    if (cur.get(id)! < target(t)) {
      cur.set(id, target(t));
      events.push(claimEvent(t, target(t)));
    }
  }

  const totalAvis = [...frontier].reduce((s, id) => s + cur.get(id)!, 0);
  events.push({ type: 'done', n_avis: totalAvis, n_themes: frontier.size });
  return events;
}

/** Play a pre-built mock event list on a timer; returns a cancel function. */
function openMockStream(
  dataset: string,
  handlers: LiveHandlers,
  intervalMs: number,
): () => void {
  const events = buildMockEvents(dataset);
  let i = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let cancelled = false;

  const tick = () => {
    if (cancelled || i >= events.length) return;
    dispatch(events[i++], handlers);
    if (i < events.length) timer = setTimeout(tick, intervalMs);
  };
  // Kick off after a beat so the caller has mounted its handlers.
  timer = setTimeout(tick, 80);

  return () => {
    cancelled = true;
    if (timer) clearTimeout(timer);
  };
}
