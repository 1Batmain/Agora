declare module 'd3-force-3d' {
  export interface SimulationNodeDatum3 {
    index?: number;
    x?: number;
    y?: number;
    z?: number;
    vx?: number;
    vy?: number;
    vz?: number;
    fx?: number | null;
    fy?: number | null;
    fz?: number | null;
  }

  export interface SimulationLinkDatum3<N extends SimulationNodeDatum3> {
    source: string | N;
    target: string | N;
    index?: number;
  }

  export interface Force<N extends SimulationNodeDatum3, L> {
    (alpha: number): void;
    initialize?(nodes: N[]): void;
  }

  export interface Simulation<N extends SimulationNodeDatum3, L> {
    nodes(): N[];
    nodes(nodes: N[]): this;
    alpha(): number;
    alpha(a: number): this;
    alphaMin(): number;
    alphaMin(a: number): this;
    alphaDecay(): number;
    alphaDecay(a: number): this;
    alphaTarget(): number;
    alphaTarget(a: number): this;
    velocityDecay(): number;
    velocityDecay(a: number): this;
    force(name: string): unknown;
    force(name: string, force: unknown): this;
    on(type: string, listener?: (...args: unknown[]) => void): this;
    restart(): this;
    stop(): this;
    tick(iter?: number): this;
  }

  export function forceSimulation<N extends SimulationNodeDatum3, L>(
    nodes?: N[],
    numDimensions?: number
  ): Simulation<N, L>;

  export interface ForceLink<N extends SimulationNodeDatum3, L extends SimulationLinkDatum3<N>>
    extends Force<N, L> {
    links(links?: L[]): L[] | this;
    id(fn: (d: N) => string): this;
    distance(fn: number | ((l: L) => number)): this;
    strength(fn: number | ((l: L) => number)): this;
  }
  export function forceLink<N extends SimulationNodeDatum3, L extends SimulationLinkDatum3<N>>(
    links?: L[]
  ): ForceLink<N, L>;

  export interface ForceManyBody<N extends SimulationNodeDatum3> extends Force<N, never> {
    strength(s: number | ((d: N) => number)): this;
    distanceMin(d: number): this;
    distanceMax(d: number): this;
    theta(t: number): this;
  }
  export function forceManyBody<N extends SimulationNodeDatum3>(): ForceManyBody<N>;

  export interface ForceCenter<N extends SimulationNodeDatum3> extends Force<N, never> {
    x(x: number): this;
    y(y: number): this;
    z(z: number): this;
    strength(s: number): this;
  }
  export function forceCenter<N extends SimulationNodeDatum3>(
    x?: number,
    y?: number,
    z?: number
  ): ForceCenter<N>;

  export interface ForceCollide<N extends SimulationNodeDatum3> extends Force<N, never> {
    radius(r: number | ((n: N) => number)): this;
    strength(s: number): this;
    iterations(n: number): this;
  }
  export function forceCollide<N extends SimulationNodeDatum3>(
    radius?: number | ((n: N) => number)
  ): ForceCollide<N>;
}
