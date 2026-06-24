import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { SpatialTheme } from './contract';
import type {
  ExplainCluster,
  ExplainPair,
  SandboxParams,
  SandboxResponse,
} from './sandboxContract';
import { SANDBOX_DEFAULTS } from './sandboxMock';
import { explainCluster, explainPair, postSandbox, type SandboxSource } from './sandboxApi';
import { applyAnalysis, type ApplyStatus } from './analysisApi';

/**
 * TOOLBOX — the réglages drawer of the MAIN analysis page. It is NOT a separate
 * dark mixing board (that isolated `ConsoleView` is gone): it is a collapsible
 * panel, IN THE SITE STYLE (DSFR / orange), that affine the main map itself.
 *
 * Moving a knob (résolution · α cible · coarsening · τ · k) POSTs `/sandbox`
 * (debounced) and the MAIN d3-pack map re-organises live (preview). « Appliquer »
 * POSTs `/analysis/apply` to PERSIST the réglages (graceful « bientôt » if 404). A
 * discreet decision-trace (clic sur une paire / un cluster → pourquoi fusionnés /
 * pas) sits in a foldable section. NO "naming" knob — structure only.
 */

/** Sentinel parent so every reclustered bubble renders as ONE flat pack on the map. */
export const TOOLBOX_LEVEL = '__toolbox__';
const DEBOUNCE_MS = 250;

/**
 * The opening state: ONLY alpha=0 is overridden. Everything else is omitted so the
 * backend derives exactly what it served → the map is unchanged on open. `resetKnobs`
 * returns here.
 */
const INITIAL_OVERRIDES: SandboxParams = { alpha: 0 };

export type ToolboxSelection =
  | { kind: 'cluster'; id: string }
  | { kind: 'pair'; a: string; b: string }
  | null;

interface KnobDef {
  key: keyof SandboxParams;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  /** Concrete impact on THIS dataset's analysis — shown on hover/focus. */
  tip: string;
  fmt: (v: number) => string;
}

// The five knobs. NO "naming" knob — structure only. Ranges bracket the derived
// defaults so the knob opens on the backend's own choice. `tip` describes the
// CONCRETE effect on the analysis (not the algorithm), shown on hover/focus.
const KNOBS: KnobDef[] = [
  { key: 'resolution', label: 'Résolution', unit: 'Leiden', min: 0.3, max: 2.5, step: 0.05, tip: '↑ communautés Leiden plus fines → plus de clusters, plus petits. ↓ regroupe en grandes familles.', fmt: (v) => v.toFixed(2) },
  { key: 'alpha', label: 'α cible', unit: 'blend', min: 0, max: 1, step: 0.02, tip: 'Pondère l’objet de la prise de position dans l’embedding — ↑ regroupe par SUJET (ex. fusionne les 3 « addiction »). À 0, regroupe par formulation.', fmt: (v) => v.toFixed(2) },
  { key: 'coarsen_mult', label: 'Coarsening', unit: '× seuil', min: 0.3, max: 2.5, step: 0.05, tip: '↑ fusionne les macros proches → moins de clusters, plus larges. ↓ garde les nuances séparées.', fmt: (v) => '×' + v.toFixed(2) },
  { key: 'tau_mult', label: 'τ subdivision', unit: '× τ', min: 0.3, max: 2.5, step: 0.05, tip: '↑ subdivise moins → arbre plus plat. ↓ éclate les clusters diffus en sous-thèmes.', fmt: (v) => '×' + v.toFixed(2) },
  { key: 'k', label: 'k voisins', unit: 'kNN', min: 4, max: 30, step: 1, tip: 'Voisins du graphe kNN — change la densité du graphe. Très sensible : ±1 peut réorganiser toute la carte.', fmt: (v) => String(Math.round(v)) },
];

/** Defaults used to label a knob before the first /sandbox reply lands. */
const KNOB_FALLBACK: Record<keyof SandboxParams, number> = SANDBOX_DEFAULTS;

/**
 * The value a knob should DISPLAY: the user override if set, otherwise the value
 * the backend actually used (echoed in `resp.params`), otherwise a sane fallback.
 * This is why opening the toolbox shows the served analysis rather than a reset.
 */
function effectiveValue(
  key: keyof SandboxParams,
  overrides: SandboxParams,
  resp: SandboxResponse | null,
): number {
  const ov = overrides[key];
  if (typeof ov === 'number') return ov;
  const fromResp = resp?.params?.[key];
  if (typeof fromResp === 'number') return fromResp;
  return KNOB_FALLBACK[key];
}

export function Toolbox({
  dataset,
  selection,
  onSelection,
  onPreview,
  onClose,
}: {
  dataset: string;
  /** The cluster/pair currently inspected (a map click sets a cluster selection). */
  selection: ToolboxSelection;
  onSelection: (sel: ToolboxSelection) => void;
  /** Push the reclustered bubbles up to the MAIN map (null = nothing yet). */
  onPreview: (themes: SpatialTheme[] | null) => void;
  onClose: () => void;
}) {
  // OVERRIDES ONLY — never a full param set. Initial = {alpha:0}: that single
  // override reproduces the SERVED analysis (k/resolution/coarsen/τ omitted →
  // backend derives the same values it already served). So opening the toolbox
  // does NOT move the map; only turning a knob (= setting an override) does.
  const [overrides, setOverrides] = useState<SandboxParams>(INITIAL_OVERRIDES);
  const [resp, setResp] = useState<SandboxResponse | null>(null);
  const [source, setSource] = useState<SandboxSource | null>(null);
  const [busy, setBusy] = useState(false);

  const [explainC, setExplainC] = useState<ExplainCluster | null>(null);
  const [explainP, setExplainP] = useState<ExplainPair | null>(null);

  const [applyState, setApplyState] = useState<{ status: ApplyStatus; busy: boolean } | null>(null);

  // Debounced recluster: a fresh `params` object re-arms the timer; only the last
  // nudge in a 250ms window actually POSTs /sandbox. A run id guards a slow response
  // from clobbering a newer one.
  const runId = useRef(0);
  const onPreviewRef = useRef(onPreview);
  onPreviewRef.current = onPreview;
  useEffect(() => {
    const t = setTimeout(() => {
      const id = ++runId.current;
      setBusy(true);
      postSandbox(dataset, overrides)
        .then(({ data, source }) => {
          if (id !== runId.current) return;
          setResp(data);
          setSource(source);
        })
        .finally(() => {
          if (id === runId.current) setBusy(false);
        });
    }, DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [dataset, overrides]);

  // Adapt SandboxCluster[] → SpatialTheme[] so we REUSE the main d3-pack renderer
  // (area = n_avis, hue = cluster id, paleness = cohesion). Flat (no drill); pushed
  // up to the MAIN map. A new recluster clears any stale selection.
  const dispByNode = useMemo(() => {
    const m = new Map<string, number>();
    resp?.trace.nodes.forEach((n) => m.set(n.id, n.dispersion));
    return m;
  }, [resp]);

  useEffect(() => {
    if (!resp) {
      onPreviewRef.current(null);
      return;
    }
    const themes: SpatialTheme[] = resp.clusters.map((c) => ({
      id: c.id,
      label: c.keywords.join(' · ') || c.id,
      x: 0,
      y: 0,
      n_avis: c.n_avis,
      n_claims: c.n_claims,
      weight: c.n_claims,
      consensus: c.cohesion,
      convergence: c.cohesion,
      dispersion: dispByNode.get(c.id) ?? 1 - c.cohesion,
      parent_id: TOOLBOX_LEVEL,
      has_children: false,
      color: '#888',
      hook: `${c.n_claims} claims · cohésion ${(c.cohesion * 100).toFixed(0)}%`,
    }));
    onPreviewRef.current(themes);
    onSelection(null); // structure changed → drop any stale pair/cluster inspection
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resp, dispByNode]);

  // Restore the persisted map on unmount.
  useEffect(() => () => onPreviewRef.current(null), []);

  // Resolve the explanation for the current selection against the latest /sandbox.
  useEffect(() => {
    if (!resp || !selection) {
      setExplainC(null);
      setExplainP(null);
      return;
    }
    let cancelled = false;
    if (selection.kind === 'cluster') {
      setExplainP(null);
      explainCluster(dataset, selection.id, resp).then((e) => !cancelled && setExplainC(e));
    } else {
      setExplainC(null);
      explainPair(dataset, selection.a, selection.b, resp).then((e) => !cancelled && setExplainP(e));
    }
    return () => {
      cancelled = true;
    };
  }, [dataset, selection, resp]);

  // Turning a knob sets (or clears) an OVERRIDE — it does not assemble a full set.
  const setKnob = useCallback((key: keyof SandboxParams, value: number) => {
    setOverrides((o) => ({ ...o, [key]: value }));
  }, []);
  const resetKnobs = useCallback(() => setOverrides(INITIAL_OVERRIDES), []);

  const onApply = useCallback(async () => {
    setApplyState({ status: 'ok', busy: true });
    const res = await applyAnalysis(dataset, overrides);
    setApplyState({ status: res.status, busy: false });
  }, [dataset, overrides]);

  const nClusters = resp?.clusters.length ?? 0;
  const merges = resp?.trace.pairs.filter((p) => p.merged).length ?? 0;
  const splits = resp?.trace.nodes.filter((n) => n.subdivided).length ?? 0;

  return (
    <section className="tbx" aria-label="Réglages de l’analyse">
      <div className="tbx__head">
        <div className="tbx__titleblock">
          <span className={`tbx__dot${busy ? ' tbx__dot--busy' : ''}`} aria-hidden />
          <strong>Affiner l’analyse</strong>
          <span className="tbx__sub">
            prévisualisation live{resp ? ` · ${nClusters} clusters · ${resp.ms} ms` : ' · …'}
          </span>
          {source && <span className={`badge badge--${source === 'live' ? 'live' : 'mock'}`}>{source}</span>}
        </div>
        <div className="tbx__actions">
          <button className="tbx__reset" onClick={resetKnobs} title="Revenir à l’analyse servie (réglages dérivés des données)">
            ⟲ analyse servie
          </button>
          <ApplyButton state={applyState} onApply={onApply} />
          <button className="tbx__close" onClick={onClose} title="Fermer les réglages">
            ✕
          </button>
        </div>
      </div>

      <div className="tbx__knobs">
        {KNOBS.map((k) => (
          <Knob
            key={k.key}
            def={k}
            value={effectiveValue(k.key, overrides, resp)}
            overridden={typeof overrides[k.key] === 'number'}
            onChange={(v) => setKnob(k.key, v)}
          />
        ))}
        {resp && (
          <div className="tbx__derived" title="valeurs réelles dérivées par le backend">
            {Object.entries(resp.params.derived).map(([k, v]) => (
              <span key={k} className="tbx__derivedrow">
                <code>{k}</code>
                <b>{typeof v === 'number' ? v : String(v)}</b>
              </span>
            ))}
          </div>
        )}
      </div>

      <details className="tbx__tracebox">
        <summary>
          Pourquoi ces regroupements ?
          <span className="tbx__tracehint">
            {merges} fusion{merges > 1 ? 's' : ''} · {splits} subdivision{splits > 1 ? 's' : ''} — clic sur une bulle ou une paire
          </span>
        </summary>
        <Trace
          resp={resp}
          sel={selection}
          explainCluster={explainC}
          explainPair={explainP}
          onSelectPair={(a, b) => onSelection({ kind: 'pair', a, b })}
          onSelectCluster={(id) => onSelection({ kind: 'cluster', id })}
          onClear={() => onSelection(null)}
        />
      </details>
    </section>
  );
}

/** « Appliquer » with inline feedback: enregistré / bientôt / erreur. */
function ApplyButton({
  state,
  onApply,
}: {
  state: { status: ApplyStatus; busy: boolean } | null;
  onApply: () => void;
}) {
  const busy = state?.busy;
  const msg =
    state && !state.busy
      ? state.status === 'ok'
        ? '✓ enregistré'
        : state.status === 'soon'
          ? 'bientôt disponible'
          : '⚠ échec'
      : null;
  return (
    <span className="tbx__applywrap">
      <button
        className="live-btn live-btn--primary tbx__apply"
        onClick={onApply}
        disabled={busy}
        title="Enregistrer ces réglages sur l’analyse (persistant)"
      >
        {busy ? 'enregistrement…' : 'Appliquer ces réglages'}
      </button>
      {msg && (
        <span
          className={`tbx__applymsg tbx__applymsg--${state!.status}`}
          title={state!.status === 'soon' ? 'l’endpoint de persistance arrive côté backend' : undefined}
        >
          {msg}
        </span>
      )}
    </span>
  );
}

// --- Rotary potard geometry: a 270° arc from bottom-left to bottom-right. ---
const ARC_START = -135; // degrees, 0 = top, clockwise positive
const ARC_SWEEP = 270;
const KNOB_R = 30; // arc radius (svg units)
const KNOB_BOX = 78; // svg viewBox size
const DRAG_RANGE_PX = 170; // vertical px to traverse the full range

/** Polar → cartesian with 0° at the TOP and clockwise-positive angles. */
function polar(cx: number, cy: number, r: number, deg: number): [number, number] {
  const rad = (deg * Math.PI) / 180;
  return [cx + r * Math.sin(rad), cy - r * Math.cos(rad)];
}

/** SVG arc path from angle a1 → a2 (clockwise). */
function arcPath(cx: number, cy: number, r: number, a1: number, a2: number): string {
  const [x1, y1] = polar(cx, cy, r, a1);
  const [x2, y2] = polar(cx, cy, r, a2);
  const largeArc = a2 - a1 > 180 ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** Snap to the knob's step and round float noise away. */
function snap(v: number, def: KnobDef): number {
  const snapped = Math.round(v / def.step) * def.step;
  return Number(clamp(snapped, def.min, def.max).toFixed(6));
}

/**
 * One ROTARY potard (console-style, but in the site's DSFR/orange skin): a graded
 * arc, an orange fill up to the current angle, a pointer, the value in the centre.
 * Drag vertically (up = ↑) to turn it; the keyboard ↑/↓/←/→/PageUp/Down/Home/End
 * works too (role="slider"). Hover/focus reveals the concrete-impact tooltip.
 */
function Knob({
  def,
  value,
  overridden,
  onChange,
}: {
  def: KnobDef;
  value: number;
  overridden: boolean;
  onChange: (v: number) => void;
}) {
  const drag = useRef<{ startY: number; startVal: number } | null>(null);

  const norm = clamp((value - def.min) / (def.max - def.min), 0, 1);
  const angle = ARC_START + norm * ARC_SWEEP;
  const c = KNOB_BOX / 2;
  const [px, py] = polar(c, c, KNOB_R, angle);
  const [hx, hy] = polar(c, c, KNOB_R - 11, angle); // inner end of the pointer

  const onPointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { startY: e.clientY, startVal: value };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    const dy = drag.current.startY - e.clientY; // up = positive
    const fine = e.shiftKey ? 0.25 : 1; // Shift = fine control
    const delta = (dy / DRAG_RANGE_PX) * (def.max - def.min) * fine;
    onChange(snap(drag.current.startVal + delta, def));
  };
  const endDrag = (e: React.PointerEvent) => {
    drag.current = null;
    try {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      /* capture may already be gone */
    }
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    const big = (def.max - def.min) / 10;
    let next: number | null = null;
    switch (e.key) {
      case 'ArrowUp':
      case 'ArrowRight':
        next = value + def.step;
        break;
      case 'ArrowDown':
      case 'ArrowLeft':
        next = value - def.step;
        break;
      case 'PageUp':
        next = value + big;
        break;
      case 'PageDown':
        next = value - big;
        break;
      case 'Home':
        next = def.min;
        break;
      case 'End':
        next = def.max;
        break;
      default:
        return;
    }
    e.preventDefault();
    onChange(snap(next, def));
  };

  return (
    <div className={`pot${overridden ? ' pot--override' : ''}`}>
      <div
        className="pot__dial"
        role="slider"
        tabIndex={0}
        aria-label={`${def.label} (${def.unit})`}
        aria-valuemin={def.min}
        aria-valuemax={def.max}
        aria-valuenow={value}
        aria-valuetext={`${def.fmt(value)} ${def.unit}`}
        aria-describedby={`pot-tip-${def.key}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onKeyDown={onKeyDown}
      >
        <svg viewBox={`0 0 ${KNOB_BOX} ${KNOB_BOX}`} className="pot__svg" aria-hidden>
          <path className="pot__track" d={arcPath(c, c, KNOB_R, ARC_START, ARC_START + ARC_SWEEP)} />
          {norm > 0 && <path className="pot__fill" d={arcPath(c, c, KNOB_R, ARC_START, angle)} />}
          <circle className="pot__hub" cx={c} cy={c} r={KNOB_R - 12} />
          <line className="pot__needle" x1={hx} y1={hy} x2={px} y2={py} />
          <circle className="pot__dot" cx={px} cy={py} r={3.2} />
        </svg>
        <span className="pot__val">{def.fmt(value)}</span>
      </div>
      <span className="pot__label">
        {def.label}
        <span className="pot__unit">{def.unit}</span>
      </span>
      <span className="pot__badge" aria-hidden>{overridden ? 'réglé' : 'auto'}</span>
      <span id={`pot-tip-${def.key}`} role="tooltip" className="pot__tip">{def.tip}</span>
    </div>
  );
}

/**
 * DECISION-TRACE (in-site). Three modes: a PAIR selected → merge criteria; a
 * CLUSTER selected → subdivision criterion + nearest clusters; nothing → the global
 * candidate-merge / subdivision lists.
 */
function Trace({
  resp,
  sel,
  explainCluster,
  explainPair,
  onSelectPair,
  onSelectCluster,
  onClear,
}: {
  resp: SandboxResponse | null;
  sel: ToolboxSelection;
  explainCluster: ExplainCluster | null;
  explainPair: ExplainPair | null;
  onSelectPair: (a: string, b: string) => void;
  onSelectCluster: (id: string) => void;
  onClear: () => void;
}) {
  if (!resp) return <div className="tbx-trace tbx-trace--empty">en attente du premier recluster…</div>;

  if (sel?.kind === 'pair' && explainPair) {
    const e = explainPair;
    return (
      <div className="tbx-trace">
        <div className="tbx-trace__head">
          <strong>Paire {e.pair[0]} ↔ {e.pair[1]}</strong>
          <button className="tbx-trace__back" onClick={onClear}>← liste</button>
        </div>
        <Verdict merged={e.merged} />
        <Crit label="similarité" value={e.sim} refLabel="seuil" refValue={e.threshold} pass={e.sim >= e.threshold} />
        <Crit label="cohésion A" value={e.cohesion_a} />
        <Crit label="cohésion B" value={e.cohesion_b} />
        <Crit label="cohésion min" value={e.cohesion_min} refLabel="garde" refValue={0.3} pass={e.cohesion_min >= 0.3} />
        <p className="tbx-trace__note">
          Fusion si <b>sim ≥ seuil</b> ET <b>cohésion min ≥ garde</b> (aucun cluster trop diffus pour absorber l’autre).
        </p>
      </div>
    );
  }

  if (sel?.kind === 'cluster' && explainCluster) {
    const e = explainCluster;
    const cluster = resp.clusters.find((c) => c.id === e.cluster);
    return (
      <div className="tbx-trace">
        <div className="tbx-trace__head">
          <strong>Cluster {e.cluster}</strong>
          <button className="tbx-trace__back" onClick={onClear}>← trace</button>
        </div>
        {cluster && (
          <>
            <div className="tbx-trace__kw">{cluster.keywords.join(' · ')}</div>
            <div className="tbx-trace__stats">
              <span>{cluster.n_claims} claims</span>
              <span>{cluster.n_avis} avis</span>
              <span>cohésion {(cluster.cohesion * 100).toFixed(0)}%</span>
            </div>
          </>
        )}
        <div className="tbx-trace__section">Subdivision</div>
        <Crit label="dispersion" value={e.node.dispersion} refLabel="τ" refValue={e.node.tau} pass={e.node.subdivided} />
        <p className="tbx-trace__note">
          {e.node.subdivided
            ? 'dispersion > τ → ce nœud se subdivise.'
            : 'dispersion ≤ τ → nœud cohérent, pas de subdivision.'}
        </p>
        <div className="tbx-trace__section">Voisins (k plus proches)</div>
        <div className="tbx-trace__rows">
          {explainCluster.neighbors.map((n) => (
            <button
              key={n.id}
              className={`tbx-trace__row${n.merged ? ' is-merged' : ''}`}
              onClick={() => onSelectPair(e.cluster, n.id)}
            >
              <span className="tbx-trace__rowid">{n.id}</span>
              <span className="tbx-trace__rowmeta">sim {n.sim.toFixed(2)}</span>
              <span className="tbx-trace__rowmeta">coh {n.cohesion.toFixed(2)}</span>
              {n.merged && <span className="tbx-trace__tag">fusionné</span>}
            </button>
          ))}
        </div>
        {cluster?.sample_claims?.length ? (
          <>
            <div className="tbx-trace__section">Exemples</div>
            <ul className="tbx-trace__samples">
              {cluster.sample_claims.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          </>
        ) : null}
      </div>
    );
  }

  // Default: candidate merges + subdivisions.
  return (
    <div className="tbx-trace">
      <div className="tbx-trace__section">Fusions candidates</div>
      <div className="tbx-trace__rows">
        {resp.trace.pairs.map((p) => (
          <button
            key={`${p.a}-${p.b}`}
            className={`tbx-trace__row${p.merged ? ' is-merged' : ''}`}
            onClick={() => onSelectPair(p.a, p.b)}
          >
            <span className="tbx-trace__rowid">{p.a} ↔ {p.b}</span>
            <span className="tbx-trace__rowmeta">
              sim {p.sim.toFixed(2)} {p.sim >= p.threshold ? '≥' : '<'} {p.threshold.toFixed(2)}
            </span>
            <span className={`tbx-trace__verdict ${p.merged ? 'is-yes' : 'is-no'}`}>
              {p.merged ? 'fusion' : 'séparé'}
            </span>
          </button>
        ))}
        {resp.trace.pairs.length === 0 && <span className="tbx-trace__note">aucune paire candidate.</span>}
      </div>
      <div className="tbx-trace__section">Subdivisions</div>
      <div className="tbx-trace__rows">
        {resp.trace.nodes.map((n) => (
          <button
            key={n.id}
            className={`tbx-trace__row${n.subdivided ? ' is-split' : ''}`}
            onClick={() => onSelectCluster(n.id)}
          >
            <span className="tbx-trace__rowid">{n.id}</span>
            <span className="tbx-trace__rowmeta">
              disp {n.dispersion.toFixed(2)} {n.subdivided ? '>' : '≤'} τ {n.tau.toFixed(2)}
            </span>
            {n.subdivided && <span className="tbx-trace__tag is-split">subdivise</span>}
          </button>
        ))}
      </div>
    </div>
  );
}

function Verdict({ merged }: { merged: boolean }) {
  return (
    <div className={`tbx-trace__bigverdict ${merged ? 'is-yes' : 'is-no'}`}>
      {merged ? '✓ Fusionnés' : '✕ Séparés'}
    </div>
  );
}

function Crit({
  label,
  value,
  refLabel,
  refValue,
  pass,
}: {
  label: string;
  value: number;
  refLabel?: string;
  refValue?: number;
  pass?: boolean;
}) {
  const hasRef = refLabel != null && refValue != null;
  return (
    <div className={`crit${pass === true ? ' crit--pass' : pass === false ? ' crit--fail' : ''}`}>
      <span className="crit__label">{label}</span>
      <span className="crit__bar">
        <span className="crit__fill" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
        {hasRef && <span className="crit__ref" style={{ left: `${Math.max(0, Math.min(1, refValue)) * 100}%` }} />}
      </span>
      <span className="crit__val">
        {value.toFixed(2)}
        {hasRef && <em> / {refLabel} {refValue.toFixed(2)}</em>}
      </span>
    </div>
  );
}
