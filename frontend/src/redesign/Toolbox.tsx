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
  hint: string;
  fmt: (v: number) => string;
}

// The five knobs. NO "naming" knob — structure only. Ranges bracket the derived
// defaults so the centre is the backend's own choice.
const KNOBS: KnobDef[] = [
  { key: 'resolution', label: 'Résolution', unit: 'Leiden', min: 0.3, max: 2.5, step: 0.05, hint: '↑ plus de clusters, plus fins', fmt: (v) => v.toFixed(2) },
  { key: 'alpha', label: 'α cible', unit: 'blend', min: 0, max: 1, step: 0.02, hint: 'poids de la cible (objet de la prise de position) dans l’embedding', fmt: (v) => v.toFixed(2) },
  { key: 'coarsen_mult', label: 'Coarsening', unit: '× seuil', min: 0.3, max: 2.5, step: 0.05, hint: '↑ fusionne plus → moins de clusters', fmt: (v) => '×' + v.toFixed(2) },
  { key: 'tau_mult', label: 'τ subdivision', unit: '× τ', min: 0.3, max: 2.5, step: 0.05, hint: '↑ subdivise moins', fmt: (v) => '×' + v.toFixed(2) },
  { key: 'k', label: 'k voisins', unit: 'kNN', min: 4, max: 30, step: 1, hint: 'densité du graphe kNN', fmt: (v) => String(Math.round(v)) },
];

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
  const [params, setParams] = useState<Required<SandboxParams>>(SANDBOX_DEFAULTS);
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
      postSandbox(dataset, params)
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
  }, [dataset, params]);

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

  const setKnob = useCallback((key: keyof SandboxParams, value: number) => {
    setParams((p) => ({ ...p, [key]: value }));
  }, []);
  const resetKnobs = useCallback(() => setParams(SANDBOX_DEFAULTS), []);

  const onApply = useCallback(async () => {
    setApplyState({ status: 'ok', busy: true });
    const res = await applyAnalysis(dataset, params);
    setApplyState({ status: res.status, busy: false });
  }, [dataset, params]);

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
          <button className="tbx__reset" onClick={resetKnobs} title="Revenir aux défauts dérivés des données">
            ⟲ défauts
          </button>
          <ApplyButton state={applyState} onApply={onApply} />
          <button className="tbx__close" onClick={onClose} title="Fermer les réglages">
            ✕
          </button>
        </div>
      </div>

      <div className="tbx__knobs">
        {KNOBS.map((k) => (
          <Knob key={k.key} def={k} value={params[k.key] as number} onChange={(v) => setKnob(k.key, v)} />
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

/** One horizontal knob (DSFR-style range) with its live value + label. */
function Knob({
  def,
  value,
  onChange,
}: {
  def: KnobDef;
  value: number;
  onChange: (v: number) => void;
}) {
  const pct = ((value - def.min) / (def.max - def.min)) * 100;
  return (
    <label className="knob" title={def.hint}>
      <span className="knob__top">
        <span className="knob__label">{def.label}</span>
        <span className="knob__val">{def.fmt(value)}</span>
      </span>
      <input
        type="range"
        className="knob__input"
        style={{ '--pct': `${pct}%` } as React.CSSProperties}
        min={def.min}
        max={def.max}
        step={def.step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-label={`${def.label} (${def.unit})`}
      />
      <span className="knob__unit">{def.unit}</span>
    </label>
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
