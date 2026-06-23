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
import { SpatialMap } from './SpatialMap';

/**
 * ANALYST MODE — "console de mixage". A dark mixing-board of FADERS (résolution ·
 * α · coarsening · τ · k) drives a LIVE recluster: nudging a fader (debounced
 * ~250ms) POSTs `/sandbox` and the d3-pack map re-organises with smooth
 * transitions, while a DECISION-TRACE panel explains why clusters merged or split.
 *
 * It is a full-screen overlay (its own board chrome) that NEVER touches the main
 * public `/analysis` view underneath — close returns there intact. No "naming"
 * knob: this console is about the cluster STRUCTURE, not labels.
 */

/** Sentinel level so every reclustered bubble renders as ONE flat pack. */
const FLAT_LEVEL = '__sandbox__';
const DEBOUNCE_MS = 250;

interface FaderDef {
  key: keyof SandboxParams;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  hint: string;
  /** how to format the live value readout */
  fmt: (v: number) => string;
}

// The five console faders. NO "naming" knob — structure only. Ranges bracket the
// derived defaults so the centre detent is the backend's own choice.
const FADERS: FaderDef[] = [
  { key: 'resolution', label: 'RÉSOLUTION', unit: 'Leiden', min: 0.3, max: 2.5, step: 0.05, hint: '↑ plus de clusters, plus fins', fmt: (v) => v.toFixed(2) },
  { key: 'alpha', label: 'α CIBLE', unit: 'blend', min: 0, max: 1, step: 0.02, hint: 'poids de la cible dans l’embedding', fmt: (v) => v.toFixed(2) },
  { key: 'coarsen_mult', label: 'COARSEN', unit: '× seuil', min: 0.3, max: 2.5, step: 0.05, hint: '↑ fusionne plus → moins de clusters', fmt: (v) => '×' + v.toFixed(2) },
  { key: 'tau_mult', label: 'τ SUBDIV.', unit: '× τ', min: 0.3, max: 2.5, step: 0.05, hint: '↑ subdivise moins', fmt: (v) => '×' + v.toFixed(2) },
  { key: 'k', label: 'k voisins', unit: 'kNN', min: 4, max: 30, step: 1, hint: 'densité du graphe kNN', fmt: (v) => String(Math.round(v)) },
];

type Selection = { kind: 'cluster'; id: string } | { kind: 'pair'; a: string; b: string } | null;

export function ConsoleView({
  dataset,
  datasetLabel,
  onClose,
}: {
  dataset: string;
  datasetLabel?: string;
  onClose: () => void;
}) {
  // Live fader values (update instantly for the readout); the recluster is debounced.
  const [params, setParams] = useState<Required<SandboxParams>>(SANDBOX_DEFAULTS);
  const [resp, setResp] = useState<SandboxResponse | null>(null);
  const [source, setSource] = useState<SandboxSource | null>(null);
  const [busy, setBusy] = useState(false);
  const [sel, setSel] = useState<Selection>(null);
  const [explainC, setExplainC] = useState<ExplainCluster | null>(null);
  const [explainP, setExplainP] = useState<ExplainPair | null>(null);

  // Debounced recluster: a fresh `params` object re-arms the timer; only the last
  // nudge in a 250ms window actually POSTs /sandbox. A run id guards against a slow
  // response overwriting a newer one.
  const runId = useRef(0);
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

  const setFader = useCallback((key: keyof SandboxParams, value: number) => {
    setParams((p) => ({ ...p, [key]: value }));
  }, []);
  const resetFaders = useCallback(() => setParams(SANDBOX_DEFAULTS), []);

  // Adapt SandboxCluster[] → SpatialTheme[] so we REUSE the d3-pack renderer (size =
  // n_avis, hue = cluster id, paleness = cohesion). All flat (no drill); a click
  // SELECTS the cluster for the trace panel rather than descending.
  const dispByNode = useMemo(() => {
    const m = new Map<string, number>();
    resp?.trace.nodes.forEach((n) => m.set(n.id, n.dispersion));
    return m;
  }, [resp]);

  const themes: SpatialTheme[] = useMemo(() => {
    if (!resp) return [];
    return resp.clusters.map((c) => ({
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
      parent_id: FLAT_LEVEL,
      has_children: false,
      color: '#888',
      hook: `${c.n_claims} claims · cohésion ${(c.cohesion * 100).toFixed(0)}%`,
    }));
  }, [resp, dispByNode]);

  // Fetch the explanation for the current selection (cluster neighbourhood OR pair
  // criteria) against the latest /sandbox response.
  useEffect(() => {
    if (!resp || !sel) {
      setExplainC(null);
      setExplainP(null);
      return;
    }
    let cancelled = false;
    if (sel.kind === 'cluster') {
      setExplainP(null);
      explainCluster(dataset, sel.id, resp).then((e) => !cancelled && setExplainC(e));
    } else {
      setExplainC(null);
      explainPair(dataset, sel.a, sel.b, resp).then((e) => !cancelled && setExplainP(e));
    }
    return () => {
      cancelled = true;
    };
  }, [dataset, sel, resp]);

  const selectedClusterId =
    sel?.kind === 'cluster' ? sel.id : null;

  const onSelectTheme = useCallback((t: SpatialTheme) => {
    setSel((cur) => (cur?.kind === 'cluster' && cur.id === t.id ? null : { kind: 'cluster', id: t.id }));
  }, []);

  const nClusters = resp?.clusters.length ?? 0;
  const merges = resp?.trace.pairs.filter((p) => p.merged).length ?? 0;
  const splits = resp?.trace.nodes.filter((n) => n.subdivided).length ?? 0;

  return (
    <div className="console">
      <header className="console__bar">
        <div className="console__title">
          <span className="console__led" data-on={busy ? 'busy' : 'idle'} />
          <strong>CONSOLE DE MIXAGE</strong>
          <span className="console__sub">recluster live · {datasetLabel || dataset}</span>
        </div>
        <div className="console__meters">
          <Meter label="clusters" value={String(nClusters)} />
          <Meter label="fusions" value={String(merges)} />
          <Meter label="subdiv." value={String(splits)} />
          <Meter label="claims" value={resp ? String(resp.n_claims) : '—'} />
          <Meter label="latence" value={resp ? `${resp.ms} ms` : '—'} />
          {source && <span className={`console__src console__src--${source}`}>{source}</span>}
        </div>
        <button className="console__close" onClick={onClose} title="Revenir à la vue principale">
          ✕ Fermer
        </button>
      </header>

      <div className="console__body">
        {/* LEFT — the fader bank (the mixing board). */}
        <aside className="console__rack">
          <div className="console__rackhead">
            <span>FADERS</span>
            <button className="console__reset" onClick={resetFaders} title="Revenir aux défauts dérivés">
              ⟲ défauts
            </button>
          </div>
          <div className="console__faders">
            {FADERS.map((f) => (
              <Fader
                key={f.key}
                def={f}
                value={params[f.key] as number}
                onChange={(v) => setFader(f.key, v)}
              />
            ))}
          </div>
          {resp && (
            <div className="console__derived">
              <span className="console__derivedhead">valeurs dérivées</span>
              {Object.entries(resp.params.derived).map(([k, v]) => (
                <span key={k} className="console__derivedrow">
                  <code>{k}</code>
                  <b>{typeof v === 'number' ? v : String(v)}</b>
                </span>
              ))}
            </div>
          )}
        </aside>

        {/* CENTRE — the live map. */}
        <main className="console__stage">
          {themes.length ? (
            <SpatialMap
              themes={themes}
              edges={[]}
              currentParentId={FLAT_LEVEL}
              selectedId={selectedClusterId}
              onSelect={onSelectTheme}
              onDrill={onSelectTheme}
              live
            />
          ) : (
            <div className="console__empty">
              <span className="spinner" /> premier recluster…
            </div>
          )}
        </main>

        {/* RIGHT — decision trace. */}
        <aside className="console__trace">
          <TracePanel
            resp={resp}
            sel={sel}
            explainCluster={explainC}
            explainPair={explainP}
            onSelectPair={(a, b) => setSel({ kind: 'pair', a, b })}
            onSelectCluster={(id) => setSel({ kind: 'cluster', id })}
            onClear={() => setSel(null)}
          />
        </aside>
      </div>
    </div>
  );
}

/** One meter readout in the board's top bar. */
function Meter({ label, value }: { label: string; value: string }) {
  return (
    <span className="console__meter">
      <span className="console__meterval">{value}</span>
      <span className="console__meterlbl">{label}</span>
    </span>
  );
}

/** A single vertical fader (range input rotated) with its live value + label. */
function Fader({
  def,
  value,
  onChange,
}: {
  def: FaderDef;
  value: number;
  onChange: (v: number) => void;
}) {
  const pct = ((value - def.min) / (def.max - def.min)) * 100;
  return (
    <div className="fader" title={def.hint}>
      <div className="fader__val">{def.fmt(value)}</div>
      <div className="fader__track" style={{ '--pct': `${pct}%` } as React.CSSProperties}>
        <input
          type="range"
          className="fader__input"
          min={def.min}
          max={def.max}
          step={def.step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          aria-label={`${def.label} (${def.unit})`}
        />
      </div>
      <div className="fader__label">{def.label}</div>
      <div className="fader__unit">{def.unit}</div>
    </div>
  );
}

/**
 * DECISION-TRACE panel. Three modes:
 *  - a CLUSTER is selected → its subdivision criterion (dispersion vs τ) + its k
 *    nearest clusters (sim / cohesion / merged), each row openable as a pair;
 *  - a PAIR is selected → the merge criteria (sim vs threshold, cohesions, verdict);
 *  - nothing selected → the global list of candidate merges from the trace.
 */
function TracePanel({
  resp,
  sel,
  explainCluster,
  explainPair,
  onSelectPair,
  onSelectCluster,
  onClear,
}: {
  resp: SandboxResponse | null;
  sel: Selection;
  explainCluster: ExplainCluster | null;
  explainPair: ExplainPair | null;
  onSelectPair: (a: string, b: string) => void;
  onSelectCluster: (id: string) => void;
  onClear: () => void;
}) {
  if (!resp) {
    return <div className="trace trace--empty">en attente du recluster…</div>;
  }

  if (sel?.kind === 'pair' && explainPair) {
    const e = explainPair;
    return (
      <div className="trace">
        <div className="trace__head">
          <strong>Paire {e.pair[0]} ↔ {e.pair[1]}</strong>
          <button className="trace__back" onClick={onClear}>← liste</button>
        </div>
        <Verdict merged={e.merged} />
        <Crit label="similarité" value={e.sim} ref_label="seuil" ref_value={e.threshold} pass={e.sim >= e.threshold} />
        <Crit label="cohésion A" value={e.cohesion_a} />
        <Crit label="cohésion B" value={e.cohesion_b} />
        <Crit label="cohésion min" value={e.cohesion_min} ref_label="garde" ref_value={0.3} pass={e.cohesion_min >= 0.3} />
        <p className="trace__note">
          Fusion si <b>sim ≥ seuil</b> ET <b>cohésion min ≥ garde</b> (aucun cluster trop diffus pour absorber l’autre).
        </p>
      </div>
    );
  }

  if (sel?.kind === 'cluster' && explainCluster) {
    const e = explainCluster;
    const cluster = resp.clusters.find((c) => c.id === e.cluster);
    return (
      <div className="trace">
        <div className="trace__head">
          <strong>Cluster {e.cluster}</strong>
          <button className="trace__back" onClick={onClear}>← trace</button>
        </div>
        {cluster && (
          <>
            <div className="trace__kw">{cluster.keywords.join(' · ')}</div>
            <div className="trace__stats">
              <span>{cluster.n_claims} claims</span>
              <span>{cluster.n_avis} avis</span>
              <span>cohésion {(cluster.cohesion * 100).toFixed(0)}%</span>
            </div>
          </>
        )}
        <div className="trace__section">SUBDIVISION</div>
        <Crit
          label="dispersion"
          value={e.node.dispersion}
          ref_label="τ"
          ref_value={e.node.tau}
          pass={e.node.subdivided}
        />
        <p className="trace__note">
          {e.node.subdivided
            ? 'dispersion > τ → ce nœud se subdivise.'
            : 'dispersion ≤ τ → nœud cohérent, pas de subdivision.'}
        </p>
        <div className="trace__section">VOISINS (k plus proches)</div>
        <div className="trace__neighbors">
          {e.neighbors.map((n) => (
            <button
              key={n.id}
              className={`trace__nb${n.merged ? ' trace__nb--merged' : ''}`}
              onClick={() => onSelectPair(e.cluster, n.id)}
            >
              <span className="trace__nbid">{n.id}</span>
              <span className="trace__nbsim">sim {n.sim.toFixed(2)}</span>
              <span className="trace__nbcoh">coh {n.cohesion.toFixed(2)}</span>
              {n.merged && <span className="trace__nbtag">fusionné</span>}
            </button>
          ))}
        </div>
        {cluster?.sample_claims?.length ? (
          <>
            <div className="trace__section">EXEMPLES</div>
            <ul className="trace__samples">
              {cluster.sample_claims.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          </>
        ) : null}
      </div>
    );
  }

  // Default: the global merge-candidate list.
  return (
    <div className="trace">
      <div className="trace__head">
        <strong>Trace de décision</strong>
      </div>
      <p className="trace__hint">
        Cliquez une bulle pour ses critères, ou une paire candidate ci-dessous.
      </p>
      <div className="trace__section">FUSIONS CANDIDATES</div>
      <div className="trace__pairs">
        {resp.trace.pairs.map((p) => (
          <button
            key={`${p.a}-${p.b}`}
            className={`trace__pair${p.merged ? ' trace__pair--merged' : ''}`}
            onClick={() => onSelectPair(p.a, p.b)}
          >
            <span className="trace__pairids">{p.a} ↔ {p.b}</span>
            <span className="trace__pairsim">
              sim {p.sim.toFixed(2)} {p.sim >= p.threshold ? '≥' : '<'} {p.threshold.toFixed(2)}
            </span>
            <span className={`trace__pairverdict ${p.merged ? 'is-yes' : 'is-no'}`}>
              {p.merged ? 'fusion' : 'séparé'}
            </span>
          </button>
        ))}
        {resp.trace.pairs.length === 0 && <span className="trace__hint">aucune paire candidate.</span>}
      </div>
      <div className="trace__section">SUBDIVISIONS</div>
      <div className="trace__nodes">
        {resp.trace.nodes.map((n) => (
          <button
            key={n.id}
            className={`trace__node${n.subdivided ? ' trace__node--split' : ''}`}
            onClick={() => onSelectCluster(n.id)}
          >
            <span className="trace__nodeid">{n.id}</span>
            <span className="trace__nodedisp">
              disp {n.dispersion.toFixed(2)} {n.subdivided ? '>' : '≤'} τ {n.tau.toFixed(2)}
            </span>
            {n.subdivided && <span className="trace__nodetag">subdivise</span>}
          </button>
        ))}
      </div>
    </div>
  );
}

/** A pass/fail verdict chip. */
function Verdict({ merged }: { merged: boolean }) {
  return (
    <div className={`trace__verdict ${merged ? 'is-yes' : 'is-no'}`}>
      {merged ? '✓ FUSIONNÉS' : '✕ SÉPARÉS'}
    </div>
  );
}

/** One criterion row: value, optional comparison to a threshold, pass/fail tint. */
function Crit({
  label,
  value,
  ref_label,
  ref_value,
  pass,
}: {
  label: string;
  value: number;
  ref_label?: string;
  ref_value?: number;
  pass?: boolean;
}) {
  const hasRef = ref_label != null && ref_value != null;
  return (
    <div className={`crit${pass === true ? ' crit--pass' : pass === false ? ' crit--fail' : ''}`}>
      <span className="crit__label">{label}</span>
      <span className="crit__bar">
        <span className="crit__fill" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
        {hasRef && (
          <span className="crit__ref" style={{ left: `${Math.max(0, Math.min(1, ref_value)) * 100}%` }} />
        )}
      </span>
      <span className="crit__val">
        {value.toFixed(2)}
        {hasRef && <em> / {ref_label} {ref_value.toFixed(2)}</em>}
      </span>
    </div>
  );
}
