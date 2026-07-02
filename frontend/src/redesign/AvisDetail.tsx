import { useEffect, useRef, useState } from 'react';
import type { AvisClaim, AvisProvenance, CharRange } from './contract';
import { deleteFlag, upsertFlag } from './analysisApi';

/**
 * Full avis text with each CLAIM rendered over the verbatim source (claim-v2):
 *  - every span of a claim is HIGHLIGHTED in the claim's cluster colour (a claim
 *    can have several non-contiguous spans);
 *  - the claim's `target` (the CIBLE — a verbatim sub-range inside one span) is
 *    UNDERLINED so it stands out within the highlight.
 * Because claims are extractive — exact substrings of the avis — every range maps
 * to a real slice of the text: a faithful render with zero drift. Overlaps are
 * resolved by a character-segment sweep (see `segments`): each output segment
 * carries ONE colour (the covering claim) + an underline flag (in a target).
 * Reading-only provenance view.
 */
export function AvisDetail({
  avis,
  loading,
  dataset,
  flagText,
  onFlagChange,
  onBack,
  backLabel = '← retour aux citations',
}: {
  avis: AvisProvenance | null;
  loading: boolean;
  /** Dataset of the open avis — routes the flag upsert/delete to the right cache. */
  dataset: string | null;
  /** Current server-side flag text for this avis (undefined = not flagged). */
  flagText?: string;
  /** Notifies the parent after a flag is saved/removed so the loaded state stays fresh. */
  onFlagChange?: (avisId: string, text: string | null) => void;
  onBack: () => void;
  /** Label of the back button — defaults to citations; the explorer overrides it. */
  backLabel?: string;
}) {
  return (
    <section className="panel avisdetail">
      <div className="avisdetail__topbar">
        <button className="link-back" onClick={onBack}>
          {backLabel}
        </button>
        {avis && dataset && (
          <FlagControl
            dataset={dataset}
            avisId={avis.id}
            flagText={flagText}
            onFlagChange={onFlagChange}
          />
        )}
      </div>
      {loading ? (
        <div className="insights__loading">
          <span className="spinner" /> chargement de l'avis…
        </div>
      ) : avis ? (
        <AvisBody avis={avis} />
      ) : (
        <p className="panel__empty">Avis indisponible.</p>
      )}
    </section>
  );
}

/** Human language names (FR) for the badge — fallback to the raw code if unknown. */
const LANG_NAMES: Record<string, string> = {
  de: 'allemand',
  it: 'italien',
  en: 'anglais',
  es: 'espagnol',
  pt: 'portugais',
  nl: 'néerlandais',
  fr: 'français',
};

function langName(code?: string): string {
  if (!code) return '';
  return LANG_NAMES[code.toLowerCase().split('-')[0]] ?? code.toUpperCase();
}

/**
 * Body of one avis. Multilingual provenance:
 *  - if `lang !== 'fr'` AND a `text_fr` exists → DEFAULT to the readable French
 *    translation (no inline highlights — the verbatim spans index the ORIGINAL, not
 *    the translation), with a « voir l'original » toggle that flips to the source text
 *    WITH claim/target highlights;
 *  - otherwise (already French, or no translation available) → the original directly,
 *    highlighted, with no useless toggle.
 * A language badge announces the translation (« traduit de l'allemand »).
 */
export function AvisBody({
  avis,
  highlight = true,
  onClaimClick,
}: {
  avis: AvisProvenance;
  highlight?: boolean;
  /** Clic sur un passage surligné → statistiques de CE claim (câblé par l'explorateur). */
  onClaimClick?: (claim: AvisClaim) => void;
}) {
  const isFr = !avis.lang || avis.lang.toLowerCase().startsWith('fr');
  const hasTranslation = !isFr && typeof avis.text_fr === 'string' && avis.text_fr.length > 0;
  const [showOriginal, setShowOriginal] = useState(false);

  // Reset the toggle whenever the open avis changes (default = French when translated).
  useEffect(() => setShowOriginal(false), [avis.id]);

  // French reading view is the default for a translated avis; the original carries highlights.
  const original = showOriginal || !hasTranslation;
  // Verbatim marks render only on the ORIGINAL text AND when the global toggle is on.
  const renderHighlights = original && highlight;

  return (
    <>
      <p className="avisdetail__meta">
        {!isFr && (
          <span className="avisdetail__langbadge" title={`Langue d'origine : ${langName(avis.lang)}`}>
            {hasTranslation ? `traduit de l'${langName(avis.lang)}` : langName(avis.lang)}
          </span>
        )}
        {avis.claims.length} claim{avis.claims.length > 1 ? 's' : ''}
        {renderHighlights ? (
          <>
            {' '}surligné{avis.claims.length > 1 ? 's' : ''} · cible soulignée · chaque couleur = un thème
          </>
        ) : original ? (
          <> · surlignages masqués</>
        ) : (
          <> · couleurs = thèmes (surlignages sur l'original)</>
        )}
        {hasTranslation && (
          <button
            type="button"
            className="avisdetail__toggle"
            aria-pressed={showOriginal}
            onClick={(e) => {
              // Ne pas remonter au corps cliquable de la carte (qui ouvre la légende d'analyse).
              e.stopPropagation();
              setShowOriginal((o) => !o);
            }}
          >
            {showOriginal ? '← voir la traduction' : "voir l'original →"}
          </button>
        )}
      </p>

      {original ? (
        <article className="avisdetail__text" lang={avis.lang}>
          {renderHighlights
            ? segments(avis.text, avis.claims).map((seg, i) => {
                if (!seg.claim) return <span key={i}>{seg.text}</span>;
                // Fond du surlignage = couleur du THÈME (cluster) ; la cible (target) porte
                // une bordure pleine de cette même couleur. Les couleurs passent par des
                // CSS-VARS : le CSS ne les APPLIQUE qu'au survol de la carte (révélation au
                // hover) — le texte reste sobre au repos. Clic → stats du claim.
                const color = seg.claim.color;
                const claim = seg.claim;
                return (
                  <mark
                    key={i}
                    className={`avisdetail__hl${seg.target ? ' avisdetail__hl--target' : ''}${
                      onClaimClick ? ' avisdetail__hl--clickable' : ''
                    }`}
                    title={claimTitle(seg.claim, seg.target)}
                    style={{
                      ['--hl-bg' as string]: tint(color),
                      ['--hl-border' as string]: seg.target ? color : tint(color),
                    }}
                    onClick={
                      onClaimClick
                        ? (e) => {
                            e.stopPropagation(); // ne pas ouvrir la légende de la carte
                            onClaimClick(claim);
                          }
                        : undefined
                    }
                  >
                    {seg.text}
                  </mark>
                );
              })
            : avis.text}
        </article>
      ) : (
        <article className="avisdetail__text avisdetail__text--fr" lang="fr">
          {avis.text_fr}
        </article>
      )}
    </>
  );
}

/**
 * Flag button (top-right of the avis) → free-text feedback on a bad extraction.
 * Click opens a small compartment with ONE text field (no select). Enter sends to
 * the server immediately (upsert); clearing the text + Enter removes the flag.
 * The button is MARKED when the avis already carries a flag — state visible at load
 * (the parent feeds `flagText` from the dataset-wide `/flags`). Re-openable to edit.
 */
export function FlagControl({
  dataset,
  avisId,
  flagText,
  onFlagChange,
}: {
  dataset: string;
  avisId: string;
  flagText?: string;
  onFlagChange?: (avisId: string, text: string | null) => void;
}) {
  const flagged = typeof flagText === 'string' && flagText.trim().length > 0;
  const [open, setOpen] = useState(false);
  const [text, setText] = useState(flagText ?? '');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Keep the field in sync when the open avis (and thus its flag) changes underneath.
  useEffect(() => {
    setText(flagText ?? '');
    setOpen(false);
    setSaved(false);
  }, [avisId, flagText]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  async function submit() {
    const value = text.trim();
    setSaving(true);
    try {
      if (!value) {
        // Empty text on an existing flag → remove it; on a non-flagged avis → no-op.
        if (flagged) await deleteFlag(dataset, avisId);
        onFlagChange?.(avisId, null);
      } else {
        const flag = await upsertFlag(dataset, avisId, value);
        onFlagChange?.(avisId, flag ? flag.text : value);
      }
      setSaved(true);
      setOpen(false);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flag">
      <button
        type="button"
        className={`flag__btn${flagged ? ' flag__btn--on' : ''}`}
        aria-pressed={flagged}
        title={flagged ? `Signalé : ${flagText}` : 'Signaler une extraction à corriger'}
        onClick={() => {
          setSaved(false);
          setOpen((o) => !o);
        }}
      >
        <span aria-hidden>⚑</span> {flagged ? 'Signalé' : 'Signaler'}
      </button>
      {open && (
        <div className="flag__panel">
          <input
            ref={inputRef}
            className="flag__input"
            type="text"
            value={text}
            placeholder="Qu'est-ce qui cloche ? (découpe, cible, extraction…)"
            disabled={saving}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                void submit();
              } else if (e.key === 'Escape') {
                setOpen(false);
              }
            }}
          />
          <p className="flag__hint">
            Entrée pour {flagged ? 'mettre à jour' : 'envoyer'}
            {flagged ? ' · videz le champ pour retirer' : ''} · Échap pour fermer
          </p>
        </div>
      )}
      {saved && !open && <span className="flag__saved">enregistré ✓</span>}
    </div>
  );
}

/** One render slice: plain text, or text covered by `claim` (+ in a `target`). */
interface Seg {
  text: string;
  claim: AvisClaim | null;
  target: boolean;
}

/**
 * Stance presentation — same transparency as the verbatim highlight: we SHOW the
 * SENTIMENT toward the cleavage object (positif / négatif / neutre) so it can be audited
 * against the text. ↑ vert = positif, ↓ rouge = négatif, ~ gris = neutre.
 * (Les clés internes restent favorable/defavorable/nuance = le sens du sentiment.)
 */
const STANCE_META: Record<string, { glyph: string; color: string; label: string }> = {
  favorable: { glyph: '↑', color: '#1a7f37', label: 'positif' },
  defavorable: { glyph: '↓', color: '#c1121f', label: 'négatif' },
  nuance: { glyph: '~', color: '#6b7280', label: 'neutre' },
};

/**
 * Confidence presentation — the model's OWN self-assessment of its stance call, shown
 * HONESTLY as such (a discreet 3-dot meter, never as a ground truth). high ●●● /
 * medium ●●○ / low ●○○. The tooltip spells out that it is an auto-évaluation du modèle.
 */
const CONFIDENCE_META: Record<string, { dots: string; label: string }> = {
  high: { dots: '●●●', label: 'élevée' },
  medium: { dots: '●●○', label: 'moyenne' },
  low: { dots: '●○○', label: 'faible' },
};
const CONFIDENCE_RANK: Record<string, number> = { low: 0, medium: 1, high: 2 };

/** Aggregate confidence over claims — the MIN (most honest: a tally is no surer than its
 *  weakest classified claim). Returns undefined when no claim carries a confidence. */
function aggConfidence(claims: AvisClaim[]): 'high' | 'medium' | 'low' | undefined {
  let min: 'high' | 'medium' | 'low' | undefined;
  for (const c of claims) {
    const lvl = c.stance_confidence;
    if (!lvl || !(lvl in CONFIDENCE_RANK)) continue;
    if (min === undefined || CONFIDENCE_RANK[lvl] < CONFIDENCE_RANK[min]) min = lvl;
  }
  return min;
}

/** Tooltip for a highlighted claim — theme, cible flag, and (if known) its stance. */
function claimTitle(claim: AvisClaim, target: boolean): string {
  let t = `Thème : ${claim.theme_title}${target ? ' · cible' : ''}`;
  const meta = claim.stance ? STANCE_META[claim.stance] : undefined;
  if (meta) {
    t += `\nAvis : ${meta.label}`;
    if (claim.proposition) t += ` envers « ${claim.proposition} »`;
    if (claim.stance_justif) t += `\n« ${claim.stance_justif} »`;
  }
  return t;
}

interface ClusterRef {
  key: string;
  label: string;
  color: string;
}

/** Distinct clusters present in this avis (for the legend), in first-seen order. */
function clustersOf(claims: AvisClaim[]): ClusterRef[] {
  const seen = new Map<string, ClusterRef>();
  for (const c of claims) {
    const key = c.cluster_id ?? c.theme_title;
    if (!key || seen.has(key)) continue;
    seen.set(key, { key, label: c.theme_title, color: c.color });
  }
  return [...seen.values()];
}

/** Stance tallies for a set of claims, in fixed order (favorable, défavorable, nuancé),
 *  skipping stances with no claim. Empty when none of the claims carry a stance. */
function stanceCounts(
  claims: AvisClaim[],
): { key: string; glyph: string; color: string; label: string; count: number; confidence?: 'high' | 'medium' | 'low' }[] {
  const byStance = new Map<string, AvisClaim[]>();
  for (const c of claims) {
    if (!c.stance) continue;
    (byStance.get(c.stance) ?? byStance.set(c.stance, []).get(c.stance)!).push(c);
  }
  return ['favorable', 'defavorable', 'nuance']
    .filter((k) => byStance.has(k))
    .map((k) => ({
      key: k,
      ...STANCE_META[k],
      count: byStance.get(k)!.length,
      confidence: aggConfidence(byStance.get(k)!),
    }));
}

/**
 * Légende d'ANALYSE d'un avis — révélée au clic (état porté par la carte). Une « fiche »
 * extensible : une ligne par cluster présent dans l'avis (pastille + nom), puis une liste
 * de FACTEURS d'analyse. La STANCE est le 1er facteur (répartition favorable/défavorable/
 * nuancé des claims de CET avis dans ce cluster) ; d'autres facteurs viendront s'ajouter.
 * Gracieux : si aucun claim n'a de stance (datasets non bakés), on montre juste les clusters.
 */
export function AvisAnalysis({ claims }: { claims: AvisClaim[] }) {
  const clusters = clustersOf(claims);
  if (clusters.length === 0) {
    return <p className="avisx__analysisempty">Aucun thème extrait pour cet avis.</p>;
  }
  return (
    <div className="avisx__analysis">
      {clusters.map((c) => {
        const own = claims.filter((cl) => (cl.cluster_id ?? cl.theme_title) === c.key);
        const stance = stanceCounts(own);
        return (
          <div key={c.key} className="avisx__analysisrow">
            <div className="avisx__analysiscluster">
              <span className="avisdetail__chip" style={{ background: c.color }} aria-hidden />
              <span className="avisx__analysisname">{c.label}</span>
              {/* Stance EN LIGNE, à la suite du nom (pas de ligne « Opinion » séparée). */}
              {stance.map((s) => {
                const conf = s.confidence ? CONFIDENCE_META[s.confidence] : undefined;
                return (
                  <span key={s.key} className="avisx__stancetag" style={{ color: s.color }}>
                    {s.glyph} {s.count > 1 ? `${s.count} ` : ''}{s.label}
                    {conf && (
                      <span
                        className={`avisx__stanceconf avisx__stanceconf--${s.confidence}`}
                        title={`Confiance du modèle : ${conf.label} — auto-évaluation, pas une vérité`}
                        aria-label={`confiance ${conf.label}`}
                      >
                        {conf.dots}
                      </span>
                    )}
                  </span>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Carte DASHBOARD des statistiques d'UN claim (clic sur un passage surligné) — même
 * langage visuel que les cartes de la synthèse : volume du cluster, sentiment agrégé,
 * interprétation du passage par le modèle (stance + justif + confiance), et
 * REPRÉSENTATIVITÉ (proximité au centroïde du cluster, via /citations).
 */
export interface ClaimStatsData {
  /** Feuille du claim (titre + volumes), depuis la carte des thèmes de /analysis. */
  leafTitle?: string | null;
  nAvis?: number | null;
  nClaims?: number | null;
  /** Répartition d'opinion de la feuille (depuis /opinion), si bakée. */
  opinion?: { fav: number; def: number; nuance: number; proposition?: string } | null;
  /** Représentativité : rang du claim par proximité au centroïde (depuis /citations). */
  citation?: { rank: number; total: number; dist: number } | null;
  /** Chargement en cours (citations fetchées à la demande). */
  loading?: boolean;
}

export function ClaimStatsCard({
  claim,
  stats,
  onClose,
}: {
  claim: AvisClaim;
  stats: ClaimStatsData;
  onClose: () => void;
}) {
  const meta = claim.stance ? STANCE_META[claim.stance] : undefined;
  const conf = claim.stance_confidence ? CONFIDENCE_META[claim.stance_confidence] : undefined;
  const op = stats.opinion;
  const pol = op ? op.fav + op.def : 0;
  const pctPos = pol > 0 ? Math.round((100 * (op?.fav ?? 0)) / pol) : null;
  const cit = stats.citation;
  // Représentativité : rang 0 = le plus proche du centroïde → percentile haut = très central.
  const reprPct = cit && cit.total > 1 ? Math.round(100 * (1 - cit.rank / (cit.total - 1))) : null;
  return (
    <div className="claimstats" role="region" aria-label="Statistiques du passage sélectionné">
      <div className="claimstats__head">
        <span className="avisdetail__chip" style={{ background: claim.color }} aria-hidden />
        <span className="claimstats__title">{stats.leafTitle || claim.theme_title}</span>
        <button type="button" className="claimstats__close" onClick={onClose} aria-label="Fermer">
          ✕
        </button>
      </div>
      <div className="claimstats__grid">
        <div className="claimstats__cell">
          <strong>{stats.nAvis != null ? stats.nAvis.toLocaleString('fr-FR') : '—'}</strong>
          <span>témoignages dans ce cluster</span>
          {stats.nClaims != null && <em>{stats.nClaims.toLocaleString('fr-FR')} idées</em>}
        </div>
        <div className="claimstats__cell">
          {pctPos != null ? (
            <>
              <strong>
                <span className="claimstats__pos">{pctPos}%</span>
                {' / '}
                <span className="claimstats__neg">{100 - pctPos}%</span>
              </strong>
              <span>sentiment du cluster (positif / négatif)</span>
              {op?.proposition && <em>envers « {op.proposition} »</em>}
            </>
          ) : (
            <>
              <strong>—</strong>
              <span>sentiment non mesuré (signal diffus)</span>
            </>
          )}
        </div>
        <div className="claimstats__cell">
          {meta ? (
            <>
              <strong style={{ color: meta.color }}>
                {meta.glyph} {meta.label}
                {conf && (
                  <span className="claimstats__conf" title={`Confiance du modèle : ${conf.label}`}>
                    {' '}{conf.dots}
                  </span>
                )}
              </strong>
              <span>lecture de CE passage par le modèle</span>
              {claim.stance_justif && <em>« {claim.stance_justif} »</em>}
            </>
          ) : (
            <>
              <strong>~</strong>
              <span>passage non classé (thème au signal diffus)</span>
            </>
          )}
        </div>
        <div className="claimstats__cell">
          {stats.loading ? (
            <>
              <strong>…</strong>
              <span>représentativité (calcul)</span>
            </>
          ) : reprPct != null && cit ? (
            <>
              <strong>{reprPct}%</strong>
              <span>représentativité du cluster</span>
              <em>
                {cit.rank + 1}ᵉ / {cit.total} par proximité au centroïde (dist {cit.dist.toFixed(2)})
              </em>
            </>
          ) : (
            <>
              <strong>—</strong>
              <span>représentativité indisponible</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/** Does range `r` cover the whole interval `[a, b)` ? */
function covers(r: CharRange, a: number, b: number): boolean {
  return r.start <= a && r.end >= b;
}

/**
 * Slice `text` into a clean, NON-overlapping cover of plain / highlighted
 * segments. We cut at every span/target boundary, then for each minimal interval
 * pick the first claim whose any span covers it (deterministic on overlap) and
 * flag it underlined when any covering claim's target covers it. Char-segment
 * rendering means overlaps never double-wrap a node.
 */
function segments(text: string, claims: AvisClaim[]): Seg[] {
  const len = text.length;
  // Boundary set: every place where coverage can change.
  const bounds = new Set<number>([0, len]);
  for (const c of claims) {
    for (const s of c.spans) {
      bounds.add(clamp(s.start, len));
      bounds.add(clamp(s.end, len));
    }
    if (c.target) {
      bounds.add(clamp(c.target.start, len));
      bounds.add(clamp(c.target.end, len));
    }
  }
  const cuts = [...bounds].sort((a, b) => a - b);

  const out: Seg[] = [];
  for (let i = 0; i < cuts.length - 1; i++) {
    const a = cuts[i];
    const b = cuts[i + 1];
    if (b <= a) continue;
    // First claim (document order) whose any span covers this interval owns the colour.
    const claim = claims.find((c) => c.spans.some((s) => covers(s, a, b))) ?? null;
    // Underline if ANY covering claim's target covers it.
    const target =
      claim != null &&
      claims.some(
        (c) => c.target != null && covers(c.target, a, b) && c.spans.some((s) => covers(s, a, b)),
      );
    const seg: Seg = { text: text.slice(a, b), claim, target };
    // Merge with the previous segment when nothing changed (fewer DOM nodes).
    const prev = out[out.length - 1];
    if (prev && prev.claim === claim && prev.target === target) prev.text += seg.text;
    else out.push(seg);
  }
  return out;
}

function clamp(n: number, len: number): number {
  return Math.max(0, Math.min(len, n));
}

/** Translucent fill from a #rrggbb (or hsl) cluster colour — keeps text readable. */
function tint(color: string): string {
  if (/^#[0-9a-f]{6}$/i.test(color)) return `${color}2e`;
  const m = color.match(/^hsl\(([^)]+)\)$/i);
  if (m) return `hsla(${m[1]} / 0.22)`;
  return 'rgba(0,0,145,0.18)';
}
