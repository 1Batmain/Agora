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
}) {
  return (
    <section className="panel avisdetail">
      <div className="avisdetail__topbar">
        <button className="link-back" onClick={onBack}>
          ← retour aux citations
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
function AvisBody({ avis }: { avis: AvisProvenance }) {
  const isFr = !avis.lang || avis.lang.toLowerCase().startsWith('fr');
  const hasTranslation = !isFr && typeof avis.text_fr === 'string' && avis.text_fr.length > 0;
  const [showOriginal, setShowOriginal] = useState(false);

  // Reset the toggle whenever the open avis changes (default = French when translated).
  useEffect(() => setShowOriginal(false), [avis.id]);

  // French reading view is the default for a translated avis; the original carries highlights.
  const original = showOriginal || !hasTranslation;

  return (
    <>
      <p className="avisdetail__meta">
        {!isFr && (
          <span className="avisdetail__langbadge" title={`Langue d'origine : ${langName(avis.lang)}`}>
            {hasTranslation ? `traduit de l'${langName(avis.lang)}` : langName(avis.lang)}
          </span>
        )}
        {avis.claims.length} claim{avis.claims.length > 1 ? 's' : ''}
        {original ? (
          <>
            {' '}surligné{avis.claims.length > 1 ? 's' : ''} · cible soulignée · chaque couleur = un thème
          </>
        ) : (
          <> · couleurs = thèmes (surlignages sur l'original)</>
        )}
        {hasTranslation && (
          <button
            type="button"
            className="avisdetail__toggle"
            aria-pressed={showOriginal}
            onClick={() => setShowOriginal((o) => !o)}
          >
            {showOriginal ? '← voir la traduction' : "voir l'original →"}
          </button>
        )}
      </p>

      {/* Cluster legend: the distinct themes present in THIS avis, each in its cluster
          colour → ties the French reading view (no inline marks) back to the map too. */}
      {clustersOf(avis.claims).length > 0 && (
        <ul className="avisdetail__legend">
          {clustersOf(avis.claims).map((c) => (
            <li key={c.key} className="avisdetail__legenditem">
              <span className="avisdetail__chip" style={{ background: c.color }} aria-hidden />
              {c.label}
            </li>
          ))}
        </ul>
      )}

      {original ? (
        <article className="avisdetail__text" lang={avis.lang}>
          {segments(avis.text, avis.claims).map((seg, i) =>
            seg.claim ? (
              <mark
                key={i}
                className={`avisdetail__hl${seg.target ? ' avisdetail__hl--target' : ''}`}
                title={`Thème : ${seg.claim.theme_title}${seg.target ? ' · cible' : ''}`}
                style={{
                  backgroundColor: tint(seg.claim.color),
                  borderBottom: seg.target
                    ? `2px solid ${seg.claim.color}`
                    : `2px solid ${tint(seg.claim.color)}`,
                }}
              >
                {seg.text}
              </mark>
            ) : (
              <span key={i}>{seg.text}</span>
            ),
          )}
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
function FlagControl({
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
  return 'rgba(200,83,18,0.18)';
}
