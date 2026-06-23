import type { AvisClaim, AvisProvenance, CharRange } from './contract';

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
  onBack,
}: {
  avis: AvisProvenance | null;
  loading: boolean;
  onBack: () => void;
}) {
  return (
    <section className="panel avisdetail">
      <button className="link-back" onClick={onBack}>
        ← retour aux citations
      </button>
      {loading ? (
        <div className="insights__loading">
          <span className="spinner" /> chargement de l'avis…
        </div>
      ) : avis ? (
        <>
          <p className="avisdetail__meta">
            {avis.claims.length} claim{avis.claims.length > 1 ? 's' : ''} surligné
            {avis.claims.length > 1 ? 's' : ''} · cible soulignée · chaque couleur = un thème
          </p>
          {/* Cluster legend: the distinct themes highlighted in THIS avis, each in
              its cluster colour → every highlight below maps to a named cluster. */}
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
          <article className="avisdetail__text">
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
        </>
      ) : (
        <p className="panel__empty">Avis indisponible.</p>
      )}
    </section>
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
