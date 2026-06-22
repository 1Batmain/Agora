import type { AvisProvenance, AvisSpan } from './contract';

/**
 * Full avis text with its extractive portions HIGHLIGHTED in their cluster colour
 * (the same colour as the map bubbles). Because claims are extractive — exact
 * substrings of the avis (`pipeline.claims.span`) — every span maps to a real range
 * of the text: a faithful highlight with zero drift. Reading-only provenance view.
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
            {avis.spans.length} portion{avis.spans.length > 1 ? 's' : ''} extraite
            {avis.spans.length > 1 ? 's' : ''} · chaque surlignage est rattaché à son thème
          </p>
          {/* Cluster legend: the distinct themes highlighted in THIS avis, each in
              its cluster colour → every highlight below maps to a named cluster. */}
          {clustersOf(avis.spans).length > 0 && (
            <ul className="avisdetail__legend">
              {clustersOf(avis.spans).map((c) => (
                <li key={c.key} className="avisdetail__legenditem">
                  <span className="avisdetail__chip" style={{ background: c.color }} aria-hidden />
                  {c.label}
                </li>
              ))}
            </ul>
          )}
          <article className="avisdetail__text">
            {segments(avis.text, avis.spans).map((seg, i) =>
              seg.span ? (
                <mark
                  key={i}
                  className="avisdetail__hl"
                  title={`Thème : ${seg.span.theme_label}`}
                  style={{
                    backgroundColor: tint(seg.span.color),
                    borderBottom: `2px solid ${seg.span.color}`,
                  }}
                >
                  {seg.text}
                  <span className="avisdetail__hltag" style={{ color: seg.span.color }}>
                    {seg.span.theme_label}
                  </span>
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

interface Seg {
  text: string;
  span: AvisSpan | null;
}

interface ClusterRef {
  key: string;
  label: string;
  color: string;
}

/** Distinct clusters present in this avis (for the legend), in first-seen order. */
function clustersOf(spans: AvisSpan[]): ClusterRef[] {
  const seen = new Map<string, ClusterRef>();
  for (const s of spans) {
    const key = s.cluster_id ?? s.theme_label;
    if (!key || seen.has(key)) continue;
    seen.set(key, { key, label: s.theme_label, color: s.color });
  }
  return [...seen.values()];
}

/**
 * Slice `text` into highlighted / plain segments. Spans are sorted by start; on
 * overlap, the earlier-starting span wins and the later one is truncated (so the
 * render is always a clean, non-overlapping cover of the text).
 */
function segments(text: string, spans: AvisSpan[]): Seg[] {
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const out: Seg[] = [];
  let cursor = 0;
  for (const s of sorted) {
    const start = Math.max(s.start, cursor);
    if (start >= s.end) continue; // fully covered by an earlier span
    if (start > cursor) out.push({ text: text.slice(cursor, start), span: null });
    out.push({ text: text.slice(start, s.end), span: s });
    cursor = s.end;
  }
  if (cursor < text.length) out.push({ text: text.slice(cursor), span: null });
  return out;
}

/** Translucent fill from a #rrggbb cluster colour (keeps the text readable). */
function tint(hex: string): string {
  return /^#[0-9a-f]{6}$/i.test(hex) ? `${hex}2e` : 'rgba(200,83,18,0.18)';
}
