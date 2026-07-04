import { useMemo, useState } from 'react';
import type { SpatialTheme } from './contract';
import { themeCaption } from './labels';
import { LOCALE } from './strings';

/**
 * F3 (v2) — the theme BUBBLES map is replaced by a « camembert » (donut chart) of
 * the CURRENT drill level: one slice per visible theme, AREA = share of `n_avis`
 * (voices) at this level, COLOUR = `t.color` (same backend palette as before, so
 * highlights stay coherent across the app). The centre hole carries the total
 * voice count for the level — always visible, whatever the number of slices.
 *
 * Navigation is UNCHANGED from the old map: click a drillable slice (`has_children`)
 * to descend into its sub-clusters (same donut, next level) ; click a leaf slice to
 * select it → its citations open in the right panel. A legend list mirrors the
 * slices with the exact NUMBER and PERCENTAGE next to each name (the donut alone
 * can't be read precisely for thin slices), and is itself clickable/hoverable in
 * sync with the chart.
 */
const SIZE = 320;
const CX = SIZE / 2;
const CY = SIZE / 2;
const R_OUTER = 150;
const R_INNER = 88;
const EXPLODE = 7; // px the hovered/selected slice is pushed outward.

interface Slice {
  theme: SpatialTheme;
  start: number;
  end: number;
  pct: number;
}

export function PieChart({
  themes,
  currentParentId,
  selectedId,
  onSelect,
  onDrill,
}: {
  themes: SpatialTheme[];
  currentParentId: string | null;
  selectedId: string | null;
  onSelect: (t: SpatialTheme) => void;
  onDrill: (t: SpatialTheme) => void;
}) {
  const [hoverId, setHoverId] = useState<string | null>(null);

  const visible = useMemo(
    () => themes.filter((t) => t.parent_id === currentParentId).sort((a, b) => b.n_avis - a.n_avis),
    [themes, currentParentId],
  );

  const total = useMemo(
    () => visible.reduce((s, t) => s + Math.max(0, t.n_avis), 0),
    [visible],
  );

  const slices = useMemo<Slice[]>(() => {
    const denom = total || 1;
    let angle = 0;
    return visible.map((t) => {
      const frac = Math.max(0, t.n_avis) / denom;
      const start = angle;
      const end = angle + frac * Math.PI * 2;
      angle = end;
      return { theme: t, start, end, pct: frac };
    });
  }, [visible, total]);

  function activate(t: SpatialTheme) {
    if (t.has_children) onDrill(t);
    else onSelect(t);
  }

  if (!visible.length) {
    return <p className="piechart__empty">Aucun cluster à ce niveau.</p>;
  }

  return (
    <div className="piechart">
      <div className="piechart__chartwrap">
        <svg
          className="piechart__svg"
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          role="img"
          aria-label="Répartition des clusters de la consultation"
        >
          {slices.length === 1 ? (
            <g
              className={`piechart__slice${slices[0].theme.id === (hoverId ?? selectedId) ? ' piechart__slice--active' : ''}`}
              onClick={() => activate(slices[0].theme)}
              onMouseEnter={() => setHoverId(slices[0].theme.id)}
              onMouseLeave={() => setHoverId(null)}
              role="button"
              tabIndex={0}
              aria-label={sliceLabel(slices[0])}
              onKeyDown={(e) => keyActivate(e, () => activate(slices[0].theme))}
            >
              <circle cx={CX} cy={CY} r={R_OUTER} fill={slices[0].theme.color || FALLBACK} />
              <circle cx={CX} cy={CY} r={R_INNER} className="piechart__hole" />
            </g>
          ) : (
            slices.map((s) => {
              const active = hoverId === s.theme.id || selectedId === s.theme.id;
              const mid = (s.start + s.end) / 2;
              const dx = active ? Math.sin(mid) * EXPLODE : 0;
              const dy = active ? -Math.cos(mid) * EXPLODE : 0;
              return (
                <path
                  key={s.theme.id}
                  className={`piechart__slice${active ? ' piechart__slice--active' : ''}`}
                  d={arcPath(CX + dx, CY + dy, R_OUTER, R_INNER, s.start, s.end)}
                  fill={s.theme.color || FALLBACK}
                  onClick={() => activate(s.theme)}
                  onMouseEnter={() => setHoverId(s.theme.id)}
                  onMouseLeave={() => setHoverId((h) => (h === s.theme.id ? null : h))}
                  role="button"
                  tabIndex={0}
                  aria-label={sliceLabel(s)}
                  onKeyDown={(e) => keyActivate(e, () => activate(s.theme))}
                >
                  <title>{sliceLabel(s)}</title>
                </path>
              );
            })
          )}
          <text x={CX} y={CY - 10} textAnchor="middle" className="piechart__totalvalue">
            {total.toLocaleString(LOCALE)}
          </text>
          <text x={CX} y={CY + 12} textAnchor="middle" className="piechart__totallabel">
            voix
          </text>
          <text x={CX} y={CY + 30} textAnchor="middle" className="piechart__totalsub">
            {visible.length} cluster{visible.length > 1 ? 's' : ''}
          </text>
        </svg>
      </div>

      <ul className="piechart__legend" aria-label="Détail des clusters">
        {slices.map((s) => {
          const t = s.theme;
          const active = hoverId === t.id || selectedId === t.id;
          return (
            <li
              key={t.id}
              className={`piechart__legenditem${active ? ' piechart__legenditem--active' : ''}`}
            >
              <button
                type="button"
                className="piechart__legendbtn"
                onClick={() => activate(t)}
                onMouseEnter={() => setHoverId(t.id)}
                onMouseLeave={() => setHoverId((h) => (h === t.id ? null : h))}
                title={t.has_children ? 'cliquer pour explorer les sous-clusters' : 'cliquer pour voir les témoignages'}
              >
                <i className="piechart__legenddot" style={{ background: t.color || FALLBACK }} aria-hidden />
                <span className="piechart__legendlabel">{themeCaption(t)}</span>
                <span className="piechart__legendstats">
                  {t.n_avis.toLocaleString(LOCALE)} · {formatPct(s.pct)}
                </span>
                <span className="piechart__legendarrow" aria-hidden>
                  {t.has_children ? '›' : '＋'}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

const FALLBACK = '#9a9ab8';

function sliceLabel(s: Slice): string {
  const t = s.theme;
  const action = t.has_children ? 'cliquer pour explorer les sous-clusters' : 'cliquer pour voir les témoignages';
  return `${themeCaption(t)} — ${t.n_avis.toLocaleString(LOCALE)} avis (${formatPct(s.pct)}), ${action}`;
}

function formatPct(frac: number): string {
  const pct = frac * 100;
  return `${pct < 10 ? pct.toFixed(1) : Math.round(pct)} %`;
}

function keyActivate(e: React.KeyboardEvent, fn: () => void) {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fn();
  }
}

/** Donut slice path: outer arc, radial edge in, inner arc back, radial edge out. */
function arcPath(
  cx: number,
  cy: number,
  rOuter: number,
  rInner: number,
  startAngle: number,
  endAngle: number,
): string {
  const large = endAngle - startAngle > Math.PI ? 1 : 0;
  const pt = (r: number, a: number): [number, number] => [cx + r * Math.sin(a), cy - r * Math.cos(a)];
  const [x1, y1] = pt(rOuter, startAngle);
  const [x2, y2] = pt(rOuter, endAngle);
  const [x3, y3] = pt(rInner, endAngle);
  const [x4, y4] = pt(rInner, startAngle);
  return (
    `M ${x1} ${y1} A ${rOuter} ${rOuter} 0 ${large} 1 ${x2} ${y2} ` +
    `L ${x3} ${y3} A ${rInner} ${rInner} 0 ${large} 0 ${x4} ${y4} Z`
  );
}
