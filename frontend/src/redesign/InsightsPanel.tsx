import { useEffect, useRef, useState } from 'react';
import { Markdown } from './Markdown';
import { ThemeNavigator } from './ThemeNavigator';
import { deleteThemeFlag, upsertThemeFlag } from './analysisApi';
import type { DataSource, SpatialTheme } from './contract';

/** Categories Bob can pin on a bad theme synthesis. */
const THEME_CATEGORIES = ['Hallucination', 'Mauvais résumé', 'Erreur de clustering'] as const;

/** Current server-side flag on the open theme synthesis (absent = not flagged). */
export interface ThemeFlagState {
  category?: string | null;
  text?: string | null;
}

/** Everything the « Signaler » button needs to upsert/remove a theme-synthesis flag. */
export interface ThemeFlagTarget {
  dataset: string;
  themeId: string;
  layer: number | null;
  flag?: ThemeFlagState;
  onChange?: (themeId: string, flag: ThemeFlagState | null) => void;
}

/**
 * F4 — right column. Renders the LLM Markdown synthesis for the CURRENT zoom
 * level (global vs selected theme). Shows a spinner during generation. At a leaf
 * the parent swaps this for the citations panel (no LLM there).
 *
 * When `flagTarget` is set (a real theme, not the global level), a « Signaler »
 * button sits in the header — same idea as the avis flag, plus a category select.
 */
export function InsightsPanel({
  title,
  markdown,
  loading,
  source,
  flagTarget,
  keywords,
  themes,
  themesTotal,
  onSelectTheme,
}: {
  title: string;
  markdown: string | null;
  loading: boolean;
  source: DataSource | null;
  flagTarget?: ThemeFlagTarget;
  keywords?: string[];
  /** Arbre COMPLET des thèmes du payload (navigateur accordéon). */
  themes?: SpatialTheme[];
  /** Voix du niveau racine (dénominateur des thèmes racine). */
  themesTotal?: number;
  /** Optionnel : drill la vue d'analyse sur le thème cliqué dans le navigateur. */
  onSelectTheme?: (themeId: string) => void;
}) {
  return (
    <section className="panel insights">
      <header className="panel__head">
        <h2 title={title}>{title}</h2>
        <div className="panel__head-right">
          {flagTarget && <ThemeFlagControl key={flagTarget.themeId} {...flagTarget} />}
        </div>
      </header>
      {keywords && keywords.length > 0 && (
        <div className="kw-chips" aria-label="Mots-clés représentatifs">
          {keywords.map((kw) => (
            <span key={kw} className="kw-chip">{kw}</span>
          ))}
        </div>
      )}
      {loading ? (
        <div className="insights__loading">
          <span className="spinner" /> génération de la synthèse…
        </div>
      ) : markdown ? (
        <Markdown source={markdown} />
      ) : source === 'building' ? (
        <div className="insights__loading">
          <span className="spinner" /> Analyse en cours…
        </div>
      ) : source === 'error' ? (
        <p className="panel__empty">Synthèse indisponible (backend).</p>
      ) : (
        <p className="panel__empty">Aucune synthèse pour ce niveau.</p>
      )}
      {themes && themes.length > 0 && (
        <>
          <h3 className="synthesis__subhead">Points de convergence</h3>
          <ThemeNavigator themes={themes} total={themesTotal ?? 0} onSelect={onSelectTheme} />
        </>
      )}
    </section>
  );
}

/**
 * « Signaler » on a theme SYNTHESIS → category (Hallucination | Mauvais résumé |
 * Erreur de clustering) + free-text, persisted to the server (upsert by theme).
 * The button is MARKED when the theme already carries a flag (state fed from the
 * dataset-wide `/flags`). Re-openable to edit; clearing the text removes it.
 */
function ThemeFlagControl({ dataset, themeId, layer, flag, onChange }: ThemeFlagTarget) {
  const flagged = typeof flag?.text === 'string' && flag.text.trim().length > 0;
  const [open, setOpen] = useState(false);
  const [category, setCategory] = useState<string>(flag?.category || THEME_CATEGORIES[0]);
  const [text, setText] = useState(flag?.text ?? '');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Keep the fields in sync when the open theme (and thus its flag) changes underneath.
  useEffect(() => {
    setCategory(flag?.category || THEME_CATEGORIES[0]);
    setText(flag?.text ?? '');
    setOpen(false);
    setSaved(false);
  }, [themeId, flag?.category, flag?.text]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  async function submit() {
    const value = text.trim();
    setSaving(true);
    try {
      if (!value) {
        // Empty text on an existing flag → remove it; otherwise no-op.
        if (flagged) await deleteThemeFlag(dataset, themeId);
        onChange?.(themeId, null);
      } else {
        const cat = category || THEME_CATEGORIES[0];
        await upsertThemeFlag(dataset, themeId, layer, cat, value);
        onChange?.(themeId, { category: cat, text: value });
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
        title={
          flagged
            ? `Signalé (${flag?.category ?? '—'}) : ${flag?.text}`
            : 'Signaler une synthèse à corriger'
        }
        onClick={() => {
          setSaved(false);
          setOpen((o) => !o);
        }}
      >
        <span aria-hidden>⚑</span> {flagged ? 'Signalé' : 'Signaler'}
      </button>
      {open && (
        <div className="flag__panel">
          <select
            className="flag__select"
            value={category}
            disabled={saving}
            onChange={(e) => setCategory(e.target.value)}
          >
            {THEME_CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <input
            ref={inputRef}
            className="flag__input"
            type="text"
            value={text}
            placeholder="Qu'est-ce qui cloche dans cette synthèse ?"
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
