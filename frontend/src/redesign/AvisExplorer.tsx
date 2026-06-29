import { useEffect, useMemo, useRef, useState } from 'react';
import type { AnalysisPayload, AvisListItem, AvisProvenance, Consultation, SpatialTheme } from './contract';
import { fetchAnalysis, fetchAvis, fetchAvisList } from './analysisApi';
import { Header } from './Header';
import { AvisDetail } from './AvisDetail';
import { LOCALE } from './strings';

const PAGE = 30; // taille d'une page « Voir plus »

/**
 * Page d'EXPLORATION DES AVIS : recense TOUS les avis d'une consultation, avec
 * recherche plein-texte (debounce), filtre par cluster (macros/thèmes de `/analysis`)
 * et pagination « Voir plus ». Cliquer un avis ouvre son détail complet (`AvisDetail`,
 * surlignages verbatim) ; `focusAvisId` ouvre directement ce détail au chargement
 * (entrée depuis une citation de la synthèse), la liste restant accessible au retour.
 */
export function AvisExplorer({
  dataset,
  focusAvisId,
  onHome,
}: {
  dataset: Consultation;
  /** Avis à ouvrir directement au chargement (deep-link `&focus=`), sinon la liste. */
  focusAvisId?: string | null;
  onHome: () => void;
}) {
  // Filtres : saisie immédiate `qInput` → `q` debouncé (300 ms) ; thème sélectionné.
  const [qInput, setQInput] = useState('');
  const [q, setQ] = useState('');
  const [themeId, setThemeId] = useState<string | null>(null);

  const [themes, setThemes] = useState<SpatialTheme[]>([]);
  const [items, setItems] = useState<AvisListItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);

  // Avis ouvert (détail). Initialisé sur le focus deep-link.
  const [openAvisId, setOpenAvisId] = useState<string | null>(focusAvisId ?? null);
  const [avis, setAvis] = useState<AvisProvenance | null>(null);
  const [avisLoading, setAvisLoading] = useState(false);

  // Debounce de la recherche : `qInput` → `q` après 300 ms d'inactivité.
  useEffect(() => {
    const t = setTimeout(() => setQ(qInput), 300);
    return () => clearTimeout(t);
  }, [qInput]);

  // Hiérarchie de thèmes pour le filtre cluster (chargée une fois par dataset).
  useEffect(() => {
    let cancelled = false;
    fetchAnalysis(dataset.id)
      .catch(() => null)
      .then((a) => {
        if (!cancelled) setThemes((a?.data as AnalysisPayload | null)?.themes ?? []);
      });
    return () => {
      cancelled = true;
    };
  }, [dataset.id]);

  // Page 0 : (re)charge dès que dataset / filtre cluster / recherche change. Remplace.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchAvisList(dataset.id, { themeId, q, limit: PAGE, offset: 0 })
      .then(({ data }) => {
        if (cancelled) return;
        setItems(data?.items ?? []);
        setTotal(data?.total ?? 0);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [dataset.id, themeId, q]);

  // Avis ouvert → charge son détail complet (surlignages verbatim).
  useEffect(() => {
    if (!openAvisId) {
      setAvis(null);
      return;
    }
    let cancelled = false;
    setAvisLoading(true);
    setAvis(null);
    fetchAvis(dataset.id, openAvisId)
      .then(({ data }) => !cancelled && setAvis(data))
      .finally(() => !cancelled && setAvisLoading(false));
    return () => {
      cancelled = true;
    };
  }, [dataset.id, openAvisId]);

  function loadMore() {
    setLoadingMore(true);
    fetchAvisList(dataset.id, { themeId, q, limit: PAGE, offset: items.length })
      .then(({ data }) => {
        if (data) {
          setItems((prev) => [...prev, ...data.items]);
          setTotal(data.total);
        }
      })
      .finally(() => setLoadingMore(false));
  }

  // Thèmes ordonnés hiérarchiquement (parent puis enfants) avec profondeur → <option> indentées.
  const themeOptions = useMemo(() => orderThemes(themes), [themes]);
  const hasMore = items.length < total;

  return (
    <div className="agora overview">
      <Header
        onHome={onHome}
        right={<span className="overview__crumb">{dataset.label}</span>}
      />

      <main className="overview__body avisx">
        {openAvisId ? (
          <AvisDetail
            avis={avis}
            loading={avisLoading}
            dataset={dataset.id}
            backLabel="← retour à la liste"
            onBack={() => setOpenAvisId(null)}
          />
        ) : (
          <>
            <section className="overview__head">
              <h1 className="overview__title">Explorer les avis</h1>
              <p className="overview__context">
                Tous les avis de la consultation : recherche plein-texte, filtre par
                thème, et accès au texte complet de chaque contribution.
              </p>
            </section>

            <div className="avisx__controls">
              <input
                className="avisx__search"
                type="search"
                value={qInput}
                placeholder="Rechercher dans les avis…"
                aria-label="Rechercher dans les avis"
                onChange={(e) => setQInput(e.target.value)}
              />
              <select
                className="avisx__filter"
                value={themeId ?? ''}
                aria-label="Filtrer par thème"
                onChange={(e) => setThemeId(e.target.value || null)}
              >
                <option value="">Tous les thèmes</option>
                {themeOptions.map(({ theme, depth }) => (
                  <option key={theme.id} value={theme.id}>
                    {'  '.repeat(depth)}
                    {theme.title || theme.label}
                  </option>
                ))}
              </select>
            </div>

            <p className="avisx__count" aria-live="polite">
              {loading
                ? 'Chargement…'
                : `${total.toLocaleString(LOCALE)} avis${q || themeId ? ' (filtrés)' : ''}`}
            </p>

            {!loading && items.length === 0 ? (
              <p className="overview__loading">Aucun avis ne correspond.</p>
            ) : (
              <ul className="avisx__list">
                {items.map((it) => (
                  <li
                    key={it.avis_id}
                    className="avisx__item"
                    role="button"
                    tabIndex={0}
                    onClick={() => setOpenAvisId(it.avis_id)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        setOpenAvisId(it.avis_id);
                      }
                    }}
                  >
                    <p className="avisx__excerpt">{it.excerpt}</p>
                    {it.themes.length > 0 && (
                      <div className="avisx__chips">
                        {it.themes.map((th) => (
                          <span
                            key={th.id}
                            className="avisx__chip"
                            style={{ borderColor: th.color, color: th.color }}
                          >
                            {th.title}
                          </span>
                        ))}
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            )}

            {hasMore && (
              <button
                type="button"
                className="btn-secondary avisx__more"
                disabled={loadingMore}
                onClick={loadMore}
              >
                {loadingMore ? 'Chargement…' : `Voir plus (${items.length}/${total})`}
              </button>
            )}
          </>
        )}
      </main>
    </div>
  );
}

/** DFS hiérarchique : chaque thème suivi de ses enfants, avec profondeur (indentation). */
function orderThemes(themes: SpatialTheme[]): { theme: SpatialTheme; depth: number }[] {
  const children = new Map<string | null, SpatialTheme[]>();
  for (const t of themes) {
    const arr = children.get(t.parent_id) ?? [];
    arr.push(t);
    children.set(t.parent_id, arr);
  }
  const out: { theme: SpatialTheme; depth: number }[] = [];
  const walk = (parent: string | null, depth: number) => {
    for (const t of children.get(parent) ?? []) {
      out.push({ theme: t, depth });
      walk(t.id, depth + 1);
    }
  };
  walk(null, 0);
  return out;
}
