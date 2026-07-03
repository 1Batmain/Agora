import { forwardRef, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type {
  AnalysisPayload,
  AvisClaim,
  AvisListItem,
  AvisProvenance,
  Citation,
  Consultation,
  SpatialTheme,
  ThemeOpinion,
} from './contract';
import {
  fetchAnalysis,
  fetchAvis,
  fetchAvisList,
  fetchCitations,
  fetchFlags,
  fetchOpinion,
} from './analysisApi';
import { Header } from './Header';
import { AvisAnalysis, AvisBody, ClaimStatsCard, FlagControl } from './AvisDetail';
import type { ClaimStatsData } from './AvisDetail';
import { LOCALE } from './strings';

const PAGE = 15; // taille d'une page « Voir plus » (items lourds : avis entiers inline)

/**
 * Page d'EXPLORATION DES AVIS — AUTO-SUFFISANTE : recense TOUS les avis d'une
 * consultation et affiche CHACUN EN ENTIER, INLINE, sous forme de carte : son texte
 * complet avec ses **surlignages** verbatim (claims, réutilise `AvisBody`) RÉVÉLÉS AU
 * SURVOL de la carte (repos sobre), un toggle **FR / original** par avis (si traduit) et
 * un bouton **« Signaler »**. CLIC sur un passage surligné → carte de STATS du claim
 * (volume du cluster, sentiment, lecture du modèle, représentativité au centroïde).
 * Filtres : recherche plein-texte (debounce), CHIPS de grands thèmes + menu du sous-arbre
 * (fini le méga-menu), chips de SENTIMENT (visibles/désactivables) et pagination. PLUS de page séparée par avis : `focusAvisId`
 * (entrée depuis une citation) épingle la carte ciblée en tête, mise en évidence et
 * scrollée en vue.
 */
export function AvisExplorer({
  dataset,
  focusAvisId,
  focusThemeId,
  focusStance,
  onHome,
}: {
  dataset: Consultation;
  /** Avis à mettre en évidence au chargement (deep-link `&focus=`) : carte épinglée. */
  focusAvisId?: string | null;
  /** Thème sur lequel PRÉ-FILTRER l'explorateur (bouton « Consulter les témoignages du thème »). */
  focusThemeId?: string | null;
  /** Sentiment sur lequel PRÉ-FILTRER (clic carte positif/négatif de la synthèse). */
  focusStance?: 'favorable' | 'defavorable' | null;
  onHome: () => void;
}) {
  // Filtres : saisie immédiate `qInput` → `q` debouncé (300 ms) ; thème sélectionné
  // (initialisé sur le thème de focus quand on arrive depuis un sous-thème).
  const [qInput, setQInput] = useState('');
  const [q, setQ] = useState('');
  const [themeId, setThemeId] = useState<string | null>(focusThemeId ?? null);
  const [stance, setStance] = useState<'favorable' | 'defavorable' | null>(focusStance ?? null);
  // Tri CLIENT du lot déjà chargé (le backend ne trie pas — ordre stable de `avis.json`).
  const [sortBy, setSortBy] = useState<SortKey>('default');

  // Synchronise les filtres si le focus change (navigation depuis un autre thème / sentiment).
  useEffect(() => {
    setThemeId(focusThemeId ?? null);
  }, [focusThemeId]);
  useEffect(() => {
    setStance(focusStance ?? null);
  }, [focusStance]);

  const [themes, setThemes] = useState<SpatialTheme[]>([]);
  const [items, setItems] = useState<AvisListItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);

  // Répartition d'opinion par thème (chargée UNE fois) — alimente la carte de stats d'un claim.
  const [opinions, setOpinions] = useState<ThemeOpinion[]>([]);
  useEffect(() => {
    let cancelled = false;
    fetchOpinion(dataset.id)
      .then((op) => !cancelled && setOpinions(op ?? []))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [dataset.id]);

  // Lookups pour la carte de stats d'un claim (feuille → volumes / opinion).
  const themeById = useMemo(() => new Map(themes.map((t) => [t.id, t])), [themes]);
  const opinionByTheme = useMemo(
    () => new Map(opinions.map((o) => [o.theme_id, o])),
    [opinions],
  );

  // Citations par feuille (représentativité au centroïde) : fetch à la demande + cache.
  const citCache = useRef<Map<string, Citation[]>>(new Map());
  useEffect(() => {
    citCache.current = new Map();
  }, [dataset.id]);
  const getCitations = useCallback(
    async (leafId: string): Promise<Citation[]> => {
      const hit = citCache.current.get(leafId);
      if (hit) return hit;
      const res = await fetchCitations(dataset.id, leafId).catch(() => null);
      const list = (res?.data as Citation[] | null) ?? [];
      citCache.current.set(leafId, list);
      return list;
    },
    [dataset.id],
  );

  // Flags d'avis du dataset (avis_id → texte), chargés une fois → restauration à l'affichage.
  const [flags, setFlags] = useState<Record<string, string>>({});

  // Avis ciblé par une citation : chargé À PART (fetchAvis) et épinglé en tête, pour
  // qu'il soit toujours présent quelle que soit la pagination.
  const [focusAvis, setFocusAvis] = useState<AvisProvenance | null>(null);
  const focusRef = useRef<HTMLLIElement>(null);

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

  // Flags du dataset (clé = avis_id) → état « Signalé » de chaque carte au chargement.
  useEffect(() => {
    let cancelled = false;
    fetchFlags(dataset.id).then((list) => {
      if (cancelled) return;
      const map: Record<string, string> = {};
      for (const f of list) if (f.avis_id) map[f.avis_id] = f.text;
      setFlags(map);
    });
    return () => {
      cancelled = true;
    };
  }, [dataset.id]);

  // Page 0 : (re)charge dès que dataset / filtre cluster / recherche change. Remplace.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchAvisList(dataset.id, { themeId, q, stance, limit: PAGE, offset: 0 })
      .then(({ data }) => {
        if (cancelled) return;
        setItems(data?.items ?? []);
        setTotal(data?.total ?? 0);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [dataset.id, themeId, q, stance]);

  // Avis ciblé (citation) → chargé une fois pour l'épingler en tête.
  useEffect(() => {
    if (!focusAvisId) {
      setFocusAvis(null);
      return;
    }
    let cancelled = false;
    setFocusAvis(null);
    fetchAvis(dataset.id, focusAvisId).then(({ data }) => !cancelled && setFocusAvis(data));
    return () => {
      cancelled = true;
    };
  }, [dataset.id, focusAvisId]);

  // Scrolle la carte épinglée en vue dès qu'elle est chargée (mise en évidence).
  useEffect(() => {
    if (focusAvis) focusRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [focusAvis]);

  function loadMore() {
    setLoadingMore(true);
    fetchAvisList(dataset.id, { themeId, q, stance, limit: PAGE, offset: items.length })
      .then(({ data }) => {
        if (data) {
          setItems((prev) => [...prev, ...data.items]);
          setTotal(data.total);
        }
      })
      .finally(() => setLoadingMore(false));
  }

  const onFlagChange = (id: string, text: string | null) =>
    setFlags((prev) => {
      const next = { ...prev };
      if (text && text.trim()) next[id] = text;
      else delete next[id];
      return next;
    });

  const hasMore = items.length < total;
  // L'avis épinglé est retiré de la liste pour éviter un doublon avec la carte en tête.
  const listItems = focusAvis ? items.filter((it) => it.avis_id !== focusAvis.id) : items;
  // Tri appliqué au lot déjà CHARGÉ uniquement (pas d'endpoint de tri côté serveur) —
  // porte sur le nombre de passages retenus ou la longueur du témoignage.
  const sortedItems = useMemo(() => sortAvisItems(listItems, sortBy), [listItems, sortBy]);
  const hasActiveFilters = Boolean(qInput || themeId || stance);
  const resetFilters = useCallback(() => {
    setQInput('');
    setThemeId(null);
    setStance(null);
  }, []);

  return (
    <div className="agora overview">
      <Header onHome={onHome} right={<span className="overview__crumb">{dataset.label}</span>} />

      <main className="overview__body avisx">
        <section className="overview__head">
          <h1 className="overview__title">Explorer les avis</h1>
          <p className="overview__context">
            Tous les avis de la consultation, en entier : chaque contribution avec ses passages
            retenus surlignés, sa traduction si besoin, et un signalement possible.
          </p>
        </section>

        <div className="avisx__toolbar">
          {/* Ligne 1 : recherche plein-texte (large, prioritaire) + tri du lot chargé. */}
          <div className="avisx__searchrow">
            <div className="avisx__searchwrap">
              <span className="avisx__searchicon" aria-hidden>⌕</span>
              <input
                className="avisx__search"
                type="search"
                value={qInput}
                placeholder="Rechercher un mot, une expression…"
                aria-label="Rechercher dans les avis"
                onChange={(e) => setQInput(e.target.value)}
              />
              {qInput && (
                <button
                  type="button"
                  className="avisx__searchclear"
                  aria-label="Effacer la recherche"
                  onClick={() => setQInput('')}
                >
                  ×
                </button>
              )}
            </div>
            <div className="avisx__sortwrap">
              <span className="avisx__grouplabel">Trier</span>
              <SortDropdown value={sortBy} onChange={setSortBy} />
            </div>
          </div>

          {/* Ligne 2 : filtres groupés et étiquetés (thème / sentiment) + réinitialisation. */}
          <div className="avisx__filterrow">
            {themes.length > 0 && (
              <div className="avisx__filtergroup">
                <span className="avisx__grouplabel">Thème</span>
                <ThemeFilterDropdown themes={themes} value={themeId} onChange={setThemeId} />
              </div>
            )}

            {/* Filtre SENTIMENT — visible et désactivable (le filtre caché causait des pages vides). */}
            <div className="avisx__filtergroup">
              <span className="avisx__grouplabel">Sentiment</span>
              <div className="avisx__chips avisx__chips--stance" role="group" aria-label="Filtrer par sentiment">
                <button
                  type="button"
                  className={`avisx__chip${stance == null ? ' avisx__chip--on' : ''}`}
                  onClick={() => setStance(null)}
                >
                  Tous
                </button>
                <button
                  type="button"
                  className={`avisx__chip avisx__chip--pos${stance === 'favorable' ? ' avisx__chip--on' : ''}`}
                  onClick={() => setStance(stance === 'favorable' ? null : 'favorable')}
                >
                  ↑ Positifs
                </button>
                <button
                  type="button"
                  className={`avisx__chip avisx__chip--neg${stance === 'defavorable' ? ' avisx__chip--on' : ''}`}
                  onClick={() => setStance(stance === 'defavorable' ? null : 'defavorable')}
                >
                  ↓ Négatifs
                </button>
              </div>
            </div>

            {hasActiveFilters && (
              <button type="button" className="avisx__clearall" onClick={resetFilters}>
                ✕ Réinitialiser les filtres
              </button>
            )}
          </div>
        </div>

        <p className="avisx__count" aria-live="polite">
          {loading
            ? 'Chargement…'
            : `${total.toLocaleString(LOCALE)} avis${
                stance ? ` · sentiment ${stance === 'favorable' ? 'positif' : 'négatif'}` : ''
              }${q || themeId ? ' (filtrés)' : ''}`}
        </p>

        <ul className="avisx__list">
          {/* Carte épinglée : l'avis ciblé par une citation, mis en évidence + scrollé. */}
          {focusAvis && (
            <AvisCard
              ref={focusRef}
              avis={focusAvis}
              dataset={dataset.id}
              flagText={flags[focusAvis.id]}
              onFlagChange={onFlagChange}
              themeById={themeById}
              opinionByTheme={opinionByTheme}
              getCitations={getCitations}
              focused
            />
          )}

          {!loading &&
            sortedItems.map((it) => (
              <AvisCard
                key={it.avis_id}
                avis={toProvenance(it)}
                dataset={dataset.id}
                flagText={flags[it.avis_id]}
                onFlagChange={onFlagChange}
                themeById={themeById}
                opinionByTheme={opinionByTheme}
                getCitations={getCitations}
              />
            ))}
        </ul>

        {!loading && sortedItems.length === 0 && !focusAvis && (
          <div className="avisx__empty">
            <p>
              Aucun avis ne correspond à ces filtres.
              {stance && (
                <>
                  {' '}Le filtre <strong>sentiment {stance === 'favorable' ? 'positif' : 'négatif'}</strong>{' '}
                  est actif — certains thèmes n'ont pas de sentiment mesuré (signal diffus).
                </>
              )}
            </p>
            <button type="button" className="btn-secondary" onClick={resetFilters}>
              Réinitialiser les filtres
            </button>
          </div>
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
      </main>
    </div>
  );
}

/** Ouverture/fermeture d'un menu déroulant + fermeture au clic extérieur / Échap — logique
 * partagée par tous les dropdowns personnalisés de la barre d'outils. */
function useDropdown() {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);
  return { open, setOpen, rootRef };
}

/**
 * Sélecteur de thème en DROPDOWN personnalisé : liste hiérarchique COMPLÈTE (macros +
 * sous-thèmes indentés) dans un panneau flottant, libellés jamais tronqués — remplace
 * l'ancien duo chips + `<select>` natif (les chips coupaient les titres longs avec « … »
 * et le `<select>` du navigateur ne reprenait pas le style du reste de la barre d'outils).
 */
function ThemeFilterDropdown({
  themes,
  value,
  onChange,
}: {
  themes: SpatialTheme[];
  value: string | null;
  onChange: (id: string | null) => void;
}) {
  const { open, setOpen, rootRef } = useDropdown();
  const ordered = useMemo(() => orderThemes(themes), [themes]);
  const themeById = useMemo(() => new Map(themes.map((t) => [t.id, t])), [themes]);
  const selected = value ? themeById.get(value) : null;

  const pick = (id: string | null) => {
    onChange(id);
    setOpen(false);
  };

  return (
    <div className="avisx__dd" ref={rootRef}>
      <button
        type="button"
        className="avisx__dd-btn"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="avisx__dd-label">
          {selected ? selected.title || selected.label : 'Tous les thèmes'}
        </span>
        <span className="avisx__dd-caret" aria-hidden>
          {open ? '▴' : '▾'}
        </span>
      </button>
      {open && (
        <ul className="avisx__dd-panel" role="listbox" aria-label="Filtrer par thème">
          <li role="option" aria-selected={value == null}>
            <button
              type="button"
              className={`avisx__dd-item${value == null ? ' avisx__dd-item--on' : ''}`}
              onClick={() => pick(null)}
            >
              Tous les thèmes
            </button>
          </li>
          {/* Hiérarchie tracée (traits ├/└) plutôt que de simples espaces d'indentation :
              chaque grand thème est une SECTION (gras, séparée), ses sous-thèmes des
              branches rattachées. Une SEULE teinte (bleu) porte l'info — plus de couleur
              par cluster, c'est la position dans l'arbre qui distingue les niveaux. */}
          {ordered.map(({ theme, depth, isLast, guides }, i) => (
            <li key={theme.id} role="option" aria-selected={value === theme.id}>
              <button
                type="button"
                className={`avisx__dd-item${depth === 0 ? ' avisx__dd-item--macro' : ' avisx__dd-item--sub'}${
                  depth === 0 && i > 0 ? ' avisx__dd-item--section' : ''
                }${value === theme.id ? ' avisx__dd-item--on' : ''}`}
                onClick={() => pick(theme.id)}
              >
                {depth > 0 && (
                  <span className="avisx__dd-guides" aria-hidden>
                    {guides.map((on, gi) => (
                      <span key={gi} className={`avisx__dd-guide${on ? ' avisx__dd-guide--on' : ''}`} />
                    ))}
                    <span className="avisx__dd-branch">{isLast ? '└' : '├'}</span>
                  </span>
                )}
                <span className="avisx__dd-dot" aria-hidden />
                <span className="avisx__dd-itemlabel">{theme.title || theme.label}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/**
 * Sélecteur de TRI en dropdown personnalisé (même composant visuel que le filtre de
 * thème) — remplace le `<select>` natif qui ne reprenait pas le style du reste de la
 * barre d'outils (bordures/rayons/police différents du navigateur à l'autre).
 */
function SortDropdown({ value, onChange }: { value: SortKey; onChange: (v: SortKey) => void }) {
  const { open, setOpen, rootRef } = useDropdown();
  const selected = SORT_OPTIONS.find((o) => o.value === value) ?? SORT_OPTIONS[0];

  const pick = (v: SortKey) => {
    onChange(v);
    setOpen(false);
  };

  return (
    <div className="avisx__dd avisx__dd--sort" ref={rootRef}>
      <button
        type="button"
        className="avisx__dd-btn"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="avisx__dd-label">{selected.label}</span>
        <span className="avisx__dd-caret" aria-hidden>
          {open ? '▴' : '▾'}
        </span>
      </button>
      {open && (
        <ul className="avisx__dd-panel" role="listbox" aria-label="Trier les avis affichés">
          {SORT_OPTIONS.map((o) => (
            <li key={o.value} role="option" aria-selected={value === o.value}>
              <button
                type="button"
                className={`avisx__dd-item${value === o.value ? ' avisx__dd-item--on' : ''}`}
                onClick={() => pick(o.value)}
              >
                {o.label}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/**
 * Une carte d'avis INLINE : par DÉFAUT le contenu seul (texte + surlignages), sans aucune
 * titraille de thèmes redondante (le survol d'un surlignage donne déjà le cluster). CLIQUER
 * sur le corps de la carte ouvre/ferme une LÉGENDE D'ANALYSE sous l'avis (état local par
 * carte) : clusters + répartition de stance, façon fiche extensible. Le clic sur « Signaler »
 * (FlagControl) ne doit PAS ouvrir la légende → la propagation y est stoppée.
 */
const AvisCard = forwardRef<
  HTMLLIElement,
  {
    avis: AvisProvenance;
    dataset: string;
    flagText?: string;
    onFlagChange: (id: string, text: string | null) => void;
    /** Lookups pour la carte de stats d'un claim (feuille → volumes / opinion / citations). */
    themeById: Map<string, SpatialTheme>;
    opinionByTheme: Map<string, ThemeOpinion>;
    getCitations: (leafId: string) => Promise<Citation[]>;
    focused?: boolean;
  }
>(({ avis, dataset, flagText, onFlagChange, themeById, opinionByTheme, getCitations,
    focused = false }, ref) => {
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const toggle = () => setAnalysisOpen((o) => !o);

  // Claim sélectionné (clic sur un passage surligné) → carte de stats dashboard.
  const [selClaim, setSelClaim] = useState<AvisClaim | null>(null);
  const [selStats, setSelStats] = useState<ClaimStatsData>({});
  const onClaimClick = (claim: AvisClaim) => {
    if (selClaim?.id === claim.id) {
      setSelClaim(null); // re-clic → referme
      return;
    }
    const leafId = claim.leaf_id || claim.cluster_id;
    const leaf = leafId ? themeById.get(leafId) : undefined;
    const op = leafId ? opinionByTheme.get(leafId) : undefined;
    setSelClaim(claim);
    setSelStats({
      leafTitle: leaf ? leaf.title || leaf.label : claim.theme_title,
      nAvis: leaf?.n_avis ?? null,
      nClaims: leaf?.n_claims ?? null,
      opinion: op ? { fav: op.fav, def: op.def, nuance: op.nuance, proposition: op.proposition } : null,
      citation: null,
      loading: Boolean(leafId),
    });
    if (!leafId) return;
    // Représentativité : retrouver CE claim dans les citations de sa feuille (triées par
    // proximité au centroïde). Match par avis_id, affiné par texte de span si plusieurs.
    getCitations(leafId).then((list) => {
      const mine = list.filter((c) => c.avis_id === avis.id);
      let found = mine[0];
      if (mine.length > 1 && claim.spans.length > 0) {
        const spanText = avis.text.slice(claim.spans[0].start, claim.spans[0].end).trim();
        found = mine.find((c) => c.text && spanText && (c.text.includes(spanText.slice(0, 60)) || spanText.includes(c.text.slice(0, 60)))) ?? mine[0];
      }
      setSelStats((s) => ({
        ...s,
        loading: false,
        citation:
          found && typeof found.rank === 'number'
            ? { rank: found.rank, total: list.length, dist: found.dist_to_centroid }
            : null,
      }));
    });
  };
  return (
    <li
      ref={ref}
      className={`avisx__card${focused ? ' avisx__card--focus' : ''}${
        analysisOpen ? ' avisx__card--open' : ''
      }`}
    >
      <div className="avisx__cardhead">
        {/* Le head ne porte plus de chips de thèmes : seulement le bouton Signaler.
            Son clic stoppe la propagation (FlagControl) → n'ouvre pas la légende. */}
        <FlagControl
          dataset={dataset}
          avisId={avis.id}
          flagText={flagText}
          onFlagChange={onFlagChange}
        />
      </div>
      <div
        className="avisx__cardbody"
        role="button"
        tabIndex={0}
        aria-expanded={analysisOpen}
        aria-label={analysisOpen ? "Masquer l'analyse de cet avis" : "Afficher l'analyse de cet avis"}
        onClick={toggle}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggle();
          }
        }}
      >
        <AvisBody avis={avis} highlight onClaimClick={onClaimClick} />
      </div>
      {/* Carte de stats du claim cliqué (dashboard : volume, sentiment, lecture, représentativité). */}
      {selClaim && (
        <ClaimStatsCard claim={selClaim} stats={selStats} onClose={() => setSelClaim(null)} />
      )}
      {analysisOpen && <AvisAnalysis claims={avis.claims} />}
    </li>
  );
});

/** Vue `AvisProvenance` d'un item de liste (qui porte déjà l'avis entier). */
function toProvenance(it: AvisListItem): AvisProvenance {
  return { id: it.avis_id, text: it.text, text_fr: it.text_fr, lang: it.lang, claims: it.claims };
}

type SortKey = 'default' | 'claims' | 'longest' | 'shortest';

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: 'default', label: 'Pertinence (par défaut)' },
  { value: 'claims', label: 'Plus de passages retenus' },
  { value: 'longest', label: 'Témoignage le plus long' },
  { value: 'shortest', label: 'Témoignage le plus court' },
];

/** Trie le lot déjà chargé ; `'default'` = ordre serveur, laissé tel quel (pas de copie). */
function sortAvisItems(items: AvisListItem[], sortBy: SortKey): AvisListItem[] {
  if (sortBy === 'default') return items;
  const copy = [...items];
  switch (sortBy) {
    case 'claims':
      copy.sort((a, b) => (b.claims?.length ?? 0) - (a.claims?.length ?? 0));
      break;
    case 'longest':
      copy.sort((a, b) => (b.text?.length ?? 0) - (a.text?.length ?? 0));
      break;
    case 'shortest':
      copy.sort((a, b) => (a.text?.length ?? 0) - (b.text?.length ?? 0));
      break;
  }
  return copy;
}

/**
 * DFS hiérarchique : chaque thème avec sa profondeur ET des repères de tracé d'arbre —
 * `isLast` (dernier enfant de son parent, pour choisir le connecteur `└` plutôt que `├`)
 * et `guides` (pour chaque profondeur ANCÊTRE, si un trait vertical doit continuer parce
 * que cet ancêtre a encore des enfants après lui). Alimente un rendu en arborescence
 * (traits de branche) plutôt qu'une simple indentation par espaces.
 */
function orderThemes(
  themes: SpatialTheme[],
): { theme: SpatialTheme; depth: number; isLast: boolean; guides: boolean[] }[] {
  const children = new Map<string | null, SpatialTheme[]>();
  for (const t of themes) {
    const arr = children.get(t.parent_id) ?? [];
    arr.push(t);
    children.set(t.parent_id, arr);
  }
  const out: { theme: SpatialTheme; depth: number; isLast: boolean; guides: boolean[] }[] = [];
  const walk = (parent: string | null, depth: number, guides: boolean[]) => {
    const kids = children.get(parent) ?? [];
    kids.forEach((t, i) => {
      const isLast = i === kids.length - 1;
      out.push({ theme: t, depth, isLast, guides });
      walk(t.id, depth + 1, [...guides, !isLast]);
    });
  };
  walk(null, 0, []);
  return out;
}
