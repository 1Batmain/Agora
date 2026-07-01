import { forwardRef, useEffect, useMemo, useRef, useState } from 'react';
import type {
  AnalysisPayload,
  AvisListItem,
  AvisProvenance,
  Consultation,
  SpatialTheme,
} from './contract';
import { fetchAnalysis, fetchAvis, fetchAvisList, fetchFlags } from './analysisApi';
import { Header } from './Header';
import { AvisAnalysis, AvisBody, FlagControl } from './AvisDetail';
import { LOCALE } from './strings';

const PAGE = 15; // taille d'une page « Voir plus » (items lourds : avis entiers inline)

/**
 * Page d'EXPLORATION DES AVIS — AUTO-SUFFISANTE : recense TOUS les avis d'une
 * consultation et affiche CHACUN EN ENTIER, INLINE, sous forme de carte : son texte
 * complet avec ses **surlignages** verbatim (claims, réutilise `AvisBody`), un toggle
 * **FR / original** par avis (si traduit) et un bouton **« Signaler »** par avis. Un
 * toggle GLOBAL « Surligner les passages retenus » (défaut ON) masque/affiche les
 * surlignages sur toutes les cartes. Recherche plein-texte (debounce), filtre par
 * cluster et pagination « Voir plus ». PLUS de page séparée par avis : `focusAvisId`
 * (entrée depuis une citation) épingle la carte ciblée en tête, mise en évidence et
 * scrollée en vue.
 */
export function AvisExplorer({
  dataset,
  focusAvisId,
  focusThemeId,
  onHome,
}: {
  dataset: Consultation;
  /** Avis à mettre en évidence au chargement (deep-link `&focus=`) : carte épinglée. */
  focusAvisId?: string | null;
  /** Thème sur lequel PRÉ-FILTRER l'explorateur (bouton « Consulter les témoignages du thème »). */
  focusThemeId?: string | null;
  onHome: () => void;
}) {
  // Filtres : saisie immédiate `qInput` → `q` debouncé (300 ms) ; thème sélectionné
  // (initialisé sur le thème de focus quand on arrive depuis un sous-thème).
  const [qInput, setQInput] = useState('');
  const [q, setQ] = useState('');
  const [themeId, setThemeId] = useState<string | null>(focusThemeId ?? null);

  // Synchronise le filtre si le thème de focus change (navigation depuis un autre sous-thème).
  useEffect(() => {
    setThemeId(focusThemeId ?? null);
  }, [focusThemeId]);

  const [themes, setThemes] = useState<SpatialTheme[]>([]);
  const [items, setItems] = useState<AvisListItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);

  // Toggle GLOBAL des surlignages (défaut ON) — pilote toutes les cartes.
  const [showHighlights, setShowHighlights] = useState(true);

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
    fetchAvisList(dataset.id, { themeId, q, limit: PAGE, offset: items.length })
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

  // Thèmes ordonnés hiérarchiquement (parent puis enfants) avec profondeur → <option> indentées.
  const themeOptions = useMemo(() => orderThemes(themes), [themes]);
  const hasMore = items.length < total;
  // L'avis épinglé est retiré de la liste pour éviter un doublon avec la carte en tête.
  const listItems = focusAvis ? items.filter((it) => it.avis_id !== focusAvis.id) : items;

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
                {'  '.repeat(depth)}
                {theme.title || theme.label}
              </option>
            ))}
          </select>
          <label className="avisx__hltoggle" title="Afficher ou masquer les passages retenus">
            <input
              type="checkbox"
              checked={showHighlights}
              onChange={(e) => setShowHighlights(e.target.checked)}
            />
            Surligner les passages retenus
          </label>
        </div>

        <p className="avisx__count" aria-live="polite">
          {loading
            ? 'Chargement…'
            : `${total.toLocaleString(LOCALE)} avis${q || themeId ? ' (filtrés)' : ''}`}
        </p>

        <ul className="avisx__list">
          {/* Carte épinglée : l'avis ciblé par une citation, mis en évidence + scrollé. */}
          {focusAvis && (
            <AvisCard
              ref={focusRef}
              avis={focusAvis}
              dataset={dataset.id}
              highlight={showHighlights}
              flagText={flags[focusAvis.id]}
              onFlagChange={onFlagChange}
              focused
            />
          )}

          {!loading &&
            listItems.map((it) => (
              <AvisCard
                key={it.avis_id}
                avis={toProvenance(it)}
                dataset={dataset.id}
                highlight={showHighlights}
                flagText={flags[it.avis_id]}
                onFlagChange={onFlagChange}
              />
            ))}
        </ul>

        {!loading && listItems.length === 0 && !focusAvis && (
          <p className="overview__loading">Aucun avis ne correspond.</p>
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
    highlight: boolean;
    flagText?: string;
    onFlagChange: (id: string, text: string | null) => void;
    focused?: boolean;
  }
>(({ avis, dataset, highlight, flagText, onFlagChange, focused = false }, ref) => {
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const toggle = () => setAnalysisOpen((o) => !o);
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
        aria-label="Cliquer pour analyser cet avis"
        onClick={toggle}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggle();
          }
        }}
      >
        <AvisBody avis={avis} highlight={highlight} />
      </div>
      <p className="avisx__analyzehint" aria-hidden>
        {analysisOpen ? '▾ analyse de l’avis' : '▸ cliquer pour analyser'}
      </p>
      {analysisOpen && <AvisAnalysis claims={avis.claims} />}
    </li>
  );
});

/** Vue `AvisProvenance` d'un item de liste (qui porte déjà l'avis entier). */
function toProvenance(it: AvisListItem): AvisProvenance {
  return { id: it.avis_id, text: it.text, text_fr: it.text_fr, lang: it.lang, claims: it.claims };
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
