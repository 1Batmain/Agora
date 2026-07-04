import { useEffect, useState } from 'react';
import type { AvisListItem } from './contract';
import { fetchAvisList } from './analysisApi';
import { LOCALE } from './strings';

const PAGE = 8; // page « Voir plus » — widget secondaire, pas la page d'exploration.

/**
 * F9 — table des RÉPONSES DES CITOYENS pour le niveau/cluster courant (mandataire
 * du camembert, cf. `PieChart`). Toujours présente (contrairement aux anciennes
 * vues Graphe/Densité/Nuage) : quel que soit le niveau de drill, on voit les avis
 * concrets qui composent le cluster affiché, pas seulement sa proportion.
 *
 * Filtrée par `themeId` (macro → tous ses sous-thèmes, feuille → elle seule, null →
 * toute la consultation), paginée (« Voir plus »), avec un lien optionnel vers la
 * page d'exploration complète de l'avis (`onOpenAvis`).
 */
export function AnswersTable({
  dataset,
  themeId,
  title,
  onOpenAvis,
}: {
  dataset: string | null;
  themeId: string | null;
  title: string;
  onOpenAvis?: (avisId: string) => void;
}) {
  const [items, setItems] = useState<AvisListItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);

  useEffect(() => {
    if (!dataset) return;
    let cancelled = false;
    setLoading(true);
    fetchAvisList(dataset, { themeId, limit: PAGE, offset: 0 })
      .then(({ data }) => {
        if (cancelled) return;
        setItems(data?.items ?? []);
        setTotal(data?.total ?? 0);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [dataset, themeId]);

  function loadMore() {
    if (!dataset) return;
    setLoadingMore(true);
    fetchAvisList(dataset, { themeId, limit: PAGE, offset: items.length })
      .then(({ data }) => {
        if (data) {
          setItems((prev) => [...prev, ...data.items]);
          setTotal(data.total);
        }
      })
      .finally(() => setLoadingMore(false));
  }

  return (
    <section className="answerstable" aria-label="Avis des citoyens">
      <header className="answerstable__head">
        <h3 className="answerstable__title">{title}</h3>
        <span className="answerstable__count" aria-live="polite">
          {loading ? 'Chargement…' : `${total.toLocaleString(LOCALE)} avis`}
        </span>
      </header>

      {loading ? (
        <div className="answerstable__loading">
          <span className="spinner" /> chargement des avis…
        </div>
      ) : items.length === 0 ? (
        <p className="answerstable__empty">Aucun avis pour cette sélection.</p>
      ) : (
        <>
          <div className="answerstable__scroll">
            <table className="answerstable__table">
              <thead>
                <tr>
                  <th scope="col" className="answerstable__thidx">#</th>
                  <th scope="col">Réponse du citoyen</th>
                  <th scope="col">Thème(s)</th>
                  {onOpenAvis && <th scope="col" className="answerstable__thaction" />}
                </tr>
              </thead>
              <tbody>
                {items.map((it, i) => (
                  <tr key={it.avis_id}>
                    <td className="answerstable__idx">{i + 1}</td>
                    <td className="answerstable__excerpt">{it.excerpt}</td>
                    <td className="answerstable__themes">
                      <div className="answerstable__themewrap">
                        {it.themes.slice(0, 3).map((th) => (
                          <span key={th.id} className="answerstable__themechip" style={{ borderColor: th.color }}>
                            {th.title}
                          </span>
                        ))}
                        {it.themes.length > 3 && (
                          <span className="answerstable__thememore">+{it.themes.length - 3}</span>
                        )}
                      </div>
                    </td>
                    {onOpenAvis && (
                      <td className="answerstable__action">
                        <button
                          type="button"
                          className="answerstable__openbtn"
                          onClick={() => onOpenAvis(it.avis_id)}
                        >
                          Voir l'avis complet →
                        </button>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {items.length < total && (
            <button
              type="button"
              className="answerstable__more"
              disabled={loadingMore}
              onClick={loadMore}
            >
              {loadingMore ? 'Chargement…' : `Voir plus (${items.length}/${total.toLocaleString(LOCALE)})`}
            </button>
          )}
        </>
      )}
    </section>
  );
}
