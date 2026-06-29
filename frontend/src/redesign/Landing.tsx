import { Header } from './Header';
import type { Consultation } from './contract';
import { LOCALE } from './strings';

/**
 * Vue d'accueil d'Agora — hero « façon framework » (grand titre mono, tagline,
 * beaucoup de blanc, curseur typewriter clignotant en déco), puis la grille de
 * consultations (depuis `/datasets`). Chaque carte porte un badge Ouvert/Clos,
 * son titre et son nombre de contributions. Le clic remonte au shell, qui route
 * vers l'analyse (clos) ou la participation (ouvert).
 */
export function Landing({
  datasets,
  loading,
  onOpen,
}: {
  datasets: Consultation[];
  loading: boolean;
  onOpen: (d: Consultation) => void;
}) {
  const openCount = datasets.filter((d) => d.status === 'open').length;
  const closedCount = datasets.length - openCount;

  return (
    <div className="agora landing">
      <Header />

      <main className="landing__body">
        <section className="hero">
          <h1 className="hero__title">
            Analyse des consultations
            <span className="hero__accent"> citoyennes </span>
            <br />
          </h1>
          <p className="hero__tagline">
            Agora fait émerger les thèmes et les idées communes des questions ouvertes
            et fournit une analyse des opinions de manière automatisée, traçable et
            transparente.
          </p>
          <ul className="hero__meta" aria-hidden={datasets.length === 0}>
            <li>
              <strong>{datasets.length}</strong> consultations
            </li>
            <li>
              <strong>{openCount}</strong> ouvertes
            </li>
            <li>
              <strong>{closedCount}</strong> analysées
            </li>
          </ul>
        </section>

        <section className="landing__list">
          <h2>Consultations</h2>
          {loading ? (
            <div className="landing__loading">
              <span className="spinner" /> chargement des consultations…
            </div>
          ) : datasets.length === 0 ? (
            <p className="landing__empty">Aucune consultation disponible.</p>
          ) : (
            <ul className="landing__grid">
              {datasets.map((d) => {
                const open = d.status === 'open';
                return (
                  <li key={d.id}>
                    <button
                      className="ds-card"
                      onClick={() => onOpen(d)}
                      aria-label={`${d.label} — ${open ? 'consultation ouverte' : 'consultation close'}`}
                    >
                      <span className={`ds-card__badge ds-card__badge--${open ? 'open' : 'closed'}`}>
                        <span className="ds-card__dot" aria-hidden />
                        {open ? 'Ouvert' : 'Clos'}
                      </span>
                      <span className="ds-card__title">{d.label}</span>
                      <span className="ds-card__meta">
                        {(() => {
                          const n = d.n_contributions ?? d.n_nodes;
                          return n ? `${n.toLocaleString(LOCALE)} contributions` : '—';
                        })()}
                      </span>
                      <span className="ds-card__cta">
                        {open ? 'Participer →' : 'Voir l’analyse →'}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section className="landing__pipeline">
          <h2>Sous le capot — le pipeline</h2>
          <p className="how__lead">
            Page d'onboarding pour le hackathon. Voici comment Agora transforme des
            milliers de contributions en une carte de thèmes navigable. Le pipeline
            s'enrichit au fil des itérations — viens en construire la suite.
          </p>
          <ol className="pipe">
            <li className="pipe__step">
              <span className="pipe__tag">Mistral&nbsp;large</span>
              <strong>1 · Découpage en claims</strong>
              <p>
                Chaque contribution est segmentée en <em>claims</em> — des prises de
                position recopiées <em>verbatim</em> (multi-segments + cible), validées
                par un gate d'ancrage exact : zéro hallucination, 100&nbsp;% fidèle à l'avis.
              </p>
            </li>
            <li className="pipe__step">
              <span className="pipe__tag">nomic-embed-v2</span>
              <strong>2 · Embeddings</strong>
              <p>
                Chaque claim est projeté dans un espace vectoriel sémantique par un
                encodeur local et multilingue.
              </p>
            </li>
            <li className="pipe__step">
              <span className="pipe__tag">k-NN · faiss</span>
              <strong>3 · Graphe de proximité</strong>
              <p>
                On relie chaque claim à ses <em>k</em> plus proches voisins (similarité
                cosinus) → un graphe où les idées sémantiquement proches sont connectées.
              </p>
            </li>
            <li className="pipe__step">
              <span className="pipe__tag">Leiden · igraph</span>
              <strong>4 · Clusters (Leiden)</strong>
              <p>
                Détection de communautés sur le graphe → les <em>thèmes</em> émergent
                des données. Hiérarchique (macro → sous-thèmes), avec subdivision
                variance-adaptative et coarsening pour éviter les fourre-tout.
              </p>
            </li>
            <li className="pipe__step">
              <span className="pipe__tag">LLM · caché</span>
              <strong>5 · Enrichissement</strong>
              <p>
                Titre + synthèse par thème, citations représentatives (proches du
                centroïde), indices honnêtes (couverture, fidélité verbatim, modularité…).
              </p>
            </li>
          </ol>
          <p className="how__footer">
            Stack : Python · FastAPI · nomic-embed-v2 · Mistral · Leiden/igraph ·
            React + Vite + D3 — souverain &amp; local, modèles ouverts.
          </p>
        </section>
      </main>
    </div>
  );
}
