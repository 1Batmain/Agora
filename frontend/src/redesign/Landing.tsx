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
          <p className="hero__eyebrow">
            <span className="hero__prompt">agora</span>
            analyse de consultations citoyennes
            <span className="hero__caret" aria-hidden />
          </p>
          <h1 className="hero__title">
            Des milliers d’avis citoyens,
            <span className="hero__accent"> les thèmes qui émergent</span>
            <br />
          </h1>
          <p className="hero__tagline">
            Agora lit l’intégralité des réponses à une consultation et en fait émerger
            les grands thèmes — <strong>sans jamais reformuler un mot</strong>. Chaque
            idée reste traçable jusqu’à la phrase exacte du citoyen.
          </p>
          <ul className="hero__diff">
            <li>
              <strong>Verbatim</strong>
              <span>zéro reformulation, zéro trahison du propos</span>
            </li>
            <li>
              <strong>Émergent</strong>
              <span>les thèmes naissent des données, pas d’une grille</span>
            </li>
            <li>
              <strong>Souverain</strong>
              <span>tourne en local, sur des modèles ouverts</span>
            </li>
          </ul>
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
              <strong>1 · Découper chaque avis en idées</strong>
              <p>
                Une contribution mélange souvent plusieurs idées. On la découpe donc en{' '}
                <em>claims</em> — une idée = une prise de position. Les phrases sont{' '}
                <em>recopiées telles quelles</em> (comme surligner dans une copie), jamais
                reformulées : ce qu'affiche Agora est toujours exactement ce qu'a écrit le
                citoyen. Un grand modèle de langue (IA) fait ce découpage.
              </p>
            </li>
            <li className="pipe__step">
              <span className="pipe__tag">embeddings · nomic</span>
              <strong>2 · Transformer chaque idée en « coordonnées de sens »</strong>
              <p>
                Chaque idée devient une liste de nombres (un <em>vecteur</em>) qui capture
                son SENS. Deux idées qui parlent de la même chose se retrouvent proches,
                même si elles n'emploient pas les mêmes mots — « addictif » et « je n'arrive
                pas à décrocher » finissent côte à côte.
              </p>
            </li>
            <li className="pipe__step">
              <span className="pipe__tag">graphe k-NN</span>
              <strong>3 · Relier les idées proches</strong>
              <p>
                On relie chaque idée à ses quelques voisines les plus ressemblantes (leurs
                « coordonnées de sens » sont proches). On obtient un grand <em>réseau</em>
                où les idées qui se ressemblent sont connectées. (C'est le « k plus proches
                voisins », ou&nbsp;k-NN.)
              </p>
            </li>
            <li className="pipe__step">
              <span className="pipe__tag">Leiden</span>
              <strong>4 · Laisser les thèmes émerger</strong>
              <p>
                Dans ce réseau, un algorithme (<em>Leiden</em>) repère les « paquets » d'idées
                très reliées entre elles : chaque paquet devient un <em>thème</em>. Les
                thèmes <em>émergent des données</em> — personne ne les a définis à l'avance.
                Les gros thèmes se subdivisent en sous-thèmes.
              </p>
            </li>
            <li className="pipe__step">
              <span className="pipe__tag">synthèse IA</span>
              <strong>5 · Nommer, résumer, garder la trace</strong>
              <p>
                Pour chaque thème, l'IA propose un titre court et un résumé, et on garde les
                avis les plus représentatifs. Tout reste <em>traçable</em> : on peut toujours
                remonter du thème jusqu'à la phrase exacte d'un citoyen.
              </p>
            </li>
          </ol>
          <p className="how__footer">
            Stack : Python · FastAPI · nomic-embed-v2 · Mistral · Leiden/igraph ·
            React + Vite + D3 — souverain &amp; local, modèles ouverts.
          </p>
        </section>

        <section className="landing__collab">
          <h2>Collaborer — hackathon</h2>
          <p className="how__lead">
            Le projet est ouvert : code à la main ou avec des agents, comme tu préfères.
            Quelques repères pour démarrer vite et travailler en parallèle sans se marcher
            dessus.
          </p>
          <div className="collab__grid">
            <div className="collab__card">
              <strong>🛣️ Choisis ta lane</strong>
              <p>
                Quatre couloirs : <em>pipeline/ML</em> (extraction, clustering) ·{' '}
                <em>backend/API</em> · <em>frontend</em> · <em>research/éval</em>. Les{' '}
                <em>contrats</em> (<code>contract.ts</code>, schéma Consultation) sont
                l'interface entre lanes — on travaille en parallèle sans collision.
              </p>
            </div>
            <div className="collab__card">
              <strong>🌿 Branche + gate</strong>
              <p>
                Une branche par feature (jamais sur <code>main</code>). Avant de merger :{' '}
                <em>pytest</em> vert + <em>npm run build</em> propre. Un mainteneur relit et
                intègre.
              </p>
            </div>
            <div className="collab__card">
              <strong>🧱 Invariants à ne jamais casser</strong>
              <p>
                <em>Fidélité verbatim</em> (les claims = mot pour mot) ·{' '}
                <em>zéro hardcoding</em> (tout dérivé des données, marche sur n'importe quelle
                consultation) · <em>build/serve séparés</em> (le serveur ne fait que lire des
                caches).
              </p>
            </div>
            <div className="collab__card">
              <strong>🔬 Le verdict d'abord</strong>
              <p>
                Tout changement d'algo (clustering, extraction…) se valide par un{' '}
                <em>A/B sur échantillon</em> + une note dans <code>research/</code> avant de
                toucher le chemin servi. On mesure, on ne devine pas.
              </p>
            </div>
            <div className="collab__card">
              <strong>🚀 Setup en 5 min</strong>
              <p>
                Clé Mistral dans <code>var/mistral.key</code> · backend <code>:8010</code>{' '}
                (FastAPI) · front <code>:5180</code> (Vite). Les contrats back↔front vivent
                dans <code>contract.ts</code>.
              </p>
            </div>
            <div className="collab__card">
              <strong>🤖 À la main ou en agents</strong>
              <p>
                Code classiquement, ou orchestre des agents dans ta lane — au choix. Le
                workflow lanes + contrats + gate marche pour les deux.
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
