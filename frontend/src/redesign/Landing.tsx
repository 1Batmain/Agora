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
  onTodo,
}: {
  datasets: Consultation[];
  loading: boolean;
  onOpen: (d: Consultation) => void;
  /** Ouvre la feuille de route IN-APP (route `?view=todo`) — câblé par le shell (`App`). */
  onTodo?: () => void;
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
            Agora analyse l’intégralité des réponses à une consultation pour en faire émerger
            les grands thèmes — de manière <strong>automatisée, transparente, rapide et
            économique</strong>.
          </p>
          <ul className="hero__diff">
            <li>
              <strong>Intelligible</strong>
              <span>restitue fidèlement les divergences et les consensus</span>
            </li>
            <li>
              <strong>Modulaire</strong>
              <span>les thèmes naissent des données, pas d’une grille — le modèle s’adapte à la donnée</span>
            </li>
            <li>
              <strong>Souverain</strong>
              <span>tourne sur des modèles ouverts, et peut être porté en local</span>
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
          <header className="sec-head">
            <span className="sec-head__num">01</span>
            <h2>Consultations</h2>
            <span className="sec-head__hint">ouvertes à participer · closes analysées</span>
          </header>
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
                      {/* Carte = le LABEL court (la question complète ne tient pas sur une
                          carte) ; la question est affichée comme titre sur la page consultation. */}
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

        <section className="landing__collab">
          <header className="sec-head">
            <span className="sec-head__num">02</span>
            <h2>Collaborer — hackathon</h2>
            <span className="sec-head__hint">repo ouvert · PR obligatoire</span>
          </header>
          <p className="how__lead">
            Tu es le bienvenu pour collaborer sur Agora.
          </p>
          <div className="collab__actions">
            <a
              className="collab__link"
              href="https://github.com/1Batmain/Analyse-des-consultations-citoyennes"
              target="_blank"
              rel="noreferrer"
            >
              <span className="collab__link-arrow" aria-hidden>→</span>
              <span className="collab__link-label">Repo GitHub</span>
              <span className="collab__link-sub">clone · crée ta branche · ouvre une PR</span>
            </a>
            <button type="button" className="collab__link" onClick={onTodo}>
              <span className="collab__link-arrow" aria-hidden>→</span>
              <span className="collab__link-label">Feuille de route</span>
              <span className="collab__link-sub">choisis une tâche · propose une amélioration</span>
            </button>
          </div>
        </section>

        <section className="landing__problem">
          <header className="sec-head">
            <span className="sec-head__num">03</span>
            <h2>Le problème à résoudre</h2>
            <span className="sec-head__hint">pourquoi Agora existe</span>
          </header>
          <div className="section-prose">
            <p>
              Une consultation publique, c'est des milliers — parfois des millions — de
              réponses en texte libre. Les analyser une par une est <strong>extrêmement long
              et coûteux</strong>.
            </p>
            <p>
              Agora est un outil qui analyse de larges volumes de données de manière{' '}
              <strong>rapide et économique</strong> — ce qui ouvre la porte à des consultations
              plus régulières et à une meilleure retranscription de la parole citoyenne.
            </p>
          </div>
        </section>

        <section className="landing__insp">
          <header className="sec-head">
            <span className="sec-head__num">04</span>
            <h2>Inspirations</h2>
            <span className="sec-head__hint">d'où l'on part</span>
          </header>
          <div className="section-prose">
            <p>
              Agora s'inscrit dans la lignée d'outils qui ont déjà fait leurs preuves.{' '}
              <strong>Pol.is</strong> a dégagé des consensus citoyens sur des sujets clivants —
              jusqu'à désamorcer le conflit Uber / taxis à Taïwan : la preuve qu'une
              consultation de masse peut éclairer une vraie décision.
            </p>
            <p>
              Agora reprend ces acquis mais part de <strong>l'expression même de chaque
              avis</strong>, pas de votes : plus souple, il s'applique à n'importe quelle
              consultation en texte libre.
            </p>
          </div>
        </section>

        <section className="landing__pipeline">
          <header className="sec-head">
            <span className="sec-head__num">05</span>
            <h2>Sous le capot — le pipeline</h2>
            <span className="sec-head__hint">5 étapes, des avis aux thèmes</span>
          </header>
          <p className="how__lead">
            Pour commencer, voilà une overview de l'état actuel du pipeline de traitement
            des consultations.
          </p>
          <ol className="pipe">
            <li className="pipe__step">
              <span className="pipe__tag">Mistral&nbsp;large</span>
              <strong>1 · Découper chaque avis en idées</strong>
              <p>
                Lorsqu'une contribution traite de différents sujets, on la segmente par sujet
                à l'aide du LLM.
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
                Les gros thèmes se subdivisent en sous-thèmes. <em>Leiden</em> nous a donné les
                meilleurs résultats jusqu'ici ; d'autres méthodes comme <em>HDBSCAN</em> ont
                aussi été testées.
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
      </main>
    </div>
  );
}
