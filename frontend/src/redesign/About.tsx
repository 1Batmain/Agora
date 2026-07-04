import { Header } from './Header';

/** Icône GitHub (octocat) inline — évite une dépendance à une lib d'icônes pour ce
 *  seul usage (badge « profil GitHub » des cartes contributeur). */
function GithubIcon() {
  return (
    <svg viewBox="0 0 16 16" width="10" height="10" fill="currentColor" aria-hidden>
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
        0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
        -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07
        -1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82
        .64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
        .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
        0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
    </svg>
  );
}

/** Contributeurs affichés en bas de page — `linkedin`/`github` optionnels (pas de lien
 *  fabriqué : seuls les profils communiqués par les intéressés sont renseignés). */
const TEAM: { name: string; role: string; linkedin?: string; github?: string }[] = [
  { name: 'Baptiste Duval', role: 'Mainteneur', linkedin: 'https://www.linkedin.com/in/baptiste-duval-4508422a9/' },
  { name: 'Eliott Pelpel', role: 'Contributeur', linkedin: 'https://www.linkedin.com/in/eliott-pelpel/' },
  { name: 'Rick Gao', role: 'Contributeur', linkedin: 'https://www.linkedin.com/in/rickgao03/' },
  { name: 'Anis Outaleb', role: 'Contributeur', linkedin: 'https://www.linkedin.com/in/aniss-outaleb/' },
  { name: 'Pierre Chhieng', role: 'Contributeur', linkedin: 'https://www.linkedin.com/in/pierre-chhieng/' },
  { name: 'Paul Pazart', role: 'Contributeur', linkedin: 'https://www.linkedin.com/in/paul-pazart/' },
  { name: 'Amine Saboni', role: 'Contributeur', linkedin: 'https://www.linkedin.com/in/amine-saboni/' },
  { name: 'Antoine Monot', role: 'Contributeur', linkedin: 'https://www.linkedin.com/in/antoine-monot/' },
  { name: 'Ksenia Ossi', role: 'Contributeur', linkedin: 'https://www.linkedin.com/in/ksenia-ossi/' },
  { name: 'Sebastien Bo', role: 'Contributeur', linkedin: 'https://www.linkedin.com/in/sebastien-bo-402096340/' },
];

/**
 * Page « À propos » — vitrine PÉDAGOGIQUE du projet : le problème qu'Agora résout, le
 * pipeline technique en détail (des avis bruts à la carte des thèmes), l'architecture
 * du dépôt, la posture sécurité/souveraineté, et comment collaborer. Contenu STATIQUE
 * (pas d'appel réseau) — accessible depuis la landing (« Comment ça marche ? »).
 */
export function About({ onHome }: { onHome: () => void }) {
  return (
    <div className="agora landing">
      <Header onHome={onHome} />

      <main className="landing__body">
        <section className="hero">
          <span className="hero__eyebrow">À propos</span>
          <h1 className="hero__title">
            Comment fonctionne
            <span className="hero__accent"> Agora</span>
            <br />
          </h1>
          <p className="hero__tagline">
            Agora transforme des milliers de témoignages citoyens en texte libre en une{' '}
            <strong>carte de thèmes navigable</strong> — sans jamais reformuler un mot, et
            sans faire sortir la moindre donnée vers un cloud tiers.
          </p>
        </section>

        <section className="landing__problem">
          <header className="sec-head">
            <span className="sec-head__num">01</span>
            <h2>Le problème</h2>
            <span className="sec-head__hint">pourquoi Agora existe</span>
          </header>
          <div className="section-prose">
            <p>
              Quand des dizaines de milliers de citoyens répondent à une consultation,
              personne ne lit tout. On résume — et en résumant, on <strong>trahit</strong> :
              on reformule, on lisse, on choisit d'avance les cases. Le citoyen ne se
              reconnaît plus dans la synthèse officielle.
            </p>
            <p>
              <strong>Agora fait l'inverse.</strong> Les thèmes ne sont pas imposés à
              l'avance : ils <strong>émergent</strong> des contributions elles-mêmes. Rien
              n'est reformulé — chaque affirmation reste le verbatim exact de la personne,
              traçable jusqu'à son témoignage d'origine.
            </p>
          </div>
          <ul className="hero__diff">
            <li>
              <strong>Fidèle</strong>
              <span>zéro reformulation — les extraits sont des sous-chaînes exactes du texte citoyen</span>
            </li>
            <li>
              <strong>Traçable</strong>
              <span>chaque thème → ses claims → l'avis complet, surligné, d'où ils viennent</span>
            </li>
            <li>
              <strong>Souverain</strong>
              <span>embeddings calculés en local, aucun texte citoyen envoyé à un cloud tiers</span>
            </li>
          </ul>
        </section>

        <section className="landing__pipeline">
          <header className="sec-head">
            <span className="sec-head__num">02</span>
            <h2>Le pipeline, en détail</h2>
            <span className="sec-head__hint">6 étapes, des avis aux thèmes</span>
          </header>
          <p className="how__lead">
            Du texte brut à la carte des opinions, sans jamais perdre le fil vers la
            source :
          </p>
          <ol className="pipe">
            <li className="pipe__step">
              <span className="pipe__tag">Mistral&nbsp;large</span>
              <strong>1 · Claims — extraction verbatim</strong>
              <p>
                Chaque avis est découpé en affirmations (<em>claims</em>) : la question de
                la consultation sert de cadre, mais aucune paraphrase — chaque claim est un
                morceau EXACT du texte d'origine (multi-passages si besoin).
              </p>
            </li>
            <li className="pipe__step">
              <span className="pipe__tag">embeddings · nomic</span>
              <strong>2 · Embeddings — coordonnées de sens</strong>
              <p>
                Chaque claim devient un vecteur, calculé <em>en local</em> (souverain, hors
                ligne). Deux idées qui parlent de la même chose se retrouvent proches, même
                si elles n'emploient pas les mêmes mots.
              </p>
            </li>
            <li className="pipe__step">
              <span className="pipe__tag">graphe k-NN</span>
              <strong>3 · Graphe k-NN — relier les idées proches</strong>
              <p>
                Chaque claim est relié à ses quelques voisins les plus ressemblants
                (similarité cosinus sur les vecteurs). On obtient un grand réseau où les
                idées proches sont connectées.
              </p>
            </li>
            <li className="pipe__step">
              <span className="pipe__tag">Leiden</span>
              <strong>4 · Leiden — les thèmes émergent</strong>
              <p>
                Un algorithme de détection de communautés (<em>Leiden</em>) repère les
                paquets de claims très reliés entre eux : chaque paquet devient un thème.
                Les gros thèmes se subdivisent en sous-thèmes — une hiérarchie
                variance-adaptative, jamais une taxonomie imposée à l'avance.
              </p>
            </li>
            <li className="pipe__step">
              <span className="pipe__tag">synthèse IA</span>
              <strong>5 · Enrichissement — nommer, résumer</strong>
              <p>
                Pour chaque thème, un LLM (sur le <em>cluster</em>, jamais sur le citoyen
                nommément) propose un titre court et une synthèse Markdown, et on retient
                les avis les plus représentatifs (proximité au centroïde).
              </p>
            </li>
            <li className="pipe__step">
              <span className="pipe__tag">stance calibrée</span>
              <strong>6 · Opinion — pour / contre / nuancé</strong>
              <p>
                Sur chaque thème, le modèle identifie l'objet de clivage et classe chaque
                claim (favorable / défavorable / nuancé) avec un niveau de confiance —
                jamais présenté comme un sondage.
              </p>
            </li>
          </ol>
          <p className="how__footer">
            Résultat : une carte navigable thèmes → sous-thèmes → témoignages surlignés,
            où l'on peut toujours remonter du thème jusqu'à la phrase exacte d'un citoyen.
          </p>
        </section>

        <section className="landing__archi">
          <header className="sec-head">
            <span className="sec-head__num">03</span>
            <h2>Architecture du dépôt</h2>
            <span className="sec-head__hint">qui fait quoi</span>
          </header>
          <p className="how__lead">
            Une séparation nette entre le calcul (fait une fois, hors ligne) et le service
            (rapide, sans clé, sans calcul lourd à la requête) :
          </p>
          <ul className="archi">
            <li className="archi__item">
              <span className="archi__tag">Python · FastAPI</span>
              <strong>backend/</strong>
              <p>
                API qui <em>sert</em> le cache précalculé (thèmes, synthèses, citations,
                opinion, avis) — aucun calcul lourd à la requête — et les scripts de
                construction (<code>build_analysis</code>, <code>build_opinion</code>,{' '}
                <code>build_cache</code>).
              </p>
            </li>
            <li className="archi__item">
              <span className="archi__tag">Python</span>
              <strong>pipeline/</strong>
              <p>
                Le cœur algorithmique : extraction des claims (<code>claims/</code>),
                vectorisation (<code>embed/</code>), constitution du graphe et détection de
                communautés (<code>cluster/</code>), ingestion des corpus bruts
                (<code>ingest/</code>).
              </p>
            </li>
            <li className="archi__item">
              <span className="archi__tag">React · Vite · TypeScript</span>
              <strong>frontend/</strong>
              <p>
                L'application que tu utilises : carte des thèmes, explorateur de
                témoignages, synthèses par niveau, répartition d'opinion — le contrat de
                types front↔back vit dans <code>src/redesign/contract.ts</code>.
              </p>
            </li>
            <li className="archi__item">
              <span className="archi__tag">scripts + Tailscale</span>
              <strong>deploy/</strong>
              <p>
                Scripts de déploiement et de promotion de cache vers la production — un
                environnement séparé du développement, sans clé LLM, qui ne sert que du
                cache déjà calculé.
              </p>
            </li>
            <li className="archi__item">
              <span className="archi__tag">GitHub Actions</span>
              <strong>.github/</strong>
              <p>
                CI (tests + build sur chaque PR) et déploiement automatique : un merge sur{' '}
                <code>main</code> redéploie seul le serveur de production.
              </p>
            </li>
            <li className="archi__item">
              <span className="archi__tag">Markdown</span>
              <strong>.agent/</strong>
              <p>
                Notes de R&amp;D (décisions d'algorithmes, arbitrages mesurés) et ledger des
                tâches en cours — la mémoire du projet, pour que chaque idée « on pourrait…
                » soit vérifiée avant d'être adoptée.
              </p>
            </li>
          </ul>
          <p className="how__footer">
            Stack : Python 3.11 · FastAPI · nomic-embed-v2 · Mistral · Leiden/igraph ·
            React + Vite + TypeScript — souverain &amp; local, modèles ouverts.
          </p>
        </section>

        <section className="landing__method">
          <header className="sec-head">
            <span className="sec-head__num">04</span>
            <h2>Comment on obtient ces résultats</h2>
            <span className="sec-head__hint">méthode &amp; transparence</span>
          </header>
          <div className="section-prose">
            <p>
              Chaque chiffre affiché (un thème, un pourcentage, une synthèse) est le
              produit du <strong>pipeline ci-dessus</strong> appliqué à un
              <strong> échantillon</strong> des contributions reçues — jamais une opinion
              ajoutée après coup. Quand l'analyse ne porte pas sur l'intégralité des
              réponses, la taille et la couverture de l'échantillon sont affichées en toutes
              lettres sur la page de la consultation (bandeau « Échantillon » ou « Couverture
              complète »).
            </p>
            <p>
              La <strong>répartition d'opinion</strong> (favorable / défavorable / nuancé)
              n'est pas un vote : pour chaque thème, un modèle identifie l'objet de clivage
              puis classe chaque passage retenu, avec un <strong>niveau de confiance
              auto-évalué</strong> (élevée / moyenne / faible) affiché à côté de sa lecture —
              jamais présenté comme une certitude.
            </p>
            <p>
              Le <strong>coût et la durée réels</strong> du traitement (tokens consommés,
              $ estimés, temps de calcul) sont exposés en pied de page de chaque
              consultation, pour comparer honnêtement à un dépouillement humain classique.
            </p>
          </div>
        </section>

        <section className="landing__warn">
          <header className="sec-head">
            <span className="sec-head__num">05</span>
            <h2>Limites &amp; contenu généré par IA</h2>
            <span className="sec-head__hint">à lire avant de faire confiance aux chiffres</span>
          </header>
          <div className="notebox" role="note">
            <p className="notebox__title">
              <span className="notebox__icon" aria-hidden>⚠</span>
              Une partie du contenu affiché est <strong>généré par un modèle de langage</strong>{' '}
              (Mistral) — il peut se tromper.
            </p>
            <ul>
              <li>
                <strong>Titres et synthèses de thèmes</strong> sont rédigés par un LLM à
                partir des verbatims du cluster. Même s'il travaille sur des extraits réels,
                un résumé peut rester <strong>approximatif ou incomplet</strong> — il ne
                remplace jamais la lecture des témoignages sources.
              </li>
              <li>
                <strong>La classification d'opinion</strong> (pour / contre / nuancé) et le
                niveau de confiance qui l'accompagne sont une <strong>estimation
                automatique</strong>, pas un jugement humain vérifié un par un.
              </li>
              <li>
                <strong>Les pourcentages ne sont pas un sondage</strong> : ils décrivent les
                contributions reçues (participation volontaire), pas l'opinion de la
                population générale.
              </li>
              <li>
                En revanche, chaque <strong>extrait verbatim</strong> (claim) reste, lui, une
                sous-chaîne EXACTE du texte citoyen — c'est l'interprétation qui peut se
                tromper, jamais la citation elle-même.
              </li>
              <li>
                Une extraction ou une lecture qui semble fausse ? Le bouton{' '}
                <strong>« Signaler »</strong>, présent sur chaque avis, permet de le
                remonter à l'équipe.
              </li>
            </ul>
          </div>
        </section>

        <section className="landing__security">
          <header className="sec-head">
            <span className="sec-head__num">06</span>
            <h2>Sécurité &amp; souveraineté</h2>
            <span className="sec-head__hint">fail-closed par défaut</span>
          </header>
          <div className="section-prose">
            <p>
              En exposition publique, un mode <strong>« fail-closed »</strong> inverse la
              posture par défaut : les endpoints de calcul ou de construction sont{' '}
              <strong>refusés</strong> sans jeton d'accès, et aucune extraction LLM n'est
              jamais déclenchée par une simple lecture. Seule la consultation du cache déjà
              calculé reste ouverte.
            </p>
            <p>
              Les embeddings tournent en <strong>local</strong> — aucun texte citoyen n'est
              envoyé à un service tiers pour cette étape — et les données sensibles
              (secrets, corpus bruts) restent hors du dépôt Git.
            </p>
          </div>
        </section>

        <section className="landing__collab">
          <header className="sec-head">
            <span className="sec-head__num">07</span>
            <h2>Collaborer</h2>
            <span className="sec-head__hint">repo ouvert · PR obligatoire</span>
          </header>
          <p className="how__lead">
            Agora est né lors du hackathon de l'Assemblée nationale et continue de vivre en
            code ouvert. Tu es le bienvenu pour y contribuer.
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
          </div>
          <p className="how__lead" style={{ marginBottom: '.9rem' }}>
            Le flux est simple : une branche, une PR vers <code>main</code>, la CI
            (tests + build) doit passer au vert, puis le serveur de production se déploie
            tout seul après le merge.
          </p>
          <ul className="team">
            {TEAM.map((m) => {
              // Priorité LinkedIn si les deux sont renseignés — un seul lien par carte.
              const link = m.linkedin ?? m.github;
              const linkTitle = m.linkedin
                ? `${m.name} sur LinkedIn`
                : `${m.name} sur GitHub`;
              const inner = (
                <>
                  <span className="team__avatar" aria-hidden>{m.name.charAt(0)}</span>
                  <span className="team__info">
                    <span className="team__name">
                      {m.name}
                      {m.linkedin && (
                        <span className="team__badge team__badge--linkedin" aria-hidden title="LinkedIn">in</span>
                      )}
                      {m.github && (
                        <span className="team__badge team__badge--github" aria-hidden title="GitHub">
                          <GithubIcon />
                        </span>
                      )}
                    </span>
                    <span className="team__role">{m.role}</span>
                  </span>
                </>
              );
              return (
                <li key={m.name}>
                  {link ? (
                    <a
                      className="team__card team__card--link"
                      href={link}
                      target="_blank"
                      rel="noreferrer"
                      title={linkTitle}
                    >
                      {inner}
                    </a>
                  ) : (
                    <span className="team__card">{inner}</span>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      </main>
    </div>
  );
}
