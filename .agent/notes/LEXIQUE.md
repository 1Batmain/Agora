# Lexique Agora — le vocabulaire du projet, expliqué

Tout le vocabulaire qui circule dans nos échanges et dans le code, avec ce que le terme veut
dire **ici** (pas la définition générale) et pourquoi il compte. Ordre : méthode → produit →
algorithmes → modèles → infra → git/agents.

> Convention : ⚠️ signale un piège ou un invariant qu'on s'est déjà fait avoir à enfreindre.

---

## 1. Méthode de travail

**Groom / Run** — les deux phases par défaut. *Groom* = transformer une idée en tâche cadrée
(but, contraintes, critère d'acceptation, dépendances) et la mettre en file, **sans rien
lancer**. *Run* = sur ton « go », on exécute. Par défaut on groome ; les agents restent froids.

**Spike** — (du vocabulaire agile) une **expérience courte et jetable dont le seul but est de
répondre à une question**, pas de livrer une fonctionnalité. On ne cherche pas du code propre :
on cherche un chiffre qui tranche. Ex. « spike live » = rejouer un dataset par tranches pour
mesurer si les thèmes gardent leur identité — après quoi le code peut être jeté, seul le
verdict compte. C'est l'inverse d'un chantier : borné, pas cumulatif.

**Verdict (OUI/NON + pourquoi)** — règle du projet : on ne change rien sans avoir **mesuré**,
et la mesure s'écrit dans une note. `.agent/notes/` et `research/*.md` sont le cimetière des
verdicts. Ça évite de re-débattre six mois plus tard.

**Witness de contrôle** — un **témoin** : un cas dont on connaît d'avance le résultat, pour
prouver que la mesure marche. Règle : *jamais de résultat négatif sans témoin positif.* Si un
test dit « aucune différence », il faut montrer qu'il **sait** détecter une différence quand
elle existe — sinon on ne distingue pas « pas d'effet » de « instrument cassé ».

**Iso-comportement** — un changement de code qui **ne change aucune sortie**. C'est l'exigence
d'un refactor honnête : on réorganise, on ne modifie pas. Se prouve, ne se déclare pas (cf.
*cache HIT*).

**Critère d'acceptation** — la condition, écrite **à l'avance**, qui dit si la tâche est
réussie. Écrite avant le code (« spec-first »), sinon on ajuste inconsciemment le critère au
résultat obtenu.

**Robustesse d'un bench** — ne jamais conclure sur un petit échantillon. En pratique : N large,
plusieurs datasets, juge répété. Un écart mesuré sur 20 avis n'est pas un résultat.

**Généricité / « zéro nom de corpus en dur »** — invariant du projet : aucun identifiant de
consultation (`tiktok`, `granddebat`…) ne doit apparaître dans la logique. Les datasets sont
**découverts** (scan de dossiers, descripteurs déclaratifs). Sinon chaque nouvelle consultation
demande de toucher le code.

**Single source of truth** — une information n'est définie **qu'à un seul endroit**. Quand elle
est dupliquée, les copies divergent en silence (on en a 14 sur les modèles, cf.
`PIPELINE_PROFILE.md`).

**Invariant** — une propriété qui doit être vraie **en permanence**, pas seulement au moment où
on l'écrit. Ex. « le verbatim est extractif ». On les protège par des assertions et des tests.

**Fail-closed** — en cas de doute, **refuser** plutôt que laisser passer. Ex. en mode public,
un endpoint d'écriture répond 403 au lieu de tenter l'opération. L'inverse (*fail-open*) rend
les incidents invisibles.

**Dette / legacy** — du code qui n'a plus de consommateur mais qu'on lit encore par erreur.
D'où les suppressions franches (−1719 lignes récemment) plutôt que le « au cas où ».

---

## 2. Le produit — objets métier

**Consultation / dataset** — une consultation citoyenne (TikTok, Grand Débat, République
numérique…). *Dataset* est son nom technique : un sous-dossier de `backend/cache/`.
Statut `open` (on collecte) ou `closed` (on analyse).

**Avis** — **une** contribution d'**une** personne. L'unité d'entrée. Mesuré : 1 avis produit
en moyenne **1,45 claim**.

**Claim** — une **affirmation** extraite d'un avis. Un avis qui dit trois choses donne trois
claims. C'est l'unité réellement analysée (clusterisée, classée en stance), pas l'avis.

**Verbatim / extractif** — ⚠️ **l'invariant central du projet**. Un claim est une **copie
exacte** d'un morceau de l'avis, jamais une reformulation. Si le LLM paraphrase, on rejette.
Raison : c'est une consultation citoyenne — faire dire à quelqu'un ce qu'il n'a pas dit
disqualifie l'outil. (Verdict mesuré : la version paraphrasée perdait face à la version
verbatim en panel aveugle.)

**Span / ancrage** — le **span** est la position `(début, fin)` du claim dans le texte de
l'avis. *Ancrer* = vérifier que le texte rendu par le LLM se retrouve **littéralement** dans
l'original. Un claim non ancré est jeté ; un avis dont rien n'est ancré retombe sur « l'avis
entier = 1 claim ». C'est le mécanisme qui **fait respecter** l'invariant verbatim.

**Thème (fin) / macro** — deux étages. Les **thèmes fins** sont le résultat direct du
clustering : précis mais redondants. Les **macros** les regroupent en familles lisibles.

**Abstraction (moteur d')** — comment on fabrique les macros. ⚠️ **Pas** en demandant au LLM de
regrouper : pour chaque thème fin, le LLM écrit un **profil** (un petit texte qui résume le
thème), ce profil est **ré-embeddé**, et on clusterise les profils. Le regroupement reste
géométrique ; le LLM ne sert qu'à décrire. Ça absorbe la redondance entre thèmes voisins.

**Stance** — la **position** d'un claim envers une proposition : `favorable` / `défavorable` /
`nuance`, avec un niveau de confiance. C'est le seul chemin qui donne des positions (cf. §3,
« les embeddings ne captent pas la position »).

**Clivage (cleavage)** — la **proposition débattable** dérivée pour chaque thème, contre
laquelle on mesure la stance. Un thème est dit *clivant* si l'opposition dépasse 15 %, sinon
*consensuel* — et on affiche la minorité sceptique au lieu de la lisser.

**Argument / V-SELECT** — les arguments affichés ne sont **pas rédigés** par le LLM : il
**sélectionne** les claims les plus représentatifs (donc du verbatim citoyen). Même logique que
l'invariant verbatim, appliquée un cran plus haut.

**Insight / synthèse** — le texte de synthèse par thème et global. Généré **bottom-up** : la
synthèse d'un parent agrège celles de ses enfants.

**Accroche / description / titre** — l'habillage éditorial d'un thème, généré par LLM et
**caché par contenu**. Le titre d'un macro est une « ombrelle » des titres de ses enfants
(sinon il dégénère en salade de mots-clés — bug corrigé récemment).

**Provenance / surlignage** — pour chaque avis, quelles portions ont nourri quel macro. C'est
ce qui permet de remonter d'une synthèse jusqu'à la phrase d'origine. La traçabilité est le
produit, pas un bonus.

**Contribution (submission)** — sur une consultation **ouverte**, ce qu'un citoyen dépose en
direct. En prod on stocke le texte seul ; l'analyse est construite plus tard.

---

## 3. Algorithmes et mesures

**Embedding** — transformer un texte en **vecteur de nombres** tel que deux textes de sens
proche donnent des vecteurs proches. Toute la géométrie du projet repose là-dessus.

**Embedder** — le modèle qui produit ces vecteurs (arctic-l aujourd'hui). ≠ le LLM qui rédige.

**Dimension (dim)** — la taille du vecteur. arctic = **1024**, nomic = **768**. ⚠️ Deux
embedders différents produisent des espaces **incomparables** : un cosinus entre un vecteur 768
et un 1024 n'a aucun sens (et plante). D'où le garde-dimension dans `correlate()`.

**Cosinus / L2-normalisé** — la mesure de proximité entre deux vecteurs (1 = identique, 0 = sans
rapport). *L2-normalisé* = tous les vecteurs ramenés à la longueur 1, ce qui rend le cosinus
égal à un simple produit scalaire (donc rapide).

**Seuil calibré** — ⚠️ un seuil de cosinus vaut **pour un embedder donné**. Les échelles
diffèrent d'un modèle à l'autre. Vécu : le passage nomic→arctic a laissé le seuil `0.68` en
place alors qu'il fallait `0.40` → la fonction « qui partage votre avis » a cessé de compter
quoi que ce soit.

**Anisotropie / recentrage (`centre`)** — les vecteurs d'un modèle ne sont pas répartis dans
toutes les directions : ils s'entassent dans un cône étroit (deux textes au hasard sont déjà à
0,59 de cosinus). On **soustrait le centroïde** pour réétaler. Zéro paramètre, gain mesuré
**+19 % d'ARI**.

**Hubness** — pathologie des espaces de grande dimension : quelques points deviennent le
« voisin » de presque tout le monde et polluent le clustering. Le recentrage la fait tomber de
3,5 à 0,9.

**kNN (k plus proches voisins) / graphe kNN** — on relie chaque claim à ses `k` voisins les plus
proches (`k = 30` ici). Ce graphe est l'entrée du clustering. *Exact* = on calcule vraiment
toutes les distances (via **faiss**, une bibliothèque de recherche vectorielle rapide).

**Leiden** — l'algorithme qui découpe le graphe en communautés (= nos thèmes). Rapide et stable.

**Modularité** — le score qui dit si un découpage est « bon » (beaucoup de liens dedans, peu
dehors). Leiden le maximise.

**Résolution γ (gamma)** — ⚠️ **le bouton de granularité**. γ haut → beaucoup de petits thèmes ;
γ bas → peu de gros thèmes. On sert `γ = 3.0` (fin), et l'abstraction remonte la structure
au-dessus. **Verdict important** : on pilotait autrefois la granularité via `k`, ce qui changeait
le *graphe* et dégénérait ; γ agit directement sur le découpage. D'où « γ, pas k ».

**Partition / membership** — le résultat du clustering : quel claim appartient à quel thème.

**c-TF-IDF** — méthode de nommage **sans LLM** : les mots caractéristiques d'un thème (fréquents
dedans, rares ailleurs). Rapide, sert de repli et alimente la vue live.

**ARI / NMI** — deux façons de mesurer si deux découpages **se ressemblent** (1 = identiques,
0 = sans rapport). Servent à comparer notre clustering à une vérité connue, ou une partition à
la précédente. ⚠️ **Résultat clé du projet** : NMI(clusters, position FAVOR/AGAINST) ≈ **0,04–0,06**
— *les embeddings captent le SUJET, pas la POSITION*. Deux personnes en désaccord frontal sont
voisines. C'est pour ça que la stance passe par le LLM.

**NPMI / silhouette / modularité** — mesures de **qualité interne** d'un clustering (les thèmes
sont-ils cohérents, bien séparés). ⚠️ Piège documenté : un modèle qui regroupe par **langue** a
d'excellentes métriques internes tout en étant **faux**. D'où le contrôle NMI(cluster, langue),
qui doit être **bas**.

**Stabilité (bootstrap)** — on relance sur des sous-échantillons : si le résultat change, il
n'était pas solide.

**UMAP / HDBSCAN** — alternatives testées (projection 2D / clustering par densité). UMAP sert
encore à l'affichage des points ; HDBSCAN a été écarté au banc.

**O(n²)** — notation de coût : quadratique. Doubler les données **quadruple** le temps. Le
calcul des voisins est en O(n²·d) — supportable en batch, rédhibitoire à chaque nouveau message
en live.

**Incrémental** — ne recalculer que ce qui a changé. L'extraction l'est déjà (cache par avis) ;
le graphe kNN et les embeddings de claims ne le sont pas.

---

## 4. Modèles et LLM

**LLM** — le modèle de langage (Mistral) qui extrait, titre, résume, classe. Payant à l'usage,
non déterministe. Tout ce qui peut être fait sans lui l'est sans lui.

**Rôles** — on n'utilise pas *un* LLM mais un par tâche : extraction, enrichissement,
abstraction, opinion, arguments, traduction. Chacun peut avoir un modèle différent (c'est
délibéré : `abstraction` et `arguments` sont volontairement sur un petit modèle).

**large / small / ministral-3b** — les tailles Mistral, du plus cher/fin au plus petit/rapide.
Aujourd'hui l'extraction tourne sur `mistral-large-latest`. Mesuré : **ministral-3b ≈ large**
pour l'extraction (Δ +0,02 sur 191 avis appariés) pour **~150× moins cher** — levier connu,
pas encore tiré.

**Token** — l'unité de facturation et de longueur des LLM (~¾ d'un mot). `max_tokens` = la
longueur maximale de réponse ; ⚠️ trop bas sur un lot, la réponse est **tronquée** et le résultat
part en vrille (nous est arrivé : 93 % de sortie inexploitable).

**Batch** — grouper plusieurs items dans un seul appel (10 claims par appel de stance). Moins
cher, mais impose d'augmenter `max_tokens` en proportion.

**429 / rate limit** — le code d'erreur « trop de requêtes ». ⚠️ C'est le **vrai facteur
limitant** de nos gros builds : le log nocturne montre un retry sur *chaque* dataset. Notre code
n'est pas le goulot, l'API l'est.

**Prompt système / nudge** — les consignes données au modèle. Un *nudge* est un ajout ciblé qui
corrige un défaut connu d'un modèle précis (le gros modèle **sous-découpe**, le petit
**sur-fragmente**).

**Juge (LLM-judge) / panel aveugle** — faire arbitrer deux sorties par un autre modèle (ou par
moi), **sans lui dire laquelle vient d'où**. L'aveugle est ce qui rend le verdict crédible.

**Comparaison appariée** — comparer deux modèles **sur les mêmes entrées**, pas sur deux
échantillons différents. Élimine la variabilité des données.

**Cascade / escalade** — stratégie de bench : tout passer au plus petit modèle, et ne
re-soumettre au gros que les cas qu'il a ratés. Si le petit y arrive, inutile de payer le gros.

**Licence de modèle** — ⚠️ contrainte juridique de 1er ordre. **Apache 2.0 / MIT** = utilisable
commercialement. **CC-BY-NC** = **non commercial** (cas de jina) → ne doit **jamais** être le
défaut du pipeline. Le registre d'embedders porte le champ `license` exprès.

**Révision épinglée** — on fige le commit exact d'un modèle téléchargé. Sans ça, une mise à jour
amont change le code exécuté chez nous sans prévenir.

---

## 5. Cache et infrastructure

**Cache** — ici, **pas** une optimisation : le **produit livré**. La prod ne calcule rien, elle
**sert** des résultats pré-calculés (`backend/cache/<dataset>/`).

**Cache HIT / MISS** — *hit* = le résultat existe déjà, zéro appel LLM. *miss* = il faut
recalculer (et payer). « Build 100 % HIT » est notre preuve d'iso-comportement.

**Clé de cache / signature** — ce qui identifie une entrée. ⚠️ **Règle** : la signature doit
contenir **tout ce qui change la sortie** (contenu + modèle + embedder + partition). Une
signature incomplète sert un résultat périmé en silence — la panne la plus vicieuse du projet.

**Invalidation** — quand la clé change, l'ancien résultat devient inutilisable. ⚠️ Un thème qui
gagne un seul claim change de contenu → titre, accroche, description et synthèse sont à
re-payer. C'est ce qui rend une restructuration coûteuse.

**Hash (sha256)** — l'empreinte d'un contenu, qui sert de clé. Même contenu → même empreinte.

**Idempotent** — relancer donne le même résultat sans re-payer. Nos builds le sont.

**meta.json / descripteur** — `meta.json` décrit un cache **construit** (combien d'avis, quel
modèle, quelles langues). Le **descripteur** (`pipeline/ingest/descriptors/*.json`) décrit une
consultation **à ingérer** (source, colonnes, question, statut). Les deux sont **déclaratifs** —
c'est ce qui permet le « zéro nom de corpus en dur ».

**Dev / prod** — ⚠️ séparation stricte. **Dev** (chez toi) a la clé Mistral et construit.
**Prod** (le runner) n'a **aucune clé** et ne fait que servir le cache. Promotion = copier un
cache validé de dev vers prod.

**`AGORA_PUBLIC=1`** — le mode public : lecture seule, endpoints d'écriture en 403,
aucun modèle chargé.

**Latence / wall-clock / débit** — *latence* = temps d'une opération ; *wall-clock* = temps réel
écoulé de bout en bout (≠ temps CPU) ; *débit* = quantité par seconde.

**Superlinéaire** — quand doubler l'entrée fait plus que doubler le temps.

---

## 6. Git, CI, agents

**Branche / PR / merge** — on travaille sur une **branche**, on ouvre une **PR** (demande de
fusion) vers `main`, la CI vérifie, puis on **merge**. ⚠️ Jamais de travail local non poussé :
le déploiement fait un `reset --hard` sur `main` et écraserait tout.

**CI / gate** — les tests automatiques (246 aujourd'hui). Le **gate** est la barrière : tant
qu'ils ne sont pas verts, rien ne part.

**Régression** — une fonctionnalité qui **marchait** et qui casse. Pire qu'un bug neuf : personne
ne la cherche.

**Refactor** — réorganiser sans changer le comportement (cf. iso-comportement).

**Worktree** — deux copies du dépôt sur des branches différentes, côte à côte, sans avoir à
changer de branche.

**`--force-with-lease`** — réécrire l'historique distant **en refusant** si quelqu'un a poussé
entre-temps. La version prudente du `--force`.

**Signed-off-by** — la signature de chaque commit, exigée par le CLA (l'accord de contribution).

**Lane** — un chantier parallèle avec son propre agent. **Pattern 12** : un agent **natif** pour
le travail mécanique et jetable (fan-out, vérifications) ; une lane **tmux** seulement quand un
humain doit **voir** ou **piloter** en direct, ou quand ça doit persister.

**Fan-out** — lancer plusieurs agents en parallèle sur des morceaux indépendants.

**Architecte** — mon rôle : cadrer, dispatcher, mesurer, garder les invariants. Pas taper tout
le code moi-même.
