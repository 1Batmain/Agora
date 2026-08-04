# Faisabilité LIVE — le pipeline peut-il tourner en continu et se restructurer ?

**Question (Bob, 2026-08-04)** : peut-on faire tourner Agora en LIVE, supporter des
restructurations de fond, et de façon scalable ? Cas d'usage projeté : les **transcripts de
l'Assemblée nationale** — modéliser les positions des orateurs et synthétiser leurs positions
sur le sujet débattu, en direct.

**Verdict court** : **l'ingestion live est à portée, la restructuration live ne l'est pas.**
Le blocage n'est pas le clustering (rapide) — c'est l'**enrichissement LLM**, qui est caché
par CONTENU de nœud : toute restructuration invalide les caches des nœuds qui ont bougé.
Et surtout : **modéliser des positions ne peut PAS passer par les embeddings** (mesuré chez
nous, cf. plus bas) — ça doit passer par `build_opinion`, qui est la partie chère.

---

## 1. Ce que le code fait déjà — l'architecture est DÉJÀ à deux vitesses

| Chemin | LLM ? | Latence | Ce qu'il produit |
|---|:--:|---|---|
| `backend/live_cluster.py` | **non** | **< 2 s** | thèmes re-clusterisés au seuil kNN, nommage c-TF-IDF, points UMAP |
| `backend/build_analysis.py` | oui | **minutes → heures** | titres/accroches/descriptions/insights, provenance verbatim, macros |
| `backend/build_opinion.py` | oui | minutes | proposition clivante par thème + stance de chaque claim |

La séparation « carte instantanée sans LLM » / « analyse rédigée avec LLM » existe déjà et
est le bon squelette pour du live. Rien à inventer là-dessus.

## 2. Coût réel d'une restructuration de fond — mesuré

Source : `var/nightly-20260721.log` (rebuild arctic complet, **extraction déjà cachée** — donc
c'est exactement le coût d'une restructuration à contenu constant).

| Dataset | avis | claims | analysis | opinion | total |
|---|--:|--:|--:|--:|--:|
| lutte-fausses-informations | 195 | 292 | ~2,5 min | 2 min | **~7 min** |
| tiktok | 1 674 | 2 419 | ~12 min | 10 min | **~22 min** |
| republique-numerique | 2 724 | 3 887 | ~37 min | 17 min | **~44 min** |
| xstance | 3 000 | — | ~17 min | — | — |
| granddebat | 22 174 | 36 275 | — | — | **~4 h** |

Lecture : le coût est **superlinéaire en pratique**, pas à cause de l'algo mais des **429
Mistral** — le log montre un retry sur *chaque* dataset (jusqu'à 2 sur republique). Le
facteur limitant d'un rebuild complet est le **rate-limit de l'API**, pas notre code.

→ Une restructuration de fond, aujourd'hui, c'est **des dizaines de minutes**. Le live, c'est
des secondes. Deux ordres de grandeur d'écart : la restructuration ne peut pas être continue,
elle doit être **déclenchée**.

## 3. Les 4 verrous techniques concrets (tous localisés, tous corrigeables)

1. **`_emb_fingerprint` est tout-ou-rien** (`backend/claims_endpoint.py:120`) : l'empreinte du
   cache d'embeddings est un sha256 de la **concaténation de TOUS les textes de claims**. Un
   seul nouveau claim → cache manqué → **ré-embedding intégral du corpus**. C'est LE verrou
   n°1 de l'incrémental, et le plus facile à lever (clé par claim au lieu d'une empreinte
   globale).
2. **`knn_search` est exact et global** (`pipeline/cluster/knn.py`, `faiss.IndexFlatIP`) :
   O(n²·d) recalculé de zéro à chaque passe. Acceptable en batch (36 k claims passent), mais
   il n'y a **aucun ajout incrémental de point** au graphe.
3. **L'identité des thèmes n'est pas stable** : `flat_partition` relance Leiden globalement ;
   rien ne garantit qu'un thème garde son id d'une passe à l'autre. En live, la carte se
   **remélangerait sous les yeux de l'utilisateur**. C'est le seul vrai point de R&D de la
   liste — nous n'avons rien dans le dépôt qui réponde à « apparier les thèmes d'avant et
   d'après une repartition ».
4. **Les caches d'enrichissement sont clés par contenu de nœud** (`backend/titles.py:131`,
   `_macro_key:180`) : dès qu'un nœud gagne/perd des claims, son hash change → titre, accroche,
   description et insight se re-paient au LLM. C'est **voulu et correct** (c'est ce qui rend le
   rebuild idempotent), mais ça signifie que le coût d'une restructuration est proportionnel au
   **nombre de nœuds qui ont bougé**, pas au nombre de nouveaux avis.

**Bonne nouvelle** : l'extraction, elle, est **déjà incrémentale**. `claims.json` est un dict
**clé = id d'avis** (`{"tiktok:101": [...]}`, `claims_endpoint.py:92`) : un nouveau tour de
parole ne coûte que sa propre extraction. Ratio mesuré : **1,45 claim par avis**.

## 4. Le coût par tour de parole — c'est là que c'est encourageant

Depuis les mesures ci-dessus : la stance de tiktok = 2 419 claims en 9,7 min ≈ **4 claims/s**
(batché par 10). Un tour de parole ≈ 1,5 claim ⇒ **extraction + stance d'une intervention
tiennent en quelques secondes d'API**. Un député parle toutes les quelques minutes.

→ **Suivre un débat au fil de l'eau est largement dans le budget.** Ce qui ne l'est pas, c'est
de *re-dessiner toute la carte* à chaque intervention.

## 5. Le point dur, et il n'est PAS de performance : positions ≠ thèmes

C'est le risque principal de la demande, et notre propre banc l'a déjà tranché :

> Banc Stance (`research/README.md`, x-stance FR, labels FAVOR/AGAINST) : Leiden **et**
> UMAP+HDBSCAN recouvrent mal le clivage, **NMI ≈ 0,04–0,06**. *« L'embedding capte le THÈME,
> pas la POSITION. »*

Donc : **on ne peut pas clusteriser les gens par position.** Deux députés en désaccord frontal
sur le même sujet sont *voisins* dans l'espace d'embedding. Modéliser les positions doit passer
par le chemin LLM `build_opinion` (proposition clivante par thème → stance par claim, avec
niveau de confiance) — qui existe, marche, et coûte. À noter aussi (mémoire projet) : la
**cible de stance ne transfère pas hors tiktok** — la cible devra être re-validée sur du
transcript.

## 6. L'écart au modèle Agora — 3 différences, additives, pas un rewrite

| | Agora aujourd'hui | Transcripts AN |
|---|---|---|
| unité | 1 avis = 1 citoyen, indépendant | 1 tour de parole ; **un orateur parle 20 fois** |
| personne | `author_hash` **salé, à sens unique**, jamais agrégé en aval | orateur **nommé** (personnage public, débat public) |
| temps | `ts` porté, jamais utilisé | axe de **1er ordre** (qui a dit quoi, quand, en réponse à qui) |

L'axe personne **existe structurellement** (`Idea.author_hash`, `Idea.ts` —
`pipeline/cluster/io.py:31-33`) mais un grep confirme qu'il n'est **utilisé nulle part en aval
de l'ingestion**. Il n'y a donc *aucune* agrégation par locuteur dans le produit. C'est un axe
à construire (orateur × thème → stance), pas à réparer.

⚠️ Inversion de conception à assumer : toute la chaîne est **anonymisante par construction**
(`correlate()` ne renvoie jamais le verbatim d'autrui, `author_hash` est irréversible). Nommer
les orateurs est légitime ici (registre public) mais **prend le contre-pied d'un invariant de
vie privée** posé partout ailleurs. À décider explicitement, pas à laisser glisser.

Enfin, un débat est **dialogique** : interruptions, procédure, réponses à un orateur précédent,
questions rhétoriques. L'extraction est **extractive** (verbatim ancré) — ça, ça survit. Mais
l'hypothèse « 1 avis = 1 opinion autonome » ne tient plus.

---

## 7. Recommandation — architecture à 3 étages, et le spike qui tranche

**Étage 0 — assignation (ms, zéro LLM)** : nouveau tour → embed local → **plus proche
centroïde parmi les thèmes EXISTANTS**. Pas de re-clustering, la carte ne bouge pas. Compteurs
mis à jour en direct.
**Étage 1 — position (secondes, LLM cheap, async)** : extraction des claims du tour + stance
contre les propositions clivantes **déjà dérivées**. C'est ce qui alimente « qui pense quoi ».
**Étage 2 — restructuration (minutes, déclenchée)** : re-partition + ré-enrichissement complet,
quand la **dérive** dépasse un seuil. Jamais en continu.

Ce découpage rend la question de Bob **mesurable** et la réduit à deux inconnues :

- **(A) Quand faut-il restructurer ?** Il faut un **indicateur de dérive** — p. ex. la part des
  nouveaux claims dont le cosinus au meilleur centroïde tombe sous le seuil dérivé. Mesurable
  en rejouant un corpus existant par tranches.
- **(B) Comment garder l'identité des thèmes à travers une restructuration ?** Sans appariement
  avant/après, l'UI se remélange. **C'est le vrai risque de la démarche** — et c'est celui à
  lever en premier.

### Spike proposé (à groomer, non lancé)

**Simuler le live par rejeu**, sans aucune infra nouvelle : prendre un dataset déjà bâti,
l'injecter par tranches de N avis dans l'ordre `ts`, et mesurer à chaque tranche —
1. **stabilité d'identité** : ARI entre partition_t et partition_{t-1} restreinte aux avis
   communs (est-ce que la carte tient ?) ;
2. **dérive** : part des nouveaux claims non couverts par un centroïde existant ;
3. **coût réel** : nœuds dont le hash de contenu change ⇒ appels LLM à re-payer.

Critère d'acceptation, dans l'esprit de la guideline de robustesse : **≥ 2 datasets, N large,
jamais un verdict sur une tranche unique**. Le witness de contrôle est gratuit ici — rejouer
dans l'ordre `ts` **vs** dans un ordre aléatoire : si les deux donnent la même dérive, c'est que
la mesure ne capte pas l'arrivée d'information, seulement du bruit d'échantillonnage.

Coût : zéro appel LLM pour (1) et (2) — tout se calcule sur les embeddings et les hashes déjà
cachés. C'est-à-dire qu'**on peut répondre à la question la plus dure du dossier sans dépenser
un euro d'API**, avant d'engager quoi que ce soit sur les transcripts AN.
