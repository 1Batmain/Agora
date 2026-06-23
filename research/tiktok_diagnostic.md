# Diagnostic — « pourquoi *tiktok* domine » (consultation TikTok / FR)

**Lane eval · analyse READ-ONLY · branche `work/diag`.**
Reproductible : `uv run --extra embed-contender python -m eval.tiktok_diag`
(numéros bruts dans `eval/tiktok_diag_results.json`, avis dans
`eval/tiktok_diag_two_clusters.md`).

## TL;DR — verdict

Le mot **tiktok** (et `tik`/`tok`/`appli`/`application`/`réseau·x`) abîme le rendu
par **DEUX** canaux, d'ampleur inégale :

| Canal | Ampleur mesurée | Réversibilité |
|---|---|---|
| **NOMMAGE** (le terme sature les labels) | **34 %** des sous-thèmes (16/47) et **50 %** des macros (4/8) ont un label « tiktok » | **Totale** : 34 %→**0 %** avec le fix (validé) |
| **GÉOMÉTRIE** (le terme tire l'espace d'embedding) | ablation ⇒ **ARI ≈ 0,21–0,27** ; **~50 %** des voisinages locaux changent | Partielle, coûteuse |

**Le nommage DOMINE le symptôme visible et se corrige à 100 % pour un coût nul.**
La géométrie est un effet **réel mais secondaire** : la composante commune
« tiktok-ité » remanie ~la moitié de la structure locale, ce qui *plafonne* la
qualité et provoque du sur-découpage, mais **ne crée pas** le « tout est tiktok »
qu'on voit à l'écran (les sous-thèmes sous-jacents restent cohérents et
distincts, cf. exp. 5).

**Reco prioritaire (nommage, à faire) :** ajouter la famille tiktok aux
**stopwords de domaine** + passer le naming en **c-TF-IDF** contrastif.
**Reco secondaire (géométrie, optionnelle) :** **all-but-the-top** (retrait des
1–2 PCs sur les vecteurs *cachés*, re-norm + re-calibrage du seuil) — préférable
au masquage pré-embed qui exige un ré-embed (le cache est tout l'intérêt du
serveur live).

---

## Données & baseline

- Subset = cache backend nomic-v2 (`backend/cache/`) = **consultation TikTok/FR**
  intégrale, 1621 avis (source/lang 100 % tiktok/fr — vérifié).
- Baseline = **défauts de prod** (contrat console `:8010`) : `min_chars=12` →
  1604, `dedup=0.95` → **1597 avis**, `k=12`, `threshold=0.60`,
  `resolution_macro=1.0`, `resolution_sub=1.5`, `min_sub_size=18`, `seed=42`.
- Clustering baseline : **8 macros → 47 sous-thèmes** (réutilise
  `pipeline.cluster.{knn,hierarchy,scoring,naming}`, vecteurs cachés).

---

## Exp. 1 — Saturation lexicale

| Mesure | Valeur | Lecture |
|---|---|---|
| % docs contenant la **famille** tiktok | **43,4 %** | présent dans ~1 avis sur 2 |
| dont `tiktok` | 32,7 % · `appli/application` 11,3 % · `réseau·x` 11,2 % · `tok` 9,1 % · `tik` 9,0 % | |
| rang de `tiktok` parmi **tous** les tokens | **#20** (519 occ.) ; `le` est #11 | dans la bande des stopwords |
| rang parmi les **mots de contenu** | **#1** (519), devant `mal` 468, `temps` 386, `vidéos` 246 | **`tiktok` est le mot plein le + fréquent du corpus** |

> **`tiktok` est un mot-vide de domaine** : par fréquence il rivalise avec les
> articles (`le`, `la`) et écrase tous les vrais mots-thèmes. Un terme aussi
> ubiquitaire ne *distingue* rien — d'où sa nocivité pour le nommage.

## Exp. 2 — Composante commune (géométrie)

| Mesure | Valeur | Lecture |
|---|---|---|
| cos moyen au **centroïde global** | **0,761** (σ=0,050) | espace **très comprimé/anisotrope** (un corpus bien étalé serait ~0,1–0,3) |
| cos moyen inter-paires | 0,579 | tout est proche de tout |
| variance expliquée **PC1..PC5** | 0,107 / 0,062 / 0,040 / 0,032 / 0,026 (cum. 0,266) | PC1 capte une direction dominante |
| **corr(PC1, mentionne tiktok)** | **−0,813** | **PC1 ≈ axe « tiktok-ité »** |
| corr(PC1, longueur du doc) | −0,512 | PC1 mêle aussi la longueur (confondu) |
| corr(PC2.., tiktok) | ≤ 0,10 | l'effet tiktok est concentré sur PC1 |

> Oui : il existe une **direction commune** de l'espace fortement corrélée au
> simple fait de mentionner tiktok (et, secondairement, à la longueur). Les
> embeddings ne sont pas isotropes — ils partagent un gros « fond tiktok ».

## Exp. 3 — Influence du token sur la GÉOMÉTRIE (ablation)

Set de nœuds **figé** (post min_chars+dedup) pour que les ARI soient bien définis.
Métrique **threshold-free** ajoutée : *neighbor-Jaccard* = recouvrement moyen des
k-PPV de chaque nœud entre l'espace baseline et l'espace ablaté (1 = géométrie
locale inchangée, 0 = remaniée).

| Variante | ARI feuilles | ARI macros | neighbor-Jaccard | Lecture |
|---|---|---|---|---|
| **(a) masquage pré-embed** (ré-embed nomic-v2) | **0,269** | 0,272 | **0,486** | retirer le token *avant* l'embed remanie ~la moitié des voisinages et ne garde que 27 % d'accord de clusters |
| **(b) all-but-the-top (PC1)**, densité appariée | 0,255 | 0,283 | 0,403 | retirer la direction commune ⇒ effet ~équivalent au masquage |
| all-but-the-top (PC1+PC2) | 0,212 | 0,223 | 0,362 | |
| all-but-the-top (PC1..PC3) | 0,205 | 0,205 | 0,330 | |

**Précision méthodo (honnêteté).** Au seuil de prod (0.60), all-but-the-top
**effondre** le graphe (ARI≈0,01, ~1500 singletons) : retirer la composante
commune *dé-comprime* l'espace (cos-centroïde 0,76→0,015), tous les cosinus
passent sous 0,60 → plus d'arêtes. Ce n'est **pas** un résultat de topologie mais
un artefact d'échelle. Les ARI ci-dessus sont donc mesurés à **densité appariée**
(seuil re-calibré pour `avg_degree≈16` sur baseline et variante). Le
neighbor-Jaccard, lui, est indépendant du seuil et **confirme** le même ordre de
grandeur.

> **Le token DIRIGE une part substantielle de la géométrie** (pas seulement le
> label) : deux ablations indépendantes (masquage *vs* retrait de PC) convergent
> vers **ARI ≈ 0,21–0,27** et **~40–50 % des voisinages remaniés**. Mais l'effet
> n'est **pas total** : ~25 % de la structure de clusters et ~50 % des
> voisinages **survivent** → la thématique réelle reste présente sous la couche
> commune.
>
> *Effet de bord révélateur* : appariée à la densité, la baseline elle-même
> éclate (8→81 macros). Les grandes communautés tiennent en partie grâce aux
> arêtes denses créées par la composante commune — fragiles dès qu'on coupe.

## Exp. 4 — Isolation du NOMMAGE : TF-IDF vs c-TF-IDF

Sur les **mêmes** clusters baseline, taux de labels contenant un terme famille
tiktok (top-3 mots) :

| Niveau | TF-IDF (prod) | c-TF-IDF | **c-TF-IDF + domain-stop (FIX)** |
|---|---|---|---|
| sous-thèmes | **16/47 (34 %)** | 18/47 (38 %) | **0/47 (0 %)** |
| macros | **4/8 (50 %)** | 4/8 (50 %) | **0/8 (0 %)** |

> **Surprise importante : c-TF-IDF SEUL ne suffit pas** (il *empire* même un peu).
> Raison : le TF-IDF inter-clusters de prod utilise un IDF *lissé* (plancher
> idf≥1), il ne *zéro-ifie* jamais un terme omniprésent ; et l'idf c-TF-IDF
> (log(1+A/f_t)) ne pénalise pas assez tiktok vu sa fréquence par classe énorme.
> **Le vrai levier de nommage = traiter la famille tiktok en stopword de
> domaine.** c-TF-IDF reste utile *en complément* : il fait remonter le **terme
> distinctif** (cf. exemples ci-dessous). Combinés → **0 %** et labels parlants :

| cluster | label TF-IDF (prod) | → **label fix (c-TF-IDF + domain-stop)** |
|---|---|---|
| 19 | `faux compte · divulgation · live` | **harcèlement · personne · personnes** |
| 11 | `estime soi · estime · baisse estime` | **soi · dépression · estime** |
| 29 | `tiktok · dopamine · scroller` | **scroller · temps · vie** |
| 44 | `tiktok · compte tiktok · test` | **enfants · contenus · dangereux** |
| 46 | `madame · monsieur · parental` | **enfants · sociaux · parents** |

**Généralisation (sans liste à la main).** Une règle « terme présent dans ≥90 %
des *clusters* » n'attrape **que** `mal` (44/47), **pas** `tiktok` (présent dans
seulement 18/47 clusters mais avec une TF colossale). Le bon signal automatique
est au niveau **document** : `tiktok` est dans **32,7 %** des avis → un cutoff
`max_df ≈ 0,25–0,30` sur le corpus d'avis le démote tout seul (per-consultation,
pas de liste figée). À coupler avec c-TF-IDF.

## Exp. 5 — Les clusters « tiktok » : nommage ou géométrie ?

**13** sous-thèmes ont pour top-terme TF-IDF un mot de la famille tiktok.
Consensus médian de *tous* les clusters = 0,696. (Avis complets :
`eval/tiktok_diag_two_clusters.md`.)

Leur consensus (0,68–0,76) est **dans la norme** — ils ne sont **pas** moins
cohérents que les autres. Et leur terme **distinctif** (c-TF-IDF, hors tiktok)
révèle des **sujets réellement différents** :

| cluster | n | consensus | sujet réel (révélé par c-TF-IDF) |
|---|---|---|---|
| 43 | 63 | 0,68 | **harcèlement scolaire** (collège, filles, harceler) |
| 44 | 46 | 0,68 | **protection des enfants** (contenus dangereux) |
| 28 | 38 | 0,74 | **TCA / image du corps** (skinny, tca) |
| 29 | 47 | 0,71 | **addiction / temps perdu** (scroller, dopamine) |
| 48 | 39 | 0,70 | **contenu anxiogène** (triste, propagande) |
| 25 | 46 | 0,73 | **désinstallation / report sur Instagram** |

> **Deux cas se cachent derrière un même label « tiktok » :**
> 1. **Sujets distincts mal nommés** (43, 44, 28, 48, 25…) : clusters cohérents
>    sur de vrais thèmes, rendus *indiscernables* par le label → **problème de
>    NOMMAGE pur**. C'est la majorité.
> 2. **Même thème sur-découpé** (12 / 27 / 32 / 53 et en partie 26 : tous
>    « culpabilité du temps passé sur l'appli ») : un thème unique éclaté en
>    plusieurs feuilles quasi-jumelles → là, la **GÉOMÉTRIE** (fond commun +
>    granularité) contribue, en plus du nommage.

---

## Verdict chiffré

- **Nommage = cause dominante du symptôme visible.** 34 % des sous-thèmes / 50 %
  des macros portent un label tiktok ; or les clusters sous-jacents sont
  cohérents et thématiquement distincts (exp. 5). Correction **complète et
  gratuite** : 34 %→0 % (exp. 4, validé).
- **Géométrie = cause réelle mais secondaire.** Espace fortement anisotrope
  (cos-centroïde 0,76), PC1 = axe « tiktok-ité » (corr −0,81). Ablation ⇒ ARI
  0,21–0,27, ~50 % des voisinages remaniés. Effet net : qualité plafonnée et
  **sur-découpage** de thèmes jumeaux — mais la thématique survit (≥25 % de
  structure, ≥50 % des voisins conservés).

**Ordre de grandeur : le nommage explique l'essentiel de ce qu'on *voit*
(100 % corrigeable) ; la géométrie explique une dégradation de fond de l'ordre de
~50 % de la structure locale, partiellement corrigeable.**

## Recommandations

1. **PRIORITÉ — nommage (lane nlp, `pipeline/cluster/naming.py`).** Risque nul,
   aucune touche à la géométrie :
   - ajouter la **famille tiktok** aux stopwords *ou* (mieux, générique) un
     **`max_df≈0,25–0,30`** calculé sur le corpus d'avis de la consultation ;
   - passer le scoring de label en **c-TF-IDF** contrastif (révèle le terme
     distinctif). Gain mesuré : **34 %→0 %** de labels pollués, thèmes lisibles.
2. **SECONDAIRE — géométrie (optionnel, si on veut + de qualité/moins de
   doublons).** **all-but-the-top** sur les vecteurs **cachés** : retirer 1–2 PCs,
   re-normaliser, **re-calibrer `threshold`** (l'espace se dé-comprime → 0,60
   devient trop haut ; viser un `avg_degree` cible ~prod). À **préférer au
   masquage pré-embed** : même effet (ARI ~0,26 vs 0,27) mais sans ré-embed —
   compatible avec l'architecture « cache figé » du serveur `:8010`.
   ⚠️ À valider au banc qualité (`eval/quality_*`) avant prod : l'ablation
   réorganise les clusters, à ne faire que si la cohérence/silhouette s'améliore.

## Honnêteté — taille d'échantillon & limites

- **n = 1597**, **une seule** consultation, **FR**, **un seul** sujet (TikTok).
  Rien ici ne se généralise tel quel à d'autres consultations/langues.
- **Confusion PC1** : corrélé tiktok (−0,81) **et** longueur (−0,51) — la
  « direction commune » mêle thème de domaine et verbosité ; on ne peut pas
  attribuer PC1 au seul mot tiktok.
- **all-but-the-top** dépend d'un **re-calibrage de seuil** (dé-compression) ;
  les ARI sont à densité appariée — choix défendable mais paramétrique. Le
  neighbor-Jaccard (sans seuil) sert de garde-fou et confirme.
- **ARI** mesure le *déplacement* de partition, pas la *qualité* : un gros
  déplacement n'est « bon » que s'il améliore le banc qualité (à tester).
- La **liste de stopwords de domaine** est corpus-spécifique ; d'où la
  préférence pour le `max_df` automatique, plus robuste au changement de
  consultation.
- `dedup`/`min_chars` figés aux défauts prod ; un autre réglage déplacerait
  légèrement les comptes (pas le sens des conclusions).
