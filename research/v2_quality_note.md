# Validation QUALITÉ renforcée des v2 — juge AVEUGLE, PANEL de 3

**Question (Bob).** Avant de payer une re-extraction complète, prouver le gain de qualité
des v2 plus solidement que « 2 flags corrigés + 12/15 fit ». On durcit la preuve sur trois
axes : **échantillon large et varié**, **juge aveugle** (anonymisé A/B, ordre randomisé),
et **panel de 3 juges** (majorité). Deux objets : **(A) extraction v2** (re-extraction =
coûteuse) et **(B) cible de clivage v2** (recompute dérivé, pas de re-extraction).

Scripts : `research/v2_extract_quality.py`, `research/v2_cleavage_quality.py`.
Caches : `research/v2_extract_cache/`, `research/v2_cleavage_cache/`.
Juge : `mistral-large-latest`, T=0.5 (diversité réelle du panel ; à T=0 les 3 juges sont
des clones). Ordre lot 1 / lot 2 (resp. A/B) **randomisé indépendamment par (item, juge)**
→ décorrèle le biais de position. Les deux bras passent le **gate verbatim dur**
(`align_spans` : tout claim non sous-chaîne exacte de l'avis est rejeté).

---

## A. Extraction v2 — gain ROBUSTE sur la sur-segmentation, SANS perte de thèmes

**Échantillon : 42 avis granddebat**, tirés au hasard (seed 42), **stratifiés par longueur**
(10 courts 30–120c · 10 moyens 120–350c · 10 longs 350–800c · 10 très-longs 800–2137c) +
les 2 avis flaggés. Longueurs 30–2137c, médiane 377c. Chaque avis extrait **v1** (snapshot
EXACT du prompt de prod d'avant le commit v2) **et v2** (`claim_sys(question)`), gate
verbatim aux deux.

**Volume & fidélité (sanity).** claims/avis : **v1 = 4.55 → v2 = 3.07** (−32 %). v2 ne
produit JAMAIS plus de claims qu'v1 (23 avis : moins ; 19 : identique ; 0 : plus). Verbatim
(claims gardés / specs émises) : v1 = 0.985, v2 = 0.992.

### Panel aveugle (majorité de 3), par critère SÉPARÉ

| Critère | v2 gagne | v1 gagne | tie | **v2 % (décidés)** | unanimité 3-0 |
|---|---|---|---|---|---|
| **sur-segmentation** | **18** | 6 | 18 | **75 %** (18/24) | 35/42 |
| **complétude** (pénalise la sur-fusion) | 2 | 3 | 37 | 40 % (2/5) | 32/42 |
| **fidélité** (verbatim) | 5 | 3 | 34 | 62 % (5/8) | 21/42 |

**Lecture.**
- **Sur-segmentation = le vrai gain, et il est ROBUSTE.** v2 l'emporte 18–6 (75 % des avis
  *décidés*), avec **35/42 verdicts unanimes** (panel très d'accord). C'est exactement la
  cible : v2 regroupe problème+solution et ne fragmente plus une idée. Corrélation
  réduction-de-claims ↔ victoire sur-segmentation = **0.48** : les gains viennent bien de la
  moindre fragmentation, pas d'un artefact.
- **Complétude = quasi-nulle de chaque côté (37/42 tie).** C'est le test du tradeoff
  redouté (« la sur-fusion perd-elle un thème ? ») : **non, presque jamais**. Même en
  restreignant aux **22 avis multi-thèmes** (≥4 claims dans un bras), la complétude reste
  **19/22 tie** (v2 = 2, v1 = 1). Le regroupement renforcé n'efface pas de thème réel.
- **Fidélité = parité** (34/42 tie). Attendu : les deux bras passent le gate verbatim dur ;
  le léger avantage v2 (5–3) est dans le bruit.

### Régression identifiée — sur-fusion sur une minorité

v2 **perd la sur-segmentation sur 6/42 avis** : ce sont des cas de **sur-FUSION** (v2
écrase des idées distinctes). Les plus nets :

| avis | v1→v2 claims | verdict | note du juge |
|---|---|---|---|
| `1-11667` | 6 → **1** | v1 | « lot 2 fusionne tout » |
| `1-9840` | 4 → **1** | v1 | « lot 2 fusionne portail web + reporting » |
| `1-22842` | 7 → 3 | v1 | idées liées trop agrégées |
| `1-2950`, `1-5131`, `1-29620` | 4→3, 6→5, 11→7 | v1 | fusion locale de 2 idées |

Sur 9 avis où v2 fusionne agressivement (n_v2 ≤ n_v1/2, n_v1≥4), **6 gardent la
sur-segmentation pour v2 et 6/9 ont une complétude *tie*** : la sur-fusion dégrade le
découpage perçu sans pour autant (le plus souvent) perdre un thème. **Coût net : ~14 % des
avis sur-fusionnés**, dont 2 cas sévères (6→1, 4→1). C'est le prix du regroupement ; il est
minoritaire et filtrable en aval (un claim « tout-en-un » est détectable par longueur /
multi-cible).

### Verdict A
v2 **améliore vraiment** (pas « juste moins de claims ») : gain franc et robuste sur la
sur-segmentation (75 % des décidés, panel unanime), **sans régression de complétude**
(37/42 tie ; 19/22 sur multi-thèmes), fidélité préservée. Régression réelle mais
minoritaire (sur-fusion sévère ~2/42). **→ Le gain JUSTIFIE la re-extraction.**

---

## B. Cible de clivage v2 — gain NON confirmé par le panel aveugle

**Échantillon : 19 feuilles granddebat** (les plus grosses avec ≥12 claims ; top-25 demandé,
19 qualifiées). Pour chacune : cible v1 (prompt prod « la plus saillante », sans titre) vs
v2 (conditionnée sur le titre + « central »). **fit = cos(emb(cible), emb(titre))** —
le fit cible↔TITRE retenu par la R&D (le fit-vs-centroïde étant connu cassé). Panel aveugle
de 3 : *« laquelle capture le mieux le débat CENTRAL du thème "<titre>" ? »*, A/B randomisé.

### Résultats

| métrique | valeur |
|---|---|
| panel : **v2 gagne / v1 gagne / tie** | **6 / 6 / 7** |
| v2 % parmi les **décidés** | **50 %** (6/12) |
| propositions v1==v2 (chaîne identique) | 6/19 |
| fit moyen v1 / v2 | 0.785 / 0.802 (+0.017) |
| fit v2 > fit v1 | 10/19 |
| **concordance fit ↔ panel** | **0.58** |
| **corr( fit_v2−fit_v1 , panel choisit v2 )** | **0.51** |
| fit moyen du gagnant / perdant (décidés) | 0.808 / 0.791 |

**Lecture.**
- Sous panel aveugle, **v2 n'est PAS meilleure que v1 : 6–6 (50 % des décidés)**. Le « 12/15 »
  d'origine était le **fit auto-déclaré**, pas un jugement humain — circulaire. Évalué à
  l'aveugle, l'avantage disparaît.
- **6/19 propositions sont identiques** entre v1 et v2 (tie mécanique). Sur les 13 où v2
  reformule, c'est encore 6 v2 / 6 v1 / 1 tie : v2 **change surtout le libellé**, rarement
  la substance. Ex. `n0` (le cas vitrine d'origine) : v1 « rendre les élus plus
  représentatifs et proches des citoyens » vs v2 « réformer la représentation politique pour
  plus de proximité » → quasi-paraphrases, **panel choisit v1**.
- **Le `cleavage_fit` est un proxy FAIBLE de la qualité** : concordance 0.58 (à peine
  au-dessus du hasard), corrélation 0.51. Le gagnant a un fit à peine supérieur (0.808 vs
  0.791). Le fit↔titre prédit « un peu » mais ne doit pas servir de juge de qualité.

### Verdict B
Le gain de clivage v2 **ne résiste pas au juge aveugle** (égalité 6–6) ; le `cleavage_fit`
prédit mal le jugement. **MAIS** : la cible de clivage est un champ **dérivé au bake/opinion,
sans re-extraction** — l'adopter ne coûte rien et ne nuit pas (fit moyen ≥, jamais de
régression franche). À garder comme amélioration neutre/cosmétique, **pas** à compter comme
justification de rebuild.

---

## Conclusion — le rebuild est-il justifié ?

**Oui, mais pour UNE raison : l'extraction v2.**

- **Extraction v2** porte un gain de qualité **réel, robuste et mesuré à l'aveugle** :
  sur-segmentation corrigée (75 % des décidés, 35/42 panel unanime) **sans perte de thèmes**
  (complétude 37/42 tie, 19/22 sur multi-thèmes), fidélité intacte. Régression minoritaire
  (sur-fusion sévère ~2/42, filtrable). C'est la partie **coûteuse** (re-extraction de tout
  le corpus) — et c'est elle qui est justifiée. **→ GO re-extraction.**
- **Clivage v2** n'apporte **pas** de gain de qualité démontrable (panel 6–6) ; mais comme
  c'est un recompute dérivé **gratuit**, on l'embarque sans le facturer au rebuild. Ne pas
  s'appuyer sur `cleavage_fit` comme métrique de qualité (proxy faible, concordance 0.58).

**Robustesse de la preuve.** Échantillon large (42 avis stratifiés / 19 clusters), juge
aveugle anonymisé, ordre randomisé par juge, panel de 3 avec forte unanimité (extraction :
35/42 sur le critère clé), critères séparés, régressions explicitement comptées. Le verdict
extraction est solide ; le verdict clivage est un **down-grade honnête** du « 12/15 »
antérieur.
