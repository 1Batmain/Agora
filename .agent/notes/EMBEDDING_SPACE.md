# Verdict — l'espace d'embedding : recentrer, ne pas blanchir (2026-07-09)

**Question.** Le nuage d'embeddings est-il exploitable tel quel ? Les cosinus entre deux
claims pris au hasard valent 0,591 — pas 0. Que perd-on, et que gagne-t-on à le corriger ?

**Déclencheur.** L'anisotropie a été invoquée trois fois dans la même journée comme cause
d'échec : nommage par centroïde, garde-fou du coarsening, détection de redondance. Chaque
fois traitée comme une fatalité. Bob : « et si on zoomait sur le quart de cercle ? »

---

## Verdicts

| | verdict | pourquoi |
|---|---|---|
| **Recentrer** (`v ← (v−μ)/‖v−μ‖`) | **OUI** | +16 à +34 % d'ARI sur gold externe ; zéro paramètre ; aucune sur-fusion sur 3 corpus ; feuilles orphelines éliminées |
| **Blanchir** (`all-but-top`) | **NON** | aucun gain d'ARI au-delà du recentrage, NMI en BAISSE, et ça ajoute un knob `k` |
| **Optimiser la hubness** | **NON** | bon diagnostic, mauvais objectif : son minimum (`k=3`) correspond à un NMI dégradé |

---

## 1. L'anisotropie appartient au MODÈLE, pas au corpus

Hypothèse initiale (« tiktok est mono-sujet, donc TikTok devient la direction commune ») :
**réfutée**. granddebat — des centaines de sujets — est PLUS anisotrope que tiktok.

```
espace BRUT           cos_moy   |centroïde|   PC1    dim_eff   hubness
tiktok                 0.591       0.769      8.9%    53/768     2.54
granddebat             0.605       0.778      6.4%    72/768     2.41
republique-numerique   0.517       0.719      5.6%    85/768     3.85
x-stance               0.534       0.731      3.6%   102/768     2.73
(référence isotrope)   0.000       0.000      0.13%  768/768     ~0
```

**Cause.** L'entraînement contrastif contraint des *différences* de similarité. Ajouter un
même vecteur à tous les embeddings laisse ces différences quasi inchangées → **le gradient
qui pousserait à centrer n'existe presque pas**. Le modèle est « expressif » selon son
propre critère tout en produisant un nuage pathologique pour un graphe de voisinage.
Ce n'est pas propre à nomic : tout modèle entraîné par contraste cosinus a cette échappatoire.
(Littérature : Ethayarajh 2019 ; Gao et al. « representation degeneration » ; Mu & Viswanath
« all-but-the-top ».)

## 2. Le mécanisme du dommage : ce n'est pas le décalage, c'est sa VARIANCE

```
alignement des points sur la direction commune : μ = 0.768   σ = 0.047
```
Si σ valait 0, le fond commun serait **inoffensif** : cosinus = constante + similarité
résiduelle, rangs préservés, aucun hub. C'est la variabilité d'un point à l'autre qui nuit.

```
corrélation(alignement, nb de fois voisin) :  brut +0.686   →  centré +0.100
les 100 points les PLUS alignés  : voisins 36.1 fois   (attendu : 10)
les 100 points les MOINS alignés : voisins  1.4 fois
après recentrage                 :        12.3  vs  9.1
```
Être aligné sur la direction commune suffisait à devenir le voisin de tout le monde.
→ **le k-NN n'était pas immunisé** ; les rangs eux-mêmes étaient faussés.

## 3. Validation EXTERNE — x-stance porte 12 topics annotés à la main

Vérité terrain gratuite, produite hors du projet. 4274 claims, Leiden vs topics réels.
15 exécutions (3 résolutions × 5 graines) ; **les plages ne se chevauchent jamais** :

```
resolution 0.8 : ARI 0.153 ± 0.010  →  0.205 ± 0.005   (+34 %)
resolution 1.0 : ARI 0.179 ± 0.009  →  0.213 ± 0.007   (+19 %)
resolution 1.2 : ARI 0.187 ± 0.003  →  0.218 ± 0.005   (+16 %)
```
Blanchiment (`all-but-top`) : ARI plat, NMI 0.374 → 0.352 (k=3) → 0.348 (k=10).

## 4. Témoins sans gold — aucune sur-fusion, orphelines éliminées

```
                macros themes feuil prof  p50av  orph  ratio_max  >0.30  coarsen  dissous
tiktok  brut      16     41     35    2     60     0     0.705      20    11→11      2
        centré    17     53     44    2   47.5     0     0.601      11    11→6       3
granddebat brut   19     54     50    1    506    20     0.262       0    37→21      2
           centré 19     35     30    2   1039     0     0.245       0    24→9       3
rép-num  brut     18     44     39    1     94     0     0.944      13    14→14      1
         centré   17     55     42    2     76     0     0.932      13    17→7       3
```
- **Aucune sur-fusion** : granddebat 19→19 macros, rép-num 18→17.
- **Orphelines** (feuille portée par <5 citoyens) : granddebat 20 → **0**.
- **Le coarsening se réveille** : `11→6`, `24→9`, `17→7` — il ne fusionnait RIEN dans
  l'espace brut sur 2 corpus sur 3.

## 5. Ce que le recentrage NE répare PAS

`dissous = 3` sur les trois corpus : **`sauce_magique` défait systématiquement le travail
du coarsening**. Il regroupe 24 racines en 9 sur granddebat ; la re-coupe en refabrique 19.
D'où un `ratio_max` qui reste à 0.601 (tiktok) et **0.932** (rép-num) — pour un seuil de
séparabilité réelle à ~0.15.

→ **La géométrie était fausse ; l'objectif l'est aussi.** T-N8 (re-coupe) n'est pas une
amélioration optionnelle : c'est le défaut dominant, désormais isolé sur 3 corpus, avec le
coarsening hors de cause.

## 6. Limites honnêtes

- **Un seul gold** (x-stance), multilingue, 12 topics grossiers. Leiden en trouve 17–19 :
  l'ARI absolu (0.21) mesure une AMÉLIORATION RELATIVE, pas une qualité absolue.
  `+34 %` ne veut pas dire « le clustering est bon ».
- granddebat passe de 54 à 35 thèmes, feuilles à **1039 avis médians**. Trente feuilles pour
  22 174 contributions est peut-être trop grossier. **Aucune de nos métriques ne le dit.**
  Ce chiffre relève du panel aveugle sur le rendu, pas d'un tableau.

## 7. Câblage — points d'attention (NON fait)

1. Centrer **à la lecture**, jamais sur disque : l'empreinte de `claims_emb.npz` ne doit pas
   changer.
2. `target_vecs` (blend α du bac à sable) est une population DISTINCTE → autre centroïde.
3. **Console live** (`live_cluster`) clusterise des SOUS-ENSEMBLES : la moyenne d'un
   sous-ensemble n'est pas celle du corpus. Naïvement centrée, elle produirait un espace
   différent à chaque filtre.
4. La cohésion chute de 0.79–0.87 à 0.28–0.61 ; `sauce_magique` la consomme → ses poids
   `α,β,γ,δ` sont à revoir, ou le recut à remplacer (T-N8).

## Reproduire
Diagnostics + gold : cf. queue `T-N10`. Le cosinus est invariant par ÉCHELLE mais PAS par
TRANSLATION — multiplier les vecteurs ne change rien, déplacer l'origine change tout.
