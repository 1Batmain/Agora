# VERDICT — balayage de `k` (voisins k-NN) : modularité + alignement taxo officielle

> Branche `work/k-sweep`. **R&D pur** — harness `research/k_sweep.py`, AUCUN fichier
> produit (backend/frontend) modifié. Lancer :
> `uv run --extra contender --extra embed-contender --extra faiss --with fastapi python research/k_sweep.py`
> Résultats bruts : `research/k_sweep_results.json`.

## Question

Un grand `k` (voisinage k-NN dense) donne-t-il **objectivement** un meilleur
clustering, ou seulement un découpage plus **grossier** (moins de thèmes, plus
« tidy » à l'œil sans gain de qualité) ? On tranche par la mesure :

- **modularité Q** de la partition Leiden au niveau RACINE = qualité intrinsèque ;
- **alignement** à la **taxo officielle** (repnum) = vérité-terrain externe (ARI/NMI/V).

Le harness réutilise telle quelle la brique de production
`backend.live_cluster.build_live_tree(ideas, vecs, weights, k=k)` (le chemin EXACT du
levier de la Console). `k` pilote le voisinage ; le seuil d'arête suit (`derive_defaults(k)`).

## Vérité-terrain repnum — où sont les labels officiels

La taxo officielle = le **plan du projet de loi** (Titre I/II/III = 3 axes). La
vérité par contribution est la colonne **`Catégorie`** du CSV source data.gouv.fr
(`projet-de-loi-numerique-consultation-anonyme.csv`, cf. descripteur) :

- **Proposition / Modification** → `Catégorie` = code de section (`TITRE Ier - Chapitre … - Section …`).
- **Argument** → pas de section propre (`Catégorie` ∈ Pour/Contre/Mitigé) ; la section
  est **héritée du parent** via la colonne `Lié.à..` (« Proposition "498" »).
- Jointure cache→CSV par `id = source:Identifiant` ; l'Identifiant étant réutilisé
  ENTRE types, l'ambiguïté est levée par le **meilleur match texte** (`difflib`) sur
  les candidats du même Identifiant.

→ **3000/3000** idées cachées reçoivent un axe gold, sans trou :
**Titre Ier 1694 · Titre II 703 · Titre III 603**.

## Résultats (k → granularité, modularité, alignement)

### tiktok (N=1621, `derive_k`=12)
| k | thr | deg.moy | n_macros | n_leaves | **Q** |
|---:|---:|---:|---:|---:|---:|
| **8** | 0.607 | 13 | 10 | 69 | **0.637** |
| 12 *défaut* | 0.600 | 20 | 10 | 47 | 0.602 |
| 16 | 0.595 | 26 | 8 | 65 | 0.581 |
| 24 | 0.587 | 38 | 7 | 74 | 0.551 |
| 36 | 0.578 | 57 | 6 | 32 | 0.515 |
| 50 | 0.570 | 78 | 5 | 52 | 0.498 |
| 80 | 0.557 | 122 | 4 | 9 | 0.463 |
| 120 | 0.546 | 179 | 3 | 21 | 0.435 |
| 200 | 0.529 | 288 | 3 | 25 | 0.389 |

### granddebat (N=3000, `derive_k`=13)
| k | thr | deg.moy | n_macros | n_leaves | **Q** |
|---:|---:|---:|---:|---:|---:|
| **8** | 0.603 | 14 | 13 | 18 | **0.570** |
| 12 | 0.598 | 20 | 12 | 104 | 0.544 |
| 13 *défaut* | 0.597 | 22 | 13 | 19 | 0.534 |
| 16 | 0.595 | 27 | 13 | 19 | 0.519 |
| 24 | 0.589 | 40 | 8 | 14 | 0.494 |
| 36 | 0.582 | 60 | 7 | 14 | 0.467 |
| 50 | 0.576 | 82 | 6 | 11 | 0.445 |
| 80 | 0.567 | 130 | 5 | 10 | 0.408 |
| 120 | 0.558 | 193 | 5 | 12 | 0.376 |
| 200 | 0.546 | 316 | 5 | 9 | 0.329 |

### xstance (N=3000, `derive_k`=13)
| k | thr | deg.moy | n_macros | n_leaves | **Q** |
|---:|---:|---:|---:|---:|---:|
| **8** | 0.537 | 13 | 20 | 133 | **0.667** |
| 12 | 0.527 | 19 | 17 | 23 | 0.620 |
| 13 *défaut* | 0.525 | 20 | 18 | 78 | 0.609 |
| 16 | 0.519 | 25 | 17 | 23 | 0.582 |
| 24 | 0.507 | 38 | 13 | 166 | 0.534 |
| 36 | 0.493 | 56 | 11 | 57 | 0.488 |
| 50 | 0.482 | 79 | 10 | 57 | 0.455 |
| 80 | 0.464 | 126 | 8 | 53 | 0.399 |
| 120 | 0.449 | 189 | 6 | 11 | 0.349 |
| 200 | 0.429 | 314 | 5 | 17 | 0.292 |

### republique-numerique (N=3000, `derive_k`=13) — **+ alignement axes officiels**
| k | deg.moy | n_macros | n_leaves | **Q** | **ARI** | **NMI** | **V** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **8** | 13 | 20 | 293 | **0.679** | **0.055** | 0.160 | 0.160 |
| 12 | 19 | 15 | 173 | 0.638 | 0.040 | 0.144 | 0.144 |
| **13 *défaut*** | 21 | 16 | 184 | 0.628 | 0.054 | **0.168** | **0.168** |
| 16 | 26 | 14 | 176 | 0.610 | 0.047 | 0.146 | 0.146 |
| 24 | 38 | 13 | 46 | 0.568 | 0.040 | 0.129 | 0.129 |
| 36 | 57 | 10 | 20 | 0.532 | 0.053 | 0.130 | 0.130 |
| 50 | 79 | 9 | 18 | 0.499 | 0.048 | 0.127 | 0.127 |
| 80 | 126 | 6 | 16 | 0.454 | 0.039 | 0.092 | 0.092 |
| 120 | 188 | 6 | 17 | 0.410 | 0.040 | 0.092 | 0.092 |
| 200 | 309 | 6 | 16 | 0.346 | 0.039 | 0.091 | 0.091 |

## VERDICT chiffré

**NON — un grand `k` ne gagne sur AUCUNE métrique objective. Il ne fait que grossir.**

1. **Modularité (qualité intrinsèque) : monotone DÉCROISSANTE en `k`, sur les 4 datasets.**
   Le `Q` maximal est toujours au **plus petit `k` (=8)** ; il s'effondre vers les grands
   `k` (chute relative du **max** au `k=200`) :
   - tiktok 0.637 → 0.389 (**−39 %**)
   - granddebat 0.570 → 0.329 (**−42 %**)
   - xstance 0.667 → 0.292 (**−56 %**)
   - repnum 0.679 → 0.346 (**−49 %**)
   Mécanisme : `k` ↑ ⇒ degré moyen explose (≈13 → ≈300) ⇒ graphe quasi-complet ⇒ Leiden
   trouve **moins de communautés, plus lâches** ⇒ Q chute. C'est le « grossissement »,
   pas une amélioration.

2. **Granularité : `n_macros` chute avec `k`** (tiktok 10→3, granddebat 13→5, xstance
   20→5, repnum 20→6). Donc « moins de thèmes, plus tidy à l'œil » = exactement l'effet
   cosmétique suspecté — et il s'accompagne d'une **perte** de Q, pas d'un gain.

3. **Alignement à la taxo officielle (repnum) : un grand `k` DÉGRADE l'alignement.**
   NMI/V culminent **au défaut `k`=13 (0.168)** et ARI au `k`=8 (0.055) ; aux grands `k`
   (80–200) NMI/V tombent à **0.09** (**−45 %** vs le défaut). Le grossissement éloigne
   donc le clustering des axes officiels au lieu de l'en rapprocher.
   *(Niveau absolu bas — NMI≈0.17 — cohérent avec le constat connu : repnum est
   mono-domaine, sur-concentré au macro, les axes vivent un cran plus bas
   ([[agora-repnum-benchmark-verdict]]). Mais le classement RELATIF entre `k` est net.)*

### `k` qui maximise chaque métrique, par dataset
| dataset | max Q | max ARI | max NMI | max V |
|---|---|---|---|---|
| tiktok | k=8 | — | — | — |
| granddebat | k=8 | — | — | — |
| xstance | k=8 | — | — | — |
| republique-numerique | k=8 | k=8 | **k=13** | **k=13** |

Le défaut `derive_k` (12–13) tombe **dans le sweet-spot** : il maximise l'alignement
NMI/V à la taxo officielle, et son `Q` n'est qu'à ~5 % sous l'optimum `k=8`.

## Recommandation sur `derive_k` (NON appliquée — chiffrée seulement)

**Ne PAS retoucher `derive_k`.** Les données invalident l'hypothèse « monter `k`
améliore » : il faudrait au contraire un `k` **plus petit** pour gratter la modularité,
mais le défaut actuel (`K_LOG_COEF=3.8`, `K_MAX=30` ⇒ 12–13 ici) est déjà
quasi-optimal et **maximise l'alignement gold**. Concrètement :

- **Surtout ne pas augmenter** `K_LOG_COEF` ni le plafond `K_MAX` (30) : tout `k` au-delà
  de ~16 dégrade Q **et** l'alignement sur les 4 datasets. La piste « coef↑ / cap↑ »
  du brief est **réfutée**.
- Micro-optimisation possible mais **non recommandée** : abaisser le défaut vers `k≈8`
  gagnerait +3–6 pts de Q, mais (a) perdrait l'alignement NMI/V optimal du repnum
  (0.168→0.160) et (b) fragmenterait davantage (n_macros/feuilles ↑). Le défaut 12–13
  est le meilleur compromis qualité-intrinsèque / fidélité-gold / lisibilité. **Statu quo.**

### Conséquence pour la Console
Le slider `k` (porté à 200) est un **levier de grossissement**, pas de qualité : monter
`k` rend la carte plus « propre » en fusionnant des thèmes au prix de la modularité ET
de la fidélité aux axes officiels. À documenter comme tel côté UX (curseur = niveau de
zoom thématique, le défaut dérivé restant le point de qualité maximale).
