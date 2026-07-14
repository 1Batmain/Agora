# Verdict — le seuil de subdivision `tau` (2026-07-09)

**Question.** L'arbre de thèmes est gouverné par trois freins : `tau` (seuil de
dispersion au-dessus duquel un nœud tente de se subdiviser), `RES_LADDER` (échelle de
résolution que `_subdivide` monte jusqu'à obtenir une coupe) et `min_sub_size` (taille
minimale d'un sous-thème viable). Lesquels servent réellement ?

**Déclencheur.** Le rebuild complet de tiktok (2026-07-08) a produit un arbre PLAT —
15 thèmes, profondeur 0, structuration 0.00 — là où la prod sert 281 thèmes, profondeur
4, structuration 0.96. Le build a écrit `status: ready` sans un signal.

---

## Verdicts

| | verdict | pourquoi |
|---|---|---|
| `tau` (milieu du plus grand écart) | **NON** | décidé par un écart de 2 claims ; fait basculer l'arbre de 304 thèmes à 15 |
| `RES_LADDER` | **NON** | code mort : ne monte JAMAIS, sur 4 corpus, avec ou sans `tau` |
| `min_sub_size` recalculé par nœud | **NON** | rétrécit avec le nœud → ne freine jamais la descente |
| `min_sub_size` à l'échelle du corpus | **OUI** | seul frein nécessaire ; rend l'arbre stable |

**Conclusion : `tau` et `RES_LADDER` sont retirés. Le frein devient `min_sub_size`,
dérivé UNE fois du nombre total de claims. Il ne reste qu'un robinet : `resolution`.**

---

## 1. `tau` est décidé à pile ou face

`_derive_tau` trie les dispersions des clusters fins et pose le seuil au milieu du plus
grand écart. Sur tiktok, les deux plus grands écarts sont **0.0216** (bas de la
distribution) et **0.0180** (haut) — 20 % d'écart entre eux. Lequel gagne décide de tout :

```
2419 claims : [0.1341, 0.1557, ...]      → tau = 0.1449 → 10/11 racines subdivisent → 304 thèmes, prof 4
2421 claims : les deux écarts s'inversent → tau = 0.2145 →  1/11 racine  subdivise  →  15 thèmes, prof 0
```

Deux claims — 0,08 % du corpus. Et l'extraction LLM n'est pas déterministe : deux appels
au MÊME modèle sur le MÊME corpus ont rendu 2421 puis 2419 claims. **La hiérarchie servie
était donc tirée au sort à chaque build.**

### Mesure de robustesse (retrait aléatoire de 1 % des claims, 6 tirages, tiktok)

| tirage | A (actuel) | F (minimal) |
|---|---|---|
| complet | 304 thèmes, prof 4 | 41 thèmes, prof 2 |
| −1 % #1 | 163, prof 3 | 46, prof 1 |
| −1 % #2 | **17, prof 0** | 50, prof 1 |
| −1 % #3 | 140, prof 3 | 50, prof 2 |
| −1 % #4 | 155, prof 3 | 52, prof 2 |
| −1 % #5 | 162, prof 3 | 53, prof 2 |
| **amplitude** | **287 thèmes, prof 0–4** | **12 thèmes, prof 1–2** |

Le tirage #2 reproduit l'effondrement du 2026-07-08 (`tau = 0.2138` contre `0.2145`).
L'effondrement n'est pas un accident : **c'est un tirage sur six.**

## 2. `RES_LADDER` ne sert à rien

`_subdivide` monte `(1.0, 1.5, 2.0, 3.0)` jusqu'à trouver ≥2 sous-communautés viables.
Mesuré : forcer l'échelle à `(1.0,)` ne change **rien** — ni sous `tau` (A' ≡ A), ni sans
(C ≡ B) — sur tiktok, tiktok-prod, république-numérique et x-stance. À résolution 1.0,
Leiden dégage déjà ≥2 communautés. L'échelle n'est jamais empruntée.

## 3. Le vrai frein est `min_sub_size`, et il était mal posé

`derive_min_sub_size(n) = max(5, 0.011·n)` était rappelé **sur chaque sous-ensemble**.
Un nœud de 60 claims accepte donc des sous-thèmes de 5 claims : le critère de viabilité
rétrécit avec le nœud, donc n'arrête jamais la descente. Retirer `tau` sans corriger ça
fait exploser l'arbre (tiktok 304 → 475, x-stance 107 → 828 thèmes, feuille médiane
8 claims, jusqu'à 18 feuilles portées par moins de 5 citoyens).

Un thème est un thème **à l'échelle du corpus**, pas à l'échelle de son parent.

### Forme de l'arbre servi (4 corpus, mistral-large)

| corpus | A : thèmes / prof / feuille méd. | F : thèmes / prof / feuille méd. | feuilles < 5 citoyens |
|---|---|---|---|
| tiktok (2419 cl.) | 304 / 4 / 9 avis | **41 / 2 / 60 avis** | 1 → 0 |
| tiktok prod (2511) | 281 / 4 / 10 avis | **52 / 1 / 46 avis** | 3 → 1 |
| république-num. (3887) | 239 / 3 / 14 avis | **47 / 1 / 86 avis** | 5 → 0 |
| x-stance (4274) | 107 / 2 / 29 avis | **49 / 1 / 90 avis** | 2 → 2 |

La façade macro (14–19 thèmes) et le score `sauce_magique` sont **inchangés** : `tau` ne
touchait que la sous-arborescence. On perd de la profondeur nominale, on gagne des
feuilles qui pèsent (9 → 60 citoyens sur tiktok) et la disparition des feuilles orphelines.

---

## Ce que ça change dans le code

- `_derive_tau` et `RES_LADDER` : **supprimés**.
- `_subdivide` : Leiden une fois, à `resolution`. « Ce nœud ne se divise pas » redevient
  une réponse valide, pas un échec à contourner.
- `min_sub_size` : dérivé une fois de `n_claims` du corpus, propagé à toutes les
  profondeurs (le graphe local garde ses `k`/seuil dérivés localement).
- `params.adaptive.dispersion_threshold` : disparaît du payload (plus de seuil).

## Reproduire

```bash
uv run --extra embed-contender --extra faiss python research/tau_robustness.py
```

## Garde-fou associé

`build_analysis._assert_tree_is_structured` (commit `2934ccf`) refuse désormais un arbre
plat AVANT l'enrichissement LLM. Il aurait fait échouer le build du 2026-07-08 au lieu
de le servir. On le garde : il protège contre la prochaine cause d'aplatissement, pas
seulement contre celle-ci.

## Ce qui n'est PAS tranché

- `resolution` reste le seul robinet, à `1.0` par défaut. `0.8` donne un arbre un peu plus
  profond (tiktok 65 thèmes, prof 2, structuration 50 %). À calibrer sur un vrai critère
  qualité — on n'en a pas : les golds existants ne mesurent que la granularité
  (cf. verdict `sauce_magique`, calibration des poids = NON).
- La profondeur utile (1–2 niveaux) est un choix produit, pas un optimum mesuré.
