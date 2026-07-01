# Segmentation par ATTENTION — e5-LARGE relève-t-il le plafond ?

*Jeu : `gold_large.json` — N=305 (104 mono, 201 multi). Encodeur : `intfloat/multilingual-e5-large` (24 couches, 16 têtes). Méthode IDENTIQUE au réglé-main e5-base (`attn_seg.py`) : seul le modèle change. CPU, seed=0.*

## 0. Faisabilité de l'extraction d'attention sur e5-large

- **OUI.** `intfloat/multilingual-e5-large` est un **XLM-R large** standard : `AutoModel(attn_implementation='eager')` + `output_attentions=True` → tuple de `[batch, heads=16, seq, seq]` × **24 couches**. Aucun hook custom requis (contrairement à nomic). Réduction token→mot par `offset_mapping`, préfixe `'passage: '` retiré comme pour e5-base.

## 1. e5-large vs e5-base réglé-main (la comparaison décisive)

| encodeur | config | Pk | WindowDiff | F1_multi | P | R | mono_FP | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **e5-base** (réglé-main, plafond) | lowmid/mean W=8 c=1.0 | 0.1493 | 0.1563 | 0.7692 | 0.8955 | 0.6742 | 0.1442 | 0.7453 |
| **e5-large** (cette expé) | mid/local W=8 c=1.0 | 0.1244 | 0.1372 | 0.8029 | 0.7899 | 0.8165 | 0.3846 | 0.7466 |
| _embedding-trajectoire e5-large_ (contrôle) | centroid_live W=5 | 0.3658 | 0.3918 | 0.3201 | 0.2962 | 0.3483 | 0.4038 | 0.2915 |

*(Pk/WindowDiff ↓ = mieux, sur multi ; F1_multi = frontières tol ±1 ; mono_FP = fraction de mono sur-coupés ↓ ; F1_global = mono+multi, objectif de sélection.)*

## 2. Top 15 configurations e5-large

| model | layers | heads | W | c | Pk | WindowDiff | F1_multi | P | R | mono_FP | mono_cuts | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e5-large | mid | local | 8 | 1.0 | 0.1244 | 0.1372 | 0.8029 | 0.7899 | 0.8165 | 0.3846 | 0.394 | 0.7466 |
| e5-large | mid | local | 12 | 1.0 | 0.1616 | 0.1784 | 0.724 | 0.7768 | 0.6779 | 0.0962 | 0.106 | 0.7084 |
| e5-large | mid | mean | 8 | 1.0 | 0.1706 | 0.1863 | 0.7199 | 0.7132 | 0.7266 | 0.2788 | 0.279 | 0.6831 |
| e5-large | lowmid | mean | 8 | 1.0 | 0.2105 | 0.216 | 0.6606 | 0.8343 | 0.5468 | 0.0769 | 0.077 | 0.6489 |
| e5-large | mid | mean | 12 | 1.0 | 0.1989 | 0.2224 | 0.6444 | 0.6777 | 0.6142 | 0.0481 | 0.048 | 0.6381 |
| e5-large | lowmid | mean | 5 | 1.0 | 0.229 | 0.2471 | 0.6525 | 0.5802 | 0.7453 | 0.4423 | 0.49 | 0.6021 |
| e5-large | lowmid | mean | 12 | 1.0 | 0.231 | 0.2497 | 0.5856 | 0.7345 | 0.4869 | 0.0 | 0.0 | 0.5856 |
| e5-large | lowmid | local | 5 | 1.0 | 0.229 | 0.2435 | 0.6436 | 0.5981 | 0.6966 | 0.5288 | 0.587 | 0.5822 |
| e5-large | lowmid | local | 8 | 1.0 | 0.2352 | 0.2365 | 0.5867 | 0.92 | 0.4307 | 0.0577 | 0.058 | 0.5779 |
| e5-large | mid | local | 5 | 1.0 | 0.2464 | 0.2742 | 0.6501 | 0.5142 | 0.8839 | 0.7404 | 1.048 | 0.5653 |
| e5-large | lowmid | local | 12 | 1.0 | 0.2484 | 0.2549 | 0.5553 | 0.8852 | 0.4045 | 0.0 | 0.0 | 0.5553 |
| e5-large | mid | mean | 5 | 1.0 | 0.2649 | 0.2927 | 0.6236 | 0.5058 | 0.8127 | 0.6442 | 0.885 | 0.5508 |
| e5-large | mid | local | 5 | 1.5 | 0.2526 | 0.2538 | 0.5316 | 0.8203 | 0.3933 | 0.1538 | 0.154 | 0.5109 |
| e5-large | midlate | local | 8 | 1.0 | 0.2788 | 0.3158 | 0.5 | 0.4747 | 0.5281 | 0.0577 | 0.058 | 0.4947 |
| e5-large | mid | local | 12 | 0.5 | 0.2606 | 0.3661 | 0.5704 | 0.4124 | 0.9251 | 0.8173 | 1.327 | 0.492 |

## 3. Couche-par-couche — e5-large vs e5-base (même jeu, meilleur c/W)

Têtes locales sélectionnées par jeu de couches : `{'early': [1, 2, 6, 7, 9, 13, 14, 15], 'lowmid': [3, 4, 6, 7, 9, 10, 13, 15], 'mid': [1, 4, 5, 6, 7, 9, 13], 'late': [0, 2, 6, 7, 9, 12, 13, 15], 'midlate': [4, 5, 6, 7, 9, 13, 15], 'all': [2, 4, 6, 7, 9, 13, 15]}` (sur 16 têtes).

| jeu | large F1_g | large F1_m | large Pk | large mono_FP | base F1_g | base F1_m | ΔF1_global |
| --- | --- | --- | --- | --- | --- | --- | --- |
| early | 0.2026 | 0.2077 | 0.4136 | 0.1346 | 0.2252 | 0.2279 | -0.0226 |
| lowmid | 0.6489 | 0.6606 | 0.2105 | 0.0769 | 0.7453 | 0.7692 | -0.0964 |
| mid | 0.7466 | 0.8029 | 0.1244 | 0.3846 | 0.7368 | 0.8072 | +0.0098 |
| late | 0.3023 | 0.3158 | 0.405 | 0.3173 | 0.2799 | 0.289 | +0.0224 |
| midlate | 0.4947 | 0.5 | 0.2788 | 0.0577 | 0.4979 | 0.5219 | -0.0032 |
| all | 0.4374 | 0.4382 | 0.3129 | 0.0096 | 0.4893 | 0.5024 | -0.0519 |

## 4. Verdict honnête

**Meilleure config e5-large (par F1_global) : `mid/local` · W=8 · c=1.0** → F1_multi=0.803 (P=0.790, R=0.816), Pk=0.124, WindowDiff=0.137, F1_global=0.747, mono_FP=0.385.

- **e5-large relève-t-il le plafond, à abstention tenue ? **NON**.** La meilleure config par F1_global affiche bien ΔF1_multi=+0.034 et ΔPk=-0.025 (négatif = mieux) vs e5-base (F1_multi=0.7692, Pk=0.1493, mono_FP=0.1442), **mais Δmono_FP=+0.240** : elle sur-coupe les mono cohérents 2.7× plus. Sur l'OBJECTIF de sélection (F1_global) le gain est nul (Δ=+0.001) — à peine positif.

- **Comparaison décisive — abstention APPARIÉE** (mono_FP ≤ 0.1442, comme e5-base) : la meilleure config e5-large sous cette contrainte est `mid/local` W=12 c=1.0 → F1_multi=0.724, Pk=0.162, mono_FP=0.096 → **ΔF1_multi=-0.045, ΔPk=+0.012**. À abstention égale, e5-large **ne bat pas** e5-base : le 24-couches ne relève donc pas le plafond du signal — le gain apparent du tableau §1 est **acheté par la sur-segmentation des mono**, pas par un meilleur signal de frontière.

- **Contrôle MÊME encodeur** : la trajectoire d'embedding de e5-large lui-même (`centroid_live` W=5) donne F1_multi=0.3201, Pk=0.3658, F1_global=0.2915. L'attention e5-large fait ΔF1_multi=+0.483, ΔPk=-0.241 → l'attention **bat** sa propre trajectoire d'embedding.

- **Coût du modèle large (honnêteté ressources)** : e5-large = 24 couches / 16 têtes / ~560M params vs e5-base 12 couches / 12 têtes / ~278M. L'extraction d'attention matérialise `[24, 16, n, n]` poids par avis (×2 couches, ×~1.3 têtes vs base) : RAM et latence du forward nettement supérieures, cache disque plus lourd. À ne payer QUE si le gain de segmentation ci-dessus le justifie.

- **Jeu** : multi = concaténation de mono-thèmes (frontières nettes par construction) → borne OPTIMISTE, identique pour e5-base et e5-large : la comparaison reste équitable.
