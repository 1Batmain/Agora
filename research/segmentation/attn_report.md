# Segmentation par ATTENTION — l'attention bat-elle l'embedding ?

*Jeu : `gold_large.json` — N=305 (104 mono, 201 multi). Modèles : intfloat/multilingual-e5-base, BAAI/bge-m3. CPU, seed=0.*

## 0. Faisabilité de l'extraction d'attention

- **e5-base** (`intfloat/multilingual-e5-base`) : **OUI.** `AutoModel(attn_implementation='eager')` + `output_attentions=True` → tuple de `[batch, heads=12, seq, seq]` × **12 couches**. Réduction token→mot par `offset_mapping` (spéciaux + préfixe `'passage: '` retirés). Attention triviale à extraire (XLM-R standard).
- **bge-m3** (`BAAI/bge-m3`) : **OUI.** `AutoModel(attn_implementation='eager')` + `output_attentions=True` → tuple de `[batch, heads=16, seq, seq]` × **24 couches**. Réduction token→mot par `offset_mapping` (spéciaux + préfixe `''` retirés). Attention triviale à extraire (XLM-R standard).

## 1. Méthode — signal de frontière par flux d'attention

- **Unité = mot** (suite de non-espaces, identique au banc embeddings). `A_word[L,H,i,j]` = moyenne des poids d'attention token(mot i)→token(mot j) (tokens spéciaux/préfixe retirés pour éviter le « puits » sur CLS).
- **Signal `cross(p)`** (frontière candidate entre mot p-1 et p) = flux d'attention MOYEN entre le bloc gauche (W mots) et le bloc droit (W mots), symétrisé (i→j + j→i). **Bas = frontière** : les mots d'un thème s'attendent entre eux ; au virage, le flux gauche↔droite s'effondre. Normalisé par la taille des blocs → comparable d'un avis à l'autre.
- **Frontières** = minima locaux de `cross` sous `μ_cross − c·σ_cross`, μ/σ **poolés GLOBALEMENT** sur tous les avis (un seuil par-avis ne peut jamais s'abstenir sur un mono cohérent). Coefficient `c` sans dimension, `min_seg=3` mots. Zéro magic-number absolu.
- **Balayage** : modèle × jeu-de-couches (early/lowmid/mid/late/midlate/all) × agrégation des têtes (`mean` = toutes ; `local` = têtes dont la localité — masse d'attention à distance ≤1 — dépasse la moyenne, sélection NON supervisée) × fenêtre W∈[3, 5, 8, 12] × seuil c∈[0.5, 1.0, 1.5, 2.0].

## 2. Attention vs change-point (trajectoire d'embedding)

| approche | config | Pk | WindowDiff | F1_multi | P | R | mono_FP | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **change-point** (embeddings nomic-v2) | W=8 pen=3.0 | 0.2815 | 0.282 | 0.4423 | 0.4545 | 0.4307 | 0.7019 | 0.384 |
| _embedding-trajectoire e5-base_ (contrôle) | changepoint W=5 | 0.3774 | 0.3827 | 0.3421 | 0.2926 | 0.412 | 0.9808 | 0.2781 |
| _embedding-trajectoire bge-m3_ (contrôle) | changepoint W=8 | 0.3355 | 0.3365 | 0.3717 | 0.369 | 0.3745 | 0.7981 | 0.3185 |
| attention e5-base | lowmid/mean W=8 c=1.0 | 0.1493 | 0.1563 | 0.7692 | 0.8955 | 0.6742 | 0.1442 | 0.7453 |
| attention bge-m3 | mid/local W=8 c=1.0 | 0.2496 | 0.2571 | 0.5752 | 0.7027 | 0.4869 | 0.1346 | 0.5579 |

*(Pk/WindowDiff ↓ = mieux, sur les multi ; F1_multi = frontières tol ±1 ; mono_FP = fraction de mono sur-coupés ; F1_global = frontières mono+multi, objectif de sélection.)*

## 3. Top 15 configurations attention

| model | layers | heads | W | c | Pk | WindowDiff | F1_multi | P | R | mono_FP | mono_cuts | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e5-base | lowmid | mean | 8 | 1.0 | 0.1493 | 0.1563 | 0.7692 | 0.8955 | 0.6742 | 0.1442 | 0.144 | 0.7453 |
| e5-base | mid | mean | 8 | 1.0 | 0.1242 | 0.1295 | 0.8072 | 0.8602 | 0.7603 | 0.4423 | 0.462 | 0.7368 |
| e5-base | lowmid | local | 8 | 1.0 | 0.1582 | 0.1595 | 0.726 | 0.9688 | 0.5805 | 0.125 | 0.125 | 0.7045 |
| e5-base | mid | local | 8 | 1.0 | 0.1359 | 0.1388 | 0.7692 | 0.9309 | 0.6554 | 0.4231 | 0.433 | 0.7 |
| e5-base | mid | mean | 12 | 1.0 | 0.1732 | 0.1822 | 0.7191 | 0.8989 | 0.5993 | 0.1731 | 0.173 | 0.6911 |
| e5-base | lowmid | local | 5 | 1.0 | 0.1566 | 0.1829 | 0.7741 | 0.6474 | 0.9625 | 0.75 | 1.01 | 0.6684 |
| e5-base | lowmid | mean | 5 | 1.0 | 0.1664 | 0.1848 | 0.7397 | 0.6419 | 0.8727 | 0.5769 | 0.712 | 0.6619 |
| e5-base | lowmid | mean | 12 | 1.0 | 0.1971 | 0.2167 | 0.6667 | 0.7828 | 0.5805 | 0.0385 | 0.038 | 0.661 |
| e5-base | mid | local | 12 | 1.0 | 0.2118 | 0.2151 | 0.5985 | 0.9435 | 0.4382 | 0.1442 | 0.144 | 0.5764 |
| e5-base | mid | local | 5 | 1.0 | 0.2296 | 0.2554 | 0.6773 | 0.5519 | 0.8764 | 0.8846 | 1.327 | 0.5645 |
| bge-m3 | mid | local | 8 | 1.0 | 0.2496 | 0.2571 | 0.5752 | 0.7027 | 0.4869 | 0.1346 | 0.135 | 0.5579 |
| e5-base | mid | mean | 5 | 1.0 | 0.2411 | 0.2654 | 0.6657 | 0.5395 | 0.8689 | 0.875 | 1.346 | 0.5544 |
| bge-m3 | mid | local | 5 | 1.0 | 0.2751 | 0.2934 | 0.5917 | 0.5296 | 0.6704 | 0.5 | 0.587 | 0.5375 |
| e5-base | mid | mean | 12 | 0.5 | 0.2367 | 0.3445 | 0.6093 | 0.4474 | 0.9551 | 0.9615 | 1.606 | 0.508 |
| e5-base | midlate | local | 5 | 1.0 | 0.3141 | 0.353 | 0.5219 | 0.4272 | 0.6704 | 0.2788 | 0.317 | 0.4979 |

## 4. Meilleure config par couche (les têtes/couches qui aident)


**e5-base** — têtes locales sélectionnées par jeu de couches : `{'early': [0, 4, 5, 6, 7, 8], 'lowmid': [0, 2, 5, 8, 9], 'mid': [1, 2, 3, 4, 11], 'late': [0, 1, 2, 4, 5, 9, 11], 'midlate': [1, 2, 3, 4, 11], 'all': [0, 1, 2, 4, 5, 8]}` (sur 12 têtes).

| model | layers | heads | W | c | Pk | WindowDiff | F1_multi | P | R | mono_FP | mono_cuts | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e5-base | early | mean | 12 | 0.5 | 0.432 | 0.543 | 0.2279 | 0.157 | 0.4157 | 0.0769 | 0.115 | 0.2252 |
| e5-base | lowmid | mean | 8 | 1.0 | 0.1493 | 0.1563 | 0.7692 | 0.8955 | 0.6742 | 0.1442 | 0.144 | 0.7453 |
| e5-base | mid | mean | 8 | 1.0 | 0.1242 | 0.1295 | 0.8072 | 0.8602 | 0.7603 | 0.4423 | 0.462 | 0.7368 |
| e5-base | late | local | 8 | 0.5 | 0.4409 | 0.543 | 0.289 | 0.2006 | 0.5169 | 0.2212 | 0.298 | 0.2799 |
| e5-base | midlate | local | 5 | 1.0 | 0.3141 | 0.353 | 0.5219 | 0.4272 | 0.6704 | 0.2788 | 0.317 | 0.4979 |
| e5-base | all | local | 5 | 1.0 | 0.3188 | 0.3496 | 0.5024 | 0.4324 | 0.5993 | 0.1635 | 0.163 | 0.4893 |


**bge-m3** — têtes locales sélectionnées par jeu de couches : `{'early': [1, 2, 6, 7, 13, 14, 15], 'lowmid': [3, 4, 6, 7, 9, 10, 11, 13, 15], 'mid': [1, 4, 5, 6, 7, 9, 13], 'late': [5, 6, 7, 8, 9, 12, 13, 15], 'midlate': [4, 5, 6, 7, 8, 9, 13, 15], 'all': [2, 4, 6, 7, 9, 13, 14, 15]}` (sur 16 têtes).

| model | layers | heads | W | c | Pk | WindowDiff | F1_multi | P | R | mono_FP | mono_cuts | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bge-m3 | early | mean | 12 | 0.5 | 0.4261 | 0.5269 | 0.2193 | 0.1557 | 0.3708 | 0.125 | 0.231 | 0.2136 |
| bge-m3 | lowmid | mean | 12 | 0.5 | 0.2726 | 0.3934 | 0.5088 | 0.3711 | 0.809 | 0.5096 | 0.663 | 0.4706 |
| bge-m3 | mid | local | 8 | 1.0 | 0.2496 | 0.2571 | 0.5752 | 0.7027 | 0.4869 | 0.1346 | 0.135 | 0.5579 |
| bge-m3 | late | local | 5 | 0.5 | 0.4709 | 0.5699 | 0.3245 | 0.2196 | 0.6217 | 0.5096 | 0.692 | 0.3032 |
| bge-m3 | midlate | mean | 8 | 0.5 | 0.3812 | 0.4614 | 0.4359 | 0.3127 | 0.7191 | 0.4615 | 0.606 | 0.4068 |
| bge-m3 | all | mean | 8 | 0.5 | 0.4017 | 0.4903 | 0.4283 | 0.3036 | 0.7266 | 0.4423 | 0.587 | 0.4012 |

## 5. Verdict honnête

**Meilleure config attention : `e5-base` · lowmid/mean · W=8 · c=1.0** → F1_multi=0.769 (P=0.896, R=0.674), Pk=0.149, WindowDiff=0.156, F1_global=0.745, mono_FP=0.144.

- **L'attention bat-elle la trajectoire d'embedding ? **OUI**.** vs change-point (F1_multi=0.4423, Pk=0.2815, F1_global=0.384) : ΔF1_multi=+0.327, ΔPk=-0.132 (négatif = mieux), ΔF1_global=+0.361.

- **Contrôle MÊME encodeur (sans confondre signal et modèle)** : la trajectoire d'embedding de `e5-base` lui-même (`changepoint` W=5) donne F1_multi=0.3421, Pk=0.3774, F1_global=0.2781. L'attention du même modèle fait ΔF1_multi=+0.427, ΔPk=-0.228, ΔF1_global=+0.467 → l'attention **bat** sa propre trajectoire d'embedding. **C'est la comparaison décisive** (le gain n'est pas un simple effet « e5 > nomic »).

- **Honnêteté têtes/couches** : l'attention de transformer est en grande partie syntaxique/positionnelle (têtes qui suivent le mot précédent/suivant, ou pointent vers la ponctuation). La sélection `local` ne garde que les têtes les plus topiques, mais rien ne garantit qu'une tête « thème » existe : XLM-R n'a jamais été entraîné à segmenter. Fait notable : la meilleure config agrège **TOUTES** les têtes (`mean`), pas la sélection `local` — le signal de cohésion thématique est **diffus** sur l'ensemble des têtes des couches basses-moyennes, pas concentré dans quelques « têtes-thème » identifiables. Les couches qui portent le signal (basses-moyennes, §4) précèdent les couches tardives plus abstraites/poolées — cohérent avec l'idée que la cohésion locale de thème vit tôt dans le réseau.

- **Jeu** : multi = concaténation de mono-thèmes (frontières nettes par construction) → borne OPTIMISTE pour les deux approches.

- **Portage nomic** : justifié UNIQUEMENT si l'attention bat nettement l'embedding ci-dessus. nomic (code custom, Wqkv fusionné + rotary) demande un hook manuel pour exposer les poids — coût non négligeable. Verdict ci-dessus = feu vert / rouge.
